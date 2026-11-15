"""Persistent Metal compute runtime implementing the compute_runtime contract on macOS."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import weakref

from .compiler import Toolchain, convert_dxil_to_metallib
from .compute_runtime import (
    ComputeContext,
    PersistentBuffer,
    PersistentComputeError,
    PersistentPipeline,
    _pack_dispatch_parameters,
    _validate_non_negative_int,
    _validate_positive_int,
)
from .viewer import ComputeProgram, _check_binary, _preview_toolchain, _run, _validate_provider, load_compute_program

_METAL_MAX_GRID_DIM = 65535
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_LIB: ctypes.CDLL | None = None


class _GfxMtlBufferBinding(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("is_uav", ctypes.c_int),
        ("set", ctypes.c_uint32),
        ("binding", ctypes.c_uint32),
    ]


def _find_metal_converter_include_dir() -> Path | None:
    candidates: list[Path] = []
    env_dir = os.environ.get("METAL_IRCONVERTER_INCLUDE_DIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            _REPO_ROOT / ".sdk" / "include",
            Path("/usr/local/include"),
            Path("/opt/homebrew/include"),
        ]
    )
    for cand in candidates:
        if (cand / "metal_irconverter_runtime" / "metal_irconverter_runtime.h").is_file():
            return cand
    return None


def _find_or_build_bridge_dylib() -> Path:
    """Locate prebuilt mlir-gpu-metal-compute.dylib or compile it from source."""
    candidates: list[Path] = []
    tools_env = os.environ.get("GPU_TOOLS_BIN")
    if tools_env:
        candidates.append(Path(tools_env) / "mlir-gpu-metal-compute.dylib")
    candidates.append(_REPO_ROOT / ".tools" / "bin" / "mlir-gpu-metal-compute.dylib")
    candidates.append(_REPO_ROOT / "build" / "native" / "mlir-gpu-metal-compute.dylib")

    src_mm = _REPO_ROOT / "native" / "metal_compute_bridge.mm"
    if not src_mm.is_file():
        for cand in candidates:
            if cand.is_file():
                return cand
        raise PersistentComputeError(f"Metal compute bridge source not found: {src_mm}")

    src_hash = hashlib.sha256(src_mm.read_bytes()).hexdigest()[:12]
    out_dir = _REPO_ROOT / "build" / "metal_compute_cache"
    cached_dylib = out_dir / f"metal_compute_{src_hash}.dylib"
    if cached_dylib.is_file():
        return cached_dylib

    for cand in candidates:
        if cand.is_file() and cand.stat().st_mtime >= src_mm.stat().st_mtime:
            return cand

    include_dir = _find_metal_converter_include_dir()
    if include_dir is None:
        raise PersistentComputeError(
            "Metal Shader Converter runtime header 'metal_irconverter_runtime/metal_irconverter_runtime.h' "
            "not found. Set METAL_IRCONVERTER_INCLUDE_DIR or install the Metal Shader Converter SDK in .sdk/."
        )

    clang_bin = shutil.which("clang++") or shutil.which("clang")
    if not clang_bin:
        raise PersistentComputeError(
            "Cannot build Metal compute bridge: clang++ not found in PATH."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dylib = out_dir / f".tmp_{os.getpid()}_{src_hash}.dylib"
    cmd = [
        clang_bin,
        "-std=c++17",
        "-O2",
        "-fobjc-arc",
        "-shared",
        f"-I{include_dir}",
        str(src_mm),
        "-framework",
        "Foundation",
        "-framework",
        "Metal",
        "-o",
        str(tmp_dylib),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode == 0 and tmp_dylib.is_file():
        tmp_dylib.replace(cached_dylib)
        return cached_dylib
    tmp_dylib.unlink(missing_ok=True)
    raise PersistentComputeError(
        f"Failed to compile Metal compute bridge ({clang_bin} exited with {res.returncode}):\n"
        f"{res.stderr or res.stdout}"
    )


def _load_bridge() -> ctypes.CDLL:
    global _BRIDGE_LIB
    if _BRIDGE_LIB is not None:
        return _BRIDGE_LIB

    dylib_path = _find_or_build_bridge_dylib()
    lib = ctypes.CDLL(str(dylib_path))

    lib.gfx_mtl_create_context.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
    lib.gfx_mtl_create_context.restype = ctypes.c_void_p

    lib.gfx_mtl_destroy_context.argtypes = [ctypes.c_void_p]
    lib.gfx_mtl_destroy_context.restype = None

    lib.gfx_mtl_context_device_name.argtypes = [ctypes.c_void_p]
    lib.gfx_mtl_context_device_name.restype = ctypes.c_char_p

    lib.gfx_mtl_create_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gfx_mtl_create_buffer.restype = ctypes.c_void_p

    lib.gfx_mtl_destroy_buffer.argtypes = [ctypes.c_void_p]
    lib.gfx_mtl_destroy_buffer.restype = None

    lib.gfx_mtl_upload_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gfx_mtl_upload_buffer.restype = ctypes.c_int

    lib.gfx_mtl_download_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gfx_mtl_download_buffer.restype = ctypes.c_int

    lib.gfx_mtl_create_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gfx_mtl_create_pipeline.restype = ctypes.c_void_p

    lib.gfx_mtl_destroy_pipeline.argtypes = [ctypes.c_void_p]
    lib.gfx_mtl_destroy_pipeline.restype = None

    lib.gfx_mtl_dispatch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(_GfxMtlBufferBinding),
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gfx_mtl_dispatch.restype = ctypes.c_int

    lib.gfx_mtl_synchronize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.gfx_mtl_synchronize.restype = ctypes.c_int

    _BRIDGE_LIB = lib
    return lib


def _decode_err(err_buf: ctypes.Array[ctypes.c_char]) -> str:
    return err_buf.value.decode("utf-8", errors="replace").strip() or "unknown Metal error"


class MetalBuffer(PersistentBuffer):
    """Metal shared storage buffer implementing PersistentBuffer."""

    def __init__(self, context: MetalComputeContext, size_bytes: int, handle: int) -> None:
        self._context = context
        self._size_bytes = size_bytes
        self._handle = handle
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    @property
    def context(self) -> MetalComputeContext:
        return self._context

    def _ensure_open(self) -> None:
        self._context._ensure_open()
        if self._closed or not self._handle:
            raise PersistentComputeError("PersistentBuffer has been closed.")

    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        self._ensure_open()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise PersistentComputeError("upload data must be bytes, bytearray, or memoryview.")
        _validate_non_negative_int(offset, "upload offset")
        view = memoryview(data).cast("B")
        nbytes = len(view)
        if offset + nbytes > self._size_bytes:
            raise PersistentComputeError(
                f"Upload range [{offset}, {offset + nbytes}) exceeds buffer size {self._size_bytes}."
            )
        if nbytes == 0:
            return
        err_buf = ctypes.create_string_buffer(1024)
        if not view.readonly and view.c_contiguous:
            src = (ctypes.c_uint8 * nbytes).from_buffer(view)
        else:
            src = (ctypes.c_uint8 * nbytes).from_buffer_copy(view)
        rc = self._context._lib.gfx_mtl_upload_buffer(
            self._context._ctx_handle,
            self._handle,
            offset,
            ctypes.cast(src, ctypes.c_void_p),
            nbytes,
            err_buf,
            len(err_buf),
        )
        if rc != 0:
            raise PersistentComputeError(_decode_err(err_buf))

    def download_into(
        self,
        dst: bytearray | memoryview,
        size_bytes: int | None = None,
        offset: int = 0,
    ) -> int:
        self._ensure_open()
        _validate_non_negative_int(offset, "download offset")
        if offset > self._size_bytes:
            raise PersistentComputeError(
                f"Download offset {offset} exceeds buffer size {self._size_bytes}."
            )
        read_size = (self._size_bytes - offset) if size_bytes is None else _validate_non_negative_int(
            size_bytes, "download size_bytes"
        )
        if offset + read_size > self._size_bytes:
            raise PersistentComputeError(
                f"Download range [{offset}, {offset + read_size}) exceeds buffer size {self._size_bytes}."
            )
        dst_view = memoryview(dst).cast("B")
        if dst_view.readonly:
            raise PersistentComputeError("download_into destination buffer must be writable.")
        if len(dst_view) < read_size:
            raise PersistentComputeError(
                f"download_into destination buffer ({len(dst_view)} bytes) is smaller than {read_size} bytes."
            )
        if read_size == 0:
            self._context.synchronize()
            return 0
        err_buf = ctypes.create_string_buffer(1024)
        if dst_view.c_contiguous:
            c_dst = (ctypes.c_uint8 * read_size).from_buffer(dst_view[:read_size])
            rc = self._context._lib.gfx_mtl_download_buffer(
                self._context._ctx_handle,
                self._handle,
                offset,
                ctypes.cast(c_dst, ctypes.c_void_p),
                read_size,
                err_buf,
                len(err_buf),
            )
            if rc != 0:
                raise PersistentComputeError(_decode_err(err_buf))
            return read_size
        tmp = (ctypes.c_uint8 * read_size)()
        rc = self._context._lib.gfx_mtl_download_buffer(
            self._context._ctx_handle,
            self._handle,
            offset,
            ctypes.cast(tmp, ctypes.c_void_p),
            read_size,
            err_buf,
            len(err_buf),
        )
        if rc != 0:
            raise PersistentComputeError(_decode_err(err_buf))
        dst_view[:read_size] = tmp
        return read_size

    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        self._ensure_open()
        _validate_non_negative_int(offset, "download offset")
        if offset > self._size_bytes:
            raise PersistentComputeError(
                f"Download offset {offset} exceeds buffer size {self._size_bytes}."
            )
        read_size = (self._size_bytes - offset) if size_bytes is None else _validate_non_negative_int(
            size_bytes, "download size_bytes"
        )
        if offset + read_size > self._size_bytes:
            raise PersistentComputeError(
                f"Download range [{offset}, {offset + read_size}) exceeds buffer size {self._size_bytes}."
            )
        if read_size == 0:
            self._context.synchronize()
            return b""
        err_buf = ctypes.create_string_buffer(1024)
        dst = ctypes.create_string_buffer(read_size)
        rc = self._context._lib.gfx_mtl_download_buffer(
            self._context._ctx_handle,
            self._handle,
            offset,
            ctypes.cast(dst, ctypes.c_void_p),
            read_size,
            err_buf,
            len(err_buf),
        )
        if rc != 0:
            raise PersistentComputeError(_decode_err(err_buf))
        return dst.raw

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handle = self._handle
        self._handle = 0
        if handle and not self._context.closed:
            self._context.synchronize()
            self._context._lib.gfx_mtl_destroy_buffer(handle)


class MetalPipeline(PersistentPipeline):
    """Reusable Metal compute pipeline compiled from DXIL-converted metallib."""

    def __init__(self, context: MetalComputeContext, program: ComputeProgram, handle: int) -> None:
        self._context = context
        self._program = program
        self._handle = handle
        self._closed = False

    @property
    def program(self) -> ComputeProgram:
        return self._program

    @property
    def context(self) -> MetalComputeContext:
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    def _ensure_open(self) -> None:
        self._context._ensure_open()
        if self._closed or not self._handle:
            raise PersistentComputeError("PersistentPipeline has been closed.")

    def dispatch(
        self,
        buffers: Mapping[str, PersistentBuffer],
        parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        self._ensure_open()
        active_grid = self._program.dispatch.grid if grid is None else tuple(grid)
        if len(active_grid) != 3 or any(type(dim) is not int or dim < 1 for dim in active_grid):
            raise PersistentComputeError("Compute dispatch grid must contain 3 positive integers.")
        gx, gy, gz = active_grid
        if gx > _METAL_MAX_GRID_DIM or gy > _METAL_MAX_GRID_DIM or gz > _METAL_MAX_GRID_DIM:
            raise PersistentComputeError(
                f"Compute dispatch grid {active_grid} exceeds Metal limit {_METAL_MAX_GRID_DIM}."
            )

        packed_params = _pack_dispatch_parameters(self._program, parameters)

        provided = dict(buffers or {})
        reflected_names = {buf["name"] for buf in self._program.buffers}
        extra = sorted(set(provided) - reflected_names)
        if extra:
            raise PersistentComputeError(f"Unknown compute buffer name(s): {', '.join(extra)}.")
        missing = sorted(reflected_names - set(provided))
        if missing:
            raise PersistentComputeError(f"Missing required compute buffer binding(s): {', '.join(missing)}.")

        binding_arr = (_GfxMtlBufferBinding * len(self._program.buffers))()
        for idx, buf_meta in enumerate(self._program.buffers):
            name = buf_meta["name"]
            buf = provided[name]
            if not isinstance(buf, MetalBuffer):
                raise PersistentComputeError(
                    f"Buffer {name!r} must be a MetalBuffer created by this context."
                )
            if buf.context is not self._context:
                raise PersistentComputeError(
                    f"Cross-device binding rejected for buffer {name!r}: "
                    "buffer belongs to a different ComputeContext."
                )
            buf._ensure_open()
            if buf.size_bytes <= 0:
                raise PersistentComputeError(f"Buffer {name!r} must have positive size.")
            is_uav = 1 if buf_meta["access"] in ("write_only", "read_write") else 0
            binding_arr[idx] = _GfxMtlBufferBinding(
                buffer=buf._handle,
                is_uav=is_uav,
                set=int(buf_meta.get("set", 0)),
                binding=int(buf_meta.get("binding", 0)),
            )

        rc_ptr = None
        rc_size = 0
        rc_buf = None
        if packed_params is not None and len(packed_params) > 0:
            rc_size = len(packed_params)
            rc_buf = (ctypes.c_uint8 * rc_size).from_buffer_copy(packed_params)
            rc_ptr = ctypes.cast(rc_buf, ctypes.c_void_p)

        err_buf = ctypes.create_string_buffer(1024)
        rc = self._context._lib.gfx_mtl_dispatch(
            self._context._ctx_handle,
            self._handle,
            binding_arr,
            len(self._program.buffers),
            rc_ptr,
            rc_size,
            gx,
            gy,
            gz,
            err_buf,
            len(err_buf),
        )
        if rc != 0:
            raise PersistentComputeError(_decode_err(err_buf))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        handle = self._handle
        self._handle = 0
        if handle and not self._context.closed:
            self._context.synchronize()
            self._context._lib.gfx_mtl_destroy_pipeline(handle)


class MetalComputeContext(ComputeContext):
    """Persistent macOS Metal compute context."""

    def __init__(self, *, validation: bool = False, provider: str | None = None) -> None:
        if sys.platform != "darwin":
            raise PersistentComputeError(
                f"Persistent Metal compute is only supported on macOS (darwin), not {sys.platform!r}."
            )
        _validate_provider(provider, "metal")
        self._lib = _load_bridge()
        self._closed = False
        self._buffers: weakref.WeakSet[MetalBuffer] = weakref.WeakSet()
        self._pipelines: weakref.WeakSet[MetalPipeline] = weakref.WeakSet()

        err_buf = ctypes.create_string_buffer(1024)
        ctx_handle = self._lib.gfx_mtl_create_context(err_buf, len(err_buf))
        if not ctx_handle:
            raise PersistentComputeError(_decode_err(err_buf))
        self._ctx_handle: int = ctx_handle
        raw_name = self._lib.gfx_mtl_context_device_name(ctx_handle)
        self._device_name = raw_name.decode("utf-8", errors="replace") if raw_name else "Metal Device"

    @property
    def target(self) -> str:
        return "metal"

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or not self._ctx_handle:
            raise PersistentComputeError("ComputeContext has been closed.")

    def create_buffer(
        self,
        size_bytes: int,
        initial_data: bytes | bytearray | memoryview | None = None,
    ) -> MetalBuffer:
        self._ensure_open()
        _validate_positive_int(size_bytes, "buffer size_bytes")
        data_ptr = None
        data_len = 0
        data_arr = None
        if initial_data is not None:
            if not isinstance(initial_data, (bytes, bytearray, memoryview)):
                raise PersistentComputeError("initial_data must be bytes, bytearray, or memoryview.")
            payload = bytes(initial_data)
            if len(payload) > size_bytes:
                raise PersistentComputeError(
                    f"initial_data length ({len(payload)}) exceeds buffer size ({size_bytes})."
                )
            data_len = len(payload)
            if data_len > 0:
                data_arr = (ctypes.c_uint8 * data_len).from_buffer_copy(payload)
                data_ptr = ctypes.cast(data_arr, ctypes.c_void_p)

        err_buf = ctypes.create_string_buffer(1024)
        handle = self._lib.gfx_mtl_create_buffer(
            self._ctx_handle,
            size_bytes,
            data_ptr,
            data_len,
            err_buf,
            len(err_buf),
        )
        if not handle:
            raise PersistentComputeError(_decode_err(err_buf))
        buf = MetalBuffer(self, size_bytes, handle)
        self._buffers.add(buf)
        return buf

    def compile_pipeline(
        self,
        program: ComputeProgram | Path | str,
        *,
        toolchain: Toolchain | None = None,
    ) -> MetalPipeline:
        self._ensure_open()
        loaded = load_compute_program(program) if isinstance(program, (Path, str)) else program
        if not isinstance(loaded, ComputeProgram):
            raise PersistentComputeError("compile_pipeline requires a ComputeProgram or manifest Path.")
        if loaded.target not in ("metal", "directx"):
            raise PersistentComputeError(
                f"MetalComputeContext requires 'metal' or 'directx' artifacts, got {loaded.target!r}."
            )

        wg_x, wg_y, wg_z = loaded.workgroup_size
        err_buf = ctypes.create_string_buffer(1024)
        with tempfile.TemporaryDirectory(prefix="mlir-gpu-metal-pipe-") as work:
            work_dir = Path(work)
            metallib_path = work_dir / "compute.metallib"
            if loaded.target == "directx":
                tools = toolchain or _preview_toolchain()
                source_dxil = work_dir / "compute.dxil"
                validated_dxil = work_dir / "compute.validated.dxil"
                source_dxil.write_bytes(loaded.compute.code)
                _run([tools.find("dxv"), str(source_dxil), "-o", str(validated_dxil)])
                _check_binary(validated_dxil, b"DXBC", 32)
                convert_dxil_to_metallib(
                    validated_dxil,
                    metallib_path,
                    loaded.compute.entry_point,
                    tools.find("metal-shaderconverter"),
                )
            else:
                metallib_path.write_bytes(loaded.compute.code)

            handle = self._lib.gfx_mtl_create_pipeline(
                self._ctx_handle,
                str(metallib_path).encode("utf-8"),
                loaded.compute.entry_point.encode("utf-8"),
                wg_x,
                wg_y,
                wg_z,
                err_buf,
                len(err_buf),
            )
        if not handle:
            raise PersistentComputeError(_decode_err(err_buf))
        pipe = MetalPipeline(self, loaded, handle)
        self._pipelines.add(pipe)
        return pipe

    def synchronize(self) -> None:
        self._ensure_open()
        err_buf = ctypes.create_string_buffer(1024)
        rc = self._lib.gfx_mtl_synchronize(self._ctx_handle, err_buf, len(err_buf))
        if rc != 0:
            raise PersistentComputeError(_decode_err(err_buf))

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._ctx_handle:
                self.synchronize()
        finally:
            for buf in list(self._buffers):
                buf.close()
            for pipe in list(self._pipelines):
                pipe.close()
            self._closed = True
            ctx_handle = self._ctx_handle
            self._ctx_handle = 0
            if ctx_handle:
                self._lib.gfx_mtl_destroy_context(ctx_handle)
