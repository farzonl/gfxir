"""Persistent Metal compute runtime implementing the compute_runtime contract on macOS."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
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
    _BoundedBufferPool,
    _pack_dispatch_parameters,
    _resolve_pool_max_bytes,
    _validate_non_negative_int,
    _validate_positive_int,
)
from .viewer import ComputeProgram, _check_binary, _preview_toolchain, _run, _validate_provider, load_compute_program

# Metal dispatchThreadgroups uses MTLSize (NSUInteger) without D3D12's 65535 cap;
# gfx_mtl_dispatch accepts uint32_t grid dimensions.
_METAL_MAX_GRID_DIM = 0xFFFFFFFF
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

    lib.gfx_mtl_begin_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.gfx_mtl_begin_batch.restype = ctypes.c_int

    lib.gfx_mtl_end_batch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.gfx_mtl_end_batch.restype = ctypes.c_int

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
        if self._context._pending_work:
            self._context.synchronize()
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
        if self._context._pending_work:
            self._context.synchronize()
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
        self._context._buffers.discard(self)
        handle = self._handle
        self._handle = 0
        if not handle:
            return
        self._context._pool.on_live_free(self._size_bytes)
        if self._context.closed:
            self._context._lib.gfx_mtl_destroy_buffer(handle)
        elif self._context._pending_work:
            self._context._defer_cleanup("pool_buffer", (self._size_bytes, handle))
        else:
            self._context._return_buffer_to_pool(self._size_bytes, handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MetalPipeline(PersistentPipeline):
    """Reusable Metal compute pipeline compiled from DXIL-converted metallib."""

    def __init__(self, context: MetalComputeContext, program: ComputeProgram, handle: int) -> None:
        self._context = context
        self._program = program
        self._handle = handle
        self._closed = False
        self._binding_arr = (_GfxMtlBufferBinding * len(program.buffers))()
        self._binding_meta = tuple(
            (
                buf_meta["name"],
                1 if buf_meta["access"] in ("write_only", "read_write") else 0,
                int(buf_meta.get("set", 0)),
                int(buf_meta.get("binding", 0)),
            )
            for buf_meta in program.buffers
        )
        self._reflected_names = frozenset(buf["name"] for buf in program.buffers)
        self._err_buf = ctypes.create_string_buffer(1024)

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

        provided = buffers or {}
        if len(provided) != len(self._reflected_names) or any(k not in self._reflected_names for k in provided):
            extra = sorted(set(provided) - self._reflected_names)
            if extra:
                raise PersistentComputeError(f"Unknown compute buffer name(s): {', '.join(extra)}.")
            missing = sorted(self._reflected_names - set(provided))
            if missing:
                raise PersistentComputeError(f"Missing required compute buffer binding(s): {', '.join(missing)}.")

        binding_arr = self._binding_arr
        for idx, (name, is_uav, set_idx, binding_idx) in enumerate(self._binding_meta):
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
            binding_arr[idx].buffer = buf._handle
            binding_arr[idx].is_uav = is_uav
            binding_arr[idx].set = set_idx
            binding_arr[idx].binding = binding_idx

        rc_ptr = None
        rc_size = 0
        rc_buf = None
        if packed_params is not None and len(packed_params) > 0:
            rc_size = len(packed_params)
            rc_buf = (ctypes.c_uint8 * rc_size).from_buffer_copy(packed_params)
            rc_ptr = ctypes.cast(rc_buf, ctypes.c_void_p)

        err_buf = self._err_buf
        rc = self._context._lib.gfx_mtl_dispatch(
            self._context._ctx_handle,
            self._handle,
            binding_arr,
            len(self._binding_meta),
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
        self._context._pending_work = True
        self._context._stats["dispatches"] += 1
        if self._context._batch_depth == 0:
            self._context._stats["submissions"] += 1
        else:
            self._context._batch_dispatches += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context._pipelines.discard(self)
        handle = self._handle
        self._handle = 0
        if handle and not self._context.closed and self._context._pending_work:
            self._context._defer_cleanup("pipeline", handle)
        elif handle:
            self._context._lib.gfx_mtl_destroy_pipeline(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class MetalComputeContext(ComputeContext):
    """Persistent macOS Metal compute context."""

    def __init__(
        self,
        *,
        validation: bool = False,
        provider: str | None = None,
        pool_max_bytes: int | None = None,
    ) -> None:
        if sys.platform != "darwin":
            raise PersistentComputeError(
                f"Persistent Metal compute is only supported on macOS (darwin), not {sys.platform!r}."
            )
        _validate_provider(provider, "metal")
        self._lib = _load_bridge()
        self._closed = False
        self._buffers: weakref.WeakSet[MetalBuffer] = weakref.WeakSet()
        self._pipelines: weakref.WeakSet[MetalPipeline] = weakref.WeakSet()
        self._pending_work = False
        self._batch_depth = 0
        self._batch_dispatches = 0
        self._retired_resources: list[tuple[str, int]] = []
        self._stats = {
            "dispatches": 0,
            "submissions": 0,
            "waits": 0,
            "deferred_cleanups": 0,
        }
        self._pool = _BoundedBufferPool(max_bytes=_resolve_pool_max_bytes(pool_max_bytes))

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
    def provider(self) -> str:
        return "hardware"

    @property
    def is_software(self) -> bool:
        return False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or not self._ctx_handle:
            raise PersistentComputeError("ComputeContext has been closed.")

    def _return_buffer_to_pool(self, size_bytes: int, handle: int) -> None:
        if self._closed:
            self._lib.gfx_mtl_destroy_buffer(handle)
            return
        self._pool.release(size_bytes, handle, self._lib.gfx_mtl_destroy_buffer)

    def create_buffer(
        self,
        size_bytes: int,
        initial_data: bytes | bytearray | memoryview | None = None,
        *,
        zero_init: bool = True,
    ) -> MetalBuffer:
        self._ensure_open()
        _validate_positive_int(size_bytes, "buffer size_bytes")
        payload: bytes | None = None
        if initial_data is not None:
            if not isinstance(initial_data, (bytes, bytearray, memoryview)):
                raise PersistentComputeError("initial_data must be bytes, bytearray, or memoryview.")
            payload = bytes(initial_data)
            if len(payload) > size_bytes:
                raise PersistentComputeError(
                    f"initial_data length ({len(payload)}) exceeds buffer size ({size_bytes})."
                )

        pooled_handle = self._pool.acquire(size_bytes)
        if pooled_handle is not None:
            buf = MetalBuffer(self, size_bytes, int(pooled_handle))  # type: ignore[arg-type]
            self._pool.on_live_alloc(size_bytes)
            try:
                if payload is not None:
                    if len(payload) > 0:
                        buf.upload(payload)
                    if zero_init and len(payload) < size_bytes:
                        buf.upload(b"\x00" * (size_bytes - len(payload)), offset=len(payload))
                elif zero_init:
                    buf.upload(b"\x00" * size_bytes)
            except Exception:
                buf._closed = True
                buf._handle = 0
                self._pool.on_live_free(size_bytes)
                self._lib.gfx_mtl_destroy_buffer(int(pooled_handle))  # type: ignore[arg-type]
                raise
            self._buffers.add(buf)
            return buf

        data_ptr = None
        data_len = 0
        data_arr = None
        if payload is not None:
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
        self._pool.record_allocation()
        self._pool.on_live_alloc(size_bytes)
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

    @property
    def stats(self) -> dict[str, int]:
        merged = dict(self._stats)
        merged.update(self._pool.stats())
        return merged

    def reset_stats(self) -> None:
        for key in self._stats:
            self._stats[key] = 0
        self._pool.reset_counters()

    def release_pool(self) -> int:
        self._ensure_open()
        if self._pending_work:
            self.synchronize()
        return self._pool.clear(self._lib.gfx_mtl_destroy_buffer)

    def _defer_cleanup(self, kind: str, handle: object) -> None:
        self._retired_resources.append((kind, handle))
        self._stats["deferred_cleanups"] += 1

    def _flush_retired(self) -> None:
        if not self._retired_resources:
            return
        retired = self._retired_resources
        self._retired_resources = []
        remaining: list[tuple[str, object]] = []
        errors: list[BaseException] = []
        for kind, handle in retired:
            try:
                if kind == "buffer":
                    self._lib.gfx_mtl_destroy_buffer(handle)
                elif kind == "pool_buffer":
                    size_bytes, buf_handle = handle  # type: ignore[misc]
                    self._return_buffer_to_pool(size_bytes, buf_handle)
                elif kind == "pipeline":
                    self._lib.gfx_mtl_destroy_pipeline(handle)
            except BaseException as exc:
                errors.append(exc)
                remaining.append((kind, handle))
        if remaining:
            self._retired_resources = remaining + self._retired_resources
        if errors:
            raise errors[0]

    @contextmanager
    def batch(self):
        self._ensure_open()
        if self._batch_depth == 0:
            self._batch_dispatches = 0
            begin_fn = getattr(self._lib, "gfx_mtl_begin_batch", None)
            if begin_fn is not None:
                err_buf = ctypes.create_string_buffer(1024)
                rc = begin_fn(self._ctx_handle, err_buf, len(err_buf))
                if rc != 0:
                    raise PersistentComputeError(_decode_err(err_buf))
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and not self._closed:
                end_fn = getattr(self._lib, "gfx_mtl_end_batch", None)
                if end_fn is not None:
                    err_buf = ctypes.create_string_buffer(1024)
                    rc = end_fn(self._ctx_handle, err_buf, len(err_buf))
                    if rc != 0:
                        raise PersistentComputeError(_decode_err(err_buf))
                if self._batch_dispatches > 0:
                    self._stats["submissions"] += 1
                    self._batch_dispatches = 0

    def synchronize(self) -> None:
        self._ensure_open()
        had_pending = self._pending_work
        err_buf = ctypes.create_string_buffer(1024)
        rc = self._lib.gfx_mtl_synchronize(self._ctx_handle, err_buf, len(err_buf))
        if rc != 0:
            raise PersistentComputeError(_decode_err(err_buf))
        self._pending_work = False
        if had_pending:
            self._stats["waits"] += 1
        self._flush_retired()

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        if self._ctx_handle:
            try:
                self.synchronize()
            except BaseException as exc:
                errors.append(exc)
        self._pending_work = False
        try:
            self._flush_retired()
        except BaseException as exc:
            errors.append(exc)
        for buf in list(self._buffers):
            try:
                buf.close()
            except BaseException as exc:
                errors.append(exc)
        for pipe in list(self._pipelines):
            try:
                pipe.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self._pool.clear(self._lib.gfx_mtl_destroy_buffer)
        except BaseException as exc:
            errors.append(exc)
        self._buffers.clear()
        self._pipelines.clear()
        self._closed = True
        ctx_handle = self._ctx_handle
        self._ctx_handle = 0
        if ctx_handle:
            try:
                self._lib.gfx_mtl_destroy_context(ctx_handle)
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
