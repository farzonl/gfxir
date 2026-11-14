"""Persistent DirectX 12 compute runtime implementing the compute_runtime contract on Windows."""

from __future__ import annotations

from collections.abc import Mapping
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import weakref

from .compute_runtime import (
    ComputeContext,
    PersistentBuffer,
    PersistentComputeError,
    PersistentPipeline,
    _pack_dispatch_parameters,
    _validate_non_negative_int,
    _validate_positive_int,
)
from .viewer import ComputeProgram, _validate_provider, load_compute_program

_D3D12_MAX_GRID_DIM = 65535
_REPO_ROOT = Path(__file__).resolve().parents[2]
_BRIDGE_LIB: ctypes.CDLL | None = None


class _GfxDx12BufferBinding(ctypes.Structure):
    _fields_ = [
        ("buffer", ctypes.c_void_p),
        ("is_uav", ctypes.c_int),
    ]


def _find_or_build_bridge_dll() -> Path:
    """Locate prebuilt mlir-gpu-directx-compute.dll or compile it from source."""
    candidates: list[Path] = []
    tools_env = os.environ.get("GPU_TOOLS_BIN")
    if tools_env:
        candidates.append(Path(tools_env) / "mlir-gpu-directx-compute.dll")
    candidates.append(_REPO_ROOT / ".tools" / "bin" / "mlir-gpu-directx-compute.dll")
    candidates.append(_REPO_ROOT / "build" / "native" / "mlir-gpu-directx-compute.dll")

    src_cpp = _REPO_ROOT / "native" / "directx_compute_bridge.cpp"
    if not src_cpp.is_file():
        for cand in candidates:
            if cand.is_file():
                return cand
        raise PersistentComputeError(f"DirectX 12 compute bridge source not found: {src_cpp}")

    src_hash = hashlib.sha256(src_cpp.read_bytes()).hexdigest()[:12]
    out_dir = _REPO_ROOT / "build" / "directx_compute_cache"
    cached_dll = out_dir / f"directx_compute_{src_hash}.dll"
    if cached_dll.is_file():
        return cached_dll

    for cand in candidates:
        if cand.is_file() and cand.stat().st_mtime >= src_cpp.stat().st_mtime:
            return cand

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_prefix = f".tmp_{os.getpid()}_{src_hash}"
    tmp_dll = out_dir / f"{tmp_prefix}.dll"
    errors: list[str] = []

    def _cleanup_tmp() -> None:
        for ext in (".dll", ".lib", ".exp", ".obj", ".pdb", ".ilk", ".cmd"):
            cand = out_dir / f"{tmp_prefix}{ext}"
            if cand.exists():
                cand.unlink(missing_ok=True)

    clang_bin = shutil.which("clang++")
    if clang_bin:
        cmd = [
            clang_bin,
            "-std=c++20",
            "-O2",
            "-shared",
            str(src_cpp),
            "-ld3d12",
            "-ldxgi",
            "-o",
            str(tmp_dll),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and tmp_dll.is_file():
            tmp_dll.replace(cached_dll)
            _cleanup_tmp()
            return cached_dll
        errors.append(f"clang++ exited with {res.returncode}:\n{res.stderr or res.stdout}")
        _cleanup_tmp()

    cl_bin = shutil.which("cl") or shutil.which("cl.exe")
    if cl_bin:
        cmd = [
            cl_bin,
            "/nologo",
            "/std:c++20",
            "/O2",
            "/EHsc",
            "/LD",
            str(src_cpp),
            f"/Fo{out_dir / (tmp_prefix + '.obj')}",
            f"/Fe{tmp_dll}",
            "d3d12.lib",
            "dxgi.lib",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and tmp_dll.is_file():
            tmp_dll.replace(cached_dll)
            _cleanup_tmp()
            return cached_dll
        errors.append(f"cl.exe exited with {res.returncode}:\n{res.stderr or res.stdout}")
        _cleanup_tmp()

    pf_x86 = os.environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)"
    vswhere = Path(pf_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        vs_res = subprocess.run(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            capture_output=True,
            text=True,
        )
        vs_root = vs_res.stdout.strip()
        if vs_res.returncode == 0 and vs_root:
            vcvars = Path(vs_root) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.is_file():
                cmd_script = out_dir / f"{tmp_prefix}.cmd"
                cmd_script.write_text(
                    f'@call "{vcvars}" >nul\r\n@if errorlevel 1 exit /b %errorlevel%\r\n'
                    f'@cl.exe /nologo /std:c++20 /O2 /EHsc /LD "{src_cpp}" '
                    f'/Fo"{out_dir / (tmp_prefix + ".obj")}" /Fe"{tmp_dll}" d3d12.lib dxgi.lib\r\n',
                    encoding="utf-8",
                )
                res = subprocess.run(["cmd.exe", "/d", "/c", str(cmd_script)], capture_output=True, text=True)
                if res.returncode == 0 and tmp_dll.is_file():
                    tmp_dll.replace(cached_dll)
                    _cleanup_tmp()
                    return cached_dll
                errors.append(f"MSVC (vcvars64) exited with {res.returncode}:\n{res.stderr or res.stdout}")
                _cleanup_tmp()

    detail = "\n".join(errors) if errors else "Neither clang++ nor MSVC cl.exe (Visual Studio C++ tools) was found."
    raise PersistentComputeError(f"Failed to load or build DirectX 12 compute bridge:\n{detail}")


def _load_bridge() -> ctypes.CDLL:
    global _BRIDGE_LIB
    if _BRIDGE_LIB is not None:
        return _BRIDGE_LIB

    dll_path = _find_or_build_bridge_dll()
    try:
        lib = ctypes.CDLL(str(dll_path))
    except OSError as exc:
        raise PersistentComputeError(
            f"Failed to load DirectX 12 compute bridge DLL '{dll_path}': {exc}"
        ) from exc

    lib.gfx_dx12_create_device.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.gfx_dx12_create_device.restype = ctypes.c_void_p

    lib.gfx_dx12_device_name.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_device_name.restype = ctypes.c_char_p

    lib.gfx_dx12_device_is_software.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_device_is_software.restype = ctypes.c_int

    lib.gfx_dx12_last_error.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_last_error.restype = ctypes.c_char_p

    lib.gfx_dx12_destroy_device.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_destroy_device.restype = None

    lib.gfx_dx12_create_buffer.argtypes = [ctypes.c_void_p, ctypes.c_uint64]
    lib.gfx_dx12_create_buffer.restype = ctypes.c_void_p

    lib.gfx_dx12_buffer_size.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_buffer_size.restype = ctypes.c_uint64

    lib.gfx_dx12_upload_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    lib.gfx_dx12_upload_buffer.restype = ctypes.c_int

    lib.gfx_dx12_download_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint64,
        ctypes.c_uint64,
    ]
    lib.gfx_dx12_download_buffer.restype = ctypes.c_int

    lib.gfx_dx12_destroy_buffer.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_destroy_buffer.restype = None

    lib.gfx_dx12_create_pipeline.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    lib.gfx_dx12_create_pipeline.restype = ctypes.c_void_p

    lib.gfx_dx12_pipeline_root_constants_bytes.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_pipeline_root_constants_bytes.restype = ctypes.c_uint32

    lib.gfx_dx12_destroy_pipeline.argtypes = [ctypes.c_void_p]
    lib.gfx_dx12_destroy_pipeline.restype = None

    lib.gfx_dx12_dispatch.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(_GfxDx12BufferBinding),
        ctypes.c_size_t,
    ]
    lib.gfx_dx12_dispatch.restype = ctypes.c_int

    _BRIDGE_LIB = lib
    return lib


class DirectXBuffer(PersistentBuffer):
    """Persistent DirectX 12 GPU buffer owned by a DirectXComputeContext."""

    def __init__(self, context: DirectXComputeContext, handle: int, size_bytes: int) -> None:
        self._context = context
        self._handle: int | None = handle
        self._size_bytes = size_bytes
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed or self._handle is None

    @property
    def context(self) -> DirectXComputeContext:
        return self._context

    def _ensure_open(self) -> int:
        self._context._ensure_open()
        if self._closed or self._handle is None:
            raise PersistentComputeError("PersistentBuffer has been closed.")
        return self._handle

    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        buf_handle = self._ensure_open()
        dev_handle = self._context._ensure_open()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise PersistentComputeError("upload data must be bytes, bytearray, or memoryview.")
        _validate_non_negative_int(offset, "upload offset")
        payload = bytes(data)
        if offset + len(payload) > self._size_bytes:
            raise PersistentComputeError(
                f"Upload range [{offset}, {offset + len(payload)}) exceeds buffer size {self._size_bytes}."
            )
        if not payload:
            return
        c_buf = (ctypes.c_ubyte * len(payload)).from_buffer_copy(payload)
        rc = self._context._lib.gfx_dx12_upload_buffer(
            ctypes.c_void_p(dev_handle),
            ctypes.c_void_p(buf_handle),
            ctypes.cast(c_buf, ctypes.c_void_p),
            ctypes.c_uint64(len(payload)),
            ctypes.c_uint64(offset),
        )
        if rc != 0:
            raise PersistentComputeError(
                f"DirectX 12 buffer upload failed: {self._context._last_error()}"
            )

    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        buf_handle = self._ensure_open()
        dev_handle = self._context._ensure_open()
        _validate_non_negative_int(offset, "download offset")
        if offset > self._size_bytes:
            raise PersistentComputeError(
                f"Download offset {offset} exceeds buffer size {self._size_bytes}."
            )
        read_size = (
            (self._size_bytes - offset)
            if size_bytes is None
            else _validate_non_negative_int(size_bytes, "download size_bytes")
        )
        if offset + read_size > self._size_bytes:
            raise PersistentComputeError(
                f"Download range [{offset}, {offset + read_size}) exceeds buffer size {self._size_bytes}."
            )
        if read_size == 0:
            return b""
        out_arr = (ctypes.c_ubyte * read_size)()
        rc = self._context._lib.gfx_dx12_download_buffer(
            ctypes.c_void_p(dev_handle),
            ctypes.c_void_p(buf_handle),
            ctypes.cast(out_arr, ctypes.c_void_p),
            ctypes.c_uint64(read_size),
            ctypes.c_uint64(offset),
        )
        if rc != 0:
            raise PersistentComputeError(
                f"DirectX 12 buffer download failed: {self._context._last_error()}"
            )
        return bytes(out_arr)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            handle = self._handle
            self._handle = None
            if handle is not None and self._context._lib is not None:
                self._context._lib.gfx_dx12_destroy_buffer(ctypes.c_void_p(handle))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DirectXPipeline(PersistentPipeline):
    """Compiled DirectX 12 compute pipeline bound to a DirectXComputeContext."""

    def __init__(
        self,
        context: DirectXComputeContext,
        handle: int,
        program: ComputeProgram,
    ) -> None:
        self._context = context
        self._handle: int | None = handle
        self._program = program
        self._closed = False

    @property
    def program(self) -> ComputeProgram:
        return self._program

    @property
    def context(self) -> DirectXComputeContext:
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed or self._handle is None

    def _ensure_open(self) -> int:
        self._context._ensure_open()
        if self._closed or self._handle is None:
            raise PersistentComputeError("PersistentPipeline has been closed.")
        return self._handle

    def dispatch(
        self,
        buffers: Mapping[str, PersistentBuffer],
        parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        pipe_handle = self._ensure_open()
        dev_handle = self._context._ensure_open()

        active_grid = self._program.dispatch.grid if grid is None else tuple(grid)
        if (
            len(active_grid) != 3
            or any(type(dim) is not int or dim < 1 or dim > _D3D12_MAX_GRID_DIM for dim in active_grid)
        ):
            raise PersistentComputeError(
                f"Compute dispatch grid must contain 3 integers in [1, {_D3D12_MAX_GRID_DIM}]."
            )

        packed_params = _pack_dispatch_parameters(self._program, parameters)

        if not isinstance(buffers, Mapping):
            raise PersistentComputeError("dispatch buffers must be a mapping of buffer names to PersistentBuffer.")

        provided = dict(buffers)
        reflected_names = {buf["name"] for buf in self._program.buffers}
        extra = sorted(set(provided) - reflected_names)
        if extra:
            raise PersistentComputeError(f"Unknown compute buffer name(s): {', '.join(extra)}.")
        missing = sorted(reflected_names - set(provided))
        if missing:
            raise PersistentComputeError(f"Missing required compute buffer binding(s): {', '.join(missing)}.")

        sorted_meta = sorted(
            self._program.buffers,
            key=lambda meta: (int(meta.get("set", 0)), int(meta.get("binding", 0))),
        )

        bound_access: dict[int, tuple[str, bool]] = {}
        resolved: list[tuple[int, bool]] = []
        for meta in sorted_meta:
            name = meta["name"]
            access = meta["access"]
            if access not in ("read_only", "write_only", "read_write"):
                raise PersistentComputeError(
                    f"Unsupported compute buffer access mode {access!r} for {name!r}."
                )
            buf = provided[name]
            if not isinstance(buf, DirectXBuffer):
                raise PersistentComputeError(
                    f"Buffer {name!r} is a {type(buf).__name__}, expected DirectXBuffer."
                )
            if buf.context is not self._context:
                raise PersistentComputeError(
                    f"Cross-device binding rejected for buffer {name!r}: "
                    f"buffer belongs to {buf.context.device_name!r}, dispatch context is "
                    f"{self._context.device_name!r}."
                )
            buf_handle = buf._ensure_open()
            is_uav = access != "read_only"
            buf_id = id(buf)
            if buf_id in bound_access:
                prev_name, prev_uav = bound_access[buf_id]
                if prev_uav != is_uav:
                    raise PersistentComputeError(
                        f"Buffer bound to {prev_name!r} and {name!r} cannot be both read_only and "
                        "writable in a single dispatch."
                    )
            else:
                bound_access[buf_id] = (name, is_uav)
            resolved.append((buf_handle, is_uav))

        if packed_params:
            c_params = (ctypes.c_ubyte * len(packed_params)).from_buffer_copy(packed_params)
            params_ptr = ctypes.cast(c_params, ctypes.c_void_p)
            params_len = len(packed_params)
        else:
            c_params = None
            params_ptr = ctypes.c_void_p(0)
            params_len = 0

        if resolved:
            binding_array = (_GfxDx12BufferBinding * len(resolved))()
            for idx, (buf_handle, is_uav) in enumerate(resolved):
                binding_array[idx].buffer = ctypes.c_void_p(buf_handle)
                binding_array[idx].is_uav = 1 if is_uav else 0
            bindings_ptr = ctypes.cast(binding_array, ctypes.POINTER(_GfxDx12BufferBinding))
            bindings_len = len(resolved)
        else:
            binding_array = None
            bindings_ptr = ctypes.POINTER(_GfxDx12BufferBinding)()
            bindings_len = 0

        rc = self._context._lib.gfx_dx12_dispatch(
            ctypes.c_void_p(dev_handle),
            ctypes.c_void_p(pipe_handle),
            ctypes.c_uint32(active_grid[0]),
            ctypes.c_uint32(active_grid[1]),
            ctypes.c_uint32(active_grid[2]),
            params_ptr,
            ctypes.c_size_t(params_len),
            bindings_ptr,
            ctypes.c_size_t(bindings_len),
        )
        if rc != 0:
            raise PersistentComputeError(
                f"DirectX 12 compute dispatch failed: {self._context._last_error()}"
            )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            handle = self._handle
            self._handle = None
            if handle is not None and self._context._lib is not None:
                self._context._lib.gfx_dx12_destroy_pipeline(ctypes.c_void_p(handle))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class DirectXComputeContext(ComputeContext):
    """In-process persistent DirectX 12 ComputeContext and resource owner.

    Blocking semantics: ``upload``, ``download``, and ``dispatch`` submit their
    D3D12 command lists and wait on the queue fence before returning, so host
    buffers can be reused immediately and subsequent dispatches or downloads
    observe all prior writes. ``synchronize()`` is therefore a fast no-op check
    once open.
    """

    def __init__(
        self,
        *,
        validation: bool = False,
        provider: str | None = None,
    ) -> None:
        if sys.platform != "win32":
            raise PersistentComputeError("Native DirectX 12 compute execution requires Windows.")

        explicit_warp = provider == "warp"
        if provider is None and os.environ.get("MLIR_GPU_TEST_WARP") == "1":
            provider = "warp"
        try:
            _validate_provider(provider, "directx")
        except ValueError as exc:
            raise PersistentComputeError(str(exc)) from exc

        self._lib = _load_bridge()
        use_warp = 1 if provider == "warp" else 0
        enable_val = 1 if validation else 0
        err_buf = ctypes.create_string_buffer(2048)
        handle = self._lib.gfx_dx12_create_device(use_warp, enable_val, err_buf, len(err_buf))
        if not handle:
            detail = err_buf.value.decode("utf-8", errors="replace").strip() or "unknown error"
            raise PersistentComputeError(f"Failed to initialize DirectX 12 compute device: {detail}")

        self._handle: int | None = int(handle)
        self._closed = False
        self._buffers: weakref.WeakSet[DirectXBuffer] = weakref.WeakSet()
        self._pipelines: weakref.WeakSet[DirectXPipeline] = weakref.WeakSet()

        raw_name = self._lib.gfx_dx12_device_name(ctypes.c_void_p(self._handle))
        self._device_name = raw_name.decode("utf-8", errors="replace") if raw_name else "DirectX 12"
        self._is_software = bool(self._lib.gfx_dx12_device_is_software(ctypes.c_void_p(self._handle)))
        is_warp_name = "basic render driver" in self._device_name.lower()

        if (explicit_warp or provider == "warp") and not (self._is_software or is_warp_name):
            self.close()
            raise PersistentComputeError(
                "Requested provider 'warp' but returned device is not a software adapter. "
                "Accidental hardware/driver fallback rejected."
            )
        if provider == "hardware" and (self._is_software or is_warp_name):
            self.close()
            raise PersistentComputeError(
                "Requested provider 'hardware' but returned device is a software adapter. "
                "Accidental hardware/driver fallback rejected."
            )

        self._provider = "warp" if (self._is_software or is_warp_name) else "hardware"

    @property
    def target(self) -> str:
        return "directx"

    @property
    def device_name(self) -> str:
        return self._device_name

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def is_software(self) -> bool:
        return self._is_software

    @property
    def closed(self) -> bool:
        return self._closed or self._handle is None

    def _ensure_open(self) -> int:
        if self._closed or self._handle is None:
            raise PersistentComputeError("ComputeContext has been closed.")
        return self._handle

    def _last_error(self) -> str:
        if self._handle is None or self._lib is None:
            return "DirectX 12 compute context is closed"
        raw = self._lib.gfx_dx12_last_error(ctypes.c_void_p(self._handle))
        return raw.decode("utf-8", errors="replace") if raw else "unknown DirectX 12 error"

    def create_buffer(
        self,
        size_bytes: int,
        initial_data: bytes | bytearray | memoryview | None = None,
    ) -> DirectXBuffer:
        dev_handle = self._ensure_open()
        validated = _validate_positive_int(size_bytes, "size_bytes")
        buf_handle = self._lib.gfx_dx12_create_buffer(
            ctypes.c_void_p(dev_handle), ctypes.c_uint64(validated)
        )
        if not buf_handle:
            raise PersistentComputeError(
                f"DirectX 12 buffer allocation failed: {self._last_error()}"
            )
        buf = DirectXBuffer(self, int(buf_handle), validated)
        self._buffers.add(buf)
        if initial_data is not None:
            try:
                buf.upload(initial_data)
            except Exception:
                buf.close()
                raise
        return buf

    def compile_pipeline(self, program: ComputeProgram | Path | str) -> DirectXPipeline:
        dev_handle = self._ensure_open()
        loaded = (
            program
            if isinstance(program, ComputeProgram)
            else load_compute_program(Path(program))
            if isinstance(program, (Path, str))
            else None
        )
        if loaded is None:
            raise PersistentComputeError(
                "compile_pipeline requires a ComputeProgram or manifest Path."
            )
        if loaded.target != "directx":
            raise PersistentComputeError(
                "DirectX 12 compute pipeline requires DirectX DXIL artifacts."
            )
        dxil = loaded.compute.code
        if not dxil:
            raise PersistentComputeError("ComputeProgram DXIL bytecode is empty.")
        c_dxil = (ctypes.c_ubyte * len(dxil)).from_buffer_copy(dxil)
        pipe_handle = self._lib.gfx_dx12_create_pipeline(
            ctypes.c_void_p(dev_handle),
            ctypes.cast(c_dxil, ctypes.c_void_p),
            ctypes.c_size_t(len(dxil)),
        )
        if not pipe_handle:
            raise PersistentComputeError(
                f"DirectX 12 compute pipeline creation failed: {self._last_error()}"
            )
        pipe = DirectXPipeline(self, int(pipe_handle), loaded)
        self._pipelines.add(pipe)
        return pipe

    def synchronize(self) -> None:
        self._ensure_open()

    def close(self) -> None:
        if not self._closed:
            for buf in list(self._buffers):
                buf.close()
            for pipe in list(self._pipelines):
                pipe.close()
            self._closed = True
            handle = self._handle
            self._handle = None
            if handle is not None and self._lib is not None:
                self._lib.gfx_dx12_destroy_device(ctypes.c_void_p(handle))

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
