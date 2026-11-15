"""Persistent GfxIR GPU backend adapter for gfxpy."""

from __future__ import annotations

from pathlib import Path
import tempfile
from typing import Any

from mlir_gpu import (
    ComputePipeline,
    Dispatch,
    Target,
    Toolchain,
    compile_pipeline,
)
from mlir_gpu.compute_runtime import (
    ComputeContext,
    PersistentBuffer,
    PersistentPipeline,
    VulkanComputeContext,
)
from mlir_gpu.viewer import _preview_toolchain

from .kernels import BINARY_KERNELS, SCALAR_KERNELS, WORKGROUP_SIZE_1D, compute_grid_1d
from .runtime import ComputeBackend, ComputeBuffer
from .targets import BackendInitializationError

_TARGET_MAP = {
    "vulkan": Target.VULKAN,
    "directx": Target.DIRECTX,
    "metal": Target.METAL,
}


def _open_context(
    target: str,
    *,
    validation: bool = False,
    provider: str | None = None,
) -> ComputeContext:
    if target == "vulkan":
        return VulkanComputeContext(validation=validation, provider=provider)
    if target == "directx":
        from mlir_gpu.directx_compute import DirectXComputeContext

        return DirectXComputeContext(validation=validation, provider=provider)
    if target == "metal":
        from mlir_gpu.metal_compute import MetalComputeContext

        return MetalComputeContext(validation=validation, provider=provider)
    raise BackendInitializationError(target, f"Unsupported target {target!r}.")


class GfxIRComputeBackend(ComputeBackend):
    """Device-resident ComputeBackend backed by mlir_gpu persistent ComputeContext."""

    def __init__(
        self,
        target: str,
        *,
        validation: bool = False,
        provider: str | None = None,
        toolchain: Toolchain | None = None,
    ) -> None:
        self._target = target
        self._closed = False
        self._toolchain = toolchain or _preview_toolchain()
        self._artifact_dir = tempfile.TemporaryDirectory(prefix=f"gfxpy-{target}-")
        self._pipelines: dict[tuple[object, ...], PersistentPipeline] = {}

        try:
            self._ctx = _open_context(target, validation=validation, provider=provider)
        except BackendInitializationError:
            self._artifact_dir.cleanup()
            raise
        except Exception as exc:
            self._artifact_dir.cleanup()
            raise BackendInitializationError(target, exc) from exc

    @property
    def target(self) -> str:
        return self._target

    @property
    def device_name(self) -> str:
        return self._ctx.device_name

    @property
    def closed(self) -> bool:
        return self._closed or self._ctx.closed

    @property
    def compiled_pipeline_count(self) -> int:
        return len(self._pipelines)

    def _pipeline_cache_key(self, op_key: str) -> tuple[object, ...]:
        return (
            self._target,
            op_key,
            "float32",
            WORKGROUP_SIZE_1D,
            id(self._toolchain),
        )

    def _get_or_compile(self, op_key: str, kernel: Any) -> PersistentPipeline:
        cache_key = self._pipeline_cache_key(op_key)
        cached = self._pipelines.get(cache_key)
        if cached is not None and not cached.closed:
            return cached

        out_dir = Path(self._artifact_dir.name) / op_key
        manifest = compile_pipeline(
            ComputePipeline(kernel, Dispatch(grid=(1, 1, 1))),
            _TARGET_MAP[self._target],
            out_dir,
            self._toolchain,
        )
        pipe = self._ctx.compile_pipeline(manifest)
        self._pipelines[cache_key] = pipe
        return pipe

    def allocate(self, size_bytes: int) -> ComputeBuffer:
        return self._ctx.create_buffer(size_bytes)

    def upload(self, buf: ComputeBuffer, data: bytes) -> None:
        assert isinstance(buf, PersistentBuffer)
        buf.upload(data)

    def download(self, buf: ComputeBuffer) -> bytes:
        assert isinstance(buf, PersistentBuffer)
        return buf.download()

    def binary_op(
        self,
        op: str,
        lhs: ComputeBuffer,
        rhs: ComputeBuffer,
        out: ComputeBuffer,
        length: int,
    ) -> None:
        if op not in BINARY_KERNELS:
            raise ValueError(f"Unsupported binary op {op!r}.")
        assert isinstance(lhs, PersistentBuffer)
        assert isinstance(rhs, PersistentBuffer)
        assert isinstance(out, PersistentBuffer)
        pipe = self._get_or_compile(f"binary_{op}", BINARY_KERNELS[op])
        pipe.dispatch(
            {"lhs": lhs, "rhs": rhs, "out": out},
            parameters={"n": length},
            grid=compute_grid_1d(length),
        )

    def scalar_op(
        self,
        op: str,
        lhs: ComputeBuffer,
        scalar: float,
        out: ComputeBuffer,
        length: int,
    ) -> None:
        if op not in SCALAR_KERNELS:
            raise ValueError(f"Unsupported scalar op {op!r}.")
        assert isinstance(lhs, PersistentBuffer)
        assert isinstance(out, PersistentBuffer)
        pipe = self._get_or_compile(f"scalar_{op}", SCALAR_KERNELS[op])
        pipe.dispatch(
            {"lhs": lhs, "out": out},
            parameters={"n": length, "scalar": float(scalar)},
            grid=compute_grid_1d(length),
        )

    def synchronize(self) -> None:
        self._ctx.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            for pipe in self._pipelines.values():
                pipe.close()
            self._pipelines.clear()
            self._ctx.close()
        finally:
            self._artifact_dir.cleanup()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
