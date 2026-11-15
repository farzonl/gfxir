"""Persistent GfxIR GPU backend adapter for gfxpy."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
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

from .kernels import (
    BINARY_KERNELS,
    SCALAR_KERNELS,
    TILE_DIM,
    WORKGROUP_SIZE_1D,
    compute_grid_1d,
    compute_matmul_grid,
    reduce_sum_f32_kernel,
    tiled_matmul_f32_kernel,
)
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
        self._validation = validation
        self._provider = provider
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

    def _toolchain_key(self) -> tuple[str | None, str | None]:
        llvm_bin = getattr(self._toolchain, "llvm_bin", None)
        tools_bin = getattr(self._toolchain, "tools_bin", None)
        return (
            str(llvm_bin) if llvm_bin is not None else None,
            str(tools_bin) if tools_bin is not None else None,
        )

    def _pipeline_cache_key(
        self,
        op_key: str,
        workgroup_size: tuple[int, int, int] = WORKGROUP_SIZE_1D,
        *,
        dtype: str = "float32",
        layout: str = "contiguous",
        contraction_policy: str = "no_fast_math",
    ) -> tuple[object, ...]:
        return (
            self._target,
            op_key,
            dtype,
            layout,
            workgroup_size,
            self._validation,
            self._provider,
            contraction_policy,
            id(self._toolchain),
            self._toolchain_key(),
        )

    def _get_or_compile(
        self,
        op_key: str,
        kernel: Any,
        workgroup_size: tuple[int, int, int] = WORKGROUP_SIZE_1D,
        *,
        cache_key: tuple[object, ...] | None = None,
        artifact_subdir: str | None = None,
    ) -> PersistentPipeline:
        resolved_key = cache_key or self._pipeline_cache_key(op_key, workgroup_size)
        cached = self._pipelines.get(resolved_key)
        if cached is not None and not cached.closed:
            return cached

        out_dir = Path(self._artifact_dir.name) / (artifact_subdir or op_key)
        manifest = compile_pipeline(
            ComputePipeline(kernel, Dispatch(grid=(1, 1, 1))),
            _TARGET_MAP[self._target],
            out_dir,
            self._toolchain,
        )
        pipe = self._ctx.compile_pipeline(manifest)
        self._pipelines[resolved_key] = pipe
        return pipe

    def allocate(self, size_bytes: int) -> ComputeBuffer:
        return self._ctx.create_buffer(size_bytes)

    def allocate_uninitialized(self, size_bytes: int) -> ComputeBuffer:
        try:
            return self._ctx.create_buffer(size_bytes, zero_init=False)
        except TypeError:
            return self._ctx.create_buffer(size_bytes)

    def release_pool(self) -> int:
        return self._ctx.release_pool()

    @property
    def memory_stats(self) -> dict[str, int]:
        return self._ctx.stats

    def upload(self, buf: ComputeBuffer, data: bytes) -> None:
        assert isinstance(buf, PersistentBuffer)
        buf.upload(data)

    def download_into(self, buf: ComputeBuffer, dst: bytearray | memoryview) -> int:
        assert isinstance(buf, PersistentBuffer)
        return buf.download_into(dst)

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

    def fused_op(
        self,
        spec: Any,
        inputs: Sequence[ComputeBuffer],
        scalars: Sequence[float],
        out: ComputeBuffer,
        length: int,
    ) -> None:
        from .fusion import build_fused_compute_kernel

        assert isinstance(out, PersistentBuffer)
        bindings: dict[str, PersistentBuffer] = {"out": out}
        for idx, buf in enumerate(inputs):
            assert isinstance(buf, PersistentBuffer)
            bindings[f"in{idx}"] = buf

        op_key = f"fused:{spec.signature}"
        cache_key = self._pipeline_cache_key(
            op_key,
            WORKGROUP_SIZE_1D,
            dtype=spec.dtype,
            layout=spec.layout,
            contraction_policy=spec.contraction_policy,
        )
        pipe = self._pipelines.get(cache_key)
        if pipe is None or pipe.closed:
            digest = hashlib.sha256(spec.signature.encode("utf-8")).hexdigest()[:16]
            subdir = f"fused_{digest}"
            kernel = build_fused_compute_kernel(
                spec,
                Path(self._artifact_dir.name) / "_fused_sources",
            )
            pipe = self._get_or_compile(
                op_key,
                kernel,
                WORKGROUP_SIZE_1D,
                cache_key=cache_key,
                artifact_subdir=subdir,
            )

        params: dict[str, object] = {"n": length}
        for idx, val in enumerate(scalars):
            params[f"s{idx}"] = float(val)
        pipe.dispatch(bindings, parameters=params, grid=compute_grid_1d(length))

    def reduce_sum_pass(
        self,
        inp: ComputeBuffer,
        out: ComputeBuffer,
        length: int,
    ) -> None:
        assert isinstance(inp, PersistentBuffer)
        assert isinstance(out, PersistentBuffer)
        pipe = self._get_or_compile("reduce_sum", reduce_sum_f32_kernel, WORKGROUP_SIZE_1D)
        pipe.dispatch(
            {"inp": inp, "out": out},
            parameters={"n": length},
            grid=compute_grid_1d(length),
        )

    def matmul(
        self,
        a: ComputeBuffer,
        b: ComputeBuffer,
        out: ComputeBuffer,
        m: int,
        n: int,
        k: int,
    ) -> None:
        assert isinstance(a, PersistentBuffer)
        assert isinstance(b, PersistentBuffer)
        assert isinstance(out, PersistentBuffer)
        pipe = self._get_or_compile("tiled_matmul", tiled_matmul_f32_kernel, (TILE_DIM, TILE_DIM, 1))
        num_tiles = (k + TILE_DIM - 1) // TILE_DIM
        pipe.dispatch(
            {"mat_a": a, "mat_b": b, "mat_c": out},
            parameters={"m": m, "n": n, "k": k, "num_tiles": num_tiles},
            grid=compute_matmul_grid(m, n),
        )

    def synchronize(self) -> None:
        self._ctx.synchronize()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[BaseException] = []
        for pipe in list(self._pipelines.values()):
            try:
                pipe.close()
            except BaseException as exc:
                errors.append(exc)
        self._pipelines.clear()
        try:
            self._ctx.close()
        except BaseException as exc:
            errors.append(exc)
        try:
            self._artifact_dir.cleanup()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
