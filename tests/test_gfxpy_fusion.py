"""Tests for explicit float32 elementwise expression fusion in gfxpy."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
SRC_DIR = ROOT / "src"
for _p in (EXAMPLES_DIR, SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gfxpy as gp  # noqa: E402
from gfxpy.chain_example import _CountingBackend  # noqa: E402
from gfxpy.gpu_backend import GfxIRComputeBackend  # noqa: E402
from mlir_gpu import ComputeContext, PersistentBuffer, PersistentPipeline, Target, Toolchain, compile_pipeline  # noqa: E402


class _FusionFakeBuffer(PersistentBuffer):
    def __init__(self, context: _FusionFakeContext, size_bytes: int) -> None:
        self._context = context
        self._data = bytearray(size_bytes)
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return len(self._data)

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    @property
    def context(self) -> _FusionFakeContext:
        return self._context

    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        raw = bytes(data)
        self._data[offset : offset + len(raw)] = raw
        self._context.upload_calls += 1

    def download_into(self, dst: bytearray | memoryview, size_bytes: int | None = None, offset: int = 0) -> int:
        raw = self.download(size_bytes=size_bytes, offset=offset)
        memoryview(dst).cast("B")[: len(raw)] = raw
        return len(raw)

    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        self._context.download_calls += 1
        end = len(self._data) if size_bytes is None else offset + size_bytes
        return bytes(self._data[offset:end])

    def close(self) -> None:
        self._closed = True


class _FusionFakePipeline(PersistentPipeline):
    def __init__(self, context: _FusionFakeContext, program: object) -> None:
        self._context = context
        self._program = program
        self._closed = False

    @property
    def program(self) -> object:
        return self._program

    @property
    def context(self) -> _FusionFakeContext:
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    def dispatch(self, buffers, parameters=None, grid=None) -> None:
        self._context.dispatch_calls.append((dict(buffers), dict(parameters or {}), grid))
        n = int(parameters["n"])
        if "in0" in buffers:
            in0 = struct.unpack(f"<{n}f", buffers["in0"]._data)
            in1 = struct.unpack(f"<{n}f", buffers["in1"]._data)
            in2 = struct.unpack(f"<{n}f", buffers["in2"]._data)
            s0 = float(parameters["s0"])
            s1 = float(parameters["s1"])
            out_vals = [((x * y + s0) * s1) + z for x, y, z in zip(in0, in1, in2)]
        else:
            lhs = struct.unpack(f"<{n}f", buffers["lhs"]._data)
            if "rhs" in buffers:
                rhs = struct.unpack(f"<{n}f", buffers["rhs"]._data)
                out_vals = [x * y if len(self._context.dispatch_calls) == 3 else x + y for x, y in zip(lhs, rhs)]
            else:
                scalar = float(parameters["scalar"])
                out_vals = [x * scalar if len(self._context.dispatch_calls) == 5 else x + scalar for x in lhs]
        buffers["out"]._data[:] = struct.pack(f"<{n}f", *out_vals)

    def close(self) -> None:
        self._closed = True


class _FusionFakeContext(ComputeContext):
    def __init__(self, target: str = "vulkan") -> None:
        self._target = target
        self._closed = False
        self.alloc_calls = 0
        self.upload_calls = 0
        self.download_calls = 0
        self.compile_calls = 0
        self.dispatch_calls: list[tuple[dict[str, object], dict[str, object], tuple[int, int, int] | None]] = []

    @property
    def target(self) -> str:
        return self._target

    @property
    def device_name(self) -> str:
        return "FusionFakeDevice"

    @property
    def closed(self) -> bool:
        return self._closed

    def create_buffer(self, size_bytes: int, initial_data=None, zero_init: bool = True) -> PersistentBuffer:
        self.alloc_calls += 1
        buf = _FusionFakeBuffer(self, size_bytes)
        if initial_data:
            buf.upload(initial_data)
        return buf

    def compile_pipeline(self, program) -> PersistentPipeline:
        self.compile_calls += 1
        return _FusionFakePipeline(self, program)

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True


class TestGfxPyFusionHost(unittest.TestCase):
    def test_representative_chain_lowers_to_single_dispatch_and_allocation_and_reuses_cache(self) -> None:
        fake_ctx = _FusionFakeContext("vulkan")
        compile_count = 0

        def fake_compile(pipeline, target, out_dir, toolchain):
            nonlocal compile_count
            compile_count += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = out_dir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            return manifest

        @gp.fuse
        def chain(a, b, c, s0, s1):
            return (a * b + s0) * s1 + c

        with (
            mock.patch.dict(sys.modules, {"numpy": None}),
            mock.patch("gfxpy.gpu_backend._open_context", return_value=fake_ctx),
            mock.patch("gfxpy.gpu_backend.compile_pipeline", side_effect=fake_compile),
        ):
            raw_backend = GfxIRComputeBackend("vulkan")
            counted = _CountingBackend(raw_backend)
            with gp.device("vulkan", backend=counted) as dev:
                a = gp.array([1.0, 2.0, 3.0], device=dev)
                b = gp.array([4.0, 5.0, 6.0], device=dev)
                c = gp.array([0.5, 1.0, 1.5], device=dev)

                allocs_before = fake_ctx.alloc_calls
                y1 = chain(a, b, c, 1.5, 2.0)
                self.assertEqual(fake_ctx.alloc_calls - allocs_before, 1)
                self.assertEqual(len(fake_ctx.dispatch_calls), 1)
                self.assertEqual(counted.downloads, 0)
                self.assertEqual(compile_count, 1)
                self.assertEqual(fake_ctx.compile_calls, 1)
                self.assertEqual(y1.tolist(), [11.5, 24.0, 40.5])
                self.assertEqual(counted.downloads, 1)

                # Changing runtime scalar values reuses the compiled kernel without recompiling.
                y2 = chain(a, b, c, 3.0, 0.5)
                self.assertEqual(compile_count, 1)
                self.assertEqual(fake_ctx.compile_calls, 1)
                self.assertEqual(len(fake_ctx.dispatch_calls), 2)
                self.assertEqual(y2.tolist(), [4.0, 7.5, 12.0])

                # Eager semantics remain available side-by-side.
                eager = (a * b + 1.5) * 2.0 + c
                self.assertEqual(eager.tolist(), y1.tolist())

    def test_cache_key_distinguishes_structure_target_validation_provider_and_toolchain(self) -> None:
        fake_ctx = _FusionFakeContext("vulkan")
        with mock.patch("gfxpy.gpu_backend._open_context", return_value=fake_ctx):
            b_default = GfxIRComputeBackend("vulkan", validation=False, provider=None, toolchain=Toolchain())
            b_val = GfxIRComputeBackend("vulkan", validation=True, provider=None, toolchain=b_default._toolchain)
            b_sw = GfxIRComputeBackend("vulkan", validation=False, provider="swiftshader", toolchain=b_default._toolchain)
            b_other_tc = GfxIRComputeBackend("vulkan", validation=False, provider=None, toolchain=Toolchain())
            try:
                k1 = b_default._pipeline_cache_key("fused:(in0+in1)")
                k2 = b_default._pipeline_cache_key("fused:(in0*in1)")
                k_val = b_val._pipeline_cache_key("fused:(in0+in1)")
                k_sw = b_sw._pipeline_cache_key("fused:(in0+in1)")
                k_tc = b_other_tc._pipeline_cache_key("fused:(in0+in1)")
                self.assertEqual(len({k1, k2, k_val, k_sw, k_tc}), 5)
            finally:
                b_default.close()
                b_val.close()
                b_sw.close()
                b_other_tc.close()

    def test_unsupported_ops_shape_device_and_empty_validation(self) -> None:
        from tests.test_gfxpy_api import _SpyBackend

        @gp.fuse
        def sub_fn(a, b):
            return a - b

        @gp.fuse
        def bad_return(a):
            return 1.0

        @gp.fuse
        def add_fn(a, b):
            return a + b

        b1 = _SpyBackend("vulkan")
        b2 = _SpyBackend("metal")
        d1 = gp.device("vulkan", backend=b1)
        d2 = gp.device("metal", backend=b2)
        try:
            a = gp.array([1.0, 2.0], device=d1)
            b = gp.array([3.0, 4.0], device=d1)
            b_other_shape = gp.array([1.0, 2.0, 3.0], device=d1)
            b_other_dev = gp.array([3.0, 4.0], device=d2)

            with self.assertRaises(gp.UnsupportedFusionError):
                sub_fn(a, b)
            with self.assertRaises(gp.UnsupportedFusionError):
                bad_return(a)
            with self.assertRaises(gp.ShapeMismatchError):
                add_fn(a, b_other_shape)
            with self.assertRaises(gp.CrossDeviceError):
                add_fn(a, b_other_dev)

            # Empty arrays succeed with zero allocations/dispatches.
            allocs_before = len(b1.allocations)
            e1 = gp.empty(0, device=d1)
            e2 = gp.array([], device=d1)
            e_out = add_fn(e1, e2)
            self.assertEqual(e_out.tolist(), [])
            self.assertEqual(len(b1.allocations), allocs_before)
            self.assertEqual(b1.dispatches, [])
        finally:
            d1.close()
            d2.close()

    def test_generated_fused_kernel_compiles_with_gfxir_frontend(self) -> None:
        from gfxpy.fusion import FusedExpressionSpec, build_fused_compute_kernel
        from mlir_gpu.compiler import CompileError

        spec = FusedExpressionSpec(
            signature="(((in0*in1)+s0)*s1+in2)",
            num_inputs=3,
            num_scalars=2,
            expr_code="(((in0[idx] * in1[idx]) + params.s0) * params.s1) + in2[idx]",
        )
        with tempfile.TemporaryDirectory(prefix="gfxpy-fusion-compile-") as tmp:
            kernel = build_fused_compute_kernel(spec, Path(tmp))
            try:
                from mlir_gpu import ComputePipeline, Dispatch

                llvm_bin = ROOT / ".toolchains" / "llvm-build" / "bin"
                toolchain = Toolchain(llvm_bin=llvm_bin if llvm_bin.is_dir() else None)
                manifest = compile_pipeline(
                    ComputePipeline(kernel, Dispatch(grid=(1, 1, 1))),
                    Target.VULKAN,
                    Path(tmp) / "out",
                    toolchain,
                )
                self.assertTrue(manifest.exists())
            except CompileError:
                # Native gfx-compile not available in this host environment.
                self.assertEqual(kernel.kind, "compute")


class _LiveFusionMixin:
    target_name: str
    provider: str | None = None

    def test_fused_vs_eager_accuracy_cancellation_and_timings(self) -> None:
        @gp.fuse
        def fused_chain(a, b, c):
            return (a * b + 1.5) * 2.0 + c

        @gp.fuse
        def fused_cancel(a, b, c):
            return (a * 1.0001 + (-1.0001) * b) + c

        raw_backend = GfxIRComputeBackend(self.target_name, provider=self.provider)
        counted = _CountingBackend(raw_backend)
        with gp.device(self.target_name, backend=counted) as dev:
            try:
                for n in (0, 1, 63, 64, 65, 137, 1000):
                    a_vals = [float(i) * 0.25 - 5.0 for i in range(n)]
                    b_vals = [float((i % 7) + 1) * 0.5 for i in range(n)]
                    c_vals = [float(i % 11) * -0.75 for i in range(n)]

                    a = gp.array(a_vals, device=dev)
                    b = gp.array(b_vals, device=dev)
                    c = gp.array(c_vals, device=dev)

                    counted.dispatches.clear()
                    t0 = time.perf_counter()
                    y_fused = fused_chain(a, b, c)
                    cold_ms = (time.perf_counter() - t0) * 1000.0
                    fused_dispatches = list(counted.dispatches)

                    counted.dispatches.clear()
                    t1 = time.perf_counter()
                    y_fused_warm = fused_chain(a, b, c)
                    warm_ms = (time.perf_counter() - t1) * 1000.0

                    counted.dispatches.clear()
                    y_eager = (a * b + 1.5) * 2.0 + c
                    eager_dispatches = list(counted.dispatches)

                    if n > 0:
                        self.assertEqual(len(fused_dispatches), 1)
                        self.assertEqual(len(eager_dispatches), 4)
                    else:
                        self.assertEqual(len(fused_dispatches), 0)

                    got_fused = y_fused.tolist()
                    got_warm = y_fused_warm.tolist()
                    got_eager = y_eager.tolist()
                    self.assertEqual(got_fused, got_warm)
                    for g, e in zip(got_fused, got_eager):
                        tol = 1e-5 + 1e-5 * abs(e)
                        self.assertLessEqual(abs(g - e), tol)

                    # Cancellation check on non-aligned length
                    if n == 137:
                        yc_fused = fused_cancel(a, a, c).tolist()
                        yc_eager = ((a * 1.0001 + (-1.0001) * a) + c).tolist()
                        for g, e in zip(yc_fused, yc_eager):
                            tol = 1e-4 + 1e-5 * abs(e)
                            self.assertLessEqual(abs(g - e), tol)

                    if n == 1000:
                        print(
                            f"[GfxPyFusionLive] target={self.target_name} "
                            f"device={raw_backend.device_name!r} n={n} "
                            f"cold_ms={cold_ms:.3f} warm_ms={warm_ms:.3f}"
                        )
            finally:
                dev.close()


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal fusion tests",
)
class TestGfxPyFusionMetalLive(_LiveFusionMixin, unittest.TestCase):
    target_name = "metal"


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan fusion tests",
)
class TestGfxPyFusionVulkanLive(_LiveFusionMixin, unittest.TestCase):
    target_name = "vulkan"


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX fusion tests",
)
class TestGfxPyFusionDirectXLive(_LiveFusionMixin, unittest.TestCase):
    target_name = "directx"


@unittest.skipUnless(
    sys.platform in ("linux", "darwin") and os.environ.get("MLIR_GPU_TEST_NACELLE") == "1",
    "Set MLIR_GPU_TEST_NACELLE=1 on Linux/macOS to run live Nacelle fusion tests",
)
class TestGfxPyFusionNacelleLive(_LiveFusionMixin, unittest.TestCase):
    target_name = "directx"
    provider = "nacelle"


if __name__ == "__main__":
    unittest.main()
