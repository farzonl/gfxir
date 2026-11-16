"""Tests for device-resident float32 elementwise arrays in gfxpy."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402
from gfxpy.gpu_backend import GfxIRComputeBackend  # noqa: E402
from mlir_gpu.compute_runtime import ComputeContext, PersistentBuffer, PersistentPipeline  # noqa: E402


class _FakePersistentBuffer(PersistentBuffer):
    def __init__(self, ctx: _FakeComputeContext, size_bytes: int) -> None:
        self._ctx = ctx
        self._data = bytearray(size_bytes)
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return len(self._data)

    @property
    def closed(self) -> bool:
        return self._closed or self._ctx.closed

    @property
    def context(self) -> ComputeContext:
        return self._ctx

    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        self._ctx.upload_calls += 1
        payload = bytes(data)
        self._data[offset : offset + len(payload)] = payload

    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        self._ctx.download_calls += 1
        n = len(self._data) - offset if size_bytes is None else size_bytes
        return bytes(self._data[offset : offset + n])

    def close(self) -> None:
        self._closed = True


class _FakePersistentPipeline(PersistentPipeline):
    def __init__(self, ctx: _FakeComputeContext, program: object, op_name: str) -> None:
        self._ctx = ctx
        self._program = program
        self._op_name = op_name
        self._closed = False

    @property
    def program(self):
        return self._program  # type: ignore[return-value]

    @property
    def context(self) -> ComputeContext:
        return self._ctx

    @property
    def closed(self) -> bool:
        return self._closed or self._ctx.closed

    def dispatch(self, buffers, parameters=None, grid=None) -> None:
        self._ctx.dispatch_calls.append((self._op_name, grid))
        n = int(parameters["n"])
        lhs = struct.unpack(f"<{n}f", buffers["lhs"]._data[: n * 4])  # type: ignore[attr-defined]
        if self._op_name == "binary_add":
            rhs = struct.unpack(f"<{n}f", buffers["rhs"]._data[: n * 4])  # type: ignore[attr-defined]
            vals = [a + b for a, b in zip(lhs, rhs)]
        elif self._op_name == "binary_mul":
            rhs = struct.unpack(f"<{n}f", buffers["rhs"]._data[: n * 4])  # type: ignore[attr-defined]
            vals = [a * b for a, b in zip(lhs, rhs)]
        elif self._op_name == "scalar_add":
            s = float(parameters["scalar"])
            vals = [a + s for a in lhs]
        elif self._op_name == "scalar_mul":
            s = float(parameters["scalar"])
            vals = [a * s for a in lhs]
        else:
            raise AssertionError(f"Unexpected op: {self._op_name}")
        buffers["out"]._data[: n * 4] = struct.pack(f"<{n}f", *vals)  # type: ignore[attr-defined]

    def close(self) -> None:
        self._closed = True


class _FakeComputeContext(ComputeContext):
    def __init__(self, target: str = "vulkan") -> None:
        self._target = target
        self._closed = False
        self.upload_calls = 0
        self.download_calls = 0
        self.dispatch_calls: list[tuple[str, tuple[int, int, int] | None]] = []
        self.compile_calls = 0
        self.created_buffers: list[_FakePersistentBuffer] = []
        self.created_pipelines: list[_FakeComputePipeline] = []

    @property
    def target(self) -> str:
        return self._target

    @property
    def device_name(self) -> str:
        return "FakeDevice"

    @property
    def closed(self) -> bool:
        return self._closed

    def create_buffer(self, size_bytes: int, initial_data=None) -> PersistentBuffer:
        buf = _FakePersistentBuffer(self, size_bytes)
        self.created_buffers.append(buf)
        if initial_data:
            buf.upload(initial_data)
        return buf

    def compile_pipeline(self, program) -> PersistentPipeline:
        self.compile_calls += 1
        op_dir = Path(str(program)).parent.name
        pipe = _FakePersistentPipeline(self, program, op_dir)
        self.created_pipelines.append(pipe)
        return pipe

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True


class TestGfxPyDeviceResidentHost(unittest.TestCase):
    def test_chained_ops_no_intermediate_host_transfers_and_pipeline_reuse(self) -> None:
        fake_ctx = _FakeComputeContext("vulkan")
        compile_count = 0

        def fake_compile(pipeline, target, out_dir, toolchain):
            nonlocal compile_count
            compile_count += 1
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = out_dir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            return manifest

        with (
            mock.patch.dict(sys.modules, {"numpy": None}),
            mock.patch("gfxpy.gpu_backend._open_context", return_value=fake_ctx),
            mock.patch("gfxpy.gpu_backend.compile_pipeline", side_effect=fake_compile),
        ):
            backend = GfxIRComputeBackend("vulkan")
            with gp.device("vulkan", backend=backend) as dev:
                x = gp.array([1.0, 2.0, 3.0], device=dev)
                y = gp.array([4.0, 5.0, 6.0], device=dev)
                # Chained expression: (x * 2.0 + 1.0) + y
                z = (x * 2.0 + 1.0) + y

                # Only initial uploads (2), zero intermediate downloads!
                self.assertEqual(fake_ctx.upload_calls, 2)
                self.assertEqual(fake_ctx.download_calls, 0)
                self.assertEqual(
                    [name for name, _ in fake_ctx.dispatch_calls],
                    ["scalar_mul", "scalar_add", "binary_add"],
                )

                # Explicit readback triggers exactly 1 download.
                self.assertEqual(z.tolist(), [7.0, 10.0, 13.0])
                self.assertEqual(fake_ctx.download_calls, 1)

                # Repeated execution reuses cached compiled pipelines.
                w = (x * 3.0 + 2.0) + y
                self.assertEqual(w.tolist(), [9.0, 13.0, 17.0])
                self.assertEqual(compile_count, 3)
                self.assertEqual(backend.compiled_pipeline_count, 3)

            dev.close()
            self.assertTrue(dev.closed)
            self.assertTrue(fake_ctx.closed)

    def test_pipeline_cache_key_accounts_for_target_and_op(self) -> None:
        fake_ctx = _FakeComputeContext("vulkan")
        with (
            mock.patch("gfxpy.gpu_backend._open_context", return_value=fake_ctx),
        ):
            backend = GfxIRComputeBackend("vulkan")
            k1 = backend._pipeline_cache_key("scalar_mul")
            k2 = backend._pipeline_cache_key("scalar_add")
            self.assertNotEqual(k1, k2)
            self.assertIn("vulkan", k1)
            self.assertIn("float32", k1)
            backend.close()

    def test_repeated_arrays_temporaries_reductions_and_close_baseline(self) -> None:
        import gc

        fake_ctx = _FakeComputeContext("vulkan")

        def fake_compile(pipeline, target, out_dir, toolchain):
            out_dir.mkdir(parents=True, exist_ok=True)
            manifest = out_dir / "manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            return manifest

        with (
            mock.patch("gfxpy.gpu_backend._open_context", return_value=fake_ctx),
            mock.patch("gfxpy.gpu_backend.compile_pipeline", side_effect=fake_compile),
        ):
            backend = GfxIRComputeBackend("vulkan")
            dev = gp.device("vulkan", backend=backend)
            try:
                with dev:
                    for _ in range(25):
                        a = gp.array([1.0, 2.0, 3.0, 4.0])
                        b = gp.array([2.0, 3.0, 4.0, 5.0])
                        res = (a * 2.0 + 1.0) + b
                        self.assertEqual(res.tolist(), [5.0, 8.0, 11.0, 14.0])
                        res.close()
                        a.close()
                        b.close()
                    gc.collect()
                    open_bufs = [buf for buf in fake_ctx.created_buffers if not buf.closed]
                    self.assertEqual(len(open_bufs), 0)
            finally:
                dev.close()
            self.assertTrue(all(buf.closed for buf in fake_ctx.created_buffers))
            self.assertTrue(all(pipe.closed for pipe in fake_ctx.created_pipelines))


class _LiveBackendMixin:
    target_name: str
    provider: str | None = None

    def test_bounds_checked_lengths_and_chained_expression(self) -> None:
        lengths = [0, 1, 63, 64, 65, 137, 1000]
        with gp.device(self.target_name, provider=self.provider) as dev:
            self.addCleanup(dev.close)
            for n in lengths:
                host_x = [float(i) * 0.5 - 3.0 for i in range(n)]
                host_y = [float(i % 7) + 1.25 for i in range(n)]
                x = gp.array(host_x, device=dev)
                y = gp.array(host_y, device=dev)
                z = (x * 2.0 + 1.0) + y
                w = x * y
                got_z = z.tolist()
                got_w = w.tolist()
                exp_z = [(a * 2.0 + 1.0) + b for a, b in zip(host_x, host_y)]
                exp_w = [a * b for a, b in zip(host_x, host_y)]
                self.assertEqual(len(got_z), n)
                self.assertEqual(len(got_w), n)
                for g, e in zip(got_z, exp_z):
                    self.assertAlmostEqual(g, e, delta=1e-3)
                for g, e in zip(got_w, exp_w):
                    self.assertAlmostEqual(g, e, delta=1e-3)

    def test_repeated_arrays_temporaries_reductions_and_close_baseline(self) -> None:
        import gc

        with gp.device(self.target_name, provider=self.provider) as dev:
            try:
                for _ in range(20):
                    a = gp.array([1.0, 2.0, 3.0, 4.0], device=dev)
                    b = gp.array([0.5, 1.5, 2.5, 3.5], device=dev)
                    s = ((a * 2.0 + 1.0) + b).sum()
                    got = s.tolist()
                    self.assertAlmostEqual(got[0], 32.0, delta=1e-3)
                    s.close()
                    a.close()
                    b.close()
                gc.collect()
                stats = dev.memory_stats
                self.assertEqual(stats.get("live_buffers", 0), 0)
                dev.release_pool()
                self.assertEqual(dev.memory_stats.get("pooled_buffers", 0), 0)
            finally:
                dev.close()


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan gfxpy tests",
)
class TestGfxPyVulkanLive(_LiveBackendMixin, unittest.TestCase):
    target_name = "vulkan"


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal gfxpy tests",
)
class TestGfxPyMetalLive(_LiveBackendMixin, unittest.TestCase):
    target_name = "metal"


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX gfxpy tests",
)
class TestGfxPyDirectXLive(_LiveBackendMixin, unittest.TestCase):
    target_name = "directx"


@unittest.skipUnless(
    sys.platform in ("linux", "darwin") and os.environ.get("MLIR_GPU_TEST_NACELLE") == "1",
    "Set MLIR_GPU_TEST_NACELLE=1 on Linux/macOS to run live Nacelle gfxpy tests",
)
class TestGfxPyNacelleLive(_LiveBackendMixin, unittest.TestCase):
    target_name = "directx"
    provider = "nacelle"


if __name__ == "__main__":
    unittest.main()
