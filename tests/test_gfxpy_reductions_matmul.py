"""Tests for bounded sum reduction and 2D float32 matmul in gfxpy."""

from __future__ import annotations

import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402
from gfxpy.kernels import reduce_sum_f32_kernel, tiled_matmul_f32_kernel  # noqa: E402
from mlir_gpu import ComputePipeline, Dispatch, Target, compile_pipeline  # noqa: E402
from test_support import TOOLS  # noqa: E402


class _SpyBuffer:
    def __init__(self, size_bytes: int) -> None:
        self.data = bytearray(size_bytes)
        self.closed = False

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def close(self) -> None:
        self.closed = True


class _SpyReductionMatmulBackend:
    def __init__(self, target: str = "vulkan") -> None:
        self.target = target
        self.closed = False
        self.uploads = 0
        self.downloads = 0
        self.reduce_passes: list[int] = []
        self.matmuls: list[tuple[int, int, int]] = []
        self.buffers: list[_SpyBuffer] = []

    def allocate(self, size_bytes: int) -> _SpyBuffer:
        buf = _SpyBuffer(size_bytes)
        self.buffers.append(buf)
        return buf

    def upload(self, buf: _SpyBuffer, data: bytes) -> None:
        self.uploads += 1
        buf.data[: len(data)] = data

    def download(self, buf: _SpyBuffer) -> bytes:
        self.downloads += 1
        return bytes(buf.data)

    def binary_op(self, op: str, lhs: _SpyBuffer, rhs: _SpyBuffer, out: _SpyBuffer, length: int) -> None:
        raise NotImplementedError

    def scalar_op(self, op: str, lhs: _SpyBuffer, scalar: float, out: _SpyBuffer, length: int) -> None:
        raise NotImplementedError

    def reduce_sum_pass(self, inp: _SpyBuffer, out: _SpyBuffer, length: int) -> None:
        self.reduce_passes.append(length)
        vals = struct.unpack(f"<{length}f", inp.data[: length * 4])
        num_groups = (length + 63) // 64
        partials = [sum(vals[g * 64 : (g + 1) * 64]) for g in range(num_groups)]
        out.data[: num_groups * 4] = struct.pack(f"<{num_groups}f", *partials)

    def matmul(self, a: _SpyBuffer, b: _SpyBuffer, out: _SpyBuffer, m: int, n: int, k: int) -> None:
        self.matmuls.append((m, n, k))
        av = struct.unpack(f"<{m * k}f", a.data[: m * k * 4])
        bv = struct.unpack(f"<{k * n}f", b.data[: k * n * 4])
        cv = [0.0] * (m * n)
        for r in range(m):
            for p in range(k):
                a_rp = av[r * k + p]
                for c in range(n):
                    cv[r * n + c] += a_rp * bv[p * n + c]
        out.data[: m * n * 4] = struct.pack(f"<{m * n}f", *cv)

    def close(self) -> None:
        self.closed = True


class TestReductionsAndMatmulHost(unittest.TestCase):
    def test_kernels_compile_for_vulkan(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            m_red = compile_pipeline(
                ComputePipeline(reduce_sum_f32_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "reduce_sum",
                TOOLS,
            )
            m_mm = compile_pipeline(
                ComputePipeline(tiled_matmul_f32_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "tiled_matmul",
                TOOLS,
            )
            self.assertTrue(Path(m_red).is_file())
            self.assertTrue(Path(m_mm).is_file())

    def test_cooperative_matrix_matmul_compiles_and_selects_with_fallback(self) -> None:
        from mlir_gpu import CompileError, TargetProfile
        from examples.gfxpy.gpu_backend import GfxIRComputeBackend
        from examples.gfxpy.kernels import make_coop_matmul_f32_kernel

        coop_kernel = make_coop_matmul_f32_kernel(16, 16, 16)
        profile = TargetProfile(spirv_version="1.6", required_capabilities=("cooperative_matrix",))
        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(
                ComputePipeline(coop_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "coop_matmul",
                TOOLS,
                target_profile=profile,
            )
            self.assertTrue(Path(manifest).is_file())
            self.assertTrue((Path(td) / "coop_matmul" / "compute.spv").is_file())

            with self.assertRaises(CompileError):
                compile_pipeline(
                    ComputePipeline(coop_kernel, Dispatch(grid=(1, 1, 1))),
                    Target.DIRECTX,
                    Path(td) / "coop_matmul_dx",
                    TOOLS,
                )

        class _FakeCtx:
            def __init__(self, configs):
                self._configs = configs

            def supports_capability(self, cap: str) -> bool:
                return cap == "cooperative_matrix" and bool(self._configs)

            @property
            def cooperative_matrix_configs(self):
                return self._configs

        backend_supported = object.__new__(GfxIRComputeBackend)
        backend_supported._ctx = _FakeCtx(((8, 8, 8), (16, 16, 16)))
        self.assertEqual(backend_supported._select_coop_matmul_shape(64, 64, 64), (16, 16, 16))
        self.assertEqual(backend_supported._select_coop_matmul_shape(24, 24, 24), (8, 8, 8))
        self.assertIsNone(backend_supported._select_coop_matmul_shape(37, 53, 29))

        backend_unsupported = object.__new__(GfxIRComputeBackend)
        backend_unsupported._ctx = _FakeCtx(())
        self.assertIsNone(backend_unsupported._select_coop_matmul_shape(64, 64, 64))

    def test_multi_dispatch_sum_stays_device_resident_and_cleans_intermediates(self) -> None:
        backend = _SpyReductionMatmulBackend("vulkan")
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with gp.device("vulkan", backend=backend) as dev:
                # Empty reduction -> [0.0] without dispatching kernel
                e = gp.array([], device=dev)
                es = gp.sum(e)
                self.assertEqual(es.shape, (1,))
                self.assertEqual(es.tolist(), [0.0])
                self.assertEqual(backend.reduce_passes, [])

                # 5000 elements -> 79 groups -> 2 groups -> 1 (3 dispatches, 0 intermediate downloads)
                host_vals = [float((i % 13) - 6) * 0.25 for i in range(5000)]
                x = gp.array(host_vals, device=dev)
                dl_before = backend.downloads
                total = x.sum()
                self.assertEqual(total.shape, (1,))
                self.assertEqual(backend.reduce_passes, [5000, 79, 2])
                self.assertEqual(backend.downloads, dl_before)
                got = total.tolist()[0]
                self.assertEqual(backend.downloads, dl_before + 1)
                self.assertAlmostEqual(got, sum(host_vals), delta=1e-2)

    def test_matmul_shapes_empty_and_validation(self) -> None:
        backend = _SpyReductionMatmulBackend("vulkan")
        other_backend = _SpyReductionMatmulBackend("metal")
        d1 = gp.device("vulkan", backend=backend)
        d2 = gp.device("metal", backend=other_backend)

        a = gp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device=d1)
        b = gp.array([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]], device=d1)
        dl_before = backend.downloads
        c = a @ b
        self.assertEqual(c.shape, (2, 2))
        self.assertEqual(backend.downloads, dl_before)
        self.assertEqual(c.tolist(), [[58.0, 64.0], [139.0, 154.0]])

        # Empty K: (2, 0) @ (0, 3) -> zero-filled (2, 3)
        z_a = gp.empty((2, 0), device=d1)
        z_b = gp.empty((0, 3), device=d1)
        z_c = gp.matmul(z_a, z_b)
        self.assertEqual(z_c.shape, (2, 3))
        self.assertEqual(z_c.tolist(), [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        # Mismatched inner dimension, 1D operand, and cross-device operand rejected
        with self.assertRaises(gp.ShapeMismatchError):
            _ = a @ a
        with self.assertRaises(gp.ShapeMismatchError):
            _ = gp.array([1.0, 2.0], device=d1) @ b
        with self.assertRaises(gp.CrossDeviceError):
            _ = a @ gp.array([[1.0], [2.0], [3.0]], device=d2)


class _LiveReductionMatmulMixin:
    target_name: str
    provider: str | None = None

    def test_live_bounded_sum_and_non_tile_aligned_matmul(self) -> None:
        with gp.device(self.target_name, provider=self.provider) as dev:
            self.addCleanup(dev.close)
            # Multi-dispatch reduction for non-workgroup-aligned 5000 elements
            vals = [float((i % 17) - 8) * 0.125 for i in range(5000)]
            x = gp.array(vals, device=dev)
            got_sum = gp.sum(x).tolist()[0]
            self.assertAlmostEqual(got_sum, sum(vals), delta=1e-2)

            # Non-tile-aligned rectangular matmul (37x53) @ (53x29)
            m, k, n = 37, 53, 29
            mat_a = [[float(((r * 7 + c * 3) % 11) - 5) * 0.25 for c in range(k)] for r in range(m)]
            mat_b = [[float(((r * 5 + c * 9) % 13) - 6) * 0.25 for c in range(n)] for r in range(k)]
            a_gpu = gp.array(mat_a, device=dev)
            b_gpu = gp.array(mat_b, device=dev)
            c_gpu = (a_gpu @ b_gpu).tolist()

            for r in range(m):
                for c in range(n):
                    exp = sum(mat_a[r][p] * mat_b[p][c] for p in range(k))
                    self.assertAlmostEqual(c_gpu[r][c], exp, delta=1e-3)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan reduction/matmul tests",
)
class TestGfxPyReductionsMatmulVulkanLive(_LiveReductionMatmulMixin, unittest.TestCase):
    target_name = "vulkan"


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal reduction/matmul tests",
)
class TestGfxPyReductionsMatmulMetalLive(_LiveReductionMatmulMixin, unittest.TestCase):
    target_name = "metal"


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX reduction/matmul tests",
)
class TestGfxPyReductionsMatmulDirectXLive(_LiveReductionMatmulMixin, unittest.TestCase):
    target_name = "directx"


@unittest.skipUnless(
    sys.platform in ("linux", "darwin") and os.environ.get("MLIR_GPU_TEST_NACELLE") == "1",
    "Set MLIR_GPU_TEST_NACELLE=1 on Linux/macOS to run live Nacelle reduction/matmul tests",
)
class TestGfxPyReductionsMatmulNacelleLive(_LiveReductionMatmulMixin, unittest.TestCase):
    target_name = "directx"
    provider = "nacelle"


if __name__ == "__main__":
    unittest.main()
