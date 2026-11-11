"""Compilation, numerical parity, and timing tests for tiled shared-memory matrix multiplication."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

from mlir_gpu import Target, compile_pipeline
from test_support import ROOT, TOOLS, available


@unittest.skipUnless(available("gfx-compile"), "Requires native gfx-compile toolchain")
class TestTiledMatMulCompilation(unittest.TestCase):
    """Hermetic compilation and shared-memory footprint checks for examples/matmul.py."""

    @classmethod
    def setUpClass(cls):
        cls.example = runpy.run_path(str(ROOT / "examples" / "matmul.py"))

    def test_matmul_compiles_without_special_capabilities_and_reflects_shared_tiles(self):
        pipe = self.example["pipeline"]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            vk_manifest_path = compile_pipeline(pipe, Target.VULKAN, td_path / "vk_mm", TOOLS)
            manifest = json.loads(vk_manifest_path.read_text(encoding="utf-8"))
            # Two 16x16 f32 tiles = 2 * 1024 = 2048 bytes, no tensor-core capabilities required.
            self.assertEqual(manifest["shared_memory_bytes"], 2048)
            self.assertEqual(manifest["workgroup_size"], [16, 16, 1])

            if available("llc") and available("dxv"):
                dx_manifest_path = compile_pipeline(pipe, Target.DIRECTX, td_path / "dx_mm", TOOLS)
                dx_manifest = json.loads(dx_manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(dx_manifest["shared_memory_bytes"], 2048)


class _TiledMatMulExecutionMixin:
    target: Target
    provider: str | None = None
    validation: bool = False
    example: dict

    def _verify_shape(self, m: int, n: int, k: int, tol: float = 1e-4) -> None:
        a, b = self.example["create_deterministic_matrices"](m, n, k)
        expected = self.example["cpu_matmul"](a, b, m, n, k)
        res = self.example["run_tiled_matmul"](
            a,
            b,
            m,
            n,
            k,
            self.target,
            validation=self.validation,
            provider=self.provider,
            toolchain=TOOLS,
        )
        self.assertEqual(len(res.c), m * n)
        self.assertGreater(res.elapsed_ms, 0.0)
        self.assertGreater(res.gflops, 0.0)
        self.assertTrue(all(math.isfinite(v) for v in res.c))
        max_err = max(abs(x - y) for x, y in zip(res.c, expected))
        self.assertLess(max_err, tol, f"Max matmul error {max_err} for ({m}, {n}, {k}) exceeded {tol}")

    def test_square_tiled_matmul_64x64(self):
        self._verify_shape(64, 64, 64)

    def test_rectangular_tiled_matmul_48x32_k80(self):
        self._verify_shape(48, 32, 80)

    def test_non_tile_aligned_boundary_matmul_37x29_k53(self):
        self._verify_shape(37, 29, 53)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan tiled matrix multiplication tests",
)
class TestTiledMatMulVulkanExecution(_TiledMatMulExecutionMixin, unittest.TestCase):
    target = Target.VULKAN

    @classmethod
    def setUpClass(cls):
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "matmul.py"))


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX 12 WARP tiled matrix multiplication tests",
)
class TestTiledMatMulDirectXExecution(_TiledMatMulExecutionMixin, unittest.TestCase):
    target = Target.DIRECTX
    provider = "warp"

    @classmethod
    def setUpClass(cls):
        if not (available("llc") and available("dxv") and available("mlir-gpu-directx-view")):
            raise unittest.SkipTest("DirectX 12 compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "matmul.py"))


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal tiled matrix multiplication tests",
)
class TestTiledMatMulMetalExecution(_TiledMatMulExecutionMixin, unittest.TestCase):
    target = Target.METAL

    @classmethod
    def setUpClass(cls):
        required = ("llc", "dxv", "metal-shaderconverter", "mlir-gpu-metal-view")
        if not all(available(tool) for tool in required):
            raise unittest.SkipTest("Metal compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "matmul.py"))


if __name__ == "__main__":
    unittest.main()
