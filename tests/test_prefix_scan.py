"""Compilation and live GPU execution tests for bounded shared-memory parallel prefix scan."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

from mlir_gpu import Target, compile_pipeline
from test_support import ROOT, TOOLS, available


@unittest.skipUnless(available("gfx-compile"), "Requires native gfx-compile toolchain")
class TestPrefixScanCompilation(unittest.TestCase):
    """Hermetic compilation, reflection, and CPU oracle checks for prefix_scan.py."""

    @classmethod
    def setUpClass(cls):
        cls.example = runpy.run_path(str(ROOT / "examples" / "prefix_scan.py"))

    def test_cpu_prefix_scan_inclusive_and_exclusive(self):
        data = [3, -1, 4, 0, -2, 7]
        self.assertEqual(self.example["cpu_prefix_scan"](data, exclusive=False), [3, 2, 6, 6, 4, 11])
        self.assertEqual(self.example["cpu_prefix_scan"](data, exclusive=True), [0, 3, 2, 6, 6, 4])

    def test_prefix_scan_compiles_with_expected_shared_memory_footprint(self):
        pipe = self.example["scan_pipeline"]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            vk_manifest_path = compile_pipeline(pipe, Target.VULKAN, td_path / "vk_scan", TOOLS)
            manifest = json.loads(vk_manifest_path.read_text(encoding="utf-8"))
            # Two 256-element i32 shared arrays = 2048 bytes.
            self.assertEqual(manifest["shared_memory_bytes"], 2048)

            if available("llc") and available("dxv"):
                dx_manifest_path = compile_pipeline(pipe, Target.DIRECTX, td_path / "dx_scan", TOOLS)
                dx_manifest = json.loads(dx_manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(dx_manifest["shared_memory_bytes"], 2048)


class _PrefixScanExecutionMixin:
    target: Target
    provider: str | None = None
    validation: bool = False
    example: dict

    def _verify_scan(self, values: list[int], *, exclusive: bool) -> None:
        expected = self.example["cpu_prefix_scan"](values, exclusive=exclusive)
        res = self.example["run_prefix_scan"](
            values,
            self.target,
            exclusive=exclusive,
            validation=self.validation,
            provider=self.provider,
            toolchain=TOOLS,
        )
        self.assertEqual(res.values, expected)
        self.assertGreaterEqual(res.elapsed_ms, 0.0)

    def test_single_workgroup_inclusive_and_exclusive_scans(self):
        for length in (1, 17, 127, 256):
            values = [((i * 31 + 5) % 43) - 20 for i in range(length)]
            with self.subTest(length=length, mode="inclusive"):
                self._verify_scan(values, exclusive=False)
            with self.subTest(length=length, mode="exclusive"):
                self._verify_scan(values, exclusive=True)

    def test_multi_workgroup_non_aligned_prefix_scans(self):
        for length in (257, 600, 2048):
            values = [((i * 53 + 11) % 61) - 30 for i in range(length)]
            with self.subTest(length=length, mode="inclusive"):
                self._verify_scan(values, exclusive=False)
            with self.subTest(length=length, mode="exclusive"):
                self._verify_scan(values, exclusive=True)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan prefix scan tests",
)
class TestPrefixScanVulkanExecution(_PrefixScanExecutionMixin, unittest.TestCase):
    target = Target.VULKAN

    @classmethod
    def setUpClass(cls):
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "prefix_scan.py"))


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX 12 WARP prefix scan tests",
)
class TestPrefixScanDirectXExecution(_PrefixScanExecutionMixin, unittest.TestCase):
    target = Target.DIRECTX
    provider = "warp"

    @classmethod
    def setUpClass(cls):
        if not (available("llc") and available("dxv") and available("mlir-gpu-directx-view")):
            raise unittest.SkipTest("DirectX 12 compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "prefix_scan.py"))


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal prefix scan tests",
)
class TestPrefixScanMetalExecution(_PrefixScanExecutionMixin, unittest.TestCase):
    target = Target.METAL

    @classmethod
    def setUpClass(cls):
        required = ("llc", "dxv", "metal-shaderconverter", "mlir-gpu-metal-view")
        if not all(available(tool) for tool in required):
            raise unittest.SkipTest("Metal compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "prefix_scan.py"))


if __name__ == "__main__":
    unittest.main()
