"""Headless execution, numerical oracle, and golden PNG imgdiff tests for Mandelbrot compute shaders."""

from __future__ import annotations

import json
import os
from pathlib import Path
import runpy
import struct
import tempfile
import unittest

from mlir_gpu import (
    Target,
    ViewerError,
    compile_pipeline,
    dispatch_compute,
    imgdiff,
    load_compute_program,
    read_png,
    render_compute_rgba,
    write_png,
)
from test_support import ROOT, TOOLS, available

GOLDEN_MANDELBROT_PNG = ROOT / "tests" / "lit" / "execution" / "Inputs" / "goldens" / "common" / "mandelbrot.png"


@unittest.skipUnless(available("gfx-compile"), "Requires native gfx-compile toolchain")
class TestMandelbrotManifestAndValidation(unittest.TestCase):
    """Hermetic manifest loading and host dispatch input validation tests."""

    @classmethod
    def setUpClass(cls):
        cls.example = runpy.run_path(str(ROOT / "examples" / "mandelbrot_compute.py"))
        cls.pipeline = cls.example["pipeline"]
        cls.rgba_pipeline = cls.example["rgba_pipeline"]
        cls.default_params = cls.example["DEFAULT_MANDELBROT_PARAMS"]

    def test_load_compute_program_validates_manifest_and_inputs(self):
        if not available("spirv-val"):
            self.skipTest("Requires spirv-val")
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "vk_mandelbrot"
            manifest_path = compile_pipeline(self.pipeline, Target.VULKAN, out_dir, TOOLS)
            program = load_compute_program(manifest_path)
            self.assertEqual(program.target, "vulkan")
            self.assertEqual(program.workgroup_size, (16, 16, 1))
            self.assertEqual(program.dispatch.grid, (16, 16, 1))
            self.assertEqual(len(program.buffers), 1)
            self.assertEqual(program.buffers[0]["name"], "output_buffer")
            self.assertIsNotNone(program.parameters)
            self.assertEqual(program.parameters["name"], "mandelbrot_params")

            # Corrupt schema_version and verify rejection
            bad_manifest = Path(td) / "bad_pipeline.json"
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            data["schema_version"] = 3
            bad_manifest.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaisesRegex(ViewerError, "schema-version 5"):
                load_compute_program(bad_manifest)

    def test_golden_mandelbrot_png_matches_cpu_reference(self):
        self.assertTrue(GOLDEN_MANDELBROT_PNG.is_file(), f"Missing golden PNG: {GOLDEN_MANDELBROT_PNG}")
        golden_img = read_png(GOLDEN_MANDELBROT_PNG)
        self.assertEqual((golden_img.width, golden_img.height), (256, 256))
        cpu_img = self.example["cpu_mandelbrot_rgba"]()
        diff = imgdiff(cpu_img, golden_img)
        self.assertTrue(diff.passed, diff.summary())
        self.assertEqual(diff.mismatched_pixels, 0)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan Mandelbrot compute execution",
)
class TestMandelbrotVulkanExecution(unittest.TestCase):
    """Live Vulkan compute dispatch, buffer readback, write_png, and imgdiff verification."""

    @classmethod
    def setUpClass(cls):
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "mandelbrot_compute.py"))
        cls.pipeline = cls.example["pipeline"]
        cls.rgba_pipeline = cls.example["rgba_pipeline"]
        cls.default_params = cls.example["DEFAULT_MANDELBROT_PARAMS"]
        cls.cpu_pixel = staticmethod(cls.example["cpu_mandelbrot_pixel"])

    def test_vulkan_mandelbrot_escape_counts_match_analytical_oracle(self):
        width, height = 256, 256
        params = {**self.default_params, "width": width, "height": height, "pitch": width}

        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(self.pipeline, Target.VULKAN, Path(td) / "vk_counts", TOOLS)
            result = dispatch_compute(
                manifest,
                buffers={"output_buffer": width * height * 4},
                parameters=params,
                grid=(16, 16, 1),
                validation=self.validation,
                toolchain=TOOLS,
            )
            self.assertTrue(result.device)
            raw = result.buffers["output_buffer"]
            self.assertEqual(len(raw), width * height * 4)
            counts = struct.unpack(f"<{width * height}I", raw)

            # Analytical probes:
            # 1. Center of main cardioid: (128, 128) -> c ~ (-0.494, +0.006) -> interior (max_iterations=100)
            self.assertEqual(counts[128 * width + 128], 100)
            # 2. Period-2 bulb center: (85, 128) -> c ~ (-0.998, +0.006) -> interior (max_iterations=100)
            self.assertEqual(counts[128 * width + 85], 100)
            # 3. Far exterior corner (0, 0) -> c ~ (-1.994, -1.494), |c|^2 > 4 -> escapes on step 1
            self.assertEqual(counts[0], 1)
            # 4. Far exterior corner (255, 0) -> c ~ (+0.994, -1.494) -> escapes on step 2
            self.assertEqual(counts[255], 2)

            # Full 256x256 comparison against single-precision CPU reference
            exact_matches = 0
            for py in range(height):
                row_offset = py * width
                for px in range(width):
                    expected = self.cpu_pixel(px, py, width=width, height=height)
                    if counts[row_offset + px] == expected:
                        exact_matches += 1
            match_ratio = exact_matches / (width * height)
            self.assertGreater(
                match_ratio,
                0.998,
                f"Expected >99.8% exact pixel escape-count agreement, got {match_ratio:.5f}",
            )

    def test_vulkan_mandelbrot_rgba_writes_png_and_passes_imgdiff(self):
        width, height = 256, 256
        params = {**self.default_params, "width": width, "height": height, "pitch": width}

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            manifest = compile_pipeline(self.rgba_pipeline, Target.VULKAN, td_path / "vk_rgba", TOOLS)
            rendered = render_compute_rgba(
                manifest,
                width,
                height,
                parameters=params,
                validation=self.validation,
                toolchain=TOOLS,
            )
            out_png = td_path / "mandelbrot_vulkan.png"
            diff_png = td_path / "mandelbrot_vulkan_diff.png"
            write_png(out_png, rendered)
            self.assertTrue(out_png.is_file())

            diff = imgdiff(
                out_png,
                GOLDEN_MANDELBROT_PNG,
                cie76_threshold=2.3,
                max_outlier_fraction=0.002,
                diff_output=diff_png,
            )
            self.assertTrue(diff.passed, diff.summary())
            self.assertTrue(diff_png.is_file())


if __name__ == "__main__":
    unittest.main()
