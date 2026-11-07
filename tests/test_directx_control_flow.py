import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import MLIRPipeline, Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen


ROOT = Path(__file__).resolve().parents[1]
@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable Vulkan and Metal GPU tests for DirectX loop comparisons",
)
class DirectXControlFlowRenderingTests(unittest.TestCase):
    def tools(self):
        return Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)

    def render(self, manifest, target, width, height):
        program = load_program(manifest)
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        return (
            render_offscreen(program, width, height, validation)
            if target == Target.VULKAN else
            render_metal_offscreen(program, width, height, validation, self.tools())
        )

    def test_sphere_while_loop_matches_vulkan_and_does_not_grow_with_cap(self):
        pipeline = runpy.run_path(str(ROOT / "examples/sphere_march.py"))["pipeline"]
        images = {}
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            for target in Target:
                manifest = compile_pipeline(pipeline, target, directory / target.value, self.tools())
                images[target] = self.render(manifest, target, 256, 256).rgba
                if target != Target.VULKAN:
                    text = (manifest.parent / "fragment.ll").read_text()
                    self.assertNotIn("@sphere_distance", text)
                    self.assertIn("phi ", text)
                    self.assertIn("br i1 ", text)
                    self.assertEqual(text.count('"hlsl.shader"='), 1)
            for target in (Target.DIRECTX, Target.METAL):
                error = max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target]))
                self.assertLessEqual(error, 2)
            sizes = []
            for cap in (1, 64, 128):
                source = directory / f"cap-{cap}.mlir"
                source.write_text(pipeline.fragment.read_text().replace(
                    "%limit = arith.constant 64 : i32", f"%limit = arith.constant {cap} : i32"))
                changed = MLIRPipeline(pipeline.vertex, source, "fullscreen", "sphere_march")
                manifest = compile_pipeline(changed, Target.METAL, directory / f"cap-out-{cap}", self.tools())
                sizes.append((manifest.parent / "fragment.dxil").stat().st_size)
                if cap == 1:
                    image = self.render(manifest, Target.METAL, 256, 256)
                    pixel = image.rgba[(128 * 256 + 128) * 4:(128 * 256 + 128) * 4 + 4]
                    for actual, expected in zip(pixel, (6, 11, 22, 255)):
                        self.assertAlmostEqual(actual, expected, delta=1)
            self.assertLessEqual(max(sizes) - min(sizes), 64)

    def test_dynamic_loop_band_example_and_vector_helper_return(self):
        pipeline = runpy.run_path(str(ROOT / "examples/loop_bands.py"))["pipeline"]
        images = {}
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                with self.subTest(target=target.value):
                    manifest = compile_pipeline(pipeline, target, Path(work) / target.value, self.tools())
                    image = self.render(manifest, target, 257, 23)
                    images[target] = image.rgba
                    for x in range(257):
                        level = ((x % 256) // 16) / 16
                        expected = (level, level * level, 0.25 + level * 0.5, 1)
                        for y in (0, 11, 22):
                            offset = (y * 257 + x) * 4
                            for actual, channel in zip(image.rgba[offset:offset + 4], expected):
                                self.assertAlmostEqual(actual, round(channel * 255), delta=1)
                    if target != Target.VULKAN:
                        text = (manifest.parent / "fragment.ll").read_text()
                        self.assertNotIn("@tint", text)
                        self.assertIn("phi ", text)
            for target in (Target.DIRECTX, Target.METAL):
                self.assertEqual(images[target], images[Target.VULKAN])

    def test_nested_loop_state_and_branched_helpers_execute_on_metal(self):
        pipeline = MLIRPipeline(ROOT / "examples/ir/fullscreen.mlir",
                                ROOT / "tests/ir/structured_graphics.mlir", "fullscreen", "structured")
        with tempfile.TemporaryDirectory() as work:
            for target in (Target.DIRECTX, Target.METAL):
                manifest = compile_pipeline(pipeline, target, Path(work) / target.value, self.tools())
                image = self.render(manifest, target, 31, 23)
                for offset in range(0, len(image.rgba), 4):
                    for actual, expected in zip(image.rgba[offset:offset + 4], (64, 128, 191, 255)):
                        self.assertAlmostEqual(actual, expected, delta=1)
