import json
import math
import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import CompileError, MLIRPipeline, Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import load_program, render_offscreen


ROOT = Path(__file__).resolve().parents[1]
@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Enable Vulkan GPU tests for native structured-loop rendering",
)
class VulkanControlFlowRenderingTests(unittest.TestCase):
    def tools(self):
        return Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)

    def render(self, manifest, width=256, height=256):
        return render_offscreen(load_program(manifest), width, height,
                                os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1")

    def test_nested_for_helper_results_and_multiple_carried_values(self):
        pipeline = MLIRPipeline(ROOT / "examples/ir/fullscreen.mlir",
                                ROOT / "tests/ir/structured_graphics.mlir", "fullscreen", "structured")
        with tempfile.TemporaryDirectory() as work:
            manifest = compile_pipeline(pipeline, Target.VULKAN, Path(work) / "out", self.tools())
            image = self.render(manifest, 31, 23)
            for offset in range(0, len(image.rgba), 4):
                for actual, expected in zip(image.rgba[offset:offset + 4], (64, 128, 191, 255)):
                    self.assertAlmostEqual(actual, expected, delta=1)
            data = json.loads(manifest.read_text())
            self.assertEqual(data["interface"]["fragment"]["name"], "structured")
            source = Path(work) / "empty.mlir"
            source.write_text(pipeline.fragment.read_text().replace(
                "%sixteen = arith.constant 16 : index", "%sixteen = arith.constant 0 : index"))
            empty = MLIRPipeline(pipeline.vertex, source, "fullscreen", "structured")
            image = self.render(compile_pipeline(empty, Target.VULKAN, Path(work) / "empty", self.tools()), 3, 3)
            self.assertEqual(image.rgba, bytes(3 * 3 * 4))

    def test_sphere_march_hits_misses_early_termination_and_cap_do_not_unroll(self):
        pipeline = runpy.run_path(str(ROOT / "examples/sphere_march.py"))["pipeline"]
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            manifest = compile_pipeline(pipeline, Target.VULKAN, directory / "out", self.tools())
            image = self.render(manifest)
            for x, y in ((128, 128), (140, 115), (0, 0), (255, 255)):
                ray = ((x + 0.5 - 128) / 128, (128 - y - 0.5) / 128, 1)
                length = math.sqrt(sum(value * value for value in ray))
                ray = tuple(value / length for value in ray)
                count, t, hit = 0, 0.0, False
                while count < 64 and t <= 8 and not hit:
                    point = (ray[0] * t, ray[1] * t, -3 + ray[2] * t)
                    distance = math.sqrt(sum(value * value for value in point)) - 0.7
                    hit = distance < 0.001
                    if not hit:
                        t += distance * 0.5
                    count += 1
                color = tuple(value / 0.7 * 0.5 + 0.5 for value in point) if hit else (0.02, 0.04, 0.08)
                expected = tuple(min(1, max(0, value + count * emission)) for value, emission in
                                 zip(color, (0.002, 0.004, 0.008))) + (1,)
                offset = (y * 256 + x) * 4
                for actual, channel in zip(image.rgba[offset:offset + 4], expected):
                    self.assertAlmostEqual(actual, round(channel * 255), delta=2)
            sizes = []
            for cap in (1, 64, 128):
                source = directory / f"cap-{cap}.mlir"
                source.write_text(pipeline.fragment.read_text().replace(
                    "%limit = arith.constant 64 : i32", f"%limit = arith.constant {cap} : i32"))
                changed = MLIRPipeline(pipeline.vertex, source, "fullscreen", "sphere_march")
                result = compile_pipeline(changed, Target.VULKAN, directory / f"out-{cap}", self.tools())
                sizes.append((result.parent / "fragment.spv").stat().st_size)
                if cap == 1:
                    limited = self.render(result, 256, 256)
                    pixel = limited.rgba[(128 * 256 + 128) * 4:(128 * 256 + 128) * 4 + 4]
                    for actual, expected in zip(pixel, (6, 11, 22, 255)):
                        self.assertAlmostEqual(actual, expected, delta=1)
            self.assertLessEqual(max(sizes) - min(sizes), 32, "Changing a bound duplicated the loop body")

    def test_native_entry_mismatch_and_missing_source_leave_no_artifacts(self):
        source = ROOT / "examples/ir/fullscreen.mlir"
        with tempfile.TemporaryDirectory() as work:
            for fragment, name, message in (
                (ROOT / "examples/ir/sphere_march.mlir", "incorrect", "Reflected entry"),
                (Path(work) / "missing.mlir", "sphere_march", "Cannot read native shader"),
            ):
                output = Path(work) / name
                with self.assertRaisesRegex(CompileError, message):
                    compile_pipeline(MLIRPipeline(source, fragment, "fullscreen", name),
                                     Target.VULKAN, output, self.tools())
                self.assertFalse(output.exists())
