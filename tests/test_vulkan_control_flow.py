import json
import math
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import CompileError, MLIRPipeline, Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import load_program, render_offscreen
from test_native_ir import TOOLS, available


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(available(), "Build the native gfx compiler for structured Vulkan IR tests")
class VulkanControlFlowIRTests(unittest.TestCase):
    def optimize(self, path, *passes):
        return subprocess.run([TOOLS.find("gfx-opt"), str(path), *passes],
                              text=True, capture_output=True, check=False)

    def test_upstream_helpers_loops_and_selections_remain_structured(self):
        for source in (
            ROOT / "tests/ir/structured_graphics.mlir",
            ROOT / "examples/ir/sphere_march.mlir",
        ):
            with self.subTest(source=source.name):
                result = self.optimize(source, "--convert-gfx-to-spirv")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("spirv.mlir.loop", result.stdout)
                self.assertIn("spirv.mlir.selection", result.stdout)
                self.assertIn("spirv.FunctionCall", result.stdout)
                self.assertNotIn("scf.", result.stdout)
                self.assertNotIn("func.call", result.stdout)

    def test_recursive_external_and_multi_result_calls_are_explicit_diagnostics(self):
        entry = """
  func.func @entry() attributes {gfx.stage = #gfx.stage<fragment>} {
    %value = func.call @helper() : () -> f32
    %zero = arith.constant 0.0 : f32
    %color = vector.from_elements %value, %zero, %zero, %zero : vector<4xf32>
    gfx.output %color 0 "color" <perspective> : vector<4xf32>
    return
  }
"""
        for helper, expected in (
            ("func.func private @helper() -> f32", "structured body"),
            ("""func.func private @helper() -> f32 {
                %value = func.call @helper() : () -> f32
                return %value : f32
              }""", "recursive shader calls"),
        ):
            with self.subTest(helper=helper), tempfile.TemporaryDirectory() as work:
                path = Path(work) / "invalid.mlir"
                path.write_text(f"module {{ {helper} {entry} }}")
                result = self.optimize(path, "--convert-gfx-to-spirv")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(expected, result.stderr)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "multiple.mlir"
            helper = """func.func private @helper() -> (f32, f32) {
              %zero = arith.constant 0.0 : f32
              return %zero, %zero : f32, f32
            }"""
            call = entry.replace("%value = func.call @helper() : () -> f32",
                                 "%value, %unused = func.call @helper() : () -> (f32, f32)")
            path.write_text(f"module {{ {helper} {call} }}")
            result = self.optimize(path, "--convert-gfx-to-spirv")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("at most one result", result.stderr)

    def test_unused_helpers_do_not_become_entry_points_or_break_reflection(self):
        shader = (ROOT / "examples/ir/sphere_march.mlir").read_text()
        unused = "func.func private @unused() { return }"
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "unused.mlir"
            path.write_text(shader.replace("module {", f"module {{ {unused}", 1))
            result = self.optimize(path, "--convert-gfx-to-spirv")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("@unused", result.stdout)
            self.assertEqual(result.stdout.count("spirv.EntryPoint"), 1)
            path.write_text(shader.replace("sphere_distance", "interface_0"))
            result = self.optimize(path, "--convert-gfx-to-spirv")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("@interface_0", result.stdout)


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
