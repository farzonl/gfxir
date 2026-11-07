import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import CompileError, MLIRPipeline, Target, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_support import ROOT, TOOLS
from compiler_inputs import SOURCE, block, constant, operations, result, source


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable Vulkan and Metal GPU tests for structured input",
)
class StructuredInputRenderingTests(unittest.TestCase):
    def render(self, manifest, target, width=128, height=128):
        program = load_program(manifest)
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        if target == Target.VULKAN:
            return render_offscreen(program, width, height, validation)
        return render_metal_offscreen(program, width, height, validation, TOOLS)

    def test_julia_iteration_state_helper_and_escape_match_all_targets(self):
        pipeline = runpy.run_path(str(ROOT / "examples/julia.py"))["pipeline"]
        images = {}
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                manifest = compile_pipeline(pipeline, target, Path(work) / target.value, TOOLS)
                images[target] = self.render(manifest, target).rgba
                for px, py in ((0, 0), (64, 64), (20, 64), (60, 50), (100, 20)):
                    x, y = (px + 0.5 - 64) / 48, (py + 0.5 - 64) / 48
                    count, escaped = 0, False
                    while count < 32 and not escaped:
                        xx, yy = x * x, y * y
                        escaped = xx + yy > 4
                        if not escaped:
                            x, y = xx - yy - 0.7, 2 * x * y + 0.27
                        count += 1
                    level = count / 32
                    expected = (level, level * level + count * 0.003, level * 0.5 + count * 0.008, 1)
                    offset = (py * 128 + px) * 4
                    for actual, channel in zip(images[target][offset:offset + 4], expected):
                        self.assertAlmostEqual(actual, round(min(channel, 1) * 255), delta=2)
            for target in (Target.DIRECTX, Target.METAL):
                error = max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target]))
                self.assertLessEqual(error, 2)

    def test_structured_for_zero_trip_and_carried_values_execute_on_all_targets(self):
        for upper in (0, 4):
            shader = {
                "schema_version": 2, "functions": [{
                    "name": "loop", "stage": "fragment", "results": [],
                    "body": {"blocks": [block([
                        constant("lower", "index", 0), constant("upper", "index", upper),
                        constant("step", "index", 1), constant("zero", "f32", 0.0),
                        constant("increment", "f32", 0.125),
                        {"op": "scf.for", "args": ["lower", "upper", "step", "zero"],
                         "results": [result("sum", "f32")],
                         "regions": [{"blocks": [block([
                             {"op": "arith.addf", "args": ["carried", "increment"], "results": [result("next", "f32")]},
                             {"op": "scf.yield", "args": ["next"]},
                         ], [result("i", "index"), result("carried", "f32")])]}]},
                        {"op": "vector.broadcast", "args": ["sum"], "results": [result("color", "f32x4")]},
                        {"op": "gfx.output", "args": ["color"],
                         "attributes": operations(source()["functions"][1])[-2]["attributes"]},
                        {"op": "func.return"},
                    ])]},
                }],
            }
            with tempfile.TemporaryDirectory() as work:
                directory = Path(work)
                fragment = directory / "loop.json"
                fragment.write_text(json.dumps(shader))
                pipeline = MLIRPipeline(ROOT / "examples/ir/fullscreen.mlir", fragment, "fullscreen", "loop")
                for target in Target:
                    with self.subTest(upper=upper, target=target.value):
                        manifest = compile_pipeline(pipeline, target, directory / target.value, TOOLS)
                        self.assertEqual(self.render(manifest, target, 1, 1).rgba,
                                         bytes([round(upper * 0.125 * 255)] * 4))

    def test_structured_entry_mismatch_leaves_no_published_artifacts(self):
        pipeline = MLIRPipeline(ROOT / "examples/ir/fullscreen.mlir", SOURCE, "fullscreen", "wrong")
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "output"
            with self.assertRaisesRegex(CompileError, "Reflected entry"):
                compile_pipeline(pipeline, Target.VULKAN, output, TOOLS)
            self.assertFalse(output.exists())
