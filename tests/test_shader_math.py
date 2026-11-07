import math
import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_frontend import evaluate


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable both Vulkan and Metal tests for shader math GPU references",
)
class ShaderMathRenderingTests(unittest.TestCase):
    def test_math_example_matches_cpu_reference_and_other_targets(self):
        tools = Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)
        pipeline = runpy.run_path(str(ROOT / "examples/math.py"))["pipeline"]
        fragment = pipeline.trace()[1]
        width, height = 63, 47
        images = {}
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                with self.subTest(target=target.value):
                    manifest = compile_pipeline(pipeline, target, Path(work) / target.value, tools)
                    program = load_program(manifest)
                    image = (
                        render_offscreen(program, width, height, validation)
                        if target == Target.VULKAN else
                        render_metal_offscreen(program, width, height, validation, tools)
                    )
                    images[target] = image.rgba
                    self.assertEqual(len(image.rgba), width * height * 4)
                    # Avoid exact procedural grid boundaries, where interpolation rounding changes floor.
                    for x, y in ((30, 23), (8, 11), (51, 38), (20, 16)):
                        uv = ((x + 0.5) / width, 1 - (y + 0.5) / height)
                        expected = evaluate(fragment.outputs[0], uv)
                        offset = (y * width + x) * 4
                        for actual, channel in zip(image.rgba[offset:offset + 4], expected):
                            self.assertTrue(math.isfinite(channel))
                            self.assertAlmostEqual(actual, round(channel * 255), delta=2)
        for target in (Target.DIRECTX, Target.METAL):
            error = max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target]))
            self.assertLessEqual(error, 2, f"{target.value} math colors differ by more than 2/255")
