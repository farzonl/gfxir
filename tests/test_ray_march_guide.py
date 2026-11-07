import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import Target, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_native_ir import ROOT, TOOLS


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable both Vulkan and Metal for executable tutorial tests",
)
class RayMarchGuideTests(unittest.TestCase):
    def test_lit_sphere_tutorial_is_portable_deterministic_and_animates_lighting(self):
        pipeline = runpy.run_path(str(ROOT / "examples/lit_sphere.py"))["pipeline"]
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        snapshots = {}
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                manifest = compile_pipeline(pipeline, target, Path(work) / target.value, TOOLS)
                program = load_program(manifest)
                render = render_offscreen if target == Target.VULKAN else render_metal_offscreen
                first = render(program, 96, 72, validation, time=0).rgba
                second = render(program, 96, 72, validation, time=1.5).rgba
                self.assertEqual(first, render(program, 96, 72, validation, time=0).rgba)
                self.assertEqual(set(first[3::4]), {255})
                changed = sum(first[i:i + 4] != second[i:i + 4] for i in range(0, len(first), 4))
                self.assertGreater(changed / (96 * 72), 0.1)
                snapshots[target] = second
        self.assertEqual(snapshots[Target.DIRECTX], snapshots[Target.METAL])
        self.assertLessEqual(max(abs(a - b) for a, b in zip(
            snapshots[Target.VULKAN], snapshots[Target.METAL])), 2)
