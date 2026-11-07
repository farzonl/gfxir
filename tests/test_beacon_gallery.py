import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import Target, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_support import ROOT, TOOLS


def gallery_pipeline():
    return runpy.run_path(str(ROOT / "examples/beacon_gallery.py"))["pipeline"]


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable both Vulkan and Metal for original-scene GPU tests",
)
class GalleryRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.programs = {}
        for target in Target:
            manifest = compile_pipeline(gallery_pipeline(), target, Path(work.name) / target.value, TOOLS)
            cls.programs[target.value] = load_program(manifest)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def test_all_preview_paths_render_and_animate_the_original_scene(self):
        snapshots = {}
        for target, program in self.programs.items():
            render = render_offscreen if target == "vulkan" else render_metal_offscreen
            images = [render(program, 96, 72, self.validation, time=time).rgba for time in (0.25, 1.75)]
            snapshots[target] = images
            self.assertEqual(images[0], render(program, 96, 72, self.validation, time=0.25).rgba)
            self.assertEqual(set(images[0][3::4]), {255})
            self.assertGreater(len(set(images[0][i:i + 3] for i in range(0, len(images[0]), 4))), 100)
            changed = sum(images[0][i:i + 4] != images[1][i:i + 4] for i in range(0, len(images[0]), 4))
            self.assertGreater(changed / (96 * 72), 0.5)
        self.assertEqual(snapshots["directx"], snapshots["metal"])
