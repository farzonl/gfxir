import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import Target, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_native_ir import ROOT, TOOLS
import test_structured_input as structured


def gallery_pipeline():
    return runpy.run_path(str(ROOT / "examples/beacon_gallery.py"))["pipeline"]


class GalleryAuthoringTests(unittest.TestCase):
    def test_scene_has_a_device_loop_and_reachable_parameter_reads(self):
        shader = gallery_pipeline().trace()[1].to_dict()
        operations = [op for function in shader["functions"] for op in structured.operations(function)]
        self.assertTrue(any(op["op"] == "scf.while" for op in operations))
        self.assertTrue(any(op["op"] == "gfx.parameter" for op in operations))
        self.assertEqual(shader["parameters"]["name"], "frame")


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
