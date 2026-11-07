import math
import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import FrameParameters, FragmentCoord, Pipeline, Target, compile_pipeline, fragment, fullscreen_vertex, vec3, vec4
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen, show_metal_window, show_window
from test_native_ir import ROOT, TOOLS


SOURCE = ROOT / "examples/light_cycle.py"
map_scene = runpy.run_path(str(SOURCE))["map_scene"]


@fragment
def geometry_diagnostic(pixel: FragmentCoord, frame: FrameParameters):
    xy = pixel / frame.resolution * 8.0 - 4.0
    point = vec3(xy, frame.time * 4.0 - 4.0)
    return vec4(map_scene(point, frame.time) / 16.0 + 0.5, 1)


def box_reference(point, size):
    q = [abs(p) - b for p, b in zip(point, size)]
    return math.sqrt(sum(max(v, 0) ** 2 for v in q)) + min(max(q), 0)


def geometry_reference(point, time):
    x, y, z = point
    x -= math.sin(z * 0.05) * 2.4 + math.cos(z * 0.025)
    y -= math.cos(z * 0.04) * 2.0 + math.sin(z * 0.02) * 0.8
    angle = z * 0.02 + time * 0.05
    x, y = math.cos(angle) * x - math.sin(angle) * y, math.sin(angle) * x + math.cos(angle) * y
    x, y = max(abs(x), abs(y)), min(abs(x), abs(y))
    cell_z = (z + 2) % 4 - 2
    recess_center = box_reference((x - 4, y, cell_z), (0.5, 1.488, 1.488)) - 0.012
    recess_side = box_reference((x - 4, y - 2.75, cell_z), (0.5, 0.738, 1.488)) - 0.012
    architecture = max(4 - x, -min(recess_center, recess_side))
    rails = min(box_reference((x - 4, y - offset), (0.05, 0.05)) for offset in (1.5, 2, 3.5))
    rings = min(box_reference((x - 4, y - 2, abs(cell_z) - offset), (0.05, 2, 0.05)) for offset in (1.5, 2))
    cyan = min(rails, rings)
    orange = box_reference((x - 3.92, y - 3.92), (0.11, 0.11))
    return min(architecture, cyan, orange), cyan, orange


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable both GPU paths for the user-provided Light Cycle example",
)
class LightCycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.programs = {}
        cls.directory = Path(work.name)
        for target in Target:
            for name, pipeline in (
                ("scene", runpy.run_path(str(SOURCE))["pipeline"]),
                ("geometry", Pipeline(fullscreen_vertex, geometry_diagnostic)),
            ):
                directory = cls.directory / f"{name}-{target.value}"
                manifest = compile_pipeline(pipeline, target, directory, TOOLS)
                cls.programs[name, target] = load_program(manifest)
                if name == "scene":
                    original = (directory / "fragment.mlir").read_text()
                    if original.count("scf.while") != 1 or "arith.constant 110 : i32" not in original:
                        raise AssertionError("Light Cycle must retain its 110-step device loop")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def render(self, name, target, width, height, time):
        render = render_offscreen if target == Target.VULKAN else render_metal_offscreen
        return render(self.programs[name, target], width, height, self.validation, time=time).rgba

    def test_geometry_matches_independent_cpu_reference_including_negative_cells_and_rotation(self):
        width, height = 17, 13
        for time in (0, 0.75, 1.25):
            expected = []
            for y in range(height):
                for x in range(width):
                    point = ((x + 0.5) / width * 8 - 4, (y + 0.5) / height * 8 - 4, time * 4 - 4)
                    distances = geometry_reference(point, time)
                    expected.extend(round(min(max(d / 16 + 0.5, 0), 1) * 255) for d in distances)
                    expected.append(255)
            for target in Target:
                image = self.render("geometry", target, width, height, time)
                self.assertLessEqual(max(abs(a - b) for a, b in zip(image, expected)), 1)

    def test_scene_is_deterministic_animated_and_portable_at_multiple_sizes(self):
        for width, height in ((64, 64), (96, 54), (97, 73)):
            snapshots = {}
            for target in Target:
                first = self.render("scene", target, width, height, 0.25)
                moved = self.render("scene", target, width, height, 1.25)
                self.assertEqual(first, self.render("scene", target, width, height, 0.25))
                self.assertEqual(set(first[3::4]), {255})
                changed = sum(first[i:i + 4] != moved[i:i + 4] for i in range(0, len(first), 4))
                self.assertGreater(changed / (width * height), 0.5)
                snapshots[target] = (first, moved)
            self.assertEqual(snapshots[Target.DIRECTX], snapshots[Target.METAL])
            for index, time in enumerate((0.25, 1.25)):
                differences = [abs(a - b) for i, (a, b) in enumerate(zip(
                    snapshots[Target.VULKAN][index], snapshots[Target.METAL][index])) if i % 4 != 3]
                peaks = [max(differences[i:i + 3]) for i in range(0, len(differences), 3)]
                mean = sum(differences) / len(differences)
                p99 = sorted(peaks)[math.ceil(len(peaks) * 0.99) - 1]
                print(f"Light Cycle {width}x{height} t={time}: mean={mean:.4f}, p99={p99}, max={max(peaks)}")
                self.assertLessEqual(mean, 1)
                self.assertLessEqual(p99, 4)
                self.assertLessEqual(sum(p > 16 for p in peaks) / len(peaks), 0.005)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable real desktop previews")
    def test_documented_workload_presents_bounded_live_frames_on_all_paths(self):
        for target in Target:
            program = self.programs["scene", target]
            show = show_window if target == Target.VULKAN else show_metal_window
            show(program, 320, 240, 4, self.validation)
