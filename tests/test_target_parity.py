import os
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import Pipeline, Target, Toolchain, Vertex, compile_pipeline, fragment, select, vec4, vertex
from mlir_gpu.viewer import load_program, render_directx_offscreen, render_metal_offscreen, render_offscreen


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = bytes([5, 5, 10, 255])


def positioned_triangle(positions):
    @vertex
    def parity_vertex(index):
        def component(values):
            return select(index == 0, float(values[0]), select(index == 1, float(values[1]), float(values[2])))

        return Vertex(
            vec4(*(component([position[axis] for position in positions]) for axis in range(4))),
            vec4(component([1, 0, 0]), component([0, 1, 0]), component([0, 0, 1]), 1.0),
        )

    @fragment
    def parity_fragment(color):
        return color

    return Pipeline(parity_vertex, parity_fragment)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and (
        os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1" if sys.platform == "win32" else
        os.environ.get("MLIR_GPU_TEST_METAL") == "1"
    ),
    "Enable Vulkan plus DirectX (Windows) or Metal (macOS) for cross-target GPU parity",
)
class TargetParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        llvm = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
        tools = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
        cls.tools = Toolchain(llvm, tools)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.positions = {
            "triangle": [(-0.6, -0.6, 0.0, 1.0), (0.6, -0.6, 0.0, 1.0), (0.0, 0.6, 0.0, 1.0)],
            "asymmetric-perspective": [(-0.75, -0.4, 0.2, 1.0), (0.65, -0.55, 0.6, 1.5), (-0.15, 0.85, 0.4, 1.2)],
            "depth-clipping": [(-0.75, -0.4, -0.5, 1.0), (0.65, -0.55, 1.8, 1.5), (-0.15, 0.85, 0.4, 1.2)],
            "quad": [(-0.8, -0.8, 0.0, 1.0), (0.8, -0.8, 0.0, 1.0), (-0.8, 0.8, 0.0, 1.0), (0.8, 0.8, 0.0, 1.0)],
        }
        cls.programs = {}
        for name, positions in cls.positions.items():
            pipeline = (
                runpy.run_path(str(ROOT / f"examples/{name}.py"))["pipeline"]
                if name in ("triangle", "quad") else positioned_triangle(positions)
            )
            cls.programs[name] = {}
            targets = (Target.VULKAN, Target.DIRECTX) if sys.platform == "win32" else tuple(Target)
            for target in targets:
                manifest = compile_pipeline(pipeline, target, Path(work.name) / f"{name}-{target.value}", cls.tools)
                cls.programs[name][target] = load_program(manifest)

    def compare_targets(self, name):
        for width, height in ((256, 256), (320, 180), (257, 193)):
            images, masks = {}, {}
            for target, program in self.programs[name].items():
                with self.subTest(scene=name, target=target.value, size=(width, height)):
                    image = (
                        render_offscreen(program, width, height, self.validation)
                        if target == Target.VULKAN else
                        (render_directx_offscreen if sys.platform == "win32" else render_metal_offscreen)(
                            program, width, height, self.validation, self.tools
                        )
                    )
                    self.assertEqual((image.width, image.height), (width, height))
                    self.assertEqual(len(image.rgba), width * height * 4)
                    images[target] = image.rgba
                    masks[target] = bytes(
                        image.rgba[offset:offset + 4] != BACKGROUND
                        for offset in range(0, len(image.rgba), 4)
                    )
                    self.assertGreater(sum(masks[target]), width * height * 0.01)
                    if name == "triangle":
                        self.assertAlmostEqual(sum(masks[target]) / (width * height), 0.18, delta=0.01)
                    elif name == "quad":
                        self.assertAlmostEqual(sum(masks[target]) / (width * height), 0.64, delta=0.01)
                        for x, y in ((width // 3, height // 3), (2 * width // 3, 2 * height // 3)):
                            u = ((x + 0.5) / width * 2 - 1 + 0.8) / 1.6
                            v = (1 - (y + 0.5) / height * 2 + 0.8) / 1.6
                            pixel = image.rgba[(y * width + x) * 4:(y * width + x) * 4 + 4]
                            for channel, expected in zip(pixel, (u, v, u * v, 1)):
                                self.assertAlmostEqual(channel, round(expected * 255), delta=2)
            reference = images[Target.VULKAN]
            for target in images:
                if target == Target.VULKAN:
                    continue
                with self.subTest(scene=name, target=target.value, size=(width, height)):
                    self.assertEqual(masks[target], masks[Target.VULKAN], "Coverage differs: orientation, scale, or clipping mismatch")
                    maximum_error = max(abs(left - right) for left, right in zip(reference, images[target]))
                    self.assertLessEqual(maximum_error, 2, "Interpolated RGBA colors differ by more than 2/255")
            if name != "depth-clipping":
                covered = [offset for offset, pixel in enumerate(masks[Target.VULKAN]) if pixel]
                actual = (
                    min(offset % width for offset in covered), min(offset // width for offset in covered),
                    max(offset % width for offset in covered), max(offset // width for offset in covered),
                )
                xs = [(position[0] / position[3] * 0.5 + 0.5) * width for position in self.positions[name]]
                ys = [(0.5 - position[1] / position[3] * 0.5) * height for position in self.positions[name]]
                expected = (min(xs), min(ys), max(xs), max(ys))
                for actual_edge, expected_edge in zip(actual, expected):
                    self.assertAlmostEqual(actual_edge + 0.5, expected_edge, delta=2, msg="Incorrect canonical projection scale or orientation")

    def test_triangle_matches_across_all_targets(self):
        self.compare_targets("triangle")

    def test_asymmetric_perspective_catches_mirroring_and_scaling(self):
        self.compare_targets("asymmetric-perspective")

    def test_zero_to_one_depth_clipping_matches_across_targets(self):
        self.compare_targets("depth-clipping")

    def test_six_vertex_quad_with_vec2_varyings_and_fragment_math(self):
        self.compare_targets("quad")


if __name__ == "__main__":
    unittest.main()
