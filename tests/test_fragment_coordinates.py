import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import (
    FragmentCoord, Pipeline, Target, Toolchain, Vertex, compile_pipeline, fragment,
    fullscreen_vertex, vec2, vec4, vertex,
)
from mlir_gpu.frontend import Type
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_frontend import evaluate


ROOT = Path(__file__).resolve().parents[1]


class FragmentCoordinateFrontendTests(unittest.TestCase):
    def test_builtin_annotation_does_not_reserve_parameter_names(self):
        @fragment
        def named(pixel: FragmentCoord):
            return vec4(pixel, 0, 1)

        vs, fs = Pipeline(fullscreen_vertex, named).trace()
        self.assertEqual(vs.output_interface[0].builtin, "position")
        self.assertEqual(len(vs.output_interface), 1)
        self.assertEqual(fs.input_interface[0].builtin, "fragment_coord")
        self.assertEqual(fs.input_interface[0].type, Type.VEC2)
        self.assertEqual(evaluate(fs.outputs[0], (0.5, 1.5)), (0.5, 1.5, 0, 1))
        for index, expected in enumerate(((-1, -1, 0, 1), (3, -1, 0, 1), (-1, 3, 0, 1))):
            self.assertEqual(evaluate(vs.outputs[0], index), expected)
        self.assertEqual(named.trace().input_interface, fs.input_interface)

    def test_future_annotations_and_ordinary_varyings_remain_distinct(self):
        @vertex
        def source(index):
            return Vertex(vec4(0, 0, 0, 1), varyings={"pixel": vec2(0, 0)})

        @fragment
        def target(pixel, other: "FragmentCoord"):
            return vec4(pixel + other, 0, 1)

        fs = Pipeline(source, target).trace()[1]
        self.assertEqual(fs.input_interface[0].location, 0)
        self.assertIsNone(fs.input_interface[0].builtin)
        self.assertEqual(fs.input_interface[1].builtin, "fragment_coord")
        @fragment
        def legacy_annotation(pixel: "PreviouslyIgnoredType") -> "PreviouslyIgnoredReturn":
            return vec4(pixel, 0, 1)

        self.assertEqual(Pipeline(source, legacy_annotation).trace()[1].input_interface[0].location, 0)

    def test_incorrect_stage_duplicate_and_unknown_varyings_are_rejected(self):
        @vertex
        def wrong_stage(coord: FragmentCoord):
            return Vertex(vec4(0, 0, 0, 1))

        @fragment
        def duplicate(a: FragmentCoord, b: FragmentCoord):
            return vec4(a + b, 0, 1)

        @fragment
        def unknown(pixel):
            return vec4(pixel, 0, 1)

        for entry, message in ((wrong_stage, "fragment entry"), (duplicate, "only once")):
            with self.assertRaisesRegex(TypeError, message):
                entry.trace()
        with self.assertRaisesRegex(ValueError, "no matching vertex varying"):
            Pipeline(fullscreen_vertex, unknown).trace()


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable Vulkan and Metal tests for canonical fragment-coordinate verification",
)
class FragmentCoordinateRenderingTests(unittest.TestCase):
    def test_fullscreen_pixel_centers_origin_and_coverage_match_all_targets(self):
        tools = Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)
        pipeline = runpy.run_path(str(ROOT / "examples/coordinates.py"))["pipeline"]
        programs = {}
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                manifest = compile_pipeline(pipeline, target, Path(work) / target.value, tools)
                programs[target] = load_program(manifest)
            for width, height in ((63, 47), (256, 256), (320, 180), (257, 193), (126, 94)):
                images = {}
                for target, program in programs.items():
                    with self.subTest(target=target.value, size=(width, height)):
                        image = (
                            render_offscreen(program, width, height, validation)
                            if target == Target.VULKAN else
                            render_metal_offscreen(program, width, height, validation, tools)
                        )
                        self.assertEqual((image.width, image.height), (width, height))
                        self.assertEqual(len(image.rgba), width * height * 4)
                        images[target] = image.rgba
                        for y in range(height):
                            for x in range(width):
                                expected = (
                                    round(((x + 0.5) % 32) / 32 * 255),
                                    round(((y + 0.5) % 32) / 32 * 255),
                                    round(((y + 0.5) % 128) / 128 * 255),
                                    255,
                                )
                                offset = (y * width + x) * 4
                                actual = image.rgba[offset:offset + 4]
                                self.assertLessEqual(max(abs(a - b) for a, b in zip(actual, expected)), 1,
                                                     f"Incorrect pixel center/origin/coverage at {(x, y)}")
                for target in (Target.DIRECTX, Target.METAL):
                    self.assertEqual(images[target], images[Target.VULKAN])

    def test_position_and_user_varying_signature_slots_do_not_collide(self):
        @vertex
        def source(index):
            position = fullscreen_vertex.function(index).position
            return Vertex(position, varyings={"uv": (position.xy + 1) / 2})

        @fragment
        def consumer(uv, pixel: FragmentCoord):
            resolution = vec2(63, 47)
            camera = (pixel - resolution * 0.5) / resolution.y
            camera = camera.with_components(y=-camera.y)
            return vec4(uv, (camera.y + 0.5) * 0.5, 1)

        tools = Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                with self.subTest(target=target.value):
                    manifest = compile_pipeline(Pipeline(source, consumer), target, Path(work) / target.value, tools)
                    program = load_program(manifest)
                    validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
                    image = (
                        render_offscreen(program, 63, 47, validation)
                        if target == Target.VULKAN else
                        render_metal_offscreen(program, 63, 47, validation, tools)
                    )
                    for x, y in ((0, 0), (31, 23), (62, 46)):
                        uv = ((x + 0.5) / 63, 1 - (y + 0.5) / 47)
                        expected = (uv[0], uv[1], uv[1] * 0.5, 1)
                        offset = (y * 63 + x) * 4
                        for actual, channel in zip(image.rgba[offset:offset + 4], expected):
                            self.assertAlmostEqual(actual, round(channel * 255), delta=1)
