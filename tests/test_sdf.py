import math
import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import (
    Pipeline, Target, Toolchain, compile_pipeline, fragment, sdf, select,
    shader_math as sm, vec2, vec3, vec4,
)
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen
from test_frontend import evaluate


ROOT = Path(__file__).resolve().parents[1]


class ShaderUtilityTests(unittest.TestCase):
    def test_vector_math_and_zero_normalization(self):
        self.assertEqual(evaluate(sm.dot(vec2(1, 2), vec2(3, 4)).node, None), 11)
        self.assertEqual(evaluate(sm.cross(vec3(1, 0, 0), vec3(0, 1, 0)).node, None), (0, 0, 1))
        self.assertEqual(evaluate(sm.length(vec3(3, 4, 0)).node, None), 5)
        self.assertEqual(evaluate(sm.normalize(vec3(0)).node, None), (0, 0, 0))
        self.assertEqual(evaluate(sm.normalize(vec2(3, 4)).node, None), (0.6, 0.8))
        rotated = evaluate(sm.rotate2(vec2(1, 0), math.pi / 2).node, None)
        self.assertAlmostEqual(rotated[0], 0, places=6)
        self.assertAlmostEqual(rotated[1], 1, places=6)

    def test_scalar_vector_interpolation_and_negative_repetition(self):
        self.assertEqual(evaluate(sm.clamp(vec3(-1, 0.5, 2), 0, 1).node, None), (0, 0.5, 1))
        self.assertEqual(evaluate(sm.mix(vec3(0), vec3(1), 0.25).node, None), (0.25,) * 3)
        self.assertEqual(evaluate(sm.smoothstep(0, 1, vec3(-1, 0.5, 2)).node, None), (0, 0.5, 1))
        self.assertEqual(evaluate(sm.fract(vec3(-1.25, -0.25, 0.25)).node, None), (0.75, 0.75, 0.25))
        self.assertEqual(evaluate(sm.mod(vec3(-3, -0.25, 3), 2).node, None), (1, 1.75, 1))
        self.assertEqual(evaluate(sdf.repeat(vec3(-3, -0.25, 3), 2).node, None), (-1, -0.25, -1))
        self.assertEqual(evaluate(sdf.fold_square(vec2(-0.25, -1)).node, None), (1, 0.25))

    def test_distance_values_and_tetrahedral_normals(self):
        for point, expected in (((0, 0, 0), -1), ((1, 0, 0), 0), ((2, 0, 0), 1)):
            self.assertEqual(evaluate(sdf.sphere(vec3(*point), 1).node, None), expected)
            self.assertEqual(evaluate(sdf.box(vec3(*point), vec3(1)).node, None), expected)
        self.assertAlmostEqual(evaluate(sdf.box(vec3(2), vec3(1)).node, None), math.sqrt(3))
        self.assertEqual(evaluate(sdf.box2(vec2(0), vec2(1)).node, None), -1)
        for point in (vec3(1, 0, 0), vec3(1, 2, 3)):
            expected = evaluate(sm.normalize(point).node, None)
            actual = evaluate(sdf.normal(lambda p: sdf.sphere(p, 1), point).node, None)
            for a, b in zip(actual, expected):
                self.assertAlmostEqual(a, b, delta=0.002)
        self.assertEqual(evaluate(sdf.normal(lambda p: p.x * 0, vec3(1)).node, None), (0, 0, 0))

    def test_invalid_types_dimensions_and_known_domains_are_diagnostics(self):
        invalid_types = (
            lambda: sm.dot(vec2(1), vec3(1)), lambda: sm.cross(vec2(1), vec2(1)),
            lambda: sm.length(1), lambda: sm.rotate2(vec3(1), 0),
            lambda: sdf.sphere(vec2(1), 1), lambda: sdf.box(vec3(1), vec2(1)),
            lambda: sdf.normal(lambda p: p, vec3(1)),
        )
        for operation in invalid_types:
            with self.assertRaises(TypeError):
                operation()
        invalid_domains = (
            lambda: sm.clamp(0, 1, 0), lambda: sm.smoothstep(1, 1, 0),
            lambda: sm.mod(1, 0), lambda: sdf.repeat(vec2(0), -1),
            lambda: sdf.sphere(vec3(0), -1), lambda: sdf.box2(vec2(0), vec2(-1, 1)),
            lambda: sdf.normal(lambda p: p.x, vec3(1), epsilon=0),
        )
        for operation in invalid_domains:
            with self.assertRaises(ValueError):
                operation()


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable Vulkan and Metal tests for reusable SDF GPU verification",
)
class SDFRenderingTests(unittest.TestCase):
    def test_helpers_execute_and_match_bounded_analytic_identities(self):
        quad = runpy.run_path(str(ROOT / "examples/quad.py"))["pipeline"]

        @fragment
        def diagnostic(uv):
            point = vec3(uv - 0.5, 0.75)
            normalized = sm.normalize(point)
            normal = sdf.normal(lambda p: sdf.sphere(p, 1.0), point)
            recovered = sm.mod(uv - 2, 1)
            repeated = sdf.repeat(uv - 2, 1)
            folded = sdf.fold_square(uv - 0.5)
            rotated = sm.rotate2(sm.rotate2(uv, 0.3), -0.3)
            expected_step = uv * uv * (3 - 2 * uv)
            valid = (
                (abs(sm.length(normalized) - 1) < 2e-5)
                & (sm.length(normal - normalized) < 0.005)
                & (sm.length(recovered - uv) < 2e-5)
                & (sm.length(repeated) < 0.71)
                & (folded.x >= folded.y) & (folded.y >= 0)
                & (sm.length(rotated - uv) < 2e-5)
                & (sm.length(sm.smoothstep(0, 1, uv) - expected_step) < 2e-5)
                & (sm.length(sm.mix(vec2(0), vec2(1), uv) - uv) < 2e-5)
                & (sdf.box2(vec2(0), vec2(1)) == -1)
                & (sdf.box(vec3(0), vec3(1)) == -1)
                & (sm.dot(sm.normalize(vec3(0)), vec3(1)) == 0)
                & (sm.cross(vec3(1, 0, 0), vec3(0, 1, 0)).z == 1)
            )
            return select(valid, vec4(1), vec4(0, 0, 0, 1))

        tools = Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                with self.subTest(target=target.value):
                    manifest = compile_pipeline(Pipeline(quad.vertex, diagnostic, quad.draw),
                                                target, Path(work) / target.value, tools)
                    image = self.render(load_program(manifest), target, tools)
                    covered = [image.rgba[offset:offset + 4] for offset in range(0, len(image.rgba), 4)
                               if image.rgba[offset:offset + 4] != bytes((5, 5, 10, 255))]
                    self.assertGreater(len(covered), 63 * 47 * 0.6)
                    self.assertTrue(all(pixel == bytes((255, 255, 255, 255)) for pixel in covered))

    def render(self, program, target, tools):
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        return (
            render_offscreen(program, 63, 47, validation)
            if target == Target.VULKAN else
            render_metal_offscreen(program, 63, 47, validation, tools)
        )

    def test_new_examples_match_cpu_reference_and_cross_target_colors(self):
        tools = Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)
        with tempfile.TemporaryDirectory() as work:
            for example in ("distance_fields", "surface_normals"):
                pipeline = runpy.run_path(str(ROOT / f"examples/{example}.py"))["pipeline"]
                fragment_stage = pipeline.trace()[1]
                images = {}
                for target in Target:
                    with self.subTest(example=example, target=target.value):
                        manifest = compile_pipeline(pipeline, target, Path(work) / f"{example}-{target.value}", tools)
                        image = self.render(load_program(manifest), target, tools)
                        images[target] = image.rgba
                        for x, y in ((3, 5), (20, 16), (31, 23), (51, 38)):
                            reference = evaluate(fragment_stage.outputs[0], (x + 0.5, y + 0.5))
                            offset = (y * 63 + x) * 4
                            for actual, channel in zip(image.rgba[offset:offset + 4], reference):
                                self.assertAlmostEqual(actual, round(channel * 255), delta=2)
                for target in (Target.DIRECTX, Target.METAL):
                    error = max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target]))
                    self.assertLessEqual(error, 2)
