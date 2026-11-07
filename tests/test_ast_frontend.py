import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import FragmentCoord, Pipeline, Target, fragment, fullscreen_vertex, select, shader_math as sm, vec3, vec4, compile_pipeline
from test_native_ir import TOOLS
import test_structured_input as structured_input


ROOT = Path(__file__).resolve().parents[1]


def count_ops(value, operation):
    if isinstance(value, dict):
        return int(value.get("op") == operation) + sum(count_ops(item, operation) for item in value.values())
    if isinstance(value, list):
        return sum(count_ops(item, operation) for item in value)
    return 0


class ASTFrontendTests(unittest.TestCase):
    def test_default_ast_loop_serialization_does_not_expand_iteration_count(self):
        lengths = []
        for limit in (1, 64, 1024):
            @fragment
            def loop(pixel: FragmentCoord):
                value = 0.0
                for i in range(limit):
                    value += pixel.x * 0.001
                return vec4(value, 0, 0, 1)
            stage = Pipeline(fullscreen_vertex, loop).trace()[1]
            module = stage.to_dict()
            self.assertEqual(module["schema_version"], 2)
            self.assertEqual(count_ops(module, "scf.while"), 1)
            lengths.append(len(json.dumps(module)))
        self.assertLessEqual(max(lengths) - min(lengths), 32)

    def test_default_ast_does_not_execute_entry_side_effects(self):
        effects = [0]
        @fragment
        def invalid(pixel: FragmentCoord):
            effects[0] += 1
            return vec4(pixel, 0, 1)
        with self.assertRaisesRegex(TypeError, "local name"):
            Pipeline(fullscreen_vertex, invalid).trace()
        self.assertEqual(effects, [0])

    def test_unsupported_source_has_file_line_and_column(self):
        @fragment
        def invalid(pixel: FragmentCoord):
            if pixel.x:
                return vec4(1)
            return vec4(0)
        with self.assertRaisesRegex(TypeError, r"test_ast_frontend.py:\d+:\d+: Device conditions"):
            Pipeline(fullscreen_vertex, invalid).trace()
        @fragment
        def missing(pixel: FragmentCoord):
            return unresolved_helper(pixel)
        with self.assertRaisesRegex(TypeError, r"test_ast_frontend.py:\d+:\d+: Unresolved"):
            Pipeline(fullscreen_vertex, missing).trace()

    def test_recursion_is_rejected_without_running_python_recursion(self):
        def recursive(value):
            return recursive(value)
        @fragment
        def shader(pixel: FragmentCoord):
            return vec4(recursive(pixel.x), 0, 0, 1)
        with self.assertRaisesRegex(TypeError, "Recursive device helper"):
            Pipeline(fullscreen_vertex, shader).trace()

    def test_constant_short_circuit_preserves_python_operand_values(self):
        @fragment
        def shader(pixel: FragmentCoord):
            a = 0.25 or unresolved_helper()
            b = 0 and unresolved_helper()
            return vec4(a, b, pixel.x, 1)
        stage = Pipeline(fullscreen_vertex, shader).trace()[1]
        self.assertNotIn("schema_version", stage.to_dict())
        self.assertEqual(stage.outputs[0].args[0].literal, 0.25)
        self.assertEqual(stage.outputs[0].args[1].literal, 0.0)

    def test_primitive_helper_captures_and_quad_metaprogramming_stay_compatible(self):
        pipeline = runpy.run_path(str(ROOT / "examples/quad.py"))["pipeline"]
        vertex, _ = pipeline.trace()
        self.assertNotIn("schema_version", vertex.to_dict())
        self.assertEqual(vertex.output_interface[1].name, "uv")
        constant = 0.25
        def helper(value):
            return value + constant
        @fragment
        def shader(pixel: FragmentCoord):
            return vec4(helper(pixel.x), 0, 0, 1)
        stage = Pipeline(fullscreen_vertex, shader).trace()[1]
        self.assertEqual(stage.outputs[0].args[0].args[1].literal, constant)

    def test_device_short_circuit_places_unsafe_operation_inside_selection(self):
        @fragment
        def shader(pixel: FragmentCoord):
            x = pixel.x - 0.5
            safe = x > 0.0 and sm.log(x) > 0.0
            return vec4(select(safe, 1.0, 0.0), 0, 0, 1)
        module = Pipeline(fullscreen_vertex, shader).trace()[1].to_dict()
        root = module["functions"][0]["body"]["blocks"][0]["operations"]
        self.assertFalse(any(op["op"] == "math.log" for op in root))
        self.assertEqual(count_ops(module, "math.log"), 1)
        self.assertGreaterEqual(count_ops(module, "scf.if"), 1)

    def test_static_local_helpers_capture_values_and_support_comprehensions(self):
        @fragment
        def shader(pixel: FragmentCoord):
            scale = 0.25
            def helper(value, offset=0.125):
                return value * scale + offset
            values = tuple(helper(pixel.x, offset=float(i)) for i in range(3))
            return vec4(*values, 1)
        stage = Pipeline(fullscreen_vertex, shader).trace()[1]
        self.assertNotIn("schema_version", stage.to_dict())
        self.assertEqual(stage.outputs[0].args[0].args[0].args[1].literal, 0.25)

    def test_rebound_local_captures_are_rejected_instead_of_using_stale_state(self):
        @fragment
        def shader(pixel: FragmentCoord):
            scale = 0.25
            def helper(value):
                return value * scale
            scale = 0.5
            return vec4(helper(pixel.x), 0, 0, 1)
        with self.assertRaisesRegex(TypeError, r"test_ast_frontend.py:\d+:\d+: Captured local 'scale'"):
            Pipeline(fullscreen_vertex, shader).trace()

    def test_domains_and_variable_type_changes_have_source_context(self):
        @fragment
        def invalid(pixel: FragmentCoord):
            return vec4(sm.sqrt(-1.0), pixel.x, 0, 1)
        with self.assertRaisesRegex(ValueError, r"test_ast_frontend.py:\d+:\d+: sqrt argument"):
            Pipeline(fullscreen_vertex, invalid).trace()
        @fragment
        def changing(pixel: FragmentCoord):
            value = 0
            for i in range(3):
                value = pixel.x
            return vec4(float(value), 0, 0, 1)
        with self.assertRaisesRegex(TypeError, r"test_ast_frontend.py:\d+:\d+: Expected I32"):
            Pipeline(fullscreen_vertex, changing).trace()

    def test_static_iteration_uses_a_trampoline_not_python_recursion(self):
        @fragment
        def shader(pixel: FragmentCoord):
            value = 0.0
            for i in reversed(range(1500)):
                value = float(i)
            return vec4(value, pixel.x, 0, 1)
        stage = Pipeline(fullscreen_vertex, shader).trace()[1]
        self.assertEqual(stage.outputs[0].args[0].literal, 0.0)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable real Vulkan and Metal rendering for Python AST tests",
)
class ASTRenderingTests(unittest.TestCase):
    render = structured_input.StructuredInputRenderingTests.render

    def assert_targets(self, shader, expected, width=1, height=1):
        pipeline = Pipeline(fullscreen_vertex, shader)
        images = {}
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                with self.subTest(target=target.value):
                    manifest = compile_pipeline(pipeline, target, Path(work) / target.value, TOOLS)
                    image = self.render(manifest, target, width, height)
                    images[target] = image.rgba
                    for pixel in range(width * height):
                        channels = expected(pixel)
                        for actual, value in zip(image.rgba[pixel * 4:pixel * 4 + 4], channels):
                            self.assertAlmostEqual(actual, round(min(1, max(0, value)) * 255), delta=2)
            for target in (Target.DIRECTX, Target.METAL):
                self.assertLessEqual(max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target])), 2)

    def test_range_break_continue_and_carried_vector_updates_preserve_order(self):
        @fragment
        def shader(pixel: FragmentCoord):
            value = 0.0
            color = vec3(0)
            for i in range(6):
                if i == 1:
                    continue
                value += 0.05
                color += vec3(0.05, 0.1, 0.15)
                if i == 3:
                    break
                value += 0.1
            return vec4(color + vec3(value), 1)
        self.assert_targets(shader, lambda pixel: (0.5, 0.65, 0.8, 1))

    def test_nested_while_loops_and_helper_early_return_execute_on_device(self):
        def helper(value):
            total = 0.0
            for i in range(2):
                j = 0
                while j < 3:
                    j += 1
                    if j == 2:
                        continue
                    total += value
                    if i == 1 and j == 3:
                        return total
            return total + 0.5
        @fragment
        def shader(pixel: FragmentCoord):
            return vec4(helper(pixel.x * 0.1), 0, 0, 1)
        self.assert_targets(shader, lambda pixel: (0.2, 0, 0, 1))

    def test_short_circuit_and_chained_comparison_do_not_evaluate_guarded_math(self):
        @fragment
        def shader(pixel: FragmentCoord):
            x = pixel.x - 0.5
            positive_log = x > 0.0 and sm.log(x) > 0.0
            reciprocal = x == 0.0 or 1.0 / x > 0.0
            chained = 0.0 < x < 3.0
            return vec4(select(positive_log, 1.0, 0.0), select(reciprocal, 1.0, 0.0),
                        select(chained, 1.0, 0.0), 1)
        self.assert_targets(shader, lambda pixel: (int(pixel > 1), 1, int(0 < pixel < 3), 1), width=4)

    def test_range_zero_trip_negative_step_and_boundary_wrap_are_correct(self):
        for lower, upper, stride, count in (
            (0, 0, 1, 0), (4, -1, -2, 3),
            (2**31 - 2, 2**31 - 1, 2, 1),
            (-2**31 + 1, -2**31, -2, 1),
            (-2**31, 2**31 - 1, 2**31 - 1, 3),
        ):
            with self.subTest(lower=lower, upper=upper, stride=stride):
                @fragment
                def shader(pixel: FragmentCoord):
                    total = 0.0
                    for i in range(lower, upper, stride):
                        total += 0.1
                    return vec4(total, 0, 0, 1)
                self.assert_targets(shader, lambda pixel: (count * 0.1, 0, 0, 1))

    def test_explicit_numeric_conversions_and_zero_base_vector_power(self):
        @fragment
        def shader(pixel: FragmentCoord):
            count = int(pixel.x)
            color = sm.pow(vec3(float(count), 0.25, 1.0), 2.0)
            return vec4(color, 1)
        self.assert_targets(shader, lambda pixel: (pixel * pixel, 0.0625, 1, 1), width=2)

    def test_python_capsule_scene_hits_misses_and_matches_all_targets(self):
        pipeline = runpy.run_path(str(ROOT / "examples/capsule_march.py"))["pipeline"]
        module = pipeline.trace()[1].to_dict()
        self.assertEqual(count_ops(module, "scf.while"), 1)
        images = {}
        with tempfile.TemporaryDirectory() as work:
            for target in Target:
                manifest = compile_pipeline(pipeline, target, Path(work) / target.value, TOOLS)
                image = self.render(manifest, target, 128, 128)
                images[target] = image.rgba
                center = image.rgba[(64 * 128 + 64) * 4:(64 * 128 + 64) * 4 + 4]
                corner = image.rgba[:4]
                self.assertGreater(center[0], corner[0] + 80)
                self.assertGreater(center[1], corner[1] + 80)
                self.assertEqual(center[3], 255)
            for target in (Target.DIRECTX, Target.METAL):
                self.assertLessEqual(max(abs(a - b) for a, b in zip(images[Target.VULKAN], images[target])), 2)
