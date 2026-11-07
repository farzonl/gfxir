import math
import runpy
import unittest
from pathlib import Path

from mlir_gpu import Draw, Pipeline, Value, Vertex, fragment, i32, maximum, minimum, select, shader_math as sm, u32, vec2, vec3, vec4, vertex
from mlir_gpu.frontend import Node, Type


ROOT = Path(__file__).resolve().parents[1]


def evaluate(node: Node, argument):
    if node.op == "input":
        return argument[node.literal] if isinstance(argument, dict) else argument
    if node.op == "constant":
        return node.literal
    args = [evaluate(arg, argument) for arg in node.args]
    if node.op in ("vec2", "vec3", "vec4"):
        return tuple(args)
    if node.op == "broadcast":
        return (args[0],) * {Type.VEC2: 2, Type.VEC3: 3, Type.VEC4: 4}[node.type]
    if node.op == "extract":
        return args[0][node.literal]
    if node.op == "select":
        return args[1] if args[0] else args[2]
    if node.op == "eq":
        return args[0] == args[1]
    if node.op == "cmp":
        return {
            "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
            "gt": lambda a, b: a > b, "ge": lambda a, b: a >= b,
            "ne": lambda a, b: a != b,
        }[node.literal[-2:]](*args)
    def numeric_extreme(a, b, operation):
        return b if math.isnan(a) else a if math.isnan(b) else operation(a, b)
    operations = {
        "add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
        "div": lambda a, b: a / b, "neg": lambda a: -a, "abs": abs, "cast": lambda a: a,
        "min": lambda a, b: numeric_extreme(a, b, min),
        "max": lambda a, b: numeric_extreme(a, b, max),
        "and": lambda a, b: a & b, "or": lambda a, b: a | b, "xor": lambda a, b: a ^ b,
        **{name: getattr(math, name) for name in ("sin", "cos", "sqrt", "exp", "exp2", "log", "log2", "floor", "pow")},
    }
    operation = operations[node.op]
    if node.type in (Type.VEC2, Type.VEC3, Type.VEC4):
        return tuple(operation(*components) for components in zip(*args))
    result = operation(*args)
    if node.type in (Type.I32, Type.U32):
        result &= 0xFFFFFFFF
        if node.type == Type.I32 and result >= 2**31:
            result -= 2**32
    return result


class FrontendTests(unittest.TestCase):
    def test_shader_math_builds_typed_nodes_and_matches_host_references(self):
        for name, values in (
            ("sin", (-12.0, -0.25, 0.0, 0.25, 12.0)),
            ("cos", (-12.0, -0.25, 0.0, 0.25, 12.0)),
            ("sqrt", (0.0, 0.001, 0.25, 1.0, 64.0)),
            ("exp", (-12.0, -1.0, 0.0, 1.0, 6.0)),
            ("exp2", (-12.0, -1.0, 0.0, 1.0, 6.0)),
            ("log", (0.001, 0.25, 1.0, 64.0)),
            ("log2", (0.001, 0.25, 1.0, 64.0)),
            ("floor", (-2.75, -0.25, 0.0, 0.25, 2.75)),
        ):
            operation = getattr(sm, name)
            for value in values:
                result = operation(value)
                self.assertEqual(result.node.op, name)
                self.assertEqual(result.node.type, Type.F32)
                expected = getattr(math, name)(result.node.args[0].literal)
                self.assertAlmostEqual(evaluate(result.node, None), expected)
            for constructor in (vec2, vec3, vec4):
                vector = constructor(0.25)
                self.assertEqual(operation(vector).node.type, vector.node.type)
            for invalid in (i32(1), u32(1), True, math.inf, math.nan):
                with self.assertRaises((TypeError, ValueError)):
                    operation(invalid)
        self.assertEqual(evaluate(sm.pow(vec3(0.0, 0.25, 1.0), 1.5).node, None), (0.0, 0.125, 1.0))
        for operation in (
            lambda: sm.sqrt(-1), lambda: sm.sqrt(vec3(-1, 0, 1)),
            lambda: sm.log(0), lambda: sm.log2(-1),
            lambda: sm.pow(-1, 2), lambda: sm.pow(0, 0),
            lambda: sm.pow(vec2(0, 1), vec2(-1, 2)),
        ):
            with self.assertRaises(ValueError):
                operation()
        with self.assertRaises(TypeError):
            sm.pow(vec2(1), vec3(1))

    def test_math_example_uses_device_math_and_finite_output(self):
        _, fs = runpy.run_path(str(ROOT / "examples/math.py"))["pipeline"].trace()
        for uv in ((0.5, 0.5), (0.125, 0.75), (0.9, 0.1)):
            output = evaluate(fs.outputs[0], uv)
            self.assertTrue(all(math.isfinite(channel) and 0 <= channel <= 1 for channel in output))
            self.assertEqual(output[3], 1)

    def test_arithmetic_example_exposes_the_new_capabilities(self):
        vs, fs = runpy.run_path(str(ROOT / "examples/arithmetic.py"))["pipeline"].trace()
        for index, expected in enumerate(((-1, -1, 0, 1), (3, -1, 0, 1), (-1, 3, 0, 1))):
            self.assertEqual(evaluate(vs.outputs[0], index), expected)
            self.assertEqual(evaluate(vs.outputs[2], index), 1)
        self.assertEqual(evaluate(fs.outputs[0], {"uv": (0.5, 0.5), "valid": 1.0}), (0.5, 0.5, 0, 1))
        outline = evaluate(fs.outputs[0], {"uv": (0.725, 0.5), "valid": 1.0})
        self.assertEqual(outline[:2], (1, 1))
        self.assertAlmostEqual(outline[2], 0.825, places=6)

    def test_float_arithmetic_broadcasts_without_changing_selection_rules(self):
        for constructor, size in ((vec2, 2), (vec3, 3), (vec4, 4)):
            value = constructor(*(float(index + 1) for index in range(size)))
            for result, expected in (
                (value + 2, tuple(index + 3 for index in range(size))),
                (2 - value, tuple(1 - index for index in range(size))),
                (value * 2, tuple(2 * (index + 1) for index in range(size))),
                (value / 2, tuple((index + 1) / 2 for index in range(size))),
                (2 / value, tuple(2 / (index + 1) for index in range(size))),
                (-value, tuple(-index - 1 for index in range(size))),
                (abs(-value), tuple(index + 1 for index in range(size))),
            ):
                self.assertEqual(evaluate(result.node, None), expected)
        for expression in (
            lambda: u32(1) / 2, lambda: i32(1) / 2,
            lambda: vec2(1) + u32(2), lambda: vec2(1) + vec3(1),
            lambda: select(u32(0) == 0, 1.0, vec4(1)),
            lambda: -u32(1), lambda: - (u32(0) == 0),
        ):
            with self.assertRaises(TypeError):
                expression()

    def test_signed_and_unsigned_integer_semantics_are_explicit(self):
        self.assertEqual(evaluate(i32(u32(2**31)).node, None), -2**31)
        self.assertEqual(evaluate(u32(i32(-1)).node, None), 2**32 - 1)
        self.assertEqual(evaluate((i32(2**31 - 1) + 1).node, None), -2**31)
        self.assertEqual(evaluate((-i32(-2**31)).node, None), -2**31)
        self.assertEqual(evaluate(abs(i32(-2**31)).node, None), -2**31)
        self.assertEqual(evaluate((u32(2**32 - 1) + 1).node, None), 0)
        for value in (-2**31 - 1, 2**31):
            with self.assertRaises(ValueError):
                i32(value)
        for value in (True, 1.0, vec2(1)):
            with self.assertRaises(TypeError):
                i32(value)
        self.assertEqual(i32(0).node.type, Type.I32)
        self.assertEqual(Type.from_schema("i32"), Type.I32)

    def test_scalar_comparisons_and_boolean_composition(self):
        for value, argument in (
            (Value(Node("input", Type.F32)), -1.0),
            (Value(Node("input", Type.I32)), -1),
            (Value(Node("input", Type.U32)), 1),
        ):
            expected = argument < 0
            if value.node.type == Type.U32:
                argument, expected = 2**31, False
            predicates = (value < 0, value <= 0, value > 0, value >= 0, value != 0)
            for predicate, result in zip(predicates, (
                expected, argument <= 0, argument > 0, argument >= 0, argument != 0,
            )):
                self.assertEqual(evaluate(predicate.node, argument), result)
            self.assertTrue(evaluate(((value != 0) & ~(value == 0)).node, argument))
            self.assertTrue(evaluate((False | (value != 0)).node, argument))
            self.assertFalse(evaluate((True ^ (value != 0)).node, argument))
        scalar = Value(Node("input", Type.F32))
        self.assertTrue(evaluate((scalar != 0).node, math.nan))
        self.assertFalse(evaluate((scalar < 0).node, math.nan))
        self.assertEqual((i32(-1) < 0).node.literal, "slt")
        self.assertEqual((u32(1) < 2).node.literal, "ult")
        for expression in (
            lambda: vec2(1) < vec2(2), lambda: i32(0) < u32(0),
            lambda: (u32(0) == 0) < True, lambda: u32(1) & True,
            lambda: ~(vec2(1)), lambda: (u32(0) == 0) | 1,
        ):
            with self.assertRaises(TypeError):
                expression()

    def test_vector_splats_composition_swizzles_and_immutable_updates(self):
        self.assertEqual(evaluate(vec3(2).node, None), (2, 2, 2))
        self.assertEqual(evaluate(vec3(vec2(1, 2), 3).node, None), (1, 2, 3))
        self.assertEqual(evaluate(vec4(vec2(1, 2), vec2(3, 4)).node, None), (1, 2, 3, 4))
        value = vec4(x=1, y=2, z=3, w=4)
        self.assertEqual(evaluate(value.yx.node, None), (2, 1))
        self.assertEqual(evaluate(value.rrr.node, None), (1, 1, 1))
        self.assertEqual(evaluate(value.with_components(xy=vec2(4, 5), w=6).node, None), (4, 5, 3, 6))
        self.assertEqual(evaluate(value.with_components(yz=9).node, None), (1, 9, 9, 4))
        self.assertEqual(evaluate(value.node, None), (1, 2, 3, 4))
        for name in ("xr", "xxxxx", "z"):
            with self.assertRaises(AttributeError):
                getattr(vec2(1), name)
        with self.assertRaises(AttributeError):
            value.x = 10
        for updates in ({"xx": 0}, {"x": 0, "r": 1}):
            with self.assertRaises(ValueError):
                value.with_components(**updates)
        for expression in (
            lambda: value.with_components(xy=vec3(1)),
            lambda: vec3(1, z=2), lambda: vec4(1, w=2),
            lambda: vec2(vec3(1)), lambda: vec3(1, 2),
        ):
            with self.assertRaises(TypeError):
                expression()

    def test_numeric_extrema_and_absolute_value_contract(self):
        scalar = Value(Node("input", Type.F32))
        self.assertEqual(evaluate(minimum(scalar, 0.5).node, math.nan), 0.5)
        self.assertEqual(evaluate(maximum(0.5, scalar).node, math.nan), 0.5)
        self.assertTrue(math.isnan(evaluate(minimum(scalar, scalar).node, math.nan)))
        self.assertEqual(evaluate(minimum(vec3(-2, 0, 3), 1).node, None), (-2, 0, 1))
        self.assertEqual(evaluate(maximum(i32(-3), -2).node, None), -2)
        self.assertEqual(evaluate(minimum(u32(2**31), 0).node, None), 0)
        self.assertEqual(math.copysign(1, evaluate(abs(scalar).node, -0.0)), 1)

    def test_graphics_interfaces_and_draw_configuration_are_independent(self):
        vs, fs = runpy.run_path(str(ROOT / "examples/quad.py"))["pipeline"].trace()
        self.assertEqual(vs.output_interface[1].name, "uv")
        self.assertEqual(vs.output_interface[1].type, Type.VEC2)
        self.assertEqual(fs.input_interface[0], vs.output_interface[1])
        self.assertEqual(evaluate(fs.outputs[0], (0.25, 0.75)), (0.25, 0.75, 0.1875, 1.0))
        for count in (0, 3, 6, 12):
            self.assertEqual(Draw(vertex_count=count).to_dict()["vertex_count"], count)
        for count in (True, -3, 1, 4, 2**32):
            with self.assertRaises(ValueError):
                Draw(vertex_count=count)

    def test_vector_construction_extraction_and_bounds(self):
        self.assertEqual(evaluate(vec2(1, 2)[1].node, None), 2)
        self.assertEqual(evaluate(vec3(1, 2, 3)[2].node, None), 3)
        for value, index in ((vec2(0, 0), 2), (vec3(0, 0, 0), -1), (vec4(0, 0, 0, 0), True)):
            with self.assertRaises(TypeError):
                value[index]

    def test_fragment_can_have_no_varying_inputs(self):
        @vertex
        def position_only(index):
            return Vertex(vec4(0, 0, 0, 1))

        @fragment
        def solid():
            return vec4(0.2, 0.4, 0.6, 1)

        vs, fs = Pipeline(position_only, solid).trace()
        self.assertEqual(len(vs.output_interface), 1)
        self.assertEqual(fs.inputs, ())

    def test_unknown_fragment_varyings_are_rejected_before_compilation(self):
        @vertex
        def entry(index):
            return Vertex(vec4(0, 0, 0, 1), varyings={"uv": vec2(0, 0), "weight": vec3(1, 1, 1)})

        @fragment
        def consumer(missing):
            return vec4(0, 0, 0, 1)

        with self.assertRaisesRegex(ValueError, "no matching vertex varying"):
            Pipeline(entry, consumer).trace()

    def test_triangle_coordinates_colors_and_fragment_passthrough(self):
        vs, fs = runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"].trace()
        positions = [(-0.6, -0.6), (0.6, -0.6), (0.0, 0.6)]
        colors = [(1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 0.0, 1.0), (0.0, 0.0, 1.0, 1.0)]
        for index, (position, color) in enumerate(zip(positions, colors)):
            actual = evaluate(vs.outputs[0], index)
            self.assertAlmostEqual(actual[0], position[0])
            self.assertAlmostEqual(actual[1], position[1])
            self.assertEqual(actual[2:], (0.0, 1.0))
            self.assertEqual(evaluate(vs.outputs[1], index), color)
            self.assertEqual(evaluate(fs.outputs[0], color), color)

    def test_host_helper_functions_and_numeric_coercion(self):
        def twice(value):
            return value * 2

        @vertex
        def entry(index):
            value = select(twice(index) == 2, 1.0, 0.0)
            return Vertex(vec4(value, 0, 0, 1), vec4(1, 0, 0, 1))

        self.assertEqual(evaluate(entry.trace().outputs[0], 1), (1.0, 0.0, 0.0, 1.0))

    def test_python_control_flow_uses_default_ast_lowering(self):
        @vertex
        def entry(index):
            if index == 0:
                return Vertex(vec4(0, 0, 0, 1), vec4(1, 1, 1, 1))
            return Vertex(vec4(1, 0, 0, 1), vec4(0, 0, 0, 1))

        source = entry.trace().to_dict()
        self.assertEqual(source["schema_version"], 2)
        self.assertIn("scf.if", str(source))

    def test_invalid_floats_rejected(self):
        for value in (math.inf, -math.inf, math.nan, 1e100, 10**1000):
            with self.subTest(value=str(value)[:20]):
                with self.assertRaises(ValueError):
                    vec4(value, 0, 0, 1)

    def test_invalid_unsigned_literals_rejected(self):
        index = Value(Node("input", Type.U32, literal="vertex_index"))
        for value in (-1, 2**32):
            with self.assertRaisesRegex(ValueError, "32 bits"):
                select(index == value, 1.0, 0.0)

    def test_mixed_types_and_boolean_arithmetic_rejected(self):
        index = Value(Node("input", Type.U32, literal="vertex_index"))
        with self.assertRaises(TypeError):
            index + 1.0
        with self.assertRaises(TypeError):
            (index == 0) + True
        with self.assertRaises(TypeError):
            vec4(index, 0, 0, 1)
        with self.assertRaises(TypeError):
            select(index, 1.0, 0.0)
        with self.assertRaises(TypeError):
            select(index == 0, 1.0, vec4(0, 0, 0, 1))

    def test_invalid_signatures_and_results_rejected(self):
        @vertex
        def no_argument():
            return None

        @vertex
        def wrong_result(index):
            return vec4(0, 0, 0, 1)

        @fragment
        def wrong_color(color):
            return 1.0

        for entry in (no_argument, wrong_result, wrong_color):
            with self.assertRaises(TypeError):
                entry.trace()
        with self.assertRaises(TypeError):
            Pipeline(wrong_color, wrong_result).trace()

    def test_device_inputs_cannot_leak_between_stages(self):
        foreign = Value(Node("input", Type.F32, literal="foreign"))

        @vertex
        def entry(index):
            return Vertex(vec4(foreign, 0, 0, 1), vec4(1, 1, 1, 1))

        with self.assertRaisesRegex(ValueError, "another stage"):
            entry.trace()

    def test_invalid_entry_name_rejected(self):
        def entry(index):
            return Vertex(vec4(0, 0, 0, 1), vec4(1, 1, 1, 1))

        entry.__name__ = "not-a-symbol"
        with self.assertRaisesRegex(ValueError, "entry-point name"):
            vertex(entry).trace()
