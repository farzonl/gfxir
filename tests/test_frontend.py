import math
import runpy
import unittest
from pathlib import Path

from mlir_gpu import Draw, Pipeline, Value, Vertex, fragment, select, vec2, vec3, vec4, vertex
from mlir_gpu.frontend import Node, Type


ROOT = Path(__file__).resolve().parents[1]


def evaluate(node: Node, argument):
    if node.op == "input":
        return argument
    if node.op == "constant":
        return node.literal
    args = [evaluate(arg, argument) for arg in node.args]
    if node.op in ("vec2", "vec3", "vec4"):
        return tuple(args)
    if node.op == "extract":
        return args[0][node.literal]
    if node.op == "select":
        return args[1] if args[0] else args[2]
    if node.op == "eq":
        return args[0] == args[1]
    operations = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b}
    operation = operations[node.op]
    if node.type in (Type.VEC2, Type.VEC3, Type.VEC4):
        return tuple(operation(a, b) for a, b in zip(*args))
    result = operation(*args)
    return result & 0xFFFFFFFF if node.type == Type.U32 else result


class FrontendTests(unittest.TestCase):
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

    def test_python_control_flow_rejected(self):
        @vertex
        def entry(index):
            if index == 0:
                return Vertex(vec4(0, 0, 0, 1), vec4(1, 1, 1, 1))
            raise AssertionError("unreachable")

        with self.assertRaisesRegex(TypeError, "cannot control Python"):
            entry.trace()

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
