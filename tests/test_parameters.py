import copy
import runpy
import struct
import unittest

from mlir_gpu import (
    FrameParameters, ParameterBlock, Pipeline, Type,
    fragment, fullscreen_vertex, pack_parameters, vec4,
)
from mlir_gpu.parameters import verify_parameters
from test_support import ROOT
import compiler_inputs as structured


def example():
    return runpy.run_path(str(ROOT / "examples/frame_bands.py"))["pipeline"]


FRAME_REFLECTION = {
    "schema_version": 1, "name": "frame", "layout": "gfx-root-f32-v1",
    "size": 16, "alignment": 16, "stages": ["fragment"],
    "ownership": "host", "access": "read-only", "addressable": False,
    "fields": [
        {"name": "time", "type": "f32", "offset": 0, "size": 4, "alignment": 4},
        {"name": "resolution", "type": "f32x2", "offset": 8, "size": 8, "alignment": 8},
    ],
    "bindings": {
        "vulkan": {"storage": "push_constant", "offset": 0},
        "directx": {"storage": "root_constants", "parameter_index": 0, "register": 0, "space": 0,
                    "num32bit_constants": 4, "visibility": "all", "root_signature_version": "1.1"},
        "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 16},
    },
}


class ParameterAuthoringTests(unittest.TestCase):
    def test_parameters_are_distinct_from_raster_inputs_and_inside_helpers(self):
        pipeline = example()
        stage = pipeline.trace()[1]
        self.assertEqual(len(stage.input_interface), 1)
        self.assertEqual(stage.input_interface[0].builtin, "fragment_coord")
        module = stage.to_dict()
        self.assertEqual(module["parameters"], FrameParameters.to_dict())
        self.assertEqual(sum(op["op"] == "gfx.parameter" for op in structured.operations(module["functions"][0])), 2)

        def color(frame):
            return vec4(frame.time, frame.resolution.x, frame.resolution.y, 1)
        @fragment
        def helper(frame: FrameParameters):
            return color(frame)
        self.assertEqual(Pipeline(fullscreen_vertex, helper).trace()[1].input_interface, ())

    def test_unsupported_parameter_declarations_and_field_access_fail(self):
        for fields in ((("a", Type.U32),), (("a", Type.F32), ("a", Type.VEC2)), (("_private", Type.F32),)):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                ParameterBlock("invalid", fields)
        @fragment
        def missing(frame: FrameParameters):
            return vec4(frame.missing, 0, 0, 1)
        with self.assertRaisesRegex(AttributeError, r"test_parameters.py:\d+:\d+: Unknown parameter field"):
            Pipeline(fullscreen_vertex, missing).trace()


class ParameterPackingTests(unittest.TestCase):
    def test_host_packing_matches_static_valid_reflection(self):
        verify_parameters(FRAME_REFLECTION)
        self.assertEqual(
            pack_parameters(FRAME_REFLECTION, {"time": 1.25, "resolution": (640, 480)}),
            struct.pack("<4f", 1.25, 0, 640, 480),
        )
        material = copy.deepcopy(FRAME_REFLECTION)
        material.update(name="material", size=32, fields=[
            {"name": "normal", "type": "f32x3", "offset": 0, "size": 12, "alignment": 16},
            {"name": "roughness", "type": "f32", "offset": 12, "size": 4, "alignment": 4},
            {"name": "pair", "type": "f32x2", "offset": 16, "size": 8, "alignment": 8},
            {"name": "weight", "type": "f32", "offset": 24, "size": 4, "alignment": 4},
        ])
        material["bindings"]["directx"]["num32bit_constants"] = 8
        material["bindings"]["metal"]["size"] = 32
        verify_parameters(material)
        self.assertEqual(
            pack_parameters(material, {"normal": (1, 2, 3), "roughness": 4, "pair": (5, 6), "weight": 7}),
            struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 0),
        )

    def test_host_packing_rejects_malformed_layouts_bindings_and_values(self):
        block = FRAME_REFLECTION
        for case in ("version", "offset", "alignment", "size", "type", "binding", "bool_binding", "stages"):
            with self.subTest(case=case):
                invalid = copy.deepcopy(block)
                if case == "version":
                    invalid["schema_version"] = 2
                elif case in ("offset", "alignment", "type"):
                    invalid["fields"][1][case] = "u32" if case == "type" else 4
                elif case == "size":
                    invalid["size"] = 32
                elif case == "stages":
                    invalid["stages"] = [{}]
                else:
                    invalid["bindings"]["metal"]["buffer_index"] = True if case == "bool_binding" else 3
                with self.assertRaises(ValueError):
                    pack_parameters(invalid, {"time": 1, "resolution": (640, 480)})
        for values in ({}, {"time": True, "resolution": (640, 480)}, {"time": float("nan"), "resolution": (640, 480)},
                       {"time": 10**400, "resolution": (640, 480)}, {"time": 0, "resolution": (640,)},
                       {"time": 0, "resolution": (640, 480), "extra": 1}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                pack_parameters(block, values)
