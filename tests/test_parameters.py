import copy
import json
import runpy
import struct as py_struct
import unittest

from mlir_gpu import (
    FrameParameters, ParameterBlock, Pipeline, StructField, Type,
    Vertex, f32, fragment, fullscreen_vertex, i32, is_struct,
    mat4, mat4x4, pack_parameters, struct, u32, vec2, vec3, vec4, vertex,
)
from mlir_gpu.parameters import verify_parameters
from mlir_gpu.interfaces import reflect_pipeline_parameters
from pathlib import Path
import tempfile
from test_support import ROOT, TOOLS, available
from mlir_gpu.compiler import Target, compile_pipeline
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
        for fields in ((("a", Type.BOOL),), (("a", Type.F32), ("a", Type.VEC2)), (("_private", Type.F32),)):
            with self.subTest(fields=fields), self.assertRaises((ValueError, TypeError)):
                ParameterBlock("invalid", fields)
        # Type.U32, Type.I32, and Type.MAT4X4 are now supported in ParameterBlock
        valid_block = ParameterBlock("valid", (("u", Type.U32), ("i", Type.I32), ("m", Type.MAT4X4)))
        self.assertEqual(len(valid_block.fields), 3)
        @fragment
        def missing(frame: FrameParameters):
            return vec4(frame.missing, 0, 0, 1)
        with self.assertRaisesRegex(AttributeError, r"test_parameters.py:\d+:\d+: Unknown parameter field"):
            Pipeline(fullscreen_vertex, missing).trace()

    def test_integer_and_float_parameter_declarations_and_packing(self):
        compute_params = ParameterBlock(
            "sim_params",
            (
                ("count", Type.U32),
                ("species_offset", Type.I32),
                ("dt", Type.F32),
                ("bounds", Type.VEC2),
            ),
        )
        self.assertEqual(
            compute_params.to_dict(),
            {
                "name": "sim_params",
                "fields": [
                    {"name": "count", "type": "u32"},
                    {"name": "species_offset", "type": "i32"},
                    {"name": "dt", "type": "f32"},
                    {"name": "bounds", "type": "f32x2"},
                ],
            },
        )
        reflected = {
            "schema_version": 2,
            "name": "sim_params",
            "layout": "gfx-std430-v1",
            "size": 24,
            "alignment": 8,
            "stages": ["compute"],
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "fields": [
                {"name": "count", "type": "u32", "offset": 0, "size": 4, "alignment": 4},
                {"name": "species_offset", "type": "i32", "offset": 4, "size": 4, "alignment": 4},
                {"name": "dt", "type": "f32", "offset": 8, "size": 4, "alignment": 4},
                {"name": "bounds", "type": "f32x2", "offset": 16, "size": 8, "alignment": 8},
            ],
            "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 0},
                "directx": {
                    "storage": "root_constants",
                    "parameter_index": 0,
                    "register": 0,
                    "space": 0,
                    "num32bit_constants": 6,
                    "visibility": "all",
                    "root_signature_version": "1.1",
                },
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 24},
            },
        }
        verify_parameters(reflected)
        packed = pack_parameters(
            reflected,
            {"count": 1024, "species_offset": -3, "dt": 0.016, "bounds": (100.0, 200.0)},
        )
        self.assertEqual(packed, py_struct.pack("<Iif4x2f", 1024, -3, 0.016, 100.0, 200.0))


class ParameterPackingTests(unittest.TestCase):
    def test_host_packing_matches_static_valid_reflection(self):
        verify_parameters(FRAME_REFLECTION)
        self.assertEqual(
            pack_parameters(FRAME_REFLECTION, {"time": 1.25, "resolution": (640, 480)}),
            py_struct.pack("<4f", 1.25, 0, 640, 480),
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
            py_struct.pack("<8f", 1, 2, 3, 4, 5, 6, 7, 0),
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


# ---------------------------------------------------------------------------
# Declarative Struct Definition Tests
# ---------------------------------------------------------------------------

@struct
class CameraParams:
    view_proj: mat4x4
    eye_pos: vec3
    time: f32 = 0.0


@struct
class ObjectParams:
    model: mat4x4
    color: vec4


@struct
class LightParams:
    direction: vec3
    intensity: f32
    color: vec3
    ambient: f32


@struct
class AllTypes:
    scalar_f: f32
    scalar_u: u32
    scalar_i: i32
    v2: vec2
    v3: vec3
    v4: vec4
    m4: mat4x4


class StructDeclarationTests(unittest.TestCase):
    def test_struct_decorator_and_typing(self):
        self.assertTrue(is_struct(CameraParams))
        self.assertTrue(is_struct(ObjectParams))
        self.assertTrue(is_struct(LightParams))
        self.assertTrue(is_struct(AllTypes))
        self.assertFalse(is_struct(object))
        self.assertFalse(is_struct(123))

        # Check field descriptor reflection
        fields = {f.name: f for f in CameraParams.__struct_fields__}
        self.assertEqual(len(fields), 3)
        self.assertEqual(fields["view_proj"].type, Type.MAT4X4)
        self.assertEqual(fields["eye_pos"].type, Type.VEC3)
        self.assertEqual(fields["time"].type, Type.F32)

    def test_struct_instantiation_defaults_and_args(self):
        # Default argument applied
        cam = CameraParams(tuple(range(16)), (1.0, 2.0, 3.0))
        self.assertTrue(is_struct(cam))
        self.assertEqual(cam.time, 0.0)
        self.assertEqual(cam.eye_pos, (1.0, 2.0, 3.0))

        # Explicit keyword arguments
        cam2 = CameraParams(view_proj=tuple(range(16)), eye_pos=(0, 0, 0), time=5.5)
        self.assertEqual(cam2.time, 5.5)

        # Repr and Eq
        self.assertEqual(cam, CameraParams(tuple(range(16)), (1.0, 2.0, 3.0), 0.0))
        self.assertIn("CameraParams", repr(cam))

        # Missing required argument
        with self.assertRaises(TypeError):
            CameraParams(tuple(range(16)))

        # Unexpected keyword argument
        with self.assertRaises(TypeError):
            CameraParams(tuple(range(16)), (0, 0, 0), time=0.0, extra=123)

        # Non-default field follows default field
        with self.assertRaises(TypeError):
            @struct
            class BadOrder:
                a: f32 = 1.0
                b: f32

    def test_invalid_struct_declarations_fail(self):
        # Empty struct
        with self.assertRaises(ValueError):
            @struct
            class Empty:
                pass

        # Private field
        with self.assertRaises(ValueError):
            @struct
            class PrivateField:
                _secret: f32

        # Invalid field name
        with self.assertRaises(ValueError):
            bad_cls = type("BadName", (), {"__annotations__": {"1invalid": f32}})
            struct(bad_cls)

        # Unsupported field type
        with self.assertRaises(TypeError):
            @struct
            class UnsupportedType:
                tag: str


class Std430LayoutTests(unittest.TestCase):
    def test_canonical_minigfx_struct_layouts(self):
        # CameraParams (DESIGN.md §9.2: 80B, align 16)
        # view_proj: offset 0, size 64, align 16
        # eye_pos: offset 64, size 12, align 16
        # time: offset 76, size 4, align 4
        self.assertEqual(CameraParams.__struct_size__, 80)
        self.assertEqual(CameraParams.__struct_alignment__, 16)
        cam_fields = {f.name: f for f in CameraParams.__struct_fields__}
        self.assertEqual((cam_fields["view_proj"].offset, cam_fields["view_proj"].size, cam_fields["view_proj"].alignment), (0, 64, 16))
        self.assertEqual((cam_fields["eye_pos"].offset, cam_fields["eye_pos"].size, cam_fields["eye_pos"].alignment), (64, 12, 16))
        self.assertEqual((cam_fields["time"].offset, cam_fields["time"].size, cam_fields["time"].alignment), (76, 4, 4))

        # ObjectParams (DESIGN.md §9.2: 80B, align 16)
        # model: offset 0, size 64, align 16
        # color: offset 64, size 16, align 16
        self.assertEqual(ObjectParams.__struct_size__, 80)
        obj_fields = {f.name: f for f in ObjectParams.__struct_fields__}
        self.assertEqual((obj_fields["model"].offset, obj_fields["model"].size, obj_fields["model"].alignment), (0, 64, 16))
        self.assertEqual((obj_fields["color"].offset, obj_fields["color"].size, obj_fields["color"].alignment), (64, 16, 16))

        # LightParams (DESIGN.md §9.2: 32B, align 16)
        # direction: offset 0, size 12, align 16
        # intensity: offset 12, size 4, align 4 (intra-vec3 packing)
        # color: offset 16, size 12, align 16
        # ambient: offset 28, size 4, align 4
        self.assertEqual(LightParams.__struct_size__, 32)
        light_fields = {f.name: f for f in LightParams.__struct_fields__}
        self.assertEqual((light_fields["direction"].offset, light_fields["direction"].size, light_fields["direction"].alignment), (0, 12, 16))
        self.assertEqual((light_fields["intensity"].offset, light_fields["intensity"].size, light_fields["intensity"].alignment), (12, 4, 4))
        self.assertEqual((light_fields["color"].offset, light_fields["color"].size, light_fields["color"].alignment), (16, 12, 16))
        self.assertEqual((light_fields["ambient"].offset, light_fields["ambient"].size, light_fields["ambient"].alignment), (28, 4, 4))

    def test_std430_alignment_gaps_and_padding(self):
        # f32 followed by vec4: gap 4..15 (12B internal padding), total 32B
        @struct
        class Gap:
            s: f32
            v: vec4

        self.assertEqual(Gap.__struct_size__, 32)
        gap_fields = {f.name: f for f in Gap.__struct_fields__}
        self.assertEqual((gap_fields["s"].offset, gap_fields["s"].size), (0, 4))
        self.assertEqual((gap_fields["v"].offset, gap_fields["v"].size), (16, 16))

        # vec3 followed by vec2: gap 12..15 (4B internal padding), trailing 24..31 (8B trailing padding)
        @struct
        class V3V2:
            v3: vec3
            v2: vec2

        self.assertEqual(V3V2.__struct_size__, 32)
        v_fields = {f.name: f for f in V3V2.__struct_fields__}
        self.assertEqual(v_fields["v3"].offset, 0)
        self.assertEqual(v_fields["v2"].offset, 16)

        # Single scalar: total rounded up to multiple of 16B
        @struct
        class Single:
            val: f32

        self.assertEqual(Single.__struct_size__, 16)


class StructBinaryPackingTests(unittest.TestCase):
    def test_camera_params_to_bytes_exact_match(self):
        mat = tuple(float(i) for i in range(16))
        pos = (10.0, 20.0, 30.0)
        time = 2.5
        cam = CameraParams(mat, pos, time)
        packed = cam.to_bytes()

        self.assertEqual(len(packed), 80)
        expected = py_struct.pack("<16f3ff", *mat, *pos, time)
        self.assertEqual(packed, expected)

        # Roundtrip via from_bytes
        restored = CameraParams.from_bytes(packed)
        self.assertEqual(restored.view_proj, mat)
        self.assertEqual(restored.eye_pos, pos)
        self.assertEqual(restored.time, time)

    def test_light_params_to_bytes_and_padding_zeroed(self):
        light = LightParams((0.0, -1.0, 0.0), 1.5, (1.0, 0.9, 0.8), 0.1)
        packed = light.to_bytes()
        self.assertEqual(len(packed), 32)
        expected = py_struct.pack("<3ff3ff", 0.0, -1.0, 0.0, 1.5, 1.0, 0.9, 0.8, 0.1)
        self.assertEqual(packed, expected)

        # Verify internal gap bytes are 0x00
        @struct
        class GapTest:
            s: f32
            v: vec4
        gap_inst = GapTest(42.0, (1.0, 2.0, 3.0, 4.0))
        gap_bytes = gap_inst.to_bytes()
        self.assertEqual(len(gap_bytes), 32)
        # Bytes 4..15 must be zero
        self.assertEqual(gap_bytes[4:16], b"\x00" * 12)

    def test_to_bytes_value_validation(self):
        # NaN float rejected
        with self.assertRaises(ValueError):
            CameraParams(tuple(range(16)), (1.0, float("nan"), 3.0), 0.0).to_bytes()

        # Inf float rejected
        with self.assertRaises(ValueError):
            CameraParams(tuple(range(16)), (1.0, float("inf"), 3.0), 0.0).to_bytes()

        # Integer range validation
        @struct
        class IntStruct:
            u: u32
            i: i32

        # Negative u32 rejected
        with self.assertRaises(ValueError):
            IntStruct(-1, 0).to_bytes()

        # u32 overflow rejected
        with self.assertRaises(ValueError):
            IntStruct(2**32, 0).to_bytes()

        # i32 overflow rejected
        with self.assertRaises(ValueError):
            IntStruct(0, 2**31).to_bytes()

        # Valid ints roundtrip
        valid = IntStruct(42, -42)
        self.assertEqual(valid.to_bytes(), py_struct.pack("<Ii8x", 42, -42))


class StructShaderTracingTests(unittest.TestCase):
    def test_vertex_shader_with_struct_and_matrix_multiplication(self):
        @vertex
        def vs(vertex_index: u32, cam: CameraParams) -> Vertex:
            world_pos = vec4(1.0, 2.0, 3.0, 1.0)
            clip_pos = cam.view_proj * world_pos
            return Vertex(clip_pos)

        @fragment
        def fs() -> vec4:
            return vec4(1.0, 1.0, 1.0, 1.0)

        pipeline = Pipeline(vs, fs)
        stage = pipeline.trace()[0]
        self.assertEqual(stage.kind, "vertex")
        self.assertEqual(len(stage.input_interface), 1)
        self.assertEqual(stage.input_interface[0].builtin, "vertex_index")

        module = stage.to_dict()
        self.assertIn("parameters", module)
        self.assertEqual(module["parameters"]["name"], "cam")
        self.assertEqual(len(module["parameters"]["fields"]), 3)

        # Check ops: matrix-vector multiplication lowered to vector arithmetic
        ops = structured.operations(module["functions"][0])
        op_names = [op["op"] for op in ops]
        self.assertIn("gfx.parameter", op_names)
        self.assertIn("vector.extract", op_names)
        self.assertIn("vector.broadcast", op_names)
        self.assertIn("arith.mulf", op_names)
        self.assertIn("arith.addf", op_names)

    def test_fragment_shader_struct_swizzle_and_helper(self):
        def helper(c: CameraParams):
            return c.eye_pos.xz

        @fragment
        def fs(cam: CameraParams) -> vec4:
            xz = helper(cam)
            return vec4(xz.x, xz.y, cam.time, 1.0)

        pipeline = Pipeline(fullscreen_vertex, fs)
        stage = pipeline.trace()[1]
        self.assertEqual(stage.kind, "fragment")
        self.assertEqual(stage.input_interface, ())
        module = stage.to_dict()
        self.assertIn("parameters", module)

    def test_shader_unknown_field_access_fails_with_source_location(self):
        @fragment
        def bad_field(cam: CameraParams):
            return vec4(cam.nonexistent, 0, 0, 1)

        with self.assertRaisesRegex(AttributeError, r"test_parameters.py:\d+:\d+: Unknown parameter field 'nonexistent'"):
            Pipeline(fullscreen_vertex, bad_field).trace()

    def test_device_boolean_control_flow_rejected(self):
        # Direct numeric truthiness in shader is rejected
        @fragment
        def bad_truthiness(cam: CameraParams):
            if cam.time:
                return vec4(1.0)
            return vec4(0.0)

        with self.assertRaisesRegex(TypeError, "Device conditions require bool values"):
            Pipeline(fullscreen_vertex, bad_truthiness).trace()

        # bool() evaluation on device values raises TypeError
        from mlir_gpu.ast_frontend import MatrixValue
        with self.assertRaisesRegex(TypeError, "Device values cannot control Python if/loops"):
            bool(vec4(1.0, 2.0, 3.0, 4.0))
        dummy_col = vec4(1.0, 0.0, 0.0, 0.0)
        m = MatrixValue((dummy_col, dummy_col, dummy_col, dummy_col))
        with self.assertRaisesRegex(TypeError, "Device values cannot control Python if/loops"):
            bool(m)


class MatrixTypeTests(unittest.TestCase):
    SHAPES = [(c, r) for c in (2, 3, 4) for r in (2, 3, 4)]

    def test_all_matrix_shapes_exist_with_std430_layout(self):
        for c, r in self.SHAPES:
            with self.subTest(shape=f"{c}x{r}"):
                typ = Type(f"mat{c}x{r}")
                self.assertEqual(typ.matrix_shape, (c, r))
                self.assertEqual(typ.schema_name, f"mat{c}x{r}")
                stride = 8 if r == 2 else 16
                self.assertEqual(typ.matrix_layout, (c * stride, stride, stride))
                self.assertIs(Type.from_schema(typ.schema_name), typ)
        self.assertFalse(Type.VEC4.is_matrix)
        self.assertFalse(hasattr(Type, "MAT4"))

    def test_struct_pack_and_unpack_round_trip_for_every_shape(self):
        for c, r in self.SHAPES:
            with self.subTest(shape=f"{c}x{r}"):
                typ = Type(f"mat{c}x{r}")

                @struct
                class Holder:
                    pad: f32
                    m: typ

                values = tuple(float(i + 1) for i in range(c * r))
                inst = Holder(1.0, values)
                data = inst.to_bytes()
                stride = typ.matrix_layout[2]
                offset = typ.matrix_layout[1]
                for col in range(c):
                    got = py_struct.unpack_from(f"<{r}f", data, offset + col * stride)
                    self.assertEqual(got, values[col * r:(col + 1) * r])
                self.assertEqual(Holder.from_bytes(data).m, values)
                block = Holder.reflection(stages=["vertex"])
                verify_parameters(block)
                self.assertEqual(pack_parameters(block, inst), data)

    def test_matrix_component_count_is_validated(self):
        @struct
        class Holder:
            m: Type.MAT3X2

        with self.assertRaises(ValueError):
            Holder(tuple(range(5))).to_bytes()
        with self.assertRaises(ValueError):
            Holder(tuple(float("inf") for _ in range(6))).to_bytes()

    def test_matrix_value_multiplication_shapes(self):
        from mlir_gpu.ast_frontend import MatrixValue

        def column(size):
            return {Type.VEC2: vec2(0.0, 0.0), Type.VEC3: vec3(0.0, 0.0, 0.0), Type.VEC4: vec4(0.0, 0.0, 0.0, 0.0)}[size]

        m3x2 = MatrixValue(tuple(column(Type.VEC2) for _ in range(3)))
        m2x3 = MatrixValue(tuple(column(Type.VEC3) for _ in range(2)))
        self.assertEqual(m3x2.type, Type.MAT3X2)
        self.assertEqual(m2x3.type, Type.MAT2X3)
        # (3x2) * (2x3) -> 2x2 ; (2x3) * (3x2) -> 3x3
        self.assertEqual((m3x2 * m2x3).type, Type.MAT2X2)
        self.assertEqual((m2x3 * m3x2).type, Type.MAT3X3)
        with self.assertRaises(TypeError):
            m3x2 * m3x2
        self.assertEqual((m3x2 * column(Type.VEC3)).type, Type.VEC2)
        self.assertEqual((column(Type.VEC2) * m3x2).type, Type.VEC3)
        with self.assertRaises(TypeError):
            m3x2 * column(Type.VEC2)


class StructReflectionTests(unittest.TestCase):
    def test_struct_reflection_conforms_to_verify_parameters(self):
        cam_refl = CameraParams.reflection(stages=["vertex"])
        verify_parameters(cam_refl)
        self.assertEqual(cam_refl["size"], 80)
        self.assertEqual(cam_refl["layout"], "gfx-std430-v1")
        self.assertEqual(cam_refl["bindings"]["directx"]["num32bit_constants"], 20)

        light_refl = LightParams.reflection(stages=["fragment"])
        verify_parameters(light_refl)
        self.assertEqual(light_refl["size"], 32)
        self.assertEqual(light_refl["bindings"]["directx"]["num32bit_constants"], 8)

        # pack_parameters with struct reflection and instance
        cam = CameraParams(tuple(range(16)), (1.0, 2.0, 3.0), 1.0)
        packed_from_refl = pack_parameters(cam_refl, cam)
        self.assertEqual(packed_from_refl, cam.to_bytes())


# ===========================================================================
# Per-Draw / Per-Object Constants & Multi-Block Parameter ABI
# ===========================================================================

class TestParameterBlock256ByteLimits(unittest.TestCase):
    def test_parameter_block_exact_256_bytes_accepted(self):
        @struct
        class Block256:
            m0: mat4
            m1: mat4
            m2: mat4
            m3: mat4

        self.assertEqual(Block256.__struct_size__, 256)
        refl = Block256.reflection(stages=["vertex"])
        verify_parameters(refl)
        self.assertEqual(refl["size"], 256)
        self.assertEqual(refl["bindings"]["directx"]["num32bit_constants"], 64)

        inst = Block256(
            tuple(range(16)), tuple(range(16, 32)),
            tuple(range(32, 48)), tuple(range(48, 64))
        )
        raw = inst.to_bytes()
        self.assertEqual(len(raw), 256)
        restored = Block256.from_bytes(raw)
        self.assertEqual(restored, inst)

    def test_parameter_block_64_fields_accepted(self):
        fields = [(f"f_{i}", Type.F32) for i in range(64)]
        block = ParameterBlock("fields64", tuple(fields))
        self.assertEqual(len(block.fields), 64)
        refl_fields = [
            {"name": f"f_{i}", "type": "f32", "offset": i * 4, "size": 4, "alignment": 4}
            for i in range(64)
        ]
        refl = {
            "schema_version": 1,
            "name": "fields64",
            "layout": "gfx-std430-v1",
            "size": 256,
            "alignment": 16,
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "stages": ["vertex"],
            "fields": refl_fields,
            "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 0},
                "directx": {"storage": "root_constants", "parameter_index": 0, "register": 0, "space": 0,
                            "num32bit_constants": 64, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 256},
            },
        }
        verify_parameters(refl)

    def test_parameter_block_65_fields_rejected(self):
        refl_fields = [
            {"name": f"f_{i}", "type": "f32", "offset": i * 4, "size": 4, "alignment": 4}
            for i in range(65)
        ]
        refl = {
            "schema_version": 1,
            "name": "fields65",
            "layout": "gfx-std430-v1",
            "size": 272,
            "alignment": 16,
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "stages": ["vertex"],
            "fields": refl_fields,
            "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 0},
                "directx": {"storage": "root_constants", "parameter_index": 0, "register": 0, "space": 0,
                            "num32bit_constants": 68, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 272},
            },
        }
        with self.assertRaises(ValueError):
            verify_parameters(refl)

    def test_parameter_block_exceeding_256_bytes_rejected(self):
        fields = [
            {"name": f"m{i}", "type": "mat4", "offset": i * 64, "size": 64, "alignment": 16}
            for i in range(5)
        ]
        refl = {
            "schema_version": 1,
            "name": "oversized",
            "layout": "gfx-std430-v1",
            "size": 320,
            "alignment": 16,
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "stages": ["vertex"],
            "fields": fields,
            "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 0},
                "directx": {"storage": "root_constants", "parameter_index": 0, "register": 0, "space": 0,
                            "num32bit_constants": 80, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 320},
            },
        }
        with self.assertRaises(ValueError):
            verify_parameters(refl)


class TestMultiBlockShaderTracing(unittest.TestCase):
    def test_vertex_shader_with_multiple_struct_blocks(self):
        @vertex
        def vs(v: u32, cam: CameraParams, obj: ObjectParams) -> Vertex:
            world_pos = obj.model * vec4(1.0, 2.0, 3.0, 1.0)
            clip_pos = cam.view_proj * world_pos
            return Vertex(clip_pos)

        @fragment
        def fs() -> vec4:
            return vec4(1.0)

        pipeline = Pipeline(vs, fs)
        stages = pipeline.trace()
        v_stage = stages[0]
        params = v_stage.to_dict()["parameters"]
        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 2)
        names = {p["name"] for p in params}
        self.assertEqual(names, {"cam", "obj"})

        ops = structured.operations(v_stage.to_dict()["functions"][0])
        param_ops = [op for op in ops if op["op"] == "gfx.parameter"]
        blocks_accessed = {op["attributes"]["block"]["value"] for op in param_ops}
        self.assertEqual(blocks_accessed, {"cam", "obj"})

    def test_multi_block_stages_and_operations(self):
        @fragment
        def fs_multi(cam: CameraParams, light: LightParams) -> vec4:
            return vec4(cam.time * light.intensity)

        pipeline = Pipeline(fullscreen_vertex, fs_multi)
        f_stage = pipeline.trace()[1]
        params = f_stage.to_dict()["parameters"]
        self.assertIsInstance(params, list)
        self.assertEqual(len(params), 2)
        names = {p["name"] for p in params}
        self.assertEqual(names, {"cam", "light"})


class TestCrossStageReconciliationAndBindings(unittest.TestCase):
    def test_cross_stage_parameter_reconciliation(self):
        cam_vs = CameraParams.reflection(stages=["vertex"], name="cam")
        obj_vs = ObjectParams.reflection(stages=["vertex"], name="obj", binding_offset=80)
        obj_vs["bindings"]["directx"]["parameter_index"] = 1
        obj_vs["bindings"]["directx"]["register"] = 1
        obj_vs["bindings"]["metal"]["buffer_index"] = 3
        obj_vs["bindings"]["metal"]["offset"] = 80
        cam_fs = CameraParams.reflection(stages=["fragment"], name="cam")
        light_fs = LightParams.reflection(stages=["fragment"], name="light", binding_offset=160)
        light_fs["bindings"]["directx"]["parameter_index"] = 2
        light_fs["bindings"]["directx"]["register"] = 2
        light_fs["bindings"]["metal"]["buffer_index"] = 4
        light_fs["bindings"]["metal"]["offset"] = 160
        interface = {
            "vertex": {"parameters": [cam_vs, obj_vs]},
            "fragment": {"parameters": [cam_fs, light_fs]},
        }
        reconciled = reflect_pipeline_parameters(interface)
        self.assertIsInstance(reconciled, list)
        self.assertEqual(len(reconciled), 3)

        # cam (80B)
        cam_info = reconciled[0]
        self.assertEqual(cam_info["name"], "cam")
        self.assertEqual(set(cam_info["stages"]), {"vertex", "fragment"})
        self.assertEqual(cam_info["bindings"]["vulkan"]["offset"], 0)
        self.assertEqual(cam_info["bindings"]["directx"]["register"], 0)
        self.assertEqual(cam_info["bindings"]["metal"]["buffer_index"], 2)

        # obj (80B)
        obj_info = reconciled[1]
        self.assertEqual(obj_info["name"], "obj")
        self.assertEqual(obj_info["stages"], ["vertex"])
        self.assertEqual(obj_info["bindings"]["vulkan"]["offset"], 80)
        self.assertEqual(obj_info["bindings"]["directx"]["register"], 1)
        self.assertEqual(obj_info["bindings"]["metal"]["buffer_index"], 3)

        # light (32B)
        light_info = reconciled[2]
        self.assertEqual(light_info["name"], "light")
        self.assertEqual(light_info["stages"], ["fragment"])
        self.assertEqual(light_info["bindings"]["vulkan"]["offset"], 160)
        self.assertEqual(light_info["bindings"]["directx"]["register"], 2)
        self.assertEqual(light_info["bindings"]["metal"]["buffer_index"], 4)

    def test_cumulative_size_exceeding_256_rejected(self):
        @struct
        class Block160:
            m1: mat4
            m2: mat4
            v1: vec4
            v2: vec4

        @struct
        class Block112:
            m1: mat4
            v1: vec4
            v2: vec4
            v3: vec4

        interface = {
            "vertex": {"parameters": Block160.reflection(stages=["vertex"])},
            "fragment": {"parameters": Block112.reflection(stages=["fragment"])},
        }
        # 160 + 112 = 272B > 256B
        with self.assertRaisesRegex(ValueError, "exceeds 256 bytes"):
            reflect_pipeline_parameters(interface)


class TestMultiBlockPacking(unittest.TestCase):
    def test_multi_block_contiguous_packing(self):
        cam = CameraParams(tuple(range(16)), (1.0, 2.0, 3.0), 10.0)
        obj = ObjectParams(tuple(range(16, 32)), (0.5, 0.5, 0.5, 1.0))
        light = LightParams((0.0, 1.0, 0.0), 2.5, (1.0, 1.0, 1.0), 0.2)

        reconciled = [
            CameraParams.reflection(stages=["vertex", "fragment"], name="cam"),
            {**ObjectParams.reflection(stages=["vertex"], name="obj"), "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 80},
                "directx": {"storage": "root_constants", "parameter_index": 1, "register": 1, "space": 0,
                            "num32bit_constants": 20, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 3, "offset": 80, "size": 80},
            }},
            {**LightParams.reflection(stages=["fragment"], name="light"), "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 160},
                "directx": {"storage": "root_constants", "parameter_index": 2, "register": 2, "space": 0,
                            "num32bit_constants": 8, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 4, "offset": 160, "size": 32},
            }},
        ]

        packed = pack_parameters(reconciled, {
            "cam": cam,
            "obj": obj,
            "light": light,
        })
        self.assertEqual(len(packed), 192)
        self.assertEqual(packed[0:80], cam.to_bytes())
        self.assertEqual(packed[80:160], obj.to_bytes())
        self.assertEqual(packed[160:192], light.to_bytes())

    def test_multi_block_packing_validation(self):
        reconciled = [
            CameraParams.reflection(stages=["vertex"]),
            {**ObjectParams.reflection(stages=["vertex"]), "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 80},
                "directx": {"storage": "root_constants", "parameter_index": 1, "register": 1, "space": 0,
                            "num32bit_constants": 20, "visibility": "all", "root_signature_version": "1.1"},
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 3, "offset": 80, "size": 80},
            }},
        ]
        # Missing block 'obj'
        with self.assertRaises(ValueError):
            pack_parameters(reconciled, {"cam": CameraParams(tuple(range(16)), (0, 0, 0), 0.0)})
        # Extra unknown block
        with self.assertRaises(ValueError):
            pack_parameters(reconciled, {
                "cam": CameraParams(tuple(range(16)), (0, 0, 0), 0.0),
                "obj": ObjectParams(tuple(range(16)), (0, 0, 0, 0)),
                "unknown": 42,
            })
        # Non-mapping
        with self.assertRaises(ValueError):
            pack_parameters(reconciled, [1, 2, 3])


# ---------------------------------------------------------------------------
# End-to-End Multi-Block Pipeline Compilation Tests
# ---------------------------------------------------------------------------

@struct
class E2ECameraParams:
    view_proj: mat4
    eye_pos: vec3
    time: f32 = 0.0

@struct
class E2EObjectParams:
    model: mat4
    color: vec4
    draw_id: u32 = 0

@struct
class E2ELightParams:
    direction: vec3
    intensity: f32
    color: vec3
    ambient: f32

@vertex
def multi_block_e2e_vs(vertex_index: u32, cam: E2ECameraParams, obj: E2EObjectParams) -> Vertex:
    world_pos = obj.model * vec4(1.0, 2.0, 3.0, 1.0)
    clip_pos = cam.view_proj * world_pos
    return Vertex(clip_pos, color=obj.color)

@fragment
def multi_block_e2e_fs(color: vec4, cam: E2ECameraParams, light: E2ELightParams) -> vec4:
    light_term = light.color * (light.intensity + light.ambient)
    animated_tint = color + vec4(light_term, cam.time * 0.0)
    return animated_tint


@unittest.skipUnless(available("gfx-compile"), "gfx-compile required for e2e compilation tests")
class TestCompilePipelineMultiBlock(unittest.TestCase):
    def test_e2e_compile_multi_block_vulkan(self):
        pipeline = Pipeline(multi_block_e2e_vs, multi_block_e2e_fs)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, out_dir / "out", TOOLS)
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(manifest["target"], "vulkan")
            self.assertEqual(manifest["schema_version"], 4)
            self.assertIn("parameters", manifest)
            params = manifest["parameters"]
            self.assertIsInstance(params, list)
            self.assertEqual(len(params), 3)

            # Reconciled blocks: cam (shared), obj (vs), light (fs)
            param_names = [p["name"] for p in params]
            self.assertEqual(param_names, ["cam", "obj", "light"])

            # Verify cumulative sequential offsets
            self.assertEqual(params[0]["bindings"]["vulkan"]["offset"], 0)
            self.assertEqual(params[1]["bindings"]["vulkan"]["offset"], 80)
            self.assertEqual(params[2]["bindings"]["vulkan"]["offset"], 176)

            # Check stages reconciliation
            self.assertEqual(sorted(params[0]["stages"]), ["fragment", "vertex"])
            self.assertEqual(params[1]["stages"], ["vertex"])
            self.assertEqual(params[2]["stages"], ["fragment"])

            # Verify SPV binary file exists on disk
            spv_file = out_dir / "out" / "vertex.spv"
            self.assertTrue(spv_file.exists())
            self.assertGreater(spv_file.stat().st_size, 0)

            # Verify fragment SPV has push constants matching pipeline layout (offset 176 for light)
            import subprocess, sys
            spv_fs = out_dir / "out" / "fragment.spv"
            self.assertTrue(spv_fs.exists())
            summarize_script = ROOT / "tests" / "lit" / "parameters" / "Inputs" / "summarize.py"
            res = subprocess.run(
                [sys.executable, str(summarize_script), "spirv", str(spv_fs)],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertIn("176", res.stdout)

            # Verify contiguous parameter byte packing
            cam_val = E2ECameraParams(tuple(range(16)), (1.0, 2.0, 3.0), 0.5)
            obj_val = E2EObjectParams(tuple(range(16, 32)), (0.1, 0.2, 0.3, 1.0), 42)
            light_val = E2ELightParams((0.0, 1.0, 0.0), 1.0, (1.0, 1.0, 1.0), 0.2)

            packed = pack_parameters(params, {
                "cam": cam_val,
                "obj": obj_val,
                "light": light_val,
            })
            self.assertEqual(len(packed), 208)
            self.assertEqual(packed[0:80], cam_val.to_bytes())
            self.assertEqual(packed[80:176], obj_val.to_bytes())
            self.assertEqual(packed[176:208], light_val.to_bytes())

    def test_e2e_compile_multi_block_directx(self):
        if not available("llc", "dxv"):
            self.skipTest("DirectX compilation tools not available")
        pipeline = Pipeline(multi_block_e2e_vs, multi_block_e2e_fs)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td)
            manifest_path = compile_pipeline(pipeline, Target.DIRECTX, out_dir / "out", TOOLS)
            manifest = json.loads(manifest_path.read_text())

            self.assertEqual(manifest["target"], "directx")
            params = manifest["parameters"]
            self.assertEqual(len(params), 3)

            # Verify separate registers assigned (b0, b1, b2)
            self.assertEqual(params[0]["bindings"]["directx"]["register"], 0)
            self.assertEqual(params[0]["bindings"]["directx"]["parameter_index"], 0)
            self.assertEqual(params[1]["bindings"]["directx"]["register"], 1)
            self.assertEqual(params[1]["bindings"]["directx"]["parameter_index"], 1)
            self.assertEqual(params[2]["bindings"]["directx"]["register"], 2)
            self.assertEqual(params[2]["bindings"]["directx"]["parameter_index"], 2)

            # Verify emitted LLVM IR contains separate handlefrombinding and rootsignatures
            ll_file = out_dir / "out" / "vertex.ll"
            self.assertTrue(ll_file.exists())
            content = ll_file.read_text()
            self.assertIn('handlefrombinding', content)
            self.assertIn('!dx.rootsignatures', content)

            ll_fs = (out_dir / "out" / "fragment.ll").read_text()
            import re
            self.assertRegex(ll_fs, r"@llvm\.dx\.resource\.handlefrombinding[^(]*\(i32 0, i32 2,")
            self.assertIn('!{!"RootConstants", i32 0, i32 2,', ll_fs)

    def test_cross_stage_float_parameter_block_reconciliation(self):
        @struct
        class CrossVsOnlyParams:
            pos: vec4

        @struct
        class CrossLightParams:
            direction: vec3
            intensity: f32
            color: vec3
            ambient: f32

        @vertex
        def cross_vs(idx: u32, vs_only: CrossVsOnlyParams, light: CrossLightParams) -> Vertex:
            return Vertex(vs_only.pos + vec4(light.direction, light.intensity))

        @fragment
        def cross_fs(light: CrossLightParams) -> vec4:
            return vec4(light.color, light.ambient)

        pipeline = Pipeline(cross_vs, cross_fs)
        for target in (Target.VULKAN, Target.DIRECTX):
            if target == Target.DIRECTX and not available("llc", "dxv"):
                continue
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td) / "out"
                manifest_path = compile_pipeline(pipeline, target, out_dir, TOOLS)
                manifest = json.loads(manifest_path.read_text())
                params = manifest["parameters"]
                self.assertEqual(len(params), 2)
                self.assertEqual(params[0]["name"], "vs_only")
                self.assertEqual(params[1]["name"], "light")
                self.assertEqual(params[1]["layout"], "gfx-std430-v1")
                self.assertEqual(params[1]["schema_version"], 2)
                if target == Target.VULKAN:
                    self.assertEqual(params[0]["bindings"]["vulkan"]["offset"], 0)
                    self.assertEqual(params[1]["bindings"]["vulkan"]["offset"], 16)
                elif target == Target.DIRECTX:
                    self.assertEqual(params[0]["bindings"]["directx"]["register"], 0)
                    self.assertEqual(params[1]["bindings"]["directx"]["register"], 1)

    def test_compile_pipeline_rejects_cumulative_size_exceeding_256(self):
        @struct
        class OversizedBlockA:
            v0: vec4; v1: vec4; v2: vec4; v3: vec4; v4: vec4
            v5: vec4; v6: vec4; v7: vec4; v8: vec4; v9: vec4  # 160B

        @struct
        class OversizedBlockB:
            v0: vec4; v1: vec4; v2: vec4; v3: vec4; v4: vec4
            v5: vec4; v6: vec4                                # 112B (total = 272B > 256B)

        @vertex
        def vs_bad(idx: u32, a: OversizedBlockA) -> Vertex:
            return Vertex(a.v0)

        @fragment
        def fs_bad(b: OversizedBlockB) -> vec4:
            return b.v0

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "exceeds 256 bytes"):
                compile_pipeline(Pipeline(vs_bad, fs_bad), Target.VULKAN, Path(td) / "bad", TOOLS)



