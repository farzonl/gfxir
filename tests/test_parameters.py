import copy
import runpy
import struct as py_struct
import unittest

from mlir_gpu import (
    FrameParameters, ParameterBlock, Pipeline, StructField, Type,
    Vertex, f32, fragment, fullscreen_vertex, i32, is_struct,
    mat4x4, pack_parameters, struct, u32, vec2, vec3, vec4, vertex,
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
# Declarative Struct Definition Tests (Issue #128)
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

