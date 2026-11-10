"""Unit and integration tests for std430 aggregate layout reflection and parameter packing."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import Toolchain, Type
from mlir_gpu.parameters import (
    AggregateField,
    ArrayType,
    StructType,
    compute_std430_layout,
    generate_c_header,
    pack_parameters,
    verify_parameters,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUILD_BIN = ROOT / "build" / "compiler" / "bin"
DEFAULT_TOOLS_BIN = ROOT / ".tools" / "bin"
if "MLIR_GPU_BIN" not in os.environ:
    for candidate in (DEFAULT_BUILD_BIN, DEFAULT_TOOLS_BIN):
        if candidate.is_dir():
            os.environ["MLIR_GPU_BIN"] = str(candidate)
            break

TOOLS = Toolchain()
try:
    GFX_COMPILE = TOOLS.find("gfx-compile")
    HAS_GFX_COMPILE = True
except Exception:
    GFX_COMPILE = None
    HAS_GFX_COMPILE = False


class TestStd430ScalarLayout(unittest.TestCase):
    """Scalar type layout calculations under std430 rules."""

    def test_scalar_sizes_and_alignments(self):
        cases = [
            (Type.F32, 4, 4),
            (Type.I32, 4, 4),
            (Type.U32, 4, 4),
            (Type.F16, 2, 2),
            (Type.I16, 2, 2),
            (Type.U16, 2, 2),
            (Type.F64, 8, 8),
            (Type.I64, 8, 8),
            (Type.U64, 8, 8),
            (Type.I8, 1, 1),
            (Type.U8, 1, 1),
            (Type.BOOL, 1, 1),
        ]
        for typ, expected_size, expected_align in cases:
            with self.subTest(typ=typ):
                layout = compute_std430_layout([AggregateField("v", typ)])
                self.assertEqual(layout.size, expected_size)
                self.assertEqual(layout.alignment, expected_align)
                self.assertEqual(layout.fields[0].offset, 0)
                self.assertEqual(layout.fields[0].size, expected_size)
                self.assertEqual(layout.fields[0].alignment, expected_align)


class TestStd430VectorLayout(unittest.TestCase):
    """Vector layout calculations under std430 rules."""

    def test_vector_layouts(self):
        cases = [
            (Type.VEC2, 8, 8),
            (Type.VEC3, 12, 16),  # std430 vec3 alignment is 16 bytes
            (Type.VEC4, 16, 16),
        ]
        for typ, expected_size, expected_align in cases:
            with self.subTest(typ=typ):
                layout = compute_std430_layout([AggregateField("v", typ)])
                self.assertEqual(layout.alignment, expected_align)
                self.assertEqual(
                    layout.size,
                    (expected_size + expected_align - 1) // expected_align * expected_align,
                )
                self.assertEqual(layout.fields[0].offset, 0)
                self.assertEqual(layout.fields[0].size, expected_size)
                self.assertEqual(layout.fields[0].alignment, expected_align)

    def test_interleaved_scalars_and_vec3(self):
        fields = [
            AggregateField("a", Type.F32),
            AggregateField("b", Type.VEC3),
            AggregateField("c", Type.F32),
        ]
        layout = compute_std430_layout(fields)
        self.assertEqual(layout.alignment, 16)
        self.assertEqual(layout.size, 32)
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[0].size, 4)
        self.assertEqual(layout.fields[1].offset, 16)
        self.assertEqual(layout.fields[1].size, 12)
        self.assertEqual(layout.fields[2].offset, 28)
        self.assertEqual(layout.fields[2].size, 4)


class TestStd430ArrayLayout(unittest.TestCase):
    """Fixed-size array layout calculations under std430 rules."""

    def test_scalar_array(self):
        arr = ArrayType(element_type=Type.F32, count=5)
        layout = compute_std430_layout([AggregateField("arr", arr)])
        self.assertEqual(layout.alignment, 4)
        self.assertEqual(layout.size, 20)
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[0].size, 20)
        self.assertEqual(layout.fields[0].stride, 4)

    def test_vec3_array_stride_padding(self):
        # Each vec3 element has size 12 and alignment 16; array stride rounds up to 16.
        arr = ArrayType(element_type=Type.VEC3, count=3)
        layout = compute_std430_layout([AggregateField("arr", arr)])
        self.assertEqual(layout.alignment, 16)
        self.assertEqual(layout.size, 48)
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[0].stride, 16)
        self.assertEqual(layout.fields[0].size, 48)

    def test_vec4_array_layout(self):
        for elem_t in (Type.VEC4, "i32x4", "u32x4"):
            with self.subTest(elem_t=elem_t):
                arr = ArrayType(element_type=elem_t, count=3)
                layout = compute_std430_layout([AggregateField("v4_arr", arr)])
                self.assertEqual(layout.alignment, 16)
                self.assertEqual(layout.size, 48)
                self.assertEqual(layout.fields[0].offset, 0)
                self.assertEqual(layout.fields[0].stride, 16)
                self.assertEqual(layout.fields[0].size, 48)

    def test_ivec3_array_layout(self):
        for elem_t in ("i32x3", "u32x3"):
            with self.subTest(elem_t=elem_t):
                arr = ArrayType(element_type=elem_t, count=2)
                layout = compute_std430_layout([AggregateField("v3i_arr", arr)])
                self.assertEqual(layout.alignment, 16)
                self.assertEqual(layout.size, 32)
                self.assertEqual(layout.fields[0].offset, 0)
                self.assertEqual(layout.fields[0].stride, 16)
                self.assertEqual(layout.fields[0].size, 32)


class TestStd430StructLayout(unittest.TestCase):
    """Heterogeneous structs, nested structs, trailing padding, and field alignment."""

    def test_trailing_padding(self):
        fields = [
            AggregateField("v", Type.VEC4),
            AggregateField("b", Type.I8),
        ]
        layout = compute_std430_layout(fields)
        self.assertEqual(layout.alignment, 16)
        self.assertEqual(layout.size, 32)
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[1].offset, 16)

    def test_zero_waste_consecutive_small_types(self):
        fields = [
            AggregateField("a", Type.I8),
            AggregateField("b", Type.I8),
            AggregateField("c", Type.I16),
            AggregateField("d", Type.I32),
        ]
        layout = compute_std430_layout(fields)
        self.assertEqual(layout.alignment, 4)
        self.assertEqual(layout.size, 8)
        self.assertEqual([f.offset for f in layout.fields], [0, 1, 2, 4])

    def test_nested_struct(self):
        inner_struct = StructType(
            "Inner",
            [
                AggregateField("val", Type.I32),
                AggregateField("pos", Type.VEC4),
            ],
        )
        outer_fields = [
            AggregateField("prefix", Type.F32),
            AggregateField("inner", inner_struct),
            AggregateField("suffix", Type.F32),
        ]
        layout = compute_std430_layout(outer_fields)
        self.assertEqual(layout.alignment, 16)
        self.assertEqual(layout.size, 64)
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[1].offset, 16)
        self.assertEqual(layout.fields[1].size, 32)
        self.assertEqual(layout.fields[2].offset, 48)

    def test_array_of_structs(self):
        s = StructType(
            "Point",
            [
                AggregateField("id", Type.F32),
                AggregateField("pos", Type.VEC2),
            ],
        )
        arr = ArrayType(element_type=s, count=4)
        layout = compute_std430_layout([AggregateField("points", arr)])
        self.assertEqual(layout.alignment, 8)
        self.assertEqual(layout.size, 64)
        self.assertEqual(layout.fields[0].stride, 16)

    def test_deeply_nested_struct(self):
        inn = StructType("Inn", [AggregateField("x", Type.F32)])
        mid = StructType("Mid", [AggregateField("inn", inn), AggregateField("y", Type.F32)])
        outer = StructType("DeepOuter", [AggregateField("mid", mid), AggregateField("z", Type.F32)])
        layout = compute_std430_layout([AggregateField("root", outer)])
        self.assertEqual(layout.alignment, 4)
        self.assertEqual(layout.size, 12)
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[0].size, 12)

    def test_mixed_alignment_field_ordering_padding(self):
        fields = [
            AggregateField("a", Type.U8),
            AggregateField("b", Type.VEC4),
            AggregateField("c", Type.U16),
            AggregateField("d", Type.F32),
            AggregateField("e", Type.VEC3),
        ]
        layout = compute_std430_layout(fields, "MixedOrder")
        self.assertEqual(layout.fields[0].offset, 0)
        self.assertEqual(layout.fields[1].offset, 16)
        self.assertEqual(layout.fields[2].offset, 32)
        self.assertEqual(layout.fields[3].offset, 36)
        self.assertEqual(layout.fields[4].offset, 48)
        self.assertEqual(layout.size, 64)
        self.assertEqual(layout.alignment, 16)


class TestAggregateReflectionSchema(unittest.TestCase):
    """JSON reflection schema validation, serialization, and native compiler parity."""

    def test_extended_schema_dict_structure(self):
        fields = [
            AggregateField("time", Type.F32),
            AggregateField("resolution", Type.VEC2),
            AggregateField("data", ArrayType(Type.F32, 4)),
        ]
        reflection = compute_std430_layout(fields).to_reflection_dict("frame_params")
        self.assertEqual(reflection["schema_version"], 2)
        self.assertEqual(reflection["name"], "frame_params")
        self.assertEqual(reflection["alignment"], 8)
        self.assertEqual(reflection["size"], 32)
        self.assertEqual(len(reflection["fields"]), 3)
        self.assertEqual(reflection["fields"][2]["type"], "array")
        self.assertEqual(reflection["fields"][2]["count"], 4)
        self.assertEqual(reflection["fields"][2]["stride"], 4)

    def test_verify_parameters_schema_v2(self):
        valid_refl = {
            "schema_version": 2,
            "name": "test_block",
            "layout": "gfx-std430-v1",
            "size": 32,
            "alignment": 16,
            "fields": [
                {"name": "a", "type": "f32", "offset": 0, "size": 4, "alignment": 4},
                {"name": "b", "type": "f32x3", "offset": 16, "size": 12, "alignment": 16},
                {"name": "c", "type": "f32", "offset": 28, "size": 4, "alignment": 4},
            ],
            "stages": ["compute"],
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "bindings": {
                "vulkan": {"storage": "storage_buffer", "set": 0, "binding": 0},
                "directx": {"storage": "structured_buffer", "register": 0, "space": 0},
                "metal": {"storage": "buffer", "buffer_index": 0},
            },
        }
        verify_parameters(valid_refl)

    def test_array_of_structs_reflection_schema_v2(self):
        pt = StructType("Point", [AggregateField("x", Type.F32), AggregateField("y", Type.F32)])
        layout = compute_std430_layout([AggregateField("pts", ArrayType(pt, 4))], "MeshBlock")
        refl = layout.to_reflection_dict()
        refl_json = json.dumps(refl)
        self.assertNotIn("StructType(", refl_json)
        self.assertEqual(refl["fields"][0]["type"], "array")
        self.assertEqual(refl["fields"][0]["count"], 4)
        self.assertEqual(refl["fields"][0]["stride"], 8)
        self.assertEqual(refl["fields"][0]["size"], 32)

    def test_deeply_nested_struct_reflection_schema_v2(self):
        inn = StructType("Inn", [AggregateField("x", Type.F32)])
        mid = StructType("Mid", [AggregateField("inn", inn)])
        layout = compute_std430_layout([AggregateField("mid", mid)], "DeepOuter")
        refl = layout.to_reflection_dict()
        refl_json = json.dumps(refl)
        self.assertNotIn("StructType(", refl_json)
        self.assertEqual(refl["fields"][0]["type"], "struct")
        self.assertEqual(refl["fields"][0]["name"], "mid")
        sub = refl["fields"][0]["fields"][0]
        self.assertEqual(sub["name"], "inn")
        self.assertEqual(sub["type"], "struct")

    @unittest.skipUnless(HAS_GFX_COMPILE, "gfx-compile binary not available")
    def test_native_compiler_single_scalar_size_parity(self):
        py_layout = compute_std430_layout([AggregateField("val", Type.I32)], "OneInt")
        shader_json = {
            "schema_version": 2,
            "parameters": {
                "name": "OneInt",
                "fields": [{"name": "val", "type": "i32"}],
            },
            "functions": [
                {
                    "name": "shader",
                    "stage": "fragment",
                    "results": [],
                    "body": {
                        "blocks": [
                            {
                                "name": "entry",
                                "arguments": [],
                                "operations": [
                                    {
                                        "op": "arith.constant",
                                        "results": [{"name": "c", "type": "f32x4"}],
                                        "attributes": {
                                            "value": {
                                                "kind": "dense",
                                                "type": "f32x4",
                                                "value": [1.0, 1.0, 1.0, 1.0],
                                            }
                                        },
                                    },
                                    {
                                        "op": "gfx.output",
                                        "args": ["c"],
                                        "results": [],
                                        "attributes": {
                                            "location": {"kind": "literal", "type": "i32", "value": 0},
                                            "name": {"kind": "string", "value": "target0"},
                                            "interpolation": {"kind": "interpolation", "value": "perspective"},
                                        },
                                    },
                                    {"op": "func.return"},
                                ],
                            }
                        ]
                    },
                }
            ],
        }

        with tempfile.TemporaryDirectory() as td:
            in_json = Path(td) / "shader.json"
            out_spv = Path(td) / "shader.spv"
            out_refl = Path(td) / "shader.refl"
            out_hdr = Path(td) / "shader.h"
            in_json.write_text(json.dumps(shader_json), encoding="utf-8")

            res = subprocess.run(
                [
                    str(GFX_COMPILE),
                    str(in_json),
                    "--target",
                    "vulkan",
                    "--output",
                    str(out_spv),
                    "--reflection",
                    str(out_refl),
                    "--c-header",
                    str(out_hdr),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"gfx-compile failed: {res.stderr}")
            native_refl = json.loads(out_refl.read_text(encoding="utf-8"))["parameters"]
            self.assertEqual(native_refl["size"], py_layout.size)


class TestPackAggregateParameters(unittest.TestCase):
    """Binary host parameter packing with std430 layout parity."""

    def test_pack_complex_parameters(self):
        refl = {
            "schema_version": 2,
            "name": "complex_block",
            "layout": "gfx-std430-v1",
            "size": 32,
            "alignment": 16,
            "fields": [
                {"name": "a", "type": "f32", "offset": 0, "size": 4, "alignment": 4},
                {"name": "b", "type": "f32x3", "offset": 16, "size": 12, "alignment": 16},
                {"name": "c", "type": "f32", "offset": 28, "size": 4, "alignment": 4},
            ],
            "stages": ["compute"],
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "bindings": {
                "vulkan": {"storage": "storage_buffer", "set": 0, "binding": 0},
                "directx": {"storage": "structured_buffer", "register": 0, "space": 0},
                "metal": {"storage": "buffer", "buffer_index": 0},
            },
        }
        packed = pack_parameters(refl, {"a": 1.0, "b": (2.0, 3.0, 4.0), "c": 5.0})
        self.assertEqual(len(packed), 32)
        self.assertEqual(struct.unpack_from("<f", packed, 0)[0], 1.0)
        self.assertEqual(struct.unpack_from("<3f", packed, 16), (2.0, 3.0, 4.0))
        self.assertEqual(struct.unpack_from("<f", packed, 28)[0], 5.0)

    def test_pack_array_of_structs(self):
        pt = StructType("Point", [AggregateField("x", Type.F32), AggregateField("y", Type.F32)])
        layout = compute_std430_layout([AggregateField("pts", ArrayType(pt, 2))], "MeshBlock")
        refl = layout.to_reflection_dict()
        packed = pack_parameters(refl, {"pts": [{"x": 1.5, "y": 2.5}, {"x": 3.5, "y": 4.5}]})
        self.assertEqual(len(packed), 16)
        self.assertEqual(struct.unpack_from("<2f", packed, 0), (1.5, 2.5))
        self.assertEqual(struct.unpack_from("<2f", packed, 8), (3.5, 4.5))

    def test_pack_deeply_nested_struct(self):
        inn = StructType("Inn", [AggregateField("val", Type.F32)])
        mid = StructType("Mid", [AggregateField("inn", inn)])
        layout = compute_std430_layout([AggregateField("mid", mid)], "Outer")
        refl = layout.to_reflection_dict()
        packed = pack_parameters(refl, {"mid": {"inn": {"val": 42.0}}})
        self.assertEqual(len(packed), 4)
        self.assertEqual(struct.unpack_from("<f", packed, 0)[0], 42.0)

    def test_pack_integer_vector_arrays(self):
        layout = compute_std430_layout([AggregateField("dirs", ArrayType("i32x3", 2))], "IVec3Block")
        refl = layout.to_reflection_dict()
        packed = pack_parameters(refl, {"dirs": [[1, 2, 3], [4, 5, 6]]})
        self.assertEqual(len(packed), 32)
        self.assertEqual(struct.unpack_from("<3i", packed, 0), (1, 2, 3))
        self.assertEqual(struct.unpack_from("<3i", packed, 16), (4, 5, 6))


if __name__ == "__main__":
    unittest.main()
