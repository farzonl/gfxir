"""Unit and integration tests for C header emission and C/C++ compiler verification."""

from __future__ import annotations

import json
import os
import re
import shutil
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


class TestCHeaderEmission(unittest.TestCase):
    """Header formatting, struct definitions, explicit padding, and static assertions."""

    def test_c_header_syntax_and_guards(self):
        fields = [
            AggregateField("time", Type.F32),
            AggregateField("resolution", Type.VEC2),
        ]
        layout = compute_std430_layout(fields)
        header = generate_c_header(layout, struct_name="FrameParams")

        self.assertIn("#ifndef FRAMEPARAMS_H", header)
        self.assertIn("#define FRAMEPARAMS_H", header)
        self.assertIn("#include <stddef.h>", header)
        self.assertIn("#include <stdint.h>", header)
        self.assertIn("#include <assert.h>", header)

        self.assertIn("typedef struct FrameParams {", header)
        self.assertIn("float time;", header)
        self.assertIn("float resolution[2];", header)
        self.assertIn("} FrameParams;", header)

        self.assertIn('static_assert(sizeof(FrameParams) == 16, "size mismatch for FrameParams");', header)
        self.assertIn('static_assert(offsetof(FrameParams, time) == 0, "offset mismatch for time");', header)
        self.assertIn('static_assert(offsetof(FrameParams, resolution) == 8, "offset mismatch for resolution");', header)

    def test_c_header_explicit_padding_injection(self):
        fields = [
            AggregateField("a", Type.F32),
            AggregateField("b", Type.VEC3),
            AggregateField("c", Type.I32),
        ]
        layout = compute_std430_layout(fields)
        header = generate_c_header(layout, struct_name="PaddedStruct")

        self.assertIn("uint8_t _pad0[12];", header)
        self.assertIn("static_assert(offsetof(PaddedStruct, a) == 0", header)
        self.assertIn("static_assert(offsetof(PaddedStruct, b) == 16", header)
        self.assertIn("static_assert(offsetof(PaddedStruct, c) == 28", header)
        self.assertIn("static_assert(sizeof(PaddedStruct) == 32", header)

    def test_c_header_vec4_array_dimensions(self):
        for t, expected_elem in ((Type.VEC4, "float"), ("i32x4", "int32_t"), ("u32x4", "uint32_t")):
            with self.subTest(type=t):
                layout = compute_std430_layout([AggregateField("arr", ArrayType(t, 2))], "Vec4Block")
                header = generate_c_header(layout, "Vec4Block")
                self.assertIn(f"{expected_elem} arr[2][4];", header)
                self.assertNotIn(f"{expected_elem} arr[2][2];", header)
                self.assertIn("static_assert(sizeof(Vec4Block) == 32", header)

    def test_c_header_integer_vec3_array_padded_struct_syntax(self):
        for t, base in (("i32x3", "int32_t"), ("u32x3", "uint32_t")):
            with self.subTest(type=t):
                layout = compute_std430_layout([AggregateField("arr", ArrayType(t, 2))], "TestI32x3Array")
                header = generate_c_header(layout, "TestI32x3Array")
                self.assertNotIn(f"{t} arr", header)
                self.assertIn(f"struct {{ {base} v[3]; uint8_t _p[4]; }} arr[2];", header)
                self.assertIn("static_assert(sizeof(TestI32x3Array) == 32", header)

    def test_c_header_array_of_structs_syntax_and_typedef(self):
        pt = StructType("Point", [AggregateField("x", Type.F32), AggregateField("y", Type.F32)])
        layout = compute_std430_layout([AggregateField("pts", ArrayType(pt, 4))], "MeshBlock")
        header = generate_c_header(layout, "MeshBlock")
        self.assertIn("typedef struct Point {", header)
        self.assertIn("Point pts[4];", header)
        self.assertIn("static_assert(sizeof(Point) == 8", header)
        self.assertIn("static_assert(sizeof(MeshBlock) == 32", header)

    def test_c_header_deeply_nested_structs_topological_order(self):
        inn = StructType("Inn", [AggregateField("x", Type.F32)])
        mid = StructType("Mid", [AggregateField("inn", inn)])
        layout = compute_std430_layout([AggregateField("mid", mid)], "DeepOuter")
        header = generate_c_header(layout, "DeepOuter")
        inn_pos = header.find("typedef struct Inn {")
        mid_pos = header.find("typedef struct Mid {")
        outer_pos = header.find("typedef struct DeepOuter {")
        self.assertTrue(0 <= inn_pos < mid_pos < outer_pos, "Typedefs must be emitted in topological order")


class TestCHeaderCompilation(unittest.TestCase):
    """Compilation of emitted C headers across Clang and MSVC C11/C++17 toolchains."""

    @classmethod
    def setUpClass(cls):
        cls.clang = shutil.which("clang") or shutil.which("clang.exe")
        cls.cl = shutil.which("cl") or shutil.which("cl.exe")

    def run_toolchain_compile(self, header_path: Path, toolchain: str, extra_flags: list[str] | None = None):
        extra_flags = extra_flags or []
        header_dir = header_path.parent
        header_name = header_path.name

        if toolchain == "clang-c11":
            wrapper = header_dir / "wrapper.c"
            wrapper.write_text(f'#include "{header_name}"\nint main(void) {{ return 0; }}\n', encoding="utf-8")
            cmd = [
                self.clang,
                "-fsyntax-only",
                "-std=c11",
                "-Wall",
                "-Wextra",
                "-Werror",
                *extra_flags,
                str(wrapper),
            ]
        elif toolchain == "clang-cpp17":
            wrapper = header_dir / "wrapper.cpp"
            wrapper.write_text(f'#include "{header_name}"\nint main() {{ return 0; }}\n', encoding="utf-8")
            cmd = [
                self.clang,
                "-fsyntax-only",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-x",
                "c++",
                *extra_flags,
                str(wrapper),
            ]
        elif toolchain == "msvc-c11":
            wrapper = header_dir / "wrapper.c"
            wrapper.write_text(f'#include "{header_name}"\nint main(void) {{ return 0; }}\n', encoding="utf-8")
            cmd = [
                self.cl,
                "/Zs",
                "/std:c11",
                "/W4",
                "/WX",
                *extra_flags,
                str(wrapper),
            ]
        elif toolchain == "msvc-cpp17":
            wrapper = header_dir / "wrapper.cpp"
            wrapper.write_text(f'#include "{header_name}"\nint main() {{ return 0; }}\n', encoding="utf-8")
            cmd = [
                self.cl,
                "/Zs",
                "/std:c++17",
                "/W4",
                "/WX",
                *extra_flags,
                str(wrapper),
            ]
        else:
            raise ValueError(f"Unknown toolchain {toolchain}")

        return subprocess.run(cmd, capture_output=True, text=True, cwd=str(header_dir))

    def available_toolchains(self) -> list[str]:
        toolchains = []
        if self.clang:
            toolchains.extend(["clang-c11", "clang-cpp17"])
        if self.cl:
            toolchains.extend(["msvc-c11", "msvc-cpp17"])
        return toolchains

    def get_test_headers(self) -> dict[str, str]:
        headers = {}

        scalars_fields = [
            AggregateField("f32_val", Type.F32),
            AggregateField("i32_val", Type.I32),
            AggregateField("u32_val", Type.U32),
            AggregateField("f16_val", Type.F16),
            AggregateField("i16_val", Type.I16),
            AggregateField("u16_val", Type.U16),
            AggregateField("f64_val", Type.F64),
            AggregateField("i64_val", Type.I64),
            AggregateField("u64_val", Type.U64),
            AggregateField("i8_val", Type.I8),
            AggregateField("u8_val", Type.U8),
            AggregateField("bool_val", Type.BOOL),
        ]
        layout1 = compute_std430_layout(scalars_fields, "ScalarPrimitives")
        headers["scalars"] = generate_c_header(layout1, "ScalarPrimitives")

        vector_fields = [
            AggregateField("v2", Type.VEC2),
            AggregateField("v3", Type.VEC3),
            AggregateField("v4", Type.VEC4),
            AggregateField("iv2", "i32x2"),
            AggregateField("iv3", "i32x3"),
            AggregateField("iv4", "i32x4"),
            AggregateField("uv2", "u32x2"),
            AggregateField("uv3", "u32x3"),
            AggregateField("uv4", "u32x4"),
        ]
        layout2 = compute_std430_layout(vector_fields, "VectorAggregates")
        headers["vectors"] = generate_c_header(layout2, "VectorAggregates")

        mixed_fields = [
            AggregateField("a", Type.F32),
            AggregateField("b", Type.VEC3),
            AggregateField("c", Type.F32),
            AggregateField("d", Type.VEC2),
            AggregateField("e", Type.I32),
        ]
        layout3 = compute_std430_layout(mixed_fields, "ComplexBlock")
        headers["complex"] = generate_c_header(layout3, "ComplexBlock")

        array_fields = [
            AggregateField("scalars", ArrayType(Type.F32, 4)),
            AggregateField("normals", ArrayType(Type.VEC3, 3)),
            AggregateField("pairs", ArrayType(Type.VEC2, 2)),
            AggregateField("colors", ArrayType(Type.VEC4, 2)),
            AggregateField("dirs", ArrayType("i32x3", 2)),
        ]
        layout4 = compute_std430_layout(array_fields, "ArrayAggregates")
        headers["arrays"] = generate_c_header(layout4, "ArrayAggregates")

        inner_fields = [
            AggregateField("x", Type.F32),
            AggregateField("y", Type.F32),
            AggregateField("vel", Type.VEC3),
        ]
        inner_struct = StructType("Particle", tuple(inner_fields))
        outer_fields = [
            AggregateField("id", Type.I32),
            AggregateField("particle", inner_struct),
            AggregateField("weight", Type.F32),
        ]
        layout5 = compute_std430_layout(outer_fields, "ParticleSystem")
        headers["nested"] = generate_c_header(layout5, "ParticleSystem")

        deep_fields = [
            AggregateField("tag", Type.U8),
            AggregateField("factor", Type.F32),
            AggregateField("coord", Type.VEC3),
            AggregateField("sub", inner_struct),
            AggregateField("anchors", ArrayType(Type.VEC3, 2)),
            AggregateField("enabled", Type.BOOL),
        ]
        layout6 = compute_std430_layout(deep_fields, "DeepComposite")
        headers["deep"] = generate_c_header(layout6, "DeepComposite")

        return headers

    def test_multi_toolchain_compilation(self):
        toolchains = self.available_toolchains()
        if not toolchains:
            self.skipTest("No C/C++ compiler available on PATH")

        headers = self.get_test_headers()
        for name, header_content in headers.items():
            for tc in toolchains:
                with self.subTest(header=name, toolchain=tc):
                    with tempfile.TemporaryDirectory() as td:
                        h_path = Path(td) / f"{name}.h"
                        h_path.write_text(header_content, encoding="utf-8")
                        res = self.run_toolchain_compile(h_path, tc)
                        self.assertEqual(
                            res.returncode,
                            0,
                            f"Compilation failed for {name} on {tc}:\n{res.stdout}\n{res.stderr}",
                        )

    @unittest.skipUnless(HAS_GFX_COMPILE, "gfx-compile binary not found")
    def test_native_gfx_compile_header_emission_and_compilation(self):
        shader = {
            "schema_version": 2,
            "parameters": {
                "name": "NativeBlock",
                "fields": [
                    {"name": "a", "type": "f32"},
                    {"name": "b", "type": "f32x3"},
                    {"name": "c", "type": "f32"},
                    {"name": "d", "type": "f32x2"},
                    {"name": "e", "type": "i32"},
                ],
            },
            "functions": [
                {
                    "name": "complex_shader",
                    "stage": "fragment",
                    "results": [],
                    "body": {
                        "blocks": [
                            {
                                "name": "entry",
                                "arguments": [],
                                "operations": [
                                    {
                                        "op": "gfx.parameter",
                                        "results": [{"name": "p0", "type": "f32"}],
                                        "attributes": {
                                            "block": {"kind": "string", "value": "NativeBlock"},
                                            "field": {"kind": "string", "value": "a"},
                                        },
                                    },
                                    {
                                        "op": "vector.from_elements",
                                        "args": ["p0", "p0", "p0", "p0"],
                                        "results": [{"name": "color", "type": "f32x4"}],
                                    },
                                    {
                                        "op": "gfx.output",
                                        "args": ["color"],
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
            json_path = Path(td) / "shader.json"
            json_path.write_text(json.dumps(shader), encoding="utf-8")
            h_path = Path(td) / "native_block.h"
            refl_path = Path(td) / "refl.json"
            out_spv = Path(td) / "out.spv"

            res = subprocess.run(
                [
                    str(GFX_COMPILE),
                    str(json_path),
                    "--target",
                    "vulkan",
                    "--output",
                    str(out_spv),
                    "--reflection",
                    str(refl_path),
                    "--c-header",
                    str(h_path),
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, f"gfx-compile failed:\n{res.stderr}")
            self.assertTrue(h_path.exists())

            header_content = h_path.read_text(encoding="utf-8")
            self.assertIn("typedef struct NativeBlock {", header_content)
            self.assertIn("static_assert(sizeof(NativeBlock) ==", header_content)

            for tc in self.available_toolchains():
                with self.subTest(toolchain=tc):
                    c_res = self.run_toolchain_compile(h_path, tc)
                    self.assertEqual(c_res.returncode, 0, f"Native header compilation failed on {tc}:\n{c_res.stderr}")

    def test_sizeof_mismatch_fails_static_assert(self):
        toolchains = self.available_toolchains()
        if not toolchains:
            self.skipTest("No C/C++ compiler available on PATH")

        header = self.get_test_headers()["complex"]
        corrupted = re.sub(
            r"static_assert\(sizeof\(ComplexBlock\) == \d+",
            "static_assert(sizeof(ComplexBlock) == 52",
            header,
        )
        self.assertNotEqual(corrupted, header)

        for tc in toolchains:
            with self.subTest(toolchain=tc):
                with tempfile.TemporaryDirectory() as td:
                    h_path = Path(td) / "corrupt_sizeof.h"
                    h_path.write_text(corrupted, encoding="utf-8")
                    res = self.run_toolchain_compile(h_path, tc)
                    self.assertNotEqual(res.returncode, 0)
                    combined_err = (res.stdout + "\n" + res.stderr).lower()
                    self.assertTrue(
                        "static_assert" in combined_err
                        or "static assertion" in combined_err
                        or "size mismatch" in combined_err
                    )

    def test_offsetof_mismatch_fails_static_assert(self):
        toolchains = self.available_toolchains()
        if not toolchains:
            self.skipTest("No C/C++ compiler available on PATH")

        header = self.get_test_headers()["complex"]
        corrupted = re.sub(
            r"static_assert\(offsetof\(ComplexBlock, b\) == \d+",
            "static_assert(offsetof(ComplexBlock, b) == 12",
            header,
        )
        self.assertNotEqual(corrupted, header)

        for tc in toolchains:
            with self.subTest(toolchain=tc):
                with tempfile.TemporaryDirectory() as td:
                    h_path = Path(td) / "corrupt_offsetof.h"
                    h_path.write_text(corrupted, encoding="utf-8")
                    res = self.run_toolchain_compile(h_path, tc)
                    self.assertNotEqual(res.returncode, 0)
                    combined_err = (res.stdout + "\n" + res.stderr).lower()
                    self.assertTrue(
                        "static_assert" in combined_err
                        or "static assertion" in combined_err
                        or "offset mismatch" in combined_err
                    )

    def test_nested_struct_mismatch_fails_static_assert(self):
        toolchains = self.available_toolchains()
        if not toolchains:
            self.skipTest("No C/C++ compiler available on PATH")

        header = self.get_test_headers()["nested"]
        corrupted_sz = re.sub(
            r"static_assert\(sizeof\(Particle\) == \d+",
            "static_assert(sizeof(Particle) == 40",
            header,
        )
        corrupted_off = re.sub(
            r"static_assert\(offsetof\(Particle, vel\) == \d+",
            "static_assert(offsetof(Particle, vel) == 8",
            header,
        )

        for tc in toolchains:
            with self.subTest(test="nested_sizeof", toolchain=tc):
                with tempfile.TemporaryDirectory() as td:
                    h_path = Path(td) / "nested_sz.h"
                    h_path.write_text(corrupted_sz, encoding="utf-8")
                    res = self.run_toolchain_compile(h_path, tc)
                    self.assertNotEqual(res.returncode, 0)

            with self.subTest(test="nested_offsetof", toolchain=tc):
                with tempfile.TemporaryDirectory() as td:
                    h_path = Path(td) / "nested_off.h"
                    h_path.write_text(corrupted_off, encoding="utf-8")
                    res = self.run_toolchain_compile(h_path, tc)
                    self.assertNotEqual(res.returncode, 0)

    def test_explicit_padding_removal_fails_static_assert(self):
        toolchains = self.available_toolchains()
        if not toolchains:
            self.skipTest("No C/C++ compiler available on PATH")

        header = self.get_test_headers()["complex"]
        unpadded = header.replace("    uint8_t _pad0[12];\n", "")
        self.assertNotEqual(unpadded, header)

        for tc in toolchains:
            with self.subTest(toolchain=tc):
                with tempfile.TemporaryDirectory() as td:
                    h_path = Path(td) / "unpadded.h"
                    h_path.write_text(unpadded, encoding="utf-8")
                    res = self.run_toolchain_compile(h_path, tc)
                    self.assertNotEqual(res.returncode, 0)

    def test_pragma_pack_preserves_explicit_padding_layout(self):
        toolchains = self.available_toolchains()
        if not toolchains:
            self.skipTest("No C/C++ compiler available on PATH")

        headers = self.get_test_headers()
        for name, header_content in headers.items():
            for pack_val in (1, 4):
                packed_header = f"#pragma pack(push, {pack_val})\n{header_content}\n#pragma pack(pop)\n"
                for tc in toolchains:
                    with self.subTest(header=name, pack=pack_val, toolchain=tc):
                        with tempfile.TemporaryDirectory() as td:
                            h_path = Path(td) / f"{name}_pack{pack_val}.h"
                            h_path.write_text(packed_header, encoding="utf-8")
                            res = self.run_toolchain_compile(h_path, tc)
                            self.assertEqual(res.returncode, 0, f"Packing {pack_val} failed for {name} on {tc}:\n{res.stderr}")


if __name__ == "__main__":
    unittest.main()
