"""Empirical limit and boundary tests for parameter blocks and pipeline parameters.

Tests:
1. Total parameter block cumulative size limits (<= 256 bytes vs > 256 bytes).
2. Cardinality boundary cases: empty parameter blocks, single-block vs multi-block pipelines.
3. Struct permutations with mixed types (mat4, vec4, u32, f32, i32) and std430 packing.
4. Validation of descriptive ValueError when parameter block limits are exceeded.
"""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from test_support import TOOLS, available

from mlir_gpu import (
    ParameterBlock,
    Pipeline,
    Target,
    Type,
    Vertex,
    compile_pipeline,
    f32,
    fragment,
    i32,
    mat4,
    pack_parameters,
    select,
    struct,
    u32,
    vec2,
    vec3,
    vec4,
    vertex,
)
from mlir_gpu.interfaces import reflect_pipeline_parameters
from mlir_gpu.parameters import verify_parameters
from test_support import TOOLS


class TestParameterBlockSizeLimits(unittest.TestCase):
    """Stress-test total parameter block size boundary conditions (<= 256B vs > 256B)."""

    def test_single_block_exact_256_bytes_accepted(self):
        @struct
        class Block256:
            ma: mat4
            mb: mat4
            mc: mat4
            md: mat4

        self.assertEqual(Block256.__struct_size__, 256)

        @vertex
        def vs(idx: u32, p: Block256) -> Vertex:
            return Vertex(p.ma * vec4(1.0) + p.mb * vec4(1.0) + p.mc * vec4(1.0) + p.md * vec4(1.0))

        @fragment
        def fs() -> vec4:
            return vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["parameters"]["size"], 256)

        with tempfile.TemporaryDirectory() as td:
            if not available("llc", "dxv"):
                self.skipTest("DirectX compilation tools not available")
            manifest_path = compile_pipeline(pipeline, Target.DIRECTX, Path(td) / "out_dx", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["parameters"]["size"], 256)

    def test_single_block_272_bytes_rejected(self):
        @struct
        class Block272:
            ma: mat4
            mb: mat4
            mc: mat4
            md: mat4
            v: vec4

        self.assertEqual(Block272.__struct_size__, 272)

        @vertex
        def vs(idx: u32, p: Block272) -> Vertex:
            return Vertex(p.v)

        @fragment
        def fs() -> vec4:
            return vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, r"Cumulative parameter blocks size \(272B\) exceeds 256 bytes\."):
                compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)

    def test_multi_block_two_blocks_exact_256_bytes_accepted(self):
        @struct
        class BlockA:
            ma: mat4
            mb: mat4  # 128B

        @struct
        class BlockB:
            ma: mat4
            mb: mat4  # 128B

        @vertex
        def vs(idx: u32, a: BlockA) -> Vertex:
            return Vertex(a.ma * vec4(1.0))

        @fragment
        def fs(b: BlockB) -> vec4:
            return b.ma * vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["parameters"]), 2)
            self.assertEqual(manifest["parameters"][0]["size"], 128)
            self.assertEqual(manifest["parameters"][0]["bindings"]["vulkan"]["offset"], 0)
            self.assertEqual(manifest["parameters"][1]["size"], 128)
            self.assertEqual(manifest["parameters"][1]["bindings"]["vulkan"]["offset"], 128)

        with tempfile.TemporaryDirectory() as td:
            if not available("llc", "dxv"):
                self.skipTest("DirectX compilation tools not available")
            manifest_path = compile_pipeline(pipeline, Target.DIRECTX, Path(td) / "out_dx", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["parameters"]), 2)
            self.assertEqual(manifest["parameters"][0]["bindings"]["directx"]["register"], 0)
            self.assertEqual(manifest["parameters"][1]["bindings"]["directx"]["register"], 1)

    def test_multi_block_two_blocks_272_bytes_rejected(self):
        @struct
        class BlockA:
            ma: mat4
            mb: mat4  # 128B

        @struct
        class BlockB:
            ma: mat4
            mb: mat4
            v: vec4  # 144B -> 128 + 144 = 272B

        @vertex
        def vs(idx: u32, a: BlockA) -> Vertex:
            return Vertex(a.ma * vec4(1.0))

        @fragment
        def fs(b: BlockB) -> vec4:
            return b.v

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, r"Cumulative parameter blocks size \(272B\) exceeds 256 bytes\."):
                compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)

    def test_multi_block_three_blocks_exact_256_bytes_accepted(self):
        @struct
        class Block80A:
            m: mat4
            v: vec4  # 80B

        @struct
        class Block80B:
            m: mat4
            v: vec4  # 80B

        @struct
        class Block96:
            m: mat4
            va: vec4
            vb: vec4  # 96B -> 80 + 80 + 96 = 256B

        @vertex
        def vs(idx: u32, a: Block80A, b: Block80B) -> Vertex:
            return Vertex(a.m * a.v + b.m * b.v)

        @fragment
        def fs(c: Block96) -> vec4:
            return c.va + c.vb

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["parameters"]), 3)
            self.assertEqual(manifest["parameters"][0]["bindings"]["vulkan"]["offset"], 0)
            self.assertEqual(manifest["parameters"][1]["bindings"]["vulkan"]["offset"], 80)
            self.assertEqual(manifest["parameters"][2]["bindings"]["vulkan"]["offset"], 160)

    def test_multi_block_three_blocks_272_bytes_rejected(self):
        @struct
        class Block80A:
            m: mat4
            v: vec4  # 80B

        @struct
        class Block80B:
            m: mat4
            v: vec4  # 80B

        @struct
        class Block112:
            m: mat4
            va: vec4
            vb: vec4
            vc: vec4  # 112B -> 80 + 80 + 112 = 272B

        @vertex
        def vs(idx: u32, a: Block80A, b: Block80B) -> Vertex:
            return Vertex(a.m * a.v + b.m * b.v)

        @fragment
        def fs(c: Block112) -> vec4:
            return c.va

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, r"Cumulative parameter blocks size \(272B\) exceeds 256 bytes\."):
                compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)

    def test_multi_block_four_blocks_exact_256_bytes_accepted(self):
        @struct
        class Block64A:
            m: mat4

        @struct
        class Block64B:
            m: mat4

        @struct
        class Block64C:
            m: mat4

        @struct
        class Block64D:
            m: mat4

        @vertex
        def vs(idx: u32, a: Block64A, b: Block64B) -> Vertex:
            return Vertex(a.m * vec4(1.0) + b.m * vec4(1.0))

        @fragment
        def fs(c: Block64C, d: Block64D) -> vec4:
            return c.m * vec4(1.0) + d.m * vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(len(manifest["parameters"]), 4)
            offsets = [b["bindings"]["vulkan"]["offset"] for b in manifest["parameters"]]
            self.assertEqual(offsets, [0, 64, 128, 192])

    def test_multi_block_five_blocks_272_bytes_rejected(self):
        @struct
        class Block64A: m: mat4
        @struct
        class Block64B: m: mat4
        @struct
        class Block64C: m: mat4
        @struct
        class Block64D: m: mat4
        @struct
        class Block16E: v: vec4  # 4 * 64 + 16 = 272B

        @vertex
        def vs(idx: u32, a: Block64A, b: Block64B, e: Block16E) -> Vertex:
            return Vertex(a.m * e.v + b.m * e.v)

        @fragment
        def fs(c: Block64C, d: Block64D) -> vec4:
            return c.m * vec4(1.0) + d.m * vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, r"Cumulative parameter blocks size \(272B\) exceeds 256 bytes\."):
                compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)

    def test_raw_reflection_dict_257_bytes_rejected(self):
        bad_refl = {
            "schema_version": 1,
            "name": "bad257",
            "layout": "gfx-std430-v1",
            "size": 257,
            "alignment": 16,
            "stages": ["vertex"],
            "ownership": "host",
            "access": "read-only",
            "addressable": False,
            "fields": [{"name": "x", "type": "f32", "offset": 0, "size": 4, "alignment": 4}],
            "bindings": {
                "vulkan": {"storage": "push_constant", "offset": 0},
                "directx": {
                    "storage": "root_constants",
                    "parameter_index": 0,
                    "register": 0,
                    "space": 0,
                    "num32bit_constants": 65,
                    "visibility": "all",
                    "root_signature_version": "1.1",
                },
                "metal": {"storage": "converter_argument_buffer", "buffer_index": 2, "offset": 0, "size": 257},
            },
        }
        with self.assertRaisesRegex(ValueError, "Parameter block has a mismatched size/alignment or exceeds 256 bytes"):
            verify_parameters(bad_refl)


class TestParameterBlockCardinality(unittest.TestCase):
    """Stress-test empty parameter blocks and single-block vs multi-block pipelines."""

    def test_pipeline_without_parameters_succeeds(self):
        @vertex
        def vs(idx: u32) -> Vertex:
            return Vertex(vec4(1.0))

        @fragment
        def fs() -> vec4:
            return vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            self.assertIsNone(manifest.get("parameters"))

    def test_empty_struct_definition_rejected(self):
        with self.assertRaisesRegex(ValueError, "Struct must declare at least one field"):
            @struct
            class EmptyStruct:
                pass

    def test_empty_parameter_block_rejected(self):
        with self.assertRaisesRegex((TypeError, ValueError), "Parameter fields must be a nonempty tuple of \\(name, Type\\) pairs"):
            ParameterBlock("empty_block", ())

    def test_single_block_vertex_only_pipeline_succeeds(self):
        @struct
        class VsBlock:
            m: mat4

        @vertex
        def vs(idx: u32, p: VsBlock) -> Vertex:
            return Vertex(p.m * vec4(1.0))

        @fragment
        def fs() -> vec4:
            return vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest.get("parameters")
            self.assertIsInstance(params, dict)
            self.assertEqual(params["name"], "p")
            self.assertEqual(params["stages"], ["vertex"])

    def test_single_block_fragment_only_pipeline_succeeds(self):
        @struct
        class FsBlock:
            col: vec4

        @vertex
        def vs(idx: u32) -> Vertex:
            return Vertex(vec4(1.0))

        @fragment
        def fs(p: FsBlock) -> vec4:
            return p.col

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest.get("parameters")
            self.assertIsInstance(params, dict)
            self.assertEqual(params["name"], "p")
            self.assertEqual(params["stages"], ["fragment"])

    def test_single_block_shared_between_stages_succeeds(self):
        @struct
        class SharedBlock:
            m: mat4
            col: vec4

        @vertex
        def vs(idx: u32, p: SharedBlock) -> Vertex:
            return Vertex(p.m * vec4(1.0))

        @fragment
        def fs(p: SharedBlock) -> vec4:
            return p.col

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest.get("parameters")
            self.assertIsInstance(params, dict)
            self.assertEqual(params["name"], "p")
            self.assertEqual(sorted(params["stages"]), ["fragment", "vertex"])


class TestStructPermutationsMixedTypes(unittest.TestCase):
    """Stress-test permutations of struct definitions with mixed types (mat4, vec4, u32, f32, i32)."""

    def test_permutation_1_mat4_vec4_u32_f32_i32(self):
        # mat4 (0..64) -> vec4 (64..80) -> u32 (80..84) -> f32 (84..88) -> i32 (88..92) -> padded to 96
        @struct
        class PermutationA:
            m: mat4
            v: vec4
            u: u32
            f: f32
            i: i32

        self.assertEqual(PermutationA.__struct_size__, 96)
        offsets = {f.name: f.offset for f in PermutationA.__struct_fields__}
        self.assertEqual(offsets, {"m": 0, "v": 64, "u": 80, "f": 84, "i": 88})

        inst = PermutationA(tuple(float(k) for k in range(16)), (1.0, 2.0, 3.0, 4.0), 42, 3.14, -7)
        raw = inst.to_bytes()
        self.assertEqual(len(raw), 96)
        restored = PermutationA.from_bytes(raw)
        self.assertEqual(restored.u, 42)
        self.assertEqual(restored.i, -7)

    def test_permutation_2_u32_i32_f32_vec4_mat4(self):
        # u32 (0..4) -> i32 (4..8) -> f32 (8..12) -> pad 4 -> vec4 (16..32) -> mat4 (32..96) = 96
        @struct
        class PermutationB:
            u: u32
            i: i32
            f: f32
            v: vec4
            m: mat4

        self.assertEqual(PermutationB.__struct_size__, 96)
        offsets = {f.name: f.offset for f in PermutationB.__struct_fields__}
        self.assertEqual(offsets, {"u": 0, "i": 4, "f": 8, "v": 16, "m": 32})

    def test_permutation_3_f32_mat4_u32_vec4_i32(self):
        # f32 (0..4) -> pad 12 -> mat4 (16..80) -> u32 (80..84) -> pad 12 -> vec4 (96..112) -> i32 (112..116) -> pad 12 = 128
        @struct
        class PermutationC:
            f: f32
            m: mat4
            u: u32
            v: vec4
            i: i32

        self.assertEqual(PermutationC.__struct_size__, 128)
        offsets = {f.name: f.offset for f in PermutationC.__struct_fields__}
        self.assertEqual(offsets, {"f": 0, "m": 16, "u": 80, "v": 96, "i": 112})

    def test_permutation_4_interleaved_scalars(self):
        # u32 (0..4) -> f32 (4..8) -> u32 (8..12) -> i32 (12..16) -> vec4 (16..32) -> mat4 (32..96) = 96
        @struct
        class PermutationD:
            ua: u32
            fa: f32
            ub: u32
            ia: i32
            vb: vec4
            mb: mat4

        self.assertEqual(PermutationD.__struct_size__, 96)
        offsets = {f.name: f.offset for f in PermutationD.__struct_fields__}
        self.assertEqual(offsets, {"ua": 0, "fa": 4, "ub": 8, "ia": 12, "vb": 16, "mb": 32})

    def test_vulkan_compile_mixed_type_permutations(self):
        @struct
        class PermA:
            m: mat4
            v: vec4
            u: u32
            f: f32
            i: i32

        @struct
        class PermB:
            f: f32
            m: mat4
            u: u32
            v: vec4
            i: i32

        @vertex
        def vs(idx: u32, a: PermA) -> Vertex:
            return Vertex(a.m * a.v)

        @fragment
        def fs(b: PermB) -> vec4:
            return b.v * b.f

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            self.assertTrue(manifest_path.exists())

    def test_directx_compile_float_matrix_vector_parameters(self):
        @struct
        class FloatMatrixBlock:
            m: mat4
            v: vec4
            f: f32

        @vertex
        def vs(idx: u32, p: FloatMatrixBlock) -> Vertex:
            return Vertex(p.m * p.v + vec4(p.f))

        @fragment
        def fs(p: FloatMatrixBlock) -> vec4:
            return p.v * p.f

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            if not available("llc", "dxv"):
                self.skipTest("DirectX compilation tools not available")
            manifest_path = compile_pipeline(pipeline, Target.DIRECTX, Path(td) / "out_dx", TOOLS)
            self.assertTrue(manifest_path.exists())

    def test_directx_compile_integer_parameter_reads(self):
        """Verifies DirectX lowering when shader reads integer parameter fields (u32/i32).

        Lowering.cpp:661 passes field->field.getType() directly to LLVM::LoadOp::create.
        Because StructuredFrontend parses u32 as ui32 and i32 as si32, LLVM::LoadOp rejects
        non-signless integer types with: 'llvm.load' op result #0 must be LLVM type with size, but got 'ui32'.
        """
        @struct
        class MixedInts:
            ua: u32
            ub: u32
            v: vec4

        @vertex
        def vs(idx: u32, p: MixedInts) -> Vertex:
            cond = p.ua == p.ub
            res = select(cond, p.v, vec4(0.0))
            return Vertex(res)

        @fragment
        def fs() -> vec4:
            return vec4(1.0)

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            if not available("llc", "dxv"):
                self.skipTest("DirectX compilation tools not available")
            manifest_path = compile_pipeline(pipeline, Target.DIRECTX, Path(td) / "out_dx", TOOLS)
            self.assertTrue(manifest_path.exists())

    def test_vulkan_compile_integer_parameter_reads(self):
        """Verifies Vulkan lowering when shader reads and operates on integer parameter fields (u32/i32).

        PushConstant structs lower integer fields to signless SPIR-V integers and bitcast loaded values
        back to their declared types. This ensures arithmetic and comparison operations against constants
        and shader inputs verify without SPIR-V dialect type mismatch errors.
        """
        @struct
        class IntegerParameters:
            u: u32
            i: i32
            v: vec4

        @vertex
        def vs(idx: u32, p: IntegerParameters) -> Vertex:
            equal_const = p.u == u32(1)
            equal_input = p.u == idx
            not_equal_const = p.u != u32(0)
            less_than_input = p.u < idx
            greater_than_signed = p.i > i32(0)
            arith_add = p.u + u32(1)
            arith_mul = p.u * idx
            arith_signed = p.i + i32(1)
            cond = (
                equal_const
                & equal_input
                & not_equal_const
                & less_than_input
                & greater_than_signed
                & (arith_add == arith_mul)
                & (arith_signed > i32(0))
            )
            res = select(cond, p.v, vec4(0.0))
            return Vertex(res)

        @fragment
        def fs(p: IntegerParameters) -> vec4:
            cond = (p.u != u32(5)) & (p.i < i32(10)) & ((p.u + u32(2)) > u32(0))
            return select(cond, p.v, vec4(1.0))

        pipeline = Pipeline(vs, fs)
        with tempfile.TemporaryDirectory() as td:
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, Path(td) / "out_vk", TOOLS)
            self.assertTrue(manifest_path.exists())


if __name__ == "__main__":
    unittest.main()
