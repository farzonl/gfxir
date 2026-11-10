"""Unit tests for MLIR pointer type formalization, storage classes, alignment, and cross-target ABI lowering."""

from __future__ import annotations

import os
import struct
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from mlir_gpu import Toolchain, Type
from mlir_gpu.frontend import PointerType, StorageClass, natural_alignment

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLVM_BIN = ROOT / ".toolchains" / "llvm-build" / "bin"
LLVM_BIN = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else (
    DEFAULT_LLVM_BIN if DEFAULT_LLVM_BIN.is_dir() else None
)
GPU_TOOLS_BIN = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
TOOLS = Toolchain(llvm_bin=LLVM_BIN, tools_bin=GPU_TOOLS_BIN)

try:
    GFX_OPT = TOOLS.find("gfx-opt")
    HAS_GFX_OPT = True
except Exception:
    GFX_OPT = None
    HAS_GFX_OPT = False

try:
    GFX_COMPILE = TOOLS.find("gfx-compile")
    HAS_GFX_COMPILE = True
except Exception:
    GFX_COMPILE = None
    HAS_GFX_COMPILE = False

SPIRV_MAGIC = 0x07230203


class TestStorageClass(unittest.TestCase):
    """Storage class enum definitions and string parsing."""

    def test_storage_class_values(self):
        self.assertEqual(StorageClass.GLOBAL.value, "global")
        self.assertEqual(StorageClass.WORKGROUP.value, "workgroup")
        self.assertEqual(StorageClass.PRIVATE.value, "private")
        self.assertEqual(StorageClass.PUSH_CONSTANT.value, "push_constant")

    def test_storage_class_from_string(self):
        self.assertEqual(StorageClass.from_str("global"), StorageClass.GLOBAL)
        self.assertEqual(StorageClass.from_str("workgroup"), StorageClass.WORKGROUP)
        self.assertEqual(StorageClass.from_str("private"), StorageClass.PRIVATE)
        self.assertEqual(StorageClass.from_str("push_constant"), StorageClass.PUSH_CONSTANT)

    def test_invalid_storage_class_rejected(self):
        for invalid in ("heap", "local", "shared", "device", "", "uniform"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    StorageClass.from_str(invalid)


class TestNaturalAlignment(unittest.TestCase):
    """Natural alignment computation semantics under std430 rules."""

    def test_scalar_natural_alignments(self):
        self.assertEqual(natural_alignment(Type.F32), 4)
        self.assertEqual(natural_alignment(Type.I32), 4)
        self.assertEqual(natural_alignment(Type.U32), 4)
        self.assertEqual(natural_alignment(Type.BOOL), 1)

    def test_vector_natural_alignments(self):
        self.assertEqual(natural_alignment(Type.VEC2), 8)
        # std430 rules require 16-byte alignment for 3-component and 4-component vectors:
        self.assertEqual(natural_alignment(Type.VEC3), 16)
        self.assertEqual(natural_alignment(Type.VEC4), 16)

    def test_natural_alignment_vectors_and_scalars_by_string(self):
        expected = {
            "f32": 4,
            "i32": 4,
            "si32": 4,
            "u32": 4,
            "f16": 2,
            "i16": 2,
            "f64": 8,
            "i64": 8,
            "i8": 1,
            "i1": 1,
            "bool": 1,
            "vector<2xf32>": 8,
            "vector<3xf32>": 16,
            "vector<4xf32>": 16,
            "vector<2xf16>": 4,
            "vector<3xf16>": 8,
            "vector<4xf16>": 8,
            "vector<2xi8>": 2,
            "vector<4xi8>": 4,
            "vector<2xf64>": 16,
            "vector<4xf64>": 32,
        }
        for type_str, exp_align in expected.items():
            with self.subTest(type_str=type_str):
                self.assertEqual(natural_alignment(type_str), exp_align)

    def test_aggregate_natural_alignments(self):
        members = [Type.F32, Type.VEC4, Type.I32]
        self.assertEqual(max(natural_alignment(m) for m in members), 16)


class TestPointerTypeModel(unittest.TestCase):
    """PointerType dataclass construction, immutability, and string formatting."""

    def test_default_natural_alignment(self):
        ptr = PointerType(Type.F32, StorageClass.GLOBAL)
        self.assertEqual(ptr.alignment, 4)
        self.assertEqual(ptr.storage_class, StorageClass.GLOBAL)
        self.assertEqual(ptr.element_type, Type.F32)

        ptr_vec4 = PointerType(Type.VEC4, StorageClass.WORKGROUP)
        self.assertEqual(ptr_vec4.alignment, 16)

        ptr_vec2 = PointerType(Type.VEC2, StorageClass.PRIVATE)
        self.assertEqual(ptr_vec2.alignment, 8)

    def test_explicit_valid_alignment(self):
        for align in (4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096, 65536):
            ptr = PointerType(Type.F32, StorageClass.GLOBAL, alignment=align)
            self.assertEqual(ptr.alignment, align)

        ptr_vec4 = PointerType(Type.VEC4, StorageClass.WORKGROUP, alignment=32)
        self.assertEqual(ptr_vec4.alignment, 32)

        ptr_bool = PointerType(Type.BOOL, StorageClass.GLOBAL, alignment=1)
        self.assertEqual(ptr_bool.alignment, 1)

        ptr_f16 = PointerType("f16", StorageClass.GLOBAL, alignment=2)
        self.assertEqual(ptr_f16.alignment, 2)

    def test_string_format(self):
        cases = [
            (PointerType(Type.F32, StorageClass.GLOBAL), "!gfx.ptr<f32, global>"),
            (PointerType(Type.F32, StorageClass.GLOBAL, 4), "!gfx.ptr<f32, global, 4>"),
            (PointerType(Type.F32, StorageClass.GLOBAL, 16), "!gfx.ptr<f32, global, 16>"),
            (PointerType(Type.VEC4, StorageClass.WORKGROUP), "!gfx.ptr<vector<4xf32>, workgroup>"),
            (PointerType(Type.VEC4, StorageClass.WORKGROUP, 32), "!gfx.ptr<vector<4xf32>, workgroup, 32>"),
            (PointerType(Type.I32, StorageClass.PRIVATE), "!gfx.ptr<si32, private>"),
            (PointerType(Type.BOOL, StorageClass.PUSH_CONSTANT), "!gfx.ptr<i1, push_constant>"),
        ]
        for ptr, expected_str in cases:
            with self.subTest(ptr=expected_str):
                self.assertEqual(str(ptr), expected_str)

    def test_immutability(self):
        ptr = PointerType(Type.F32, StorageClass.GLOBAL, 16)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ptr.element_type = Type.I32  # type: ignore
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ptr.storage_class = StorageClass.WORKGROUP  # type: ignore
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ptr.alignment = 32  # type: ignore
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ptr._explicit = False  # type: ignore
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            del ptr.alignment

    def test_equality_and_hashing(self):
        p1 = PointerType(Type.F32, StorageClass.GLOBAL, alignment=16)
        p2 = PointerType(Type.F32, StorageClass.GLOBAL, alignment=16)
        p3 = PointerType(Type.F32, StorageClass.GLOBAL, alignment=32)
        p4 = PointerType(Type.F32, StorageClass.WORKGROUP, alignment=16)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertNotEqual(p1, p4)
        self.assertEqual(hash(p1), hash(p2))
        self.assertEqual(len({p1, p2, p3, p4}), 3)


class TestPointerTypeRejections(unittest.TestCase):
    """PointerType validation for invalid alignments, pointees, and storage classes."""

    def test_non_power_of_two_alignment_rejected(self):
        for align in (0, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 24, 30, 31, 33, 48, 65, 100, 127):
            with self.subTest(align=align):
                with self.assertRaises(ValueError) as ctx:
                    PointerType(Type.F32, StorageClass.GLOBAL, alignment=align)
                self.assertIn("power of two", str(ctx.exception).lower())

    def test_negative_alignments_rejected(self):
        for align in (-1, -2, -4, -8, -16, -1024):
            with self.subTest(align=align):
                with self.assertRaises(ValueError):
                    PointerType(Type.F32, StorageClass.GLOBAL, alignment=align)

    def test_alignment_zero_explicit_message(self):
        with self.assertRaises(ValueError) as ctx:
            PointerType(Type.F32, StorageClass.GLOBAL, alignment=0)
        self.assertEqual(str(ctx.exception), "pointer alignment must be a power of two, got 0")

    def test_sub_natural_alignment_rejected(self):
        cases = [
            (Type.F32, [1, 2]),
            (Type.I32, [1, 2]),
            (Type.U32, [1, 2]),
            (Type.VEC2, [1, 2, 4]),
            (Type.VEC3, [1, 2, 4, 8]),
            (Type.VEC4, [1, 2, 4, 8]),
            ("f64", [1, 2, 4]),
            ("i64", [1, 2, 4]),
            ("f16", [1]),
            ("i16", [1]),
        ]
        for elem_type, bad_aligns in cases:
            for align in bad_aligns:
                with self.subTest(elem_type=elem_type, align=align):
                    with self.assertRaises(ValueError) as ctx:
                        PointerType(elem_type, StorageClass.GLOBAL, alignment=align)
                    self.assertIn("smaller than", str(ctx.exception).lower())

    def test_invalid_argument_types(self):
        for invalid_align in (4.0, 16.0, "4", "16", True, False, [4], {"align": 4}, (4,)):
            with self.subTest(invalid_align=invalid_align):
                with self.assertRaises(TypeError):
                    PointerType(Type.F32, StorageClass.GLOBAL, alignment=invalid_align)  # type: ignore

        with self.assertRaises(TypeError):
            PointerType(None, StorageClass.GLOBAL)  # type: ignore
        with self.assertRaises(TypeError):
            PointerType(Type.F32, "not_a_storage_class")  # type: ignore
        with self.assertRaises(TypeError):
            PointerType(Type.F32, None)  # type: ignore
        with self.assertRaises(ValueError):
            PointerType("!gfx.ptr<f32, global>", StorageClass.GLOBAL)
        with self.assertRaises(ValueError):
            PointerType("!gfx.handle<sampler>", StorageClass.GLOBAL)


@unittest.skipUnless(HAS_GFX_OPT, "gfx-opt tool required")
class TestPointerMLIRRoundTrip(unittest.TestCase):
    """MLIR dialect parser, printer, and verifier tests via gfx-opt."""

    def run_opt(self, mlir_source: str, extra_args: list[str] | None = None) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "input.mlir"
            path.write_text(mlir_source, encoding="utf-8")
            cmd = [str(GFX_OPT), str(path)]
            if extra_args:
                cmd.extend(extra_args)
            return subprocess.run(cmd, capture_output=True, text=True)

    def test_valid_pointer_roundtrip(self):
        mlir_content = """module {
  func.func private @types(
      !gfx.ptr<f32, global>,
      !gfx.ptr<vector<4xf32>, workgroup>,
      !gfx.ptr<f32, private>,
      !gfx.ptr<f32, push_constant>,
      !gfx.ptr<f32, global, 16>)
}
"""
        res = self.run_opt(mlir_content)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        self.assertIn("!gfx.ptr<f32, global>", res.stdout)
        self.assertIn("!gfx.ptr<vector<4xf32>, workgroup>", res.stdout)
        self.assertIn("!gfx.ptr<f32, global, 16>", res.stdout)

    def test_python_str_roundtrip_with_gfx_opt(self):
        pointers = [
            PointerType(Type.F32, StorageClass.GLOBAL),
            PointerType(Type.F32, StorageClass.GLOBAL, 16),
            PointerType(Type.VEC4, StorageClass.WORKGROUP),
            PointerType(Type.VEC4, StorageClass.WORKGROUP, 32),
            PointerType(Type.I32, StorageClass.PRIVATE),
            PointerType(Type.BOOL, StorageClass.PUSH_CONSTANT),
            PointerType("vector<2xf32>", StorageClass.GLOBAL, 8),
            PointerType("f64", StorageClass.GLOBAL, 8),
        ]
        args_str = ", ".join(str(ptr) for ptr in pointers)
        mlir = f"module {{ func.func private @roundtrip({args_str}) }}"
        res = self.run_opt(mlir)
        self.assertEqual(res.returncode, 0, msg=res.stderr)
        for ptr in pointers:
            self.assertIn(str(ptr), res.stdout)

    def test_subnatural_alignment_rejected_by_compiler(self):
        cases = [
            "!gfx.ptr<f32, global, 1>",
            "!gfx.ptr<f32, global, 2>",
            "!gfx.ptr<vector<2xf32>, global, 4>",
            "!gfx.ptr<vector<4xf32>, global, 8>",
            "!gfx.ptr<f64, global, 4>",
        ]
        for ptr_type in cases:
            with self.subTest(ptr_type=ptr_type):
                res = self.run_opt(f"module {{ func.func private @bad({ptr_type}) }}")
                self.assertNotEqual(res.returncode, 0)
                self.assertIn("smaller than the natural alignment", res.stderr)

    def test_npot_alignment_rejected_by_compiler(self):
        for align in (0, 3, 5, 6, 7, 9, 10, 12, 14, 15, 24):
            with self.subTest(align=align):
                res = self.run_opt(f"module {{ func.func private @bad(!gfx.ptr<f32, global, {align}>) }}")
                self.assertNotEqual(res.returncode, 0)
                self.assertIn("pointer alignment must be a power of two", res.stderr)

    def test_alignment_zero_rejected_by_compiler(self):
        res = self.run_opt("module { func.func private @bad(!gfx.ptr<f32, global, 0>) }")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("pointer alignment must be a power of two, got 0", res.stderr)

    def test_nested_pointer_rejected_by_compiler(self):
        res = self.run_opt("module { func.func private @bad(!gfx.ptr<!gfx.ptr<f32, global>, global>) }")
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("pointer element type must be a valid shader data type", res.stderr)

    def test_invalid_struct_pointees_rejected_by_compiler(self):
        cases = [
            "!gfx.ptr<!llvm.struct<(!gfx.ptr<f32, global>)>, global>",
            "!gfx.ptr<!llvm.struct<(!gfx.handle<texture_2d>)>, global>",
            "!gfx.ptr<!llvm.struct<(vector<16xf32>)>, global>",
            "!gfx.ptr<!spirv.struct<(!spirv.ptr<f32, StorageBuffer>)>, global>",
            '!gfx.ptr<!llvm.struct<"Opaque", opaque>, global>',
            '!gfx.ptr<!llvm.struct<"Node", (!llvm.struct<"Node">)>, global>',
        ]
        for ptr_type in cases:
            with self.subTest(ptr_type=ptr_type):
                res = self.run_opt(f"module {{ func.func private @bad({ptr_type}) }}")
                self.assertNotEqual(res.returncode, 0)
                self.assertIn("pointer element type must be a valid shader data type", res.stderr)

    def test_workgroup_pointer_rejected_on_vertex_entry_point(self):
        mlir = """module {
  func.func @vertex_main(%arg: !gfx.ptr<f32, workgroup>) attributes {
    gfx.stage = #gfx.stage<vertex>
  } {
    %pos = arith.constant dense<0.0> : vector<4xf32>
    gfx.position %pos : vector<4xf32>
    return
  }
}
"""
        res = self.run_opt(mlir)
        self.assertNotEqual(res.returncode, 0)
        self.assertIn("workgroup pointers are only allowed in compute entry points", res.stderr)


@unittest.skipUnless(HAS_GFX_COMPILE and HAS_GFX_OPT, "gfx-compile and gfx-opt tools required")
class TestCrossTargetPointerLowering(unittest.TestCase):
    """Compilation and capability-gated pointer lowering across Vulkan and DirectX."""

    COMPUTE_DESCRIPTOR_MLIR = """module {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>, %wg: !gfx.ptr<vector<4xf32>, workgroup>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
"""

    COMPUTE_BDA_MLIR = """module attributes {gfx.required_capabilities = ["PhysicalStorageBuffer64"]} {
  func.func @compute_main(%buf: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
"""

    def run_compile(
        self,
        mlir_source: str,
        target: str = "vulkan",
        extra_args: list[str] | None = None,
    ) -> tuple[int, str, str, bytes, str, str]:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            in_mlir = tmp / "input.mlir"
            out_bin = tmp / "out.bin"
            out_refl = tmp / "refl.json"
            out_lowered = tmp / "lowered.mlir"
            in_mlir.write_text(mlir_source, encoding="utf-8")

            cmd = [
                str(GFX_COMPILE),
                str(in_mlir),
                "--input-format=mlir",
                f"--target={target}",
                f"--output={out_bin}",
                f"--reflection={out_refl}",
                f"--lowered-output={out_lowered}",
            ]
            if extra_args:
                cmd.extend(extra_args)

            proc = subprocess.run(cmd, capture_output=True, text=True)
            bin_bytes = out_bin.read_bytes() if out_bin.is_file() else b""
            refl_text = out_refl.read_text(encoding="utf-8") if out_refl.is_file() else ""
            lowered_text = out_lowered.read_text(encoding="utf-8") if out_lowered.is_file() else ""
            return proc.returncode, proc.stdout, proc.stderr, bin_bytes, refl_text, lowered_text

    def test_vulkan_default_descriptor_lowering_and_spirv_header(self):
        code, _, stderr, bin_bytes, _, lowered = self.run_compile(
            self.COMPUTE_DESCRIPTOR_MLIR, target="vulkan"
        )
        self.assertEqual(code, 0, msg=stderr)
        self.assertIn("!spirv.ptr<f32, StorageBuffer>", lowered)
        self.assertIn("!spirv.ptr<vector<4xf32>, Workgroup>", lowered)
        self.assertIn('spirv.ExecutionMode @compute_main "LocalSize", 8, 8, 1', lowered)
        self.assertIn("spirv.module Logical GLSL450", lowered)

        self.assertGreaterEqual(len(bin_bytes), 20)
        magic, version, _, bound, schema = struct.unpack("<5I", bin_bytes[:20])
        self.assertEqual(magic, SPIRV_MAGIC)
        self.assertEqual(version, 0x00010000)
        self.assertGreater(bound, 0)
        self.assertEqual(schema, 0)

    def test_vulkan_all_four_storage_classes_with_helper(self):
        mlir = """module {
  func.func private @helper(%priv: !gfx.ptr<f32, private>) {
    return
  }

  func.func @compute_main(
      %priv: !gfx.ptr<f32, private>,
      %glob: !gfx.ptr<f32, global>,
      %pc: !gfx.ptr<f32, push_constant>,
      %wg: !gfx.ptr<f32, workgroup>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 4, 4, 1>
  } {
    func.call @helper(%priv) : (!gfx.ptr<f32, private>) -> ()
    return
  }
}
"""
        code, _, stderr, bin_bytes, _, lowered = self.run_compile(mlir, target="vulkan")
        self.assertEqual(code, 0, msg=stderr)
        self.assertIn("!spirv.ptr<f32, Function>", lowered)
        self.assertIn("!spirv.ptr<f32, StorageBuffer>", lowered)
        self.assertIn("!spirv.ptr<f32, PushConstant>", lowered)
        self.assertIn("!spirv.ptr<f32, Workgroup>", lowered)
        self.assertIn("spirv.FunctionCall @helper", lowered)
        self.assertGreaterEqual(len(bin_bytes), 20)
        self.assertEqual(struct.unpack("<I", bin_bytes[:4])[0], SPIRV_MAGIC)

    def test_vulkan_bda_capability_gating(self):
        # Without PhysicalStorageBuffer64 capability: rejected
        code_fail, _, stderr_fail, _, _, _ = self.run_compile(
            self.COMPUTE_BDA_MLIR, target="vulkan"
        )
        self.assertNotEqual(code_fail, 0)
        self.assertIn(
            "target capability profile does not support required capability 'PhysicalStorageBuffer64'",
            stderr_fail,
        )

        # With PhysicalStorageBuffer64 capability: succeeds
        code_pass, _, stderr_pass, bin_bytes, _, lowered = self.run_compile(
            self.COMPUTE_BDA_MLIR,
            target="vulkan",
            extra_args=["--required-capability=PhysicalStorageBuffer64"],
        )
        self.assertEqual(code_pass, 0, msg=stderr_pass)
        self.assertIn("PhysicalStorageBuffer64", lowered)
        self.assertIn("PhysicalStorageBufferAddresses", lowered)
        self.assertIn("!spirv.ptr<f32, PhysicalStorageBuffer>", lowered)
        self.assertIn("spirv.decoration = #spirv.decoration<Aliased>", lowered)
        self.assertGreaterEqual(len(bin_bytes), 20)
        self.assertEqual(struct.unpack("<I", bin_bytes[:4])[0], SPIRV_MAGIC)

    def test_vulkan_bda_gating_via_inline_target_profile(self):
        profile_no_bda = '{"spirv_version": "1.3", "required_capabilities": ["Shader"]}'
        code_fail, _, stderr_fail, _, _, _ = self.run_compile(
            self.COMPUTE_BDA_MLIR,
            target="vulkan",
            extra_args=[f"--target-profile={profile_no_bda}"],
        )
        self.assertNotEqual(code_fail, 0)
        self.assertIn(
            "target capability profile does not support required capability 'PhysicalStorageBuffer64'",
            stderr_fail,
        )

        profile_with_bda = (
            '{"spirv_version": "1.3", "required_capabilities": ["Shader", "PhysicalStorageBuffer64"]}'
        )
        code_pass, _, stderr_pass, _, _, lowered = self.run_compile(
            self.COMPUTE_BDA_MLIR,
            target="vulkan",
            extra_args=[f"--target-profile={profile_with_bda}"],
        )
        self.assertEqual(code_pass, 0, msg=stderr_pass)
        self.assertIn("PhysicalStorageBuffer64", lowered)
        self.assertIn("!spirv.ptr<f32, PhysicalStorageBuffer>", lowered)

    def test_unknown_and_multiple_required_capabilities(self):
        mlir_unknown = """module attributes {gfx.required_capabilities = ["NonExistentGpuFeature"]} {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
"""
        code, _, stderr, _, _, _ = self.run_compile(mlir_unknown, target="vulkan")
        self.assertNotEqual(code, 0)
        self.assertIn(
            "target capability profile does not support required capability 'NonExistentGpuFeature'",
            stderr,
        )

        mlir_multi = """module attributes {gfx.required_capabilities = ["Shader", "PhysicalStorageBuffer64"]} {
  func.func @compute_main(%bda: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
"""
        code1, _, stderr1, _, _, _ = self.run_compile(mlir_multi, target="vulkan")
        self.assertNotEqual(code1, 0)
        self.assertIn("PhysicalStorageBuffer64", stderr1)

        code2, _, stderr2, _, _, lowered2 = self.run_compile(
            mlir_multi,
            target="vulkan",
            extra_args=["--required-capability=PhysicalStorageBuffer64"],
        )
        self.assertEqual(code2, 0, msg=stderr2)
        self.assertIn("PhysicalStorageBuffer64", lowered2)

    def test_directx_all_storage_classes_and_compute_lowering(self):
        mlir = """module {
  func.func private @helper(
      %priv: !gfx.ptr<f32, private>,
      %glob: !gfx.ptr<f32, global>,
      %pc: !gfx.ptr<f32, push_constant>,
      %wg: !gfx.ptr<f32, workgroup>) {
    return
  }

  func.func @compute_main(
      %priv: !gfx.ptr<f32, private>,
      %glob: !gfx.ptr<f32, global>,
      %pc: !gfx.ptr<f32, push_constant>,
      %wg: !gfx.ptr<f32, workgroup>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    func.call @helper(%priv, %glob, %pc, %wg) : (!gfx.ptr<f32, private>, !gfx.ptr<f32, global>, !gfx.ptr<f32, push_constant>, !gfx.ptr<f32, workgroup>) -> ()
    return
  }
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "dx_classes.mlir"
            src.write_text(mlir, encoding="utf-8")
            res = subprocess.run(
                [str(GFX_OPT), str(src), "--gfx-to-directx"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertIn("!llvm.ptr", res.stdout)
            self.assertIn("!llvm.ptr<1>", res.stdout)
            self.assertIn("!llvm.ptr<2>", res.stdout)
            self.assertIn("!llvm.ptr<3>", res.stdout)
            self.assertIn('hlsl.numthreads = "8,8,1"', res.stdout)
            self.assertIn('llvm.target_triple = "dxil-pc-shadermodel6.6-compute"', res.stdout)

        code, _, stderr, bin_bytes, _, lowered = self.run_compile(mlir, target="directx")
        self.assertEqual(code, 0, msg=stderr)
        self.assertIn("!llvm.ptr", lowered)
        self.assertIn("!llvm.ptr<1>", lowered)
        self.assertIn("!llvm.ptr<2>", lowered)
        self.assertIn("!llvm.ptr<3>", lowered)
        self.assertIn('hlsl.numthreads = "8,8,1"', lowered)

        dxil_text = bin_bytes.decode("utf-8", errors="replace")
        self.assertIn("ptr addrspace(1)", dxil_text)
        self.assertIn("ptr addrspace(2)", dxil_text)
        self.assertIn("ptr addrspace(3)", dxil_text)
        self.assertIn('"hlsl.shader"="compute"', dxil_text)


if __name__ == "__main__":
    unittest.main()
