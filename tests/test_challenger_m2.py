"""Comprehensive Empirical Challenger Test Suite for Milestone 2 (Issue #44).

Adversarially challenges and stress-tests:
1. Cross-target ABI lowering via gfx-compile.exe (Vulkan Default Portable ABI, Vulkan BDA Gating, DirectX lowering).
2. MLIR dialect verifier enforcement via gfx-opt.exe.
3. Python PointerType dataclass (immutability, type checks, alignment rules, string roundtrip).
"""

from __future__ import annotations

import dataclasses
import os
import struct
import subprocess
import tempfile
import unittest

from mlir_gpu.frontend import PointerType, StorageClass, Type, natural_alignment

GFX_COMPILE = os.path.abspath(".toolchains/llvm-build/bin/gfx-compile.exe")
GFX_OPT = os.path.abspath(".toolchains/llvm-build/bin/gfx-opt.exe")
SPIRV_MAGIC = 0x07230203  # Little-endian in file: \x03\x02\x23\x07


class TestMilestone2LoweringAndGating(unittest.TestCase):
    """Empirical tests for compiler lowering and capability gating."""

    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GFX_COMPILE):
            raise FileNotFoundError(f"gfx-compile not found at {GFX_COMPILE}")
        if not os.path.exists(GFX_OPT):
            raise FileNotFoundError(f"gfx-opt not found at {GFX_OPT}")

    def run_compile(
        self,
        mlir_source: str,
        target: str = "vulkan",
        extra_args: list[str] | None = None,
    ) -> tuple[int, str, str, bytes, str, str]:
        """Runs gfx-compile.exe on MLIR input.
        Returns: (exit_code, stdout, stderr, spv_or_dxil_bytes, reflection_text, lowered_mlir_text)
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            in_mlir = os.path.join(tmpdir, "input.mlir")
            out_bin = os.path.join(tmpdir, "out.bin")
            out_refl = os.path.join(tmpdir, "refl.json")
            out_lowered = os.path.join(tmpdir, "lowered.mlir")

            with open(in_mlir, "w", encoding="utf-8") as f:
                f.write(mlir_source)

            cmd = [
                GFX_COMPILE,
                in_mlir,
                "--input-format=mlir",
                f"--target={target}",
                f"--output={out_bin}",
                f"--reflection={out_refl}",
                f"--lowered-output={out_lowered}",
            ]
            if extra_args:
                cmd.extend(extra_args)

            proc = subprocess.run(cmd, capture_output=True, text=True)

            bin_bytes = b""
            if os.path.exists(out_bin):
                with open(out_bin, "rb") as f:
                    bin_bytes = f.read()

            refl_text = ""
            if os.path.exists(out_refl):
                with open(out_refl, "r", encoding="utf-8") as f:
                    refl_text = f.read()

            lowered_text = ""
            if os.path.exists(out_lowered):
                with open(out_lowered, "r", encoding="utf-8") as f:
                    lowered_text = f.read()

            return proc.returncode, proc.stdout, proc.stderr, bin_bytes, refl_text, lowered_text

    def run_opt(self, mlir_source: str, extra_args: list[str] | None = None) -> tuple[int, str, str]:
        """Runs gfx-opt.exe on MLIR input. Returns (exit_code, stdout, stderr)."""
        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as f:
            f.write(mlir_source)
            temp_path = f.name
        try:
            cmd = [GFX_OPT, temp_path]
            if extra_args:
                cmd.extend(extra_args)
            proc = subprocess.run(cmd, capture_output=True, text=True)
            return proc.returncode, proc.stdout, proc.stderr
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    # =========================================================================
    # 1. Vulkan Default Portable ABI Lowering
    # =========================================================================

    def test_vulkan_default_portable_abi(self):
        """Compile compute shader declaring !gfx.ptr<f32, global> and
        !gfx.ptr<vector<4xf32>, workgroup> to SPIR-V.
        Verify spv.ptr<..., StorageBuffer>, spv.ptr<..., Workgroup>, and SPIR-V magic.
        """
        mlir = """
module {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>, %wg: !gfx.ptr<vector<4xf32>, workgroup>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
"""
        code, stdout, stderr, bin_bytes, refl, lowered = self.run_compile(mlir, target="vulkan")
        self.assertEqual(code, 0, f"Compilation failed: {stderr}")

        # Check lowered MLIR types
        self.assertIn("!spirv.ptr<f32, StorageBuffer>", lowered)
        self.assertIn("!spirv.ptr<vector<4xf32>, Workgroup>", lowered)
        self.assertIn('spirv.ExecutionMode @compute_main "LocalSize", 8, 8, 1', lowered)
        self.assertIn('spirv.module Logical GLSL450', lowered)

        # Check SPIR-V binary magic number (0x07230203)
        self.assertGreaterEqual(len(bin_bytes), 20, "SPIR-V binary is too small")
        magic = struct.unpack("<I", bin_bytes[:4])[0]
        self.assertEqual(magic, SPIRV_MAGIC, f"Invalid SPIR-V magic: {hex(magic)}")

        # Check SPIR-V version (1.0 = 0x00010000)
        version = struct.unpack("<I", bin_bytes[4:8])[0]
        self.assertEqual(version, 0x00010000, f"Expected SPIR-V 1.0, got {hex(version)}")

    def test_vulkan_all_four_storage_classes(self):
        """Verify lowering of all storage classes: global, workgroup, private, push_constant
        both in entry point and in reachable helper functions.
        """
        mlir = """
module {
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
        code, stdout, stderr, bin_bytes, refl, lowered = self.run_compile(mlir, target="vulkan")
        self.assertEqual(code, 0, f"Storage classes compilation failed: {stderr}")
        self.assertIn("!spirv.ptr<f32, Function>", lowered)
        self.assertIn("!spirv.ptr<f32, StorageBuffer>", lowered)
        self.assertIn("!spirv.ptr<f32, PushConstant>", lowered)
        self.assertIn("!spirv.ptr<f32, Workgroup>", lowered)
        self.assertIn("spirv.FunctionCall @helper", lowered)

        # Check SPIR-V binary magic
        self.assertGreaterEqual(len(bin_bytes), 20)
        magic = struct.unpack("<I", bin_bytes[:4])[0]
        self.assertEqual(magic, SPIRV_MAGIC)

    # =========================================================================
    # 2. Vulkan BDA Capability Gating
    # =========================================================================

    def test_vulkan_bda_rejection_without_capability(self):
        """Module declaring gfx.required_capabilities = ["PhysicalStorageBuffer64"]
        without profile capability MUST BE REJECTED with:
        'target capability profile does not support required capability 'PhysicalStorageBuffer64''
        """
        mlir = """
module attributes {gfx.required_capabilities = ["PhysicalStorageBuffer64"]} {
  func.func @compute_main(%bda_ptr: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
"""
        code, stdout, stderr, bin_bytes, refl, lowered = self.run_compile(mlir, target="vulkan")
        self.assertNotEqual(code, 0, "Compilation should have failed when capability is absent")
        expected_diag = "target capability profile does not support required capability 'PhysicalStorageBuffer64'"
        self.assertIn(expected_diag, stderr)

    def test_vulkan_bda_success_with_required_capability_flag(self):
        """Module declaring gfx.required_capabilities = ["PhysicalStorageBuffer64"]
        with --required-capability=PhysicalStorageBuffer64 MUST SUCCEED, emitting:
        AddressingModel PhysicalStorageBuffer64, capability PhysicalStorageBufferAddresses,
        storage class PhysicalStorageBuffer, and Aliased decoration.
        """
        mlir = """
module attributes {gfx.required_capabilities = ["PhysicalStorageBuffer64"]} {
  func.func @compute_main(%bda_ptr: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
"""
        code, stdout, stderr, bin_bytes, refl, lowered = self.run_compile(
            mlir,
            target="vulkan",
            extra_args=["--required-capability=PhysicalStorageBuffer64"],
        )
        self.assertEqual(code, 0, f"BDA compilation failed: {stderr}")

        # Check addressing model, capability, and storage class
        self.assertIn("PhysicalStorageBuffer64", lowered)
        self.assertIn("PhysicalStorageBufferAddresses", lowered)
        self.assertIn("!spirv.ptr<f32, PhysicalStorageBuffer>", lowered)
        self.assertIn("spirv.decoration = #spirv.decoration<Aliased>", lowered)

        # Check valid SPIR-V magic
        self.assertGreaterEqual(len(bin_bytes), 20)
        magic = struct.unpack("<I", bin_bytes[:4])[0]
        self.assertEqual(magic, SPIRV_MAGIC, f"Invalid SPIR-V magic: {hex(magic)}")

    def test_vulkan_bda_gating_via_inline_target_profile(self):
        """Verify gating when target profile JSON is passed explicitly."""
        mlir = """
module attributes {gfx.required_capabilities = ["PhysicalStorageBuffer64"]} {
  func.func @compute_main(%bda_ptr: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 8, 8, 1>
  } {
    return
  }
}
"""
        # Inline profile WITHOUT PhysicalStorageBuffer64
        profile_no_bda = '{"spirv_version": "1.3", "required_capabilities": ["Shader"]}'
        code, stdout, stderr, _, _, _ = self.run_compile(
            mlir, target="vulkan", extra_args=[f"--target-profile={profile_no_bda}"]
        )
        self.assertNotEqual(code, 0)
        self.assertIn("target capability profile does not support required capability 'PhysicalStorageBuffer64'", stderr)

        # Inline profile WITH PhysicalStorageBuffer64
        profile_with_bda = '{"spirv_version": "1.3", "required_capabilities": ["Shader", "PhysicalStorageBuffer64"]}'
        code, stdout, stderr, bin_bytes, _, lowered = self.run_compile(
            mlir, target="vulkan", extra_args=[f"--target-profile={profile_with_bda}"]
        )
        self.assertEqual(code, 0, f"Failed with BDA target profile: {stderr}")
        self.assertIn("PhysicalStorageBuffer64", lowered)
        self.assertIn("!spirv.ptr<f32, PhysicalStorageBuffer>", lowered)

    # =========================================================================
    # 3. DirectX Lowering & Address Space Mapping
    # =========================================================================

    def test_directx_address_space_mapping(self):
        """Compile with --target directx and verify address space mapping:
        0 = private
        1 = global
        2 = push_constant
        3 = workgroup
        """
        mlir = """
module {
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
        # Test 1: Direct pass verification via gfx-opt --gfx-to-directx
        code, opt_out, opt_err = self.run_opt(mlir, ["--gfx-to-directx"])
        self.assertEqual(code, 0, f"gfx-to-directx pass failed: {opt_err}")
        self.assertIn("!llvm.ptr", opt_out)       # private (addrspace 0)
        self.assertIn("!llvm.ptr<1>", opt_out)    # global (addrspace 1)
        self.assertIn("!llvm.ptr<2>", opt_out)    # push_constant (addrspace 2)
        self.assertIn("!llvm.ptr<3>", opt_out)    # workgroup (addrspace 3)
        self.assertIn('hlsl.numthreads = "8,8,1"', opt_out)
        self.assertIn('llvm.target_triple = "dxil-pc-shadermodel6.6-compute"', opt_out)

        # Test 2: Full compilation via gfx-compile --target directx
        code, stdout, stderr, bin_bytes, refl, lowered = self.run_compile(mlir, target="directx")
        self.assertEqual(code, 0, f"gfx-compile directx failed: {stderr}")
        self.assertIn("!llvm.ptr", lowered)
        self.assertIn("!llvm.ptr<1>", lowered)
        self.assertIn("!llvm.ptr<2>", lowered)
        self.assertIn("!llvm.ptr<3>", lowered)
        self.assertIn('hlsl.numthreads = "8,8,1"', lowered)

        # Check emitted DXIL / LLVM IR disassembly
        dxil_text = bin_bytes.decode("utf-8", errors="replace")
        self.assertIn("ptr addrspace(1)", dxil_text)
        self.assertIn("ptr addrspace(2)", dxil_text)
        self.assertIn("ptr addrspace(3)", dxil_text)
        self.assertIn('"hlsl.shader"="compute"', dxil_text)

    # =========================================================================
    # 4. Dialect Verifier Negative Tests
    # =========================================================================

    def test_verifier_rejection_subnatural_alignment(self):
        """Reject sub-natural alignments for f32, vector, and f64."""
        cases = [
            ("!gfx.ptr<f32, global, 1>", "smaller than the natural alignment"),
            ("!gfx.ptr<f32, global, 2>", "smaller than the natural alignment"),
            ("!gfx.ptr<vector<2xf32>, global, 4>", "smaller than the natural alignment"),
            ("!gfx.ptr<vector<4xf32>, global, 8>", "smaller than the natural alignment"),
            ("!gfx.ptr<f64, global, 4>", "smaller than the natural alignment"),
        ]
        for ptr_type, err_sub in cases:
            mlir = f"module {{ func.func private @bad({ptr_type}) }}"
            code, stdout, stderr = self.run_opt(mlir)
            self.assertNotEqual(code, 0, f"Expected failure for {ptr_type}")
            self.assertIn(err_sub, stderr)

    def test_verifier_rejection_non_power_of_two_alignment(self):
        """Reject non-power-of-two alignments."""
        cases = [0, 3, 5, 6, 7, 9, 10, 12, 14, 15, 24]
        for align in cases:
            mlir = f"module {{ func.func private @bad(!gfx.ptr<f32, global, {align}>) }}"
            code, stdout, stderr = self.run_opt(mlir)
            self.assertNotEqual(code, 0, f"Expected failure for align {align}")
            self.assertIn("pointer alignment must be a power of two", stderr)

    def test_verifier_rejection_struct_members(self):
        """Reject structs containing invalid pointee types like pointers, handles, or cycles."""
        cases = [
            "!gfx.ptr<!llvm.struct<(!gfx.ptr<f32, global>)>, global>",
            "!gfx.ptr<!llvm.struct<(!gfx.handle<texture_2d>)>, global>",
            "!gfx.ptr<!llvm.struct<(vector<16xf32>)>, global>",
            "!gfx.ptr<!spirv.struct<(!spirv.ptr<f32, StorageBuffer>)>, global>",
            '!gfx.ptr<!llvm.struct<"Opaque", opaque>, global>',
            '!gfx.ptr<!llvm.struct<"Node", (!llvm.struct<"Node">)>, global>',
        ]
        for ptr_type in cases:
            mlir = f"module {{ func.func private @bad({ptr_type}) }}"
            code, stdout, stderr = self.run_opt(mlir)
            self.assertNotEqual(code, 0, f"Expected failure for {ptr_type}")
            self.assertIn("pointer element type must be a valid shader data type", stderr)

    def test_verifier_rejection_nested_pointers(self):
        """Reject nested pointer types."""
        mlir = "module { func.func private @bad(!gfx.ptr<!gfx.ptr<f32, global>, global>) }"
        code, stdout, stderr = self.run_opt(mlir)
        self.assertNotEqual(code, 0)
        self.assertIn("pointer element type must be a valid shader data type", stderr)

    def test_verifier_rejection_stage_restrictions(self):
        """Reject workgroup pointers on vertex entry points."""
        mlir = """
module {
  func.func @vertex_main(%arg: !gfx.ptr<f32, workgroup>) attributes {
    gfx.stage = #gfx.stage<vertex>
  } {
    %pos = arith.constant dense<0.0> : vector<4xf32>
    gfx.position %pos : vector<4xf32>
    return
  }
}
"""
        code, stdout, stderr = self.run_opt(mlir)
        self.assertNotEqual(code, 0)
        self.assertIn("workgroup pointers are only allowed in compute entry points", stderr)


    def test_spirv_header_structure(self):
        """Verify the 5-word SPIR-V header structure in emitted binary."""
        mlir = """
module {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
"""
        code, stdout, stderr, bin_bytes, refl, lowered = self.run_compile(mlir, target="vulkan")
        self.assertEqual(code, 0, f"Compilation failed: {stderr}")
        self.assertGreaterEqual(len(bin_bytes), 20)
        magic, version, generator, bound, schema = struct.unpack("<5I", bin_bytes[:20])
        self.assertEqual(magic, SPIRV_MAGIC, "Magic word must be 0x07230203")
        self.assertEqual(version, 0x00010000, "Version word must be SPIR-V 1.0 (0x00010000)")
        self.assertGreater(bound, 0, "ID bound must be non-zero")
        self.assertEqual(schema, 0, "Schema word must be 0")

    def test_unknown_capability_rejection(self):
        """Module declaring an unsupported capability is rejected with exact diagnostic."""
        mlir = """
module attributes {gfx.required_capabilities = ["NonExistentGpuFeature"]} {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
"""
        code, stdout, stderr, _, _, _ = self.run_compile(mlir, target="vulkan")
        self.assertNotEqual(code, 0)
        self.assertIn("target capability profile does not support required capability 'NonExistentGpuFeature'", stderr)

    def test_multiple_required_capabilities(self):
        """Module declaring multiple capabilities enforces all required capabilities."""
        mlir = """
module attributes {gfx.required_capabilities = ["Shader", "PhysicalStorageBuffer64"]} {
  func.func @compute_main(%bda: !gfx.ptr<f32, global, 16>) attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 1, 1, 1>
  } {
    return
  }
}
"""
        # Default profile has Shader but not PhysicalStorageBuffer64 -> must fail
        code1, _, stderr1, _, _, _ = self.run_compile(mlir, target="vulkan")
        self.assertNotEqual(code1, 0)
        self.assertIn("PhysicalStorageBuffer64", stderr1)

        # Supplying PhysicalStorageBuffer64 -> succeeds
        code2, _, stderr2, bin_bytes2, _, lowered2 = self.run_compile(
            mlir, target="vulkan", extra_args=["--required-capability=PhysicalStorageBuffer64"]
        )
        self.assertEqual(code2, 0, f"Failed with PhysicalStorageBuffer64: {stderr2}")
        self.assertIn("PhysicalStorageBuffer64", lowered2)


class TestPythonPointerType(unittest.TestCase):
    """Empirical stress-testing of Python PointerType dataclass."""

    def test_immutability(self):
        """Verify dataclass is frozen (cannot mutate attributes)."""
        ptr = PointerType(Type.F32, StorageClass.GLOBAL, 16)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ptr.element_type = Type.I32
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ptr.storage_class = StorageClass.WORKGROUP
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ptr.alignment = 32
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ptr._explicit = False
        with self.assertRaises(dataclasses.FrozenInstanceError):
            del ptr.alignment

    def test_invalid_alignment_types(self):
        """Reject floats, strings, booleans, lists, and dicts for alignment."""
        invalid_types = [
            4.0,
            16.0,
            "4",
            "16",
            True,   # Crucial: isinstance(True, int) is True in Python
            False,
            [4],
            {"align": 4},
            (4,),
        ]
        for invalid_align in invalid_types:
            with self.assertRaises(TypeError, msg=f"Should reject {invalid_align!r} ({type(invalid_align).__name__})"):
                PointerType(Type.F32, StorageClass.GLOBAL, invalid_align)

    def test_negative_and_zero_alignments(self):
        """Reject negative and zero alignments."""
        bad_numbers = [0, -1, -2, -4, -8, -16, -1024]
        for num in bad_numbers:
            with self.assertRaises(ValueError, msg=f"Should reject alignment {num}"):
                PointerType(Type.F32, StorageClass.GLOBAL, num)

    def test_non_power_of_two_alignments(self):
        """Reject non-power-of-two alignments."""
        npot_numbers = [3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 17, 18, 20, 24, 30, 31, 33, 48, 65, 100, 127]
        for num in npot_numbers:
            with self.assertRaises(ValueError, msg=f"Should reject non-power-of-two alignment {num}"):
                PointerType(Type.F32, StorageClass.GLOBAL, num)

    def test_subnatural_alignments(self):
        """Reject alignments smaller than pointee natural alignment."""
        cases = [
            (Type.F32, [1, 2]),                 # nat = 4
            (Type.I32, [1, 2]),                 # nat = 4
            (Type.U32, [1, 2]),                 # nat = 4
            (Type.VEC2, [1, 2, 4]),             # nat = 8
            (Type.VEC3, [1, 2, 4, 8]),          # nat = 16
            (Type.VEC4, [1, 2, 4, 8]),          # nat = 16
            ("f64", [1, 2, 4]),                 # nat = 8
            ("i64", [1, 2, 4]),                 # nat = 8
            ("f16", [1]),                       # nat = 2
            ("i16", [1]),                       # nat = 2
        ]
        for elem_type, bad_aligns in cases:
            for align in bad_aligns:
                with self.assertRaises(ValueError, msg=f"Should reject sub-natural align {align} for {elem_type}"):
                    PointerType(elem_type, StorageClass.GLOBAL, align)

    def test_natural_and_supernatural_alignments(self):
        """Accept default natural alignment and valid super-natural alignments."""
        # Default alignment adopted
        ptr_f32 = PointerType(Type.F32, StorageClass.GLOBAL)
        self.assertEqual(ptr_f32.alignment, 4)

        ptr_vec4 = PointerType(Type.VEC4, StorageClass.WORKGROUP)
        self.assertEqual(ptr_vec4.alignment, 16)

        ptr_vec2 = PointerType(Type.VEC2, StorageClass.PRIVATE)
        self.assertEqual(ptr_vec2.alignment, 8)

        # Super-natural alignments succeed
        valid_aligns = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096]
        for align in valid_aligns:
            ptr = PointerType(Type.F32, StorageClass.GLOBAL, align)
            self.assertEqual(ptr.alignment, align)

    def test_string_formatting_roundtrip(self):
        """Verify string formatting matches MLIR syntax for both default and explicit alignments."""
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
            self.assertEqual(str(ptr), expected_str)

    def test_invalid_element_type_and_storage_class(self):
        """Reject invalid element types and storage classes."""
        # Disallow None
        with self.assertRaises(TypeError):
            PointerType(None, StorageClass.GLOBAL)

        # Disallow nested pointer string
        with self.assertRaises(ValueError):
            PointerType("!gfx.ptr<f32, global>", StorageClass.GLOBAL)

        # Disallow handle string
        with self.assertRaises(ValueError):
            PointerType("!gfx.handle<sampler>", StorageClass.GLOBAL)

        # Disallow invalid storage class type
        with self.assertRaises(TypeError):
            PointerType(Type.F32, "global")  # String instead of StorageClass enum

        with self.assertRaises(TypeError):
            PointerType(Type.F32, None)

    def test_equality_and_hashing(self):
        """Verify value equality and hashing (can be used in sets and dicts)."""
        p1 = PointerType(Type.F32, StorageClass.GLOBAL, 16)
        p2 = PointerType(Type.F32, StorageClass.GLOBAL, 16)
        p3 = PointerType(Type.F32, StorageClass.GLOBAL, 32)
        p4 = PointerType(Type.F32, StorageClass.WORKGROUP, 16)

        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertNotEqual(p1, p4)
        self.assertEqual(hash(p1), hash(p2))

        # Check set deduplication
        s = {p1, p2, p3, p4}
        self.assertEqual(len(s), 3)

    def test_natural_alignment_vectors_and_scalars(self):
        """Verify std430 natural alignment calculation across scalars and vectors."""
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
            self.assertEqual(
                natural_alignment(type_str),
                exp_align,
                f"Mismatch for natural_alignment({type_str!r})",
            )

    def test_boundary_alignments(self):
        """Test boundary alignment cases like 1-byte types, large powers of two."""
        # 1-byte alignment valid for i1/bool
        ptr_bool = PointerType(Type.BOOL, StorageClass.GLOBAL, 1)
        self.assertEqual(ptr_bool.alignment, 1)

        # 2-byte alignment valid for f16
        ptr_f16 = PointerType("f16", StorageClass.GLOBAL, 2)
        self.assertEqual(ptr_f16.alignment, 2)

        # Large power-of-two alignment
        ptr_large = PointerType(Type.F32, StorageClass.GLOBAL, 65536)
        self.assertEqual(ptr_large.alignment, 65536)

    def test_python_str_roundtrip_with_gfx_opt(self):
        """Verify that string representations formatted by Python PointerType
        are strictly legal MLIR types accepted by gfx-opt.exe.
        """
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
        args_str = ", ".join(f"%arg{i}: {ptr}" for i, ptr in enumerate(pointers))
        mlir = f"module {{ func.func private @roundtrip({args_str}) }}"

        with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False, encoding="utf-8") as f:
            f.write(mlir)
            temp_path = f.name
        try:
            cmd = [GFX_OPT, temp_path]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                proc.returncode,
                0,
                f"gfx-opt failed to parse Python PointerType strings:\n{proc.stderr}\nMLIR:\n{mlir}",
            )
            for ptr in pointers:
                self.assertIn(str(ptr), proc.stdout)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if __name__ == "__main__":
    unittest.main()
