"""Unit tests for MLIR pointer type formalization, storage classes, alignment, and ABI (Issue #44)."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from mlir_gpu import (
    CompileError,
    HIGH_END_COMPUTE,
    Target,
    TargetProfile,
    Toolchain,
    Type,
    VULKAN_PORTABLE_1_0,
)
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
    HAS_GFX_OPT = False

try:
    GFX_COMPILE = TOOLS.find("gfx-compile")
    HAS_GFX_COMPILE = True
except Exception:
    HAS_GFX_COMPILE = False


class TestStorageClass(unittest.TestCase):
    """Tier 1: Storage class enum definitions and values."""

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
    """Tier 1 & 2: Natural alignment computation semantics."""

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

    def test_aggregate_natural_alignments(self):
        # Struct alignment equals maximum member alignment:
        members = [Type.F32, Type.VEC4, Type.I32]
        self.assertEqual(max(natural_alignment(m) for m in members), 16)


class TestPointerTypeModel(unittest.TestCase):
    """Tier 1: PointerType dataclass construction, immutability, and string representation."""

    def test_default_natural_alignment(self):
        ptr = PointerType(Type.F32, StorageClass.GLOBAL)
        self.assertEqual(ptr.alignment, 4)
        self.assertEqual(ptr.storage_class, StorageClass.GLOBAL)
        self.assertEqual(ptr.element_type, Type.F32)

    def test_explicit_valid_alignment(self):
        ptr = PointerType(Type.F32, StorageClass.GLOBAL, alignment=16)
        self.assertEqual(ptr.alignment, 16)

        ptr_vec4 = PointerType(Type.VEC4, StorageClass.WORKGROUP, alignment=32)
        self.assertEqual(ptr_vec4.alignment, 32)

    def test_string_format(self):
        # Default natural alignment omitted in standard shorthand:
        ptr1 = PointerType(Type.F32, StorageClass.GLOBAL)
        self.assertEqual(str(ptr1), "!gfx.ptr<f32, global>")

        ptr2 = PointerType(Type.VEC4, StorageClass.WORKGROUP)
        self.assertEqual(str(ptr2), "!gfx.ptr<vector<4xf32>, workgroup>")

        # Explicit over-alignment included:
        ptr3 = PointerType(Type.F32, StorageClass.GLOBAL, alignment=16)
        self.assertEqual(str(ptr3), "!gfx.ptr<f32, global, 16>")

    def test_immutability(self):
        ptr = PointerType(Type.F32, StorageClass.GLOBAL)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ptr.alignment = 16  # type: ignore

    def test_equality_and_hashing(self):
        p1 = PointerType(Type.F32, StorageClass.GLOBAL, alignment=16)
        p2 = PointerType(Type.F32, StorageClass.GLOBAL, alignment=16)
        p3 = PointerType(Type.F32, StorageClass.WORKGROUP, alignment=16)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertEqual(hash(p1), hash(p2))


class TestPointerTypeRejections(unittest.TestCase):
    """Tier 2: Boundary & Corner case validation."""

    def test_non_power_of_two_alignment_rejected(self):
        for align in (0, 3, 5, 6, 7, 10, 12, 18, 24, 33):
            with self.subTest(align=align):
                with self.assertRaises(ValueError) as ctx:
                    PointerType(Type.F32, StorageClass.GLOBAL, alignment=align)
                self.assertIn("power of two", str(ctx.exception).lower())

    def test_sub_natural_alignment_rejected(self):
        # f32 natural is 4; alignment 2 is sub-natural:
        with self.assertRaises(ValueError) as ctx:
            PointerType(Type.F32, StorageClass.GLOBAL, alignment=2)
        self.assertIn("smaller than", str(ctx.exception).lower())

        # vec2 natural is 8; alignment 4 is sub-natural:
        with self.assertRaises(ValueError) as ctx:
            PointerType(Type.VEC2, StorageClass.GLOBAL, alignment=4)
        self.assertIn("smaller than", str(ctx.exception).lower())

        # vec4 natural is 16; alignment 8 is sub-natural:
        with self.assertRaises(ValueError) as ctx:
            PointerType(Type.VEC4, StorageClass.WORKGROUP, alignment=8)
        self.assertIn("smaller than", str(ctx.exception).lower())

    def test_invalid_argument_types(self):
        with self.assertRaises(TypeError):
            PointerType(Type.F32, StorageClass.GLOBAL, alignment=16.0)  # type: ignore
        with self.assertRaises(TypeError):
            PointerType(None, StorageClass.GLOBAL)  # type: ignore
        with self.assertRaises(TypeError):
            PointerType(Type.F32, "not_a_storage_class")  # type: ignore


@unittest.skipUnless(HAS_GFX_OPT, "gfx-opt tool required")
class TestPointerMLIRRoundTrip(unittest.TestCase):
    """Tier 1 & 2: 100% Unmocked integration tests invoking gfx-opt."""

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
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "types.mlir"
            path.write_text(mlir_content, encoding="utf-8")
            res = subprocess.run([str(GFX_OPT), str(path)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertIn("!gfx.ptr<f32, global>", res.stdout)
            self.assertIn("!gfx.ptr<vector<4xf32>, workgroup>", res.stdout)
            self.assertIn("!gfx.ptr<f32, global, 16>", res.stdout)

    def test_subnatural_alignment_rejected_by_compiler(self):
        mlir_content = """module {
  func.func private @bad(!gfx.ptr<vector<4xf32>, global, 8>)
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.mlir"
            path.write_text(mlir_content, encoding="utf-8")
            res = subprocess.run([str(GFX_OPT), str(path)], capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("smaller than the natural alignment", res.stderr)

    def test_npot_alignment_rejected_by_compiler(self):
        mlir_content = """module {
  func.func private @bad(!gfx.ptr<f32, global, 6>)
}
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "bad.mlir"
            path.write_text(mlir_content, encoding="utf-8")
            res = subprocess.run([str(GFX_OPT), str(path)], capture_output=True, text=True)
            self.assertNotEqual(res.returncode, 0)
            self.assertIn("power of two", res.stderr)


@unittest.skipUnless(HAS_GFX_COMPILE, "gfx-compile tool required")
class TestCrossTargetPointerLowering(unittest.TestCase):
    """Tier 3: 100% Unmocked compilation and capability-gated lowering."""

    COMPUTE_DESCRIPTOR_MLIR = """module {
  func.func @compute_main(%buf: !gfx.ptr<f32, global>) attributes {
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

    def test_vulkan_default_descriptor_lowering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "desc.mlir"
            out = Path(tmpdir) / "out.spv"
            refl = Path(tmpdir) / "out.json"
            src.write_text(self.COMPUTE_DESCRIPTOR_MLIR, encoding="utf-8")

            res = subprocess.run([
                str(GFX_COMPILE), str(src),
                "--input-format=mlir",
                "--target=vulkan",
                "--output", str(out),
                "--reflection", str(refl)
            ], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, msg=res.stderr)
            self.assertTrue(out.is_file())

    def test_vulkan_bda_capability_gating(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "bda.mlir"
            out = Path(tmpdir) / "out.spv"
            refl = Path(tmpdir) / "out.json"
            src.write_text(self.COMPUTE_BDA_MLIR, encoding="utf-8")

            # Without PhysicalStorageBuffer64 capability: MUST FAIL
            res_fail = subprocess.run([
                str(GFX_COMPILE), str(src),
                "--input-format=mlir",
                "--target=vulkan",
                "--output", str(out),
                "--reflection", str(refl)
            ], capture_output=True, text=True)
            self.assertNotEqual(res_fail.returncode, 0)
            self.assertIn("PhysicalStorageBuffer64", res_fail.stderr)

            # With PhysicalStorageBuffer64 capability: MUST SUCCEED
            res_pass = subprocess.run([
                str(GFX_COMPILE), str(src),
                "--input-format=mlir",
                "--target=vulkan",
                "--required-capability=PhysicalStorageBuffer64",
                "--output", str(out),
                "--reflection", str(refl)
            ], capture_output=True, text=True)
            self.assertEqual(res_pass.returncode, 0, msg=res_pass.stderr)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
