"""Unit tests for Buffer[T, access] type annotations, subscript loads/stores, and access diagnostics."""

from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from mlir_gpu import Type, compute, f32, global_invocation_id, vec4
from mlir_gpu.frontend import AccessMode, Buffer, PointerType, StorageClass


class TestBufferTypeModel(unittest.TestCase):
    """Buffer[T, access] construction, parameterization, and immutability."""

    def test_buffer_valid_access_modes(self):
        b_ro = Buffer[Type.F32, "read_only"]
        self.assertEqual(b_ro.element_type, Type.F32)
        self.assertEqual(b_ro.access, AccessMode.READ_ONLY)
        self.assertEqual(b_ro.storage_class, StorageClass.GLOBAL)

        b_wo = Buffer[Type.VEC4, "write_only"]
        self.assertEqual(b_wo.element_type, Type.VEC4)
        self.assertEqual(b_wo.access, AccessMode.WRITE_ONLY)

        b_rw = Buffer[Type.I32, "read_write"]
        self.assertEqual(b_rw.element_type, Type.I32)
        self.assertEqual(b_rw.access, AccessMode.READ_WRITE)

    def test_buffer_default_access_mode(self):
        b = Buffer[Type.F32]
        self.assertEqual(b.access, AccessMode.READ_WRITE)

    def test_buffer_string_representations(self):
        self.assertEqual(str(Buffer[Type.F32, "read_only"]), "Buffer[f32, read_only]")
        self.assertEqual(str(Buffer[Type.VEC4, "write_only"]), "Buffer[vector<4xf32>, write_only]")
        self.assertEqual(str(Buffer[Type.I32, "read_write"]), "Buffer[si32, read_write]")

    def test_buffer_immutability_and_hashing(self):
        b1 = Buffer[Type.F32, "read_only"]
        b2 = Buffer[Type.F32, "read_only"]
        b3 = Buffer[Type.F32, "write_only"]
        self.assertEqual(b1, b2)
        self.assertNotEqual(b1, b3)
        self.assertEqual(hash(b1), hash(b2))

        with self.assertRaises((FrozenInstanceError, AttributeError)):
            b1.access = AccessMode.WRITE_ONLY  # type: ignore

    def test_buffer_invalid_access_mode_rejected(self):
        for invalid in ("readonly", "READ_ONLY", "read", "write", "rw", "none", "", None, 123):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    Buffer[Type.F32, invalid]  # type: ignore

    def test_buffer_invalid_element_type_rejected(self):
        for invalid in (None, 42, "unknown_type", "!gfx.ptr<f32, global>"):
            with self.subTest(invalid=invalid):
                with self.assertRaises((TypeError, ValueError)):
                    Buffer[invalid, "read_only"]  # type: ignore


class TestKernelSignatureGeneration(unittest.TestCase):
    """Kernel parameter reflection, workgroup size normalization, and buffer binding assignments."""

    def test_kernel_signature_binding_assignment(self):
        @compute(workgroup_size=(64, 1, 1))
        def kernel(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
        ):
            pass

        bindings = kernel.reflected_buffer_bindings
        self.assertEqual(len(bindings), 2)
        self.assertEqual(bindings[0].name, "src")
        self.assertEqual(bindings[0].set_index, 0)
        self.assertEqual(bindings[0].binding_index, 0)
        self.assertEqual(bindings[0].access, AccessMode.READ_ONLY)
        self.assertEqual(bindings[0].pointer_type, PointerType(Type.F32, StorageClass.GLOBAL))

        self.assertEqual(bindings[1].name, "dst")
        self.assertEqual(bindings[1].set_index, 0)
        self.assertEqual(bindings[1].binding_index, 1)
        self.assertEqual(bindings[1].access, AccessMode.WRITE_ONLY)
        self.assertEqual(bindings[1].pointer_type, PointerType(Type.F32, StorageClass.GLOBAL))

    def test_mixed_explicit_and_auto_bindings(self):
        @compute
        def kernel(
            a: Buffer[f32, "read_only"],
            b: Buffer[f32, "read_only", 5],
            c: Buffer[f32, "write_only"],
        ):
            pass

        bindings = kernel.reflected_buffer_bindings
        self.assertEqual(len(bindings), 3)
        self.assertEqual(bindings[0].binding_index, 0)
        self.assertEqual(bindings[1].binding_index, 5)
        self.assertEqual(bindings[2].binding_index, 6)

    def test_workgroup_size_normalization(self):
        @compute(64)
        def k1(b: Buffer[f32]):
            pass

        self.assertEqual(k1.workgroup_size, (64, 1, 1))

        @compute((32,))
        def k2(b: Buffer[f32]):
            pass

        self.assertEqual(k2.workgroup_size, (32, 1, 1))

        @compute((16, 8))
        def k3(b: Buffer[f32]):
            pass

        self.assertEqual(k3.workgroup_size, (16, 8, 1))

        @compute((8, 4, 2))
        def k4(b: Buffer[f32]):
            pass

        self.assertEqual(k4.workgroup_size, (8, 4, 2))

        with self.assertRaises(TypeError):
            @compute((8, 4, 2, 1))
            def k5(b: Buffer[f32]):
                pass


class TestASTSubscriptCompilation(unittest.TestCase):
    """AST parsing of buffer subscript expressions to ptr_offset and load/store operations."""

    def test_ast_subscript_read_emits_offset_and_load(self):
        @compute(workgroup_size=(64, 1, 1))
        def read_kernel(src: Buffer[Type.F32, "read_only"]):
            idx = global_invocation_id().x
            x = src[idx]
            return x

        stage = read_kernel.trace()
        ops = [op["op"] for op in stage.structured["functions"][0]["body"]["blocks"][0]["operations"]]
        self.assertIn("gfx.ptr_offset", ops)
        self.assertIn("gfx.load", ops)

    def test_ast_subscript_write_emits_offset_and_store(self):
        @compute(workgroup_size=(64, 1, 1))
        def write_kernel(dst: Buffer[Type.F32, "write_only"]):
            idx = global_invocation_id().x
            dst[idx] = 42.0

        stage = write_kernel.trace()
        ops = [op["op"] for op in stage.structured["functions"][0]["body"]["blocks"][0]["operations"]]
        self.assertIn("gfx.ptr_offset", ops)
        self.assertIn("gfx.store", ops)

    def test_ast_subscript_read_write_compound_update(self):
        @compute(workgroup_size=(64, 1, 1))
        def update_kernel(buf: Buffer[Type.F32, "read_write"]):
            idx = global_invocation_id().x
            buf[idx] = buf[idx] + 1.0

        stage = update_kernel.trace()
        ops = [op["op"] for op in stage.structured["functions"][0]["body"]["blocks"][0]["operations"]]
        self.assertGreaterEqual(ops.count("gfx.ptr_offset"), 1)
        self.assertIn("gfx.load", ops)
        self.assertIn("gfx.store", ops)

    def test_ast_subscript_augmented_assignments(self):
        @compute(workgroup_size=(64, 1, 1))
        def math_kernel(b: Buffer[f32, "read_write"]):
            idx = global_invocation_id().x
            b[idx] *= 2.0
            b[idx] -= 1.0
            b[idx] /= 4.0

        stage = math_kernel.trace()
        ops = [op["op"] for op in stage.structured["functions"][0]["body"]["blocks"][0]["operations"]]
        self.assertEqual(ops.count("gfx.load"), 3)
        self.assertEqual(ops.count("gfx.store"), 3)


class TestAccessPermissionViolations(unittest.TestCase):
    """Compile-time rejection of invalid buffer access permissions and subscript expressions."""

    def test_reject_store_to_read_only_buffer(self):
        @compute(workgroup_size=(64, 1, 1))
        def invalid_store(src: Buffer[Type.F32, "read_only"]):
            idx = global_invocation_id().x
            src[idx] = 1.0

        with self.assertRaisesRegex(TypeError, r":\d+:\d+: Cannot write to read-only buffer 'src'"):
            invalid_store.trace()

    def test_reject_load_from_write_only_buffer(self):
        @compute(workgroup_size=(64, 1, 1))
        def invalid_load(dst: Buffer[Type.F32, "write_only"]):
            idx = global_invocation_id().x
            x = dst[idx]

        with self.assertRaisesRegex(TypeError, r":\d+:\d+: Cannot read from write-only buffer 'dst'"):
            invalid_load.trace()

    def test_reject_compound_in_place_on_read_only(self):
        @compute(workgroup_size=(64, 1, 1))
        def invalid_update(src: Buffer[Type.F32, "read_only"]):
            idx = global_invocation_id().x
            src[idx] += 1.0

        with self.assertRaisesRegex(TypeError, r"Cannot write to read-only buffer 'src'"):
            invalid_update.trace()

    def test_reject_compound_in_place_on_write_only(self):
        @compute(workgroup_size=(64, 1, 1))
        def invalid_update(dst: Buffer[Type.F32, "write_only"]):
            idx = global_invocation_id().x
            dst[idx] += 1.0

        with self.assertRaisesRegex(TypeError, r"Cannot read from write-only buffer 'dst'"):
            invalid_update.trace()

    def test_reject_store_type_mismatch(self):
        @compute(workgroup_size=(64, 1, 1))
        def bad_type(dst: Buffer[Type.F32, "write_only"]):
            idx = global_invocation_id().x
            dst[idx] = vec4(1.0, 2.0, 3.0, 4.0)

        with self.assertRaisesRegex(TypeError, r"Type mismatch: cannot store vector<4xf32> into Buffer\[f32\]"):
            bad_type.trace()

    def test_reject_buffer_slice_subscript(self):
        @compute(workgroup_size=(64, 1, 1))
        def bad_slice(b: Buffer[f32, "read_only"]):
            x = b[0:4]

        with self.assertRaises(TypeError):
            bad_slice.trace()

    def test_reject_read_from_write_only_in_expression(self):
        @compute(workgroup_size=(64, 1, 1))
        def bad_expr_read(
            wo: Buffer[f32, "write_only"],
            ro: Buffer[f32, "read_only"],
        ):
            x = ro[0] + wo[0]

        with self.assertRaisesRegex(TypeError, r"Cannot read from write-only buffer 'wo'"):
            bad_expr_read.trace()

    def test_reject_store_to_read_only_in_conditional(self):
        @compute(workgroup_size=(64, 1, 1))
        def bad_cond_write(ro: Buffer[f32, "read_only"]):
            idx = global_invocation_id().x
            if idx > 0:
                ro[idx] = 1.0

        with self.assertRaisesRegex(TypeError, r"Cannot write to read-only buffer 'ro'"):
            bad_cond_write.trace()


if __name__ == "__main__":
    unittest.main()
