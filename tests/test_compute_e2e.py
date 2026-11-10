"""End-to-end compute kernel compilation and reflection integration tests across Vulkan SPIR-V and DirectX DXIL."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

from mlir_gpu import Buffer, Target, Toolchain, Type, compute, global_invocation_id, vec4
from mlir_gpu.compiler import CompileError, compile_pipeline
from test_support import ROOT, TOOLS

COMPUTE_TOOLS = TOOLS


def assert_valid_dxil(test_case: unittest.TestCase, dxil_path: Path, dxv_exe: str | None = None) -> None:
    """Assert that a DXIL file exists, is non-empty, has b'DXBC' magic, and validates with dxv."""
    test_case.assertTrue(dxil_path.is_file(), f"DXIL file not found: {dxil_path}")
    test_case.assertGreater(dxil_path.stat().st_size, 0, f"DXIL file is empty: {dxil_path}")
    raw_bytes = dxil_path.read_bytes()
    test_case.assertTrue(raw_bytes.startswith(b"DXBC"), f"DXIL file {dxil_path.name} missing DXBC magic header")
    if dxv_exe and Path(dxv_exe).is_file():
        with tempfile.TemporaryDirectory() as td:
            val_out = Path(td) / "val.dxil"
            res = subprocess.run([dxv_exe, str(dxil_path), "-o", str(val_out)], capture_output=True, text=True)
            combined = res.stdout + "\n" + res.stderr
            test_case.assertEqual(
                res.returncode, 0, f"dxv validation failed on {dxil_path.name} (code {res.returncode}):\n{combined}"
            )
            test_case.assertIn("dxv: succeeded", combined, f"dxv did not report success on {dxil_path.name}:\n{combined}")


def assert_valid_spirv(test_case: unittest.TestCase, spv_path: Path, spirv_val_exe: str | None = None) -> None:
    """Assert that a SPIR-V file exists, is non-empty, has SPIR-V magic header, and validates with spirv-val."""
    test_case.assertTrue(spv_path.is_file(), f"SPIR-V file not found: {spv_path}")
    test_case.assertGreater(spv_path.stat().st_size, 0, f"SPIR-V file is empty: {spv_path}")
    raw_bytes = spv_path.read_bytes()
    test_case.assertTrue(
        raw_bytes.startswith(b"\x03\x02\x23\x07"), f"SPIR-V file {spv_path.name} missing magic header"
    )
    if spirv_val_exe and Path(spirv_val_exe).is_file():
        res = subprocess.run(
            [spirv_val_exe, "--target-env", "vulkan1.0", str(spv_path)], capture_output=True, text=True
        )
        combined = res.stdout + "\n" + res.stderr
        test_case.assertEqual(
            res.returncode, 0, f"spirv-val failed on {spv_path.name} (code {res.returncode}):\n{combined}"
        )


class TestComputeE2EIntegration(unittest.TestCase):
    """End-to-end compilation and validation of compute kernels across Vulkan and DirectX."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.dxv = COMPUTE_TOOLS.find("dxv")
        except CompileError:
            cls.dxv = None
        try:
            cls.spirv_val = COMPUTE_TOOLS.find("spirv-val")
        except CompileError:
            cls.spirv_val = None

    def test_vector_add_compiles_to_vulkan_spv_and_validates(self):
        @compute(workgroup_size=(64, 1, 1))
        def vector_add(
            a: Buffer[Type.F32, "read_only"],
            b: Buffer[Type.F32, "read_only"],
            c: Buffer[Type.F32, "write_only"],
        ):
            idx = global_invocation_id().x
            c[idx] = a[idx] + b[idx]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "vector_add.spv"
            refl_out = Path(td) / "vector_add.json"
            manifest = compile_pipeline(
                vector_add, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_out
            )
            assert_valid_spirv(self, spv_out, self.spirv_val)
            self.assertEqual(manifest["target"], "vulkan")
            self.assertEqual(manifest["stage"], "compute")
            self.assertEqual(manifest["entry_point"], "vector_add")
            self.assertEqual(manifest["binary"], str(spv_out))

            self.assertTrue(refl_out.is_file())
            refl = json.loads(refl_out.read_text(encoding="utf-8"))
            self.assertEqual(refl["stage"], "compute")
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertEqual(len(bufs), 3)
            self.assertEqual(bufs["a"]["access"], "read_only")
            self.assertEqual(bufs["a"]["element_type"], "f32")
            self.assertEqual(bufs["b"]["access"], "read_only")
            self.assertEqual(bufs["b"]["element_type"], "f32")
            self.assertEqual(bufs["c"]["access"], "write_only")
            self.assertEqual(bufs["c"]["element_type"], "f32")

    def test_vector_add_compiles_to_directx_dxil_and_validates(self):
        @compute(workgroup_size=(64, 1, 1))
        def vector_add(
            a: Buffer[Type.F32, "read_only"],
            b: Buffer[Type.F32, "read_only"],
            c: Buffer[Type.F32, "write_only"],
        ):
            idx = global_invocation_id().x
            c[idx] = a[idx] + b[idx]

        with tempfile.TemporaryDirectory() as td:
            dx_out = Path(td) / "vector_add.dxil"
            refl_out = Path(td) / "vector_add.json"
            manifest = compile_pipeline(
                vector_add, Target.DIRECTX, dx_out, COMPUTE_TOOLS, reflection_path=refl_out
            )
            assert_valid_dxil(self, dx_out, self.dxv)
            self.assertEqual(manifest["target"], "directx")
            self.assertEqual(manifest["stage"], "compute")
            self.assertEqual(manifest["entry_point"], "vector_add")
            self.assertEqual(manifest["binary"], str(dx_out))

            self.assertTrue(refl_out.is_file())
            refl = json.loads(refl_out.read_text(encoding="utf-8"))
            self.assertEqual(refl["stage"], "compute")
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertEqual(len(bufs), 3)
            self.assertEqual(bufs["a"]["access"], "read_only")
            self.assertEqual(bufs["b"]["access"], "read_only")
            self.assertEqual(bufs["c"]["access"], "write_only")

    def test_vector_add_compiles_to_directx_ll_intermediate(self):
        @compute(workgroup_size=(64, 1, 1))
        def vector_add(
            a: Buffer[Type.F32, "read_only"],
            b: Buffer[Type.F32, "read_only"],
            c: Buffer[Type.F32, "write_only"],
        ):
            idx = global_invocation_id().x
            c[idx] = a[idx] + b[idx]

        with tempfile.TemporaryDirectory() as td:
            dx_out = Path(td) / "vector_add.ll"
            compile_pipeline(vector_add, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            self.assertTrue(dx_out.is_file())
            self.assertGreater(dx_out.stat().st_size, 0)
            dx_content = dx_out.read_text(encoding="utf-8")
            self.assertIn('"hlsl.shader"="compute"', dx_content)
            self.assertIn('"hlsl.numthreads"', dx_content)
            self.assertIn("RootSRV", dx_content)
            self.assertIn("RootUAV", dx_content)

    def test_buffer_copy_vector4_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(64, 1, 1))
        def buffer_copy(
            src: Buffer[Type.VEC4, "read_only"],
            dst: Buffer[Type.VEC4, "write_only"],
        ):
            idx = global_invocation_id().x
            dst[idx] = src[idx]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "copy.spv"
            refl_spv = Path(td) / "copy.spv.json"
            compile_pipeline(buffer_copy, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            refl = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertEqual(bufs["src"]["element_type"], "vector<4xf32>")
            self.assertEqual(bufs["dst"]["element_type"], "vector<4xf32>")

            dx_out = Path(td) / "copy.dxil"
            refl_dx = Path(td) / "copy.dx.json"
            compile_pipeline(buffer_copy, Target.DIRECTX, dx_out, COMPUTE_TOOLS, reflection_path=refl_dx)
            assert_valid_dxil(self, dx_out, self.dxv)
            refl_d = json.loads(refl_dx.read_text(encoding="utf-8"))
            bufs_d = {b["name"]: b for b in refl_d["buffers"]}
            self.assertEqual(bufs_d["src"]["element_type"], "vector<4xf32>")
            self.assertEqual(bufs_d["dst"]["element_type"], "vector<4xf32>")

    def test_vector4_arithmetic_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(64, 1, 1))
        def vector4_saxpy(
            x: Buffer[Type.VEC4, "read_only"],
            y: Buffer[Type.VEC4, "read_only"],
            res: Buffer[Type.VEC4, "write_only"],
        ):
            idx = global_invocation_id().x
            res[idx] = x[idx] * 2.0 + y[idx]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "saxpy.spv"
            compile_pipeline(vector4_saxpy, Target.VULKAN, spv_out, COMPUTE_TOOLS)
            assert_valid_spirv(self, spv_out, self.spirv_val)

            dx_out = Path(td) / "saxpy.dxil"
            compile_pipeline(vector4_saxpy, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_multi_buffer_pipeline_mixed_access_modes(self):
        @compute(workgroup_size=(64, 1, 1))
        def multi_buf_pipeline(
            in_a: Buffer[Type.F32, "read_only"],
            in_b: Buffer[Type.F32, "read_only"],
            accum: Buffer[Type.F32, "read_write"],
            out_c: Buffer[Type.F32, "write_only"],
        ):
            idx = global_invocation_id().x
            val = in_a[idx] * 2.0 + in_b[idx]
            accum[idx] = accum[idx] + val
            out_c[idx] = accum[idx] * 0.5

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "multi.spv"
            refl_spv = Path(td) / "multi.spv.json"
            compile_pipeline(multi_buf_pipeline, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            refl = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertEqual(len(bufs), 4)
            self.assertEqual(bufs["in_a"]["access"], "read_only")
            self.assertEqual(bufs["in_b"]["access"], "read_only")
            self.assertEqual(bufs["accum"]["access"], "read_write")
            self.assertEqual(bufs["out_c"]["access"], "write_only")

            dx_out = Path(td) / "multi.dxil"
            refl_dx = Path(td) / "multi.dx.json"
            compile_pipeline(multi_buf_pipeline, Target.DIRECTX, dx_out, COMPUTE_TOOLS, reflection_path=refl_dx)
            assert_valid_dxil(self, dx_out, self.dxv)
            refl_d = json.loads(refl_dx.read_text(encoding="utf-8"))
            bufs_d = {b["name"]: b for b in refl_d["buffers"]}
            self.assertEqual(len(bufs_d), 4)
            self.assertEqual(bufs_d["in_a"]["access"], "read_only")
            self.assertEqual(bufs_d["in_b"]["access"], "read_only")
            self.assertEqual(bufs_d["accum"]["access"], "read_write")
            self.assertEqual(bufs_d["out_c"]["access"], "write_only")

    def test_integer_buffers_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(64, 1, 1))
        def u32_kernel(
            in_ids: Buffer[Type.U32, "read_only"],
            counts: Buffer[Type.U32, "read_write"],
            out_ids: Buffer[Type.U32, "write_only"],
        ):
            idx = global_invocation_id().x
            val = in_ids[idx]
            counts[idx] = counts[idx] + 1
            out_ids[idx] = val + counts[idx]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "u32_k.spv"
            refl_spv = Path(td) / "u32_k.json"
            compile_pipeline(u32_kernel, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            refl = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertEqual(bufs["in_ids"]["element_type"], "i32")

            dx_out = Path(td) / "u32_k.dxil"
            compile_pipeline(u32_kernel, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_i32_buffer_copy_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(64, 1, 1))
        def i32_copy(
            in_ids: Buffer[Type.I32, "read_only"],
            out_ids: Buffer[Type.I32, "write_only"],
        ):
            idx = global_invocation_id().x
            out_ids[idx] = in_ids[idx]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "i32_copy.spv"
            refl_spv = Path(td) / "i32_copy.json"
            compile_pipeline(i32_copy, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            refl = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertIn(bufs["in_ids"]["element_type"], ("i32", "si32"))

            dx_out = Path(td) / "i32_copy.dxil"
            compile_pipeline(i32_copy, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_i32_lookup_table_indirect_indexing_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(64, 1, 1))
        def lut_gather(
            indices: Buffer[Type.I32, "read_only"],
            lut: Buffer[Type.F32, "read_only"],
            out: Buffer[Type.F32, "write_only"],
        ):
            idx = global_invocation_id().x
            entry = indices[idx]
            out[idx] = lut[entry]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "lut_gather.spv"
            refl_spv = Path(td) / "lut_gather.spv.json"
            compile_pipeline(lut_gather, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            refl_v = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs_v = {b["name"]: b for b in refl_v["buffers"]}
            self.assertIn(bufs_v["indices"]["element_type"], ("i32", "si32"))
            self.assertEqual(bufs_v["lut"]["element_type"], "f32")
            self.assertEqual(bufs_v["out"]["element_type"], "f32")

            dx_out = Path(td) / "lut_gather.dxil"
            refl_dx = Path(td) / "lut_gather.dx.json"
            compile_pipeline(lut_gather, Target.DIRECTX, dx_out, COMPUTE_TOOLS, reflection_path=refl_dx)
            assert_valid_dxil(self, dx_out, self.dxv)
            refl_d = json.loads(refl_dx.read_text(encoding="utf-8"))
            bufs_d = {b["name"]: b for b in refl_d["buffers"]}
            self.assertIn(bufs_d["indices"]["element_type"], ("i32", "si32"))
            self.assertEqual(bufs_d["lut"]["element_type"], "f32")
            self.assertEqual(bufs_d["out"]["element_type"], "f32")

    def test_i32_multilevel_indirect_subscript_pipeline(self):
        @compute(workgroup_size=(64, 1, 1))
        def multi_indirect(
            map_a: Buffer[Type.I32, "read_only"],
            map_b: Buffer[Type.I32, "read_only"],
            data: Buffer[Type.F32, "read_only"],
            result: Buffer[Type.F32, "write_only"],
        ):
            gid = global_invocation_id()
            step1 = map_a[gid.x]
            step2 = map_b[step1]
            result[gid.x] = data[step2]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "multi_ind.spv"
            compile_pipeline(multi_indirect, Target.VULKAN, spv_out, COMPUTE_TOOLS)
            assert_valid_spirv(self, spv_out, self.spirv_val)

            dx_out = Path(td) / "multi_ind.dxil"
            compile_pipeline(multi_indirect, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_i32_buffer_arithmetic_and_accumulate(self):
        @compute(workgroup_size=(64, 1, 1))
        def i32_accum(
            in_offsets: Buffer[Type.I32, "read_only"],
            accum: Buffer[Type.I32, "read_write"],
            out_offsets: Buffer[Type.I32, "write_only"],
        ):
            idx = global_invocation_id().x
            val = in_offsets[idx]
            accum[idx] = accum[idx] + val
            out_offsets[idx] = accum[idx]

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "i32_accum.spv"
            compile_pipeline(i32_accum, Target.VULKAN, spv_out, COMPUTE_TOOLS)
            assert_valid_spirv(self, spv_out, self.spirv_val)

            dx_out = Path(td) / "i32_accum.dxil"
            compile_pipeline(i32_accum, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_workgroup_2d_dimensions_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(16, 16, 1))
        def kernel_2d(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
        ):
            gid = global_invocation_id()
            idx = gid.y * 16 + gid.x
            dst[idx] = src[idx]

        self.assertEqual(kernel_2d.workgroup_size, (16, 16, 1))

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "k2d.spv"
            compile_pipeline(kernel_2d, Target.VULKAN, spv_out, COMPUTE_TOOLS)
            assert_valid_spirv(self, spv_out, self.spirv_val)

            dx_out = Path(td) / "k2d.dxil"
            compile_pipeline(kernel_2d, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_workgroup_3d_dimensions_compiles_to_vulkan_and_directx(self):
        @compute(workgroup_size=(8, 8, 4))
        def kernel_3d(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
        ):
            gid = global_invocation_id()
            idx = gid.z * 64 + gid.y * 8 + gid.x
            dst[idx] = src[idx]

        self.assertEqual(kernel_3d.workgroup_size, (8, 8, 4))

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "k3d.spv"
            compile_pipeline(kernel_3d, Target.VULKAN, spv_out, COMPUTE_TOOLS)
            assert_valid_spirv(self, spv_out, self.spirv_val)

            dx_out = Path(td) / "k3d.dxil"
            compile_pipeline(kernel_3d, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_unreferenced_buffers_preserved_across_compilation(self):
        @compute(workgroup_size=(64, 1, 1))
        def partial_unused(
            active_in: Buffer[Type.F32, "read_only"],
            unused_ro: Buffer[Type.I32, "read_only"],
            unused_wo: Buffer[Type.VEC4, "write_only"],
            active_out: Buffer[Type.F32, "write_only"],
        ):
            idx = global_invocation_id().x
            active_out[idx] = active_in[idx] * 3.0

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "unused.spv"
            refl_spv = Path(td) / "unused.spv.json"
            compile_pipeline(partial_unused, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            refl = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs = {b["name"]: b for b in refl["buffers"]}
            self.assertEqual(len(bufs), 4)
            self.assertIn("active_in", bufs)
            self.assertIn("unused_ro", bufs)
            self.assertIn("unused_wo", bufs)
            self.assertIn("active_out", bufs)

            dx_out = Path(td) / "unused.dxil"
            refl_dx = Path(td) / "unused.dx.json"
            compile_pipeline(partial_unused, Target.DIRECTX, dx_out, COMPUTE_TOOLS, reflection_path=refl_dx)
            assert_valid_dxil(self, dx_out, self.dxv)
            refl_d = json.loads(refl_dx.read_text(encoding="utf-8"))
            bufs_d = {b["name"]: b for b in refl_d["buffers"]}
            self.assertEqual(len(bufs_d), 4)

    def test_all_unused_buffers_compile_and_preserve_reflection(self):
        @compute(workgroup_size=(64, 1, 1))
        def all_unused(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.VEC4, "write_only"],
            acc: Buffer[Type.U32, "read_write"],
        ):
            pass

        with tempfile.TemporaryDirectory() as td:
            spv_out = Path(td) / "all_unused.spv"
            refl_spv = Path(td) / "all_unused.spv.json"
            compile_pipeline(all_unused, Target.VULKAN, spv_out, COMPUTE_TOOLS, reflection_path=refl_spv)
            assert_valid_spirv(self, spv_out, self.spirv_val)
            data = json.loads(refl_spv.read_text(encoding="utf-8"))
            bufs = {b["name"]: b for b in data["buffers"]}
            self.assertEqual(len(bufs), 3)

            dx_out = Path(td) / "all_unused.dxil"
            compile_pipeline(all_unused, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)

    def test_illegal_buffer_writes_rejected_at_compile_time(self):
        @compute(workgroup_size=(32, 1, 1))
        def bad_write_direct(ro: Buffer[Type.F32, "read_only"]):
            ro[0] = 1.0

        with self.assertRaisesRegex(TypeError, r"Cannot write to read-only buffer 'ro'"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(bad_write_direct, Target.VULKAN, Path(td) / "bad.spv", COMPUTE_TOOLS)

        @compute(workgroup_size=(32, 1, 1))
        def bad_write_augmented(ro: Buffer[Type.F32, "read_only"]):
            ro[0] += 1.0

        with self.assertRaisesRegex(TypeError, r"Cannot write to read-only buffer 'ro'"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(bad_write_augmented, Target.DIRECTX, Path(td) / "bad.dxil", COMPUTE_TOOLS)

    def test_illegal_buffer_reads_rejected_at_compile_time(self):
        @compute(workgroup_size=(32, 1, 1))
        def bad_read_direct(wo: Buffer[Type.F32, "write_only"]):
            val = wo[0]

        with self.assertRaisesRegex(TypeError, r"Cannot read from write-only buffer 'wo'"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(bad_read_direct, Target.VULKAN, Path(td) / "bad.spv", COMPUTE_TOOLS)

        @compute(workgroup_size=(32, 1, 1))
        def bad_read_augmented(wo: Buffer[Type.F32, "write_only"]):
            wo[0] += 1.0

        with self.assertRaisesRegex(TypeError, r"Cannot read from write-only buffer 'wo'"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(bad_read_augmented, Target.DIRECTX, Path(td) / "bad.dxil", COMPUTE_TOOLS)

    def test_buffer_type_mismatch_rejected_at_compile_time(self):
        @compute(workgroup_size=(32, 1, 1))
        def type_mismatch(dst: Buffer[Type.F32, "write_only"]):
            dst[0] = vec4(1.0, 2.0, 3.0, 4.0)

        with self.assertRaisesRegex(TypeError, r"Type mismatch: cannot store vector<4xf32> into Buffer\[f32\]"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(type_mismatch, Target.DIRECTX, Path(td) / "bad.dxil", COMPUTE_TOOLS)

    def test_non_integer_buffer_subscript_rejected_at_compile_time(self):
        @compute(workgroup_size=(32, 1, 1))
        def bad_subscript_store(buf: Buffer[Type.F32, "read_write"]):
            idx = buf[0]
            buf[idx] = 2.0

        with self.assertRaisesRegex(TypeError, r"Buffer subscript index must be an integer, got Type\.F32"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(bad_subscript_store, Target.DIRECTX, Path(td) / "bad.dxil", COMPUTE_TOOLS)

        @compute(workgroup_size=(32, 1, 1))
        def bad_subscript_load(buf: Buffer[Type.F32, "read_write"]):
            idx = buf[0]
            val = buf[idx]

        with self.assertRaisesRegex(TypeError, r"Buffer subscript index must be an integer, got Type\.F32"):
            with tempfile.TemporaryDirectory() as td:
                compile_pipeline(bad_subscript_load, Target.DIRECTX, Path(td) / "bad.dxil", COMPUTE_TOOLS)

    def test_directx_64bit_offset_buffer_gather_dxil_and_validation(self):
        @dataclass(frozen=True)
        class MLIRComputePipeline:
            source: Path
            entry_point: str = "compute_main"
            kind: str = "compute"

            def trace(self):
                from mlir_gpu.compiler import _MLIRStage
                return _MLIRStage("compute", self.entry_point, Path(self.source))

        cases = [
            ("i64", "f32"),
            ("si64", "vector<4xf32>"),
            ("ui64", "i32"),
        ]

        for off_t, elem_t in cases:
            with self.subTest(offset_type=off_t, element_type=elem_t):
                mlir_text = f"""module {{
  func.func @compute_main() attributes {{
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  }} {{
    %data = gfx.buffer_binding 0, 0, "data", #gfx.access_mode<read_write> : !gfx.ptr<{elem_t}, global>
    %idx_buf = gfx.buffer_binding 0, 1, "idx_buf", #gfx.access_mode<read_only> : !gfx.ptr<{off_t}, global>
    %c0 = arith.constant 0 : i32
    %idx_ptr = gfx.ptr_offset %idx_buf, %c0 : (!gfx.ptr<{off_t}, global>, i32) -> !gfx.ptr<{off_t}, global>
    %idx = gfx.load %idx_ptr : (!gfx.ptr<{off_t}, global>) -> {off_t}
    %elem_ptr = gfx.ptr_offset %data, %idx : (!gfx.ptr<{elem_t}, global>, {off_t}) -> !gfx.ptr<{elem_t}, global>
    %val = gfx.load %elem_ptr : (!gfx.ptr<{elem_t}, global>) -> {elem_t}
    gfx.store %val, %elem_ptr : {elem_t}, !gfx.ptr<{elem_t}, global>
    return
  }}
}}"""
                with tempfile.TemporaryDirectory() as td:
                    src_mlir = Path(td) / "shader64.mlir"
                    src_mlir.write_text(mlir_text)
                    dx_out = Path(td) / "shader64.dxil"
                    refl_out = Path(td) / "shader64.json"
                    pipeline = MLIRComputePipeline(src_mlir)
                    manifest = compile_pipeline(
                        pipeline, Target.DIRECTX, dx_out, COMPUTE_TOOLS, reflection_path=refl_out
                    )
                    assert_valid_dxil(self, dx_out, self.dxv)
                    self.assertEqual(manifest["target"], "directx")
                    self.assertEqual(manifest["stage"], "compute")

    def test_directx_64bit_constant_offset_compiles_and_validates(self):
        @dataclass(frozen=True)
        class MLIRComputePipeline:
            source: Path
            entry_point: str = "compute_main"
            kind: str = "compute"

            def trace(self):
                from mlir_gpu.compiler import _MLIRStage
                return _MLIRStage("compute", self.entry_point, Path(self.source))

        mlir_text = """module {
  func.func @compute_main() attributes {
    gfx.stage = #gfx.stage<compute>,
    gfx.workgroup_size = array<i32: 64, 1, 1>
  } {
    %bufA = gfx.buffer_binding 0, 0, "bufA", #gfx.access_mode<read_write> : !gfx.ptr<vector<4xf32>, global>
    %c7 = arith.constant 7 : i64
    %elem_ptr = gfx.ptr_offset %bufA, %c7 : (!gfx.ptr<vector<4xf32>, global>, i64) -> !gfx.ptr<vector<4xf32>, global>
    %val = gfx.load %elem_ptr : (!gfx.ptr<vector<4xf32>, global>) -> vector<4xf32>
    gfx.store %val, %elem_ptr : vector<4xf32>, !gfx.ptr<vector<4xf32>, global>
    return
  }
}"""
        with tempfile.TemporaryDirectory() as td:
            src_mlir = Path(td) / "const64.mlir"
            src_mlir.write_text(mlir_text)
            dx_out = Path(td) / "const64.dxil"
            pipeline = MLIRComputePipeline(src_mlir)
            compile_pipeline(pipeline, Target.DIRECTX, dx_out, COMPUTE_TOOLS)
            assert_valid_dxil(self, dx_out, self.dxv)


if __name__ == "__main__":
    unittest.main()
