import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import (
    Buffer, Pipeline, Target, compile_pipeline, compute, struct,
    vec2, vec3, vec4, mat4, f32, u32, i32,
    Vertex, fragment, vertex, ParameterBlock, Type,
)
from test_support import TOOLS


@struct
class PipelineCameraBlock:
    view_proj: mat4  # 64B
    eye: vec3        # 12B
    time: f32        # 4B -> 80B


@struct
class PipelineObjectBlock:
    model: mat4      # 64B
    color: vec4      # 16B
    scale: vec4      # 16B -> 96B


@struct
class PipelineLightBlock:
    direction: vec3  # 12B
    intensity: f32   # 4B
    color: vec3      # 12B
    ambient: f32     # 4B -> 32B


@struct
class PipelineVertexOnlyBlock:
    pos: vec4        # 16B


@struct
class PipelineMaterialBlock:
    albedo: vec4     # 16B
    roughness: f32   # 4B -> 32B (aligned)


@struct
class PipelineSimBlock:
    gravity: vec3    # 12B
    delta_time: f32  # 4B -> 16B


@vertex
def disjoint_vs_ab(idx: u32, cam: PipelineCameraBlock, obj: PipelineObjectBlock) -> Vertex:
    return Vertex(cam.eye.x * obj.color + obj.scale)


@fragment
def disjoint_fs_ac(cam: PipelineCameraBlock, light: PipelineLightBlock) -> vec4:
    return vec4(cam.time + light.intensity) * vec4(light.color, light.ambient)


class DisjointStageParameterRegisterTests(unittest.TestCase):
    def test_reconciled_push_constant_offsets_match_spirv_binary(self):
        pipeline = Pipeline(disjoint_vs_ab, disjoint_fs_ac)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out_vk"
            manifest_path = compile_pipeline(pipeline, Target.VULKAN, out_dir, TOOLS)
            manifest = json.loads(manifest_path.read_text())

            params = manifest["parameters"]
            offsets = {p["name"]: p["bindings"]["vulkan"]["offset"] for p in params}
            self.assertEqual(offsets["cam"], 0)
            self.assertEqual(offsets["obj"], 80)
            self.assertEqual(offsets["light"], 176)

            # Check fragment.spv member offsets
            res = subprocess.run(
                [sys.executable, "tests/lit/parameters/Inputs/summarize.py", "spirv", str(out_dir / "fragment.spv")],
                capture_output=True,
                text=True,
                check=True,
            )
            # Member offsets must include 176 for light block (and not start at 80)
            self.assertIn("176", res.stdout)
            self.assertRegex(res.stdout, r"176,\s*188,\s*192,\s*204")

    def test_reconciled_directx_registers_and_rootsignatures_match_dxil(self):
        from test_support import available
        if not available("llc", "dxv"):
            self.skipTest("DirectX compilation tools not available")
        pipeline = Pipeline(disjoint_vs_ab, disjoint_fs_ac)
        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out_dx"
            manifest_path = compile_pipeline(pipeline, Target.DIRECTX, out_dir, TOOLS)
            manifest = json.loads(manifest_path.read_text())

            params = manifest["parameters"]
            regs = {p["name"]: p["bindings"]["directx"]["register"] for p in params}
            self.assertEqual(regs["cam"], 0)
            self.assertEqual(regs["obj"], 1)
            self.assertEqual(regs["light"], 2)

            ll_fs = (out_dir / "fragment.ll").read_text()

            # Verify fragment accesses register b2 for light
            handle_regs = re.findall(r"@llvm\.dx\.resource\.handlefrombinding[^(]*\(i32 0,\s*i32\s*(\d+)", ll_fs)
            self.assertIn("2", handle_regs)
            self.assertNotIn("1", handle_regs)

            # Verify root signature contains register b2 with 8 dwords (32B / 4)
            # and register b0 with 20 dwords (80B / 4)
            self.assertRegex(ll_fs, r'!\{!"RootConstants",\s*i32\s*\d+,\s*i32\s*0,\s*i32\s*\d+,\s*i32\s*20\}')
            self.assertRegex(ll_fs, r'!\{!"RootConstants",\s*i32\s*\d+,\s*i32\s*2,\s*i32\s*\d+,\s*i32\s*8\}')

    def test_float_struct_cross_stage_layout_reconciliation_multi_block(self):
        @vertex
        def vs_local(idx: u32, vs_pos: PipelineVertexOnlyBlock, light: PipelineLightBlock) -> Vertex:
            return Vertex(vs_pos.pos + vec4(light.direction, light.intensity))

        @fragment
        def fs_local(light: PipelineLightBlock) -> vec4:
            return vec4(light.color, light.ambient)

        for target in (Target.VULKAN, Target.DIRECTX):
            from test_support import available
            if target == Target.DIRECTX and not available("llc", "dxv"):
                continue
            with self.subTest(target=target):
                with tempfile.TemporaryDirectory() as td:
                    out_dir = Path(td) / "out"
                    manifest_path = compile_pipeline(Pipeline(vs_local, fs_local), target, out_dir, TOOLS)
                    manifest = json.loads(manifest_path.read_text())
                    params = manifest["parameters"]
                    self.assertEqual(len(params), 2)
                    self.assertEqual(params[0]["name"], "vs_pos")
                    self.assertEqual(params[1]["name"], "light")
                    self.assertEqual(params[1]["layout"], "gfx-std430-v1")
                    self.assertEqual(params[1]["schema_version"], 2)
                    if target == Target.VULKAN:
                        self.assertEqual(params[0]["bindings"]["vulkan"]["offset"], 0)
                        self.assertEqual(params[1]["bindings"]["vulkan"]["offset"], 16)
                    else:
                        self.assertEqual(params[0]["bindings"]["directx"]["register"], 0)
                        self.assertEqual(params[1]["bindings"]["directx"]["register"], 1)

    def test_float_struct_cross_stage_layout_reconciliation_index_2(self):
        @vertex
        def vs_idx2(idx: u32, cam: PipelineCameraBlock, obj: PipelineObjectBlock, light: PipelineLightBlock) -> Vertex:
            return Vertex(cam.eye.x * obj.color + vec4(light.direction, light.intensity))

        @fragment
        def fs_idx0(light: PipelineLightBlock) -> vec4:
            return vec4(light.color, light.ambient)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            manifest_path = compile_pipeline(Pipeline(vs_idx2, fs_idx0), Target.VULKAN, out_dir, TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest["parameters"]
            self.assertEqual(len(params), 3)
            light_entry = [p for p in params if p["name"] == "light"][0]
            self.assertEqual(light_entry["layout"], "gfx-std430-v1")
            self.assertEqual(light_entry["schema_version"], 2)
            self.assertEqual(light_entry["bindings"]["vulkan"]["offset"], 176)

    def test_float_struct_single_block_reconciliation(self):
        @vertex
        def vs_single(idx: u32, light: PipelineLightBlock) -> Vertex:
            return Vertex(vec4(light.direction, light.intensity))

        @fragment
        def fs_single(light: PipelineLightBlock) -> vec4:
            return vec4(light.color, light.ambient)

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            manifest_path = compile_pipeline(Pipeline(vs_single, fs_single), Target.VULKAN, out_dir, TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest["parameters"]
            entry = params[0] if isinstance(params, list) else params
            self.assertEqual(entry["name"], "light")
            self.assertEqual(entry["bindings"]["vulkan"]["offset"], 0)

    def test_inverted_parameter_block_order_across_stages(self):
        @vertex
        def vs_order_ab(idx: u32, a: PipelineVertexOnlyBlock, b: PipelineMaterialBlock) -> Vertex:
            return Vertex(a.pos + b.albedo)

        @fragment
        def fs_order_ba(b: PipelineMaterialBlock, a: PipelineVertexOnlyBlock) -> vec4:
            return a.pos * b.albedo

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            manifest_path = compile_pipeline(Pipeline(vs_order_ab, fs_order_ba), Target.VULKAN, out_dir, TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest["parameters"]
            self.assertEqual(len(params), 2)
            self.assertEqual(params[0]["name"], "a")
            self.assertEqual(params[1]["name"], "b")
            self.assertEqual(params[0]["bindings"]["vulkan"]["offset"], 0)
            self.assertEqual(params[1]["bindings"]["vulkan"]["offset"], 16)

    def test_alternating_disjoint_parameter_blocks_across_stages(self):
        @vertex
        def vs_bc(idx: u32, b: PipelineVertexOnlyBlock, c: PipelineMaterialBlock) -> Vertex:
            return Vertex(b.pos + c.albedo)

        @fragment
        def fs_ab(a: PipelineLightBlock, b: PipelineVertexOnlyBlock) -> vec4:
            return vec4(a.color, a.ambient) + b.pos

        with tempfile.TemporaryDirectory() as td:
            out_dir = Path(td) / "out"
            manifest_path = compile_pipeline(Pipeline(vs_bc, fs_ab), Target.VULKAN, out_dir, TOOLS)
            manifest = json.loads(manifest_path.read_text())
            params = manifest["parameters"]
            self.assertEqual(len(params), 3)
            self.assertEqual(params[0]["name"], "b")
            self.assertEqual(params[1]["name"], "c")
            self.assertEqual(params[2]["name"], "a")
            self.assertEqual(params[0]["bindings"]["vulkan"]["offset"], 0)
            self.assertEqual(params[1]["bindings"]["vulkan"]["offset"], 16)
            self.assertEqual(params[2]["bindings"]["vulkan"]["offset"], 48)

    def test_compute_pipeline_with_parameter_block(self):
        @compute(workgroup_size=(64, 1, 1))
        def sim_compute(buf: Buffer[Type.F32, "read_write"], sim: PipelineSimBlock):
            buf[0] = sim.gravity.x * sim.delta_time

        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "sim.spv"
            ref_file = Path(td) / "sim.reflection.json"
            compile_pipeline(sim_compute, Target.VULKAN, out_file, reflection_path=ref_file)
            self.assertTrue(out_file.is_file())
            ref = json.loads(ref_file.read_text())
            sim_param = ref["parameters"]
            self.assertEqual(sim_param["name"], "sim")
            self.assertEqual(sim_param["layout"], "gfx-std430-v1")
            self.assertEqual(sim_param["schema_version"], 2)

    def test_mismatched_parameter_block_field_names_raise_descriptive_error(self):
        block_v1 = ParameterBlock("mismatched_block", (("color", Type.VEC4),))
        block_v2 = ParameterBlock("mismatched_block", (("tint", Type.VEC4),))

        @vertex
        def vs_mismatch(idx: u32, p: block_v1) -> Vertex:
            return Vertex(p.color)

        @fragment
        def fs_mismatch(p: block_v2) -> vec4:
            return p.tint

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "Graphics stages require matching parameter block fields: 'mismatched_block'"):
                compile_pipeline(Pipeline(vs_mismatch, fs_mismatch), Target.VULKAN, Path(td) / "out", TOOLS)

    def test_mismatched_parameter_block_field_types_raise_descriptive_error(self):
        block_v1 = ParameterBlock("mismatched_block", (("color", Type.VEC4),))
        block_v2 = ParameterBlock("mismatched_block", (("color", Type.VEC3),))

        @vertex
        def vs_mismatch(idx: u32, p: block_v1) -> Vertex:
            return Vertex(p.color)

        @fragment
        def fs_mismatch(p: block_v2) -> vec4:
            return vec4(p.color, 1.0)

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "Graphics stages require matching parameter block fields: 'mismatched_block'"):
                compile_pipeline(Pipeline(vs_mismatch, fs_mismatch), Target.VULKAN, Path(td) / "out", TOOLS)

    def test_cumulative_size_limit_validation(self):
        @struct
        class BigBlock1:
            matrix_a: mat4; matrix_b: mat4; v0: vec4  # 64 + 64 + 16 = 144B

        @struct
        class BigBlock2:
            matrix_a: mat4; matrix_b: mat4            # 64 + 64 = 128B (total 272B > 256B)

        @vertex
        def vs_big(idx: u32, b1: BigBlock1) -> Vertex:
            return Vertex(b1.v0)

        @fragment
        def fs_big(b2: BigBlock2) -> vec4:
            return b2.matrix_a[0]

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaisesRegex(ValueError, "Cumulative parameter blocks size .* exceeds 256 bytes"):
                compile_pipeline(Pipeline(vs_big, fs_big), Target.VULKAN, Path(td) / "out", TOOLS)


if __name__ == "__main__":
    unittest.main()
