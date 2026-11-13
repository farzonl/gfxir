"""Unit and GPU parity tests for MiniGfx 2D textures, samplers, and textured PBR materials."""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
_TESTS_DIR = _REPO_ROOT / "tests"
for _p in (_REPO_ROOT, _SRC_DIR, _TESTS_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

from examples.minigfx import Material, MaterialParams, Sampler, Texture
from examples.minigfx.camera import Camera
from examples.minigfx.material import (
    CameraParams,
    LightParams,
    ObjectParams,
    pbr_fragment,
    pbr_vertex,
    textured_pbr_fragment,
)
from examples.minigfx.math import Mat4x4, Vec2, Vec3, Vec4
from examples.minigfx.renderer import ForwardRenderer
from examples.minigfx.scene import Mesh, Node, Scene, Vertex as MeshVertex
from examples.minigfx.window import compile_minigfx_shaders, pack_mesh_triangles

from mlir_gpu import Buffer, Draw, Pipeline, Sampler as GpuSampler, Target, Texture2D, Type, Vertex, compile_pipeline, fragment, sample_texture, u32, vec2, vec3, vec4, vertex
import mlir_gpu.shader_math as sm
from mlir_gpu.viewer import load_program, render_directx_offscreen, render_offscreen
from test_support import TOOLS, available


def _has_vulkan_runtime() -> bool:
    if os.environ.get("MLIR_GPU_TEST_VULKAN") == "1":
        return True
    try:
        import vulkan as vk

        inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(), None)
        try:
            return bool(vk.vkEnumeratePhysicalDevices(inst))
        finally:
            vk.vkDestroyInstance(inst, None)
    except Exception:
        return False


def _has_directx_runtime() -> bool:
    return sys.platform == "win32" and available("gfx-opt", "gfx-compile", "llc", "dxv", "mlir-gpu-directx-view")


def _make_uv_quad_mesh(half_extent: float = 1.0) -> Mesh:
    """Create a +Z-facing XY quad spanning [-half_extent, +half_extent] with standard [0, 1] UVs."""
    h = float(half_extent)
    n = Vec3(0.0, 0.0, 1.0)
    verts = [
        MeshVertex(Vec3(-h, h, 0.0), n, Vec2(0.0, 0.0)),
        MeshVertex(Vec3(h, h, 0.0), n, Vec2(1.0, 0.0)),
        MeshVertex(Vec3(h, -h, 0.0), n, Vec2(1.0, 1.0)),
        MeshVertex(Vec3(-h, -h, 0.0), n, Vec2(0.0, 1.0)),
    ]
    return Mesh(verts, [0, 3, 2, 0, 2, 1])


class TestMiniGfxTextureAndSampler(unittest.TestCase):
    def test_sampler_construction_and_conversion(self):
        samp = Sampler()
        self.assertEqual(samp.mag_filter, "linear")
        self.assertEqual(samp.min_filter, "linear")
        self.assertEqual(samp.address_mode_u, "repeat")
        self.assertEqual(samp.address_mode_v, "repeat")

        desc = samp.to_sampler_desc()
        self.assertEqual(desc.mag_filter, "linear")
        self.assertEqual(desc.address_mode_u, "repeat")

        custom = Sampler(
            mag_filter="nearest",
            min_filter="nearest",
            address_mode_u="clamp_to_edge",
            address_mode_v="mirror_repeat",
        )
        cdesc = custom.to_sampler_desc()
        self.assertEqual(cdesc.mag_filter, "nearest")
        self.assertEqual(cdesc.address_mode_u, "clamp_to_edge")
        self.assertEqual(cdesc.address_mode_v, "mirror_repeat")

        with self.assertRaises(ValueError):
            Sampler(mag_filter="anisotropic")
        with self.assertRaises(ValueError):
            Sampler(address_mode_v="border")

    def test_texture_factories_and_descriptor_conversion(self):
        solid = Texture.solid((0.25, 0.5, 0.75, 1.0), width=2, height=2)
        self.assertEqual((solid.width, solid.height, solid.format), (2, 2, "rgba8_unorm"))
        td = solid.to_texture_data()
        self.assertEqual((td.width, td.height, td.format), (2, 2, "rgba8_unorm"))
        self.assertEqual(len(td.data), 16)

        checker = Texture.checkerboard(width=4, height=4, tile_size=2)
        self.assertEqual((checker.width, checker.height), (4, 4))
        c0 = checker.load(0, 0)
        c2 = checker.load(2, 0)
        self.assertNotEqual(c0, c2)

        f32_tex = Texture.from_rgba_f32(2, 1, [(0.1, 0.2, 0.3, 1.0), (0.9, 0.8, 0.7, 0.5)])
        self.assertEqual(f32_tex.format, "rgba32_sfloat")
        p1 = f32_tex.load(1, 0)
        self.assertAlmostEqual(p1.x, 0.9, places=5)
        self.assertAlmostEqual(p1.w, 0.5, places=5)

    def test_cpu_texture_sampling_filters_and_wrapping(self):
        tex = Texture.from_rgba8(
            2,
            2,
            [
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
            ],
        )
        near_clamp = Sampler(mag_filter="nearest", min_filter="nearest", address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge")
        lin_clamp = Sampler(mag_filter="linear", min_filter="linear", address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge")
        near_repeat = Sampler(mag_filter="nearest", min_filter="nearest", address_mode_u="repeat", address_mode_v="repeat")
        near_mirror = Sampler(mag_filter="nearest", min_filter="nearest", address_mode_u="mirror_repeat", address_mode_v="mirror_repeat")

        self.assertEqual(tex.sample(Vec2(0.1, 0.1), near_clamp), Vec4(1.0, 0.0, 0.0, 1.0))
        self.assertEqual(tex.sample(Vec2(0.9, 0.1), near_clamp), Vec4(0.0, 1.0, 0.0, 1.0))
        self.assertEqual(tex.sample(Vec2(0.1, 0.9), near_clamp), Vec4(0.0, 0.0, 1.0, 1.0))
        self.assertEqual(tex.sample(Vec2(0.9, 0.9), near_clamp), Vec4(1.0, 1.0, 0.0, 1.0))

        mid = tex.sample(Vec2(0.5, 0.5), lin_clamp)
        self.assertAlmostEqual(mid.x, 0.5, places=4)
        self.assertAlmostEqual(mid.y, 0.5, places=4)
        self.assertAlmostEqual(mid.z, 0.25, places=4)
        self.assertAlmostEqual(mid.w, 1.0, places=4)

        self.assertEqual(tex.sample(Vec2(-0.25, 0.25), near_clamp), Vec4(1.0, 0.0, 0.0, 1.0))
        self.assertEqual(tex.sample(Vec2(-0.25, 0.25), near_repeat), Vec4(0.0, 1.0, 0.0, 1.0))
        self.assertEqual(tex.sample(Vec2(1.25, 0.25), near_mirror), Vec4(0.0, 1.0, 0.0, 1.0))


class TestMiniGfxMaterialAndRendererIntegration(unittest.TestCase):
    def test_procedural_and_textured_material_cpu_shading(self):
        cam = CameraParams(view_proj=Mat4x4.identity(), eye_pos=Vec3(0.0, 0.0, 3.0), time=0.0)
        light = LightParams(direction=Vec3(0.0, 0.4, 1.0), intensity=1.2, color=Vec3(1.0, 1.0, 1.0), ambient=0.1)
        params = MaterialParams(albedo=(0.8, 0.6, 0.4, 1.0), roughness=0.45, metallic=0.1, ambient=0.05)

        untextured = Material(params=params)
        self.assertFalse(untextured.is_textured)
        self.assertEqual(untextured.gpu_resources(), ({}, {}))

        white_tex = Texture.solid((1.0, 1.0, 1.0, 1.0), width=2, height=2)
        samp = Sampler(mag_filter="nearest", min_filter="nearest")
        textured_white = Material(params=params, albedo_texture=white_tex, sampler=samp)
        self.assertTrue(textured_white.is_textured)
        tex_map, samp_map = textured_white.gpu_resources()
        self.assertIn("albedo_tex", tex_map)
        self.assertIn("albedo_sampler", samp_map)

        # Solid white texture with textured_pbr_fragment matches procedural pbr_fragment exactly
        c_proc = pbr_fragment(Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 0.0), Vec2(0.25, 0.25), cam, params, light)
        c_tex = textured_pbr_fragment(
            Vec3(0.0, 0.0, 1.0),
            Vec3(0.0, 0.0, 0.0),
            Vec2(0.25, 0.25),
            white_tex,
            samp,
            cam,
            params,
            light,
        )
        for ch_a, ch_b in zip((c_proc.x, c_proc.y, c_proc.z, c_proc.w), (c_tex.x, c_tex.y, c_tex.z, c_tex.w)):
            self.assertAlmostEqual(ch_a, ch_b, places=5)

        # Red/Blue texture modulates left vs right UV halves distinctly
        rb_tex = Texture.from_rgba8(2, 1, [(255, 0, 0, 255), (0, 0, 255, 255)])
        sharp = MaterialParams(roughness=0.15, metallic=0.0, ambient=0.05)
        oblique_light = LightParams(direction=Vec3(1.0, 0.5, 0.4), intensity=1.0, color=Vec3(1.0, 1.0, 1.0), ambient=0.1)
        c_left = textured_pbr_fragment(
            Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 0.0), Vec2(0.25, 0.5), rb_tex, samp, cam, sharp, oblique_light
        )
        c_right = textured_pbr_fragment(
            Vec3(0.0, 0.0, 1.0), Vec3(0.0, 0.0, 0.0), Vec2(0.75, 0.5), rb_tex, samp, cam, sharp, oblique_light
        )
        self.assertGreater(c_left.x, c_left.z + 0.3)
        self.assertGreater(c_right.z, c_right.x + 0.3)

    def test_forward_renderer_populates_textured_draw_commands_without_breaking_procedural(self):
        scene = Scene()
        quad = _make_uv_quad_mesh(0.5)

        proc_node = Node("proc_quad", mesh=quad)
        proc_node.material = MaterialParams(albedo=(0.9, 0.2, 0.2, 1.0))

        tex = Texture.checkerboard(width=4, height=4, tile_size=2)
        samp = Sampler(mag_filter="nearest", min_filter="nearest")
        tex_mat = Material(params=MaterialParams(albedo=(1.0, 1.0, 1.0, 1.0)), albedo_texture=tex, sampler=samp)
        tex_node = Node("tex_quad", translation=Vec3(0.5, 0.0, 0.0), mesh=quad)
        tex_node.material = tex_mat

        scene.add_child(proc_node)
        scene.add_child(tex_node)

        cam = Camera(eye=Vec3(0.0, 0.0, 5.0), target=Vec3(0.0, 0.0, 0.0))
        light = LightParams()
        renderer = ForwardRenderer()
        cmds = renderer.render_frame(camera=cam, lights=light, scene=scene)
        self.assertEqual(len(cmds), 2)

        by_name = {c.node.name: c for c in cmds}
        self.assertEqual(by_name["proc_quad"].textures, {})
        self.assertEqual(by_name["proc_quad"].samplers, {})

        self.assertEqual(by_name["tex_quad"].pipeline_id, "textured_pbr")
        self.assertIn("albedo_tex", by_name["tex_quad"].textures)
        self.assertIn("albedo_sampler", by_name["tex_quad"].samplers)


@unittest.skipUnless(available("gfx-opt", "gfx-compile"), "Native GfxIR compiler tools required")
class TestMiniGfxTexturedGpuParity(unittest.TestCase):
    def test_window_vulkan_shader_bundle_includes_textured_pbr(self):
        if not available("spirv-val"):
            self.skipTest("spirv-val required for Vulkan shader bundle compilation")
        compiled = compile_minigfx_shaders("vulkan")
        self.assertIn("textured_pbr_fs", compiled)
        self.assertTrue(compiled["textured_pbr_fs"].is_file())
        self.assertGreater(compiled["textured_pbr_fs"].stat().st_size, 0)

    def test_window_directx_shader_bundle_includes_textured_pbr(self):
        if not available("llc", "dxv"):
            self.skipTest("llc and dxv required for DirectX shader bundle compilation")
        compiled = compile_minigfx_shaders("directx")
        self.assertIn("textured_pbr_fs", compiled)
        dxil_path = compiled["textured_pbr_fs"]
        self.assertTrue(dxil_path.is_file())
        raw = dxil_path.read_bytes()
        self.assertGreater(len(raw), 32)
        self.assertEqual(raw[:4], b"DXBC")

    def test_textured_pbr_material_cpu_vulkan_and_directx_parity(self):
        has_vk = _has_vulkan_runtime()
        has_dx = _has_directx_runtime()
        if not has_vk and not has_dx:
            self.skipTest("Neither Vulkan nor DirectX 12 GPU runtime is available")

        vk_val = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

        @vertex
        def mesh_storage_vs(
            vertex_index: u32,
            verts: Buffer[Type.VEC4, "read_only"],
        ) -> Vertex:
            p = verts[vertex_index * u32(2)]
            n = verts[vertex_index * u32(2) + u32(1)]
            pos = vec4(p.xyz, 1.0)
            return Vertex(
                pos,
                varyings={"normal": n.xyz, "world_pos": p.xyz, "uv": vec2(p.w, n.w)},
            )

        @fragment
        def mesh_textured_pbr_fs(
            normal,
            world_pos,
            uv,
            scene_ub: Buffer[Type.VEC4, "read_only"],
            albedo_tex: Texture2D,
            albedo_sampler: GpuSampler,
        ):
            eye_pos = scene_ub[0].xyz
            mat_albedo = scene_ub[1]
            mat_rma = scene_ub[2]
            light_dir_int = scene_ub[3]
            light_amb = scene_ub[4].w

            texel = sample_texture(albedo_tex, albedo_sampler, uv)
            base_albedo = mat_albedo * texel
            n = sm.normalize(normal)
            l = sm.normalize(light_dir_int.xyz)
            n_dot_l = sm.clamp(sm.dot(n, l), 0.0, 1.0)
            diffuse = base_albedo.xyz * n_dot_l * light_dir_int.w * (1.0 - mat_rma.y)
            ambient = base_albedo.xyz * (light_amb + mat_rma.z)

            v = sm.normalize(eye_pos - world_pos)
            h = sm.normalize(l + v)
            n_dot_h = sm.clamp(sm.dot(n, h), 0.0, 1.0)
            specular = sm.pow(n_dot_h, 2.0 / (mat_rma.x * mat_rma.x + 0.001)) * light_dir_int.w
            hdr = ambient + diffuse + vec3(specular, specular, specular)
            ldr = sm.clamp((hdr * (hdr * 2.51 + 0.03)) / (hdr * (hdr * 2.43 + 0.59) + 0.14), 0.0, 1.0)
            srgb = sm.pow(ldr, 1.0 / 2.2)
            return vec4(srgb, base_albedo.w)

        quad_mesh = _make_uv_quad_mesh(half_extent=1.0)
        vert_bytes, vert_count = pack_mesh_triangles(quad_mesh)

        cam = CameraParams(view_proj=Mat4x4.identity(), eye_pos=Vec3(0.0, 0.0, 2.0), time=0.0)
        mat_params = MaterialParams(albedo=(0.9, 0.9, 0.9, 1.0), roughness=0.7, metallic=0.0, ambient=0.1)
        light = LightParams(direction=Vec3(0.5, 0.5, 1.0), intensity=1.0, color=Vec3(1.0, 1.0, 1.0), ambient=0.15)
        scene_ub_bytes = struct.pack(
            "<20f",
            cam.eye_pos.x, cam.eye_pos.y, cam.eye_pos.z, cam.time,
            mat_params.albedo.x, mat_params.albedo.y, mat_params.albedo.z, mat_params.albedo.w,
            mat_params.roughness, mat_params.metallic, mat_params.ambient, 0.0,
            light.direction.x, light.direction.y, light.direction.z, light.intensity,
            light.color.x, light.color.y, light.color.z, light.ambient,
        )

        tex = Texture.from_rgba8(
            2,
            2,
            [
                (255, 32, 32, 255),
                (32, 255, 32, 255),
                (32, 32, 255, 255),
                (255, 220, 32, 255),
            ],
        )
        samp = Sampler(mag_filter="nearest", min_filter="nearest", address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge")
        textured_mat = Material(params=mat_params, albedo_texture=tex, sampler=samp)
        gpu_tex, gpu_samp = textured_mat.gpu_resources()
        gpu_bufs = {"verts": vert_bytes, "scene_ub": scene_ub_bytes}

        pipe = Pipeline(mesh_storage_vs, mesh_textured_pbr_fs, Draw(vertex_count=vert_count))

        renders: dict[str, bytes] = {}
        with tempfile.TemporaryDirectory() as tmp:
            if has_vk:
                vk_manifest = compile_pipeline(pipe, Target.VULKAN, Path(tmp) / "vk_tex_mat", TOOLS)
                vk_prog = load_program(
                    vk_manifest,
                    buffers=gpu_bufs,
                    textures=gpu_tex,
                    samplers=gpu_samp,
                )
                renders["vulkan"] = render_offscreen(vk_prog, 8, 8, validation=vk_val).rgba

            if has_dx:
                dx_manifest = compile_pipeline(pipe, Target.DIRECTX, Path(tmp) / "dx_tex_mat", TOOLS)
                for prov in ("hardware", "warp"):
                    dx_prog = load_program(
                        dx_manifest,
                        buffers=gpu_bufs,
                        textures=gpu_tex,
                        samplers=gpu_samp,
                    )
                    renders[f"directx-{prov}"] = render_directx_offscreen(
                        dx_prog, 8, 8, validation=True, toolchain=TOOLS, provider=prov
                    ).rgba

        # Compare GPU quadrants against CPU textured_pbr_fragment evaluation at exact fragment world_pos & UV
        sample_points = [
            (1, 1),
            (6, 1),
            (1, 6),
            (6, 6),
        ]

        for backend_name, rgba in renders.items():
            for px_x, px_y in sample_points:
                u = (px_x + 0.5) / 8.0
                v = (px_y + 0.5) / 8.0
                wx = u * 2.0 - 1.0
                wy = 1.0 - v * 2.0
                cpu_col = textured_pbr_fragment(
                    Vec3(0.0, 0.0, 1.0),
                    Vec3(wx, wy, 0.0),
                    Vec2(u, v),
                    tex,
                    samp,
                    cam,
                    mat_params,
                    light,
                )
                expected_u8 = (
                    round(cpu_col.x * 255),
                    round(cpu_col.y * 255),
                    round(cpu_col.z * 255),
                    round(cpu_col.w * 255),
                )
                off = (px_y * 8 + px_x) * 4
                actual_u8 = tuple(rgba[off : off + 4])
                # Note: GPU shader uses fast gamma x^(1/2.2) while CPU tonemap_and_gamma uses IEC 61966-2-1 sRGB
                for ch in range(4):
                    self.assertAlmostEqual(
                        actual_u8[ch],
                        expected_u8[ch],
                        delta=6,
                        msg=f"CPU vs {backend_name} mismatch at ({px_x}, {px_y}) ch={ch}: got {actual_u8}, want {expected_u8}",
                    )

        # Also verify MiniGfx's builtin pbr_vertex + textured_pbr_fragment compiles cleanly via GfxIR!
        with tempfile.TemporaryDirectory() as tmp:
            pbr_pipe = Pipeline(
                pbr_vertex,
                textured_pbr_fragment,
                Draw(vertex_count=vert_count),
            )
            if available("spirv-val"):
                pbr_vk = compile_pipeline(pbr_pipe, Target.VULKAN, Path(tmp) / "minigfx_builtin_vk", TOOLS)
                self.assertTrue(pbr_vk.is_file())
            if available("dxv"):
                pbr_dx = compile_pipeline(pbr_pipe, Target.DIRECTX, Path(tmp) / "minigfx_builtin_dx", TOOLS)
                self.assertTrue(pbr_dx.is_file())

        # Cross-GPU parity check (Vulkan vs DirectX hardware vs DirectX WARP)
        ref_name, ref_rgba = next(iter(renders.items()))
        for name, rgba in renders.items():
            max_err = max(abs(a - b) for a, b in zip(rgba, ref_rgba))
            self.assertLessEqual(max_err, 2, f"Cross-GPU mismatch between {ref_name} and {name}: max_err={max_err}")


if __name__ == "__main__":
    unittest.main()
