from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import (
    Buffer,
    CompileError,
    ComputePipeline,
    Dispatch,
    Draw,
    GlobalInvocationId,
    Pipeline,
    Sampler,
    SamplerBinding,
    SamplerDesc,
    Target,
    Texture2D,
    TextureBinding,
    TextureData,
    Type,
    Vertex,
    compile_pipeline,
    compute,
    fragment,
    load_texture,
    sample_texture,
    sample_texture_level,
    select,
    u32,
    vec2,
    vec4,
    vertex,
)
from mlir_gpu.viewer import (
    ViewerError,
    _write_native_graphics_resources,
    dispatch_compute,
    load_compute_program,
    load_program,
    render_directx_offscreen,
    render_metal_offscreen,
    render_offscreen,
)
from test_support import TOOLS, available


def _has_vulkan_runtime() -> bool:
    if os.environ.get("MLIR_GPU_TEST_VULKAN") == "1":
        return True
    try:
        import vulkan as vk

        inst = vk.vkCreateInstance(vk.VkInstanceCreateInfo(), None)
        try:
            devs = vk.vkEnumeratePhysicalDevices(inst)
            return bool(devs)
        finally:
            vk.vkDestroyInstance(inst, None)
    except Exception:
        return False


def _has_directx_runtime() -> bool:
    if sys.platform != "win32":
        return False
    return available("gfx-opt", "gfx-compile", "llc", "dxv", "mlir-gpu-directx-view")


def _fullscreen_uv_vertex():
    @vertex
    def fs_tri_vs(index: u32) -> Vertex:
        x = select(index == u32(1), 3.0, -1.0)
        y = select(index == u32(2), -3.0, 1.0)
        u = select(index == u32(1), 2.0, 0.0)
        v = select(index == u32(2), 2.0, 0.0)
        return Vertex(vec4(x, y, 0.0, 1.0), varyings={"uv": vec2(u, v)})

    return fs_tri_vs


class TextureDescriptorValidationTests(unittest.TestCase):
    def test_texture_data_rgba8_and_rgba32f_construction(self):
        tex8 = TextureData.from_rgba8(
            2,
            2,
            [
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 255, 255),
            ],
        )
        self.assertEqual((tex8.width, tex8.height, tex8.format), (2, 2, "rgba8_unorm"))
        self.assertEqual(tex8.bytes_per_pixel, 4)
        self.assertEqual(len(tex8.data), 16)

        tex32 = TextureData.from_rgba_f32(
            2,
            1,
            [
                (0.25, 0.5, 0.75, 1.0),
                (1.0, 0.0, 0.5, 1.0),
            ],
        )
        self.assertEqual((tex32.width, tex32.height, tex32.format), (2, 1, "rgba32_sfloat"))
        self.assertEqual(tex32.bytes_per_pixel, 16)
        self.assertEqual(len(tex32.data), 32)

        checker = TextureData.checkerboard(width=4, height=4, tile_size=2)
        self.assertEqual((checker.width, checker.height), (4, 4))
        self.assertEqual(len(checker.data), 64)

    def test_texture_data_rejects_invalid_arguments(self):
        with self.assertRaises(ValueError):
            TextureData(width=0, height=2, data=b"\x00" * 8)
        with self.assertRaises(ValueError):
            TextureData(width=True, height=2, data=b"\x00" * 8)
        with self.assertRaises(ValueError):
            TextureData(width=2, height=2, data=b"\x00" * 16, format="bgra8_unorm")
        with self.assertRaises(ValueError):
            TextureData(width=2, height=2, data=b"\x00" * 12, format="rgba8_unorm")
        with self.assertRaises(ValueError):
            TextureData(width=2, height=2, data=b"\x00" * 16, format="rgba32_sfloat")

    def test_sampler_desc_validation(self):
        default_samp = SamplerDesc()
        self.assertEqual(default_samp.mag_filter, "linear")
        self.assertEqual(default_samp.min_filter, "linear")
        self.assertEqual(default_samp.address_mode_u, "repeat")
        self.assertEqual(default_samp.address_mode_v, "repeat")

        custom_samp = SamplerDesc(
            mag_filter="nearest",
            min_filter="linear",
            address_mode_u="clamp_to_edge",
            address_mode_v="mirror_repeat",
        )
        self.assertEqual(custom_samp.mag_filter, "nearest")
        self.assertEqual(custom_samp.address_mode_u, "clamp_to_edge")
        self.assertEqual(custom_samp.address_mode_v, "mirror_repeat")

        with self.assertRaises(ValueError):
            SamplerDesc(mag_filter="cubic")
        with self.assertRaises(ValueError):
            SamplerDesc(address_mode_u="clamp_to_border")

    def test_subscript_resource_annotations(self):
        t0 = Texture2D[3]
        self.assertIsInstance(t0, TextureBinding)
        self.assertEqual((t0.set, t0.binding), (0, 3))

        t1 = Texture2D[1, 4]
        self.assertEqual((t1.set, t1.binding), (1, 4))

        s0 = Sampler[2]
        self.assertIsInstance(s0, SamplerBinding)
        self.assertEqual((s0.set, s0.binding), (0, 2))

        s1 = Sampler[2, 5]
        self.assertEqual((s1.set, s1.binding), (2, 5))


@unittest.skipUnless(available("gfx-opt", "gfx-compile"), "Native GfxIR compiler tools required")
class TextureFrontendAndCompilerTests(unittest.TestCase):
    def test_implicit_lod_sample_rejected_outside_fragment(self):
        @vertex
        def bad_vs(index: u32, tex: Texture2D, samp: Sampler) -> Vertex:
            c = sample_texture(tex, samp, vec2(0.5, 0.5))
            return Vertex(c)

        @fragment
        def pass_fs() -> vec4:
            return vec4(1.0, 1.0, 1.0, 1.0)

        with self.assertRaises((TypeError, ValueError, CompileError)) as ctx_vs:
            Pipeline(bad_vs, pass_fs).trace()
        self.assertIn("fragment", str(ctx_vs.exception).lower())

        @compute(workgroup_size=(1, 1, 1))
        def bad_cs(tex: Texture2D, samp: Sampler, out: Buffer[Type.VEC4, "write_only"]):
            out[0] = sample_texture(tex, samp, vec2(0.5, 0.5))

        with self.assertRaises((TypeError, ValueError, CompileError)) as ctx_cs:
            ComputePipeline(bad_cs).trace()
        self.assertIn("fragment", str(ctx_cs.exception).lower())

    def test_conflicting_descriptor_bindings_rejected(self):
        @fragment
        def conflict_fs(uv, tex: Texture2D[0], samp: Sampler[0]):
            return sample_texture(tex, samp, uv)

        pipe = Pipeline(_fullscreen_uv_vertex(), conflict_fs, Draw(vertex_count=3))
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises((ValueError, CompileError)):
                compile_pipeline(pipe, Target.VULKAN, Path(tmp) / "conflict", TOOLS)

    def test_cross_stage_unindexed_resource_auto_offset(self):
        @vertex
        def vs_with_tex(index: u32, vtex: Texture2D, vsamp: Sampler) -> Vertex:
            tint = sample_texture_level(vtex, vsamp, vec2(0.25, 0.25), 0.0)
            x = select(index == u32(1), 3.0, -1.0)
            y = select(index == u32(2), -3.0, 1.0)
            return Vertex(vec4(x, y, 0.0, 1.0), varyings={"tint": tint, "uv": vec2(0.5, 0.5)})

        @fragment
        def fs_with_tex(tint, uv, ftex: Texture2D, fsamp: Sampler):
            return tint * sample_texture(ftex, fsamp, uv)

        pipe = Pipeline(vs_with_tex, fs_with_tex, Draw(vertex_count=3))
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = compile_pipeline(pipe, Target.VULKAN, Path(tmp) / "cross_stage", TOOLS)
            manifest = json.loads(manifest_path.read_text())
            tex_slots = {(t["name"], t["set"], t["binding"]) for t in manifest["textures"]}
            samp_slots = {(s["name"], s["set"], s["binding"]) for s in manifest["samplers"]}
            self.assertEqual(tex_slots, {("vtex", 0, 0), ("ftex", 0, 2)})
            self.assertEqual(samp_slots, {("vsamp", 0, 1), ("fsamp", 0, 3)})

    def test_manifest_and_program_loader_resource_validation(self):
        @fragment
        def textured_fs(uv, albedo: Texture2D[0], linear_samp: Sampler[1]):
            return sample_texture(albedo, linear_samp, uv)

        pipe = Pipeline(_fullscreen_uv_vertex(), textured_fs, Draw(vertex_count=3))
        tex = TextureData.checkerboard(width=2, height=2, tile_size=1)
        samp = SamplerDesc()

        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = compile_pipeline(pipe, Target.VULKAN, Path(tmp) / "prog", TOOLS)

            # Missing texture binding raises ViewerError when bindings are validated during render/write
            prog_missing = load_program(manifest_path, samplers={"linear_samp": samp})
            with self.assertRaises(ViewerError):
                _write_native_graphics_resources(Path(tmp), prog_missing)

            # Unknown texture name raises ViewerError in load_program
            with self.assertRaises(ViewerError):
                load_program(manifest_path, textures={"wrong_name": tex}, samplers={"linear_samp": samp})

            # Wrong resource descriptor type raises ViewerError
            with self.assertRaises(ViewerError):
                load_program(manifest_path, textures={"albedo": b"raw_bytes"}, samplers={"linear_samp": samp})  # type: ignore[arg-type]

            # Valid graphics resources write native resources.json cleanly (used by DirectX and Metal viewers)
            prog_ok = load_program(manifest_path, textures={"albedo": tex}, samplers={"linear_samp": samp})
            res_json = _write_native_graphics_resources(Path(tmp), prog_ok)
            self.assertIsNotNone(res_json)
            payload = json.loads(res_json.read_text())
            self.assertEqual(len(payload["fragment"]["textures"]), 1)
            self.assertEqual(payload["fragment"]["textures"][0]["name"], "albedo")
            self.assertEqual(payload["fragment"]["textures"][0]["format"], "rgba8_unorm")
            self.assertEqual(len(payload["fragment"]["samplers"]), 1)
            self.assertEqual(payload["fragment"]["samplers"][0]["name"], "linear_samp")

        @compute(workgroup_size=(1, 1, 1))
        def comp_kernel(src: Texture2D[0], s: Sampler[1], dst: Buffer[Type.VEC4, "write_only", 2]):
            dst[0] = sample_texture_level(src, s, vec2(0.5, 0.5), 0.0)

        with tempfile.TemporaryDirectory() as tmp:
            comp_manifest = compile_pipeline(ComputePipeline(comp_kernel), Target.VULKAN, Path(tmp) / "comp", TOOLS)
            cprog = load_compute_program(comp_manifest)
            self.assertEqual(cprog.textures[0]["binding"], 0)
            self.assertEqual(cprog.samplers[0]["binding"], 1)
            self.assertEqual(cprog.buffers[0]["binding"], 2)


@unittest.skipUnless(_has_vulkan_runtime() or _has_directx_runtime(), "Vulkan or DirectX 12 GPU runtime required")
class TextureGpuExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_vulkan = _has_vulkan_runtime()
        cls.has_directx = _has_directx_runtime()
        cls.vk_validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def _active_graphics_backends(self):
        backends = []
        if self.has_vulkan:
            backends.append(("vulkan", Target.VULKAN, None))
        if self.has_directx:
            backends.append(("directx-hw", Target.DIRECTX, "hardware"))
            backends.append(("directx-warp", Target.DIRECTX, "warp"))
        return backends

    def _render_backend(self, target: Target, program, width: int, height: int, provider: str | None):
        if target == Target.VULKAN:
            return render_offscreen(program, width, height, validation=self.vk_validation)
        if target == Target.DIRECTX:
            return render_directx_offscreen(program, width, height, validation=True, toolchain=TOOLS, provider=provider)
        return render_metal_offscreen(program, width, height, validation=True, toolchain=TOOLS)

    def test_graphics_nearest_and_linear_sampling_rgba8_and_rgba32f(self):
        @fragment
        def sample_fs(uv, tex: Texture2D[0], samp: Sampler[1]):
            return sample_texture(tex, samp, uv)

        pipe = Pipeline(_fullscreen_uv_vertex(), sample_fs, Draw(vertex_count=3))

        tex_rgba8 = TextureData.from_rgba8(
            2,
            2,
            [
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
            ],
        )
        tex_rgba32f = TextureData.from_rgba_f32(
            2,
            2,
            [
                (1.0, 0.0, 0.0, 1.0),
                (0.0, 1.0, 0.0, 1.0),
                (0.0, 0.0, 1.0, 1.0),
                (1.0, 1.0, 0.0, 1.0),
            ],
        )
        nearest = SamplerDesc(mag_filter="nearest", min_filter="nearest", address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge")
        linear = SamplerDesc(mag_filter="linear", min_filter="linear", address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge")

        with tempfile.TemporaryDirectory() as tmp:
            manifests = {}
            for _, target, _ in self._active_graphics_backends():
                if target not in manifests:
                    manifests[target] = compile_pipeline(pipe, target, Path(tmp) / f"quad-{target.value}", TOOLS)

            for fmt_label, tex in (("rgba8", tex_rgba8), ("rgba32f", tex_rgba32f)):
                nearest_renders = {}
                linear_renders = {}
                for label, target, provider in self._active_graphics_backends():
                    with self.subTest(format=fmt_label, backend=label):
                        prog_near = load_program(manifests[target], textures={"tex": tex}, samplers={"samp": nearest})
                        img_near = self._render_backend(target, prog_near, 8, 8, provider)
                        nearest_renders[label] = img_near.rgba

                        def px(rgba: bytes, x: int, y: int) -> tuple[int, int, int, int]:
                            off = (y * 8 + x) * 4
                            return tuple(rgba[off : off + 4])  # type: ignore[return-value]

                        self.assertEqual(px(img_near.rgba, 1, 1), (255, 0, 0, 255))
                        self.assertEqual(px(img_near.rgba, 6, 1), (0, 255, 0, 255))
                        self.assertEqual(px(img_near.rgba, 1, 6), (0, 0, 255, 255))
                        self.assertEqual(px(img_near.rgba, 6, 6), (255, 255, 0, 255))

                        prog_lin = load_program(manifests[target], textures={"tex": tex}, samplers={"samp": linear})
                        img_lin = self._render_backend(target, prog_lin, 9, 9, provider)
                        linear_renders[label] = img_lin.rgba
                        # Pixel (4, 4) has center UV = (4.5/9, 4.5/9) = (0.5, 0.5): exact average of the 4 texels = (0.5, 0.5, 0.25, 1.0)
                        c_off = (4 * 9 + 4) * 4
                        center = tuple(img_lin.rgba[c_off : c_off + 4])
                        self.assertAlmostEqual(center[0], 128, delta=2)
                        self.assertAlmostEqual(center[1], 128, delta=2)
                        self.assertAlmostEqual(center[2], 64, delta=2)
                        self.assertEqual(center[3], 255)

                ref_near = next(iter(nearest_renders.values()))
                for label, rgba in nearest_renders.items():
                    self.assertEqual(rgba, ref_near, f"Nearest render mismatch on {label} ({fmt_label})")

                ref_lin = next(iter(linear_renders.values()))
                for label, rgba in linear_renders.items():
                    max_diff = max(abs(a - b) for a, b in zip(rgba, ref_lin))
                    self.assertLessEqual(max_diff, 2, f"Linear render mismatch on {label} ({fmt_label})")

    def test_uv_address_modes_repeat_clamp_and_mirror(self):
        # Fullscreen triangle mapping screen x in [0, 1] -> u in [-1.0, 2.0] (span = 3.0, so index=1 at x=2 -> u=-1 + 6 = 5.0)
        @vertex
        def wide_uv_vs(index: u32) -> Vertex:
            x = select(index == u32(1), 3.0, -1.0)
            y = select(index == u32(2), -3.0, 1.0)
            u = select(index == u32(1), 5.0, -1.0)
            v = select(index == u32(2), 0.25, 0.25)
            return Vertex(vec4(x, y, 0.0, 1.0), varyings={"uv": vec2(u, v)})

        @fragment
        def wrap_fs(uv, tex: Texture2D[0], samp: Sampler[1]):
            return sample_texture(tex, samp, uv)

        pipe = Pipeline(wide_uv_vs, wrap_fs, Draw(vertex_count=3))
        # 2x1 horizontal texture: left=Red (x=0), right=Blue (x=1)
        tex = TextureData.from_rgba8(2, 1, [(255, 0, 0, 255), (0, 0, 255, 255)])
        RED = (255, 0, 0, 255)
        BLUE = (0, 0, 255, 255)

        # Width = 12 -> pixel x has u = -1.0 + 3.0 * (x + 0.5) / 12.0:
        # x=0 -> u=-0.875; x=3 -> u=-0.125; x=4 -> u=+0.125; x=7 -> u=+0.875; x=8 -> u=+1.125; x=11 -> u=+1.875
        expectations = {
            "repeat": {0: RED, 3: BLUE, 4: RED, 7: BLUE, 8: RED, 11: BLUE},
            "clamp_to_edge": {0: RED, 3: RED, 4: RED, 7: BLUE, 8: BLUE, 11: BLUE},
            "mirror_repeat": {0: BLUE, 3: RED, 4: RED, 7: BLUE, 8: BLUE, 11: RED},
        }

        with tempfile.TemporaryDirectory() as tmp:
            manifests = {}
            for _, target, _ in self._active_graphics_backends():
                if target not in manifests:
                    manifests[target] = compile_pipeline(pipe, target, Path(tmp) / f"wrap-{target.value}", TOOLS)

            for mode, expected_cols in expectations.items():
                samp = SamplerDesc(mag_filter="nearest", min_filter="nearest", address_mode_u=mode, address_mode_v=mode)
                mode_renders = {}
                for label, target, provider in self._active_graphics_backends():
                    with self.subTest(mode=mode, backend=label):
                        prog = load_program(manifests[target], textures={"tex": tex}, samplers={"samp": samp})
                        img = self._render_backend(target, prog, 12, 2, provider)
                        mode_renders[label] = img.rgba
                        for x_col, want in expected_cols.items():
                            got = tuple(img.rgba[x_col * 4 : x_col * 4 + 4])
                            self.assertEqual(got, want, f"Column {x_col} mismatch for mode={mode} on {label}")
                ref = next(iter(mode_renders.values()))
                for label, rgba in mode_renders.items():
                    self.assertEqual(rgba, ref, f"Cross-backend parity mismatch for mode={mode} on {label}")

    def test_vertex_sample_level_and_fragment_load_texture(self):
        @vertex
        def vs_tex_fetch(index: u32, pal_tex: Texture2D[0], pal_samp: Sampler[1]) -> Vertex:
            x = select(index == u32(1), 3.0, -1.0)
            y = select(index == u32(2), -3.0, 1.0)
            tint = sample_texture_level(pal_tex, pal_samp, vec2(0.75, 0.25), 0.0)
            return Vertex(vec4(x, y, 0.0, 1.0), varyings={"tint": tint})

        @fragment
        def fs_tex_load(tint, detail_tex: Texture2D[2]):
            texel = load_texture(detail_tex, (1, 1), 0)
            return tint * texel

        pipe = Pipeline(vs_tex_fetch, fs_tex_load, Draw(vertex_count=3))
        pal = TextureData.from_rgba8(2, 1, [(0, 0, 0, 255), (255, 128, 255, 255)])
        detail = TextureData.from_rgba8(
            2,
            2,
            [
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (0, 0, 0, 255),
                (128, 255, 64, 255),
            ],
        )
        samp = SamplerDesc(mag_filter="nearest", min_filter="nearest")

        with tempfile.TemporaryDirectory() as tmp:
            for label, target, provider in self._active_graphics_backends():
                with self.subTest(backend=label):
                    mpath = compile_pipeline(pipe, target, Path(tmp) / f"vsfs-{label}", TOOLS)
                    prog = load_program(
                        mpath,
                        textures={"pal_tex": pal, "detail_tex": detail},
                        samplers={"pal_samp": samp},
                    )
                    img = self._render_backend(target, prog, 4, 4, provider)
                    px = tuple(img.rgba[0:4])
                    self.assertAlmostEqual(px[0], 128, delta=2)
                    self.assertAlmostEqual(px[1], 128, delta=2)
                    self.assertAlmostEqual(px[2], 64, delta=2)
                    self.assertEqual(px[3], 255)

    def test_compute_sample_level_and_load_texture_parity(self):
        @compute(workgroup_size=(2, 2, 1))
        def tex_compute(
            src_tex: Texture2D[0],
            src_samp: Sampler[1],
            sampled_out: Buffer[Type.VEC4, "write_only", 2],
            loaded_out: Buffer[Type.VEC4, "write_only", 3],
            gid: GlobalInvocationId,
        ):
            gx = gid.x
            gy = gid.y
            idx = gy * u32(2) + gx
            uv = vec2((float(gx) + 0.5) * 0.5, (float(gy) + 0.5) * 0.5)
            sampled_out[idx] = sample_texture_level(src_tex, src_samp, uv, 0.0)
            loaded_out[idx] = load_texture(src_tex, (gx, gy), 0)

        cpipe = ComputePipeline(tex_compute)
        tex_f32 = TextureData.from_rgba_f32(
            2,
            2,
            [
                (0.125, 0.25, 0.5, 1.0),
                (0.75, 0.5, 0.25, 1.0),
                (0.0, 1.0, 0.5, 1.0),
                (1.0, 0.25, 0.75, 0.5),
            ],
        )
        samp = SamplerDesc(mag_filter="nearest", min_filter="nearest", address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge")

        expected = [
            (0.125, 0.25, 0.5, 1.0),
            (0.75, 0.5, 0.25, 1.0),
            (0.0, 1.0, 0.5, 1.0),
            (1.0, 0.25, 0.75, 0.5),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            for label, target, provider, val in [
                *([("vulkan", Target.VULKAN, None, self.vk_validation)] if self.has_vulkan else []),
                *(
                    [
                        ("directx-hw", Target.DIRECTX, "hardware", True),
                        ("directx-warp", Target.DIRECTX, "warp", True),
                    ]
                    if self.has_directx
                    else []
                ),
            ]:
                with self.subTest(backend=label):
                    mpath = compile_pipeline(cpipe, target, Path(tmp) / f"comp-{label}", TOOLS)
                    prog = load_compute_program(mpath)
                    res = dispatch_compute(
                        prog,
                        grid=(1, 1, 1),
                        buffers={"sampled_out": bytes(64), "loaded_out": bytes(64)},
                        textures={"src_tex": tex_f32},
                        samplers={"src_samp": samp},
                        validation=val,
                        toolchain=TOOLS,
                        provider=provider,
                    )
                    sampled_vals = [struct.unpack_from("<4f", res.buffers["sampled_out"], i * 16) for i in range(4)]
                    loaded_vals = [struct.unpack_from("<4f", res.buffers["loaded_out"], i * 16) for i in range(4)]
                    for i, want in enumerate(expected):
                        for ch in range(4):
                            self.assertAlmostEqual(sampled_vals[i][ch], want[ch], places=5)
                            self.assertAlmostEqual(loaded_vals[i][ch], want[ch], places=5)
