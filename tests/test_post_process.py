from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from mlir_gpu import (
    DepthStencil, Draw, FrameParameters, Pipeline, Sampler, SamplerDesc, Target, Texture2D, TextureData, compile_pipeline,
    fragment, fullscreen_uv_vertex, load_texture, sample_texture, vec4,
)
from mlir_gpu.viewer import (
    GraphicsProgram, PostProcess, Shader, ViewerError, _Vulkan, _directx_shaders, _metal_libraries,
    _offscreen_renderer, _write_native_graphics_resources, load_program,
    render_directx_offscreen, render_metal_offscreen, render_offscreen,
    show_directx_window, show_metal_window, show_window, with_post_process,
)
from test_support import TOOLS, available


@fragment
def hdr_scene(uv, frame: FrameParameters):
    return vec4(uv.x * 4.0 + frame.time, uv.y * 2.0, 2.0, 0.75)


@fragment
def post_fragment(uv, scene_color: Texture2D[0], scene_sampler: Sampler[1]):
    color = sample_texture(scene_color, scene_sampler, uv)
    return vec4(color.x / (1.0 + color.x), color.y / (1.0 + color.y), color.z / (1.0 + color.z), color.w)


@fragment
def post_load_fragment(uv, frame: FrameParameters, scene_color: Texture2D[0]):
    color = load_texture(scene_color, (int(uv.x * frame.resolution.x), int(uv.y * frame.resolution.y)), 0)
    return vec4(color.x / (1.0 + color.x), color.y / (1.0 + color.y), color.z / (1.0 + color.z), color.w)


def _post_program():
    shader = Shader("unused", b"")
    return GraphicsProgram(
        shader, shader,
        texture_bindings=({"name": "scene_color", "set": 0, "binding": 0, "stages": ["fragment"]},),
        sampler_bindings=({"name": "scene_sampler", "set": 0, "binding": 1, "stages": ["fragment"]},),
    )


class PostProcessValidationTests(unittest.TestCase):
    def test_invalid_passes_rejected(self):
        p = _post_program()
        for bad in (
            replace(p, draw=Draw(vertex_count=6)),
            replace(p, texture_bindings=()),
            replace(p, depth_stencil={"format": "d32_sfloat"}),
            replace(p, vertex_buffer=b"data"),
            replace(p, sampler_bindings=p.sampler_bindings, samplers={}),
            replace(p, texture_bindings=({**p.texture_bindings[0], "stages": ["vertex"]},)),
        ):
            with self.subTest(program=bad), self.assertRaises(ViewerError):
                PostProcess(bad)
        with self.assertRaises(ViewerError):
            PostProcess(p, format="bgra8_unorm")
        with self.assertRaises(ViewerError):
            with_post_process(p, replace(p, target="metal"))
        combined = with_post_process(GraphicsProgram(p.vertex, p.fragment), p)
        with self.assertRaises(ViewerError):
            with_post_process(combined, p)
        with self.assertRaises(ViewerError):
            PostProcess(combined)

    def test_native_render_target_has_no_cpu_payload(self):
        p = PostProcess(_post_program()).program
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = _write_native_graphics_resources(directory, p, render_target="scene_color")
            spec = json.loads(path.read_text())
            self.assertEqual(spec["fragment"]["textures"], [{
                "name": "scene_color", "set": 0, "binding": 0, "render_target": True,
            }])
            self.assertEqual(spec["fragment"]["samplers"][0]["min_filter"], "nearest")
            self.assertEqual(list(directory.glob("*.bin")), [])

    def test_native_post_spec_keeps_both_shader_sets_alive(self):
        for platform, target, prepare in (
            ("darwin", "metal", _metal_libraries), ("win32", "directx", _directx_shaders),
        ):
            post = replace(_post_program(), target=target)
            scene = GraphicsProgram(post.vertex, post.fragment, target=target)
            program = with_post_process(scene, post)
            with self.subTest(target=target), patch("mlir_gpu.viewer.sys.platform", platform):
                tools = Mock()
                tools.find.return_value = "viewer"
                with prepare(program, tools) as (directory, arguments):
                    self.assertEqual(arguments[arguments.index("--color-format") + 1], "rgba32_sfloat")
                    spec = json.loads(Path(arguments[arguments.index("--post-spec") + 1]).read_text())
                    for key in ("vertex", "fragment", "resources"):
                        self.assertTrue(Path(spec[key]).exists())
                    self.assertEqual(spec["vertex_count"], "3")
                self.assertFalse(directory.exists())
                self.assertFalse(Path(spec["vertex"]).exists())


@unittest.skipUnless(available("gfx-opt", "gfx-compile"), "Native compiler required")
class PostProcessGpuTests(unittest.TestCase):
    def _backends(self):
        backends = []
        if os.environ.get("MLIR_GPU_TEST_VULKAN") == "1":
            backends.append((Target.VULKAN, render_offscreen))
        if sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1":
            backends.append((Target.METAL, render_metal_offscreen))
            backends.append((Target.DIRECTX, render_metal_offscreen))
        if sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1":
            backends.append((Target.DIRECTX, render_directx_offscreen))
        if not backends:
            self.skipTest("Enable MLIR_GPU_TEST_VULKAN, METAL, or DIRECTX for GPU execution")
        return backends

    def _compile(self, directory, target):
        scene_path = compile_pipeline(
            Pipeline(fullscreen_uv_vertex, hdr_scene, depth_stencil=DepthStencil()),
            target, directory / "scene", TOOLS,
        )
        post_path = compile_pipeline(Pipeline(fullscreen_uv_vertex, post_fragment), target, directory / "post", TOOLS)
        return with_post_process(load_program(scene_path), load_program(post_path))

    def _assert_pixels(self, image, time, ldr=False):
        self.assertEqual(len(image.rgba), image.width * image.height * 4)
        for y in range(image.height):
            for x in range(image.width):
                hdr = (4 * (x + 0.5) / image.width + time, 2 * (y + 0.5) / image.height, 2.0)
                if ldr:
                    hdr = tuple(min(1.0, c) for c in hdr)
                expected = tuple(round(c / (1 + c) * 255) for c in hdr) + (191,)
                offset = (y * image.width + x) * 4
                for actual, want in zip(image.rgba[offset:offset + 4], expected):
                    self.assertAlmostEqual(actual, want, delta=2)

    def test_hdr_post_pass_orientation_alpha_and_dimensions(self):
        for target, render in self._backends():
            with self.subTest(target=target), tempfile.TemporaryDirectory() as tmp:
                program = self._compile(Path(tmp), target)
                for width, height, time in ((11, 7, 0.0), (5, 9, 1.5)):
                    kwargs = {"toolchain": TOOLS} if target != Target.VULKAN else {}
                    image = render(program, width, height, validation=True, time=time, **kwargs)
                    self._assert_pixels(image, time)
                ldr = replace(program, post_process=replace(program.post_process, format="rgba8_unorm"))
                self._assert_pixels(render(ldr, 5, 3, validation=True, **kwargs), 0.0, ldr=True)

    def test_persistent_vulkan_target_is_rendered_again_each_frame(self):
        if os.environ.get("MLIR_GPU_TEST_VULKAN") != "1":
            self.skipTest("Vulkan GPU execution is opt-in")
        with tempfile.TemporaryDirectory() as tmp:
            program = self._compile(Path(tmp), Target.VULKAN)
            with _offscreen_renderer(program, 8, 6, True) as render:
                for time in (0.0, 2.0, 0.5):
                    self._assert_pixels(render(time), time)

    def test_rgba32_texel_load_does_not_require_filtering(self):
        for target, render in self._backends():
            with self.subTest(target=target, renderer=render), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                scene_path = compile_pipeline(
                    Pipeline(fullscreen_uv_vertex, hdr_scene), target, directory / "scene", TOOLS,
                )
                post_path = compile_pipeline(
                    Pipeline(fullscreen_uv_vertex, post_load_fragment), target, directory / "post", TOOLS,
                )
                program = with_post_process(load_program(scene_path), load_program(post_path), format="rgba32_sfloat")
                kwargs = {"toolchain": TOOLS} if render is not render_offscreen else {}
                self._assert_pixels(render(program, 11, 7, validation=True, time=1.5, **kwargs), 1.5)

    def test_post_shader_reads_uploaded_hdr_texture(self):
        width, height, time = 11, 7, 1.5
        texture = TextureData.from_rgba_f32(width, height, [
            (4 * (x + 0.5) / width + time, 2 * (y + 0.5) / height, 2.0, 0.75)
            for y in range(height) for x in range(width)
        ])
        sampler = SamplerDesc(
            min_filter="nearest", mag_filter="nearest",
            address_mode_u="clamp_to_edge", address_mode_v="clamp_to_edge",
        )
        for target, render in self._backends():
            for shader in (post_fragment, post_load_fragment):
                with self.subTest(target=target, shader=shader), tempfile.TemporaryDirectory() as tmp:
                    path = compile_pipeline(Pipeline(fullscreen_uv_vertex, shader), target, Path(tmp) / "post", TOOLS)
                    program = load_program(
                        path, textures={"scene_color": texture},
                        samplers={"scene_sampler": sampler} if shader is post_fragment else {},
                    )
                    kwargs = {"toolchain": TOOLS} if render is not render_offscreen else {}
                    for validation in (False, True):
                        with self.subTest(validation=validation):
                            self._assert_pixels(
                                render(program, width, height, validation=validation, time=time, **kwargs), time,
                            )

    def test_emissive_demo_hdr_highlights_differ_from_clipped_ldr(self):
        from examples.basics.post_process import make_post, make_scene

        for target, render in self._backends():
            with self.subTest(target=target, renderer=render), tempfile.TemporaryDirectory() as tmp:
                directory = Path(tmp)
                scene_path = compile_pipeline(make_scene(split=True), target, directory / "scene", TOOLS)
                post_path = compile_pipeline(make_post(), target, directory / "post", TOOLS)
                program = with_post_process(load_program(scene_path), load_program(post_path))
                kwargs = {"toolchain": TOOLS} if render is not render_offscreen else {}
                image = render(program, 320, 180, validation=True, time=2.0, **kwargs)
                self.assertEqual(len(image.rgba), 320 * 180 * 4)
                different = 0
                dark = 0
                for y in range(180):
                    for x in range(159):
                        left = (y * 320 + x) * 4
                        right = left + 160 * 4
                        a = image.rgba[left:left + 3]
                        b = image.rgba[right:right + 3]
                        if max(abs(i - j) for i, j in zip(a, b)) > 24:
                            different += 1
                        if max(b) < 16:
                            dark += 1
                self.assertGreater(different, 320 * 180 * 0.01)
                self.assertGreater(dark, 320 * 180 * 0.01)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Window tests require a desktop session")
    def test_post_pass_window_three_frames(self):
        for target, render in self._backends():
            with self.subTest(target=target, renderer=render), tempfile.TemporaryDirectory() as tmp:
                program = self._compile(Path(tmp), target)
                if render is render_metal_offscreen:
                    show_metal_window(program, 160, 120, frames=3, validation=True, toolchain=TOOLS, time=1.0)
                elif target == Target.DIRECTX:
                    show_directx_window(program, 160, 120, frames=3, validation=True, toolchain=TOOLS, time=1.0)
                else:
                    self.assertEqual(show_window(program, 160, 120, frames=3, validation=True, time=1.0), 3)

    @unittest.skipUnless(
        os.environ.get("MLIR_GPU_TEST_WINDOW") == "1" and os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
        "Vulkan window resize requires a desktop session",
    )
    def test_vulkan_post_target_recreated_at_drawable_size(self):
        import glfw
        original_size = glfw.get_framebuffer_size
        original_image = _Vulkan.image
        target_sizes = []
        queries = 0

        def image(gpu, resources, width, height, format, **kwargs):
            if kwargs.get("sampled"):
                target_sizes.append((width, height))
            return original_image(gpu, resources, width, height, format, **kwargs)

        def resize(window):
            nonlocal queries
            queries += 1
            if queries == 2:
                glfw.set_window_size(window, 240, 180)
                glfw.poll_events()
            return original_size(window)

        with tempfile.TemporaryDirectory() as tmp:
            program = self._compile(Path(tmp), Target.VULKAN)
            with patch.object(glfw, "get_framebuffer_size", side_effect=resize), patch.object(_Vulkan, "image", image):
                self.assertEqual(show_window(program, 160, 120, frames=3, validation=True), 3)
        self.assertGreater(queries, 2)
        self.assertGreater(len(set(target_sizes)), 1)
        self.assertGreater(target_sizes[-1][0], target_sizes[0][0])
        self.assertGreater(target_sizes[-1][1], target_sizes[0][1])


if __name__ == "__main__":
    unittest.main()
