"""Nacelle provider routing and opt-in native DXIL execution."""

import hashlib
import io
import os
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import Mock, patch

from mlir_gpu import (
    Buffer, ComputePipeline, DepthStencil, Pipeline, Target, Type, compile_pipeline,
    compute, fullscreen_uv_vertex, global_invocation_id, open_compute_context,
)
from mlir_gpu.viewer import (
    ComputeProgram, GraphicsProgram, RenderedImage, Shader, ViewerError, _check_nacelle_device,
    _offscreen_renderer, _validate_provider, compute_session, dispatch_compute,
    get_device_capabilities, load_compute_program, load_program, render_directx_offscreen,
    render_offscreen, show_directx_window, with_post_process,
)
from test_post_process import hdr_scene, post_fragment
from test_support import TOOLS
from tools import setup_nacelle


@compute(workgroup_size=(8, 1, 1))
def nacelle_increment(source: Buffer[Type.U32, "read_only"], output: Buffer[Type.U32, "write_only"]):
    index = global_invocation_id().x
    output[index] = source[index] + 7


class NacelleProviderTests(unittest.TestCase):
    def test_minigfx_provider_validation_and_offscreen_forwarding(self):
        from examples.minigfx.app import App
        from examples.minigfx.samples.model_viewer import ModelViewerApp

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                App.parse_args(["--target", "vulkan", "--provider", "nacelle"])
        with patch("sys.platform", "win32"), patch("sys.stderr"), self.assertRaises(SystemExit):
            App.parse_args(["--target", "directx", "--provider", "nacelle"])
        with patch("sys.platform", "linux"):
            args = ModelViewerApp.parse_args([
                "--target", "directx", "--provider", "nacelle", "--offscreen", "--frames", "0",
                "--width", "8", "--height", "8",
            ])
        app = ModelViewerApp(args)
        try:
            with patch("examples.minigfx.samples.model_viewer.render_offscreen",
                       return_value=RenderedImage(8, 8, bytes(8 * 8 * 4), "Nacelle")) as render:
                app.render_offscreen_frame()
            self.assertEqual(render.call_args.kwargs["provider"], "nacelle")
        finally:
            app.cleanup()

    def test_vulkan_pixel_presenter_suboptimal_and_out_of_date(self):
        from examples.minigfx.window import _VulkanWindowBackend

        class Suboptimal(Exception):
            pass

        class OutOfDate(Exception):
            pass

        for acquire_error, present_error in ((Suboptimal, None), (None, Suboptimal), (OutOfDate, None)):
            with self.subTest(acquire_error=acquire_error, present_error=present_error):
                backend = _VulkanWindowBackend.__new__(_VulkanWindowBackend)
                vk = backend.vk = Mock()
                vk.VkSuboptimalKhr = Suboptimal
                vk.VkErrorOutOfDateKhr = OutOfDate
                vk.ffi.new.return_value = [0]
                acquire, present = Mock(side_effect=acquire_error), Mock(side_effect=present_error)
                vk.vkGetDeviceProcAddr.side_effect = (acquire, present)
                for attr in ("device", "swapchain", "fence", "img_avail", "render_done", "cmd_buf", "queue",
                             "_pixel_memory", "_pixel_staging"):
                    setattr(backend, attr, object())
                backend._pixels_only = backend._pixel_bgra = True
                backend.sc_w, backend.sc_h = 2, 1
                backend._pixel_images = [object()]
                backend._destroy_swapchain = Mock()
                backend._create_swapchain_and_pipelines = Mock()
                backend.present_pixels(2, 1, bytes((240, 32, 96, 255, 12, 24, 48, 255)))
                backend._create_swapchain_and_pipelines.assert_called_once_with(2, 1)
                if acquire_error is OutOfDate:
                    vk.vkResetFences.assert_not_called()
                    vk.vkQueueSubmit.assert_not_called()
                    present.assert_not_called()
                else:
                    vk.ffi.memmove.assert_called_once_with(
                        vk.vkMapMemory.return_value, bytes((96, 32, 240, 255, 48, 24, 12, 255)), 8,
                    )
                    vk.vkQueueSubmit.assert_called_once()
                    present.assert_called_once()
                    vk.vkQueueWaitIdle.assert_called_once()

    def test_vulkan_initialization_preserves_shader_loading_failure(self):
        from examples.minigfx.window import _VulkanWindowBackend

        missing_shader = Mock()
        missing_shader.read_bytes.side_effect = FileNotFoundError("missing pixel test shader")
        with patch.dict(sys.modules, {"vulkan": Mock()}), self.assertRaisesRegex(FileNotFoundError, "missing pixel"):
            _VulkanWindowBackend(Mock(), object(), 2, 1, {"vs": missing_shader})

    def test_explicit_provider_routes_macos_graphics_and_compute_to_directx(self):
        program = GraphicsProgram(Shader("main", b"DXBC"), Shader("main", b"DXBC"), target="directx")
        for platform in ("linux", "darwin"):
            with self.subTest(platform=platform), patch("mlir_gpu.viewer.sys.platform", platform):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer") as renderer:
                    with _offscreen_renderer(program, 3, 2, False, provider="nacelle"):
                        pass
                    self.assertEqual(renderer.call_args.kwargs["provider"], "nacelle")
                with patch("mlir_gpu.viewer._directx_compute_session") as session:
                    compiled = ComputeProgram(Shader("main", b"DXBC"), target="directx")
                    with compute_session(compiled, provider="nacelle"):
                        pass
                    self.assertEqual(session.call_args.kwargs["provider"], "nacelle")

    def test_wrong_backend_window_and_driver_fallback_rejected(self):
        _validate_provider("nacelle", "directx")
        for backend in ("vulkan", "metal"):
            with self.assertRaisesRegex(ValueError, "not supported"):
                _validate_provider("nacelle", backend)
        with self.assertRaisesRegex(ViewerError, "fallback rejected"):
            _check_nacelle_device("nacelle", "Microsoft Basic Render Driver")
        program = GraphicsProgram(Shader("main", b"DXBC"), Shader("main", b"DXBC"), target="directx")
        with self.assertRaisesRegex(ViewerError, "headless"):
            show_directx_window(program, provider="nacelle")

    def test_offscreen_rejects_non_nacelle_readback(self):
        program = GraphicsProgram(Shader("main", b"DXBC"), Shader("main", b"DXBC"), target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "linux"), \
             patch.object(type(TOOLS), "find", return_value="native"), \
             patch("mlir_gpu.viewer._directx_run", return_value="{}"), \
             patch("mlir_gpu.viewer._readback_result", return_value=RenderedImage(1, 1, b"\0" * 4, "Other")):
            with self.assertRaisesRegex(ViewerError, "fallback rejected"):
                render_directx_offscreen(program, 1, 1, provider="nacelle")


class NacelleInstallerTests(unittest.TestCase):
    def test_verified_archive_installs_and_checksum_mismatch_leaves_no_sdk(self):
        archive = io.BytesIO()
        package = f"nacelle-{setup_nacelle.VERSION}-macos-arm64"
        with tarfile.open(fileobj=archive, mode="w:gz") as output:
            for name in ("include/nacelle/nacelle.h", "lib/libnacelle_d3d12.dylib"):
                member = tarfile.TarInfo(f"{package}/{name}")
                member.size = 3
                output.addfile(member, io.BytesIO(b"sdk"))
        data = archive.getvalue()
        for valid in (True, False):
            expected = hashlib.sha256(data).hexdigest() if valid else "0" * 64
            with self.subTest(valid=valid), tempfile.TemporaryDirectory() as tmp, \
                 patch.object(setup_nacelle.platform, "system", return_value="Darwin"), \
                 patch.object(setup_nacelle.platform, "machine", return_value="arm64"), \
                 patch.object(setup_nacelle, "RELEASES", {("Darwin", "arm64"): ("macos-arm64", expected)}), \
                 patch.object(setup_nacelle.urllib.request, "urlopen", return_value=io.BytesIO(data)):
                prefix = Path(tmp) / "sdk"
                if valid:
                    setup_nacelle.install(prefix)
                    self.assertEqual((prefix / "lib/libnacelle_d3d12.dylib").read_bytes(), b"sdk")
                else:
                    with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                        setup_nacelle.install(prefix)
                    self.assertFalse(prefix.exists())

    def test_unsupported_architecture_does_not_download(self):
        with patch.object(setup_nacelle.platform, "system", return_value="Linux"), \
             patch.object(setup_nacelle.platform, "machine", return_value="aarch64"), \
             patch.object(setup_nacelle.urllib.request, "urlopen") as download:
            with self.assertRaisesRegex(RuntimeError, "no supported release"):
                setup_nacelle.install(Path("unused"))
            download.assert_not_called()


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_NACELLE") == "1", "Set MLIR_GPU_TEST_NACELLE=1")
class NacelleExecutionTests(unittest.TestCase):
    def test_minigfx_persistent_bridge_materials_hdr_and_resize(self):
        from examples.minigfx.material import Material, MaterialParams, Sampler, Texture
        from examples.minigfx.samples.model_viewer import ModelViewerApp
        from examples.minigfx.window import InteractiveWindow, _NacelleWindowBackend, compile_minigfx_shaders

        shaders = compile_minigfx_shaders("directx")
        for hdr, comparison, prefix in ((True, False, "hdr"), (True, True, "compare")):
            compiled = compile_minigfx_shaders("directx", hdr=hdr, comparison=comparison)
            for label in ("pbr", "textured_pbr", "normal"):
                shaders[f"{prefix}_{label}_fs"] = compiled[f"{label}_fs"]
            shaders.update(post_vs=compiled["post_vs"], post_fs=compiled["post_fs"])
        presenter = Mock()
        presenter_class = "_MetalPixelPresenter" if sys.platform == "darwin" else "_VulkanWindowBackend"
        app = ModelViewerApp(ModelViewerApp.parse_args([
            "--target", "directx", "--provider", "nacelle", "--frames", "0", "--width", "320", "--height", "240",
        ]))
        window = InteractiveWindow.__new__(InteractiveWindow)
        window.window = object()
        window.target = "directx"
        window.glfw = Mock()
        window.glfw.get_framebuffer_size.side_effect = lambda _: (app.width, app.height)
        window.glfw.get_window_size.side_effect = lambda _: (app.width, app.height)
        with patch(f"examples.minigfx.window.{presenter_class}", return_value=presenter):
            window.backend = _NacelleWindowBackend(window.glfw, window.window, 320, 240, shaders)
        app._window = window
        app.is_running = True
        backend = window.backend
        try:
            self.assertEqual(backend.device_name, "Nacelle")
            context = backend._ctx
            captures = []
            for normal, hdr, exposure, dimensions in (
                (False, False, 1.0, (320, 240)),
                (True, False, 1.0, (320, 240)),
                (False, True, 0.5, (320, 240)),
                (False, True, 3.0, (336, 252)),
            ):
                app.handle_key_press("3" if normal else "1")
                app.renderer.hdr = hdr
                app.renderer.exposure = exposure
                app.resize(*dimensions)
                backend.request_frame_capture()
                app.render()
                width, height, rgba = backend.last_captured_frame
                self.assertEqual((width, height), dimensions)
                self.assertEqual(len(rgba), width * height * 4)
                center = (height // 2 * width + width // 2) * 4
                captures.append(rgba[center:center + 3])
                self.assertGreater(sum(captures[-1]), 20)
                self.assertEqual(backend._ctx, context)
                presenter.present_pixels.assert_called_with(width, height, rgba)
            self.assertNotEqual(captures[0], captures[1])
            self.assertNotEqual(captures[2], captures[3])
            app.renderer.hdr = False
            app.model_node.material = Material(
                params=MaterialParams(), albedo_texture=Texture.checkerboard(width=8, height=8, tile_size=2),
                sampler=Sampler.nearest(),
            )
            backend.request_frame_capture()
            app.render()
            self.assertEqual(presenter.present_pixels.call_count, 5)
        finally:
            app.cleanup()
        self.assertIsNone(backend._ctx)
        presenter.destroy.assert_called_once_with()

    def test_capabilities_and_unsupported_validation_fail_explicitly(self):
        capabilities = get_device_capabilities("directx", provider="nacelle", toolchain=TOOLS)
        self.assertEqual(capabilities["provider"], "nacelle")
        self.assertEqual(capabilities["status"], "EXECUTED")
        self.assertTrue(capabilities["is_software"])
        with self.assertRaisesRegex(RuntimeError, "does not provide D3D12 validation"):
            with open_compute_context("directx", provider="nacelle", validation=True):
                pass

    def test_hdr_post_process_pixels_dimensions_alpha_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            scene = compile_pipeline(
                Pipeline(fullscreen_uv_vertex, hdr_scene, depth_stencil=DepthStencil()),
                Target.DIRECTX, directory / "scene", TOOLS,
            )
            post = compile_pipeline(
                Pipeline(fullscreen_uv_vertex, post_fragment), Target.DIRECTX, directory / "post", TOOLS,
            )
            program = with_post_process(load_program(scene), load_program(post))
            for width, height, time in ((11, 7, 0.0), (5, 9, 1.5)):
                image = render_offscreen(program, width, height, time=time, provider="nacelle")
                self.assertEqual(image.device, "Nacelle")
                self.assertEqual((image.width, image.height), (width, height))
                self.assertEqual(len(image.rgba), width * height * 4)
                for y in range(height):
                    for x in range(width):
                        hdr = (4 * (x + 0.5) / width + time, 2 * (y + 0.5) / height, 2.0)
                        expected = tuple(round(c / (1 + c) * 255) for c in hdr) + (191,)
                        offset = (y * width + x) * 4
                        for actual, want in zip(image.rgba[offset:offset + 4], expected):
                            self.assertAlmostEqual(actual, want, delta=2)

    def test_subprocess_compute_executes_dxil_and_reads_back_exact_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = compile_pipeline(
                ComputePipeline(nacelle_increment), Target.DIRECTX, Path(tmp) / "compute", TOOLS,
            )
            program = load_compute_program(manifest)
            for start in (0, 100):
                values = tuple(range(start, start + 16))
                result = dispatch_compute(
                    program, provider="nacelle", grid=(2, 1, 1), toolchain=TOOLS,
                    buffers={"source": struct.pack("<16I", *values), "output": 64},
                )
                self.assertEqual(result.device, "Nacelle")
                self.assertEqual(struct.unpack("<16I", result.buffers["output"]), tuple(v + 7 for v in values))


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_NACELLE") == "1" and os.environ.get("MLIR_GPU_TEST_WINDOW") == "1",
    "Set MLIR_GPU_TEST_NACELLE=1 and MLIR_GPU_TEST_WINDOW=1 in a desktop session",
)
class NacelleWindowTests(unittest.TestCase):
    def test_live_glfw_input_resize_and_capture(self):
        from examples.minigfx.samples.model_viewer import ModelViewerApp

        app = ModelViewerApp(ModelViewerApp.parse_args([
            "--target", "directx", "--provider", "nacelle", "--width", "320", "--height", "240", "--frames", "3",
        ]))
        try:
            app.startup()
            app.is_running = True
            window = app._window
            backend = window.backend
            self.assertEqual(backend.device_name, "Nacelle")
            self.assertTrue(window.poll_and_sync(app))
            for key, width, height in (("3", 320, 240), ("H", 336, 252), ("C", 352, 264)):
                window._on_key(window.window, getattr(window.glfw, f"KEY_{key}"), 0, window.glfw.PRESS, 0)
                window.glfw.set_window_size(window.window, width, height)
                self.assertTrue(window.poll_and_sync(app))
                backend.request_frame_capture()
                app.render()
                w, h, rgba = backend.last_captured_frame
                self.assertEqual((w, h), window.glfw.get_framebuffer_size(window.window))
                self.assertEqual(len(rgba), w * h * 4)
                self.assertGreater(len(set(rgba)), 16)
        finally:
            app.cleanup()

    def test_vulkan_pixel_only_presenter_and_resize(self):
        import glfw
        from examples.minigfx.window import _VulkanWindowBackend

        self.assertTrue(glfw.init())
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        window = glfw.create_window(80, 60, "Nacelle pixel presenter test", None, None)
        self.assertTrue(window)
        presenter = None
        try:
            w, h = glfw.get_framebuffer_size(window)
            presenter = _VulkanWindowBackend(glfw, window, w, h, {}, pixels_only=True)
            for width, height in ((80, 60), (96, 72)):
                glfw.set_window_size(window, width, height)
                glfw.poll_events()
                w, h = glfw.get_framebuffer_size(window)
                presenter.present_pixels(w, h, bytes((240, 32, 96, 255)) * (w * h))
                self.assertEqual((presenter.sc_w, presenter.sc_h), (w, h))
            self.assertEqual(presenter.shaders, {})
        finally:
            if presenter is not None:
                presenter.destroy()
            glfw.destroy_window(window)
            glfw.terminate()
