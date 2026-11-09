import contextlib
import io
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlir_gpu import Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import (
    GraphicsProgram, RenderedImage, Shader, ViewerError, _Vulkan, _video_encoder,
    export_mp4, load_program, main, render_offscreen,
)
from test_support import ROOT, TOOLS


def encoder_available():
    try:
        _video_encoder()
    except ViewerError:
        return False
    return True


class FakeEncoder:
    def __init__(self, arguments, **kwargs):
        self.arguments = arguments
        self.stdin = io.BytesIO()
        self.written_bytes = bytearray()
        orig_write = self.stdin.write

        def record_write(data):
            self.written_bytes.extend(data)
            return orig_write(data)

        self.stdin.write = record_write
        self.returncode = None
        self.terminated = False
        self.payload = b"\0\0\0\x20ftypisom" + bytes(32)
        self.on_wait = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            Path(self.arguments[-1]).write_bytes(self.payload)
            self.returncode = 0
        if self.on_wait:
            self.on_wait()
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


class VideoTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)
        self.path = self.directory / "animation.mp4"
        self.program = GraphicsProgram(Shader("vertex", b""), Shader("fragment", b""))
        self.times, self.encoders = [], []
        self.cleaned = False

    @contextlib.contextmanager
    def renderer(self, program, width, height, validation):
        def render(time):
            self.times.append(time)
            return RenderedImage(width, height, bytes(width * height * 4), "test GPU")
        try:
            yield render
        finally:
            self.cleaned = True

    def encoder(self, *args, **kwargs):
        encoder = FakeEncoder(*args, **kwargs)
        self.encoders.append(encoder)
        return encoder

    def test_fixed_sampling_rounds_duration_up_and_reuses_one_renderer(self):
        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer) as renderer:
                self.assertEqual(export_mp4(self.program, self.path, 16, 12, duration=0.8, fps=4, start_time=1.25), 4)
                renderer.assert_called_once_with(self.program, 16, 12, False)
        self.assertEqual(self.times, [1.25, 1.5, 1.75, 2.0])
        self.assertTrue(self.cleaned)
        arguments = self.encoders[0].arguments
        self.assertIn("libx264", arguments)
        self.assertIn("yuv420p", arguments)
        self.assertIn("+faststart", arguments)
        self.assertTrue(self.path.is_file())
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_invalid_video_arguments_fail_before_starting_encoder(self):
        for changes in (
            {"width": 15}, {"height": 13}, {"fps": 0}, {"fps": 241}, {"fps": True},
            {"fps": 29.97}, {"duration": 0}, {"duration": -1}, {"duration": float("nan")},
            {"start_time": -1}, {"start_time": float("inf")}, {"path": self.directory / "bad.png"},
            {"duration": 3e38, "start_time": 3e38},
        ):
            with self.subTest(changes=changes), patch("mlir_gpu.viewer.subprocess.Popen") as encoder:
                with self.assertRaises(ViewerError):
                    export_mp4(self.program, **{"path": self.path, "width": 16, "height": 12, **changes})
                encoder.assert_not_called()
        with self.assertRaisesRegex(ViewerError, "Vulkan"):
            export_mp4(GraphicsProgram(self.program.vertex, self.program.fragment, target="compute"), self.path, 16, 12)

    def test_decimal_duration_does_not_add_a_spurious_frame(self):
        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                self.assertEqual(export_mp4(self.program, self.path, 16, 12, duration=0.14, fps=100), 14)
        self.assertEqual(len(self.times), 14)

    def test_missing_encoder_is_actionable_and_does_not_render(self):
        with patch("mlir_gpu.viewer.shutil.which", return_value=None), patch.dict("sys.modules", {"imageio_ffmpeg": None}):
            with patch("mlir_gpu.viewer._offscreen_renderer") as render:
                with self.assertRaisesRegex(ViewerError, r"\.\[video\]"):
                    export_mp4(self.program, self.path, 16, 12)
                render.assert_not_called()

    def test_existing_output_and_publication_race_never_overwrite(self):
        self.path.write_bytes(b"keep")
        with patch("mlir_gpu.viewer._video_encoder") as find:
            with self.assertRaisesRegex(ViewerError, "already exists"):
                export_mp4(self.program, self.path, 16, 12)
            find.assert_not_called()
        self.path.unlink()

        def race(*args, **kwargs):
            encoder = self.encoder(*args, **kwargs)
            encoder.on_wait = lambda: self.path.write_bytes(b"keep")
            return encoder

        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=race):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                with self.assertRaisesRegex(ViewerError, "already exists"):
                    export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4)
        self.assertEqual(self.path.read_bytes(), b"keep")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_mp4_publication_falls_back_when_link_fails(self):
        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                with patch("mlir_gpu.viewer.os.link", side_effect=PermissionError("hard link forbidden")) as mock_link:
                    self.assertEqual(export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4), 1)
                    mock_link.assert_called_once()
        self.assertTrue(self.path.is_file())
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_mp4_publication_falls_back_to_copy_when_link_and_rename_fail(self):
        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                with patch("mlir_gpu.viewer.os.link", side_effect=PermissionError("hard link forbidden")):
                    with patch("mlir_gpu.viewer.os.rename", side_effect=OSError("cross-device link")):
                        self.assertEqual(export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4), 1)
        self.assertTrue(self.path.is_file())
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_mp4_publication_race_with_link_permission_error_never_overwrites(self):
        def race(*args, **kwargs):
            encoder = self.encoder(*args, **kwargs)
            encoder.on_wait = lambda: self.path.write_bytes(b"keep")
            return encoder

        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=race):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                with patch("mlir_gpu.viewer.os.link", side_effect=PermissionError("hard link forbidden")):
                    with self.assertRaisesRegex(ViewerError, "already exists"):
                        export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4)
        self.assertEqual(self.path.read_bytes(), b"keep")
        self.assertEqual(list(self.directory.iterdir()), [self.path])

    def test_mp4_publication_copy_failure_cleans_up_destination(self):
        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                with patch("mlir_gpu.viewer.os.link", side_effect=PermissionError("hard link forbidden")):
                    with patch("mlir_gpu.viewer.os.rename", side_effect=OSError("cross-device link")):
                        with patch("mlir_gpu.viewer.shutil.copyfileobj", side_effect=OSError("disk full")):
                            with self.assertRaisesRegex(ViewerError, "Failed to publish MP4 output"):
                                export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_render_and_invalid_encoder_output_fail_without_partial_video(self):
        @contextlib.contextmanager
        def failed(*args):
            def render(time):
                raise ViewerError("GPU readback failed")
            yield render

        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=failed):
                with self.assertRaisesRegex(ViewerError, "GPU readback failed"):
                    export_mp4(self.program, self.path, 16, 12)
        self.assertTrue(self.encoders[-1].terminated)
        self.assertFalse(self.path.exists())

        def corrupt(*args, **kwargs):
            encoder = self.encoder(*args, **kwargs)
            encoder.payload = b"not an mp4" + bytes(32)
            return encoder
        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=corrupt):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                with self.assertRaisesRegex(ViewerError, "valid MP4"):
                    export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_partial_pipe_writes_and_wrong_frame_shape(self):
        class PartialInput(io.BytesIO):
            total = 0

            def write(self, data):
                written = super().write(data[:17])
                self.total += written
                return written

        def partial(*args, **kwargs):
            encoder = self.encoder(*args, **kwargs)
            encoder.stdin = PartialInput()
            return encoder

        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=partial):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=self.renderer):
                self.assertEqual(export_mp4(self.program, self.path, 16, 12, duration=0.25, fps=4), 1)
        self.assertEqual(self.encoders[-1].stdin.total, 16 * 12 * 4)
        self.path.unlink()

        @contextlib.contextmanager
        def wrong_shape(*args):
            yield lambda time: RenderedImage(15, 12, bytes(15 * 12 * 4), "test GPU")

        with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
            with patch("mlir_gpu.viewer._offscreen_renderer", side_effect=wrong_shape):
                with self.assertRaisesRegex(ViewerError, "RGBA frame"):
                    export_mp4(self.program, self.path, 16, 12)
        self.assertTrue(self.encoders[-1].terminated)
        self.assertFalse(self.path.exists())

    def test_cli_video_controls_are_explicit_and_do_not_change_png_or_window_modes(self):
        with patch("sys.argv", ["mlir-gpu-view", "pipeline.json", "--video", str(self.path), "--duration", "2",
                                "--fps", "24", "--time", "1.25"]), patch("mlir_gpu.viewer.load_program", return_value=self.program):
            with patch("mlir_gpu.viewer.export_mp4", return_value=48) as export, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
                self.assertEqual(export.call_args.args[1:], (self.path, 800, 600, 2.0, 24, 1.25, False))
        for extra in (["--video", str(self.path), "--output", "image.png"],
                      ["--video", str(self.path), "--frames", "0"], ["--fps", "24"], ["--duration", "2"]):
            with patch("sys.argv", ["mlir-gpu-view", "pipeline.json", *extra]), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)

    def test_directx_video_routing_on_windows_and_macos_and_rejects_linux(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        # 1. On Windows, export succeeds and routes through _directx_offscreen_renderer
        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=self.renderer) as render:
                    self.assertEqual(export_mp4(program, self.path, 16, 12, duration=0.5, fps=4, start_time=1.25), 2)
                    render.assert_called_once_with(program, 16, 12, False)
        self.assertEqual(self.times, [1.25, 1.5])
        self.assertTrue(self.cleaned)
        self.assertTrue(self.path.is_file())
        self.path.unlink()
        self.times.clear()
        self.cleaned = False

        # 2. On Linux (non-Windows/non-macOS), rejects with ViewerError before launching encoder
        with patch("mlir_gpu.viewer.sys.platform", "linux"), patch("mlir_gpu.viewer._video_encoder") as encoder:
            with self.assertRaisesRegex(ViewerError, "Native DirectX 12 MP4"):
                export_mp4(program, self.path, 16, 12)
            encoder.assert_not_called()

        # 3. On macOS, continues routing through _metal_offscreen_renderer
        with patch("mlir_gpu.viewer.sys.platform", "darwin"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._metal_offscreen_renderer", side_effect=self.renderer) as render:
                    self.assertEqual(export_mp4(program, self.path, 16, 12, duration=0.5, fps=4, start_time=1.25), 2)
                    render.assert_called_once_with(program, 16, 12, False)
        self.assertEqual(self.times, [1.25, 1.5])

    # Backward compatibility alias
    test_directx_video_preserves_metal_routing_and_rejects_unsupported_windows_export = (
        test_directx_video_routing_on_windows_and_macos_and_rejects_linux
    )

    def test_directx_video_export_frame_count_dimensions_and_piping(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        width, height = 32, 24
        duration, fps, start_time = 0.75, 4, 10.0
        expected_frames = 3
        expected_times = [10.0, 10.25, 10.5]
        frame_size = width * height * 4
        rendered_payloads = [bytes([(i * 37) % 256] * frame_size) for i in range(expected_frames)]
        frame_idx = 0

        @contextlib.contextmanager
        def mock_dx_renderer(prog, w, h, val):
            self.assertEqual(prog, program)
            self.assertEqual((w, h), (width, height))
            self.assertFalse(val)

            def render(t):
                nonlocal frame_idx
                self.times.append(t)
                p = rendered_payloads[frame_idx]
                frame_idx += 1
                return RenderedImage(w, h, p, "DirectX 12 (NVIDIA)")

            try:
                yield render
            finally:
                self.cleaned = True

        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=mock_dx_renderer) as mock_render:
                    count = export_mp4(
                        program, self.path, width, height,
                        duration=duration, fps=fps, start_time=start_time, validation=False,
                    )
                    self.assertEqual(count, expected_frames)
                    mock_render.assert_called_once_with(program, width, height, False)

        self.assertEqual(self.times, expected_times)
        self.assertTrue(self.cleaned)
        self.assertTrue(self.path.is_file())

        encoder = self.encoders[-1]
        args = encoder.arguments
        self.assertIn("-video_size", args)
        self.assertEqual(args[args.index("-video_size") + 1], f"{width}x{height}")
        self.assertIn("-framerate", args)
        self.assertEqual(args[args.index("-framerate") + 1], str(fps))
        self.assertIn("-frames:v", args)
        self.assertEqual(args[args.index("-frames:v") + 1], str(expected_frames))
        self.assertIn("-pix_fmt", args)
        self.assertEqual(args[args.index("-pix_fmt") + 1], "yuv420p")
        self.assertIn("-c:v", args)
        self.assertEqual(args[args.index("-c:v") + 1], "libx264")
        self.assertIn("-movflags", args)
        self.assertEqual(args[args.index("-movflags") + 1], "+faststart")
        self.assertEqual(encoder.written_bytes, b"".join(rendered_payloads))

    def test_directx_video_export_validation_propagation(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=self.renderer) as render:
                    count = export_mp4(program, self.path, 16, 12, duration=0.25, fps=4, validation=True)
                    self.assertEqual(count, 1)
                    render.assert_called_once_with(program, 16, 12, True)

    def test_directx_video_export_frame_shape_mismatch_terminates_encoder(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")

        @contextlib.contextmanager
        def wrong_shape(*args):
            yield lambda t: RenderedImage(15, 12, bytes(15 * 12 * 4), "DirectX 12")

        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=wrong_shape):
                    with self.assertRaisesRegex(ViewerError, "RGBA frame"):
                        export_mp4(program, self.path, 16, 12, duration=0.5, fps=4)
        self.assertTrue(self.encoders[-1].terminated)
        self.assertFalse(self.path.exists())

    def test_directx_video_export_render_failure_terminates_encoder_and_cleans_up(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        call_count = 0

        @contextlib.contextmanager
        def failing_renderer(*args):
            def render(t):
                nonlocal call_count
                call_count += 1
                if call_count >= 2:
                    raise ViewerError("DirectX 12 GPU command list failed (0x887A0005)")
                return RenderedImage(16, 12, bytes(16 * 12 * 4), "DirectX 12")

            yield render

        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=failing_renderer):
                    with self.assertRaisesRegex(ViewerError, "DirectX 12 GPU command list failed"):
                        export_mp4(program, self.path, 16, 12, duration=1.0, fps=4)
        self.assertTrue(self.encoders[-1].terminated)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_directx_video_export_broken_pipe_during_streaming(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")

        class BrokenPipeStdin(io.BytesIO):
            def write(self, data):
                raise BrokenPipeError("Encoder pipe closed unexpectedly")

        def broken_encoder(*args, **kwargs):
            enc = self.encoder(*args, **kwargs)
            enc.stdin = BrokenPipeStdin()
            enc.returncode = 1
            return enc

        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=broken_encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=self.renderer):
                    with self.assertRaisesRegex(ViewerError, "MP4 encoder failed"):
                        export_mp4(program, self.path, 16, 12, duration=0.5, fps=4)
        self.assertFalse(self.path.exists())
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_directx_video_export_with_frame_parameters(self):
        from test_parameters import FRAME_REFLECTION
        program = GraphicsProgram(
            self.program.vertex, self.program.fragment, target="directx",
            parameters=FRAME_REFLECTION,
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=self.renderer) as render:
                    count = export_mp4(program, self.path, 16, 12, duration=0.25, fps=4)
                    self.assertEqual(count, 1)
                    render.assert_called_once_with(program, 16, 12, False)

    def test_offscreen_renderer_delegates_to_directx_on_windows(self):
        from mlir_gpu.viewer import _offscreen_renderer
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=self.renderer) as dx_render:
                with _offscreen_renderer(program, 16, 12, False) as render:
                    image = render(2.5)
                    self.assertEqual((image.width, image.height), (16, 12))
                dx_render.assert_called_once_with(program, 16, 12, False)
        self.assertEqual(self.times, [2.5])

    def test_directx_video_exact_time_stepping(self):
        """Verify that DirectX video export queries exact timestamps start_time + i / fps."""
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        width, height = 32, 24
        duration, fps, start_time = 0.5, 10, 1.0  # 5 frames: 1.0, 1.1, 1.2, 1.3, 1.4
        expected_times = [round(1.0 + i * 0.1, 7) for i in range(5)]
        recorded_times = []

        @contextlib.contextmanager
        def mock_dx(prog, w, h, val):
            def render(t):
                recorded_times.append(round(t, 7))
                return RenderedImage(w, h, bytes(w * h * 4), "DirectX 12")
            yield render

        with patch("mlir_gpu.viewer.sys.platform", "win32"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
                 patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._directx_offscreen_renderer", side_effect=mock_dx):
                    count = export_mp4(program, self.path, width, height, duration=duration, fps=fps, start_time=start_time)
                    self.assertEqual(count, 5)

        self.assertEqual(recorded_times, expected_times)


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1" and encoder_available(),
    "Enable DirectX tests on Windows and install FFmpeg for native DirectX video verification",
)
class DirectXVideoRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if sys.platform != "win32":
            raise unittest.SkipTest("DirectX rendering requires Windows")
        cls.tools = Toolchain(
            Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None,
            Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None,
        )
        try:
            cls.tools.find("mlir-gpu-directx-view")
        except Exception:
            raise unittest.SkipTest("mlir-gpu-directx-view binary not found")
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.directory = Path(work.name)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

        from mlir_gpu import FrameParameters, Pipeline, fragment, fullscreen_vertex, vec4

        @fragment
        def moving_pattern(frame: FrameParameters):
            return vec4(frame.resolution / 255.0, frame.time / 4.0, 1.0)

        pipe = Pipeline(fullscreen_vertex, moving_pattern)
        manifest = compile_pipeline(pipe, Target.DIRECTX, cls.directory / "dx_pipe", cls.tools)
        cls.program = load_program(manifest)

    def test_real_directx_mp4_animation_and_decoding(self):
        from mlir_gpu.viewer import render_directx_offscreen
        output = self.directory / "dx_animation.mp4"
        width, height = 64, 48
        duration, fps, start_time = 1.0, 4, 0.0
        expected_frames = 4

        count = export_mp4(
            self.program, output, width, height,
            duration=duration, fps=fps, start_time=start_time,
            validation=self.validation,
        )
        self.assertEqual(count, expected_frames)
        self.assertTrue(output.is_file())

        decoded = subprocess.run(
            [_video_encoder(), "-hide_banner", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
            capture_output=True, check=True,
        )
        report = decoded.stderr.decode()
        self.assertIn("Video: h264", report)
        self.assertIn(f"{width}x{height}", report)
        self.assertIn("4 fps", report)
        self.assertIn("Duration: 00:00:01.00", report)

        frame_size = width * height * 4
        self.assertEqual(len(decoded.stdout), expected_frames * frame_size)

        frame0 = decoded.stdout[:frame_size]
        frame3 = decoded.stdout[-frame_size:]
        self.assertNotEqual(frame0, frame3, "Frame 0 and Frame 3 must differ due to animation")

        for idx, t in enumerate([0.0, 0.25, 0.5, 0.75]):
            expected = render_directx_offscreen(self.program, width, height, self.validation, self.tools, time=t).rgba
            actual = decoded.stdout[idx * frame_size:(idx + 1) * frame_size]
            mean_diff = sum(abs(a - b) for i, (a, b) in enumerate(zip(actual, expected)) if i % 4 != 3) / (width * height * 3)
            self.assertLess(mean_diff, 8.0)


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and encoder_available(),
                     "Enable Vulkan tests and install FFmpeg or the video extra")
class VideoRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.directory = Path(work.name)
        pipeline = runpy.run_path(str(ROOT / "examples/orbiting_sphere.py"))["pipeline"]
        cls.program = load_program(compile_pipeline(pipeline, Target.VULKAN, cls.directory / "pipeline", TOOLS))
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def test_real_gpu_mp4_decodes_correct_shape_rate_duration_and_visible_animation(self):
        output = self.directory / "orbit.mp4"
        original = _Vulkan.pipeline
        with patch.object(_Vulkan, "pipeline", autospec=True, side_effect=original) as pipelines:
            count = export_mp4(self.program, output, 64, 48, duration=2, fps=4, validation=self.validation)
            self.assertEqual(pipelines.call_count, 1)
        self.assertEqual(count, 8)
        decoded = subprocess.run(
            [_video_encoder(), "-hide_banner", "-i", str(output), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"],
            capture_output=True, check=True,
        )
        report = decoded.stderr.decode()
        self.assertIn("Video: h264", report)
        self.assertIn("64x48", report)
        self.assertIn("4 fps", report)
        self.assertIn("Duration: 00:00:02.00", report)
        size = 64 * 48 * 4
        self.assertEqual(len(decoded.stdout), count * size)
        first, last = decoded.stdout[:size], decoded.stdout[-size:]
        changed = sum(first[i:i + 3] != last[i:i + 3] for i in range(0, size, 4))
        self.assertGreater(changed / (64 * 48), 0.10)
        def center(frame):
            columns = [index % 64 for index in range(64 * 48)
                       if max(frame[index * 4:index * 4 + 3]) > 60]
            return sum(columns) / len(columns)
        self.assertGreater(center(last) - center(first), 6)
        for index in (0, count - 1):
            expected = render_offscreen(self.program, 64, 48, self.validation, time=index / 4).rgba
            actual = decoded.stdout[index * size:(index + 1) * size]
            mean_error = sum(abs(a - b) for i, (a, b) in enumerate(zip(actual, expected)) if i % 4 != 3) / (64 * 48 * 3)
            self.assertLess(mean_error, 6)

    def test_real_encoder_failure_preserves_diagnostics_and_cleans_up(self):
        output = self.directory / "failed.mp4"
        original = subprocess.Popen
        processes = []

        def invalid_codec(arguments, **kwargs):
            arguments = list(arguments)
            arguments[arguments.index("-c:v") + 1] = "gfxir_missing_encoder"
            process = original(arguments, **kwargs)
            processes.append(process)
            return process

        with patch("mlir_gpu.viewer.subprocess.Popen", side_effect=invalid_codec):
            with self.assertRaisesRegex(ViewerError, "gfxir_missing_encoder"):
                export_mp4(self.program, output, 64, 48, duration=2, fps=4, validation=self.validation)
        self.assertTrue(all(process.poll() is not None for process in processes))
        self.assertFalse(output.exists())
        self.assertFalse(list(self.directory.glob(".mlir-gpu-video-*")))
