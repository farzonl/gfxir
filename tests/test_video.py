import contextlib
import io
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlir_gpu import Target, compile_pipeline
from mlir_gpu.viewer import (
    GraphicsProgram, RenderedImage, Shader, ViewerError, _Vulkan, _video_encoder,
    export_mp4, load_program, main, render_offscreen,
)
from test_native_ir import ROOT, TOOLS


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

    def test_directx_video_preserves_metal_routing_and_rejects_unsupported_windows_export(self):
        program = GraphicsProgram(self.program.vertex, self.program.fragment, target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch("mlir_gpu.viewer._video_encoder") as encoder:
            with self.assertRaisesRegex(ViewerError, "Native DirectX 12 MP4"):
                export_mp4(program, self.path, 16, 12)
            encoder.assert_not_called()
        with patch("mlir_gpu.viewer.sys.platform", "darwin"):
            with patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), patch("mlir_gpu.viewer.subprocess.Popen", side_effect=self.encoder):
                with patch("mlir_gpu.viewer._metal_offscreen_renderer", side_effect=self.renderer) as render:
                    self.assertEqual(export_mp4(program, self.path, 16, 12, duration=0.5, fps=4, start_time=1.25), 2)
                    render.assert_called_once_with(program, 16, 12, False)
        self.assertEqual(self.times, [1.25, 1.5])


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
