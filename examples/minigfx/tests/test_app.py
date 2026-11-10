"""Unit tests for minigfx.app lifecycle, CLI parsing, timing, and cleanup guarantees."""

from __future__ import annotations

import argparse
import unittest
from unittest.mock import patch

import examples.minigfx as minigfx
from examples.minigfx.app import App, get_default_target, main


class TestPackageExport(unittest.TestCase):
    """Test package __init__ exports and metadata."""

    def test_package_exports(self):
        self.assertTrue(hasattr(minigfx, "App"))
        self.assertEqual(minigfx.__version__, "0.1.0")
        self.assertEqual(minigfx.__all__, ["App", "__version__"])


class TestCLIArgs(unittest.TestCase):
    """Test CLI argument parsing and validation."""

    def test_cli_defaults(self):
        args = App.parse_args([])
        self.assertEqual(args.target, get_default_target())
        self.assertIsNone(args.frames)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertFalse(args.offscreen)

    def test_explicit_targets(self):
        for target in ("vulkan", "directx", "metal"):
            args = App.parse_args(["--target", target])
            self.assertEqual(args.target, target)

    def test_invalid_target_rejected(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                App.parse_args(["--target", "opengl"])

    def test_explicit_frames(self):
        args = App.parse_args(["--frames", "10"])
        self.assertEqual(args.frames, 10)

    def test_zero_frames(self):
        args = App.parse_args(["--frames", "0"])
        self.assertEqual(args.frames, 0)

    def test_negative_frames_rejected(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                App.parse_args(["--frames", "-1"])

    def test_invalid_frames_type_rejected(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                App.parse_args(["--frames", "abc"])

    def test_dimensions(self):
        args = App.parse_args(["--width", "1920", "--height", "1080"])
        self.assertEqual(args.width, 1920)
        self.assertEqual(args.height, 1080)

    def test_zero_or_negative_dimensions_rejected(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                App.parse_args(["--width", "0"])
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                App.parse_args(["--height", "-10"])

    def test_offscreen_flag(self):
        args = App.parse_args(["--offscreen"])
        self.assertTrue(args.offscreen)

    def test_combined_flags(self):
        args = App.parse_args(
            ["--target", "metal", "--frames", "5", "--width", "800", "--height", "600", "--offscreen"]
        )
        self.assertEqual(args.target, "metal")
        self.assertEqual(args.frames, 5)
        self.assertEqual(args.width, 800)
        self.assertEqual(args.height, 600)
        self.assertTrue(args.offscreen)


class TestAppInit(unittest.TestCase):
    """Test App initialization and state defaults."""

    def test_init_with_defaults(self):
        app = App()
        self.assertEqual(app.target, get_default_target())
        self.assertIsNone(app.frames)
        self.assertEqual(app.width, 1280)
        self.assertEqual(app.height, 720)
        self.assertFalse(app.offscreen)
        self.assertFalse(app.is_running)
        self.assertEqual(app.frame_count, 0)
        self.assertEqual(app.elapsed_time, 0.0)
        self.assertEqual(app.dt, 0.0)
        self.assertEqual(app.fps, 0.0)

    def test_init_with_custom_namespace(self):
        ns = argparse.Namespace(target="vulkan", frames=3, width=640, height=480, offscreen=True)
        app = App(ns)
        self.assertEqual(app.target, "vulkan")
        self.assertEqual(app.frames, 3)
        self.assertEqual(app.width, 640)
        self.assertEqual(app.height, 480)
        self.assertTrue(app.offscreen)


class RecordingApp(App):
    """Test helper that records lifecycle events."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: list[str] = []
        self.dts: list[float] = []

    def startup(self) -> None:
        self.events.append("startup")

    def update(self, dt: float) -> None:
        self.events.append(f"update:{self.frame_count}")
        self.dts.append(dt)

    def render(self) -> None:
        self.events.append(f"render:{self.frame_count}")

    def cleanup(self) -> None:
        self.events.append("cleanup")


class TestAppLifecycle(unittest.TestCase):
    """Test lifecycle invocation order, frame limiting, and early termination."""

    def test_lifecycle_order_normal(self):
        app = RecordingApp(App.parse_args(["--frames", "2"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(
            app.events, ["startup", "update:0", "render:0", "update:1", "render:1", "cleanup"]
        )
        self.assertEqual(app.frame_count, 2)
        self.assertFalse(app.is_running)

    def test_finite_frame_count(self):
        app = RecordingApp(App.parse_args(["--frames", "5"]))
        app.run()
        self.assertEqual(app.frame_count, 5)
        update_count = sum(1 for e in app.events if e.startswith("update"))
        render_count = sum(1 for e in app.events if e.startswith("render"))
        self.assertEqual(update_count, 5)
        self.assertEqual(render_count, 5)

    def test_single_frame(self):
        app = RecordingApp(App.parse_args(["--frames", "1"]))
        app.run()
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "cleanup"])
        self.assertEqual(app.frame_count, 1)

    def test_zero_frames(self):
        app = RecordingApp(App.parse_args(["--frames", "0"]))
        app.run()
        self.assertEqual(app.events, ["startup", "cleanup"])
        self.assertEqual(app.frame_count, 0)

    def test_programmatic_stop(self):
        class StopApp(RecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 1:
                    self.stop()

        app = StopApp(App.parse_args(["--frames", "10"]))
        app.run()
        self.assertEqual(app.frame_count, 2)
        self.assertEqual(
            app.events, ["startup", "update:0", "render:0", "update:1", "render:1", "cleanup"]
        )


class TestAppTiming(unittest.TestCase):
    """Test high-resolution frame timing and FPS tracking."""

    def test_dt_positive_and_elapsed(self):
        app = RecordingApp(App.parse_args(["--frames", "3"]))
        app.run()
        self.assertEqual(len(app.dts), 3)
        for dt in app.dts:
            self.assertGreaterEqual(dt, 0.0)
        self.assertAlmostEqual(app.elapsed_time, sum(app.dts), places=6)

    def test_timing_with_mocked_perf_counter(self):
        mock_times = [10.0, 10.016, 10.032, 10.048, 10.064]
        with patch("time.perf_counter", side_effect=mock_times):
            app = RecordingApp(App.parse_args(["--frames", "3"]))
            app.run()
            self.assertAlmostEqual(app.dts[0], 0.016, places=5)
            self.assertAlmostEqual(app.dts[1], 0.016, places=5)
            self.assertAlmostEqual(app.dts[2], 0.016, places=5)
            self.assertGreater(app.fps, 0.0)

    def test_zero_dt_no_division_error(self):
        with patch("time.perf_counter", side_effect=[5.0, 5.0, 5.0, 5.0]):
            app = RecordingApp(App.parse_args(["--frames", "2"]))
            app.run()
            self.assertEqual(app.dt, 0.0)


class TestAppExceptions(unittest.TestCase):
    """Test exception handling and guaranteed cleanup invocations."""

    def test_startup_exception_calls_cleanup(self):
        class FailStartupApp(RecordingApp):
            def startup(self) -> None:
                super().startup()
                raise RuntimeError("startup failure")

        app = FailStartupApp(App.parse_args(["--frames", "2"]))
        with self.assertRaises(RuntimeError) as ctx:
            app.run()
        self.assertIn("startup failure", str(ctx.exception))
        self.assertEqual(app.events, ["startup", "cleanup"])
        self.assertFalse(app.is_running)

    def test_update_exception_calls_cleanup(self):
        class FailUpdateApp(RecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 1:
                    raise ValueError("update failure")

        app = FailUpdateApp(App.parse_args(["--frames", "5"]))
        with self.assertRaises(ValueError) as ctx:
            app.run()
        self.assertIn("update failure", str(ctx.exception))
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "update:1", "cleanup"])
        self.assertFalse(app.is_running)

    def test_render_exception_calls_cleanup(self):
        class FailRenderApp(RecordingApp):
            def render(self) -> None:
                super().render()
                if self.frame_count == 0:
                    raise RuntimeError("render failure")

        app = FailRenderApp(App.parse_args(["--frames", "5"]))
        with self.assertRaises(RuntimeError) as ctx:
            app.run()
        self.assertIn("render failure", str(ctx.exception))
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "cleanup"])
        self.assertFalse(app.is_running)

    def test_keyboard_interrupt_calls_cleanup(self):
        class InterruptApp(RecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                raise KeyboardInterrupt()

        app = InterruptApp(App.parse_args(["--frames", "5"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.events, ["startup", "update:0", "cleanup"])
        self.assertFalse(app.is_running)


class TestAppMain(unittest.TestCase):
    """Test main CLI entrypoint."""

    def test_main_runs_finite_frames(self):
        exit_code = main(["--frames", "2", "--offscreen"])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
