"""Comprehensive empirical stress harness for MiniGfx App lifecycle and timing."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
import unittest
from typing import List, Optional
from unittest.mock import patch

from examples.minigfx.app import App, get_default_target


class CustomFatalError(Exception):
    """Custom exception type for failure testing."""
    pass


class EventRecordingApp(App):
    """App subclass that records lifecycle events and timestamps."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.events: List[str] = []
        self.recorded_dts: List[float] = []

    def startup(self) -> None:
        self.events.append("startup")

    def update(self, dt: float) -> None:
        self.events.append(f"update:{self.frame_count}")
        self.recorded_dts.append(dt)

    def render(self) -> None:
        self.events.append(f"render:{self.frame_count}")

    def cleanup(self) -> None:
        self.events.append("cleanup")


class TestStressApp(unittest.TestCase):
    """Stress tests covering edge cases, clock quirks, and failure modes."""

    # -------------------------------------------------------------------------
    # 1. Early Stop Scenarios
    # -------------------------------------------------------------------------

    def test_stop_inside_startup(self):
        """Test behavior when app.stop() is called inside startup()."""
        class StartupStopApp(EventRecordingApp):
            def startup(self) -> None:
                super().startup()
                self.stop()

        app = StartupStopApp(App.parse_args(["--frames", "5"]))
        exit_code = app.run()
        print(f"\n[test_stop_inside_startup] events: {app.events}, frame_count: {app.frame_count}")
        # Does the app actually stop or does self.is_running = True override self.stop()?
        # We check empirical behavior:
        # If is_running was overwritten, frame_count would be > 0.

    def test_stop_on_frame_0_in_update(self):
        """app.stop() on frame 0 in update()."""
        class Frame0UpdateStopApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 0:
                    self.stop()

        app = Frame0UpdateStopApp(App.parse_args(["--frames", "100"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 1)
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "cleanup"])
        self.assertFalse(app.is_running)

    def test_stop_on_frame_0_in_render(self):
        """app.stop() on frame 0 in render()."""
        class Frame0RenderStopApp(EventRecordingApp):
            def render(self) -> None:
                super().render()
                if self.frame_count == 0:
                    self.stop()

        app = Frame0RenderStopApp(App.parse_args(["--frames", "100"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 1)
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "cleanup"])
        self.assertFalse(app.is_running)

    def test_stop_on_frame_1_in_update(self):
        """app.stop() on frame 1 in update()."""
        class Frame1UpdateStopApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 1:
                    self.stop()

        app = Frame1UpdateStopApp(App.parse_args(["--frames", "100"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 2)
        self.assertEqual(app.events, [
            "startup",
            "update:0", "render:0",
            "update:1", "render:1",
            "cleanup",
        ])
        self.assertFalse(app.is_running)

    def test_stop_before_run(self):
        """app.stop() called before app.run()."""
        app = EventRecordingApp(App.parse_args(["--frames", "5"]))
        app.stop()
        exit_code = app.run()
        print(f"\n[test_stop_before_run] events: {app.events}, frame_count: {app.frame_count}")

    # -------------------------------------------------------------------------
    # 2. Zero and Boundary Frames
    # -------------------------------------------------------------------------

    def test_zero_frames_cli(self):
        """--frames 0 terminates after startup and cleanup with 0 frames."""
        app = EventRecordingApp(App.parse_args(["--frames", "0"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 0)
        self.assertEqual(app.events, ["startup", "cleanup"])
        self.assertFalse(app.is_running)

    def test_zero_frames_manual_attribute(self):
        """app.frames = 0 set directly."""
        app = EventRecordingApp()
        app.frames = 0
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 0)
        self.assertEqual(app.events, ["startup", "cleanup"])

    def test_negative_frames_manual_attribute(self):
        """app.frames = -1 set directly."""
        app = EventRecordingApp()
        app.frames = -1
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 0)
        self.assertEqual(app.events, ["startup", "cleanup"])

    # -------------------------------------------------------------------------
    # 3. High Frame Counts (Headless & Offscreen)
    # -------------------------------------------------------------------------

    def test_high_frame_count_1000(self):
        """1000 frames headless executed cleanly."""
        app = EventRecordingApp(App.parse_args(["--frames", "1000", "--offscreen"]))
        t0 = time.perf_counter()
        exit_code = app.run()
        t1 = time.perf_counter()
        elapsed_wall = t1 - t0

        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 1000)
        self.assertEqual(len(app.recorded_dts), 1000)
        self.assertFalse(app.is_running)
        self.assertGreater(app.fps, 0.0)
        self.assertAlmostEqual(app.elapsed_time, sum(app.recorded_dts), places=4)
        print(f"\n[test_high_frame_count_1000] Wall time: {elapsed_wall:.4f}s, Sim elapsed: {app.elapsed_time:.4f}s, Final FPS: {app.fps:.1f}")

    def test_high_frame_count_5000(self):
        """5000 frames stress test."""
        app = App(App.parse_args(["--frames", "5000", "--offscreen"]))
        t0 = time.perf_counter()
        exit_code = app.run()
        t1 = time.perf_counter()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 5000)
        print(f"\n[test_high_frame_count_5000] 5000 frames completed in {t1 - t0:.4f}s, FPS: {app.fps:.1f}")

    # -------------------------------------------------------------------------
    # 4. Clock Quirks
    # -------------------------------------------------------------------------

    def test_clock_freeze_zero_dt(self):
        """Simulated clock freeze: time.perf_counter() returns constant."""
        with patch("time.perf_counter", return_value=123.456):
            app = EventRecordingApp(App.parse_args(["--frames", "20"]))
            exit_code = app.run()
            self.assertEqual(exit_code, 0)
            self.assertEqual(app.frame_count, 20)
            self.assertEqual(app.dt, 0.0)
            self.assertEqual(app.elapsed_time, 0.0)
            self.assertEqual(app.fps, 0.0)
            for dt in app.recorded_dts:
                self.assertEqual(dt, 0.0)

    def test_clock_backwards_jump(self):
        """Simulated clock going backwards (e.g. system clock reset / NTP glitch)."""
        # Times: startup=100.0, frame0=100.016, frame1=90.0 (jump backwards), frame2=90.016
        times = [100.0, 100.016, 90.0, 90.016, 90.032]
        with patch("time.perf_counter", side_effect=times):
            app = EventRecordingApp(App.parse_args(["--frames", "4"]))
            exit_code = app.run()
            self.assertEqual(exit_code, 0)
            self.assertEqual(app.frame_count, 4)
            # Frame 0: dt = 100.016 - 100.0 = 0.016
            self.assertAlmostEqual(app.recorded_dts[0], 0.016, places=5)
            # Frame 1: now = 90.0, last_time = 100.016 -> max(90.0 - 100.016, 0.0) = 0.0
            self.assertEqual(app.recorded_dts[1], 0.0)
            # Frame 2: now = 90.016, last_time = 90.0 -> dt = 0.016
            self.assertAlmostEqual(app.recorded_dts[2], 0.016, places=5)
            # Frame 3: now = 90.032, last_time = 90.016 -> dt = 0.016
            self.assertAlmostEqual(app.recorded_dts[3], 0.016, places=5)

    def test_clock_large_delta_time(self):
        """Simulated debugger pause or huge dt (e.g. 1 hour). Verify EMA updates and no crash."""
        # Frame 0: dt = 0.016 -> instant = 62.5 -> fps = 62.5
        # Frame 1: dt = 3600.0 -> instant = 1/3600 -> fps = 0.9 * 62.5 + 0.1 * (1/3600) = 56.25
        times = [10.0, 10.016, 3610.016]
        with patch("time.perf_counter", side_effect=times):
            app = EventRecordingApp(App.parse_args(["--frames", "2"]))
            exit_code = app.run()
            self.assertEqual(exit_code, 0)
            self.assertEqual(app.frame_count, 2)
            self.assertAlmostEqual(app.recorded_dts[1], 3600.0, places=3)
            # instant fps for frame 1 is 1/3600 (~0.00028)
            # EMA with alpha=0.1: 0.9 * 62.5 + 0.1 * (1/3600) = ~56.25
            expected_fps = 0.9 * 62.5 + 0.1 * (1.0 / 3600.0)
            self.assertAlmostEqual(app.fps, expected_fps, places=2)

    def test_continuous_loop_stop_at_frame_50(self):
        """Continuous loop (frames=None) stopped programmatically at frame 50."""
        class StopAt50App(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 50:
                    self.stop()

        app = StopAt50App(App.parse_args([]))  # frames=None by default
        self.assertIsNone(app.frames)
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 51)  # 0 to 50 inclusive
        self.assertEqual(len(app.recorded_dts), 51)
        self.assertFalse(app.is_running)

    def test_dynamic_frames_modification(self):
        """Subclass dynamically modifies self.frames during the loop."""
        class DynamicFramesApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 2:
                    self.frames = 5  # Extend or shorten dynamically

        app = DynamicFramesApp(App.parse_args(["--frames", "20"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 5)

    def test_keyboard_interrupt_during_startup(self):
        """KeyboardInterrupt during startup() is caught cleanly and returns 0."""
        class InterruptStartupApp(EventRecordingApp):
            def startup(self) -> None:
                super().startup()
                raise KeyboardInterrupt()

        app = InterruptStartupApp(App.parse_args(["--frames", "5"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.events, ["startup", "cleanup"])
        self.assertFalse(app.is_running)

    def test_sigint_signal_in_update(self):
        """Simulate real OS SIGINT signal during frame update."""
        class SigintApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 1:
                    os.kill(os.getpid(), signal.SIGINT)

        app = SigintApp(App.parse_args(["--frames", "10"]))
        exit_code = app.run()
        self.assertEqual(exit_code, 0)
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "update:1", "cleanup"])
        self.assertFalse(app.is_running)

    def test_clock_nan_handling(self):
        """What happens if perf_counter returns NaN?"""
        times = [10.0, float("nan"), 10.032]
        with patch("time.perf_counter", side_effect=times):
            app = EventRecordingApp(App.parse_args(["--frames", "2"]))
            # max(nan, 0.0) in Python is nan!
            app.run()
            print(f"\n[test_clock_nan_handling] dt: {app.dt}, elapsed: {app.elapsed_time}, fps: {app.fps}")

    # -------------------------------------------------------------------------
    # 5. Sequential Invocations
    # -------------------------------------------------------------------------

    def test_multiple_distinct_app_instances_sequentially(self):
        """Multiple distinct App instances executed one after another."""
        for i in range(5):
            app = EventRecordingApp(App.parse_args(["--frames", "3"]))
            code = app.run()
            self.assertEqual(code, 0)
            self.assertEqual(app.frame_count, 3)
            self.assertEqual(app.events, ["startup", "update:0", "render:0", "update:1", "render:1", "update:2", "render:2", "cleanup"])
            self.assertFalse(app.is_running)

    def test_same_instance_run_called_twice(self):
        """What happens when app.run() is called twice on the exact same instance?"""
        app = EventRecordingApp(App.parse_args(["--frames", "3"]))
        code1 = app.run()
        self.assertEqual(code1, 0)
        self.assertEqual(app.frame_count, 3)

        # Call run() a second time on same instance
        code2 = app.run()
        print(f"\n[test_same_instance_run_called_twice] Run 2 code: {code2}, frame_count: {app.frame_count}, events: {app.events}")

    # -------------------------------------------------------------------------
    # 6. Exception Handling and Cleanup Guarantees
    # -------------------------------------------------------------------------

    def test_custom_exception_in_startup(self):
        """Custom exception in startup() guarantees cleanup() and re-raises."""
        class FailStartupApp(EventRecordingApp):
            def startup(self) -> None:
                super().startup()
                raise CustomFatalError("GPU initialization failed")

        app = FailStartupApp()
        with self.assertRaises(CustomFatalError):
            app.run()
        self.assertEqual(app.events, ["startup", "cleanup"])
        self.assertFalse(app.is_running)

    def test_custom_exception_in_update(self):
        """Custom exception in update() guarantees cleanup() and re-raises."""
        class FailUpdateApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                if self.frame_count == 2:
                    raise CustomFatalError("Simulation NaN detected")

        app = FailUpdateApp(App.parse_args(["--frames", "10"]))
        with self.assertRaises(CustomFatalError):
            app.run()
        self.assertEqual(app.frame_count, 2)
        self.assertEqual(app.events, [
            "startup",
            "update:0", "render:0",
            "update:1", "render:1",
            "update:2",
            "cleanup",
        ])
        self.assertFalse(app.is_running)

    def test_custom_exception_in_render(self):
        """Custom exception in render() guarantees cleanup() and re-raises."""
        class FailRenderApp(EventRecordingApp):
            def render(self) -> None:
                super().render()
                if self.frame_count == 1:
                    raise CustomFatalError("VK_ERROR_DEVICE_LOST")

        app = FailRenderApp(App.parse_args(["--frames", "10"]))
        with self.assertRaises(CustomFatalError):
            app.run()
        self.assertEqual(app.frame_count, 1)
        self.assertEqual(app.events, [
            "startup",
            "update:0", "render:0",
            "update:1", "render:1",
            "cleanup",
        ])
        self.assertFalse(app.is_running)

    def test_system_exit_in_update(self):
        """sys.exit(42) in update() still triggers cleanup()."""
        class SystemExitApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                sys.exit(42)

        app = SystemExitApp(App.parse_args(["--frames", "5"]))
        with self.assertRaises(SystemExit) as ctx:
            app.run()
        self.assertEqual(ctx.exception.code, 42)
        self.assertEqual(app.events, ["startup", "update:0", "cleanup"])
        self.assertFalse(app.is_running)

    def test_generator_exit_in_update(self):
        """GeneratorExit in update() triggers cleanup()."""
        class GeneratorExitApp(EventRecordingApp):
            def update(self, dt: float) -> None:
                super().update(dt)
                raise GeneratorExit()

        app = GeneratorExitApp(App.parse_args(["--frames", "5"]))
        with self.assertRaises(GeneratorExit):
            app.run()
        self.assertEqual(app.events, ["startup", "update:0", "cleanup"])
        self.assertFalse(app.is_running)

    def test_cleanup_exception_does_not_mask_is_running_flag(self):
        """Even if cleanup() raises, is_running is already set to False."""
        class ExplodingCleanupApp(EventRecordingApp):
            def cleanup(self) -> None:
                super().cleanup()
                raise RuntimeError("Cleanup crashed")

        app = ExplodingCleanupApp(App.parse_args(["--frames", "1"]))
        with self.assertRaises(RuntimeError):
            app.run()
        self.assertFalse(app.is_running)
        self.assertEqual(app.events, ["startup", "update:0", "render:0", "cleanup"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
