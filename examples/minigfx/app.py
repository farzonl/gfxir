"""Application lifecycle host and main loop for MiniGfx."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Sequence


def get_default_target() -> str:
    """Return platform-appropriate default graphics backend."""
    if sys.platform == "darwin":
        return "metal"
    if sys.platform == "win32":
        return "directx"
    return "vulkan"


def _non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"Value must be non-negative, got {value}")
    return ivalue


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Value must be positive, got {value}")
    return ivalue


class App:
    """Base application lifecycle host managing timing, CLI configuration, and main loop."""

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        if args is None:
            args = self.parse_args([])
        self.args: argparse.Namespace = args
        self.target: str = getattr(args, "target", get_default_target())
        self.frames: Optional[int] = getattr(args, "frames", None)
        self.width: int = getattr(args, "width", 1280)
        self.height: int = getattr(args, "height", 720)
        self.offscreen: bool = getattr(args, "offscreen", False)

        self.is_running: bool = False
        self.frame_count: int = 0
        self.elapsed_time: float = 0.0
        self.dt: float = 0.0
        self.fps: float = 0.0
        self._last_time: float = 0.0

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Construct the CLI argument parser for MiniGfx applications."""
        parser = argparse.ArgumentParser(description="MiniGfx Application")
        parser.add_argument(
            "--target",
            choices=["vulkan", "directx", "metal"],
            default=get_default_target(),
            help="Target graphics backend (default: %(default)s)",
        )
        parser.add_argument(
            "--frames",
            type=_non_negative_int,
            default=None,
            help="Number of frames to render before exiting (default: continuous loop)",
        )
        parser.add_argument(
            "--width",
            type=_positive_int,
            default=1280,
            help="Window / framebuffer width in pixels (default: %(default)s)",
        )
        parser.add_argument(
            "--height",
            type=_positive_int,
            default=720,
            help="Window / framebuffer height in pixels (default: %(default)s)",
        )
        parser.add_argument(
            "--offscreen",
            action="store_true",
            default=False,
            help="Render offscreen without opening a live window (default: %(default)s)",
        )
        return parser

    @classmethod
    def parse_args(cls, args: Optional[Sequence[str]] = None) -> argparse.Namespace:
        """Parse CLI arguments into a namespace."""
        return cls.build_parser().parse_args(args)

    def startup(self) -> None:
        """Lifecycle hook invoked once before entering the main frame loop."""

    def update(self, dt: float) -> None:
        """Lifecycle hook invoked once per frame to advance simulation and input state."""

    def render(self) -> None:
        """Lifecycle hook invoked once per frame to record and submit draw commands."""

    def cleanup(self) -> None:
        """Lifecycle hook invoked on exit or exception to release engine resources."""

    def stop(self) -> None:
        """Signal the main loop to terminate after the current frame."""
        self.is_running = False

    def run(self) -> int:
        """Execute application lifecycle: startup -> loop(update, render) -> cleanup."""
        try:
            self.startup()
            self.is_running = True
            self._last_time = time.perf_counter()
            while self.is_running:
                if self.frames is not None and self.frame_count >= self.frames:
                    break
                now = time.perf_counter()
                dt = max(now - self._last_time, 0.0)
                self._last_time = now
                self.dt = dt
                self.elapsed_time += dt
                if dt > 0.0:
                    instant = 1.0 / dt
                    self.fps = (0.9 * self.fps + 0.1 * instant) if self.fps > 0.0 else instant
                self.update(dt)
                self.render()
                self.frame_count += 1
            return 0
        except KeyboardInterrupt:
            return 0
        finally:
            self.is_running = False
            self.cleanup()


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for running a MiniGfx application from CLI."""
    parsed = App.parse_args(args)
    app = App(parsed)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
