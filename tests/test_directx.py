import ctypes
import os
import queue
import runpy
import subprocess
import tempfile
import threading
import time
import unittest
from ctypes import wintypes
from pathlib import Path

from mlir_gpu import Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import _directx_shaders, load_program, render_directx_offscreen, show_directx_window
from test_target_parity import positioned_triangle


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = bytes([5, 5, 10, 255])


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 for real Windows D3D12 rendering",
)
class DirectXRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.tools = Toolchain(
            Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None,
            Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None,
        )
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.programs = {}
        pipelines = {
            name: runpy.run_path(str(ROOT / "examples" / f"{name}.py"))["pipeline"]
            for name in ("triangle", "quad")
        }
        for name, depth in (("before-near", -0.1), ("after-far", 1.1)):
            pipelines[name] = positioned_triangle([
                (-0.6, -0.6, depth, 1), (0.6, -0.6, depth, 1), (0, 0.6, depth, 1),
            ])
        for name, pipeline in pipelines.items():
            manifest = compile_pipeline(pipeline, Target.DIRECTX, Path(work.name) / name, cls.tools)
            cls.programs[name] = load_program(manifest)

    def test_triangle_and_quad_pixels_at_square_nonsquare_and_unaligned_widths(self):
        for name in ("triangle", "quad"):
            for width, height in ((256, 256), (320, 180), (257, 193)):
                with self.subTest(scene=name, size=(width, height)):
                    image = render_directx_offscreen(self.programs[name], width, height, self.validation, self.tools)
                    self.assertTrue(image.device)
                    self.assertEqual(len(image.rgba), width * height * 4)

                    def pixel(x, y):
                        offset = (y * width + x) * 4
                        return image.rgba[offset:offset + 4]

                    self.assertEqual(pixel(0, 0), BACKGROUND)
                    self.assertEqual(pixel(width - 1, height - 1), BACKGROUND)
                    covered = sum(
                        image.rgba[offset:offset + 4] != BACKGROUND
                        for offset in range(0, len(image.rgba), 4)
                    ) / (width * height)
                    self.assertAlmostEqual(covered, 0.18 if name == "triangle" else 0.64, delta=0.005)
                    samples = (
                        ((width // 2, height // 3), (width // 3, 2 * height // 3), (2 * width // 3, 2 * height // 3))
                        if name == "triangle" else
                        ((width // 3, height // 3), (2 * width // 3, 2 * height // 3))
                    )
                    for x, y in samples:
                        ndc_x = (x + 0.5) / width * 2 - 1
                        ndc_y = 1 - (y + 0.5) / height * 2
                        if name == "triangle":
                            blue = (ndc_y + 0.6) / 1.2
                            green = ((1 - blue) + ndc_x / 0.6) / 2
                            expected = (1 - green - blue, green, blue, 1)
                        else:
                            u, v = (ndc_x + 0.8) / 1.6, (ndc_y + 0.8) / 1.6
                            expected = (u, v, u * v, 1)
                        for channel, value in zip(pixel(x, y), expected):
                            self.assertAlmostEqual(channel, round(value * 255), delta=2)

    def test_zero_to_one_depth_range_clips_near_and_far(self):
        for name in ("before-near", "after-far"):
            with self.subTest(scene=name):
                image = render_directx_offscreen(self.programs[name], 257, 193, self.validation, self.tools)
                self.assertEqual(image.rgba, BACKGROUND * (257 * 193))

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_bounded_native_window(self):
        show_directx_window(self.programs["triangle"], 160, 120, 3, self.validation, self.tools)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_unbounded_window_survives_resize_and_minimize(self):
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
        user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
        user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
        user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
        with _directx_shaders(self.programs["quad"], self.tools) as (_, arguments):
            for frame_arguments in ([], ["-frames", "0"], ["--frames", "0"]):
                with self.subTest(frames=frame_arguments):
                    process = subprocess.Popen(
                        [*arguments, "--width", "160", "--height", "120", *frame_arguments,
                         *(["--validation"] if self.validation else [])],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    )
                    readiness = queue.Queue()
                    reader = threading.Thread(target=lambda: readiness.put(process.stdout.readline()), daemon=True)
                    reader.start()
                    try:
                        self.assertIn("DirectX 12 viewer ready", readiness.get(timeout=30))
                        reader.join(timeout=1)
                        windows = []

                        @callback_type
                        def collect(window, parameter):
                            pid = wintypes.DWORD()
                            user32.GetWindowThreadProcessId(window, ctypes.byref(pid))
                            if pid.value == process.pid:
                                windows.append(window)
                            return True

                        self.assertTrue(user32.EnumWindows(collect, 0))
                        self.assertEqual(len(windows), 1)
                        window = windows[0]
                        self.assertTrue(user32.SetWindowPos(window, None, 0, 0, 300, 220, 0x0002 | 0x0004))
                        time.sleep(0.3)
                        user32.ShowWindow(window, 6)
                        time.sleep(0.3)
                        self.assertIsNone(process.poll(), "Unbounded window exited while minimized")
                        user32.ShowWindow(window, 9)
                        time.sleep(0.3)
                        self.assertIsNone(process.poll(), "Unbounded window exited after resize/restore")
                        self.assertTrue(user32.PostMessageW(window, 0x0010, 0, 0))
                        stdout, stderr = process.communicate(timeout=30)
                        self.assertEqual(process.returncode, 0, stderr)
                        self.assertRegex(stdout, r"Presented [1-9][0-9]* DirectX 12 frames")
                    finally:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate(timeout=10)
                        reader.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
