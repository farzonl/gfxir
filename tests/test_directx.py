import copy
import ctypes
import json
import os
import queue
import runpy
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from ctypes import wintypes
from pathlib import Path

from mlir_gpu import (
    FrameParameters, Pipeline, Target, Toolchain, Vertex, compile_pipeline,
    fragment, fullscreen_vertex, select, vec4, vertex,
)
from mlir_gpu.viewer import (
    ViewerError, _directx_shaders, load_program, render_directx_offscreen, show_directx_window, write_png,
)
from test_parameters import FRAME_REFLECTION
from test_target_parity import positioned_triangle


ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = bytes([5, 5, 10, 255])


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows for real Windows D3D12 rendering",
)
class DirectXRenderingTests(unittest.TestCase):
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
        @fragment
        def frame_diagnostic(frame: FrameParameters):
            return vec4(frame.resolution / 255.0, frame.time / 4.0, 1)

        pipelines["diagnostic"] = Pipeline(fullscreen_vertex, frame_diagnostic)
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

    def test_exact_framebuffer_dimensions_and_changed_time_reach_directx_shaders(self):
        for width, height, time_val in ((37, 23, 0.0), (151, 19, 1.0), (43, 71, 1.5)):
            with self.subTest(dimensions=(width, height), time=time_val):
                image = render_directx_offscreen(self.programs["diagnostic"], width, height, self.validation, self.tools, time=time_val)
                self.assertEqual(image.rgba, bytes([width, height, round(time_val / 4 * 255), 255]) * (width * height))

    def test_deterministic_time_sampling_and_zero_time_propagation(self):
        """Verify deterministic time parameter propagation including t=0.0 and bitwise repeatability."""
        program = self.programs["diagnostic"]
        width, height = 64, 48

        # 1. Verify t=0.0 is explicitly passed and produces blue channel 0
        img_t0_a = render_directx_offscreen(program, width, height, self.validation, self.tools, time=0.0)
        expected_blue_0 = round(0.0 / 4 * 255)  # 0
        expected_pixel_0 = bytes([width, height, expected_blue_0, 255])
        self.assertEqual(img_t0_a.rgba[:4], expected_pixel_0)
        self.assertEqual(img_t0_a.rgba, expected_pixel_0 * (width * height))

        # 2. Verify bitwise repeatability at t=0.0
        img_t0_b = render_directx_offscreen(program, width, height, self.validation, self.tools, time=0.0)
        self.assertEqual(img_t0_a.rgba, img_t0_b.rgba, "Independent runs at t=0.0 must be bitwise identical")

        # 3. Verify t=1.0 produces distinct blue channel 64
        img_t1_a = render_directx_offscreen(program, width, height, self.validation, self.tools, time=1.0)
        expected_blue_1 = round(1.0 / 4 * 255)  # 64
        expected_pixel_1 = bytes([width, height, expected_blue_1, 255])
        self.assertEqual(img_t1_a.rgba[:4], expected_pixel_1)
        self.assertEqual(img_t1_a.rgba, expected_pixel_1 * (width * height))

        # 4. Assert divergence between t=0.0 and t=1.0
        self.assertNotEqual(img_t0_a.rgba, img_t1_a.rgba, "Frames at t=0.0 and t=1.0 must differ")

        # 5. Verify bitwise repeatability at t=1.0
        img_t1_b = render_directx_offscreen(program, width, height, self.validation, self.tools, time=1.0)
        self.assertEqual(img_t1_a.rgba, img_t1_b.rgba, "Independent runs at t=1.0 must be bitwise identical")

        # 6. Verify negative or invalid time rejects before native invocation
        for bad_time in (-0.001, -1.0, float("nan"), float("inf"), 1e39):
            with self.subTest(bad_time=bad_time), self.assertRaises(ViewerError):
                render_directx_offscreen(program, width, height, self.validation, self.tools, time=bad_time)

    def test_resource_schema_validation_failures(self):
        """Assert rejection of malformed, invalid, or unsupported parameter schemas before native launch."""
        base_program = self.programs["diagnostic"]

        def with_params(params):
            return type(base_program)(
                base_program.vertex, base_program.fragment, base_program.target, base_program.draw,
                parameters=params,
            )

        # 1. Empty / missing schema fields
        with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
            render_directx_offscreen(with_params({}), 16, 12, self.validation, self.tools)

        # 2. Unsupported schema version
        bad_schema = copy.deepcopy(FRAME_REFLECTION)
        bad_schema["schema_version"] = 2
        with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
            render_directx_offscreen(with_params(bad_schema), 16, 12, self.validation, self.tools)

        # 3. Unsupported layout name
        bad_layout = copy.deepcopy(FRAME_REFLECTION)
        bad_layout["layout"] = "std140"
        with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
            render_directx_offscreen(with_params(bad_layout), 16, 12, self.validation, self.tools)

        # 4. Invalid stage visibility (empty)
        bad_stages = copy.deepcopy(FRAME_REFLECTION)
        bad_stages["stages"] = []
        with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
            render_directx_offscreen(with_params(bad_stages), 16, 12, self.validation, self.tools)

        # 5. Unsupported stage (compute)
        bad_stage = copy.deepcopy(FRAME_REFLECTION)
        bad_stage["stages"] = ["compute"]
        with self.assertRaisesRegex(ViewerError, "supplies FrameParameters only"):
            render_directx_offscreen(with_params(bad_stage), 16, 12, self.validation, self.tools)

        # 6. Duplicate stages
        bad_dup = copy.deepcopy(FRAME_REFLECTION)
        bad_dup["stages"] = ["vertex", "vertex"]
        with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
            render_directx_offscreen(with_params(bad_dup), 16, 12, self.validation, self.tools)

        # 7. Mismatched binding storage
        bad_binding = copy.deepcopy(FRAME_REFLECTION)
        bad_binding["bindings"]["directx"]["storage"] = "descriptor_table"
        with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
            render_directx_offscreen(with_params(bad_binding), 16, 12, self.validation, self.tools)

    def test_preserved_unparameterized_rendering_invariants(self):
        """Verify that unparameterized triangle and quad render without regression."""
        for scene in ("triangle", "quad"):
            prog = self.programs[scene]
            self.assertIsNone(prog.parameters, f"Scene {scene} must not have parameter reflection")

            for width, height in ((256, 256), (320, 180), (257, 193)):
                with self.subTest(scene=scene, dimensions=(width, height)):
                    image = render_directx_offscreen(prog, width, height, self.validation, self.tools)
                    self.assertEqual(len(image.rgba), width * height * 4)

                    def get_pixel(x, y):
                        idx = (y * width + x) * 4
                        return image.rgba[idx:idx + 4]

                    # Corner pixels must be clear background
                    self.assertEqual(get_pixel(0, 0), BACKGROUND)
                    self.assertEqual(get_pixel(width - 1, height - 1), BACKGROUND)

                    # Coverage invariant
                    covered_count = sum(
                        image.rgba[i:i + 4] != BACKGROUND
                        for i in range(0, len(image.rgba), 4)
                    )
                    coverage = covered_count / (width * height)
                    expected_cov = 0.18 if scene == "triangle" else 0.64
                    self.assertAlmostEqual(coverage, expected_cov, delta=0.005)

    def test_stage_visibility_in_root_signature(self):
        """Verify vertex, fragment, and both-stage FrameParameters pipelines via embedded DXIL root signatures."""
        @vertex
        def vertex_with_frame(index, frame: FrameParameters):
            return Vertex(vec4(select(index == 1, 3.0, -1.0) + frame.time * 0.0,
                               select(index == 2, 3.0, -1.0), 0.0, 1.0))

        @fragment
        def solid_fragment():
            return vec4(0.25, 0.5, 0.75, 1.0)

        @fragment
        def fragment_with_frame(frame: FrameParameters):
            return vec4(frame.resolution / 255.0, frame.time / 4.0, 1.0)

        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            stage_pipelines = {
                "fragment": self.programs["diagnostic"],
                "vertex": load_program(
                    compile_pipeline(Pipeline(vertex_with_frame, solid_fragment), Target.DIRECTX, dir_path / "vertex", self.tools)
                ),
                "all": load_program(
                    compile_pipeline(Pipeline(vertex_with_frame, fragment_with_frame), Target.DIRECTX, dir_path / "all", self.tools)
                ),
            }
            self.assertIn(b"RTS0", stage_pipelines["fragment"].fragment.code)
            self.assertNotIn(b"RTS0", stage_pipelines["fragment"].vertex.code)
            self.assertIn(b"RTS0", stage_pipelines["vertex"].vertex.code)
            self.assertNotIn(b"RTS0", stage_pipelines["vertex"].fragment.code)
            self.assertIn(b"RTS0", stage_pipelines["all"].vertex.code)
            self.assertIn(b"RTS0", stage_pipelines["all"].fragment.code)

            for label, prog in stage_pipelines.items():
                with self.subTest(visibility=label):
                    img = render_directx_offscreen(prog, 16, 12, self.validation, self.tools, time=1.0)
                    self.assertEqual(len(img.rgba), 16 * 12 * 4)

    def test_embedded_and_fallback_root_signature_creation(self):
        """Verify loading embedded root signatures from DXIL and fallback to serialized descriptor."""
        # 1. Pipeline with embedded root signature in fragment DXIL (RTS0 FourCC chunk present)
        self.assertIn(b"RTS0", self.programs["diagnostic"].fragment.code)
        self.assertNotIn(b"RTS0", self.programs["diagnostic"].vertex.code)
        img_embedded = render_directx_offscreen(
            self.programs["diagnostic"], 32, 24, self.validation, self.tools, time=1.0,
        )
        self.assertEqual(len(img_embedded.rgba), 32 * 24 * 4)

        # 2. Pipeline without embedded root signature (RTS0 absent) falling back to procedural serialization
        self.assertNotIn(b"RTS0", self.programs["triangle"].fragment.code)
        self.assertNotIn(b"RTS0", self.programs["triangle"].vertex.code)
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            vertex_dxil = dir_path / "vertex.dxil"
            fragment_dxil = dir_path / "fragment.dxil"
            out_file = dir_path / "fallback.rgba"

            vertex_dxil.write_bytes(self.programs["triangle"].vertex.code)
            fragment_dxil.write_bytes(self.programs["triangle"].fragment.code)

            viewer_bin = self.tools.find("mlir-gpu-directx-view")
            cmd = [
                viewer_bin,
                *(["--warp"] if os.environ.get("MLIR_GPU_TEST_WARP") == "1" else []),
                "--vertex", str(vertex_dxil),
                "--fragment", str(fragment_dxil),
                "--vertex-count", "3",
                "--readback", str(out_file),
                "--width", "32", "--height", "24",
                "--time", "1.0",
            ]
            if self.validation:
                cmd.append("--validation")
            res = subprocess.run(cmd, capture_output=True, text=True, check=False)
            self.assertEqual(res.returncode, 0, f"Fallback descriptor serialization failed: {res.stderr}")
            self.assertTrue(out_file.is_file())
            self.assertEqual(len(out_file.read_bytes()), 32 * 24 * 4)

    def test_deterministic_png_reproducibility(self):
        """Verify PNG export byte-for-byte reproducibility on fixed-time DirectX captures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dir_path = Path(temp_dir)
            png1 = dir_path / "capture1.png"
            png2 = dir_path / "capture2.png"

            img1 = render_directx_offscreen(self.programs["diagnostic"], 64, 48, self.validation, self.tools, time=2.0)
            img2 = render_directx_offscreen(self.programs["diagnostic"], 64, 48, self.validation, self.tools, time=2.0)

            write_png(png1, img1)
            write_png(png2, img2)

            self.assertEqual(png1.read_bytes(), png2.read_bytes(), "PNG files at identical time must be bitwise identical")
            # Validate PNG header
            header = png1.read_bytes()[:8]
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")

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
