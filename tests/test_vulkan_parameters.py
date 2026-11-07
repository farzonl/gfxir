import contextlib
import io
import json
import os
import runpy
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlir_gpu import FrameParameters, Pipeline, Target, Vertex, compile_pipeline, fragment, fullscreen_vertex, vec4, vertex
from mlir_gpu.viewer import ViewerError, _Vulkan, _binding, _frame_data, _time_value, load_program, main, render_offscreen, show_window
from test_support import ROOT, TOOLS


@fragment
def frame_diagnostic(frame: FrameParameters):
    return vec4(frame.resolution / 255.0, frame.time / 4.0, 1)


@vertex
def frame_vertex(index, frame: FrameParameters):
    return Vertex(fullscreen_vertex.function(index).position,
                  color=vec4(frame.resolution / 255.0, frame.time / 8.0, 1))


@fragment
def both_stages(color, frame: FrameParameters):
    return color.with_components(z=frame.time / 4.0)


class FrameTimeTests(unittest.TestCase):
    def test_time_rejects_nonfinite_negative_out_of_range_and_non_numeric_values(self):
        for value in (True, -0.01, float("nan"), float("inf"), 10**400, "1", None):
            with self.subTest(value=value), self.assertRaisesRegex(ViewerError, "finite"):
                _time_value(value)
        for value in (0, 1.25, 3.4028234663852886e38):
            self.assertEqual(_time_value(value), value)

    def test_invalid_time_fails_before_allocating_vulkan_resources(self):
        from mlir_gpu.viewer import GraphicsProgram, Shader
        program = GraphicsProgram(Shader("vertex", b""), Shader("fragment", b""))
        with patch("mlir_gpu.viewer._binding") as binding:
            for render in (render_offscreen, show_window):
                with self.assertRaises(ViewerError):
                    render(program, time=float("nan"))
            binding.assert_not_called()


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_VULKAN") == "1", "Enable real Vulkan parameter tests")
class VulkanParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.directory = Path(work.name)
        cls.programs = {}
        for name, pipeline in (
            ("fragment", Pipeline(fullscreen_vertex, frame_diagnostic)),
            ("both", Pipeline(frame_vertex, both_stages)),
            ("sphere", runpy.run_path(str(ROOT / "examples/orbiting_sphere.py"))["pipeline"]),
            ("bands", runpy.run_path(str(ROOT / "examples/frame_bands.py"))["pipeline"]),
        ):
            manifest = compile_pipeline(pipeline, Target.VULKAN, cls.directory / name, TOOLS)
            cls.programs[name] = load_program(manifest)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def test_exact_framebuffer_dimensions_and_changed_time_reach_both_shader_stages(self):
        for name in ("fragment", "both"):
            for width, height, time in ((37, 23, 0), (151, 19, 1), (43, 71, 1.5)):
                with self.subTest(name=name, dimensions=(width, height), time=time):
                    image = render_offscreen(self.programs[name], width, height, self.validation, time=time)
                    self.assertEqual(image.rgba, bytes([width, height, round(time / 4 * 255), 255]) * width * height)

    def test_snapshots_repeat_and_animation_changes_measurable_coverage(self):
        program = self.programs["sphere"]
        first = render_offscreen(program, 96, 72, self.validation, time=0)
        repeated = render_offscreen(program, 96, 72, self.validation, time=0)
        moved = render_offscreen(program, 96, 72, self.validation, time=1.5)
        self.assertEqual(first.rgba, repeated.rgba)
        changed = sum(first.rgba[i:i + 4] != moved.rgba[i:i + 4] for i in range(0, len(first.rgba), 4))
        self.assertGreater(changed / (96 * 72), 0.10)
        self.assertEqual(render_offscreen(program, 96, 72, self.validation).rgba, first.rgba)

    def test_frame_bands_use_pixel_centers_and_fixed_time(self):
        import math
        image = render_offscreen(self.programs["bands"], 37, 23, self.validation, time=1.25)
        for x, y in ((0, 0), (18, 11), (36, 22)):
            uv = (x + 0.5) / 37, (y + 0.5) / 23
            expected = (*uv, math.sin((uv[0] + uv[1]) * 12 - 2.5) * 0.5 + 0.5, 1)
            pixel = image.rgba[(y * 37 + x) * 4:(y * 37 + x + 1) * 4]
            for actual, value in zip(pixel, expected):
                self.assertAlmostEqual(actual, round(value * 255), delta=1)

    def test_cli_forwards_fixed_time_and_reports_invalid_time(self):
        manifest = self.directory / "sphere/pipeline.json"
        output = self.directory / "snapshot.png"
        with patch("sys.argv", ["mlir-gpu-view", str(manifest), "--output", str(output), "--time", "1.5",
                                "--width", "37", "--height", "23"]), contextlib.redirect_stdout(io.StringIO()):
            with patch("mlir_gpu.viewer.render_offscreen", wraps=render_offscreen) as render:
                self.assertEqual(main(), 0)
                self.assertEqual(render.call_args.kwargs, {"time": 1.5})
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))
        with patch("sys.argv", ["mlir-gpu-view", str(manifest), "--time", "nan"]), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(), 1)

    def test_bad_binding_missing_layout_and_zero_framebuffer_fail_explicitly(self):
        parameters = self.programs["fragment"].parameters
        for width, height in ((0, 23), (37, 0)):
            with self.assertRaisesRegex(ViewerError, "positive"):
                _frame_data(parameters, width, height, 0)
        data = json.loads((self.directory / "fragment/pipeline.json").read_text())
        data["interface"]["fragment"]["parameters"]["bindings"]["vulkan"]["offset"] = 4
        invalid = self.directory / "invalid.json"
        invalid.write_text(json.dumps(data))
        with self.assertRaisesRegex(ViewerError, "bindings"):
            load_program(invalid)
        gpu = object.__new__(_Vulkan)
        gpu.vk = None
        with self.assertRaisesRegex(ViewerError, "pipeline layout"):
            gpu.command(None, None, None, None, 37, 23, self.programs["fragment"].draw, parameters=parameters)
        original = _Vulkan.__init__

        def limited(current, *args, **kwargs):
            original(current, *args, **kwargs)
            current.max_push_constants = 0

        with patch.object(_Vulkan, "__init__", limited), self.assertRaisesRegex(ViewerError, "push-constant limit"):
            render_offscreen(self.programs["fragment"], 37, 23, self.validation)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_live_commands_refresh_clock_after_resize_and_zero_iconified_pause(self):
        glfw, vk = _binding("glfw"), _binding("vulkan")
        original_size, original_attrib = glfw.get_framebuffer_size, glfw.get_window_attrib
        pauses, zero_queries = 0, 0
        resized = False
        packets, reported_sizes = [], []
        original_push = vk.vkCmdPushConstants

        def size(window):
            nonlocal resized, zero_queries
            submitted = sum(packet[0] > 0 for packet in packets)
            if submitted == 1 and not resized:
                glfw.set_window_size(window, 240, 180)
                glfw.poll_events()
                resized = True
            if submitted == 1 and zero_queries < 2:
                zero_queries += 1
                return (0, 0)
            actual = original_size(window)
            reported_sizes.append(actual)
            return actual

        def attrib(window, attribute):
            nonlocal pauses
            if attribute == glfw.ICONIFIED and pauses == 0:
                pauses += 1
                return True
            return original_attrib(window, attribute)

        def push(command, layout, stages, offset, count, data):
            packets.append(struct.unpack("<4f", bytes(vk.ffi.buffer(data, count))))
            return original_push(command, layout, stages, offset, count, data)

        ticks = iter(100 + i * 0.25 for i in range(100))
        with patch.object(glfw, "get_framebuffer_size", side_effect=size), patch.object(glfw, "get_window_attrib", side_effect=attrib):
            with patch.object(vk, "vkCmdPushConstants", side_effect=push), patch("mlir_gpu.viewer.clock.monotonic", side_effect=lambda: next(ticks)):
                self.assertEqual(show_window(self.programs["fragment"], 160, 120, frames=3, validation=self.validation), 3)
        submitted = [packet for packet in packets if packet[0] > 0]
        self.assertEqual([packet[0] for packet in submitted], [0.25, 0.5, 0.75])
        self.assertTrue(all((packet[2], packet[3]) in reported_sizes for packet in packets))
        self.assertTrue(all(packet[2] > 0 and packet[3] > 0 for packet in packets))
        self.assertEqual(pauses, 1)
        self.assertEqual(zero_queries, 2)
        self.assertNotEqual(submitted[0][2:], submitted[-1][2:])

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_surface_extent_not_logical_window_size_is_packed(self):
        glfw, vk = _binding("glfw"), _binding("vulkan")
        original_size, original_push = glfw.get_framebuffer_size, vk.vkCmdPushConstants
        sizes, packets = [], []

        def logical_size(window):
            actual = original_size(window)
            sizes.append(actual)
            return tuple(max(value // 2, 1) for value in actual)

        def push(command, layout, stages, offset, count, data):
            packets.append(struct.unpack("<4f", bytes(vk.ffi.buffer(data, count))))
            return original_push(command, layout, stages, offset, count, data)

        with patch.object(glfw, "get_framebuffer_size", side_effect=logical_size), patch.object(vk, "vkCmdPushConstants", side_effect=push):
            self.assertEqual(show_window(self.programs["fragment"], 120, 90, frames=2, validation=self.validation, time=1), 2)
        self.assertTrue(all((packet[2], packet[3]) in sizes for packet in packets))
        self.assertGreater(packets[-1][2], sizes[-1][0] // 2)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_real_minimize_restore_resumes_resource_window(self):
        glfw = _binding("glfw")
        original_create, original_attrib, original_wait = glfw.create_window, glfw.get_window_attrib, glfw.wait_events_timeout
        window = None
        minimized, restored = False, False

        def create(*args):
            nonlocal window
            window = original_create(*args)
            return window

        def attrib(current, attribute):
            nonlocal minimized
            if attribute == glfw.ICONIFIED and not minimized:
                glfw.iconify_window(current)
                glfw.poll_events()
                minimized = True
                return True
            return original_attrib(current, attribute)

        def wait(timeout):
            nonlocal restored
            original_wait(timeout)
            if minimized and not restored:
                glfw.restore_window(window)
                glfw.poll_events()
                restored = True

        with patch.object(glfw, "create_window", side_effect=create), patch.object(glfw, "get_window_attrib", side_effect=attrib):
            with patch.object(glfw, "wait_events_timeout", side_effect=wait):
                self.assertEqual(show_window(self.programs["fragment"], 120, 90, frames=2, validation=self.validation), 2)
        self.assertTrue(minimized and restored)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_live_fixed_time_is_frozen_and_cli_preserves_it(self):
        vk = _binding("vulkan")
        original_push = vk.vkCmdPushConstants
        packets = []

        def push(command, layout, stages, offset, count, data):
            packets.append(struct.unpack("<4f", bytes(vk.ffi.buffer(data, count))))
            return original_push(command, layout, stages, offset, count, data)

        with patch.object(vk, "vkCmdPushConstants", side_effect=push):
            self.assertEqual(show_window(self.programs["both"], 120, 90, frames=3, validation=self.validation, time=2.5), 3)
        self.assertEqual([packet[0] for packet in packets if packet[0] != 0], [2.5] * 3)
        with patch("sys.argv", ["mlir-gpu-view", str(self.directory / "both/pipeline.json"), "--frames", "2", "--time", "2.5"]):
            with patch("mlir_gpu.viewer.show_window", return_value=2) as show, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(), 0)
                self.assertEqual(show.call_args.kwargs, {"time": 2.5})
