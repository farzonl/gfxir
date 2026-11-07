import copy
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlir_gpu import Pipeline, Target, compile_pipeline, fullscreen_vertex
from mlir_gpu.viewer import (
    ViewerError, _metal_libraries, _metal_run, convert_dxil_to_metallib, export_mp4, load_program,
    render_metal_offscreen, show_metal_window,
)
from test_support import ROOT, TOOLS
from test_vulkan_parameters import both_stages, frame_diagnostic, frame_vertex


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_METAL") == "1", "Enable real Metal parameter tests")
class MetalParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.directory = Path(work.name)
        cls.programs = {}
        for target in (Target.DIRECTX, Target.METAL):
            for name, pipeline in (
                ("fragment", Pipeline(fullscreen_vertex, frame_diagnostic)),
                ("both", Pipeline(frame_vertex, both_stages)),
                ("rings", runpy.run_path(str(ROOT / "examples/pulse_rings.py"))["pipeline"]),
            ):
                manifest = compile_pipeline(pipeline, target, cls.directory / f"{target.value}-{name}", TOOLS)
                cls.programs[target.value, name] = load_program(manifest)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def test_exact_dimensions_and_time_reach_fragment_and_vertex_resources(self):
        for target in ("directx", "metal"):
            for name in ("fragment", "both"):
                for width, height, time in ((37, 23, 0), (151, 19, 1), (43, 71, 1.5)):
                    with self.subTest(target=target, stages=name, dimensions=(width, height), time=time):
                        image = render_metal_offscreen(self.programs[target, name], width, height,
                                                       self.validation, TOOLS, time=time)
                        self.assertEqual(image.rgba, bytes([width, height, round(time / 4 * 255), 255]) * width * height)

    def test_native_rejects_invalid_layout_visibility_and_time(self):
        with _metal_libraries(self.programs["metal", "fragment"], TOOLS) as (directory, arguments):
            path = Path(arguments[arguments.index("--frame-layout") + 1])
            original = json.loads(path.read_text())
            for change in ("offset", "binding", "boolean", "visibility"):
                invalid = copy.deepcopy(original)
                if change == "offset":
                    invalid["fields"][1]["offset"] = 4
                elif change == "binding":
                    invalid["bindings"]["metal"]["buffer_index"] = 1
                elif change == "boolean":
                    invalid["fields"][0]["offset"] = False
                else:
                    invalid["stages"] = ["compute"]
                path.write_text(json.dumps(invalid))
                with self.subTest(change=change), self.assertRaisesRegex(ViewerError, "frame"):
                    _metal_run([*arguments, "--width", "37", "--height", "23", "--readback", str(directory / "bad.rgba")],
                               self.validation, True)
            path.write_text(json.dumps(original))
            for time in ("nan", "inf", "-1", "1.2bad", "3.5e38"):
                with self.subTest(time=time), self.assertRaisesRegex(ViewerError, "Time"):
                    _metal_run([*arguments, "--width", "37", "--height", "23", "--time", time,
                                "--readback", str(directory / "bad.rgba")], self.validation, True)

    def test_rings_are_deterministic_and_change_without_recompilation(self):
        for target in ("directx", "metal"):
            program = self.programs[target, "rings"]
            first = render_metal_offscreen(program, 96, 72, self.validation, TOOLS)
            self.assertEqual(first.rgba, render_metal_offscreen(program, 96, 72, self.validation, TOOLS, time=0).rgba)
            moved = render_metal_offscreen(program, 96, 72, self.validation, TOOLS, time=0.4)
            changed = sum(first.rgba[i:i + 4] != moved.rgba[i:i + 4] for i in range(0, len(first.rgba), 4))
            self.assertGreater(changed / (96 * 72), 0.20)

    def test_video_converts_libraries_once_and_renders_all_fixed_times(self):
        program = self.programs["directx", "rings"]
        output = self.directory / "rings.mp4"
        with patch("mlir_gpu.viewer.convert_dxil_to_metallib", wraps=convert_dxil_to_metallib) as convert:
            self.assertEqual(export_mp4(program, output, 96, 72, duration=0.5, fps=8,
                                       start_time=0.25, validation=self.validation), 4)
            self.assertEqual(convert.call_count, 2)
        self.assertGreater(output.stat().st_size, 1000)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window tests")
    def test_live_and_frozen_resource_windows_present(self):
        show_metal_window(self.programs["directx", "rings"], 160, 120, 3, self.validation, TOOLS)
        show_metal_window(self.programs["metal", "both"], 160, 120, 3, self.validation, TOOLS, time=1.5)
