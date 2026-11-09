import contextlib
import copy
import io
import json
import os
import runpy
import select
import struct
import subprocess
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from mlir_gpu import CompileError, Target, Toolchain, compile_pipeline
from mlir_gpu.compiler import PIPELINE_SCHEMA_VERSION, RASTER_CONVENTION
from mlir_gpu.frontend import Draw
from mlir_gpu.viewer import (
    RenderedImage, ViewerError, _binding, _dimensions, _directx_run, _directx_shaders, _metal_libraries, _metal_run, _Vulkan,
    load_program, main, render_directx_offscreen, render_metal_offscreen, render_offscreen,
    show_directx_window, show_metal_window, show_window, write_png,
)


ROOT = Path(__file__).resolve().parents[1]


def manifest_data():
    varying = {"name": "color", "type": "f32x4", "location": 0, "interpolation": "perspective"}
    return {
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "target": "vulkan",
        "interface": {
            "vertex": {
                "name": "triangle_vertex", "stage": "vertex",
                "inputs": [{"name": "vertex_index", "builtin": "vertex_index", "type": "u32"}],
                "outputs": [{"name": "position", "builtin": "position", "type": "f32x4"}, dict(varying)],
            },
            "fragment": {
                "name": "triangle_fragment", "stage": "fragment", "inputs": [dict(varying)],
                "outputs": [{"name": "target0", "type": "f32x4", "location": 0, "interpolation": "perspective"}],
            },
        },
        "draw": Draw().to_dict(),
        "raster_convention": dict(RASTER_CONVENTION),
        "entries": [
            {"stage": "vertex", "entry_point": "triangle_vertex", "binary": "vertex.spv"},
            {"stage": "fragment", "entry_point": "triangle_fragment", "binary": "fragment.spv"},
        ],
    }


class ViewerTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)
        self.manifest = self.directory / "pipeline.json"
        self.data = manifest_data()
        self.save_manifest()
        for name in ("vertex.spv", "fragment.spv"):
            (self.directory / name).write_bytes(b"\x03\x02\x23\x07" + bytes(16))

    def save_manifest(self):
        self.manifest.write_text(json.dumps(self.data))

    def test_load_compiled_artifact_contract(self):
        program = load_program(self.manifest)
        self.assertEqual(program.vertex.entry_point, "triangle_vertex")
        self.assertEqual(program.fragment.entry_point, "triangle_fragment")
        self.assertEqual(len(program.vertex.code), 20)

    def test_reject_unsupported_contracts(self):
        for key, value in (
            ("schema_version", 99), ("schema_version", True), ("target", "opengl"),
            ("raster_convention", {}), ("raster_convention", {**RASTER_CONVENTION, "clip_y": "down"}),
            ("interface", {}), ("draw", {"topology": "triangle-list", "vertex_count": 4}),
            ("entries", []), ("entries", [{}, {}]), ("entries", [None, None]),
        ):
            with self.subTest(key=key, value=value):
                self.data = manifest_data()
                self.data[key] = value
                self.save_manifest()
                with self.assertRaises(ViewerError):
                    load_program(self.manifest)

    def test_reflected_vec2_interfaces_and_six_vertex_draw_are_not_triangle_abi(self):
        self.data["draw"] = Draw(vertex_count=6).to_dict()
        self.data["interface"]["vertex"]["outputs"][1]["type"] = "f32x2"
        self.data["interface"]["fragment"]["inputs"][0]["type"] = "f32x2"
        self.save_manifest()
        self.assertEqual(load_program(self.manifest).draw.vertex_count, 6)

    def test_reflection_linking_rejects_mismatched_varying_types(self):
        self.data["interface"]["fragment"]["inputs"][0]["type"] = "f32x2"
        self.save_manifest()
        with self.assertRaisesRegex(ViewerError, "does not match"):
            load_program(self.manifest)

    def test_corrected_version_two_artifacts_remain_viewable(self):
        self.data["schema_version"] = 2
        self.data["interface"] = {
            "vertex_input": "vertex_index:u32",
            "vertex_output": {"position": "clip_space:f32x4", "color": "location0:f32x4"},
            "fragment_input": {"color": "location0:f32x4"},
            "fragment_output": {"color": "target0:f32x4"},
        }
        self.save_manifest()
        self.assertEqual(load_program(self.manifest).draw, Draw())

    def test_old_artifacts_require_recompilation_not_silent_coordinate_changes(self):
        self.data["schema_version"] = 1
        self.save_manifest()
        with self.assertRaisesRegex(ViewerError, "Recompile"):
            load_program(self.manifest)

    def test_default_and_explicit_zero_frames_open_live_windows(self):
        for frame_arguments in ([], ["-frames", "0"], ["--frames", "0"]):
            for target in ("vulkan", "directx", "metal"):
                with self.subTest(frames=frame_arguments, target=target):
                    if target == "vulkan":
                        self.data = manifest_data()
                        for name in ("vertex.spv", "fragment.spv"):
                            (self.directory / name).write_bytes(b"\x03\x02\x23\x07" + bytes(16))
                        self.save_manifest()
                    else:
                        self.native_program(target)
                    arguments = ["mlir-gpu-view", str(self.manifest), *frame_arguments]
                    for platform in ("win32", "darwin"):
                        with patch("sys.argv", arguments), patch("mlir_gpu.viewer.sys.platform", platform), contextlib.redirect_stdout(io.StringIO()):
                            with patch("mlir_gpu.viewer.show_window", return_value=0) as vulkan, patch("mlir_gpu.viewer.show_metal_window") as metal, patch("mlir_gpu.viewer.show_directx_window") as directx, patch("mlir_gpu.viewer.write_png") as png:
                                self.assertEqual(main(), 0)
                                active = vulkan if target == "vulkan" else directx if target == "directx" and platform == "win32" else metal
                                self.assertEqual(active.call_args.args[3], 0)
                                for inactive in (vulkan, metal, directx):
                                    if inactive is not active:
                                        inactive.assert_not_called()
                                png.assert_not_called()

    def test_png_and_live_modes_are_explicitly_exclusive(self):
        arguments = ["mlir-gpu-view", str(self.manifest), "--output", "preview.png", "-frames", "0"]
        with patch("sys.argv", arguments), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                main()
        self.assertEqual(result.exception.code, 2)

    def native_program(self, target):
        self.data["target"] = target
        magic, minimum = (b"DXBC", 32) if target == "directx" else (b"MTLB", 16)
        for entry in self.data["entries"]:
            (self.directory / entry["binary"]).write_bytes(magic + bytes(minimum - len(magic)))
        self.save_manifest()
        return load_program(self.manifest)

    def test_load_native_binary_formats(self):
        for target in ("directx", "metal"):
            self.assertEqual(self.native_program(target).target, target)

    def test_directx_png_routing_on_windows_and_macos(self):
        self.native_program("directx")
        for platform in ("win32", "darwin"):
            with patch("sys.argv", ["mlir-gpu-view", str(self.manifest), "--output", str(self.directory / "preview.png")]):
                with patch("mlir_gpu.viewer.sys.platform", platform), contextlib.redirect_stdout(io.StringIO()):
                    with patch("mlir_gpu.viewer.render_directx_offscreen") as directx, patch("mlir_gpu.viewer.render_metal_offscreen") as metal, patch("mlir_gpu.viewer.write_png"):
                        self.assertEqual(main(), 0)
                        active, inactive = (directx, metal) if platform == "win32" else (metal, directx)
                        active.assert_called_once()
                        inactive.assert_not_called()

    def test_directx_cli_preserves_fixed_time_on_windows_and_macos(self):
        self.native_program("directx")
        for platform in ("win32", "darwin"):
            for png in (False, True):
                arguments = ["mlir-gpu-view", str(self.manifest), "--time", "1.5"]
                if png:
                    arguments.extend(["--output", str(self.directory / "snapshot.png")])
                native = "render_directx_offscreen" if png else "show_directx_window"
                metal = "render_metal_offscreen" if png else "show_metal_window"
                with self.subTest(platform=platform, png=png):
                    with patch("sys.argv", arguments), patch("mlir_gpu.viewer.sys.platform", platform), contextlib.redirect_stdout(io.StringIO()):
                        with patch(f"mlir_gpu.viewer.{native}") as directx, patch(f"mlir_gpu.viewer.{metal}") as converted, patch("mlir_gpu.viewer.write_png"):
                            self.assertEqual(main(), 0)
                            active, inactive = (directx, converted) if platform == "win32" else (converted, directx)
                            self.assertEqual(active.call_args.kwargs, {"time": 1.5})
                            inactive.assert_not_called()

    def test_directx_resource_and_invalid_time_limits_fail_before_launch(self):
        program = self.native_program("directx")
        parameterized = type(program)(program.vertex, program.fragment, program.target, program.draw, parameters={})
        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find") as find:
            for render in (render_directx_offscreen, show_directx_window):
                with self.subTest(render=render.__name__):
                    with self.assertRaisesRegex(ViewerError, "Invalid frame parameter resource"):
                        render(parameterized)
                    with self.assertRaisesRegex(ViewerError, "Time"):
                        render(program, time=float("nan"))
            find.assert_not_called()

    def test_directx_parameterized_offscreen_passes_time(self):
        from test_parameters import FRAME_REFLECTION
        program = self.native_program("directx")
        parameterized = type(program)(
            program.vertex, program.fragment, program.target, program.draw,
            parameters=FRAME_REFLECTION,
        )

        def mock_renderer(arguments, validation, capture):
            self.assertNotIn("--frame-layout", arguments)
            self.assertIn("--time", arguments)
            self.assertEqual(arguments[arguments.index("--time") + 1], "2.5")
            readback_path = Path(arguments[arguments.index("--readback") + 1])
            readback_path.write_bytes(bytes([10, 20, 30, 255]))
            return '{"device":"D3D12 Test GPU","width":1,"height":1}'

        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find", side_effect=lambda name: name):
            with patch("mlir_gpu.viewer._directx_run", side_effect=mock_renderer):
                image = render_directx_offscreen(parameterized, 1, 1, time=2.5)
        self.assertEqual(image.width, 1)
        self.assertEqual(image.height, 1)
        self.assertEqual(image.rgba, bytes([10, 20, 30, 255]))

    def test_directx_parameterized_window_passes_fixed_time(self):
        from test_parameters import FRAME_REFLECTION
        program = self.native_program("directx")
        parameterized = type(program)(
            program.vertex, program.fragment, program.target, program.draw,
            parameters=FRAME_REFLECTION,
        )

        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find", side_effect=lambda name: name):
            with patch("mlir_gpu.viewer._directx_run") as renderer:
                show_directx_window(parameterized, 800, 600, frames=10, time=3.0)
                arguments, validation, capture = renderer.call_args.args
                self.assertNotIn("--frame-layout", arguments)
                self.assertIn("--time", arguments)
                self.assertEqual(arguments[arguments.index("--time") + 1], "3.0")
                self.assertIn("--frames", arguments)
                self.assertEqual(arguments[arguments.index("--frames") + 1], "10")

    def test_directx_custom_parameter_block_rejected_before_launch(self):
        import copy
        from test_parameters import FRAME_REFLECTION
        program = self.native_program("directx")
        custom_block = copy.deepcopy(FRAME_REFLECTION)
        custom_block["name"] = "material"
        parameterized = type(program)(
            program.vertex, program.fragment, program.target, program.draw,
            parameters=custom_block,
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find") as find:
            for render in (render_directx_offscreen, show_directx_window):
                with self.subTest(render=render.__name__):
                    with self.assertRaisesRegex(ViewerError, "supplies FrameParameters only"):
                        render(parameterized)
            find.assert_not_called()

    def test_directx_native_readback_uses_original_dxil_and_draw_count(self):
        program = self.native_program("directx")
        program = type(program)(program.vertex, program.fragment, program.target, Draw(vertex_count=6))
        paths = []

        def renderer(arguments, validation, capture):
            self.assertTrue(validation)
            self.assertTrue(capture)
            self.assertEqual(arguments[0], "mlir-gpu-directx-view")
            self.assertEqual(arguments[arguments.index("--vertex-count") + 1], "6")
            for stage, code in (("vertex", program.vertex.code), ("fragment", program.fragment.code)):
                path = Path(arguments[arguments.index(f"--{stage}") + 1])
                paths.append(path)
                self.assertEqual(path.read_bytes(), code)
            Path(arguments[arguments.index("--readback") + 1]).write_bytes(bytes([1, 2, 3, 255]))
            return '{"device":"D3D12 test GPU","width":1,"height":1}'

        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find", side_effect=lambda name: name):
            with patch("mlir_gpu.viewer._directx_run", side_effect=renderer), patch("mlir_gpu.viewer._run") as compiler:
                image = render_directx_offscreen(program, 1, 1, True)
                compiler.assert_not_called()
        self.assertEqual(image, RenderedImage(1, 1, bytes([1, 2, 3, 255]), "D3D12 test GPU"))
        self.assertTrue(all(not path.exists() for path in paths))

    def test_directx_validation_and_failure_diagnostics(self):
        result = subprocess.CompletedProcess(["native"], 1, "", "D3D12 shader signature rejected")
        with patch("mlir_gpu.viewer.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(ViewerError, "D3D12 shader signature rejected"):
                _directx_run(["native"], True, True)
            self.assertEqual(run.call_args.args[0], ["native", "--validation"])
        with patch("mlir_gpu.viewer.subprocess.run", side_effect=OSError("missing executable")):
            with self.assertRaisesRegex(ViewerError, "Cannot start.*DirectX 12"):
                _directx_run(["native"], False, True)

    def test_directx_native_readback_rejects_invalid_reports_and_sizes(self):
        program = self.native_program("directx")
        for report, rgba in (
            ('{}', bytes(4)),
            ('{"device":"GPU","width":true,"height":1}', bytes(4)),
            ('{"device":"","width":1,"height":1}', bytes(4)),
            ('{"device":"GPU","width":2,"height":1}', bytes(4)),
            ('{"device":"GPU","width":1,"height":1}', bytes(3)),
        ):
            def renderer(arguments, validation, capture):
                Path(arguments[arguments.index("--readback") + 1]).write_bytes(rgba)
                return report

            with self.subTest(report=report, length=len(rgba)):
                with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find", return_value="native"), patch("mlir_gpu.viewer._directx_run", side_effect=renderer):
                    with self.assertRaises(ViewerError):
                        render_directx_offscreen(program, 1, 1)

    def test_directx_native_window_defaults_and_platform_guards(self):
        program = self.native_program("directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"), patch.object(Toolchain, "find", return_value="native"):
            with patch("mlir_gpu.viewer._directx_run") as renderer:
                show_directx_window(program)
                arguments, validation, capture = renderer.call_args.args
                self.assertEqual(arguments[arguments.index("--frames") + 1], "0")
                self.assertFalse(validation)
                self.assertFalse(capture)
            for frames in (-1, True, 1.5):
                with self.assertRaisesRegex(ViewerError, "Frame count"):
                    show_directx_window(program, frames=frames)
            with self.assertRaisesRegex(ViewerError, "requires DirectX artifacts"):
                render_directx_offscreen(self.native_program("metal"))
        with patch("mlir_gpu.viewer.sys.platform", "linux"):
            with self.assertRaisesRegex(ViewerError, "requires Windows"):
                render_directx_offscreen(program)

    def test_directx_preview_validates_converts_and_reads_back(self):
        program = self.native_program("directx")
        calls = []

        def compiler(arguments):
            calls.append(arguments)
            output = Path(arguments[arguments.index("-o") + 1])
            output.write_bytes(b"MTLB" + bytes(12) if output.suffix == ".metallib" else b"DXBC" + bytes(28))
            return ""

        def renderer(arguments, validation, capture):
            self.assertTrue(validation)
            self.assertTrue(capture)
            Path(arguments[arguments.index("--readback") + 1]).write_bytes(bytes([1, 2, 3, 255]))
            return '{"device": "native test GPU", "width": 1, "height": 1}'

        with patch("mlir_gpu.viewer.sys.platform", "darwin"), patch.object(Toolchain, "find", side_effect=lambda name: name):
            with patch("mlir_gpu.viewer._run", side_effect=compiler), patch("mlir_gpu.compiler._run", side_effect=compiler):
                with patch("mlir_gpu.viewer._metal_run", side_effect=renderer):
                    image = render_metal_offscreen(program, 1, 1, True)
        self.assertEqual(image.rgba, bytes([1, 2, 3, 255]))
        self.assertEqual(image.device, "native test GPU")
        self.assertEqual([call[0] for call in calls], ["dxv", "metal-shaderconverter"] * 2)
        for call in calls:
            if call[0] == "metal-shaderconverter":
                self.assertTrue(any(arg.startswith("--rename-entry-point=") for arg in call))

    def test_metal_window_has_no_default_frame_limit_or_reconversion(self):
        program = self.native_program("metal")
        with patch("mlir_gpu.viewer.sys.platform", "darwin"), patch.object(Toolchain, "find", side_effect=lambda name: name):
            with patch("mlir_gpu.viewer._run") as compile_tool, patch("mlir_gpu.viewer._metal_run", return_value="") as renderer:
                show_metal_window(program)
                compile_tool.assert_not_called()
                arguments, validation, capture = renderer.call_args.args
        self.assertEqual(arguments[arguments.index("--frames") + 1], "0")
        self.assertFalse(capture)
        self.assertFalse(validation)

    def test_native_renderer_failure_preserves_diagnostics(self):
        result = subprocess.CompletedProcess(["native"], 1, "", "Metal pipeline rejected")
        with patch("mlir_gpu.viewer.subprocess.run", return_value=result) as run:
            with self.assertRaisesRegex(ViewerError, "Metal pipeline rejected"):
                _metal_run(["native"], True, True)
            self.assertEqual(run.call_args.kwargs["env"]["MTL_DEBUG_LAYER"], "1")
            self.assertEqual(run.call_args.kwargs["env"]["MTL_SHADER_VALIDATION"], "1")

    def test_metal_preview_requires_macos(self):
        program = self.native_program("metal")
        with patch("mlir_gpu.viewer.sys.platform", "linux"):
            with self.assertRaisesRegex(ViewerError, "requires macOS"):
                render_metal_offscreen(program)

    def test_reject_duplicate_stages_and_invalid_names(self):
        for key, value in (("stage", "fragment"), ("entry_point", "invalid-name"), ("binary", "")):
            with self.subTest(key=key):
                self.data = manifest_data()
                self.data["entries"][0][key] = value
                self.save_manifest()
                with self.assertRaises(ViewerError):
                    load_program(self.manifest)

    def test_reject_escaping_shader_paths(self):
        for filename in ("../outside.spv", str(self.directory / "vertex.spv")):
            with self.subTest(filename=filename):
                self.data["entries"][0]["binary"] = filename
                self.save_manifest()
                with self.assertRaisesRegex(ViewerError, "inside the pipeline"):
                    load_program(self.manifest)

    def test_reject_missing_invalid_and_unaligned_spirv(self):
        binary = self.directory / "vertex.spv"
        binary.unlink()
        with self.assertRaises(CompileError):
            load_program(self.manifest)
        binary.write_bytes(bytes(20))
        with self.assertRaises(CompileError):
            load_program(self.manifest)
        binary.write_bytes(b"\x03\x02\x23\x07" + bytes(17))
        with self.assertRaisesRegex(ViewerError, "word-aligned"):
            load_program(self.manifest)

    def test_png_pixels_and_chunk_checksums(self):
        pixels = bytes([255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255])
        image = RenderedImage(2, 2, pixels, "test device")
        output = self.directory / "image.png"
        write_png(output, image)
        encoded = output.read_bytes()
        self.assertEqual(encoded[:8], b"\x89PNG\r\n\x1a\n")
        offset, chunks = 8, {}
        while offset < len(encoded):
            length = struct.unpack_from(">I", encoded, offset)[0]
            kind = encoded[offset + 4:offset + 8]
            payload = encoded[offset + 8:offset + 8 + length]
            crc = struct.unpack_from(">I", encoded, offset + 8 + length)[0]
            self.assertEqual(crc, zlib.crc32(kind + payload))
            chunks[kind] = payload
            offset += length + 12
        self.assertEqual(struct.unpack(">IIBBBBB", chunks[b"IHDR"]), (2, 2, 8, 6, 0, 0, 0))
        self.assertEqual(zlib.decompress(chunks[b"IDAT"]), b"\0" + pixels[:8] + b"\0" + pixels[8:])
        self.assertEqual(chunks[b"IEND"], b"")
        with self.assertRaises(FileExistsError):
            write_png(output, image)

    def test_invalid_dimensions_and_readback_rejected(self):
        for width, height in ((0, 1), (1, -1), (4097, 1), (True, 1), (1.0, 1)):
            with self.assertRaises(ViewerError):
                _dimensions(width, height)
        with self.assertRaisesRegex(ViewerError, "readback size"):
            write_png(self.directory / "invalid.png", RenderedImage(2, 2, b"", "test"))

    def test_missing_optional_dependencies_and_loader_are_actionable(self):
        for error in (ModuleNotFoundError("missing vulkan"), OSError("missing loader")):
            with patch("mlir_gpu.viewer.importlib.import_module", side_effect=error):
                with self.assertRaisesRegex(ViewerError, r"\.\[viewer\]"):
                    _binding("vulkan")

    def test_validation_errors_cannot_report_success(self):
        gpu = object.__new__(_Vulkan)
        gpu.validation_errors = ["invalid resource lifetime"]
        with self.assertRaisesRegex(ViewerError, "invalid resource lifetime"):
            gpu.check_validation()

    def test_cli_reports_errors_and_preserves_existing_images(self):
        output = self.directory / "keep.png"
        output.write_bytes(b"keep")
        arguments = ["mlir-gpu-view", str(self.manifest), "--output", str(output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch("sys.argv", arguments), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            with patch("mlir_gpu.viewer.render_offscreen") as render:
                self.assertEqual(main(), 1)
                render.assert_not_called()
        self.assertEqual(output.read_bytes(), b"keep")
        self.assertIn("already exists", stderr.getvalue())
        self.assertEqual(stdout.getvalue(), "")

    def test_cli_writes_real_render_result(self):
        output = self.directory / "preview.png"
        image = RenderedImage(1, 1, bytes([255, 0, 0, 255]), "test GPU")
        arguments = ["mlir-gpu-view", str(self.manifest), "--output", str(output), "--width", "1", "--height", "1"]
        with patch("sys.argv", arguments), contextlib.redirect_stdout(io.StringIO()):
            with patch("mlir_gpu.viewer.render_offscreen", return_value=image) as render:
                self.assertEqual(main(), 0)
                self.assertEqual(render.call_args.args[1:], (1, 1, False))
        self.assertTrue(output.read_bytes().startswith(b"\x89PNG"))


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_VULKAN") == "1", "Set MLIR_GPU_TEST_VULKAN=1 to run real Vulkan rendering")
class VulkanRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        llvm = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
        tools = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
        pipeline = runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"]
        manifest = compile_pipeline(pipeline, Target.VULKAN, Path(work.name) / "pipeline", Toolchain(llvm, tools))
        cls.program = load_program(manifest)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def test_gpu_rasterization_coverage_and_interpolated_colors(self):
        image = render_offscreen(self.program, 256, 256, self.validation)
        self.assertTrue(image.device)
        self.assertEqual(len(image.rgba), 256 * 256 * 4)

        def pixel(x, y):
            offset = (y * 256 + x) * 4
            return image.rgba[offset:offset + 4]

        for x, y in ((0, 0), (255, 255), (128, 15), (20, 128)):
            self.assertEqual(pixel(x, y), bytes([5, 5, 10, 255]))
        for x, y in ((128, 100), (65, 185), (190, 185), (128, 65)):
            ndc_x, ndc_y = (x + 0.5) / 128 - 1, 1 - (y + 0.5) / 128
            blue = (ndc_y + 0.6) / 1.2
            green = ((1 - blue) + ndc_x / 0.6) / 2
            red = 1 - green - blue
            actual = pixel(x, y)
            for channel, expected in zip(actual[:3], (red, green, blue)):
                self.assertAlmostEqual(channel, round(expected * 255), delta=2)
            self.assertEqual(actual[3], 255)
        coverage = sum(
            image.rgba[offset:offset + 3] != bytes([5, 5, 10])
            for offset in range(0, len(image.rgba), 4)
        ) / (256 * 256)
        self.assertGreater(coverage, 0.175)
        self.assertLess(coverage, 0.185)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Set MLIR_GPU_TEST_WINDOW=1 in a desktop session")
    def test_real_window_presents_requested_frames(self):
        self.assertEqual(show_window(self.program, 160, 120, frames=3, validation=self.validation), 3)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Set MLIR_GPU_TEST_WINDOW=1 in a desktop session")
    def test_window_recreates_swapchain_after_resize(self):
        glfw = _binding("glfw")
        original_size = glfw.get_framebuffer_size
        queries = 0

        def resize(window):
            nonlocal queries
            queries += 1
            if queries == 2:
                glfw.set_window_size(window, 240, 180)
                glfw.poll_events()
            return original_size(window)

        with patch.object(glfw, "get_framebuffer_size", side_effect=resize):
            self.assertEqual(show_window(self.program, 160, 120, frames=3, validation=self.validation), 3)
        self.assertGreater(queries, 2)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Set MLIR_GPU_TEST_WINDOW=1 in a desktop session")
    def test_window_pauses_while_iconified(self):
        glfw = _binding("glfw")
        original_attrib = glfw.get_window_attrib
        iconified = True

        def attrib(window, attribute):
            nonlocal iconified
            if attribute == glfw.ICONIFIED and iconified:
                iconified = False
                return 1
            return original_attrib(window, attribute)

        with patch.object(glfw, "get_window_attrib", side_effect=attrib):
            with patch.object(glfw, "wait_events_timeout", wraps=glfw.wait_events_timeout) as wait:
                self.assertEqual(show_window(self.program, 160, 120, frames=1, validation=self.validation), 1)
                wait.assert_called_with(0.1)


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_METAL") == "1", "Set MLIR_GPU_TEST_METAL=1 for real DXIL-to-Metal rendering")
class MetalRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        llvm = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
        tools = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
        cls.tools = Toolchain(llvm, tools)
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        pipeline = runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"]
        cls.programs = {}
        for target in (Target.DIRECTX, Target.METAL):
            manifest = compile_pipeline(pipeline, target, Path(work.name) / target.value, cls.tools)
            cls.programs[target.value] = load_program(manifest)

    def test_native_dxil_and_metallib_pixels(self):
        images = []
        for target, program in self.programs.items():
            with self.subTest(target=target):
                image = render_metal_offscreen(program, 256, 256, self.validation, self.tools)
                self.assertTrue(image.device)
                self.assertEqual(len(image.rgba), 256 * 256 * 4)
                images.append(image.rgba)

                def pixel(x, y):
                    offset = (y * 256 + x) * 4
                    return image.rgba[offset:offset + 4]

                for x, y in ((0, 0), (255, 255), (128, 15), (20, 128)):
                    self.assertEqual(pixel(x, y), bytes([5, 5, 10, 255]))
                for x, y in ((128, 100), (65, 185), (190, 185), (128, 65)):
                    ndc_x, ndc_y = (x + 0.5) / 128 - 1, 1 - (y + 0.5) / 128
                    blue = (ndc_y + 0.6) / 1.2
                    green = ((1 - blue) + ndc_x / 0.6) / 2
                    red = 1 - green - blue
                    for channel, expected in zip(pixel(x, y)[:3], (red, green, blue)):
                        self.assertAlmostEqual(channel, round(expected * 255), delta=2)
                    self.assertEqual(pixel(x, y)[3], 255)
                coverage = sum(
                    image.rgba[offset:offset + 3] != bytes([5, 5, 10])
                    for offset in range(0, len(image.rgba), 4)
                ) / (256 * 256)
                self.assertGreater(coverage, 0.175)
                self.assertLess(coverage, 0.185)
        self.assertEqual(images[0], images[1])

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Set MLIR_GPU_TEST_WINDOW=1 in a desktop session")
    def test_directx_manifest_opens_a_real_metal_window(self):
        show_metal_window(self.programs["directx"], 160, 120, 3, self.validation, self.tools)

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Set MLIR_GPU_TEST_WINDOW=1 in a desktop session")
    def test_native_window_stays_open_without_a_frame_limit(self):
        with _metal_libraries(self.programs["metal"], self.tools) as (_, arguments):
            for frame_arguments in ([], ["-frames", "0"], ["--frames", "0"]):
                with self.subTest(frames=frame_arguments):
                    process = subprocess.Popen(
                        [*arguments, "--width", "160", "--height", "120", *frame_arguments],
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                    )
                    try:
                        ready, _, _ = select.select([process.stdout], [], [], 15)
                        self.assertTrue(ready, "Native Metal window did not report readiness")
                        self.assertIn("DXIL-to-Metal viewer ready", process.stdout.readline())
                        time.sleep(0.5)
                        self.assertIsNone(process.poll(), "Unbounded Metal window closed without user input")
                    finally:
                        if process.poll() is None:
                            process.terminate()
                        try:
                            process.communicate(timeout=10)
                        except subprocess.TimeoutExpired:
                            process.kill()
                            process.communicate(timeout=10)


if __name__ == "__main__":
    unittest.main()
