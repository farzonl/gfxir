import contextlib
import io
import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlir_gpu import CompileError, Pipeline, Target, Toolchain, Vertex, fragment, i32, maximum, minimum, select, u32, vec3, vec4, vertex
from mlir_gpu.__main__ import main
from mlir_gpu.compiler import PIPELINE_SCHEMA_VERSION, RASTER_CONVENTION, _check_binary, _run
from mlir_gpu.compiler import compile_pipeline


ROOT = Path(__file__).resolve().parents[1]
LLVM_BIN = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
GPU_TOOLS_BIN = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
TOOLS = Toolchain(LLVM_BIN, GPU_TOOLS_BIN)


def triangle():
    return runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"]


def arithmetic_pipeline():
    @vertex
    def arithmetic_vertex(index):
        integer = (2 + index) * 2 - 1
        predicate = (integer == 3) == True
        x = select(predicate, 0.25, 0.5)
        position = vec4(x + 0.25, 1.0 - x, x * 0.0, 1.0)
        white = vec4(1, 1, 1, 1)
        color = select(x == 0.25, white, vec4(0, 0, 0, 1))
        return Vertex(position, (color + white - white) * white)

    @fragment
    def arithmetic_fragment(color):
        return color

    return Pipeline(arithmetic_vertex, arithmetic_fragment)


def unused_varying_pipeline():
    quad = runpy.run_path(str(ROOT / "examples/quad.py"))["pipeline"]

    @vertex
    def extra_vertex(index):
        result = quad.vertex.function(index)
        return Vertex(result.position, varyings={"unused": vec3(1, 0, 0), **result.varyings})

    @fragment
    def extra_fragment(unused, uv):
        return quad.fragment.function(uv)

    return Pipeline(extra_vertex, extra_fragment, draw=quad.draw)


def expanded_arithmetic_pipeline():
    quad = runpy.run_path(str(ROOT / "examples/quad.py"))["pipeline"]

    @vertex
    def expanded_vertex(index):
        result = quad.vertex.function(index)
        unsigned = index + 2**31
        signed = i32(unsigned)
        absolute_ok = select(index == 0, abs(signed) == signed, abs(signed) >= 0)
        flag = (
            (signed < 0) & (unsigned >= 2**31) & (signed <= i32(index) + (-2**31))
            & (minimum(signed, 0) == signed) & (maximum(signed, 0) == 0)
            & (minimum(unsigned, 0) == 0) & (maximum(unsigned, 0) == unsigned)
            & ((-signed) == (0 - signed)) & (u32(signed) == unsigned) & absolute_ok
        )
        return Vertex(result.position, varyings={
            "uv": result.varyings["uv"], "integer_ok": select(flag, 1.0, 0.0),
        })

    @fragment
    def expanded_fragment(uv, integer_ok):
        point = vec3(uv * 2 - 1, -uv.x)
        surface = minimum(maximum(abs(-point), 0.0), 1.0).with_components(z=minimum(uv.x, uv.y))
        value = maximum(surface.xy, 0.125) / 2 + 0.125 / maximum(surface.xy, 0.5)
        predicate = (point.x < 0) & ~(point.y >= 0)
        zero = uv.x - uv.x
        nan = zero / zero
        numeric_ok = (
            (minimum(nan, 0.5) == 0.5) & (maximum(0.25, nan) == 0.25)
            & (minimum(nan, nan) != nan) & (abs(-zero) == 0)
        )
        color = vec4(value.x * integer_ok * select(numeric_ok, 1.0, 0.0),
                     surface.z, select(predicate | False, 0.25, 0.75), 1)
        return minimum(maximum(abs(-color), 0.0), 1.0)

    return Pipeline(expanded_vertex, expanded_fragment, draw=quad.draw)


def available(*names):
    try:
        for name in names:
            TOOLS.find(name)
    except CompileError:
        return False
    return True


class CompilerTests(unittest.TestCase):
    def test_missing_tool_has_actionable_diagnostic(self):
        with patch("mlir_gpu.compiler.shutil.which", return_value=None):
            with self.assertRaisesRegex(CompileError, "no MSL/HLSL fallback"):
                Toolchain().find("llc")

    def test_explicit_tool_directory_does_not_silently_fall_back(self):
        with tempfile.TemporaryDirectory() as work:
            with self.assertRaisesRegex(CompileError, "missing or not executable"):
                Toolchain(llvm_bin=Path(work)).find("llc")

    def test_directx_preflight_uses_upstream_llvm_directx_backend(self):
        with patch.object(Toolchain, "find", side_effect=lambda name: name):
            tools = Toolchain().preflight(Target.DIRECTX)
        self.assertIn("gfx-compile", tools)
        self.assertIn("dxv", tools)
        self.assertIn("llc", tools)
        self.assertNotIn("dxil-assemble", tools)

    def test_failed_command_preserves_diagnostics(self):
        result = subprocess.CompletedProcess(["tool"], 7, "", "invalid stage interface")
        with patch("mlir_gpu.compiler.subprocess.run", return_value=result):
            with self.assertRaisesRegex(CompileError, r"Command failed \(7\).*"):
                _run(["tool", "some path"])
            try:
                _run(["tool"])
            except CompileError as error:
                self.assertIn("invalid stage interface", str(error))

    def test_unexecutable_command_reports_os_error(self):
        with patch("mlir_gpu.compiler.subprocess.run", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(CompileError, "Could not execute"):
                _run(["tool"])

    def test_invalid_or_missing_binaries_rejected(self):
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "shader.dxil"
            with self.assertRaisesRegex(CompileError, "did not produce"):
                _check_binary(path, b"DXBC", 32)
            path.write_bytes(b"DXBC")
            with self.assertRaisesRegex(CompileError, "invalid"):
                _check_binary(path, b"DXBC", 32)
            path.write_bytes(b"not DXIL".ljust(32, b"\0"))
            with self.assertRaisesRegex(CompileError, "invalid"):
                _check_binary(path, b"DXBC", 32)

    def test_existing_output_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as work:
            marker = Path(work) / "keep"
            marker.write_text("unchanged")
            with self.assertRaisesRegex(CompileError, "already exists"):
                compile_pipeline(triangle(), Target.VULKAN, Path(work))
            self.assertEqual(marker.read_text(), "unchanged")

    def test_failed_stage_leaves_no_partial_artifacts(self):
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "out"
            with patch.object(Toolchain, "preflight", return_value={}):
                with patch("mlir_gpu.compiler._compile_stage", side_effect=CompileError("failed")):
                    with self.assertRaisesRegex(CompileError, "failed"):
                        compile_pipeline(triangle(), Target.VULKAN, output)
            self.assertEqual(list(Path(work).iterdir()), [])

    def test_second_stage_failure_discards_first_stage_artifacts(self):
        with tempfile.TemporaryDirectory() as work:
            output = Path(work) / "out"

            def compile_stage(stage, target, directory, tools):
                if stage.kind == "fragment":
                    raise CompileError("fragment validation failed")
                (directory / "vertex.spv").write_bytes(b"partial output")
                return "vertex.spv"

            with patch.object(Toolchain, "preflight", return_value={}):
                with patch("mlir_gpu.compiler._compile_stage", side_effect=compile_stage):
                    with self.assertRaisesRegex(CompileError, "fragment validation"):
                        compile_pipeline(triangle(), Target.VULKAN, output)
            self.assertEqual(list(Path(work).iterdir()), [])

    def test_cli_reports_missing_native_tools_without_success_output(self):
        with tempfile.TemporaryDirectory() as work:
            arguments = [
                "mlir-gpu", str(ROOT / "examples/triangle.py"),
                "--target", "metal", "--output", str(Path(work) / "out"),
            ]
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch("sys.argv", arguments), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with patch.object(Toolchain, "preflight", side_effect=CompileError("Metal converter missing")):
                    self.assertEqual(main(), 1)
            self.assertIn("Metal converter missing", stderr.getvalue())
            self.assertEqual(stdout.getvalue(), "")
            self.assertEqual(list(Path(work).iterdir()), [])

    def test_frontend_protocol_is_a_typed_graph_not_textual_ir(self):
        for stage in arithmetic_pipeline().trace():
            graph = stage.to_dict()
            self.assertEqual(graph["stage"], stage.kind)
            self.assertEqual(graph["name"], stage.name)
            for index, node in enumerate(graph["nodes"]):
                self.assertIsInstance(node["type"], str)
                self.assertTrue(all(type(operand) is int and operand < index for operand in node["args"]))
            text = json.dumps(graph)
            self.assertNotIn("llvm.", text)
            self.assertNotIn("spirv.", text)
            self.assertNotIn("dx.op", text)

    def test_adapter_commands_and_manifests(self):
        for target in Target:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as work:
                calls = []

                def tool(arguments):
                    calls.append(arguments)
                    if arguments[0] == "gfx-compile":
                        source = json.loads(Path(arguments[1]).read_text())
                        for flag in ("--ir-output", "--lowered-output"):
                            Path(arguments[arguments.index(flag) + 1]).write_text("module {}\n")
                        reflected = {
                            "name": source["name"], "stage": source["stage"],
                            "inputs": source["inputs"],
                            "outputs": [{key: value for key, value in io.items() if key != "value"} for io in source["outputs"]],
                        }
                        Path(arguments[arguments.index("--reflection") + 1]).write_text(json.dumps(reflected))
                        output = Path(arguments[arguments.index("--output") + 1])
                        output.write_bytes(b"\x03\x02\x23\x07" + bytes(16) if output.suffix == ".spv" else b"; LLVM IR\n")
                    if "-o" in arguments:
                        output = Path(arguments[arguments.index("-o") + 1])
                        suffix = output.suffix
                        payload = {
                            ".spv": b"\x03\x02\x23\x07" + bytes(16),
                            ".dxil": b"DXBC" + bytes(28),
                            ".metallib": b"MTLB" + bytes(12),
                        }.get(suffix, b"; test IR\n" if suffix == ".ll" else b"module {}\n")
                        output.write_bytes(payload)
                    return ""

                names = ["gfx-compile", "llc", "dxv", "spirv-val", "metal-shaderconverter"]
                with patch.object(Toolchain, "preflight", return_value={name: name for name in names}):
                    with patch("mlir_gpu.compiler._run", side_effect=tool):
                        manifest = compile_pipeline(triangle(), target, Path(work) / "out")
                data = json.loads(manifest.read_text())
                self.assertFalse(data["runtime_verified"])
                self.assertEqual(data["draw"]["vertex_count"], 3)
                self.assertEqual(data["target"], target.value)
                self.assertEqual(data["schema_version"], PIPELINE_SCHEMA_VERSION)
                self.assertEqual(data["raster_convention"], RASTER_CONVENTION)
                self.assertEqual(len(data["entries"]), 2)
                commands = [args[0] for args in calls]
                if target == Target.VULKAN:
                    self.assertEqual(commands.count("spirv-val"), 2)
                else:
                    self.assertEqual(commands.count("llc"), 2)
                    self.assertEqual(commands.count("dxv"), 2)
                    for args in calls:
                        if args[0] == "dxv":
                            self.assertIn("-o", args)
                if target == Target.METAL:
                    self.assertEqual(commands.count("metal-shaderconverter"), 2)
                    for args in calls:
                        if args[0] == "metal-shaderconverter":
                            self.assertTrue(any(arg.startswith("--rename-entry-point=") for arg in args))
                for entry in data["entries"]:
                    self.assertTrue((manifest.parent / entry["binary"]).is_file())
                self.assertFalse(any(path.suffix in {".metal", ".hlsl", ".glsl"} for path in manifest.parent.iterdir()))


@unittest.skipUnless(available("gfx-compile", "spirv-val"), "Native gfx/SPIR-V tools unavailable")
class VulkanIntegrationTests(unittest.TestCase):
    def test_triangle_and_all_supported_operations(self):
        for pipeline in (triangle(), arithmetic_pipeline()):
            with self.subTest(pipeline=pipeline.vertex.function.__name__), tempfile.TemporaryDirectory() as work:
                manifest = compile_pipeline(pipeline, Target.VULKAN, Path(work) / "out", TOOLS)
                data = json.loads(manifest.read_text())
                self.assertEqual(data["backend_status"], "validated-binary")
                for entry in data["entries"]:
                    binary = manifest.parent / entry["binary"]
                    _check_binary(binary, b"\x03\x02\x23\x07", 20)
                    _run([TOOLS.find("spirv-val"), "--target-env", "vulkan1.0", str(binary)])


@unittest.skipUnless(available("gfx-compile", "llc", "dxv"), "Native gfx/LLVM DirectX tools unavailable")
class LLVMIntegrationTests(unittest.TestCase):
    def test_directx_uses_modern_intrinsics_and_upstream_signature_translation(self):
        for pipeline in (triangle(), arithmetic_pipeline()):
            with tempfile.TemporaryDirectory() as work:
                manifest = compile_pipeline(pipeline, Target.DIRECTX, Path(work) / "out", TOOLS)
                for stage in pipeline.trace():
                    with self.subTest(stage=stage.name):
                        text = (manifest.parent / f"{stage.kind}.ll").read_text()
                        self.assertIn("llvm.dx.store.output", text)
                        self.assertIn("!dx.semantic.signatures", text)
                        self.assertNotIn("dx.op.storeOutput", text)
                        self.assertNotIn("!dx.entryPoints", text)
                        self.assertIn("<4 x float>", text)


class NativeTargetIntegrationTests(unittest.TestCase):
    def test_expanded_arithmetic_validates_and_executes_on_all_targets(self):
        for target in Target:
            with self.subTest(target=target.value), tempfile.TemporaryDirectory() as work:
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                manifest = compile_pipeline(expanded_arithmetic_pipeline(), target, Path(work) / "out", TOOLS)
                flag = "MLIR_GPU_TEST_VULKAN" if target == Target.VULKAN else "MLIR_GPU_TEST_METAL"
                if os.environ.get(flag) == "1":
                    from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen

                    program = load_program(manifest)
                    validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
                    image = (
                        render_offscreen(program, 63, 47, validation)
                        if target == Target.VULKAN else
                        render_metal_offscreen(program, 63, 47, validation, TOOLS)
                    )
                    center = (23 * 63 + 31) * 4
                    for actual, expected in zip(image.rgba[center:center + 4], (80, 128, 191, 255)):
                        self.assertAlmostEqual(actual, expected, delta=2)

    def test_unused_input_preserves_location_and_renumbers_live_signature(self):
        for target in Target:
            with self.subTest(target=target.value), tempfile.TemporaryDirectory() as work:
                try:
                    TOOLS.preflight(target)
                except CompileError as error:
                    self.skipTest(str(error))
                manifest = compile_pipeline(unused_varying_pipeline(), target, Path(work) / "out", TOOLS)
                data = json.loads(manifest.read_text())
                self.assertEqual(data["interface"]["fragment"]["inputs"], [
                    {"name": "uv", "type": "f32x2", "location": 1, "interpolation": "perspective"},
                ])
                self.assertEqual(data["draw"]["vertex_count"], 6)
                gpu_flag = "MLIR_GPU_TEST_VULKAN" if target == Target.VULKAN else "MLIR_GPU_TEST_METAL"
                if os.environ.get(gpu_flag) == "1":
                    from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen

                    program = load_program(manifest)
                    validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
                    image = (
                        render_offscreen(program, 63, 47, validation)
                        if target == Target.VULKAN else
                        render_metal_offscreen(program, 63, 47, validation, TOOLS)
                    )
                    center = (23 * 63 + 31) * 4
                    for actual, expected in zip(image.rgba[center:center + 4], (128, 128, 64, 255)):
                        self.assertAlmostEqual(actual, expected, delta=2)

    def compile_target(self, target):
        try:
            TOOLS.preflight(target)
        except CompileError as error:
            self.skipTest(str(error))
        for pipeline in (triangle(), arithmetic_pipeline()):
            with self.subTest(program=pipeline.vertex.function.__name__), tempfile.TemporaryDirectory() as work:
                manifest = compile_pipeline(pipeline, target, Path(work) / "out", TOOLS)
                data = json.loads(manifest.read_text())
                self.assertEqual(data["target"], target.value)
                self.assertFalse(data["runtime_verified"])
                self.assertEqual(len(data["entries"]), 2)
                for stage in ("vertex", "fragment"):
                    dxil = manifest.parent / f"{stage}.dxil"
                    self.assertNotEqual(dxil.read_bytes()[4:20], bytes(16))
                    _run([TOOLS.find("dxv"), str(dxil), "-o", str(manifest.parent / f"{stage}.revalidated.dxil")])

    def test_native_dxil_validation_and_signing(self):
        self.compile_target(Target.DIRECTX)

    def test_native_metallib_conversion(self):
        self.compile_target(Target.METAL)

    @unittest.skipUnless(available("dxv", "dxil-assemble"), "Native DXC frontends unavailable")
    def test_native_validator_and_assembler_reject_bad_inputs(self):
        for tool, filename, content in (
            ("dxv", "bad.dxil", b"DXBC" + bytes(28)),
            ("dxil-assemble", "bad.ll", b"not LLVM IR"),
        ):
            with self.subTest(tool=tool), tempfile.TemporaryDirectory() as work:
                source, output = Path(work) / filename, Path(work) / "out.dxil"
                source.write_bytes(content)
                result = subprocess.run(
                    [TOOLS.find(tool), str(source), "-o", str(output)],
                    capture_output=True, text=True, check=False,
                )
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertTrue(result.stderr.strip())
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
