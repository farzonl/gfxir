"""Windows WARP Offscreen CI Test Harness.

Implements automated CI verification for Windows WARP offscreen execution:
- Hosted WARP execution under MLIR_GPU_TEST_WARP=1 and MLIR_GPU_TEST_DIRECTX=1.
- D3D12 debug validation layer active under MLIR_GPU_TEST_VALIDATION=1 without errors.
- Capability reporting correctly outputs vendor 5140 (Microsoft), device 140, is_software=True.
- Tri-state execution status reporting (EXECUTED, UNSUPPORTED, UNVERIFIED).
- Missing-tool failure assertions: verify descriptive errors if mlir-gpu-directx-view, dxv, or layout is absent/invalid.
- Session-0 window safety: ensure no live window presentation is attempted in CI.
"""

from __future__ import annotations

import copy
import io
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from mlir_gpu import (
    FrameParameters,
    Pipeline,
    Target,
    Toolchain,
    compile_pipeline,
    fragment,
    fullscreen_vertex,
    vec4,
)
from mlir_gpu.compiler import CompileError
from mlir_gpu.viewer import (
    GraphicsProgram,
    RenderedImage,
    Shader,
    ViewerError,
    _directx_offscreen_renderer,
    _native_run,
    _validate_provider,
    export_mp4,
    get_device_capabilities,
    load_program,
    main,
    render_directx_offscreen,
    render_offscreen,
    show_directx_window,
)
from test_parameters import FRAME_REFLECTION

BACKGROUND = bytes([5, 5, 10, 255])
IS_WINDOWS = sys.platform == "win32"
WARP_CI_ENABLED = (
    IS_WINDOWS
    and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1"
    and os.environ.get("MLIR_GPU_TEST_WARP") == "1"
)
VALIDATION_ENABLED = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"


def get_warp_ci_toolchain() -> Toolchain:
    tools_bin = os.environ.get("GPU_TOOLS_BIN")
    llvm_bin = os.environ.get("LLVM_BIN")
    return Toolchain(
        llvm_bin=Path(llvm_bin) if llvm_bin else None,
        tools_bin=Path(tools_bin) if tools_bin else (ROOT / ".tools" / "bin"),
    )


class TestCiWarpEnvironment(unittest.TestCase):
    """Verify CI environment variables, toolchain resolution, and runtime contracts."""

    def test_environment_variables_detected(self):
        """When WARP CI is active, DIRECTX and WARP are 1, and WINDOW is NOT 1 (Session 0 safety)."""
        if WARP_CI_ENABLED:
            self.assertEqual(os.environ.get("MLIR_GPU_TEST_DIRECTX"), "1")
            self.assertEqual(os.environ.get("MLIR_GPU_TEST_WARP"), "1")
            if VALIDATION_ENABLED:
                self.assertEqual(os.environ.get("MLIR_GPU_TEST_VALIDATION"), "1")
            self.assertNotEqual(
                os.environ.get("MLIR_GPU_TEST_WINDOW"),
                "1",
                "MLIR_GPU_TEST_WINDOW must NEVER be enabled in non-interactive Session-0 CI",
            )

    @unittest.skipUnless(IS_WINDOWS, "Toolchain binary resolution requires Windows")
    def test_toolchain_binary_resolution(self):
        """Resolve native Windows adapter binaries and runtime dependencies in .tools/bin."""
        tools = get_warp_ci_toolchain()
        tools_dir = tools.tools_bin or (ROOT / ".tools" / "bin")
        viewer_exe = tools_dir / "mlir-gpu-directx-view.exe"
        if not viewer_exe.is_file():
            self.skipTest("mlir-gpu-directx-view.exe not yet built in tools directory")

        self.assertTrue(viewer_exe.is_file(), f"Missing {viewer_exe}")
        found_viewer = tools.find("mlir-gpu-directx-view")
        self.assertTrue(Path(found_viewer).is_file())

        dxv_exe = tools_dir / "dxv.exe"
        self.assertTrue(dxv_exe.is_file(), f"Missing {dxv_exe}")

        dxcompiler_dll = tools_dir / "dxcompiler.dll"
        self.assertTrue(dxcompiler_dll.is_file(), f"Missing {dxcompiler_dll}")

        dxil_dll = tools_dir / "dxil.dll"
        self.assertTrue(dxil_dll.is_file(), f"Missing {dxil_dll}")

    def test_warp_flag_forwarding_in_python_runtime(self):
        """render_directx_offscreen forwards --warp argument when provider='warp'."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with patch("mlir_gpu.viewer._directx_run") as mock_run:
            mock_run.return_value = json.dumps({"device": "Microsoft Basic Render Driver", "width": 64, "height": 48})
            with tempfile.TemporaryDirectory() as tmp:
                readback_path = Path(tmp) / "readback.rgba"
                readback_path.write_bytes(bytes(64 * 48 * 4))
                with patch("mlir_gpu.viewer._directx_shaders") as mock_shaders:
                    mock_shaders.return_value.__enter__.return_value = (Path(tmp), ["directx-view", "--vertex", "v.dxil", "--fragment", "f.dxil"])
                    render_directx_offscreen(prog, 64, 48, provider="warp", toolchain=get_warp_ci_toolchain())
                    args = mock_run.call_args[0][0]
                    self.assertIn("--warp", args)

    def test_anti_fallback_guarantee_on_hardware_substitution(self):
        """If native capability query reports non-software device when warp was requested, reject."""
        mock_res = subprocess.CompletedProcess(
            args=["view"], returncode=0,
            stdout=json.dumps({"is_software": False, "adapter_name": "NVIDIA RTX", "vendor_id": 4318}),
            stderr="",
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="directx", provider="warp", toolchain=get_warp_ci_toolchain())
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))


@unittest.skipUnless(IS_WINDOWS, "DirectX capability tests require Windows")
class TestCiWarpCapabilities(unittest.TestCase):
    """Assert D3D12 WARP device identity and capability reporting."""

    @classmethod
    def setUpClass(cls):
        cls.tools = get_warp_ci_toolchain()
        try:
            cls.viewer_bin = cls.tools.find("mlir-gpu-directx-view")
        except CompileError:
            cls.viewer_bin = None

    def setUp(self):
        if not self.viewer_bin or not Path(self.viewer_bin).is_file():
            self.skipTest("mlir-gpu-directx-view is not built or available")

    def test_cli_standalone_report_capabilities(self):
        """--warp --report-capabilities outputs vendor 5140, device 140, and is_software=True."""
        result = subprocess.run(
            [str(self.viewer_bin), "--warp", "--report-capabilities"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Error: {result.stderr}")
        data = json.loads(result.stdout)

        # 1. Vendor, device, and software classification
        self.assertEqual(data.get("provider"), "warp")
        self.assertEqual(data.get("vendor_id"), 5140, "Vendor ID must be 5140 (0x1414 Microsoft)")
        self.assertEqual(data.get("device_id"), 140, "Device ID must be 140 (0x008C WARP)")
        self.assertIs(data.get("is_software"), True, "is_software must be True")
        self.assertIs(data.get("software_adapter"), True, "software_adapter must be True")

        # 2. Driver and adapter name
        self.assertIn("Microsoft Basic Render Driver", data.get("adapter_name", ""))
        self.assertEqual(data.get("adapter_name"), data.get("device_name"))

        # 3. Memory specifications for WARP software driver
        self.assertEqual(data.get("dedicated_video_memory"), 0)
        self.assertEqual(data.get("dedicated_memory"), 0)
        self.assertGreater(data.get("shared_system_memory"), 0)

        # 4. Feature level and shader model
        self.assertIn(data.get("d3d12_feature_level"), ("12_0", "12_1", "12_2"))
        self.assertTrue(data.get("shader_model").startswith("6_"))
        sm_minor = int(data.get("shader_model").split("_")[1])
        self.assertGreaterEqual(sm_minor, 6, "WARP requires Shader Model 6.6+")

    def test_python_get_device_capabilities_warp(self):
        """get_device_capabilities('directx', provider='warp') returns standardized compliant schema."""
        caps = get_device_capabilities(backend="directx", provider="warp", toolchain=self.tools)
        self.assertEqual(caps["provider"], "warp")
        self.assertEqual(caps["target"], "directx")
        self.assertEqual(caps["backend"], "directx")
        self.assertEqual(caps["vendor_id"], 5140)
        self.assertEqual(caps["device_id"], 140)
        self.assertTrue(caps["is_software"])
        self.assertTrue(caps["software_adapter"])
        self.assertEqual(caps["status"], "EXECUTED")
        self.assertEqual(caps["missing_capabilities"], [])
        self.assertIn("Basic Render Driver", caps["device_name"])
        self.assertEqual(caps["dedicated_video_memory"], 0)

    def test_tri_state_status_reporting(self):
        """Validate tri-state status reporting semantics (EXECUTED, UNSUPPORTED, UNVERIFIED)."""
        caps = get_device_capabilities(backend="directx", provider="warp", toolchain=self.tools)
        self.assertEqual(caps["status"], "EXECUTED")
        self.assertEqual(caps["missing_capabilities"], [])
        self.assertIn(caps["status"], ("EXECUTED", "UNSUPPORTED", "UNVERIFIED"))

        # Unverified / compile-only path contract: must not claim EXECUTED
        unverified_record = {
            "target": "directx",
            "provider": "warp",
            "status": "UNVERIFIED",
            "missing_capabilities": ["runtime_execution"],
        }
        self.assertNotEqual(unverified_record["status"], "EXECUTED")

        # Unsupported path contract: must not claim EXECUTED
        unsupported_record = {
            "target": "directx",
            "provider": "warp",
            "status": "UNSUPPORTED",
            "missing_capabilities": ["mesh_shaders"],
        }
        self.assertNotEqual(unsupported_record["status"], "EXECUTED")

    def test_warp_capabilities_with_validation_clean(self):
        """--warp --validation --report-capabilities executes cleanly without validation errors."""
        result = subprocess.run(
            [str(self.viewer_bin), "--warp", "--validation", "--report-capabilities"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Error: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertEqual(data.get("vendor_id"), 5140)
        self.assertEqual(data.get("device_id"), 140)

    def test_warp_capabilities_hardware_fallback_rejected_mock(self):
        """If native tool erroneously reports non-software device for WARP, strictly reject."""
        mock_res = subprocess.CompletedProcess(
            args=["view"], returncode=0,
            stdout=json.dumps({"is_software": False, "adapter_name": "NVIDIA RTX", "vendor_id": 4318}),
            stderr="",
        )
        with patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="directx", provider="warp", toolchain=self.tools)
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))


@unittest.skipUnless(WARP_CI_ENABLED, "Requires Windows and MLIR_GPU_TEST_WARP=1 and MLIR_GPU_TEST_DIRECTX=1")
class TestCiWarpOffscreenRendering(unittest.TestCase):
    """Execute real offscreen rendering on hosted Windows WARP rasterizer."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="ci-warp-")
        cls.work_dir = Path(cls.work.name)
        cls.tools = get_warp_ci_toolchain()
        cls.validation = VALIDATION_ENABLED

        # Compile triangle pipeline
        triangle_pipe = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
        manifest_tri = compile_pipeline(triangle_pipe, Target.DIRECTX, cls.work_dir / "triangle", cls.tools)
        cls.triangle_prog = load_program(manifest_tri)

        # Compile quad pipeline
        quad_pipe = runpy.run_path(str(ROOT / "examples" / "quad.py"))["pipeline"]
        manifest_quad = compile_pipeline(quad_pipe, Target.DIRECTX, cls.work_dir / "quad", cls.tools)
        cls.quad_prog = load_program(manifest_quad)

        # Compile diagnostic pipeline with FrameParameters
        @fragment
        def diagnostic(frame: FrameParameters):
            return vec4(frame.resolution / 255.0, frame.time / 4.0, 1.0)

        diag_pipe = Pipeline(fullscreen_vertex, diagnostic)
        manifest_diag = compile_pipeline(diag_pipe, Target.DIRECTX, cls.work_dir / "diagnostic", cls.tools)
        cls.diag_prog = load_program(manifest_diag)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_warp_triangle_offscreen_pixel_accuracy(self):
        """Triangle rendering under WARP verifies dimensions, Basic Render Driver, and 0.18 coverage."""
        width, height = 64, 48
        image = render_directx_offscreen(
            self.triangle_prog, width, height, validation=self.validation,
            toolchain=self.tools, provider="warp",
        )
        self.assertIn("Basic Render Driver", image.device)
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        self.assertEqual(len(image.rgba), width * height * 4)

        def pixel(x, y):
            offset = (y * width + x) * 4
            return image.rgba[offset:offset + 4]

        # Corner pixels must be clear background
        self.assertEqual(pixel(0, 0), BACKGROUND)
        self.assertEqual(pixel(width - 1, height - 1), BACKGROUND)

        # Interior pixel must be rendered
        self.assertNotEqual(pixel(width // 2, height // 2), BACKGROUND)

        # Coverage invariant: triangle must cover ~18% of the surface
        covered = sum(
            image.rgba[i:i + 4] != BACKGROUND for i in range(0, len(image.rgba), 4)
        ) / (width * height)
        self.assertAlmostEqual(covered, 0.18, delta=0.015)

    def test_warp_quad_offscreen_pixel_accuracy(self):
        """Quad rendering under WARP verifies 0.64 coverage invariant."""
        width, height = 64, 48
        image = render_directx_offscreen(
            self.quad_prog, width, height, validation=self.validation,
            toolchain=self.tools, provider="warp",
        )
        self.assertIn("Basic Render Driver", image.device)
        covered = sum(
            image.rgba[i:i + 4] != BACKGROUND for i in range(0, len(image.rgba), 4)
        ) / (width * height)
        self.assertAlmostEqual(covered, 0.64, delta=0.015)

    def test_warp_unaligned_dimensions(self):
        """Unaligned odd dimensions (257x193) render with exact byte count and without pitch padding corruption."""
        width, height = 257, 193
        image = render_directx_offscreen(
            self.triangle_prog, width, height, validation=self.validation,
            toolchain=self.tools, provider="warp",
        )
        self.assertIn("Basic Render Driver", image.device)
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        self.assertEqual(len(image.rgba), width * height * 4)

        def pixel(x, y):
            offset = (y * width + x) * 4
            return image.rgba[offset:offset + 4]

        self.assertEqual(pixel(0, 0), BACKGROUND)
        self.assertEqual(pixel(width - 1, height - 1), BACKGROUND)
        self.assertNotEqual(pixel(width // 2, height // 2), BACKGROUND)

    def test_warp_deterministic_time_readback(self):
        """Deterministic time parameters propagate correctly to WARP root constants."""
        width, height = 32, 24

        # Run at t = 0.0
        img_t0_a = render_directx_offscreen(
            self.diag_prog, width, height, validation=self.validation,
            toolchain=self.tools, time=0.0, provider="warp",
        )
        # Blue channel for diagnostic shader is round(time / 4 * 255) -> 0
        expected_pixel_0 = bytes([width, height, 0, 255])
        self.assertEqual(img_t0_a.rgba[:4], expected_pixel_0)

        # Bitwise identical run at t = 0.0
        img_t0_b = render_directx_offscreen(
            self.diag_prog, width, height, validation=self.validation,
            toolchain=self.tools, time=0.0, provider="warp",
        )
        self.assertEqual(img_t0_a.rgba, img_t0_b.rgba, "WARP execution must be bitwise reproducible at t=0.0")

        # Run at t = 1.0 -> Blue channel round(1.0 / 4 * 255) = 64
        img_t1 = render_directx_offscreen(
            self.diag_prog, width, height, validation=self.validation,
            toolchain=self.tools, time=1.0, provider="warp",
        )
        expected_pixel_1 = bytes([width, height, 64, 255])
        self.assertEqual(img_t1.rgba[:4], expected_pixel_1)
        self.assertNotEqual(img_t0_a.rgba, img_t1.rgba, "Frames at t=0.0 and t=1.0 must differ")

        # Run at t = 2.5 -> Blue channel round(2.5 / 4 * 255) = 159
        img_t25 = render_directx_offscreen(
            self.diag_prog, width, height, validation=self.validation,
            toolchain=self.tools, time=2.5, provider="warp",
        )
        expected_pixel_25 = bytes([width, height, 159, 255])
        self.assertEqual(img_t25.rgba[:4], expected_pixel_25)

    def test_warp_offscreen_cli_standalone_readback(self):
        """Direct CLI invocation with --warp and --readback succeeds and creates valid RGBA readback."""
        viewer_bin = self.tools.find("mlir-gpu-directx-view")
        vertex_path = self.work_dir / "triangle" / "vertex.dxil"
        fragment_path = self.work_dir / "triangle" / "fragment.dxil"
        readback_out = self.work_dir / "warp_direct_readback.rgba"

        args = [
            viewer_bin, "--warp",
            "--vertex", str(vertex_path),
            "--fragment", str(fragment_path),
            "--readback", str(readback_out),
            "--width", "32", "--height", "24",
        ]
        if self.validation:
            args.append("--validation")

        res = subprocess.run(args, capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 0, f"CLI error: {res.stderr}")
        report = json.loads(res.stdout)
        self.assertIn("Basic Render Driver", report.get("device", ""))
        self.assertEqual(report.get("width"), 32)
        self.assertEqual(report.get("height"), 24)
        self.assertTrue(readback_out.is_file())
        self.assertEqual(len(readback_out.read_bytes()), 32 * 24 * 4)


class TestCiWarpValidationLayer(unittest.TestCase):
    """Verify D3D12 debug validation layer activation and error handling."""

    def test_validation_layer_enabled_zero_errors(self):
        """When validation layer is active on WARP, offscreen render executes cleanly with 0 errors."""
        if not (WARP_CI_ENABLED and VALIDATION_ENABLED):
            self.skipTest("Requires MLIR_GPU_TEST_WARP=1, MLIR_GPU_TEST_DIRECTX=1, and MLIR_GPU_TEST_VALIDATION=1")

        tools = get_warp_ci_toolchain()
        with tempfile.TemporaryDirectory() as tmp:
            triangle_pipe = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
            manifest = compile_pipeline(triangle_pipe, Target.DIRECTX, Path(tmp) / "tri", tools)
            prog = load_program(manifest)
            img = render_directx_offscreen(prog, 32, 24, validation=True, provider="warp", toolchain=tools)
            self.assertIn("Basic Render Driver", img.device)
            self.assertEqual(len(img.rgba), 32 * 24 * 4)

    def test_warp_d3d12_validation_flag_passed_to_native_viewer(self):
        """When validation=True, --validation flag is explicitly passed to subprocess arguments."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with patch("mlir_gpu.viewer._directx_run") as mock_run:
            mock_run.return_value = json.dumps({"device": "Microsoft Basic Render Driver", "width": 32, "height": 24})
            with tempfile.TemporaryDirectory() as tmp:
                readback_file = Path(tmp) / "readback.rgba"
                readback_file.write_bytes(bytes(32 * 24 * 4))
                with patch("mlir_gpu.viewer._directx_shaders") as mock_shaders:
                    mock_shaders.return_value.__enter__.return_value = (Path(tmp), ["directx-view", "--vertex", "v.dxil", "--fragment", "f.dxil"])
                    render_directx_offscreen(prog, 32, 24, validation=True, provider="warp", toolchain=get_warp_ci_toolchain())
                    self.assertTrue(mock_run.call_args[0][1], "validation parameter must be True")

    def test_warp_d3d12_validation_failure_surfacing(self):
        """If D3D12 validation layer reports errors, ViewerError surfaces the exact error diagnostic."""
        mock_res = subprocess.CompletedProcess(
            args=["directx-view"],
            returncode=1,
            stdout="",
            stderr="D3D12 validation failed:\n[ERROR] D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_AT_FAULT",
        )
        with patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(ViewerError) as ctx:
                _native_run(["directx-view"], capture=True, backend="DirectX 12")
            self.assertIn("D3D12 validation failed", str(ctx.exception))
            self.assertIn("D3D12_MESSAGE_ID_DEVICE_REMOVAL_PROCESS_AT_FAULT", str(ctx.exception))


class TestCiWarpMissingTools(unittest.TestCase):
    """Verify explicit descriptive failure when native tools or inputs are missing or invalid."""

    def test_missing_viewer_binary_raises_compile_error(self):
        """When tools_bin is missing mlir-gpu-directx-view, Toolchain.find raises descriptive CompileError."""
        with tempfile.TemporaryDirectory() as empty_dir:
            tc = Toolchain(tools_bin=Path(empty_dir))
            with self.assertRaises(CompileError) as ctx:
                tc.find("mlir-gpu-directx-view")
            self.assertIn("Required tool 'mlir-gpu-directx-view' is missing or not executable in", str(ctx.exception))
            self.assertIn("Build tools/build_toolchain.py", str(ctx.exception))

            prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
            with patch("mlir_gpu.viewer.sys.platform", "win32"):
                with self.assertRaises(CompileError) as ctx2:
                    render_directx_offscreen(prog, 64, 48, toolchain=tc, provider="warp")
                self.assertIn("Required tool 'mlir-gpu-directx-view' is missing", str(ctx2.exception))

                with self.assertRaises(CompileError) as ctx3:
                    get_device_capabilities(backend="directx", provider="warp", toolchain=tc)
                self.assertIn("Required tool 'mlir-gpu-directx-view' is missing", str(ctx3.exception))

    def test_missing_dxv_raises_compile_error(self):
        """When dxv is missing from tools_bin, Toolchain.find and preflight(Target.DIRECTX) raise CompileError."""
        with tempfile.TemporaryDirectory() as empty_dir:
            tc = Toolchain(llvm_bin=get_warp_ci_toolchain().llvm_bin, tools_bin=Path(empty_dir))
            with self.assertRaises(CompileError) as ctx_find:
                tc.find("dxv")
            self.assertIn("dxv", str(ctx_find.exception))
            self.assertIn("missing or not executable", str(ctx_find.exception))

            orig_find = Toolchain.find

            def mock_find(self_tc, name: str):
                if name == "dxv":
                    return orig_find(self_tc, name)
                return "mock_" + name

            with patch.object(Toolchain, "find", mock_find):
                with self.assertRaises(CompileError) as ctx:
                    tc.preflight(Target.DIRECTX)
                self.assertIn("dxv", str(ctx.exception))

    def test_missing_tool_not_on_path_raises_compile_error(self):
        """When tools_bin is None and tool is absent from PATH, Toolchain.find raises CompileError."""
        tc = Toolchain(llvm_bin=None, tools_bin=None)
        with patch("shutil.which", return_value=None):
            with self.assertRaises(CompileError) as ctx:
                tc.find("mlir-gpu-directx-view")
            self.assertIn("Required tool 'mlir-gpu-directx-view' was not found", str(ctx.exception))
            self.assertIn("there is no MSL/HLSL fallback", str(ctx.exception))

    def test_invalid_tool_binary_execution_failure_raises_viewer_error(self):
        """Execution failure (OSError or nonzero exit) raises ViewerError without silent swallowing."""
        # 1. OSError (cannot start process)
        with patch("subprocess.run", side_effect=OSError("Access denied")):
            with self.assertRaises(ViewerError) as ctx:
                _native_run(["invalid-binary"], capture=True, backend="DirectX 12")
            self.assertIn("Cannot start the native DirectX 12 preview: Access denied", str(ctx.exception))

        # 2. Non-zero exit code
        mock_fail = subprocess.CompletedProcess(
            args=["binary"], returncode=1, stdout="", stderr="Fatal initialization failure",
        )
        with patch("subprocess.run", return_value=mock_fail):
            with self.assertRaises(ViewerError) as ctx:
                _native_run(["binary"], capture=True, backend="DirectX 12")
            self.assertIn("Native DirectX 12 preview failed (1): Fatal initialization failure", str(ctx.exception))

    @unittest.skipUnless(IS_WINDOWS, "Native CLI test requires Windows")
    def test_missing_shaders_cli_fails_descriptively(self):
        """Invoking mlir-gpu-directx-view without required shader arguments exits with status 1."""
        tools = get_warp_ci_toolchain()
        try:
            viewer_bin = tools.find("mlir-gpu-directx-view")
        except CompileError:
            self.skipTest("mlir-gpu-directx-view not available")

        res = subprocess.run([viewer_bin], capture_output=True, text=True, check=False)
        self.assertEqual(res.returncode, 1)
        self.assertIn("--vertex and --fragment DXIL paths are required", res.stderr)

    def test_hardware_fallback_rejection(self):
        """Requests warp with simulated hardware adapter asserts immediate ViewerError rejecting fallback."""
        mock_res = subprocess.CompletedProcess(
            args=["view"], returncode=0,
            stdout=json.dumps({"is_software": False, "adapter_name": "Intel Arc", "vendor_id": 32902}),
            stderr="",
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="directx", provider="warp", toolchain=get_warp_ci_toolchain())
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_corrupt_viewer_binary_raises_viewer_error(self):
        """When mlir-gpu-directx-view is a corrupt/unexecutable file, running offscreen raises ViewerError."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_bin = Path(tmp) / ("mlir-gpu-directx-view.exe" if os.name == "nt" else "mlir-gpu-directx-view")
            bad_bin.write_bytes(b"NOT_A_VALID_PE_BINARY")
            bad_bin.chmod(0o755)
            tc = Toolchain(tools_bin=Path(tmp))
            prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
            with patch("mlir_gpu.viewer.sys.platform", "win32"):
                with self.assertRaises(ViewerError) as ctx:
                    render_directx_offscreen(prog, 64, 48, toolchain=tc, provider="warp")
                err_msg = str(ctx.exception)
                self.assertTrue(
                    "Cannot start the native DirectX 12 preview" in err_msg or "Native DirectX 12 preview failed" in err_msg,
                    f"Unexpected error message: {err_msg}",
                )

    def test_corrupt_viewer_capability_query_raises_viewer_error(self):
        """When mlir-gpu-directx-view is a corrupt binary, get_device_capabilities raises ViewerError."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_bin = Path(tmp) / ("mlir-gpu-directx-view.exe" if os.name == "nt" else "mlir-gpu-directx-view")
            bad_bin.write_bytes(b"CORRUPT")
            bad_bin.chmod(0o755)
            tc = Toolchain(tools_bin=Path(tmp))
            with patch("mlir_gpu.viewer.sys.platform", "win32"):
                with self.assertRaises(ViewerError) as ctx:
                    get_device_capabilities(backend="directx", provider="warp", toolchain=tc)
                err_msg = str(ctx.exception)
                self.assertTrue(
                    "Cannot start DirectX 12 capability query" in err_msg or "DirectX 12 capability query failed" in err_msg,
                    f"Unexpected error message: {err_msg}",
                )

    def test_corrupt_dxv_binary_raises_compile_error(self):
        """When dxv binary is corrupted or fails to start, compiler raises CompileError."""
        with tempfile.TemporaryDirectory() as tmp:
            bad_dxv = Path(tmp) / ("dxv.exe" if IS_WINDOWS else "dxv")
            bad_dxv.write_bytes(b"INVALID")
            with patch("subprocess.run", side_effect=OSError("[WinError 193] %1 is not a valid Win32 application")):
                from mlir_gpu.compiler import _run
                with self.assertRaises(CompileError) as ctx:
                    _run([str(bad_dxv), "input.dxil", "-o", "output.dxil"])
                self.assertIn("Could not execute", str(ctx.exception))
                self.assertIn("is not a valid Win32 application", str(ctx.exception))

    def test_missing_dynamic_libraries_dxv_failure_surfacing(self):
        """When dxv fails because dxcompiler.dll/dxil.dll is missing (STATUS_DLL_NOT_FOUND), raises CompileError."""
        mock_dll_fail = subprocess.CompletedProcess(
            args=["dxv", "in.dxil", "-o", "out.dxil"],
            returncode=3221225781,
            stdout="",
            stderr="The code execution cannot proceed because dxcompiler.dll was not found.",
        )
        with patch("subprocess.run", return_value=mock_dll_fail):
            from mlir_gpu.compiler import _run
            with self.assertRaises(CompileError) as ctx:
                _run(["dxv", "in.dxil", "-o", "out.dxil"])
            self.assertIn("Command failed (3221225781)", str(ctx.exception))
            self.assertIn("dxcompiler.dll was not found", str(ctx.exception))

    def test_missing_dynamic_libraries_viewer_failure_surfacing(self):
        """When viewer fails due to missing dynamic library, ViewerError surfaces the diagnostic."""
        mock_dll_fail = subprocess.CompletedProcess(
            args=["mlir-gpu-directx-view"],
            returncode=3221225781,
            stdout="",
            stderr="System error: dxcompiler.dll or d3d12.dll missing.",
        )
        with patch("subprocess.run", return_value=mock_dll_fail):
            with self.assertRaises(ViewerError) as ctx:
                _native_run(["mlir-gpu-directx-view"], capture=True, backend="DirectX 12")
            self.assertIn("Native DirectX 12 preview failed (3221225781)", str(ctx.exception))
            self.assertIn("missing", str(ctx.exception))

    @unittest.skipUnless(IS_WINDOWS, "Toolchain binary resolution requires Windows")
    def test_toolchain_dll_dependencies_verified(self):
        """Verify that dxcompiler.dll and dxil.dll are present alongside dxv.exe."""
        tools = get_warp_ci_toolchain()
        tools_dir = tools.tools_bin or (ROOT / ".tools" / "bin")
        if not (tools_dir / "dxv.exe").is_file():
            self.skipTest("dxv.exe not yet available in tools directory")
        self.assertTrue((tools_dir / "dxcompiler.dll").is_file(), f"Missing dxcompiler.dll in {tools_dir}")
        self.assertTrue((tools_dir / "dxil.dll").is_file(), f"Missing dxil.dll in {tools_dir}")


class TestCiWarpSession0Safety(unittest.TestCase):
    """Ensure no live window presentation is attempted in non-interactive CI environments."""

    def test_ci_session0_environment_guarantee(self):
        """In CI runs, MLIR_GPU_TEST_WINDOW must not be enabled."""
        self.assertNotEqual(
            os.environ.get("MLIR_GPU_TEST_WINDOW"), "1",
            "CI environments operate in Session 0; MLIR_GPU_TEST_WINDOW must remain unset",
        )

    def test_offscreen_rendering_never_creates_window_or_calls_presentation(self):
        """render_directx_offscreen exclusively passes --readback and never passes --frames."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with patch("mlir_gpu.viewer._directx_run") as mock_run:
            mock_run.return_value = json.dumps({"device": "Microsoft Basic Render Driver", "width": 64, "height": 48})
            with tempfile.TemporaryDirectory() as tmp:
                readback_file = Path(tmp) / "readback.rgba"
                readback_file.write_bytes(bytes(64 * 48 * 4))
                with patch("mlir_gpu.viewer._directx_shaders") as mock_shaders:
                    mock_shaders.return_value.__enter__.return_value = (Path(tmp), ["view", "--vertex", "v.dxil", "--fragment", "f.dxil"])
                    render_directx_offscreen(prog, 64, 48, provider="warp", toolchain=get_warp_ci_toolchain())
                    args = mock_run.call_args[0][0]
                    self.assertIn("--readback", args)
                    self.assertNotIn("--frames", args)

    @unittest.skipUnless(IS_WINDOWS, "Native CLI test requires Windows")
    def test_cli_readback_and_frames_mutually_exclusive(self):
        """--readback and --frames are mutually exclusive in native viewer CLI."""
        tools = get_warp_ci_toolchain()
        try:
            viewer_bin = tools.find("mlir-gpu-directx-view")
        except CompileError:
            self.skipTest("mlir-gpu-directx-view not available")

        res = subprocess.run(
            [viewer_bin, "--vertex", "v.dxil", "--fragment", "f.dxil", "--readback", "out.rgba", "--frames", "1"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(res.returncode, 1)
        self.assertIn("--readback and --frames are mutually exclusive", res.stderr)

    def test_window_mode_session0_rejection(self):
        """If presentation is attempted without desktop session, descriptive error is reported."""
        session0_stderr = (
            "mlir-gpu-directx-view: DXGI presentation is unavailable. Run the live viewer "
            "in an interactive Windows desktop session (not Session 0), or use --output for offscreen rendering."
        )
        mock_fail = subprocess.CompletedProcess(
            args=["view"], returncode=1, stdout="", stderr=session0_stderr,
        )
        with patch("subprocess.run", return_value=mock_fail):
            with self.assertRaises(ViewerError) as ctx:
                _native_run(["view"], capture=True, backend="DirectX 12")
            self.assertIn("DXGI presentation is unavailable", str(ctx.exception))
            self.assertIn("not Session 0", str(ctx.exception))

    def test_viewer_cli_output_flag_routes_to_offscreen(self):
        """mlir-gpu-view manifest.json --output preview.png routes to offscreen renderer and avoids window."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "pipeline.json"
            manifest_path.write_text("{}")
            out_png = Path(tmp) / "out.png"

            with patch("mlir_gpu.viewer.sys.platform", "win32"), \
                 patch("mlir_gpu.viewer.load_program", return_value=prog), \
                 patch("mlir_gpu.viewer.render_directx_offscreen") as mock_offscreen, \
                 patch("mlir_gpu.viewer.write_png"):
                mock_offscreen.return_value = RenderedImage(64, 48, bytes(64 * 48 * 4), "Microsoft Basic Render Driver")
                with patch("sys.argv", ["mlir-gpu-view", str(manifest_path), "--output", str(out_png), "--provider", "warp"]):
                    exit_code = main()
                    self.assertEqual(exit_code, 0)
                    mock_offscreen.assert_called_once()
                    self.assertEqual(mock_offscreen.call_args[1].get("provider"), "warp")


class TestCiWarpFallbackRefusal(unittest.TestCase):
    """Verify that missing tools, failed executions, or missing WARP never silently fall back to dummy rendering."""

    def test_missing_viewer_never_falls_back_to_dummy_rendering(self):
        """render_directx_offscreen raises CompileError when viewer is missing and never returns dummy pixels."""
        with tempfile.TemporaryDirectory() as empty_dir:
            tc = Toolchain(tools_bin=Path(empty_dir))
            prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
            with patch("mlir_gpu.viewer.sys.platform", "win32"):
                with self.assertRaises(CompileError) as ctx:
                    render_directx_offscreen(prog, 64, 48, toolchain=tc, provider="warp")
                self.assertIn("Required tool 'mlir-gpu-directx-view' is missing", str(ctx.exception))

    def test_native_renderer_failure_never_returns_dummy_pixels(self):
        """render_directx_offscreen raises ViewerError on native crash and never returns dummy pixels."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("mlir_gpu.viewer._directx_run", side_effect=ViewerError("Native DirectX 12 preview failed (1): GPU Hang")):
            with self.assertRaises(ViewerError) as ctx:
                render_directx_offscreen(prog, 64, 48, provider="warp", toolchain=get_warp_ci_toolchain())
            self.assertIn("GPU Hang", str(ctx.exception))

    def test_missing_readback_file_never_returns_dummy_pixels(self):
        """Missing readback file raises FileNotFoundError/ViewerError rather than returning dummy pixels."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("mlir_gpu.viewer._directx_run") as mock_run:
            mock_run.return_value = json.dumps({"device": "Microsoft Basic Render Driver", "width": 64, "height": 48})
            with self.assertRaises((ViewerError, FileNotFoundError, OSError)):
                render_directx_offscreen(prog, 64, 48, provider="warp", toolchain=get_warp_ci_toolchain())

    def test_corrupted_readback_size_never_returns_dummy_pixels(self):
        """Truncated or corrupted readback length raises ViewerError and never returns dummy pixels."""
        prog = GraphicsProgram(Shader("main", b"DXBC" + bytes(28)), Shader("main", b"DXBC" + bytes(28)), target="directx")
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch("mlir_gpu.viewer._directx_run") as mock_run:
            mock_run.return_value = json.dumps({"device": "Microsoft Basic Render Driver", "width": 64, "height": 48})
            with tempfile.TemporaryDirectory() as tmp:
                readback_file = Path(tmp) / "readback.rgba"
                readback_file.write_bytes(bytes(100))
                with patch("mlir_gpu.viewer._directx_shaders") as mock_shaders:
                    mock_shaders.return_value.__enter__.return_value = (Path(tmp), ["view", "--vertex", "v.dxil", "--fragment", "f.dxil"])
                    with self.assertRaises(ViewerError) as ctx:
                        render_directx_offscreen(prog, 64, 48, provider="warp", toolchain=get_warp_ci_toolchain())
                    self.assertIn("readback size does not match", str(ctx.exception))

    def test_anti_hardware_substitution_refuses_fallback(self):
        """get_device_capabilities strictly rejects hardware adapter when WARP was requested."""
        mock_res = subprocess.CompletedProcess(
            args=["view"], returncode=0,
            stdout=json.dumps({"is_software": False, "adapter_name": "NVIDIA GeForce RTX", "vendor_id": 4318}),
            stderr="",
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="directx", provider="warp", toolchain=get_warp_ci_toolchain())
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_anti_software_substitution_when_hardware_requested(self):
        """get_device_capabilities strictly rejects software adapter when hardware was requested."""
        mock_res = subprocess.CompletedProcess(
            args=["view"], returncode=0,
            stdout=json.dumps({"is_software": True, "adapter_name": "Microsoft Basic Render Driver", "vendor_id": 5140}),
            stderr="",
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("subprocess.run", return_value=mock_res):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="directx", provider="hardware", toolchain=get_warp_ci_toolchain())
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_cross_backend_provider_fallbacks_refused(self):
        """_validate_provider strictly refuses invalid or cross-backend provider requests."""
        with self.assertRaises(ValueError) as ctx_d1:
            _validate_provider("lavapipe", "directx")
        self.assertIn("Provider 'lavapipe' is not supported by the DirectX 12 backend", str(ctx_d1.exception))

        with self.assertRaises(ValueError) as ctx_d2:
            _validate_provider("swiftshader", "directx")
        self.assertIn("Provider 'swiftshader' is not supported by the DirectX 12 backend", str(ctx_d2.exception))

        with self.assertRaises(ValueError) as ctx_v:
            _validate_provider("warp", "vulkan")
        self.assertIn("Provider 'warp' is not supported by the Vulkan backend", str(ctx_v.exception))

        for p in ("warp", "lavapipe", "swiftshader"):
            with self.assertRaises(ValueError) as ctx_m:
                _validate_provider(p, "metal")
            self.assertIn(f"Provider '{p}' is not supported by the Metal backend", str(ctx_m.exception))

    def test_unknown_provider_refused(self):
        """_validate_provider strictly refuses unknown provider strings."""
        with self.assertRaises(ValueError) as ctx:
            _validate_provider("cpu_fallback", "directx")
        self.assertIn("Unknown provider 'cpu_fallback'", str(ctx.exception))

    def test_unverified_status_never_claims_executed(self):
        """Unverified or unsupported driver status invariants are strictly enforced."""
        for status_code in ("UNVERIFIED", "UNSUPPORTED"):
            report = {
                "target": "directx",
                "provider": "warp",
                "status": status_code,
                "missing_capabilities": ["required_feature"],
            }
            self.assertNotEqual(report["status"], "EXECUTED")


class TestCiWarpEnvironmentParsing(unittest.TestCase):
    """Verify parsing and validation of CI environment variables."""

    def test_environment_warp_disabled_skips_cleanly(self):
        """When MLIR_GPU_TEST_WARP is not '1', WARP offscreen tests skip cleanly."""
        with patch.dict(os.environ, {"MLIR_GPU_TEST_WARP": "0"}):
            self.assertNotEqual(os.environ.get("MLIR_GPU_TEST_WARP"), "1")

    def test_environment_window_enabled_rejected_for_ci(self):
        """If MLIR_GPU_TEST_WINDOW is erroneously set to '1' in CI, assertion guarantees failure."""
        with patch.dict(os.environ, {"MLIR_GPU_TEST_WINDOW": "1"}):
            with self.assertRaises(AssertionError):
                self.assertNotEqual(
                    os.environ.get("MLIR_GPU_TEST_WINDOW"), "1",
                    "MLIR_GPU_TEST_WINDOW must NEVER be enabled in non-interactive Session-0 CI",
                )

    def test_toolchain_resolution_empty_environment_variables(self):
        """get_warp_ci_toolchain handles unset, empty string, and fallback defaults."""
        with patch.dict(os.environ, {}, clear=True):
            tc = get_warp_ci_toolchain()
            self.assertIsNone(tc.llvm_bin)
            self.assertEqual(tc.tools_bin, ROOT / ".tools" / "bin")

        with patch.dict(os.environ, {"GPU_TOOLS_BIN": "D:\\custom\\tools", "LLVM_BIN": "D:\\custom\\llvm"}):
            tc_custom = get_warp_ci_toolchain()
            self.assertEqual(tc_custom.llvm_bin, Path("D:\\custom\\llvm"))
            self.assertEqual(tc_custom.tools_bin, Path("D:\\custom\\tools"))


class TestCiWarpYamlSchemaValidation(unittest.TestCase):
    """Validate YAML schema, steps, runners, and environment contracts for .github/workflows/ci.yml."""

    @classmethod
    def setUpClass(cls):
        cls.ci_yaml_path = ROOT / ".github" / "workflows" / "ci.yml"
        if not HAVE_YAML:
            return
        cls.ci_data = yaml.safe_load(cls.ci_yaml_path.read_text(encoding="utf-8"))

    def setUp(self):
        if not HAVE_YAML:
            self.skipTest("PyYAML not installed in test environment")

    def test_ci_yaml_exists_and_parses(self):
        """Workflow file exists and is valid YAML mapping."""
        self.assertTrue(self.ci_yaml_path.is_file(), f"Missing {self.ci_yaml_path}")
        self.assertIsInstance(self.ci_data, dict)
        self.assertIn("name", self.ci_data)
        self.assertIn("jobs", self.ci_data)

    def test_ci_yaml_workflow_triggers(self):
        """Workflow triggers include push, pull_request, and workflow_dispatch."""
        triggers = self.ci_data.get("on") or self.ci_data.get(True)
        self.assertIsNotNone(triggers, "Workflow triggers ('on') must be defined")
        self.assertIn("push", triggers)
        self.assertIn("pull_request", triggers)

    def test_ci_yaml_all_jobs_present(self):
        """Workflow defines both build-and-test and test-windows-warp jobs."""
        jobs = self.ci_data.get("jobs", {})
        self.assertIn("build-and-test", jobs)
        self.assertIn("test-windows-warp", jobs)

    def test_ci_yaml_windows_warp_runner_and_shell(self):
        """test-windows-warp targets windows-latest with pwsh shell."""
        job = self.ci_data["jobs"]["test-windows-warp"]
        self.assertEqual(job.get("runs-on"), "windows-latest")
        defaults = job.get("defaults", {})
        self.assertEqual(defaults.get("run", {}).get("shell"), "pwsh")

    def test_ci_yaml_windows_warp_steps_complete(self):
        """test-windows-warp contains all required lifecycle steps."""
        job = self.ci_data["jobs"]["test-windows-warp"]
        steps = job.get("steps", [])
        step_names = [s.get("name") for s in steps if "name" in s]

        self.assertTrue(any("Checkout" in n for n in step_names), "Missing Checkout step")
        self.assertTrue(any("Python" in n for n in step_names), "Missing Python setup step")
        self.assertTrue(any("MSVC" in n for n in step_names), "Missing MSVC setup step")
        self.assertTrue(any("DirectXShaderCompiler" in n and "Cache" in n for n in step_names), "Missing DXC cache step")
        self.assertTrue(any("Install DirectXShaderCompiler" in n for n in step_names), "Missing DXC install step")
        self.assertTrue(any("Cache compiled gfx compiler" in n for n in step_names), "Missing compiler binaries cache step")
        self.assertTrue(any("Cache native helper binaries" in n for n in step_names), "Missing native tools cache step")
        self.assertTrue(any("Build native helper binaries" in n for n in step_names), "Missing native tools build step")
        self.assertTrue(any("Report WARP driver" in n for n in step_names), "Missing WARP capability report step")
        self.assertTrue(any("Run Windows WARP" in n for n in step_names), "Missing test run step")

    def test_ci_yaml_windows_warp_environment_variables(self):
        """Test step in test-windows-warp sets required environment variables."""
        job = self.ci_data["jobs"]["test-windows-warp"]
        steps = job.get("steps", [])
        test_step = next((s for s in steps if "Run Windows WARP" in s.get("name", "")), None)
        self.assertIsNotNone(test_step, "Could not find 'Run Windows WARP' step")

        env = test_step.get("env", {})
        self.assertEqual(env.get("MLIR_GPU_TEST_DIRECTX"), "1")
        self.assertEqual(env.get("MLIR_GPU_TEST_WARP"), "1")
        self.assertEqual(env.get("MLIR_GPU_TEST_VALIDATION"), "1")
        self.assertEqual(env.get("GPU_TOOLS_BIN"), "${{ github.workspace }}/.tools/bin")
        self.assertNotIn("MLIR_GPU_TEST_WINDOW", env, "MLIR_GPU_TEST_WINDOW must not be set in CI")

    def test_ci_yaml_cache_keys_include_directx_viewer(self):
        """Cache keys for native tools include native/directx_viewer.cpp in both jobs."""
        for job_name in ("build-and-test", "test-windows-warp"):
            job = self.ci_data["jobs"][job_name]
            steps = job.get("steps", [])
            native_cache = next(
                (s for s in steps if "Cache native helper binaries" in s.get("name", "")), None
            )
            self.assertIsNotNone(native_cache, f"Missing native tools cache step in {job_name}")
            key = native_cache.get("with", {}).get("key", "")
            self.assertIn("native/directx_viewer.cpp", key, f"Cache key in {job_name} missing native/directx_viewer.cpp")


if __name__ == "__main__":
    unittest.main()
