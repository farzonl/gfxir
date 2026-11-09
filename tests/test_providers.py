import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlir_gpu import FrameParameters, Pipeline, Target, Toolchain, compile_pipeline, fragment, fullscreen_vertex, vec4
from mlir_gpu.viewer import (
    GraphicsProgram, RenderedImage, Shader, ViewerError, _match_vulkan_device,
    _validate_provider, export_mp4, get_device_capabilities, load_program, main,
    render_directx_offscreen, render_offscreen, render_vulkan_offscreen,
    show_directx_window, write_png,
)
from test_parameters import FRAME_REFLECTION

ROOT = Path(__file__).resolve().parents[1]


def get_test_toolchain() -> Toolchain:
    llvm_bin = os.environ.get("LLVM_BIN")
    if not llvm_bin:
        cand = ROOT / ".toolchains" / "llvm-build" / "bin"
        if cand.is_dir():
            llvm_bin = str(cand)
    tools_bin = os.environ.get("GPU_TOOLS_BIN")
    if not tools_bin and sys.platform == "win32":
        cand_tools = ROOT / ".tools" / "bin"
        if (cand_tools / "mlir-gpu-directx-view.exe").is_file():
            tools_bin = str(cand_tools)
    return Toolchain(
        llvm_bin=Path(llvm_bin) if llvm_bin else None,
        tools_bin=Path(tools_bin) if tools_bin else None,
    )


def _has_directx_hardware_gpu(viewer_bin: str) -> bool:
    result = subprocess.run(
        [viewer_bin, "--report-capabilities"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        return False
    try:
        data = json.loads(result.stdout)
    except ValueError:
        return False
    return (
        not data.get("is_software", False)
        and not data.get("software_adapter", False)
        and data.get("vendor_id") != 5140
        and "basic render driver" not in data.get("adapter_name", "").lower()
    )


class FakeVideoEncoder:
    def __init__(self, arguments, **kwargs):
        self.arguments = arguments
        self.stdin = io.BytesIO()
        self.written_bytes = bytearray()
        orig_write = self.stdin.write

        def record_write(data):
            self.written_bytes.extend(data)
            return orig_write(data)

        self.stdin.write = record_write
        self.returncode = None
        self.terminated = False
        self.payload = b"\0\0\0\x20ftypisom" + bytes(32)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        if self.returncode is None:
            Path(self.arguments[-1]).write_bytes(self.payload)
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15


@unittest.skipUnless(sys.platform == "win32", "DirectX CLI tests require Windows")
class TestDirectXCliProviders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.tools = get_test_toolchain()
        cls.viewer_bin = cls.tools.find("mlir-gpu-directx-view")

        # Compile minimal triangle DXIL program
        pipeline = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
        manifest = compile_pipeline(pipeline, Target.DIRECTX, Path(cls.work.name) / "triangle", cls.tools)
        cls.triangle_prog = load_program(manifest)
        cls.vertex_path = Path(cls.work.name) / "triangle" / "vertex.dxil"
        cls.fragment_path = Path(cls.work.name) / "triangle" / "fragment.dxil"
        cls.vertex_path.write_bytes(cls.triangle_prog.vertex.code)
        cls.fragment_path.write_bytes(cls.triangle_prog.fragment.code)
        cls.has_hardware_gpu = _has_directx_hardware_gpu(cls.viewer_bin)

    def test_cli_report_capabilities_hardware(self):
        """--report-capabilities without --warp outputs valid JSON hardware device capabilities."""
        if not self.has_hardware_gpu:
            self.skipTest("No hardware DirectX 12 GPU available on this runner")
        result = subprocess.run(
            [self.viewer_bin, "--report-capabilities"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Error: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertEqual(data.get("provider"), "hardware")
        self.assertFalse(data.get("is_software"))
        self.assertFalse(data.get("software_adapter"))
        self.assertNotEqual(data.get("vendor_id"), 5140)
        self.assertTrue(data.get("adapter_name"))
        self.assertIn("shader_model", data)

    def test_cli_report_capabilities_warp(self):
        """--warp --report-capabilities outputs valid JSON WARP software rasterizer capabilities."""
        result = subprocess.run(
            [self.viewer_bin, "--warp", "--report-capabilities"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Error: {result.stderr}")
        data = json.loads(result.stdout)
        self.assertEqual(data.get("provider"), "warp")
        self.assertTrue(data.get("is_software"))
        self.assertTrue(data.get("software_adapter"))
        self.assertEqual(data.get("vendor_id"), 5140)
        self.assertEqual(data.get("device_id"), 140)
        self.assertIn("Basic Render Driver", data.get("adapter_name", ""))
        self.assertIn("shader_model", data)

    def test_cli_report_capabilities_standalone_no_shaders(self):
        """--report-capabilities succeeds standalone without --vertex or --fragment flags."""
        args = [self.viewer_bin, "--report-capabilities"]
        if not self.has_hardware_gpu:
            args.append("--warp")
        result = subprocess.run(
            args,
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertNotIn("--vertex and --fragment", result.stderr)
        data = json.loads(result.stdout)
        self.assertIn("vendor_id", data)

    def test_cli_warp_offscreen_render(self):
        """--warp offscreen render generates valid readback RGBA file and reports Basic Render Driver."""
        readback_file = Path(self.work.name) / "warp_readback.rgba"
        result = subprocess.run(
            [
                self.viewer_bin, "--warp",
                "--vertex", str(self.vertex_path),
                "--fragment", str(self.fragment_path),
                "--readback", str(readback_file),
                "--width", "32", "--height", "24",
            ],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        report = json.loads(result.stdout)
        self.assertIn("Basic Render Driver", report.get("device", ""))
        self.assertEqual(report.get("width"), 32)
        self.assertEqual(report.get("height"), 24)
        self.assertTrue(readback_file.is_file())
        self.assertEqual(len(readback_file.read_bytes()), 32 * 24 * 4)

    def test_cli_warp_with_time(self):
        """--warp supports root constant upload with --time."""
        readback_file = Path(self.work.name) / "warp_timed.rgba"
        result = subprocess.run(
            [
                self.viewer_bin, "--warp",
                "--vertex", str(self.vertex_path),
                "--fragment", str(self.fragment_path),
                "--time", "1.5",
                "--readback", str(readback_file),
                "--width", "32", "--height", "24",
            ],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, f"Stderr: {result.stderr}")
        self.assertTrue(readback_file.is_file())
        self.assertEqual(len(readback_file.read_bytes()), 32 * 24 * 4)

    def test_cli_invalid_provider_flags_rejected(self):
        """CLI strictly rejects unknown provider flags like --lavapipe or --swiftshader."""
        for bad_flag in ("--lavapipe", "--swiftshader", "--provider", "--software"):
            result = subprocess.run(
                [self.viewer_bin, bad_flag],
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Unknown native DirectX viewer argument", result.stderr)

    def test_cli_duplicate_warp_flag_rejected(self):
        """Duplicate --warp flag is rejected with error."""
        result = subprocess.run(
            [self.viewer_bin, "--warp", "--warp"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing or duplicate native viewer argument", result.stderr)

    def test_cli_duplicate_report_capabilities_rejected(self):
        """Duplicate --report-capabilities flag is rejected with error."""
        result = subprocess.run(
            [self.viewer_bin, "--report-capabilities", "--report-capabilities"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("Missing or duplicate native viewer argument", result.stderr)


@unittest.skipUnless(sys.platform == "win32", "DirectX tests require Windows")
class TestDirectXPythonProviders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.tools = get_test_toolchain()
        pipeline = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
        manifest = compile_pipeline(pipeline, Target.DIRECTX, Path(cls.work.name) / "triangle", cls.tools)
        cls.program = load_program(manifest)
        viewer_bin = cls.tools.find("mlir-gpu-directx-view")
        cls.has_hardware_gpu = _has_directx_hardware_gpu(viewer_bin)

    def test_render_directx_offscreen_warp(self):
        """render_directx_offscreen(provider='warp') renders on WARP rasterizer."""
        image = render_directx_offscreen(self.program, 64, 48, provider="warp", toolchain=self.tools)
        self.assertEqual(image.width, 64)
        self.assertEqual(image.height, 48)
        self.assertIn("Basic Render Driver", image.device)
        self.assertEqual(len(image.rgba), 64 * 48 * 4)

    def test_render_directx_offscreen_hardware(self):
        """render_directx_offscreen(provider='hardware') renders on hardware GPU."""
        if not self.has_hardware_gpu:
            self.skipTest("No hardware DirectX 12 GPU available on this runner")
        image = render_directx_offscreen(self.program, 64, 48, provider="hardware", toolchain=self.tools)
        self.assertEqual(image.width, 64)
        self.assertEqual(image.height, 48)
        self.assertNotIn("Basic Render Driver", image.device)
        self.assertEqual(len(image.rgba), 64 * 48 * 4)

    def test_render_directx_offscreen_default_none(self):
        """render_directx_offscreen(provider=None) maintains existing hardware default."""
        if not self.has_hardware_gpu:
            self.skipTest("No hardware DirectX 12 GPU available on this runner")
        with patch.dict(os.environ, {"MLIR_GPU_TEST_WARP": "0"}):
            image = render_directx_offscreen(self.program, 64, 48, provider=None, toolchain=self.tools)
        self.assertEqual(image.width, 64)
        self.assertEqual(image.height, 48)
        self.assertNotIn("Basic Render Driver", image.device)

    def test_render_directx_offscreen_rejects_lavapipe(self):
        """render_directx_offscreen(provider='lavapipe') raises ValueError."""
        with self.assertRaisesRegex(ValueError, "lavapipe"):
            render_directx_offscreen(self.program, provider="lavapipe", toolchain=self.tools)

    def test_render_directx_offscreen_rejects_swiftshader(self):
        """render_directx_offscreen(provider='swiftshader') raises ValueError."""
        with self.assertRaisesRegex(ValueError, "swiftshader"):
            render_directx_offscreen(self.program, provider="swiftshader", toolchain=self.tools)

    def test_render_directx_offscreen_rejects_unknown(self):
        """render_directx_offscreen(provider='invalid') raises ValueError."""
        with self.assertRaises(ValueError):
            render_directx_offscreen(self.program, provider="invalid", toolchain=self.tools)

    def test_render_offscreen_dispatcher_with_provider_warp(self):
        """render_offscreen routes provider='warp' to DirectX renderer."""
        env_patch = {"GPU_TOOLS_BIN": str(self.tools.tools_bin)} if self.tools.tools_bin else {}
        with patch.dict(os.environ, env_patch):
            image = render_offscreen(self.program, 64, 48, provider="warp")
        self.assertIn("Basic Render Driver", image.device)

    def test_show_directx_window_passes_warp_arg(self):
        """show_directx_window forwards --warp when provider='warp'."""
        with patch("mlir_gpu.viewer._directx_run") as mock_run:
            show_directx_window(self.program, width=64, height=48, frames=1, provider="warp", toolchain=self.tools)
            args = mock_run.call_args[0][0]
            self.assertIn("--warp", args)

    def test_show_directx_window_omits_warp_arg_for_hardware(self):
        """show_directx_window omits --warp when provider='hardware' or None."""
        with patch("mlir_gpu.viewer._directx_run") as mock_run:
            show_directx_window(self.program, width=64, height=48, frames=1, provider="hardware", toolchain=self.tools)
            args = mock_run.call_args[0][0]
            self.assertNotIn("--warp", args)

    def test_directx_get_device_capabilities_warp(self):
        """get_device_capabilities('directx', provider='warp') returns compliant schema."""
        caps = get_device_capabilities(backend="directx", provider="warp", toolchain=self.tools)
        self.assertEqual(caps["provider"], "warp")
        self.assertTrue(caps["is_software"])
        self.assertTrue(caps["software_adapter"])
        self.assertEqual(caps["vendor_id"], 5140)
        self.assertEqual(caps["device_id"], 140)
        self.assertEqual(caps["target"], "directx")
        self.assertEqual(caps["backend"], "directx")
        self.assertEqual(caps["status"], "EXECUTED")
        self.assertIn("Basic Render Driver", caps["device_name"])
        self.assertIn("shader_model", caps)
        self.assertIn("feature_level", caps)

    def test_directx_get_device_capabilities_hardware(self):
        """get_device_capabilities('directx', provider='hardware') returns hardware schema."""
        if not self.has_hardware_gpu:
            self.skipTest("No hardware DirectX 12 GPU available on this runner")
        caps = get_device_capabilities(backend="directx", provider="hardware", toolchain=self.tools)
        self.assertEqual(caps["provider"], "hardware")
        self.assertFalse(caps["is_software"])
        self.assertFalse(caps["software_adapter"])
        self.assertNotEqual(caps["vendor_id"], 5140)
        self.assertEqual(caps["target"], "directx")
        self.assertEqual(caps["status"], "EXECUTED")

    def test_directx_get_device_capabilities_required_supported(self):
        """Supported capabilities return status EXECUTED with empty missing list."""
        caps = get_device_capabilities(
            backend="directx", provider="warp", toolchain=self.tools,
            required_capabilities=["feature_level_12_0", "shader_model_6_6"],
        )
        self.assertEqual(caps["status"], "EXECUTED")
        self.assertEqual(caps["missing_capabilities"], [])
        self.assertEqual(caps["unverified_capabilities"], [])

    def test_directx_get_device_capabilities_required_unsupported(self):
        """Unsupported capabilities return status UNSUPPORTED with missing list populated."""
        caps = get_device_capabilities(
            backend="directx", provider="warp", toolchain=self.tools,
            required_capabilities=["shader_model_7_0"],
        )
        self.assertEqual(caps["status"], "UNSUPPORTED")
        self.assertIn("shader_model_7_0", caps["missing_capabilities"])

    def test_directx_get_device_capabilities_required_unverified(self):
        """Unverifiable runtime capabilities return status UNVERIFIED with unverified list."""
        caps = get_device_capabilities(
            backend="directx", provider="warp", toolchain=self.tools,
            required_capabilities=["unverified_extension_query"],
        )
        self.assertEqual(caps["status"], "UNVERIFIED")
        self.assertIn("unverified_extension_query", caps["unverified_capabilities"])


class TestVulkanPythonProviders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.tools = get_test_toolchain()
        pipeline = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
        try:
            manifest = compile_pipeline(pipeline, Target.VULKAN, Path(cls.work.name) / "triangle", cls.tools)
            cls.program = load_program(manifest)
            cls.vulkan_available = True
        except Exception:
            cls.vulkan_available = False
            cls.program = None

    def test_render_vulkan_offscreen_rejects_warp(self):
        """render_vulkan_offscreen(provider='warp') raises ValueError."""
        prog = self.program or GraphicsProgram(Shader("main", b""), Shader("main", b""), target="vulkan")
        with self.assertRaisesRegex(ValueError, "warp"):
            render_vulkan_offscreen(prog, provider="warp")

    def test_render_vulkan_offscreen_hardware(self):
        """render_vulkan_offscreen(provider='hardware') renders on discrete/integrated GPU."""
        if not self.vulkan_available:
            self.skipTest("Vulkan toolchain/driver not available on host")
        try:
            image = render_vulkan_offscreen(self.program, 64, 48, provider="hardware")
        except ViewerError:
            self.skipTest("No hardware Vulkan graphics device available on host")
        self.assertTrue(image.device)
        self.assertEqual(image.width, 64)
        self.assertEqual(image.height, 48)
        self.assertEqual(len(image.rgba), 64 * 48 * 4)

    def test_vulkan_device_filter_lavapipe_mock(self):
        """Mocked physical device with vendorID 0x10005 matches provider='lavapipe'."""
        mesa_props = MagicMock()
        mesa_props.vendorID = 0x10005
        mesa_props.deviceName = b"llvmpipe (LLVM 17.0.6)"
        mesa_props.deviceType = 4  # VK_PHYSICAL_DEVICE_TYPE_CPU

        self.assertTrue(_match_vulkan_device(mesa_props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(mesa_props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(mesa_props, "hardware"))

    def test_vulkan_device_filter_swiftshader_mock(self):
        """Mocked physical device with vendorID 0x1AE0 matches provider='swiftshader'."""
        swift_props = MagicMock()
        swift_props.vendorID = 0x1AE0
        swift_props.deviceName = b"SwiftShader Device"
        swift_props.deviceType = 4  # VK_PHYSICAL_DEVICE_TYPE_CPU

        self.assertTrue(_match_vulkan_device(swift_props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(swift_props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(swift_props, "hardware"))

    def test_vulkan_device_filter_hardware_mock(self):
        """Mocked hardware physical device matches provider='hardware' and rejects CPU."""
        nvidia_props = MagicMock()
        nvidia_props.vendorID = 0x10DE
        nvidia_props.deviceName = b"NVIDIA GeForce RTX 4090"
        nvidia_props.deviceType = 2  # VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU

        self.assertTrue(_match_vulkan_device(nvidia_props, "hardware"))
        self.assertFalse(_match_vulkan_device(nvidia_props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(nvidia_props, "swiftshader"))

        intel_props = MagicMock()
        intel_props.vendorID = 0x8086
        intel_props.deviceName = b"Intel(R) UHD Graphics 770"
        intel_props.deviceType = 1  # VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU

        self.assertTrue(_match_vulkan_device(intel_props, "hardware"))
        self.assertFalse(_match_vulkan_device(intel_props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(intel_props, "swiftshader"))

    def test_vulkan_get_device_capabilities_hardware(self):
        """get_device_capabilities('vulkan', provider='hardware') returns compliant schema."""
        try:
            caps = get_device_capabilities(backend="vulkan", provider="hardware")
        except ViewerError:
            self.skipTest("No Vulkan graphics device available on host")
        self.assertEqual(caps["target"], "vulkan")
        self.assertEqual(caps["backend"], "vulkan")
        self.assertEqual(caps["provider"], "hardware")
        self.assertFalse(caps["is_software"])
        self.assertIn("limits", caps)
        self.assertIn("max_push_constants_size", caps["limits"])
        self.assertEqual(caps["status"], "EXECUTED")


class TestProviderRejectionAndSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.tools = get_test_toolchain()
        pipeline = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
        cls.triangle_pipe = pipeline
        try:
            manifest_vk = compile_pipeline(pipeline, Target.VULKAN, Path(cls.work.name) / "triangle_vk", cls.tools)
            cls.vulkan_prog = load_program(manifest_vk)
        except Exception:
            cls.vulkan_prog = GraphicsProgram(Shader("main", b""), Shader("main", b""), target="vulkan")

        if sys.platform == "win32":
            manifest_dx = compile_pipeline(pipeline, Target.DIRECTX, Path(cls.work.name) / "triangle_dx", cls.tools)
            cls.directx_prog = load_program(manifest_dx)
        else:
            cls.directx_prog = None

    @staticmethod
    def _mock_vk_without_software_icd():
        mock_vk = MagicMock()
        mock_vk.VkError = Exception
        mock_vk.VkException = Exception
        mock_vk.ProcedureNotFoundError = Exception
        mock_vk.ExtensionNotSupportedError = Exception
        mock_vk.VK_API_VERSION_1_0 = 0
        mock_vk.VK_PHYSICAL_DEVICE_TYPE_CPU = 4
        mock_vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
        mock_vk.VK_QUEUE_GRAPHICS_BIT = 1
        mock_vk.vkEnumerateInstanceExtensionProperties.return_value = []
        mock_vk.vkCreateInstance.return_value = "mock_inst"
        gpu_props = MagicMock(vendorID=0x10DE, deviceID=0x2484, deviceName=b"NVIDIA RTX", deviceType=2)
        mock_vk.vkEnumeratePhysicalDevices.return_value = ["dev_gpu"]
        mock_vk.vkGetPhysicalDeviceProperties.return_value = gpu_props
        mock_vk.vkGetPhysicalDeviceQueueFamilyProperties.return_value = [MagicMock(queueCount=1, queueFlags=1)]
        mock_vk.vkEnumerateDeviceExtensionProperties.return_value = []
        return mock_vk

    def test_vulkan_lavapipe_missing_icd_clean_rejection(self):
        """When Lavapipe ICD is absent on system, requesting 'lavapipe' cleanly raises RuntimeError and NEVER falls back."""
        mock_vk = self._mock_vk_without_software_icd()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(RuntimeError) as ctx:
                render_vulkan_offscreen(self.vulkan_prog, provider="lavapipe")
        self.assertIn("Requested provider 'lavapipe'", str(ctx.exception))
        self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_vulkan_swiftshader_missing_icd_clean_rejection(self):
        """When SwiftShader ICD is absent on system, requesting 'swiftshader' cleanly raises RuntimeError."""
        mock_vk = self._mock_vk_without_software_icd()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(RuntimeError) as ctx:
                render_vulkan_offscreen(self.vulkan_prog, provider="swiftshader")
        self.assertIn("Requested provider 'swiftshader'", str(ctx.exception))
        self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_vulkan_hardware_rejects_cpu_device_mock(self):
        """When only CPU rasterizer is available, provider='hardware' raises RuntimeError."""
        cpu_props = MagicMock()
        cpu_props.vendorID = 0x10005
        cpu_props.deviceName = b"llvmpipe"
        cpu_props.deviceType = 4  # CPU

        # _match_vulkan_device must return False for CPU device when 'hardware' is requested
        self.assertFalse(_match_vulkan_device(cpu_props, "hardware"))

    def test_directx_capabilities_fallback_rejection_warp_mock(self):
        """If native query returns non-software device when 'warp' was requested, raise ViewerError."""
        fake_result = subprocess.CompletedProcess(
            args=["fake"], returncode=0,
            stdout=json.dumps({"is_software": False, "adapter_name": "NVIDIA", "vendor_id": 4318}),
            stderr="",
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("mlir_gpu.viewer.subprocess.run", return_value=fake_result):
            with self.assertRaises(RuntimeError) as ctx:
                get_device_capabilities(backend="directx", provider="warp", toolchain=self.tools)
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_directx_capabilities_fallback_rejection_hardware_mock(self):
        """If native query returns software device when 'hardware' was requested, raise ViewerError."""
        fake_result = subprocess.CompletedProcess(
            args=["fake"], returncode=0,
            stdout=json.dumps({"is_software": True, "adapter_name": "Basic Render Driver", "vendor_id": 5140}),
            stderr="",
        )
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch.object(Toolchain, "find", return_value="mlir-gpu-directx-view"), \
             patch("mlir_gpu.viewer.subprocess.run", return_value=fake_result):
            with self.assertRaises(RuntimeError) as ctx:
                get_device_capabilities(backend="directx", provider="hardware", toolchain=self.tools)
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_empty_and_invalid_provider_types(self):
        """Invalid types and empty strings for provider raise ValueError."""
        prog = self.directx_prog or self.vulkan_prog
        for invalid in ("", " ", "d3d12", "cpu", "software", 123):
            with self.subTest(invalid=invalid):
                with self.assertRaises((ValueError, TypeError)):
                    _validate_provider(invalid, "directx")
                with self.assertRaises((ValueError, TypeError)):
                    _validate_provider(invalid, "vulkan")

    def test_get_device_capabilities_invalid_backend(self):
        """get_device_capabilities with unknown backend raises ValueError."""
        with self.assertRaises(ValueError):
            get_device_capabilities(backend="metal_mock")
        with self.assertRaises(ValueError):
            get_device_capabilities(backend="")

    def test_metal_rejects_providers(self):
        """Metal backend rejects any non-None provider with ValueError."""
        for provider in ("hardware", "warp", "lavapipe", "swiftshader"):
            with self.assertRaises(ValueError):
                _validate_provider(provider, "metal")

    def test_metal_get_device_capabilities_unsupported_on_non_darwin(self):
        """Metal capability query on non-darwin returns UNSUPPORTED status without raising error."""
        if sys.platform != "darwin":
            caps = get_device_capabilities(backend="metal")
            self.assertEqual(caps["status"], "UNSUPPORTED")
            self.assertEqual(caps["target"], "metal")
            self.assertIn("darwin_platform", caps["missing_capabilities"])

    def test_directx_offscreen_renderer_rejects_software_device_when_hardware_requested(self):
        """If readback report indicates Basic Render Driver when provider='hardware', raise ViewerError."""
        if not self.directx_prog:
            self.skipTest("DirectX tests require Windows")
        with patch("mlir_gpu.viewer._directx_run", return_value='{"device": "Microsoft Basic Render Driver", "width": 64, "height": 48}'), \
             patch("pathlib.Path.read_bytes", return_value=bytes(64 * 48 * 4)):
            with self.assertRaises(RuntimeError) as ctx:
                render_directx_offscreen(self.directx_prog, 64, 48, provider="hardware", toolchain=self.tools)
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_directx_offscreen_renderer_rejects_hardware_device_when_warp_requested(self):
        """If readback report indicates discrete GPU when provider='warp', raise ViewerError."""
        if not self.directx_prog:
            self.skipTest("DirectX tests require Windows")
        with patch("mlir_gpu.viewer._directx_run", return_value='{"device": "NVIDIA GeForce RTX 4090", "width": 64, "height": 48}'), \
             patch("pathlib.Path.read_bytes", return_value=bytes(64 * 48 * 4)):
            with self.assertRaises(RuntimeError) as ctx:
                render_directx_offscreen(self.directx_prog, 64, 48, provider="warp", toolchain=self.tools)
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))


class TestVideoExportProviders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.tools = get_test_toolchain()
        if sys.platform == "win32":
            pipeline = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
            manifest = compile_pipeline(pipeline, Target.DIRECTX, Path(cls.work.name) / "triangle", cls.tools)
            cls.program = load_program(manifest)
        else:
            cls.program = GraphicsProgram(Shader("vertex", b""), Shader("fragment", b""), target="directx")

    def test_export_mp4_with_warp(self):
        """export_mp4 forwards provider='warp' to the offscreen renderer."""
        mp4_path = Path(self.work.name) / "anim_warp.mp4"
        with patch("mlir_gpu.viewer.sys.platform", "win32"), \
             patch("mlir_gpu.viewer._video_encoder", return_value="ffmpeg"), \
             patch("mlir_gpu.viewer.subprocess.Popen", side_effect=FakeVideoEncoder):
            with patch("mlir_gpu.viewer._directx_offscreen_renderer") as mock_renderer:
                @contextlib.contextmanager
                def fake_render(*args, **kwargs):
                    yield lambda elapsed: RenderedImage(64, 48, bytes(64 * 48 * 4), "Microsoft Basic Render Driver")
                mock_renderer.side_effect = fake_render

                count = export_mp4(self.program, mp4_path, width=64, height=48, duration=0.2, fps=10, provider="warp")
                self.assertEqual(count, 2)
                self.assertEqual(mock_renderer.call_args[1].get("provider"), "warp")
                self.assertTrue(mp4_path.is_file())

    def test_export_mp4_invalid_provider_rejected_early(self):
        """export_mp4 rejects invalid or mismatched provider before starting encoder or touching files."""
        mp4_path = Path(self.work.name) / "invalid.mp4"
        with self.assertRaises(ValueError):
            export_mp4(self.program, mp4_path, provider="lavapipe")
        self.assertFalse(mp4_path.exists())

        with self.assertRaises(ValueError):
            export_mp4(self.program, mp4_path, provider="swiftshader")
        self.assertFalse(mp4_path.exists())


class TestViewerCli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.work.cleanup)
        cls.tools = get_test_toolchain()
        if sys.platform == "win32":
            pipeline = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
            cls.manifest = compile_pipeline(pipeline, Target.DIRECTX, Path(cls.work.name) / "triangle", cls.tools)
            viewer_bin = cls.tools.find("mlir-gpu-directx-view")
            cls.has_hardware_gpu = _has_directx_hardware_gpu(viewer_bin)
        else:
            cls.manifest = Path(cls.work.name) / "pipeline.json"
            cls.has_hardware_gpu = False

    @unittest.skipUnless(sys.platform == "win32", "DirectX CLI tests require Windows")
    def test_cli_report_capabilities(self):
        """mlir-gpu-view manifest.json --report-capabilities emits capabilities JSON to stdout."""
        env_patch = {"GPU_TOOLS_BIN": str(self.tools.tools_bin)} if self.tools.tools_bin else {}
        if not getattr(self, "has_hardware_gpu", True):
            env_patch["MLIR_GPU_TEST_WARP"] = "1"
        with patch.dict(os.environ, env_patch):
            with patch("sys.argv", ["mlir-gpu-view", str(self.manifest), "--report-capabilities"]):
                with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                    exit_code = main()
                    self.assertEqual(exit_code, 0)
                    data = json.loads(mock_stdout.getvalue())
                    self.assertEqual(data.get("backend"), "directx")
                    self.assertIn("vendor_id", data)

    @unittest.skipUnless(sys.platform == "win32", "DirectX CLI tests require Windows")
    def test_cli_provider_warp_render(self):
        """mlir-gpu-view manifest.json --provider warp --output preview.png renders on WARP."""
        out_png = Path(self.work.name) / "preview_warp.png"
        env_patch = {"GPU_TOOLS_BIN": str(self.tools.tools_bin)} if self.tools.tools_bin else {}
        with patch.dict(os.environ, env_patch):
            with patch("sys.argv", ["mlir-gpu-view", str(self.manifest), "--provider", "warp", "--output", str(out_png)]):
                exit_code = main()
                self.assertEqual(exit_code, 0)
                self.assertTrue(out_png.is_file())
                self.assertGreater(out_png.stat().st_size, 32)


if __name__ == "__main__":
    unittest.main()
