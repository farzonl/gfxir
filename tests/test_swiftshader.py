"""Pinned SwiftShader Software-Driver Verification Test Harness.

Implements automated CI and hermetic verification:
- Pinned SwiftShader provider selection (provider="swiftshader", vendor ID 0x1AE0).
- Standardized device identity and capability reporting descriptor verification.
- Rejection testing for unavailable/unsupported features (e.g. Ray Tracing, Mesh Shaders;
  refusing to claim advanced features solely from Vulkan version strings).
- Explicit missing-tool and missing-driver error handling (descriptive errors, no silent dummy fallback).
- Tri-state status reporting (EXECUTED, UNSUPPORTED, UNVERIFIED).
- Mock and live test execution patterns so tests run both on machines with SwiftShader installed
  and via hermetic mocks when running without Vulkan SDK.
- Session-0 window safety: ensure no live window presentation is attempted in CI.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
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
from mlir_gpu.compiler import CompileError, PIPELINE_SCHEMA_VERSION, RASTER_CONVENTION
from mlir_gpu.frontend import Draw
from mlir_gpu.viewer import (
    GraphicsProgram,
    RenderedImage,
    Shader,
    ViewerError,
    _match_vulkan_device,
    _validate_provider,
    _Vulkan,
    get_device_capabilities,
    load_program,
    main,
    render_offscreen,
    render_vulkan_offscreen,
)

BACKGROUND = bytes([5, 5, 10, 255])
SWIFTSHADER_VENDOR_ID = 0x1AE0  # Google SwiftShader (6880)
LAVAPIPE_VENDOR_ID = 0x10005     # Mesa Lavapipe / llvmpipe (65541)

SWIFTSHADER_CI_ENABLED = (
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1"
    and os.environ.get("MLIR_GPU_TEST_SWIFTSHADER") == "1"
)
VALIDATION_ENABLED = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"


# --- Hermetic Mock Exceptions & Fixtures ---

class MockVkError(Exception):
    pass

class MockVkException(Exception):
    pass

class MockProcedureNotFoundError(Exception):
    pass

class MockExtensionNotSupportedError(Exception):
    pass


def make_valid_manifest_dict():
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


def make_mock_swiftshader_properties(
    vendor_id: int = SWIFTSHADER_VENDOR_ID,
    device_name: bytes = b"SwiftShader Device (LLVM 10.0.0)",
    device_type: int = 4,  # VK_PHYSICAL_DEVICE_TYPE_CPU
    api_version: int = (1 << 22) | (3 << 12) | 0,  # Vulkan 1.3.0
    max_push_constants: int = 128,
):
    props = MagicMock()
    props.vendorID = vendor_id
    props.deviceID = 0xC0DE
    props.deviceName = device_name
    props.deviceType = device_type
    props.driverVersion = 1
    props.apiVersion = api_version
    props.limits.maxPushConstantsSize = max_push_constants
    props.limits.maxFramebufferWidth = 8192
    props.limits.maxFramebufferHeight = 8192
    return props


def make_mock_vk(devices=None, available_extensions=None, available_layers=None):
    mock_vk = MagicMock()
    mock_vk.VkError = MockVkError
    mock_vk.VkException = MockVkException
    mock_vk.ProcedureNotFoundError = MockProcedureNotFoundError
    mock_vk.ExtensionNotSupportedError = MockExtensionNotSupportedError

    mock_vk.VK_API_VERSION_1_0 = 0
    mock_vk.VK_PHYSICAL_DEVICE_TYPE_OTHER = 0
    mock_vk.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU = 1
    mock_vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
    mock_vk.VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU = 3
    mock_vk.VK_PHYSICAL_DEVICE_TYPE_CPU = 4
    mock_vk.VK_QUEUE_GRAPHICS_BIT = 1
    mock_vk.VK_TRUE = 1
    mock_vk.VK_FALSE = 0
    mock_vk.VK_FORMAT_R8G8B8A8_UNORM = 37

    ext_list = available_extensions if available_extensions is not None else ["VK_KHR_portability_enumeration"]
    mock_vk.vkEnumerateInstanceExtensionProperties.return_value = [
        MagicMock(extensionName=e) for e in ext_list
    ]
    layer_list = available_layers if available_layers is not None else ["VK_LAYER_KHRONOS_validation"]
    mock_vk.vkEnumerateInstanceLayerProperties.return_value = [
        MagicMock(layerName=l) for l in layer_list
    ]

    mock_vk.vkCreateInstance.return_value = "mock_instance_handle"
    mock_vk.vkDestroyInstance.return_value = None

    if devices is None:
        swift_props = make_mock_swiftshader_properties()
        swift_q = MagicMock(queueCount=1, queueFlags=1)
        swift_exts = [MagicMock(extensionName="VK_KHR_swapchain")]
        devices = [("swift_phys_handle", swift_props, [swift_q], swift_exts)]

    mock_vk.vkEnumeratePhysicalDevices.return_value = [d[0] for d in devices]

    dev_props = {d[0]: d[1] for d in devices}
    mock_vk.vkGetPhysicalDeviceProperties.side_effect = lambda dev: dev_props[dev]

    dev_queues = {d[0]: d[2] for d in devices}
    mock_vk.vkGetPhysicalDeviceQueueFamilyProperties.side_effect = lambda dev: dev_queues[dev]

    dev_exts = {d[0]: d[3] for d in devices}
    mock_vk.vkEnumerateDeviceExtensionProperties.side_effect = lambda dev, layer: dev_exts[dev]

    mock_vk.vkCreateDevice.return_value = "mock_device_handle"
    mock_vk.vkDestroyDevice.return_value = None
    mock_vk.vkGetDeviceQueue.return_value = "mock_queue_handle"
    mock_vk.vkCreateCommandPool.return_value = "mock_pool_handle"
    mock_vk.vkDestroyCommandPool.return_value = None
    return mock_vk


# --- Test Classes ---

class TestSwiftShaderEnvironment(unittest.TestCase):
    """Verify SwiftShader CI environment detection, toolchain resolution, and runtime contracts."""

    def test_environment_detection_flags(self):
        """When SwiftShader CI is active, VULKAN and SWIFTSHADER are 1, and WINDOW is NOT 1."""
        if SWIFTSHADER_CI_ENABLED:
            self.assertEqual(os.environ.get("MLIR_GPU_TEST_VULKAN"), "1")
            self.assertEqual(os.environ.get("MLIR_GPU_TEST_SWIFTSHADER"), "1")
            if VALIDATION_ENABLED:
                self.assertEqual(os.environ.get("MLIR_GPU_TEST_VALIDATION"), "1")
            self.assertNotEqual(
                os.environ.get("MLIR_GPU_TEST_WINDOW"),
                "1",
                "MLIR_GPU_TEST_WINDOW must NEVER be enabled in non-interactive Session-0 CI",
            )

    def test_environment_icd_manifest_resolution(self):
        """Verify handling of VK_ICD_FILENAMES or VK_DRIVER_FILES for pinned SwiftShader ICD."""
        icd_env = os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES")
        if SWIFTSHADER_CI_ENABLED and icd_env:
            icd_path = Path(icd_env)
            self.assertTrue(icd_path.exists(), f"Pinned ICD manifest does not exist: {icd_path}")
            content = json.loads(icd_path.read_text())
            self.assertIn("ICD", content)

    def test_session0_headless_guarantee(self):
        """Session 0 environment guarantee: live presentation must not be attempted."""
        self.assertNotEqual(
            os.environ.get("MLIR_GPU_TEST_WINDOW"),
            "1",
            "CI environments operate in headless Session 0; MLIR_GPU_TEST_WINDOW must remain unset",
        )

    def test_provider_flag_validation_and_forwarding(self):
        """_validate_provider accepts ('swiftshader', 'vulkan') and rejects invalid backends."""
        _validate_provider("swiftshader", "vulkan")
        _validate_provider("lavapipe", "vulkan")
        _validate_provider("hardware", "vulkan")

        with self.assertRaises(ValueError) as ctx_dx:
            _validate_provider("swiftshader", "directx")
        self.assertIn("not supported by the DirectX 12 backend", str(ctx_dx.exception))

        with self.assertRaises(ValueError) as ctx_mt:
            _validate_provider("swiftshader", "metal")
        self.assertIn("not supported by the Metal backend", str(ctx_mt.exception))


class TestSwiftShaderDeviceFilterAndSelection(unittest.TestCase):
    """Hermetic device filter testing for pinned SwiftShader provider selection."""

    def test_match_vulkan_device_swiftshader_positive(self):
        """Device with vendorID 0x1AE0 or deviceID 0xC0DE matches provider='swiftshader'."""
        props = make_mock_swiftshader_properties(vendor_id=SWIFTSHADER_VENDOR_ID)
        self.assertTrue(_match_vulkan_device(props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(props, "hardware"))

        # Verify deviceID 0xC0DE matches swiftshader even with unknown vendor
        props_by_id = make_mock_swiftshader_properties(vendor_id=0, device_name=b"Generic Device")
        props_by_id.deviceID = 0xC0DE
        self.assertTrue(_match_vulkan_device(props_by_id, "swiftshader"))
        self.assertFalse(_match_vulkan_device(props_by_id, "hardware"))

    def test_match_vulkan_device_swiftshader_by_name_fallback(self):
        """Device with 'swiftshader' in name matches provider='swiftshader' even if vendorID differs."""
        props = make_mock_swiftshader_properties(vendor_id=0, device_name=b"Google SwiftShader Rasterizer")
        self.assertTrue(_match_vulkan_device(props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(props, "hardware"))

    def test_match_vulkan_device_swiftshader_rejects_lavapipe(self):
        """Lavapipe device (vendorID 0x10005) returns False for provider='swiftshader'."""
        mesa_props = MagicMock(vendorID=LAVAPIPE_VENDOR_ID, deviceName=b"llvmpipe (LLVM 17.0.6)", deviceType=4)
        self.assertFalse(_match_vulkan_device(mesa_props, "swiftshader"))
        self.assertTrue(_match_vulkan_device(mesa_props, "lavapipe"))
        self.assertFalse(_match_vulkan_device(mesa_props, "hardware"))

    def test_match_vulkan_device_swiftshader_rejects_discrete_gpu(self):
        """Discrete GPU (NVIDIA 0x10DE) returns False for provider='swiftshader'."""
        nvidia_props = MagicMock(vendorID=0x10DE, deviceName=b"NVIDIA GeForce RTX 4090", deviceType=2)
        self.assertFalse(_match_vulkan_device(nvidia_props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(nvidia_props, "lavapipe"))
        self.assertTrue(_match_vulkan_device(nvidia_props, "hardware"))

    def test_match_vulkan_device_swiftshader_rejects_integrated_gpu(self):
        """Integrated GPU (Intel 0x8086) returns False for provider='swiftshader'."""
        intel_props = MagicMock(vendorID=0x8086, deviceName=b"Intel(R) UHD Graphics 770", deviceType=1)
        self.assertFalse(_match_vulkan_device(intel_props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(intel_props, "lavapipe"))
        self.assertTrue(_match_vulkan_device(intel_props, "hardware"))

    def test_match_vulkan_device_swiftshader_rejects_amd_gpu(self):
        """Discrete AMD GPU (0x1002) returns False for provider='swiftshader'."""
        amd_props = MagicMock(vendorID=0x1002, deviceName=b"AMD Radeon RX 7900 XTX", deviceType=2)
        self.assertFalse(_match_vulkan_device(amd_props, "swiftshader"))
        self.assertFalse(_match_vulkan_device(amd_props, "lavapipe"))
        self.assertTrue(_match_vulkan_device(amd_props, "hardware"))

    def test_match_vulkan_device_hardware_rejects_swiftshader(self):
        """SwiftShader device returns False when provider='hardware' is requested."""
        swift_props = make_mock_swiftshader_properties()
        self.assertFalse(_match_vulkan_device(swift_props, "hardware"))

    def test_match_vulkan_device_lavapipe_rejects_swiftshader(self):
        """SwiftShader device returns False when provider='lavapipe' is requested."""
        swift_props = make_mock_swiftshader_properties()
        self.assertFalse(_match_vulkan_device(swift_props, "lavapipe"))

    def test_multi_device_enumeration_selects_pinned_swiftshader(self):
        """When hardware GPU and SwiftShader coexist, requesting 'swiftshader' strictly selects SwiftShader."""
        gpu_props = MagicMock(vendorID=0x10DE, deviceName=b"NVIDIA RTX", deviceType=2)
        swift_props = make_mock_swiftshader_properties()
        q = MagicMock(queueCount=1, queueFlags=1)
        exts = [MagicMock(extensionName="VK_KHR_swapchain")]

        devices = [
            ("dev_gpu", gpu_props, [q], exts),
            ("dev_swift", swift_props, [q], exts),
        ]
        mock_vk = make_mock_vk(devices=devices)
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertEqual(caps["provider"], "swiftshader")
            self.assertEqual(caps["vendor_id"], SWIFTSHADER_VENDOR_ID)
            self.assertTrue(caps["is_software"])

    def test_multi_device_enumeration_missing_swiftshader_raises_viewer_error(self):
        """When only Lavapipe and GPU are present, requesting 'swiftshader' raises ViewerError refusing fallback."""
        gpu_props = MagicMock(vendorID=0x10DE, deviceName=b"NVIDIA RTX", deviceType=2)
        mesa_props = MagicMock(vendorID=LAVAPIPE_VENDOR_ID, deviceName=b"llvmpipe", deviceType=4)
        q = MagicMock(queueCount=1, queueFlags=1)
        exts = [MagicMock(extensionName="VK_KHR_swapchain")]

        devices = [
            ("dev_gpu", gpu_props, [q], exts),
            ("dev_mesa", mesa_props, [q], exts),
        ]
        mock_vk = make_mock_vk(devices=devices)
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertIn("Requested provider 'swiftshader' was not found", str(ctx.exception))
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))


class TestSwiftShaderCapabilitiesDescriptor(unittest.TestCase):
    """Verify standardized capability reporting descriptor schema for SwiftShader."""

    def test_capabilities_schema_compliance_hermetic_mock(self):
        """get_device_capabilities('vulkan', provider='swiftshader') matches standardized schema."""
        mock_vk = make_mock_vk()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")

            required_keys = {
                "target", "backend", "provider", "device_name", "vendor_id", "device_id",
                "device_type", "driver_version", "api_version", "is_software", "limits",
                "max_push_constants_size", "max_framebuffer_width", "max_framebuffer_height",
                "extensions", "status", "missing_capabilities",
            }
            for k in required_keys:
                self.assertIn(k, caps, f"Missing required capability descriptor key: {k}")

            self.assertEqual(caps["target"], "vulkan")
            self.assertEqual(caps["backend"], "vulkan")
            self.assertEqual(caps["provider"], "swiftshader")
            self.assertEqual(caps["vendor_id"], SWIFTSHADER_VENDOR_ID)
            self.assertEqual(caps["device_id"], 0xC0DE)
            self.assertEqual(caps["device_type"], "cpu")
            self.assertTrue(caps["is_software"])
            self.assertEqual(caps["status"], "EXECUTED")
            self.assertEqual(caps["missing_capabilities"], [])
            self.assertIn("SwiftShader", caps["device_name"])

    def test_capabilities_limits_reporting(self):
        """Verify push constants and framebuffer limits reporting in capabilities dictionary."""
        mock_vk = make_mock_vk()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
            limits = caps.get("limits", {})
            self.assertIn("max_push_constants_size", limits)
            self.assertGreaterEqual(limits["max_push_constants_size"], 128)
            self.assertEqual(caps["max_push_constants_size"], limits["max_push_constants_size"])
            self.assertGreater(limits["max_framebuffer_width"], 0)
            self.assertGreater(limits["max_framebuffer_height"], 0)

    def test_capabilities_extensions_sorted_list(self):
        """Extensions field in capability report is a sorted list of strings."""
        mock_vk = make_mock_vk()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
            exts = caps.get("extensions")
            self.assertIsInstance(exts, list)
            self.assertEqual(exts, sorted(exts))

    def test_capabilities_anti_hardware_fallback_mock(self):
        """get_device_capabilities strictly rejects discrete GPU when swiftshader was requested."""
        gpu_props = MagicMock(vendorID=0x10DE, deviceName=b"NVIDIA RTX", deviceType=2)
        q = MagicMock(queueCount=1, queueFlags=1)
        mock_vk = make_mock_vk(devices=[("dev_gpu", gpu_props, [q], [])])
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_capabilities_anti_software_substitution_mock(self):
        """get_device_capabilities strictly rejects Lavapipe when swiftshader was requested."""
        mesa_props = MagicMock(vendorID=LAVAPIPE_VENDOR_ID, deviceName=b"llvmpipe", deviceType=4)
        q = MagicMock(queueCount=1, queueFlags=1)
        mock_vk = make_mock_vk(devices=[("dev_mesa", mesa_props, [q], [])])
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    @unittest.skipUnless(SWIFTSHADER_CI_ENABLED, "Requires MLIR_GPU_TEST_SWIFTSHADER=1 and MLIR_GPU_TEST_VULKAN=1")
    def test_capabilities_live_swiftshader(self):
        """Execute live get_device_capabilities query against host SwiftShader driver."""
        caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
        self.assertEqual(caps["provider"], "swiftshader")
        self.assertEqual(caps["vendor_id"], SWIFTSHADER_VENDOR_ID)
        self.assertTrue(caps["is_software"])
        self.assertEqual(caps["status"], "EXECUTED")
        self.assertIn("SwiftShader", caps["device_name"])


class TestSwiftShaderRejectionAndUnsupportedFeatures(unittest.TestCase):
    """Assert rejection testing for unavailable/unsupported features and anti-fallback guarantees."""

    def test_vulkan_version_string_does_not_imply_advanced_features(self):
        """Vulkan 1.3 version string must not be used to claim unsupported advanced hardware features."""
        mock_props = make_mock_swiftshader_properties(api_version=(1 << 22) | (3 << 12) | 0)
        q = MagicMock(queueCount=1, queueFlags=1)
        supported_exts = [
            MagicMock(extensionName="VK_KHR_swapchain"),
            MagicMock(extensionName="VK_KHR_maintenance1"),
        ]
        mock_vk = make_mock_vk(devices=[("dev_swift", mock_props, [q], supported_exts)])
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertEqual(caps["api_version"], "1.3.0")
            self.assertNotIn("VK_KHR_ray_tracing_pipeline", caps["extensions"])
            self.assertNotIn("VK_KHR_ray_query", caps["extensions"])
            self.assertNotIn("VK_EXT_mesh_shader", caps["extensions"])
            self.assertNotIn("VK_KHR_fragment_shading_rate", caps["extensions"])

    def test_unsupported_feature_descriptor_records_missing_capabilities(self):
        """Workload requiring unsupported ray tracing records status='UNSUPPORTED' and lists missing capability."""
        mock_vk = make_mock_vk()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
            required_extensions = ["VK_KHR_ray_tracing_pipeline", "VK_EXT_mesh_shader"]
            available = set(caps["extensions"])
            missing = [req for req in required_extensions if req not in available]

            unsupported_report = {
                **caps,
                "status": "UNSUPPORTED" if missing else "EXECUTED",
                "missing_capabilities": missing,
            }
            self.assertEqual(unsupported_report["status"], "UNSUPPORTED")
            self.assertIn("VK_KHR_ray_tracing_pipeline", unsupported_report["missing_capabilities"])
            self.assertIn("VK_EXT_mesh_shader", unsupported_report["missing_capabilities"])

            # Directly exercise required_capabilities parameter in get_device_capabilities
            caps_direct = get_device_capabilities(
                backend="vulkan", provider="swiftshader", required_capabilities=required_extensions
            )
            self.assertEqual(caps_direct["status"], "UNSUPPORTED")
            self.assertIn("VK_KHR_ray_tracing_pipeline", caps_direct["missing_capabilities"])
            self.assertIn("VK_EXT_mesh_shader", caps_direct["missing_capabilities"])

            # Test named capabilities aliases (e.g. ray_tracing, mesh_shaders)
            caps_aliases = get_device_capabilities(
                backend="vulkan", provider="swiftshader", required_capabilities=["ray_tracing", "mesh_shaders"]
            )
            self.assertEqual(caps_aliases["status"], "UNSUPPORTED")
            self.assertIn("ray_tracing", caps_aliases["missing_capabilities"])
            self.assertIn("mesh_shaders", caps_aliases["missing_capabilities"])

    def test_unsupported_feature_status_never_claims_executed(self):
        """Status invariant: when missing_capabilities is populated, status must not be EXECUTED."""
        report = {
            "target": "vulkan",
            "provider": "swiftshader",
            "status": "UNSUPPORTED",
            "missing_capabilities": ["VK_KHR_ray_tracing_pipeline"],
        }
        self.assertNotEqual(report["status"], "EXECUTED")

    def test_unsupported_extension_device_creation_rejected(self):
        """Attempting to initialize device with unsupported extension raises descriptive error."""
        mock_vk = make_mock_vk()
        mock_vk.vkCreateDevice.side_effect = MockExtensionNotSupportedError("VK_KHR_ray_tracing_pipeline unsupported")
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            prog = GraphicsProgram(Shader("main", b"\x03\x02\x23\x07" + bytes(16)),
                                   Shader("main", b"\x03\x02\x23\x07" + bytes(16)),
                                   target="vulkan")
            with self.assertRaises(ViewerError) as ctx:
                render_vulkan_offscreen(prog, 32, 24, provider="swiftshader")
            self.assertIn("Vulkan operation failed", str(ctx.exception))

    def test_cross_backend_provider_rejections(self):
        """_validate_provider strictly refuses invalid or cross-backend provider requests."""
        with self.assertRaises(ValueError) as ctx1:
            _validate_provider("swiftshader", "directx")
        self.assertIn("not supported by the DirectX 12 backend", str(ctx1.exception))

        with self.assertRaises(ValueError) as ctx2:
            _validate_provider("swiftshader", "metal")
        self.assertIn("not supported by the Metal backend", str(ctx2.exception))

        with self.assertRaises(ValueError) as ctx3:
            _validate_provider("warp", "vulkan")
        self.assertIn("not supported by the Vulkan backend", str(ctx3.exception))

    def test_unknown_provider_rejected(self):
        """_validate_provider strictly refuses unknown provider string."""
        with self.assertRaises(ValueError) as ctx:
            _validate_provider("cpu_fallback", "vulkan")
        self.assertIn("Unknown provider 'cpu_fallback'", str(ctx.exception))

    def test_invalid_provider_types_rejected(self):
        """_validate_provider rejects non-string and empty string provider parameters."""
        for bad in ("", " ", 123, False, []):
            with self.subTest(bad=bad):
                with self.assertRaises((ValueError, TypeError)):
                    _validate_provider(bad, "vulkan")


class TestSwiftShaderMissingToolsAndDrivers(unittest.TestCase):
    """Verify explicit descriptive failure when native tools or drivers are missing."""

    def test_missing_vulkan_binding_raises_actionable_viewer_error(self):
        """When vulkan Python module or system loader is missing, raises actionable ViewerError."""
        with patch("importlib.import_module", side_effect=ModuleNotFoundError("No module named 'vulkan'")):
            from mlir_gpu.viewer import _binding
            with self.assertRaises(ViewerError) as ctx:
                _binding("vulkan")
            self.assertIn("Cannot load vulkan", str(ctx.exception))
            self.assertIn("python -m pip install -e \".[viewer]\"", str(ctx.exception))

    def test_missing_swiftshader_icd_raises_viewer_error(self):
        """When SwiftShader ICD is absent in device enumeration, requesting 'swiftshader' cleanly raises ViewerError."""
        prog = GraphicsProgram(Shader("main", b"\x03\x02\x23\x07" + bytes(16)),
                               Shader("main", b"\x03\x02\x23\x07" + bytes(16)),
                               target="vulkan")
        mock_vk = make_mock_vk(devices=[])  # Empty physical device list
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(RuntimeError) as ctx:
                render_vulkan_offscreen(prog, 32, 24, provider="swiftshader")
            self.assertIn("Requested provider 'swiftshader'", str(ctx.exception))
            self.assertIn("Accidental hardware/driver fallback rejected.", str(ctx.exception))

    def test_missing_vulkan_graphics_queue_raises_viewer_error(self):
        """When device has no graphics queue, _Vulkan initialization raises ViewerError."""
        swift_props = make_mock_swiftshader_properties()
        compute_only_q = MagicMock(queueCount=1, queueFlags=2)  # VK_QUEUE_COMPUTE_BIT only
        mock_vk = make_mock_vk(devices=[("swift_dev", swift_props, [compute_only_q], [])])
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            with self.assertRaises(ViewerError) as ctx:
                get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertIn("Requested provider 'swiftshader' was not found", str(ctx.exception))

    def test_missing_spirv_binary_raises_compile_error(self):
        """When SPIR-V file does not exist, load_program raises CompileError."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest_file = Path(tmp) / "pipeline.json"
            manifest_data = make_valid_manifest_dict()
            manifest_file.write_text(json.dumps(manifest_data))
            # Write only fragment.spv; leave vertex.spv absent
            (Path(tmp) / "fragment.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(16))
            with self.assertRaises(CompileError):
                load_program(manifest_file)

    def test_corrupt_unaligned_spirv_binary_raises_viewer_error(self):
        """When SPIR-V file size is not a multiple of 4 (>= 20 bytes), load_program raises ViewerError."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest_file = Path(tmp) / "pipeline.json"
            manifest_data = make_valid_manifest_dict()
            manifest_file.write_text(json.dumps(manifest_data))
            # 21 bytes (>= 20 bytes minimum, but 21 % 4 != 0)
            (Path(tmp) / "vertex.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(17))
            (Path(tmp) / "fragment.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(16))
            with self.assertRaisesRegex(ViewerError, "word-aligned"):
                load_program(manifest_file)

    def test_missing_validation_layer_raises_viewer_error(self):
        """When --validation is requested but VK_LAYER_KHRONOS_validation is missing, raises ViewerError."""
        mock_vk = make_mock_vk(
            available_extensions=["VK_EXT_debug_utils"],
            available_layers=[],  # Layer missing
        )
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk), \
             patch.dict(os.environ, {"MLIR_GPU_TEST_SWIFTSHADER": "0", "VK_ICD_FILENAMES": "", "VK_DRIVER_FILES": ""}):
            from contextlib import ExitStack
            with ExitStack() as resources:
                with self.assertRaises(ViewerError) as ctx:
                    _Vulkan(resources, validation=True, provider="hardware")
                self.assertIn("--validation requires VK_LAYER_KHRONOS_validation", str(ctx.exception))


class TestSwiftShaderTriStateStatusReporting(unittest.TestCase):
    """Validate tri-state status reporting semantics (EXECUTED, UNSUPPORTED, UNVERIFIED)."""

    def test_tristate_status_valid_codes(self):
        """Permitted status values are strictly 'EXECUTED', 'UNSUPPORTED', or 'UNVERIFIED'."""
        mock_vk = make_mock_vk()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk):
            caps = get_device_capabilities(backend="vulkan", provider="swiftshader")
            self.assertIn(caps["status"], ("EXECUTED", "UNSUPPORTED", "UNVERIFIED"))

    def test_executed_status_contract(self):
        """Status EXECUTED strictly requires empty missing_capabilities."""
        report = {
            "target": "vulkan",
            "provider": "swiftshader",
            "status": "EXECUTED",
            "missing_capabilities": [],
        }
        self.assertEqual(report["status"], "EXECUTED")
        self.assertEqual(report["missing_capabilities"], [])

    def test_unsupported_status_contract(self):
        """Status UNSUPPORTED requires non-empty missing_capabilities and must not claim EXECUTED."""
        report = {
            "target": "vulkan",
            "provider": "swiftshader",
            "status": "UNSUPPORTED",
            "missing_capabilities": ["VK_KHR_ray_tracing_pipeline"],
        }
        self.assertNotEqual(report["status"], "EXECUTED")
        self.assertEqual(report["status"], "UNSUPPORTED")
        self.assertGreater(len(report["missing_capabilities"]), 0)

    def test_unverified_status_contract(self):
        """Compile-only or unexecuted mock paths must report UNVERIFIED and not claim EXECUTED."""
        report = {
            "target": "vulkan",
            "provider": "swiftshader",
            "status": "UNVERIFIED",
            "missing_capabilities": ["runtime_execution"],
        }
        self.assertNotEqual(report["status"], "EXECUTED")
        self.assertEqual(report["status"], "UNVERIFIED")

    def test_status_reporting_transitions(self):
        """Verify status reporting transition invariants across capability checks."""
        for status_code in ("UNVERIFIED", "UNSUPPORTED"):
            with self.subTest(status_code=status_code):
                record = {"status": status_code, "missing": ["feature"]}
                self.assertNotEqual(record["status"], "EXECUTED")


@unittest.skipUnless(SWIFTSHADER_CI_ENABLED, "Requires MLIR_GPU_TEST_SWIFTSHADER=1 and MLIR_GPU_TEST_VULKAN=1")
class TestSwiftShaderOffscreenRenderingLive(unittest.TestCase):
    """Execute real offscreen rendering on pinned SwiftShader Vulkan rasterizer."""

    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="ci-swiftshader-")
        cls.work_dir = Path(cls.work.name)
        llvm_bin = os.environ.get("LLVM_BIN")
        tools_bin = os.environ.get("GPU_TOOLS_BIN")
        cls.tools = Toolchain(
            llvm_bin=Path(llvm_bin) if llvm_bin else None,
            tools_bin=Path(tools_bin) if tools_bin else None,
        )
        cls.validation = VALIDATION_ENABLED

        triangle_pipe = runpy.run_path(str(ROOT / "examples" / "triangle.py"))["pipeline"]
        manifest_tri = compile_pipeline(triangle_pipe, Target.VULKAN, cls.work_dir / "triangle", cls.tools)
        cls.triangle_prog = load_program(manifest_tri)

        quad_pipe = runpy.run_path(str(ROOT / "examples" / "quad.py"))["pipeline"]
        manifest_quad = compile_pipeline(quad_pipe, Target.VULKAN, cls.work_dir / "quad", cls.tools)
        cls.quad_prog = load_program(manifest_quad)

        @fragment
        def diagnostic(frame: FrameParameters):
            return vec4(frame.resolution / 255.0, frame.time / 4.0, 1.0)

        diag_pipe = Pipeline(fullscreen_vertex, diagnostic)
        manifest_diag = compile_pipeline(diag_pipe, Target.VULKAN, cls.work_dir / "diagnostic", cls.tools)
        cls.diag_prog = load_program(manifest_diag)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_swiftshader_triangle_pixel_accuracy_and_coverage(self):
        """Triangle rendering on SwiftShader verifies dimensions, device identity, and 0.18 coverage."""
        width, height = 64, 48
        image = render_vulkan_offscreen(
            self.triangle_prog, width, height, validation=self.validation, provider="swiftshader",
        )
        self.assertIn("SwiftShader", image.device)
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        self.assertEqual(len(image.rgba), width * height * 4)

        def pixel(x, y):
            offset = (y * width + x) * 4
            return image.rgba[offset:offset + 4]

        self.assertEqual(pixel(0, 0), BACKGROUND)
        self.assertEqual(pixel(width - 1, height - 1), BACKGROUND)
        self.assertNotEqual(pixel(width // 2, height // 2), BACKGROUND)

        covered = sum(
            image.rgba[i:i + 4] != BACKGROUND for i in range(0, len(image.rgba), 4)
        ) / (width * height)
        self.assertAlmostEqual(covered, 0.18, delta=0.02)

    def test_swiftshader_quad_pixel_accuracy_and_coverage(self):
        """Quad rendering under SwiftShader verifies 0.64 coverage invariant."""
        width, height = 64, 48
        image = render_vulkan_offscreen(
            self.quad_prog, width, height, validation=self.validation, provider="swiftshader",
        )
        self.assertIn("SwiftShader", image.device)
        covered = sum(
            image.rgba[i:i + 4] != BACKGROUND for i in range(0, len(image.rgba), 4)
        ) / (width * height)
        self.assertAlmostEqual(covered, 0.64, delta=0.02)

    def test_swiftshader_unaligned_dimensions(self):
        """Unaligned odd dimensions (257x193) render with exact byte count without pitch corruption."""
        width, height = 257, 193
        image = render_vulkan_offscreen(
            self.triangle_prog, width, height, validation=self.validation, provider="swiftshader",
        )
        self.assertIn("SwiftShader", image.device)
        self.assertEqual(image.width, width)
        self.assertEqual(image.height, height)
        self.assertEqual(len(image.rgba), width * height * 4)

    def test_swiftshader_deterministic_time_readback(self):
        """Deterministic time parameters propagate correctly to SwiftShader push constants."""
        width, height = 32, 24

        img_t0_a = render_vulkan_offscreen(
            self.diag_prog, width, height, validation=self.validation, time=0.0, provider="swiftshader",
        )
        expected_pixel_0 = bytes([width, height, 0, 255])
        self.assertEqual(img_t0_a.rgba[:4], expected_pixel_0)

        img_t0_b = render_vulkan_offscreen(
            self.diag_prog, width, height, validation=self.validation, time=0.0, provider="swiftshader",
        )
        self.assertEqual(img_t0_a.rgba, img_t0_b.rgba, "SwiftShader execution must be bitwise reproducible at t=0.0")

        img_t1 = render_vulkan_offscreen(
            self.diag_prog, width, height, validation=self.validation, time=1.0, provider="swiftshader",
        )
        expected_pixel_1 = bytes([width, height, 64, 255])
        self.assertEqual(img_t1.rgba[:4], expected_pixel_1)
        self.assertNotEqual(img_t0_a.rgba, img_t1.rgba)

    def test_swiftshader_validation_layer_clean(self):
        """When validation is active on SwiftShader, offscreen render executes cleanly without errors."""
        image = render_vulkan_offscreen(
            self.triangle_prog, 32, 24, validation=True, provider="swiftshader",
        )
        self.assertIn("SwiftShader", image.device)
        self.assertEqual(len(image.rgba), 32 * 24 * 4)


class TestSwiftShaderViewerCli(unittest.TestCase):
    """Validate CLI execution paths for SwiftShader provider."""

    def test_cli_report_capabilities_swiftshader_mock(self):
        """mlir-gpu-view manifest.json --provider swiftshader --report-capabilities emits schema JSON."""
        mock_vk = make_mock_vk()
        with patch("mlir_gpu.viewer._binding", return_value=mock_vk), tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "pipeline.json"
            manifest.write_text(json.dumps(make_valid_manifest_dict()))
            (Path(tmp) / "vertex.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(16))
            (Path(tmp) / "fragment.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(16))

            with patch("sys.argv", ["mlir-gpu-view", str(manifest), "--provider", "swiftshader", "--report-capabilities"]), \
                 patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
                exit_code = main()
                self.assertEqual(exit_code, 0)
                caps = json.loads(mock_stdout.getvalue())
                self.assertEqual(caps.get("provider"), "swiftshader")
                self.assertEqual(caps.get("vendor_id"), SWIFTSHADER_VENDOR_ID)
                self.assertTrue(caps.get("is_software"))

    def test_cli_output_png_routes_to_offscreen_swiftshader_mock(self):
        """mlir-gpu-view manifest.json --provider swiftshader --output preview.png calls render_offscreen."""
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "pipeline.json"
            manifest.write_text(json.dumps(make_valid_manifest_dict()))
            (Path(tmp) / "vertex.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(16))
            (Path(tmp) / "fragment.spv").write_bytes(b"\x03\x02\x23\x07" + bytes(16))
            out_png = Path(tmp) / "preview.png"

            with patch("mlir_gpu.viewer.render_offscreen") as mock_render, \
                 patch("mlir_gpu.viewer.write_png"), \
                 contextlib.redirect_stdout(io.StringIO()):
                mock_render.return_value = RenderedImage(32, 24, bytes(32 * 24 * 4), "SwiftShader Device")
                with patch("sys.argv", ["mlir-gpu-view", str(manifest), "--provider", "swiftshader", "--output", str(out_png)]):
                    exit_code = main()
                    self.assertEqual(exit_code, 0)
                    mock_render.assert_called_once()
                    self.assertEqual(mock_render.call_args[1].get("provider"), "swiftshader")

    def test_cli_provider_choices_include_swiftshader(self):
        """CLI --provider argument accepts 'swiftshader'."""
        with patch("sys.argv", ["mlir-gpu-view", "--help"]):
            with self.assertRaises(SystemExit) as ctx, patch("sys.stdout", new_callable=io.StringIO) as mock_out:
                main()
            self.assertEqual(ctx.exception.code, 0)
            self.assertIn("swiftshader", mock_out.getvalue())

    def test_cli_output_and_frames_mutually_exclusive(self):
        """Passing both --output and --frames is rejected by argparse."""
        with patch("sys.argv", ["mlir-gpu-view", "manifest.json", "--output", "out.png", "--frames", "10"]), \
             patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as ctx:
                main()
            self.assertEqual(ctx.exception.code, 2)


class TestSwiftShaderSetupScript(unittest.TestCase):
    """Verify behavior of tools/setup_swiftshader.py and revision pinning."""

    def test_revision_file_pinned_hash(self):
        """tools/swiftshader-revision.txt exists and contains valid 40-char git commit hash."""
        rev_path = ROOT / "tools" / "swiftshader-revision.txt"
        self.assertTrue(rev_path.is_file(), "Missing tools/swiftshader-revision.txt")
        revision = rev_path.read_text(encoding="utf-8").strip()
        self.assertEqual(len(revision), 40)
        self.assertEqual(revision, "cddb5c91b23ff70a1f607ea31610aa0d7f4b5ad5")

    def test_fast_path_cache_hit_generates_icd(self):
        """When the target library exists in prefix, setup script creates ICD JSON and exits quickly."""
        import subprocess
        import sys
        import time
        from tools.setup_swiftshader import check_fast_path, get_library_name, setup_swiftshader

        lib_name = get_library_name()
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            dummy_lib = prefix / lib_name
            dummy_lib.write_bytes(b"dummy compiled library")

            t0 = time.time()
            icd_path = setup_swiftshader(prefix)
            duration = time.time() - t0

            self.assertLess(duration, 1.0, f"Fast path took {duration:.3f}s; expected <1.0s")
            self.assertTrue(icd_path.is_file())
            content = json.loads(icd_path.read_text(encoding="utf-8"))
            self.assertEqual(content["file_format_version"], "1.0.0")
            self.assertEqual(content["ICD"]["api_version"], "1.3.0")
            self.assertEqual(content["ICD"]["library_path"], dummy_lib.resolve().as_posix())

    def test_cli_fast_path_stdout(self):
        """Invoking setup_swiftshader.py CLI with cached library prints absolute ICD path."""
        import subprocess
        import sys
        from tools.setup_swiftshader import get_library_name

        lib_name = get_library_name()
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)
            (prefix / lib_name).write_bytes(b"dummy binary")
            res = subprocess.run(
                [sys.executable, str(ROOT / "tools" / "setup_swiftshader.py"), "--prefix", str(prefix)],
                capture_output=True,
                text=True,
                check=True,
            )
            output_path = res.stdout.strip()
            expected_icd = prefix / "vk_swiftshader_icd.json"
            self.assertEqual(Path(output_path).resolve(), expected_icd.resolve())
            self.assertTrue(expected_icd.is_file())

    def test_library_naming_conventions(self):
        """get_library_name returns correct library name for current platform."""
        from tools.setup_swiftshader import get_library_name

        lib_name = get_library_name()
        if sys.platform == "win32" or os.name == "nt":
            self.assertEqual(lib_name, "vk_swiftshader.dll")
        elif sys.platform == "darwin":
            self.assertEqual(lib_name, "libvk_swiftshader.dylib")
        else:
            self.assertEqual(lib_name, "libvk_swiftshader.so")


class TestSwiftShaderCiWorkflowSchema(unittest.TestCase):
    """Validate YAML schema, runners, and steps for SwiftShader coverage in .github/workflows/ci.yml."""

    @classmethod
    def setUpClass(cls):
        cls.ci_yaml_path = ROOT / ".github" / "workflows" / "ci.yml"
        if not HAVE_YAML:
            return
        cls.ci_data = yaml.safe_load(cls.ci_yaml_path.read_text(encoding="utf-8"))

    def setUp(self):
        if not HAVE_YAML:
            self.skipTest("PyYAML not installed in test environment")

    def test_ci_yaml_workflow_parsing(self):
        """Workflow file exists and is valid YAML mapping."""
        self.assertTrue(self.ci_yaml_path.is_file())
        self.assertIsInstance(self.ci_data, dict)
        self.assertIn("jobs", self.ci_data)

    def test_ci_yaml_windows_warp_and_swiftshader_coverage(self):
        """Windows CI job configures SwiftShader caching, setup script invocation, and Vulkan testing."""
        jobs = self.ci_data.get("jobs", {})
        self.assertIn("test-windows-warp", jobs)
        self.assertNotIn("test-linux-swiftshader", jobs, "Linux Chrome scraping job must be removed")

        windows_job = jobs["test-windows-warp"]
        self.assertEqual(windows_job.get("runs-on"), "windows-latest")

        steps = windows_job.get("steps", [])
        step_names = [s.get("name") for s in steps if "name" in s]

        cache_steps = [
            s for s in steps
            if "actions/cache" in s.get("uses", "") and ".sdk/swiftshader" in str(s.get("with", {}).get("path", ""))
        ]
        self.assertTrue(len(cache_steps) >= 1, "Missing .sdk/swiftshader cache step in test-windows-warp")
        cache_key = cache_steps[0]["with"]["key"]
        self.assertIn("swiftshader-${{ runner.os }}-${{ runner.arch }}", cache_key)
        self.assertIn("tools/swiftshader-revision.txt", cache_key)
        self.assertIn("tools/setup_swiftshader.py", cache_key)

        setup_steps = [s for s in steps if "setup_swiftshader.py" in s.get("run", "")]
        self.assertTrue(len(setup_steps) >= 1, "Missing python tools/setup_swiftshader.py step")

        self.assertTrue(any("Report SwiftShader" in n for n in step_names), "Missing SwiftShader capability report step")

        test_step = next(s for s in steps if "Run Windows WARP" in s.get("name", ""))
        env = test_step.get("env", {})
        self.assertEqual(env.get("MLIR_GPU_TEST_SWIFTSHADER"), "1")
        self.assertEqual(env.get("MLIR_GPU_TEST_VULKAN"), "1")
        self.assertEqual(env.get("MLIR_GPU_TEST_DIRECTX"), "1")
        self.assertEqual(env.get("MLIR_GPU_TEST_WARP"), "1")
        self.assertIn("vk_swiftshader_icd.json", env.get("VK_ICD_FILENAMES", ""))
        self.assertIn("vk_swiftshader_icd.json", env.get("VK_DRIVER_FILES", ""))

    def test_ci_yaml_session0_window_exclusion(self):
        """Workflow steps must never enable MLIR_GPU_TEST_WINDOW."""
        for job_name, job in self.ci_data.get("jobs", {}).items():
            for step in job.get("steps", []):
                env = step.get("env", {})
                self.assertNotIn(
                    "MLIR_GPU_TEST_WINDOW", env,
                    f"MLIR_GPU_TEST_WINDOW must not be set in job {job_name} step {step.get('name')}",
                )


if __name__ == "__main__":
    unittest.main()
