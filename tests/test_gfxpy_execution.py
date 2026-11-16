"""Cross-backend execution and host validation tests for the NumPy-free gfxpy example."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.append(str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402
from gfxpy.chain_example import _CountingBackend, _default_toolchain, format_provider_label  # noqa: E402
from gfxpy.gpu_backend import GfxIRComputeBackend  # noqa: E402
from mlir_gpu import ComputeContext, PersistentBuffer, PersistentComputeError, PersistentPipeline, VulkanComputeContext  # noqa: E402
from mlir_gpu.viewer import ComputeProgram, Shader  # noqa: E402


class _StubBuffer(PersistentBuffer):
    def __init__(self, context: _StubContext, size_bytes: int) -> None:
        self._context = context
        self._data = bytearray(size_bytes)
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return len(self._data)

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    @property
    def context(self) -> _StubContext:
        return self._context

    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        raw = bytes(data)
        self._data[offset : offset + len(raw)] = raw

    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        end = len(self._data) if size_bytes is None else offset + size_bytes
        return bytes(self._data[offset:end])

    def close(self) -> None:
        self._closed = True


class _StubPipeline(PersistentPipeline):
    def __init__(self, context: _StubContext, program: ComputeProgram) -> None:
        self._context = context
        self._program = program
        self._closed = False

    @property
    def program(self) -> ComputeProgram:
        return self._program

    @property
    def context(self) -> _StubContext:
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    def dispatch(self, buffers, parameters=None, grid=None) -> None:
        pass

    def close(self) -> None:
        self._closed = True


class _StubContext(ComputeContext):
    def __init__(self, target: str = "vulkan") -> None:
        self._target = target
        self._closed = False
        self.compiled_count = 0

    @property
    def target(self) -> str:
        return self._target

    @property
    def device_name(self) -> str:
        return "StubDevice"

    @property
    def closed(self) -> bool:
        return self._closed

    def create_buffer(self, size_bytes: int, initial_data=None) -> _StubBuffer:
        buf = _StubBuffer(self, size_bytes)
        if initial_data is not None:
            buf.upload(initial_data)
        return buf

    def compile_pipeline(self, program) -> _StubPipeline:
        self.compiled_count += 1
        dummy = ComputeProgram(compute=Shader(entry_point="main", code=b"stub"), target=self._target)
        return _StubPipeline(self, dummy)

    def synchronize(self) -> None:
        pass

    def close(self) -> None:
        self._closed = True


class TestGfxPyBackendHost(unittest.TestCase):
    """Host-only tests for GfxIRComputeBackend caching, error handling, and NumPy-free import."""

    def test_pipeline_cache_reuses_compiled_kernels_without_numpy(self) -> None:
        ctx = _StubContext("vulkan")
        with mock.patch.dict(sys.modules, {"numpy": None}), mock.patch(
            "gfxpy.gpu_backend._open_context", return_value=ctx
        ), mock.patch(
            "gfxpy.gpu_backend.compile_pipeline", return_value="dummy_manifest.json"
        ) as compile_mock:
            raw_backend = GfxIRComputeBackend("vulkan")
            counted = _CountingBackend(raw_backend)
            dev = gp.device("vulkan", backend=counted)
            try:
                with dev:
                    a = gp.array([1.0, 2.0, 3.0])
                    b = gp.array([4.0, 5.0, 6.0])
                    _ = ((a + b) + (a + b)).tolist()
                    self.assertEqual(compile_mock.call_count, 1)
                    self.assertEqual(ctx.compiled_count, 1)
                    self.assertEqual(raw_backend.compiled_pipeline_count, 1)
                    self.assertEqual(counted.downloads, 1)
            finally:
                dev.close()

    def test_missing_device_or_tools_raises_without_cpu_fallback(self) -> None:
        with mock.patch(
            "gfxpy.gpu_backend._open_context",
            side_effect=PersistentComputeError("No GPU device available"),
        ):
            with self.assertRaises(gp.BackendInitializationError) as ctx:
                gp.device("vulkan")
            self.assertIn("No GPU device available", str(ctx.exception))
            self.assertIn("No fallback", str(ctx.exception))

    def test_implicit_operations_reuse_context_and_compiled_pipelines(self) -> None:
        ctx = _StubContext("vulkan")
        gp.close_default_device()
        try:
            with (
                mock.patch.object(sys, "platform", "linux"),
                mock.patch("gfxpy.gpu_backend._open_context", return_value=ctx) as open_ctx,
                mock.patch(
                    "gfxpy.gpu_backend.compile_pipeline",
                    return_value="dummy_manifest.json",
                ) as compile_mock,
            ):
                a = gp.array([1.0, 2.0, 3.0])
                b = gp.array([4.0, 5.0, 6.0])
                c = gp.empty(3)
                self.assertIs(a.device, b.device)
                self.assertIs(a.device, c.device)

                _ = ((a + b) * 2.0 + (a + b)).tolist()
                d = gp.array([7.0, 8.0, 9.0])
                _ = (d + a).tolist()

                self.assertEqual(open_ctx.call_count, 1)
                self.assertEqual(compile_mock.call_count, 2)
                self.assertEqual(ctx.compiled_count, 2)
                self.assertEqual(gp.current_device().backend.compiled_pipeline_count, 2)
        finally:
            gp.close_default_device()

    def test_provider_label_is_unknown_when_metadata_absent(self) -> None:
        self.assertEqual(format_provider_label(_StubContext("vulkan")), "unknown")

    def test_vulkan_compute_context_classifies_software_and_hardware_providers(self) -> None:
        cases = [
            ("llvmpipe (LLVM 15.0.7, 256 bits)", 0x10005, 0x0, 4, "lavapipe", True),
            ("SwiftShader Device (Subzero)", 0x1AE0, 0xC0DE, 4, "swiftshader", True),
            ("Generic CPU Vulkan", 0x9999, 0x1, 4, "software", True),
            ("NVIDIA GeForce RTX 4090", 0x10DE, 0x2684, 2, "hardware", False),
        ]
        for dev_name, vendor_id, device_id, dev_type, expected_provider, expected_sw in cases:
            with self.subTest(dev_name=dev_name):
                fake_props = SimpleNamespace(
                    deviceName=dev_name.encode("utf-8"),
                    vendorID=vendor_id,
                    deviceID=device_id,
                    deviceType=dev_type,
                    limits=SimpleNamespace(
                        maxFramebufferWidth=16384,
                        maxFramebufferHeight=16384,
                        maxPushConstantsSize=256,
                        maxComputeWorkGroupCount=(65535, 65535, 65535),
                        maxComputeWorkGroupSize=(1024, 1024, 64),
                        maxComputeWorkGroupInvocations=1024,
                    ),
                )
                fake_queue = SimpleNamespace(queueCount=1, queueFlags=0x2)
                fake_vk = mock.MagicMock()
                fake_vk.VK_PHYSICAL_DEVICE_TYPE_CPU = 4
                fake_vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU = 2
                fake_vk.VK_QUEUE_COMPUTE_BIT = 0x2
                fake_vk.VK_QUEUE_GRAPHICS_BIT = 0x1
                fake_vk.VK_API_VERSION_1_0 = 1 << 22
                fake_vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
                fake_vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = 0x2
                fake_vk.VK_SUCCESS = 0
                fake_vk.VkError = RuntimeError
                fake_vk.VkException = RuntimeError
                fake_vk.ProcedureNotFoundError = RuntimeError
                fake_vk.ExtensionNotSupportedError = RuntimeError
                fake_vk.vkEnumerateInstanceExtensionProperties.return_value = []
                fake_vk.vkEnumeratePhysicalDevices.return_value = ["phys"]
                fake_vk.vkGetPhysicalDeviceProperties.return_value = fake_props
                fake_vk.vkGetPhysicalDeviceQueueFamilyProperties.return_value = [fake_queue]
                fake_vk.vkEnumerateDeviceExtensionProperties.return_value = []
                fake_vk.vkAllocateCommandBuffers.return_value = ["cmd"]
                with (
                    mock.patch.dict(os.environ, {"MLIR_GPU_TEST_SWIFTSHADER": "0"}),
                    mock.patch("mlir_gpu.viewer._binding", return_value=fake_vk),
                ):
                    ctx = VulkanComputeContext()
                    try:
                        self.assertEqual(ctx.provider, expected_provider)
                        self.assertIs(ctx.is_software, expected_sw)
                        self.assertEqual(format_provider_label(ctx), expected_provider)
                    finally:
                        ctx.close()

    def test_clean_subprocess_execution_without_numpy(self) -> None:
        script = f"""
import importlib.abc
import json
import sys
from unittest import mock

class _BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ModuleNotFoundError(f"Blocked NumPy import: {{fullname}}")
        return None

sys.meta_path.insert(0, _BlockNumpy())
sys.path.append({str(ROOT / "src")!r})
sys.path.append({str(EXAMPLES_DIR)!r})

import gfxpy as gp
import gfxpy.chain_example as chain_example
from mlir_gpu import ComputeContext, PersistentBuffer, PersistentPipeline
from mlir_gpu.viewer import ComputeProgram, Shader

class _Buf(PersistentBuffer):
    def __init__(self, ctx, n):
        self._ctx = ctx
        self._data = bytearray(n)
        self._closed = False
    @property
    def size_bytes(self): return len(self._data)
    @property
    def closed(self): return self._closed
    @property
    def context(self): return self._ctx
    def upload(self, data, offset=0):
        raw = bytes(data)
        self._data[offset:offset + len(raw)] = raw
    def download(self, size_bytes=None, offset=0):
        return bytes(self._data[offset:] if size_bytes is None else self._data[offset:offset + size_bytes])
    def close(self): self._closed = True

class _Pipe(PersistentPipeline):
    def __init__(self, ctx, prog):
        self._ctx = ctx
        self._prog = prog
        self._closed = False
    @property
    def program(self): return self._prog
    @property
    def context(self): return self._ctx
    @property
    def closed(self): return self._closed
    def dispatch(self, buffers, parameters=None, grid=None):
        import struct
        lhs = struct.unpack("<2f", buffers["lhs"].download(8))
        scalar = float(parameters["scalar"])
        buffers["out"].upload(struct.pack("<2f", *(x + scalar for x in lhs)))
    def close(self): self._closed = True

class _Ctx(ComputeContext):
    def __init__(self): self._closed = False
    @property
    def target(self): return "vulkan"
    @property
    def device_name(self): return "CleanEnvStub"
    @property
    def provider(self): return "lavapipe"
    @property
    def is_software(self): return True
    @property
    def closed(self): return self._closed
    def create_buffer(self, size_bytes, initial_data=None):
        b = _Buf(self, size_bytes)
        if initial_data is not None: b.upload(initial_data)
        return b
    def compile_pipeline(self, program):
        return _Pipe(self, ComputeProgram(compute=Shader(entry_point="main", code=b"stub"), target="vulkan"))
    def synchronize(self): pass
    def close(self): self._closed = True

with mock.patch("gfxpy.gpu_backend._open_context", return_value=_Ctx()), mock.patch(
    "gfxpy.gpu_backend.compile_pipeline", return_value="dummy.json"
):
    with gp.device("vulkan") as dev:
        res = (gp.array([1.0, 2.0]) + 3.0).tolist()
        assert res == [4.0, 5.0], res
        assert chain_example.format_provider_label(dev._backend._ctx) == "lavapipe"

print(json.dumps({{"numpy_loaded": "numpy" in sys.modules, "result": res}}))
"""
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}")
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertFalse(payload["numpy_loaded"])
        self.assertEqual(payload["result"], [4.0, 5.0])


def _run_live_chain_check(
    test_case: unittest.TestCase,
    *,
    target: str,
    provider: str | None,
    validation: bool,
) -> None:
    length = 137  # Non-workgroup-aligned (workgroup_size=64)
    a_vals = [float(i) * 0.25 - 5.0 for i in range(length)]
    b_vals = [float((i % 7) + 1) * 0.5 for i in range(length)]
    c_vals = [float(i % 11) * -0.75 for i in range(length)]
    expected = [((a * b + 1.5) * 2.0 + c) for a, b, c in zip(a_vals, b_vals, c_vals)]

    with mock.patch.dict(sys.modules, {"numpy": None}):
        raw_backend = GfxIRComputeBackend(
            target,
            provider=provider,
            validation=validation,
            toolchain=_default_toolchain(),
        )
        counted = _CountingBackend(raw_backend)
        dev = gp.device(target, backend=counted)
        try:
            with dev:
                provider_label = format_provider_label(raw_backend._ctx)
                if provider == "nacelle":
                    test_case.assertEqual(provider_label, "nacelle")
                    test_case.assertTrue(raw_backend._ctx.is_software)
                print(
                    f"[GfxPyLive] target={dev.target} provider={provider_label} "
                    f"device={raw_backend.device_name!r}"
                )

                a = gp.array(a_vals, dtype=gp.float32)
                b = gp.array(b_vals, dtype=gp.float32)
                c = gp.array(c_vals, dtype=gp.float32)
                y = (a * b + 1.5) * 2.0 + c

                test_case.assertEqual(counted.downloads, 0)
                actual = y.tolist()
                test_case.assertEqual(counted.downloads, 1)
                test_case.assertEqual(
                    counted.dispatches,
                    ["binary:mul", "scalar:add", "scalar:mul", "binary:add"],
                )

                for idx, (got, exp) in enumerate(zip(actual, expected)):
                    test_case.assertAlmostEqual(
                        got,
                        exp,
                        places=4,
                        msg=f"Mismatch at index {idx}: got {got}, expected {exp}",
                    )

                empty_res = (gp.array([]) * 2.0 + 1.0).tolist()
                test_case.assertEqual(empty_res, [])
                test_case.assertEqual(counted.downloads, 1)
        finally:
            dev.close()


class TestGfxPyLiveExecution(unittest.TestCase):
    """Opt-in live GPU execution checks for DirectX 12, Vulkan, and Metal."""

    @unittest.skipUnless(
        sys.platform in ("linux", "darwin") and os.environ.get("MLIR_GPU_TEST_NACELLE") == "1",
        "Set MLIR_GPU_TEST_NACELLE=1 on Linux/macOS to run live Nacelle execution tests",
    )
    def test_nacelle_chained_execution(self) -> None:
        _run_live_chain_check(self, target="directx", provider="nacelle", validation=False)

    @unittest.skipUnless(
        sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
        "Opt-in Windows DirectX 12 check (set MLIR_GPU_TEST_DIRECTX=1)",
    )
    def test_directx_chained_execution(self) -> None:
        provider = "warp" if os.environ.get("MLIR_GPU_TEST_WARP") == "1" else "hardware"
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        _run_live_chain_check(self, target="directx", provider=provider, validation=validation)

    @unittest.skipUnless(
        os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
        "Opt-in Vulkan check (set MLIR_GPU_TEST_VULKAN=1)",
    )
    def test_vulkan_chained_execution(self) -> None:
        provider = "swiftshader" if os.environ.get("MLIR_GPU_TEST_SWIFTSHADER") == "1" else None
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        _run_live_chain_check(self, target="vulkan", provider=provider, validation=validation)

    @unittest.skipUnless(
        sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
        "Opt-in macOS Metal check (set MLIR_GPU_TEST_METAL=1)",
    )
    def test_metal_chained_execution(self) -> None:
        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        _run_live_chain_check(self, target="metal", provider=None, validation=validation)

    @unittest.skipUnless(
        (sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1")
        or (sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1")
        or (sys.platform.startswith("linux") and os.environ.get("MLIR_GPU_TEST_VULKAN") == "1"),
        "Opt-in live platform-default check (set MLIR_GPU_TEST_METAL=1, MLIR_GPU_TEST_DIRECTX=1, or MLIR_GPU_TEST_VULKAN=1)",
    )
    def test_implicit_default_device_execution(self) -> None:
        gp.close_default_device()
        try:
            a = gp.array([1.0, 2.0, 3.0], dtype=gp.float32)
            b = gp.array([4.0, 5.0, 6.0], dtype=gp.float32)
            e = gp.empty(3, dtype=gp.float32)
            self.assertIs(a.device, b.device)
            self.assertIs(a.device, e.device)
            self.assertEqual((a + b).tolist(), [5.0, 7.0, 9.0])
            self.assertEqual((a * b).tolist(), [4.0, 10.0, 18.0])
        finally:
            gp.close_default_device()
