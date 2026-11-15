"""Cross-backend execution and host validation tests for the NumPy-free gfxpy example."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402
from gfxpy.chain_example import _CountingBackend, _default_toolchain  # noqa: E402
from gfxpy.gpu_backend import GfxIRComputeBackend  # noqa: E402
from mlir_gpu import ComputeContext, PersistentBuffer, PersistentComputeError, PersistentPipeline  # noqa: E402
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
                ctx = raw_backend._ctx
                provider_label = getattr(ctx, "provider", None) or (
                    "software" if getattr(ctx, "is_software", False) else "hardware"
                )
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
