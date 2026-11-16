"""GPU constructor values, transfer boundaries, and ownership checks."""

from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

import gfxpy as gp
from test_gfxpy_api import _SpyBackend


class _ConstructorBackend(_SpyBackend):
    def fill(self, out, value, length):
        out._data[:] = struct.pack("<f", value) * length
        self.dispatches.append("fill")

    def copy_buffer(self, src, out, length):
        out._data[:] = src._data
        self.dispatches.append("copy")


class TestConstructors(unittest.TestCase):
    def setUp(self):
        self.backend = _ConstructorBackend()
        self.dev = gp.device("vulkan", backend=self.backend)
        self.addCleanup(self.dev.close)

    def test_fill_and_like_variants_dispatch_without_uploading_host_arrays(self):
        with mock.patch.dict(sys.modules, {"numpy": None}):
            source = gp.full((3, 37), 2.5, device=self.dev)
            arrays = [
                (source, 2.5),
                (gp.zeros((3, 37), device=self.dev), 0.0),
                (gp.ones((3, 37), device=self.dev), 1.0),
                (gp.zeros_like(source), 0.0),
                (gp.ones_like(source), 1.0),
                (gp.full_like(source, -4.25), -4.25),
            ]
        self.assertEqual(self.backend.uploads, 0)
        self.assertEqual(self.backend.downloads, 0)
        self.assertEqual(self.backend.dispatches, ["fill"] * 6)
        for arr, value in arrays:
            self.assertEqual(arr.shape, (3, 37))
            self.assertIs(arr.device, self.dev)
            self.assertIs(arr.dtype, gp.float32)
            self.assertEqual(arr.tolist(), [[value] * 37] * 3)

    def test_empty_shapes_need_no_buffer_or_dispatch(self):
        for shape in ((0,), (3, 0), (0, 7)):
            source = gp.zeros(shape, device=self.dev)
            for result in (gp.ones_like(source), gp.copy(source)):
                self.assertEqual(result.shape, shape)
                self.assertEqual(result.size, 0)
                self.assertIsNot(result, source)
        self.assertEqual(self.backend.allocations, [])
        self.assertEqual(self.backend.dispatches, [])

    def test_asarray_reuses_storage_and_copy_is_independent_without_readback(self):
        source = gp.asarray([[1.0, -0.0], [float("inf"), float("nan")]], device=self.dev)
        self.assertIs(gp.asarray(source, dtype="float32"), source)
        result = gp.copy(source)
        self.assertEqual(self.backend.uploads, 1)
        self.assertEqual(self.backend.downloads, 0)
        self.assertEqual(self.backend.dispatches, ["copy"])
        self.assertIsNot(result._buffer, source._buffer)
        expected = source.tobytes()
        source.close()
        self.assertEqual(result.tobytes(), expected)
        self.assertEqual(result.shape, (2, 2))
        uploaded = gp.copy([3.0, 4.0], device=self.dev)
        self.assertEqual(uploaded.tolist(), [3.0, 4.0])

    def test_invalid_inputs_and_limits_fail_before_allocation(self):
        for make, error in (
            (lambda: gp.full(3, True, device=self.dev), TypeError),
            (lambda: gp.full(3, 1e99, device=self.dev), OverflowError),
            (lambda: gp.zeros((1, 2, 3), device=self.dev), gp.ShapeMismatchError),
            (lambda: gp.ones(2, dtype="float64", device=self.dev), gp.UnsupportedDTypeError),
            (lambda: gp.zeros((0, gp.I32_MAX + 1), device=self.dev), gp.IndexingLimitError),
        ):
            with self.assertRaises(error):
                make()
        self.backend.limits = gp.ComputeLimits(max_workgroup_count=(1, 1, 1))
        with self.assertRaises(gp.IndexingLimitError):
            gp.zeros(65, device=self.dev)
        self.assertEqual(self.backend.allocations, [])

    def test_closed_and_cross_device_inputs_never_fall_back_to_host(self):
        source = gp.ones(3, device=self.dev)
        other = gp.device("vulkan", backend=_ConstructorBackend())
        self.addCleanup(other.close)
        with self.assertRaises(gp.CrossDeviceError):
            gp.asarray(source, device=other)
        with self.assertRaises(gp.CrossDeviceError):
            gp.copy(source, device=other)
        source.close()
        for make in (gp.zeros_like, gp.asarray, gp.copy):
            with self.assertRaises(gp.ArrayClosedError):
                make(source)
        self.assertEqual(self.backend.downloads, 0)

    def test_dispatch_failures_release_only_the_new_buffer(self):
        with mock.patch.object(self.backend, "fill", side_effect=RuntimeError("fill failed")):
            with self.assertRaisesRegex(RuntimeError, "fill failed"):
                gp.ones(3, device=self.dev)
        self.assertTrue(self.backend.buffers[-1].closed)
        source = gp.ones(3, device=self.dev)
        with mock.patch.object(self.backend, "copy_buffer", side_effect=RuntimeError("copy failed")):
            with self.assertRaisesRegex(RuntimeError, "copy failed"):
                gp.copy(source)
        self.assertFalse(source.closed)
        self.assertTrue(self.backend.buffers[-1].closed)


class _LiveConstructorMixin:
    target: str
    provider: str | None = None

    def test_fill_copy_values_nonfinite_and_pipeline_reuse(self):
        dev = gp.device(self.target, provider=self.provider)
        self.addCleanup(dev.close)
        with dev, mock.patch.dict(sys.modules, {"numpy": None}):
            source = gp.full((3, 37), 2.5)
            self.assertEqual(gp.zeros_like(source).tolist(), [[0.0] * 37] * 3)
            self.assertEqual(gp.ones_like(source).tolist(), [[1.0] * 37] * 3)
            self.assertEqual(gp.full_like(source, -3.25).tolist(), [[-3.25] * 37] * 3)
            result = gp.copy(source)
            self.assertIs(gp.asarray(source), source)
            self.assertEqual(result.tobytes(), source.tobytes())
            source.close()
            self.assertEqual(result.tolist(), [[2.5] * 37] * 3)
            for value in (float("inf"), float("-inf"), float("nan"), -0.0):
                original = gp.full(137, value)
                copied = gp.copy(original)
                self.assertEqual(copied.tobytes(), original.tobytes())
                got = copied.tolist()
                for item in got:
                    if math.isnan(value):
                        self.assertTrue(math.isnan(item))
                    else:
                        self.assertEqual(item, value)
                        self.assertEqual(math.copysign(1.0, item), math.copysign(1.0, value))
            self.assertEqual(dev.backend.compiled_pipeline_count, 2)


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_NACELLE") == "1", "Set MLIR_GPU_TEST_NACELLE=1")
class TestConstructorsNacelleLive(_LiveConstructorMixin, unittest.TestCase):
    target = "directx"
    provider = "nacelle"


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_VULKAN") == "1", "Set MLIR_GPU_TEST_VULKAN=1")
class TestConstructorsVulkanLive(_LiveConstructorMixin, unittest.TestCase):
    target = "vulkan"


@unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_METAL") == "1", "Set MLIR_GPU_TEST_METAL=1")
class TestConstructorsMetalLive(_LiveConstructorMixin, unittest.TestCase):
    target = "metal"


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1", "Set MLIR_GPU_TEST_DIRECTX=1 on Windows",
)
class TestConstructorsDirectXLive(_LiveConstructorMixin, unittest.TestCase):
    target = "directx"
