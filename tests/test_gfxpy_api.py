"""Host-only unit tests for the NumPy-free gfxpy array API contract."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402


class _SpyBuffer:
    def __init__(self, size_bytes: int) -> None:
        self._data = bytearray(size_bytes)
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return len(self._data)

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        self._closed = True


class _SpyBackend:
    """In-memory test double verifying Device/GfxArray contract calls."""

    def __init__(self, target: str = "vulkan") -> None:
        self._target = target
        self._closed = False
        self.allocations: list[int] = []
        self.uploads: int = 0
        self.downloads: int = 0
        self.dispatches: list[str] = []
        self.buffers: list[_SpyBuffer] = []

    @property
    def target(self) -> str:
        return self._target

    @property
    def closed(self) -> bool:
        return self._closed

    def allocate(self, size_bytes: int) -> _SpyBuffer:
        if size_bytes <= 0:
            raise AssertionError(f"Zero-byte or negative allocation requested: {size_bytes}")
        buf = _SpyBuffer(size_bytes)
        self.allocations.append(size_bytes)
        self.buffers.append(buf)
        return buf

    def upload(self, buf: _SpyBuffer, data: bytes) -> None:
        if len(data) != buf.size_bytes:
            raise AssertionError("Upload size mismatch")
        buf._data[:] = data
        self.uploads += 1

    def download(self, buf: _SpyBuffer) -> bytes:
        self.downloads += 1
        return bytes(buf._data)

    def binary_op(self, op: str, lhs: _SpyBuffer, rhs: _SpyBuffer, out: _SpyBuffer, length: int) -> None:
        a = struct.unpack(f"<{length}f", lhs._data)
        b = struct.unpack(f"<{length}f", rhs._data)
        if op == "add":
            vals = [x + y for x, y in zip(a, b)]
        elif op == "mul":
            vals = [x * y for x, y in zip(a, b)]
        else:
            raise ValueError(f"Unsupported op: {op}")
        out._data[:] = struct.pack(f"<{length}f", *vals)
        self.dispatches.append(f"binary:{op}")

    def scalar_op(self, op: str, lhs: _SpyBuffer, scalar: float, out: _SpyBuffer, length: int) -> None:
        a = struct.unpack(f"<{length}f", lhs._data)
        if op == "add":
            vals = [x + scalar for x in a]
        elif op == "mul":
            vals = [x * scalar for x in a]
        else:
            raise ValueError(f"Unsupported op: {op}")
        out._data[:] = struct.pack(f"<{length}f", *vals)
        self.dispatches.append(f"scalar:{op}")

    def close(self) -> None:
        self._closed = True
        for buf in self.buffers:
            buf.close()


class TestTargetSelection(unittest.TestCase):
    def test_platform_defaults(self) -> None:
        self.assertEqual(gp.default_target("darwin"), "metal")
        self.assertEqual(gp.default_target("win32"), "directx")
        self.assertEqual(gp.default_target("linux"), "vulkan")
        self.assertEqual(gp.default_target("linux2"), "vulkan")
        with self.assertRaises(gp.UnsupportedTargetError):
            gp.default_target("freebsd14")

    def test_explicit_override_and_rejection(self) -> None:
        self.assertEqual(gp.normalize_target("Vulkan"), "vulkan")
        self.assertEqual(gp.normalize_target("METAL"), "metal")
        self.assertEqual(gp.normalize_target("directx"), "directx")
        self.assertEqual(gp.normalize_target("auto", platform_name="darwin"), "metal")
        with self.assertRaises(gp.UnsupportedTargetError):
            gp.normalize_target("cuda")
        with self.assertRaises(gp.UnsupportedTargetError):
            gp.normalize_target(123)  # type: ignore[arg-type]

    def test_backend_init_failure_surfaces_target_and_cause(self) -> None:
        def failing_factory(target: str):
            raise RuntimeError(f"missing {target} driver")

        with self.assertRaises(gp.BackendInitializationError) as ctx:
            gp.device("metal", backend_factory=failing_factory)
        self.assertEqual(ctx.exception.target, "metal")
        self.assertIn("metal", str(ctx.exception))
        self.assertIn("missing metal driver", str(ctx.exception))
        self.assertIn("No fallback", str(ctx.exception))

    def test_uninitialized_backend_does_not_fall_back_to_cpu(self) -> None:
        from gfxpy.runtime import UninitializedBackend

        with gp.device("vulkan", backend=UninitializedBackend("vulkan")) as dev:
            self.assertEqual(dev.target, "vulkan")
            with self.assertRaises(gp.BackendNotImplementedError) as ctx:
                gp.array([1.0, 2.0])
            self.assertIn("vulkan", str(ctx.exception))
            self.assertIn("No CPU fallback", str(ctx.exception))


class TestArrayContract(unittest.TestCase):
    def test_import_and_execute_without_numpy(self) -> None:
        backend = _SpyBackend("vulkan")
        with mock.patch.dict(sys.modules, {"numpy": None}):
            with gp.device("vulkan", backend=backend):
                x = gp.array([1.0, 2.0, 3.0], dtype=gp.float32)
                y = x * 2.0 + 1.0
                self.assertEqual(y.shape, (3,))
                self.assertEqual(y.ndim, 1)
                self.assertEqual(y.size, 3)
                self.assertEqual(y.nbytes, 12)
                self.assertEqual(y.dtype, gp.float32)
                self.assertEqual(backend.downloads, 0)
                self.assertEqual(y.tolist(), [3.0, 5.0, 7.0])
                self.assertEqual(backend.downloads, 1)
                self.assertEqual(backend.dispatches, ["scalar:mul", "scalar:add"])

    def test_reverse_scalar_and_binary_ops(self) -> None:
        backend = _SpyBackend("metal")
        with gp.device("metal", backend=backend):
            a = gp.array((1, 2, 4))
            b = gp.array(range(3))
            self.assertEqual((a + b).tolist(), [1.0, 3.0, 6.0])
            self.assertEqual((a * b).tolist(), [0.0, 2.0, 8.0])
            self.assertEqual((2.0 * a).tolist(), [2.0, 4.0, 8.0])
            self.assertEqual((10.0 + b).tolist(), [10.0, 11.0, 12.0])

    def test_empty_arrays_avoid_zero_byte_allocations_and_dispatches(self) -> None:
        backend = _SpyBackend("directx")
        with gp.device("directx", backend=backend):
            e1 = gp.array([])
            e2 = gp.empty(0)
            e3 = e1 * 2.0 + e2
            self.assertEqual(e1.shape, (0,))
            self.assertEqual(e3.tolist(), [])
            self.assertEqual(backend.allocations, [])
            self.assertEqual(backend.dispatches, [])
            self.assertEqual(backend.uploads, 0)
            self.assertEqual(backend.downloads, 0)

    def test_validation_rejects_bad_shapes_dtypes_and_types(self) -> None:
        backend = _SpyBackend("vulkan")
        with gp.device("vulkan", backend=backend):
            with self.assertRaises(gp.UnsupportedDTypeError):
                gp.array([1.0], dtype="float64")
            with self.assertRaises(gp.UnsupportedDTypeError):
                gp.empty(4, dtype="int32")
            with self.assertRaises(gp.ShapeMismatchError):
                gp.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # type: ignore[arg-type]
            with self.assertRaises(gp.ShapeMismatchError):
                gp.array([[1.0], [2.0, 3.0]])
            with self.assertRaises(TypeError):
                gp.array("123")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                gp.array(b"123")  # type: ignore[arg-type]
            with self.assertRaises(TypeError):
                gp.array([True, False])
            with self.assertRaises(ValueError):
                gp.empty(-1)
            with self.assertRaises(TypeError):
                gp.empty(True)

            a = gp.array([1.0, 2.0])
            b = gp.array([1.0, 2.0, 3.0])
            with self.assertRaises(gp.ShapeMismatchError):
                _ = a + b
            with self.assertRaises(TypeError):
                _ = a + True  # type: ignore[operator]

    def test_ownership_cross_device_and_use_after_close(self) -> None:
        b1 = _SpyBackend("vulkan")
        b2 = _SpyBackend("metal")
        d1 = gp.device("vulkan", backend=b1)
        d2 = gp.device("metal", backend=b2)

        a1 = gp.array([1.0, 2.0], device=d1)
        a2 = gp.array([3.0, 4.0], device=d2)
        with self.assertRaises(gp.CrossDeviceError):
            _ = a1 + a2

        a1.close()
        self.assertTrue(a1.closed)
        with self.assertRaises(gp.ArrayClosedError):
            a1.tolist()
        with self.assertRaises(gp.ArrayClosedError):
            a1.tobytes()
        with self.assertRaises(gp.ArrayClosedError):
            a1.toarray()
        with self.assertRaises(gp.ArrayClosedError):
            _ = a1 + 1.0

        d1.close()
        self.assertTrue(d1.closed)
        self.assertTrue(b1.closed)
        with self.assertRaises(gp.DeviceClosedError):
            gp.empty(4, device=d1)

    def test_tobytes_toarray_and_tolist_1d_2d_empty(self) -> None:
        import array as stdlib_array
        import struct

        backend = _SpyBackend("vulkan")
        with gp.device("vulkan", backend=backend):
            a1 = gp.array([1.5, -2.0, 3.25])
            self.assertEqual(a1.tobytes(), struct.pack("<3f", 1.5, -2.0, 3.25))
            arr1 = a1.toarray()
            self.assertIsInstance(arr1, stdlib_array.array)
            self.assertEqual(arr1.typecode, "f")
            self.assertEqual(arr1.tolist(), [1.5, -2.0, 3.25])
            self.assertEqual(a1.tolist(), [1.5, -2.0, 3.25])

            a2 = gp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            self.assertEqual(a2.tobytes(), struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
            self.assertEqual(a2.toarray().tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            self.assertEqual(a2.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            empty_1d = gp.empty(0)
            self.assertEqual(empty_1d.tobytes(), b"")
            self.assertEqual(empty_1d.toarray().tolist(), [])
            self.assertEqual(empty_1d.tolist(), [])

            empty_2d = gp.empty((2, 0))
            self.assertEqual(empty_2d.tobytes(), b"")
            self.assertEqual(empty_2d.toarray().tolist(), [])
            self.assertEqual(empty_2d.tolist(), [[], []])

            mv1 = a1.tomemoryview()
            self.assertEqual(mv1.shape, (3,))
            self.assertEqual(mv1.tolist(), [1.5, -2.0, 3.25])

            mv2 = a2.tomemoryview()
            self.assertEqual(mv2.shape, (2, 3))
            self.assertEqual(mv2.tolist(), [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

            dst = bytearray(24)
            self.assertEqual(a2.download_into(dst), 24)
            self.assertEqual(bytes(dst), struct.pack("<6f", 1.0, 2.0, 3.0, 4.0, 5.0, 6.0))

            with self.assertRaises(TypeError):
                a1.download_into(memoryview(b"0" * 12))
            with self.assertRaises(ValueError):
                a1.download_into(bytearray(4))


class TestDefaultDevicePolicy(unittest.TestCase):
    def setUp(self) -> None:
        gp.close_default_device()

    def tearDown(self) -> None:
        gp.close_default_device()

    def test_implicit_array_and_empty_share_default_device_on_simulated_platforms(self) -> None:
        for plat, expected_target in (
            ("darwin", "metal"),
            ("win32", "directx"),
            ("linux", "vulkan"),
        ):
            with self.subTest(platform=plat):
                gp.close_default_device()
                created: list[_SpyBackend] = []

                def make_backend(target: str, **_kwargs: object) -> _SpyBackend:
                    backend = _SpyBackend(target)
                    created.append(backend)
                    return backend

                with mock.patch.object(sys, "platform", plat), mock.patch(
                    "gfxpy.gpu_backend.GfxIRComputeBackend",
                    side_effect=make_backend,
                ) as backend_cls:
                    a = gp.array([1.0, 2.0], dtype=gp.float32)
                    b = gp.array([3.0, 4.0], dtype=gp.float32)
                    e = gp.empty(2, dtype=gp.float32)

                    self.assertIs(a.device, b.device)
                    self.assertIs(a.device, e.device)
                    self.assertIs(a.device, gp.current_device())
                    self.assertEqual(a.device.target, expected_target)
                    self.assertEqual(backend_cls.call_count, 1)
                    self.assertEqual(len(created), 1)
                    self.assertEqual((a + b).tolist(), [4.0, 6.0])
                    self.assertEqual((a * b).tolist(), [3.0, 8.0])
                gp.close_default_device()

    def test_nested_explicit_scopes_restore_and_isolate_devices(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "gfxpy.gpu_backend.GfxIRComputeBackend",
            side_effect=lambda target, **_kw: _SpyBackend(target),
        ):
            d_default = gp.current_device()
            a_default = gp.array([1.0, 2.0])

            with gp.device("vulkan", backend=_SpyBackend("vulkan")) as d1:
                self.assertIs(gp.current_device(), d1)
                a1 = gp.array([3.0, 4.0])
                self.assertIs(a1.device, d1)
                with self.assertRaises(gp.CrossDeviceError):
                    _ = a_default + a1

                with gp.device("metal", backend=_SpyBackend("metal")) as d2:
                    self.assertIs(gp.current_device(), d2)
                    a2 = gp.array([5.0, 6.0])
                    self.assertIs(a2.device, d2)
                    with self.assertRaises(gp.CrossDeviceError):
                        _ = a1 + a2

                self.assertIs(gp.current_device(), d1)
                a1_b = gp.array([7.0, 8.0])
                self.assertEqual((a1 + a1_b).tolist(), [10.0, 12.0])

            self.assertIs(gp.current_device(), d_default)
            a_default_2 = gp.array([9.0, 10.0])
            self.assertEqual((a_default + a_default_2).tolist(), [10.0, 12.0])

    def test_explicit_separate_device_isolated_from_default(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "gfxpy.gpu_backend.GfxIRComputeBackend",
            side_effect=lambda target, **_kw: _SpyBackend(target),
        ):
            d_sep = gp.device("vulkan", backend=_SpyBackend("vulkan"))
            try:
                a_def = gp.array([1.0, 2.0])
                a_sep = gp.array([3.0, 4.0], device=d_sep)
                with self.assertRaises(gp.CrossDeviceError):
                    _ = a_def + a_sep
            finally:
                d_sep.close()

    def test_closed_default_device_cleanup_and_recreation(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "gfxpy.gpu_backend.GfxIRComputeBackend",
            side_effect=lambda target, **_kw: _SpyBackend(target),
        ):
            d1 = gp.current_device()
            b1 = d1.backend
            a1 = gp.array([1.0, 2.0])

            gp.close_default_device()
            self.assertTrue(d1.closed)
            self.assertTrue(b1.closed)
            with self.assertRaises(gp.DeviceClosedError):
                gp.array([1.0], device=d1)
            with self.assertRaises(gp.DeviceClosedError):
                a1.tolist()

            gp.close_default_device()  # idempotent

            d2 = gp.current_device()
            self.assertIsNot(d2, d1)
            self.assertFalse(d2.closed)
            a2 = gp.array([3.0, 4.0])
            self.assertIs(a2.device, d2)
            self.assertEqual((a2 + gp.array([1.0, 2.0])).tolist(), [4.0, 6.0])

            d2.close()
            self.assertTrue(d2.closed)
            d3 = gp.current_device()
            self.assertIsNot(d3, d2)
            self.assertFalse(d3.closed)

    def test_closed_explicit_scope_does_not_reopen_or_fall_back(self) -> None:
        with mock.patch.object(sys, "platform", "linux"), mock.patch(
            "gfxpy.gpu_backend.GfxIRComputeBackend",
            side_effect=lambda target, **_kw: _SpyBackend(target),
        ):
            d_default = gp.current_device()
            with gp.device("vulkan", backend=_SpyBackend("vulkan")) as d_scope:
                d_scope.close()
                with self.assertRaises(gp.DeviceClosedError):
                    gp.current_device()
                with self.assertRaises(gp.DeviceClosedError):
                    gp.array([1.0])
                with self.assertRaises(gp.DeviceClosedError):
                    gp.empty(1)
            self.assertIs(gp.current_device(), d_default)


if __name__ == "__main__":
    unittest.main()
