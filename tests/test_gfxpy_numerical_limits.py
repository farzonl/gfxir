"""Boundary validation, numerical accuracy, and non-finite tests for gfxpy."""

from __future__ import annotations

import math
import os
from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402
from gfxpy.limits import (  # noqa: E402
    FLOAT32_EPSILON,
    FLOAT32_MAX,
    I32_MAX,
    ComputeLimits,
    validate_1d_dispatch,
    validate_elementwise_dispatch,
    validate_matmul_dispatch,
)


def _f32(x: float) -> float:
    try:
        return struct.unpack("<f", struct.pack("<f", x))[0]
    except OverflowError:
        return math.copysign(math.inf, x)


class _SpyBuffer:
    def __init__(self, size_bytes: int) -> None:
        self.data = bytearray(size_bytes)
        self.closed = False

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def close(self) -> None:
        self.closed = True


class _SpyLimitBackend:
    """In-memory float32 test double with configurable ComputeLimits and call counters."""

    def __init__(self, target: str = "vulkan", limits: ComputeLimits | None = None) -> None:
        self.target = target
        self.limits = limits or ComputeLimits()
        self.closed = False
        self.allocations: list[int] = []
        self.dispatches: list[str] = []

    def allocate(self, size_bytes: int) -> _SpyBuffer:
        self.allocations.append(size_bytes)
        return _SpyBuffer(size_bytes)

    def upload(self, buf: _SpyBuffer, data: bytes) -> None:
        buf.data[: len(data)] = data

    def download(self, buf: _SpyBuffer) -> bytes:
        return bytes(buf.data)

    def binary_op(self, op: str, lhs: _SpyBuffer, rhs: _SpyBuffer, out: _SpyBuffer, length: int) -> None:
        self.dispatches.append(f"binary:{op}:{length}")
        lv = struct.unpack(f"<{length}f", lhs.data[: length * 4])
        rv = struct.unpack(f"<{length}f", rhs.data[: length * 4])
        if op == "add":
            res = [_f32(a + b) for a, b in zip(lv, rv)]
        else:
            res = [_f32(a * b) for a, b in zip(lv, rv)]
        out.data[: length * 4] = struct.pack(f"<{length}f", *res)

    def scalar_op(self, op: str, lhs: _SpyBuffer, scalar: float, out: _SpyBuffer, length: int) -> None:
        self.dispatches.append(f"scalar:{op}:{length}")
        lv = struct.unpack(f"<{length}f", lhs.data[: length * 4])
        s = _f32(scalar)
        if op == "add":
            res = [_f32(a + s) for a in lv]
        else:
            res = [_f32(a * s) for a in lv]
        out.data[: length * 4] = struct.pack(f"<{length}f", *res)

    def fused_op(self, spec: object, inputs: list[_SpyBuffer], scalars: list[float], out: _SpyBuffer, length: int) -> None:
        self.dispatches.append(f"fused:{length}")
        arrs = [struct.unpack(f"<{length}f", buf.data[: length * 4]) for buf in inputs]
        s_f32 = [_f32(v) for v in scalars]
        sig: str = getattr(spec, "signature")

        def _parse(pos: int) -> tuple[list[float], int]:
            if sig[pos] == "(":
                left, pos = _parse(pos + 1)
                op = sig[pos]
                right, pos = _parse(pos + 1)
                assert sig[pos] == ")"
                if op == "+":
                    return [_f32(x + y) for x, y in zip(left, right)], pos + 1
                return [_f32(x * y) for x, y in zip(left, right)], pos + 1
            if sig[pos : pos + 2] == "in":
                end = pos + 2
                while end < len(sig) and sig[end].isdigit():
                    end += 1
                return list(arrs[int(sig[pos + 2 : end])]), end
            assert sig[pos] == "s"
            end = pos + 1
            while end < len(sig) and sig[end].isdigit():
                end += 1
            return [s_f32[int(sig[pos + 1 : end])]] * length, end

        res, _ = _parse(0)
        out.data[: length * 4] = struct.pack(f"<{length}f", *res)

    def reduce_sum_pass(self, inp: _SpyBuffer, out: _SpyBuffer, length: int) -> None:
        self.dispatches.append(f"reduce:{length}")
        vals = struct.unpack(f"<{length}f", inp.data[: length * 4])
        num_groups = (length + 63) // 64
        partials: list[float] = []
        for g in range(num_groups):
            lane = list(vals[g * 64 : (g + 1) * 64])
            lane.extend([0.0] * (64 - len(lane)))
            stride = 32
            while stride > 0:
                for lx in range(stride):
                    lane[lx] = _f32(lane[lx] + lane[lx + stride])
                stride //= 2
            partials.append(lane[0])
        out.data[: num_groups * 4] = struct.pack(f"<{num_groups}f", *partials)

    def matmul(self, a: _SpyBuffer, b: _SpyBuffer, out: _SpyBuffer, m: int, n: int, k: int) -> None:
        self.dispatches.append(f"matmul:{m}x{n}x{k}")
        av = struct.unpack(f"<{m * k}f", a.data[: m * k * 4])
        bv = struct.unpack(f"<{k * n}f", b.data[: k * n * 4])
        cv = [0.0] * (m * n)
        for r in range(m):
            for c in range(n):
                acc = 0.0
                for p in range(k):
                    acc = _f32(acc + _f32(av[r * k + p] * bv[p * n + c]))
                cv[r * n + c] = acc
        out.data[: m * n * 4] = struct.pack(f"<{m * n}f", *cv)

    def close(self) -> None:
        self.closed = True


class TestIndexingAndResourceLimits(unittest.TestCase):
    """Metadata and test-double boundary validation without enormous allocations."""

    def test_i32_dimension_and_product_limits_rejected_before_allocation(self) -> None:
        backend = _SpyLimitBackend("metal")
        with gp.device("metal", backend=backend) as dev:
            with self.assertRaises(gp.IndexingLimitError):
                gp.empty(I32_MAX + 1, device=dev)
            with self.assertRaises(gp.IndexingLimitError):
                gp.empty((50_000, 50_000), device=dev)
            self.assertEqual(backend.allocations, [])
            self.assertEqual(backend.dispatches, [])

    def test_matmul_output_overflow_rejected_even_when_inputs_fit(self) -> None:
        backend = _SpyLimitBackend("metal")
        with gp.device("metal", backend=backend) as dev:
            a = gp.empty((50_000, 10), device=dev)
            b = gp.empty((10, 50_000), device=dev)
            allocs_before = len(backend.allocations)
            with self.assertRaises(gp.IndexingLimitError):
                _ = a @ b
            self.assertEqual(len(backend.allocations), allocs_before)
            self.assertEqual(backend.dispatches, [])

    def test_max_buffer_bytes_boundary(self) -> None:
        backend = _SpyLimitBackend("vulkan", limits=ComputeLimits(max_buffer_bytes=16))
        with gp.device("vulkan", backend=backend) as dev:
            ok = gp.empty(4, device=dev)
            self.assertEqual(ok.nbytes, 16)
            allocs_before = len(backend.allocations)
            with self.assertRaises(gp.IndexingLimitError):
                gp.empty(5, device=dev)
            with self.assertRaises(gp.IndexingLimitError):
                gp.array([1.0, 2.0, 3.0, 4.0, 5.0], device=dev)
            self.assertEqual(len(backend.allocations), allocs_before)

    def test_workgroup_count_boundaries_for_1d_fusion_sum_and_matmul(self) -> None:
        limits = ComputeLimits(max_workgroup_count=(4, 2, 1))
        backend = _SpyLimitBackend("directx", limits=limits)
        with gp.device("directx", backend=backend) as dev:
            # Reductions remain 1D; elementwise operations can also use the Y axis.
            a_ok = gp.empty(256, device=dev)
            b_ok = gp.empty(256, device=dev)
            _ = a_ok + b_ok
            _ = a_ok + 1.0
            _ = a_ok.sum()
            _ = gp.fuse(lambda x: x * 2.0 + 1.0)(a_ok)

            a_two_rows = gp.empty(257, device=dev)
            _ = a_two_rows + 1.0
            with self.assertRaises(gp.IndexingLimitError):
                a_two_rows.sum()

            a_bad = gp.empty(513, device=dev)
            b_bad = gp.empty(513, device=dev)
            allocs_before = len(backend.allocations)
            dispatches_before = len(backend.dispatches)

            for fn in (
                lambda: a_bad + b_bad,
                lambda: a_bad + 1.0,
                lambda: a_bad.sum(),
                lambda: gp.fuse(lambda x: x * 2.0 + 1.0)(a_bad),
            ):
                with self.assertRaises(gp.IndexingLimitError):
                    fn()
            self.assertEqual(len(backend.allocations), allocs_before)
            self.assertEqual(len(backend.dispatches), dispatches_before)

            # MatMul tile is 16x16 -> max grid (4, 2, 1) allows up to (32, 64).
            m_ok_a = gp.empty((32, 17), device=dev)
            m_ok_b = gp.empty((17, 64), device=dev)
            _ = m_ok_a @ m_ok_b

            m_bad_rows = gp.empty((33, 17), device=dev)
            m_bad_cols = gp.empty((17, 65), device=dev)
            allocs_before = len(backend.allocations)
            dispatches_before = len(backend.dispatches)
            with self.assertRaises(gp.IndexingLimitError):
                _ = m_bad_rows @ m_ok_b
            with self.assertRaises(gp.IndexingLimitError):
                _ = m_ok_a @ m_bad_cols
            self.assertEqual(len(backend.allocations), allocs_before)
            self.assertEqual(len(backend.dispatches), dispatches_before)

    def test_elementwise_dispatch_spills_into_y_without_relaxing_limits(self) -> None:
        limits = ComputeLimits(max_workgroup_count=(65535, 65535, 65535))
        self.assertEqual(validate_elementwise_dispatch(65535 * 64, 64, limits), (65535, 1, 1))
        self.assertEqual(validate_elementwise_dispatch(4_194_304, 64, limits), (65535, 2, 1))
        with self.assertRaises(gp.IndexingLimitError):
            validate_elementwise_dispatch(513, 64, ComputeLimits(max_workgroup_count=(4, 2, 1)))
        with self.assertRaises(gp.IndexingLimitError):
            validate_elementwise_dispatch(I32_MAX, 64, limits)
        with self.assertRaises(gp.IndexingLimitError):
            validate_elementwise_dispatch(I32_MAX, 96, ComputeLimits())

    def test_rounded_up_invocation_index_overflow_rejected(self) -> None:
        # length <= I32_MAX, but rounding up to non-power-of-two workgroup_size_x=96 makes gid.x > I32_MAX.
        with self.assertRaises(gp.IndexingLimitError):
            validate_1d_dispatch(I32_MAX, 96, ComputeLimits(), op="test_1d")
        with self.assertRaises(gp.IndexingLimitError):
            validate_matmul_dispatch(I32_MAX, 1, 1, 4, 24, 24, 16, ComputeLimits())

    def test_float32_scalar_and_array_overflow_and_underflow(self) -> None:
        backend = _SpyLimitBackend("vulkan")
        with gp.device("vulkan", backend=backend) as dev:
            for bad in (1e39, -1e39, 10**400):
                with self.assertRaises(OverflowError):
                    gp.array([bad], device=dev)
            a = gp.array([1.0, 2.0], device=dev)
            empty = gp.array([], device=dev)
            allocs_before = len(backend.allocations)
            for fn in (
                lambda: a + 1e39,
                lambda: 1e39 * a,
                lambda: empty + 1e39,
                lambda: gp.fuse(lambda x: x + 1e39)(a),
            ):
                with self.assertRaises(OverflowError):
                    fn()
            self.assertEqual(len(backend.allocations), allocs_before)
            self.assertEqual(backend.dispatches, [])

            # Sub-minimum values underflow to 0.0; inf/nan are preserved.
            u = gp.array([1e-50, float("inf"), float("-inf"), float("nan")], device=dev)
            vals = u.tolist()
            self.assertEqual(vals[0], 0.0)
            self.assertTrue(math.isinf(vals[1]) and vals[1] > 0)
            self.assertTrue(math.isinf(vals[2]) and vals[2] < 0)
            self.assertTrue(math.isnan(vals[3]))


class _NumericalAccuracyMixin:
    """Shared numerical accuracy, cancellation, non-exact-decimal, and non-finite checks."""

    target_name: str

    def _run_numerical_checks(self, dev: gp.Device) -> None:
        # 1. Non-exact-decimal inputs + cancellation in elementwise & fused ops
        #    (0.1, 0.2, 0.3 are not exact binary fractions in IEEE 754)
        a_vals = [0.1, 10000.3, -10000.1, 1e-4, 0.33333334]
        b_vals = [0.2, -10000.1, 10000.3, 1.0, 0.6666667]
        a = gp.array(a_vals, device=dev)
        b = gp.array(b_vals, device=dev)

        fused = gp.fuse(lambda x, y: (x + y) * 0.3 + 0.1)(a, b)
        got_fused = fused.tolist()
        for idx, (x, y) in enumerate(zip(a_vals, b_vals)):
            ref = (x + y) * 0.3 + 0.1
            # Scale error bound by intermediate magnitude to account for cancellation.
            scale = max(1.0, abs(x) + abs(y), abs(ref))
            self.assertAlmostEqual(got_fused[idx], ref, delta=16.0 * FLOAT32_EPSILON * scale)

        # 2. Mixed-magnitude & cancellation sum (non-64-aligned length = 131)
        #    Includes large cancelling pairs + small non-exact decimals.
        mix = [1e4 if i % 2 == 0 else -1e4 for i in range(130)] + [0.1]
        mix[3] += 0.3
        mix[77] += 0.7
        x_mix = gp.array(mix, device=dev)
        got_sum = x_mix.sum().tolist()[0]
        ref_sum = math.fsum(mix)  # 1.1
        cond_sum = sum(abs(v) for v in mix)  # ~1.3e6
        # Parallel float32 tree reduction error is bounded by O(log2(N) * eps * sum(|x_i|)).
        self.assertAlmostEqual(got_sum, ref_sum, delta=32.0 * FLOAT32_EPSILON * cond_sum)

        # 3. Non-tile-aligned matmul (17, 19) @ (19, 23) with mixed magnitudes & non-exact decimals
        m, k, n = 17, 19, 23
        mat_a_vals = [
            [((r * 7 + c * 3) % 11 - 5) * 0.1 + (1e3 if (r + c) % 5 == 0 else 0.0) for c in range(k)]
            for r in range(m)
        ]
        mat_b_vals = [
            [((r * 5 + c * 11) % 13 - 6) * 0.1 - (1e2 if (r + c) % 4 == 0 else 0.0) for c in range(n)]
            for r in range(k)
        ]
        ma = gp.array(mat_a_vals, device=dev)
        mb = gp.array(mat_b_vals, device=dev)
        mc = (ma @ mb).tolist()
        for r in range(m):
            for c in range(n):
                ref = math.fsum(mat_a_vals[r][p] * mat_b_vals[p][c] for p in range(k))
                cond = math.fsum(abs(mat_a_vals[r][p]) * abs(mat_b_vals[p][c]) for p in range(k))
                self.assertAlmostEqual(mc[r][c], ref, delta=4.0 * k * FLOAT32_EPSILON * max(1.0, cond))

        # 4. Arithmetic overflow -> +inf and non-finite propagation
        ov = gp.array([3.0e38, 1.0e20], device=dev)
        ov_add = (ov + 3.0e38).tolist()
        self.assertTrue(math.isinf(ov_add[0]) and ov_add[0] > 0)

        ov_mul = (ov * 1.0e20).tolist()
        self.assertTrue(math.isinf(ov_mul[1]) and ov_mul[1] > 0)

        s_inf = gp.array([1.0, float("inf"), 2.0], device=dev).sum().tolist()[0]
        self.assertTrue(math.isinf(s_inf) and s_inf > 0)

        s_inf_nan = gp.array([float("inf"), float("-inf"), 1.0], device=dev).sum().tolist()[0]
        self.assertTrue(math.isnan(s_inf_nan))

        s_nan = gp.array([float("nan"), 1.0, 2.0], device=dev).sum().tolist()[0]
        self.assertTrue(math.isnan(s_nan))

        mm_nf_a = gp.array([[1.0, float("nan")], [float("inf"), 1.0]], device=dev)
        mm_nf_b = gp.array([[1.0, 0.0], [1.0, 1.0]], device=dev)
        mm_nf = (mm_nf_a @ mm_nf_b).tolist()
        self.assertTrue(math.isnan(mm_nf[0][0]))
        self.assertTrue(math.isinf(mm_nf[1][0]) and mm_nf[1][0] > 0)


class TestNumericalAccuracyHost(_NumericalAccuracyMixin, unittest.TestCase):
    target_name = "vulkan"

    def test_host_float32_accuracy_cancellation_and_non_finite(self) -> None:
        backend = _SpyLimitBackend("vulkan")
        with gp.device("vulkan", backend=backend) as dev:
            self._run_numerical_checks(dev)


class TestNumericalAccuracyMetalLive(_NumericalAccuracyMixin, unittest.TestCase):
    target_name = "metal"

    @unittest.skipUnless(
        os.environ.get("MLIR_GPU_TEST_METAL") == "1",
        "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal numerical accuracy tests",
    )
    def test_live_numerical_accuracy_and_non_finite(self) -> None:
        with gp.device(self.target_name, validation=os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1") as dev:
            self._run_numerical_checks(dev)


class TestNumericalAccuracyVulkanLive(_NumericalAccuracyMixin, unittest.TestCase):
    target_name = "vulkan"

    @unittest.skipUnless(
        os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
        "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan numerical accuracy tests",
    )
    def test_live_numerical_accuracy_and_non_finite(self) -> None:
        provider = "swiftshader" if os.environ.get("MLIR_GPU_TEST_SWIFTSHADER") == "1" else None
        with gp.device(
            self.target_name,
            validation=os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1",
            provider=provider,
        ) as dev:
            self._run_numerical_checks(dev)


@unittest.skipUnless(
    sys.platform in ("linux", "darwin") and os.environ.get("MLIR_GPU_TEST_NACELLE") == "1",
    "Set MLIR_GPU_TEST_NACELLE=1 on Linux/macOS to run live Nacelle numerical accuracy tests",
)
class TestNumericalAccuracyNacelleLive(_NumericalAccuracyMixin, unittest.TestCase):
    target_name = "directx"

    def test_live_numerical_accuracy_and_non_finite(self) -> None:
        dev = gp.device(self.target_name, provider="nacelle")
        self.addCleanup(dev.close)
        with dev:
            self._run_numerical_checks(dev)


class TestNumericalAccuracyDirectXLive(_NumericalAccuracyMixin, unittest.TestCase):
    target_name = "directx"

    @unittest.skipUnless(
        os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
        "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX numerical accuracy tests",
    )
    def test_live_numerical_accuracy_and_non_finite(self) -> None:
        provider = "warp" if os.environ.get("MLIR_GPU_TEST_WARP") == "1" else None
        with gp.device(
            self.target_name,
            validation=os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1",
            provider=provider,
        ) as dev:
            self._run_numerical_checks(dev)


if __name__ == "__main__":
    unittest.main()
