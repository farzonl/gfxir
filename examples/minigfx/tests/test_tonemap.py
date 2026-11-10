"""Comprehensive unit test suite for minigfx.tonemap.

Covers:
- TestAcesTonemap: zero, negative clamping, NaN handling, HDR limits, midgray, analytical reference points
- TestLinearToSrgb: boundary values, piecewise IEC vs power-law, junction continuity, monotonicity
- TestSrgbToLinear: boundary values, piecewise IEC vs power-law, junction continuity, monotonicity
- TestGammaRoundtrip: srgb_to_linear(linear_to_srgb(x)) ≈ x across [0.0, 1.0], discrete 8-bit roundtrip
- TestTonemapAndGamma: combined pass equivalence, reference values, extreme HDR/negative bounds
- TestPolymorphism: float, int, Vec3, Vec4 inputs, alpha channel preservation
- TestTypeSafety: rejection of bool, str, None, list, tuple, Vec2, Mat4 with TypeError
- TestMonotonicity: strict monotonicity across [0.0, 100.0]
"""

from __future__ import annotations

import math
import unittest

try:
    from examples.minigfx.math import Mat4, Vec2, Vec3, Vec4
except (ImportError, ModuleNotFoundError):
    from examples.minigfx.tonemap import Vec3, Vec4  # type: ignore[assignment]

    class Vec2:  # type: ignore[no-redef]
        def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
            self.x, self.y = float(x), float(y)

    class Mat4:  # type: ignore[no-redef]
        @classmethod
        def identity(cls):
            return cls()
from examples.minigfx.tonemap import (
    aces_tonemap,
    linear_to_srgb,
    srgb_gamma,
    srgb_to_linear,
    tonemap_and_gamma,
)


class TestAcesTonemap(unittest.TestCase):
    """Test Krzysztof Narkowicz ACES filmic tonemapping curve."""

    def test_zero_and_near_zero(self):
        self.assertEqual(aces_tonemap(0.0), 0.0)
        self.assertAlmostEqual(aces_tonemap(1e-6), 2.142857e-7, places=6)
        subnormal = 1e-40
        res = aces_tonemap(subnormal)
        self.assertGreaterEqual(res, 0.0)
        self.assertLessEqual(res, 0.001)

    def test_negative_clamping(self):
        for neg in (-1e-30, -1e-6, -0.01, -0.1, -0.5, -1.0, -100.0, -1e6, -float("inf")):
            with self.subTest(neg=neg):
                self.assertEqual(aces_tonemap(neg), 0.0)

    def test_nan_handling(self):
        val = aces_tonemap(float("nan"))
        self.assertFalse(math.isnan(val))
        self.assertEqual(val, 0.0)

    def test_hdr_limits(self):
        # Saturation crossing point: 3.5 + sqrt(14) ≈ 7.24165738677
        self.assertAlmostEqual(aces_tonemap(7.241657), 1.0, places=5)
        for hdr in (10.0, 50.0, 100.0, 1000.0, 1e6, 1e12, 1e20, 1e30, float("inf")):
            with self.subTest(hdr=hdr):
                self.assertEqual(aces_tonemap(hdr), 1.0)

    def test_midgray(self):
        # Photographic 18% middle gray
        expected = 21681.0 / 81233.0  # ≈ 0.26689892
        val = aces_tonemap(0.18)
        self.assertAlmostEqual(val, expected, places=6)
        self.assertGreaterEqual(val, 0.20)
        self.assertLessEqual(val, 0.35)

    def test_analytical_reference_points(self):
        # g(x) = (x*(2.51*x + 0.03)) / (x*(2.43*x + 0.59) + 0.14)
        reference_table = [
            (0.001, 2513.0 / 10868400.0),      # ≈ 0.00023122
            (0.01, 254.0 / 67365.0),           # ≈ 0.00377050
            (0.05, 0.007775 / 0.175575),       # ≈ 0.04428307
            (0.10, 281.0 / 2233.0),            # ≈ 0.12583968
            (0.18, 21681.0 / 81233.0),         # ≈ 0.26689892
            (0.20, 0.1064 / 0.3552),           # ≈ 0.29954955
            (0.50, 257.0 / 417.0),             # ≈ 0.61630695
            (0.80, 1.6304 / 2.1672),           # ≈ 0.75230712
            (1.00, 127.0 / 158.0),             # ≈ 0.80379747
            (1.50, 5.6925 / 6.4925),           # ≈ 0.87678090
            (2.00, 505.0 / 552.0),             # ≈ 0.91485507
            (5.00, 3145.0 / 3192.0),           # ≈ 0.98527569
        ]
        for x, expected in reference_table:
            with self.subTest(x=x):
                self.assertAlmostEqual(aces_tonemap(x), expected, places=6)


class TestLinearToSrgb(unittest.TestCase):
    """Test linear to sRGB gamma transfer function."""

    def test_boundary_values(self):
        self.assertEqual(linear_to_srgb(0.0), 0.0)
        self.assertEqual(linear_to_srgb(1.0), 1.0)
        self.assertEqual(linear_to_srgb(-0.5), 0.0)
        self.assertEqual(linear_to_srgb(2.0), 1.0)
        self.assertEqual(linear_to_srgb(float("nan")), 0.0)

    def test_piecewise_iec_vs_power_law(self):
        # Linear segment test (x <= 0.0031308)
        self.assertAlmostEqual(linear_to_srgb(0.002, precise=True), 12.92 * 0.002, places=6)
        self.assertAlmostEqual(linear_to_srgb(0.002, precise=False), 0.002 ** (1.0 / 2.2), places=6)

        # Power segment test (x > 0.0031308)
        self.assertAlmostEqual(linear_to_srgb(0.5, precise=True), 1.055 * (0.5 ** (1.0 / 2.4)) - 0.055, places=6)
        self.assertAlmostEqual(linear_to_srgb(0.5, precise=False), 0.5 ** (1.0 / 2.2), places=6)

        # srgb_gamma convenience alias equivalence
        self.assertEqual(srgb_gamma(0.5), linear_to_srgb(0.5, precise=False))

        # Junction continuity at x = 0.0031308
        x0 = 0.0031308
        lin_val = 12.92 * x0
        pow_val = 1.055 * (x0 ** (1.0 / 2.4)) - 0.055
        self.assertAlmostEqual(lin_val, pow_val, places=6)

    def test_strict_monotonicity(self):
        for precise in (True, False):
            prev = -1.0
            for i in range(501):
                x = i / 500.0
                val = linear_to_srgb(x, precise=precise)
                self.assertGreaterEqual(val, prev)
                prev = val


class TestSrgbToLinear(unittest.TestCase):
    """Test inverse sRGB gamma to linear luminance transfer function."""

    def test_boundary_values(self):
        self.assertEqual(srgb_to_linear(0.0), 0.0)
        self.assertEqual(srgb_to_linear(1.0), 1.0)
        self.assertEqual(srgb_to_linear(-0.5), 0.0)
        self.assertEqual(srgb_to_linear(2.0), 1.0)
        self.assertEqual(srgb_to_linear(float("nan")), 0.0)

    def test_piecewise_iec_vs_power_law(self):
        # Linear segment test (y <= 0.04045)
        self.assertAlmostEqual(srgb_to_linear(0.02, precise=True), 0.02 / 12.92, places=6)
        self.assertAlmostEqual(srgb_to_linear(0.02, precise=False), 0.02 ** 2.2, places=6)

        # Power segment test (y > 0.04045)
        self.assertAlmostEqual(srgb_to_linear(0.5, precise=True), ((0.5 + 0.055) / 1.055) ** 2.4, places=6)
        self.assertAlmostEqual(srgb_to_linear(0.5, precise=False), 0.5 ** 2.2, places=6)

        # Junction continuity at y = 0.04045
        y0 = 0.04045
        lin_val = y0 / 12.92
        pow_val = ((y0 + 0.055) / 1.055) ** 2.4
        self.assertAlmostEqual(lin_val, pow_val, places=6)

    def test_strict_monotonicity(self):
        for precise in (True, False):
            prev = -1.0
            for i in range(501):
                y = i / 500.0
                val = srgb_to_linear(y, precise=precise)
                self.assertGreaterEqual(val, prev)
                prev = val


class TestGammaRoundtrip(unittest.TestCase):
    """Test forward and inverse gamma roundtrip accuracy."""

    def test_roundtrip_precise(self):
        for i in range(1001):
            x = i / 1000.0
            forward_rt = srgb_to_linear(linear_to_srgb(x, precise=True), precise=True)
            self.assertAlmostEqual(forward_rt, x, places=5)
            reverse_rt = linear_to_srgb(srgb_to_linear(x, precise=True), precise=True)
            self.assertAlmostEqual(reverse_rt, x, places=5)

    def test_roundtrip_power_law(self):
        for i in range(1001):
            x = i / 1000.0
            forward_rt = srgb_to_linear(linear_to_srgb(x, precise=False), precise=False)
            self.assertAlmostEqual(forward_rt, x, places=5)
            reverse_rt = linear_to_srgb(srgb_to_linear(x, precise=False), precise=False)
            self.assertAlmostEqual(reverse_rt, x, places=5)

    def test_discrete_8bit_roundtrip(self):
        for c in range(256):
            x = c / 255.0
            # Precise IEC roundtrip
            srgb = linear_to_srgb(srgb_to_linear(x, precise=True), precise=True)
            self.assertEqual(round(srgb * 255.0), c)

            # Power-law roundtrip
            srgb_p = linear_to_srgb(srgb_to_linear(x, precise=False), precise=False)
            self.assertEqual(round(srgb_p * 255.0), c)


class TestTonemapAndGamma(unittest.TestCase):
    """Test combined tonemap_and_gamma pass."""

    def test_combined_equivalence(self):
        test_inputs = (0.0, 0.001, 0.05, 0.1, 0.18, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0)
        for x in test_inputs:
            with self.subTest(x=x):
                # Precise mode
                expected_precise = linear_to_srgb(aces_tonemap(x), precise=True)
                self.assertAlmostEqual(tonemap_and_gamma(x, precise=True), expected_precise, places=7)

                # Power-law mode
                expected_approx = linear_to_srgb(aces_tonemap(x), precise=False)
                self.assertAlmostEqual(tonemap_and_gamma(x, precise=False), expected_approx, places=7)

    def test_combined_reference_values(self):
        self.assertEqual(tonemap_and_gamma(0.0), 0.0)
        self.assertAlmostEqual(tonemap_and_gamma(0.1, precise=True), 0.389812, places=5)
        self.assertAlmostEqual(tonemap_and_gamma(0.18, precise=True), 0.553458, places=5)
        self.assertAlmostEqual(tonemap_and_gamma(0.5, precise=True), 0.807319, places=5)
        self.assertAlmostEqual(tonemap_and_gamma(1.0, precise=True), 0.908230, places=5)
        self.assertAlmostEqual(tonemap_and_gamma(2.0, precise=True), 0.961598, places=5)
        self.assertEqual(tonemap_and_gamma(10.0), 1.0)
        self.assertEqual(tonemap_and_gamma(100.0), 1.0)

    def test_combined_bounds_and_nan(self):
        self.assertEqual(tonemap_and_gamma(-10.0), 0.0)
        self.assertEqual(tonemap_and_gamma(float("nan")), 0.0)
        self.assertEqual(tonemap_and_gamma(1e20), 1.0)


class TestPolymorphism(unittest.TestCase):
    """Test polymorphic dispatch across scalar, Vec3, and Vec4."""

    def test_scalar_types(self):
        self.assertIsInstance(aces_tonemap(0.5), float)
        self.assertIsInstance(aces_tonemap(1), float)
        self.assertIsInstance(linear_to_srgb(0.5), float)
        self.assertIsInstance(linear_to_srgb(1), float)
        self.assertIsInstance(srgb_to_linear(0.5), float)
        self.assertIsInstance(srgb_to_linear(1), float)
        self.assertIsInstance(srgb_gamma(0.5), float)
        self.assertIsInstance(tonemap_and_gamma(0.5), float)

    def test_vec3(self):
        v = Vec3(0.0, 0.5, 1.0)
        res = aces_tonemap(v)
        self.assertIsInstance(res, Vec3)
        self.assertEqual(res.x, 0.0)
        self.assertAlmostEqual(res.y, aces_tonemap(0.5), places=6)
        self.assertAlmostEqual(res.z, aces_tonemap(1.0), places=6)

        # Vector roundtrip
        v_rt = srgb_to_linear(linear_to_srgb(v))
        self.assertIsInstance(v_rt, Vec3)
        self.assertAlmostEqual(v_rt.x, v.x, places=5)
        self.assertAlmostEqual(v_rt.y, v.y, places=5)
        self.assertAlmostEqual(v_rt.z, v.z, places=5)

    def test_vec4_alpha_invariance(self):
        # Alpha channel (w) MUST remain bitwise identical across all operations
        for alpha in (0.0, 0.001, 0.25, 0.5, 0.75, 1.0, -0.5, 99.0):
            with self.subTest(alpha=alpha):
                v = Vec4(0.1, 0.5, 2.0, alpha)

                # aces_tonemap
                res_aces = aces_tonemap(v)
                self.assertIsInstance(res_aces, Vec4)
                self.assertEqual(res_aces.w, alpha)
                self.assertAlmostEqual(res_aces.x, aces_tonemap(0.1), places=6)
                self.assertAlmostEqual(res_aces.y, aces_tonemap(0.5), places=6)
                self.assertAlmostEqual(res_aces.z, aces_tonemap(2.0), places=6)

                # linear_to_srgb
                res_lin = linear_to_srgb(v)
                self.assertIsInstance(res_lin, Vec4)
                self.assertEqual(res_lin.w, alpha)
                self.assertAlmostEqual(res_lin.x, linear_to_srgb(0.1), places=6)

                # srgb_to_linear
                res_srgb = srgb_to_linear(v)
                self.assertIsInstance(res_srgb, Vec4)
                self.assertEqual(res_srgb.w, alpha)
                self.assertAlmostEqual(res_srgb.x, srgb_to_linear(0.1), places=6)

                # srgb_gamma
                res_gamma = srgb_gamma(v)
                self.assertIsInstance(res_gamma, Vec4)
                self.assertEqual(res_gamma.w, alpha)
                self.assertAlmostEqual(res_gamma.x, srgb_gamma(0.1), places=6)

                # tonemap_and_gamma
                res_comb = tonemap_and_gamma(v)
                self.assertIsInstance(res_comb, Vec4)
                self.assertEqual(res_comb.w, alpha)
                self.assertAlmostEqual(res_comb.x, tonemap_and_gamma(0.1), places=6)


class TestTypeSafety(unittest.TestCase):
    """Verify strict type checking and rejection of invalid types."""

    def test_invalid_types_raise_type_error(self):
        invalid_inputs = [
            True,
            False,
            "1.0",
            "red",
            None,
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3, 1.0],
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.3, 1.0),
            Vec2(0.5, 0.5),
            Mat4.identity(),
            {"r": 1.0},
        ]
        funcs = [aces_tonemap, linear_to_srgb, srgb_to_linear, srgb_gamma, tonemap_and_gamma]
        for val in invalid_inputs:
            for fn in funcs:
                with self.subTest(fn=fn.__name__, val=val):
                    with self.assertRaises(TypeError):
                        fn(val)


class TestMonotonicity(unittest.TestCase):
    """Verify strict monotonicity across [0.0, 100.0]."""

    def test_tonemap_monotonicity(self):
        prev = -1.0
        # Linear range [0, 1]
        for i in range(501):
            x = i / 500.0
            val = aces_tonemap(x)
            self.assertGreaterEqual(val, prev)
            prev = val

        # Mid-HDR range [1, 10]
        for i in range(1, 501):
            x = 1.0 + (9.0 * i / 500.0)
            val = aces_tonemap(x)
            self.assertGreaterEqual(val, prev)
            prev = val

        # High-HDR range [10, 100]
        for i in range(1, 501):
            x = 10.0 + (90.0 * i / 500.0)
            val = aces_tonemap(x)
            self.assertGreaterEqual(val, prev)
            prev = val

    def test_combined_monotonicity(self):
        prev = -1.0
        for i in range(1001):
            x = 100.0 * (i / 1000.0)
            val = tonemap_and_gamma(x)
            self.assertGreaterEqual(val, prev)
            prev = val

    def test_strictly_increasing_before_saturation(self):
        # Before saturation threshold (x_sat ≈ 7.24), ACES curve is strictly increasing
        prev = -1.0
        for i in range(701):
            x = 7.0 * (i / 700.0)
            val = aces_tonemap(x)
            self.assertGreater(val, prev)
            prev = val


if __name__ == "__main__":
    unittest.main()
