"""Numerical limits and edge-case testing suite for MiniGfx CPU math and camera controller.

Covers:
1. Mat4.inverse() numerical stability across 1,000 random non-singular matrices (T * R * S).
2. Singular matrix detection: zero, rank-deficient (plane projection, identical cols/rows, col multiples), near-singular (|det| < 1e-12).
3. Quaternion vs matrix rotation equivalence across 100 random axis-angles (norm(Q.to_mat4() * v - R_axis(theta) * v) < 1e-5).
4. Quaternion slerp boundary conditions: identical (t in [0, 1]), antipodal (q and -q), orthogonal, and near-parallel.
5. Vector operations: high magnitude, subnormal/denormalized floats, zero-vector normalization, division by zero, bounds checking.
6. Camera and frustum culling boundary conditions: extreme aspect/FOV, near-singular look-at, controller extreme dt/damping.
"""

from __future__ import annotations

import math
import random
import struct
import unittest

from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.math import Mat4, Quat, Vec2, Vec3, Vec4


class MatrixInverseNumericalLimitsTests(unittest.TestCase):
    """Numerical limits testing of Mat4.inverse() precision and singular detection."""

    def test_inverse_precision_1000_random_matrices(self):
        """Stress-test Mat4.inverse() over 1,000 random non-singular SRT matrices.

        Verifies ||M * M^-1 - I||_inf < 1e-4 and ||M^-1 * M - I||_inf < 1e-4.
        """
        rng = random.Random(42)  # Deterministic seed for reproducibility
        max_err_mm_inv = 0.0
        max_err_inv_mm = 0.0

        for _ in range(1000):
            # Random translation in [-100, 100]
            tx = rng.uniform(-100.0, 100.0)
            ty = rng.uniform(-100.0, 100.0)
            tz = rng.uniform(-100.0, 100.0)
            t_mat = Mat4.translation(tx, ty, tz)

            # Random rotation axis and angle in [-pi, pi]
            ax = rng.uniform(-1.0, 1.0)
            ay = rng.uniform(-1.0, 1.0)
            az = rng.uniform(-1.0, 1.0)
            axis_len = math.sqrt(ax * ax + ay * ay + az * az)
            if axis_len < 1e-6:
                ax, ay, az = 0.0, 1.0, 0.0
            else:
                ax, ay, az = ax / axis_len, ay / axis_len, az / axis_len
            angle = rng.uniform(-math.pi, math.pi)
            r_mat = Mat4.rotation_axis((ax, ay, az), angle)

            # Random scale in [-10, -0.1] U [0.1, 10] to avoid near-zero scale
            def _rand_scale():
                val = rng.uniform(0.1, 10.0)
                return -val if rng.random() < 0.3 else val

            sx, sy, sz = _rand_scale(), _rand_scale(), _rand_scale()
            s_mat = Mat4.scaling(sx, sy, sz)

            # Composite transformation matrix M = T * R * S
            m = t_mat * r_mat * s_mat
            m_inv = m.inverse()

            # Check M * M^-1 == I
            prod1 = m * m_inv
            for r in range(4):
                for c in range(4):
                    expected = 1.0 if r == c else 0.0
                    err = abs(prod1[r, c] - expected)
                    if err > max_err_mm_inv:
                        max_err_mm_inv = err
                    self.assertLess(
                        err,
                        1e-4,
                        f"M * M^-1 failed at [{r}, {c}]: got {prod1[r, c]}, expected {expected}, err={err}",
                    )

            # Check M^-1 * M == I
            prod2 = m_inv * m
            for r in range(4):
                for c in range(4):
                    expected = 1.0 if r == c else 0.0
                    err = abs(prod2[r, c] - expected)
                    if err > max_err_inv_mm:
                        max_err_inv_mm = err
                    self.assertLess(
                        err,
                        1e-4,
                        f"M^-1 * M failed at [{r}, {c}]: got {prod2[r, c]}, expected {expected}, err={err}",
                    )

        # Confirm max observed error is well within 1e-4
        self.assertLess(max_err_mm_inv, 1e-4)
        self.assertLess(max_err_inv_mm, 1e-4)

    def test_singular_matrix_detection(self):
        """Verify robust detection of singular and rank-deficient matrices."""
        # 1. Zero matrix
        with self.assertRaises(ValueError) as ctx:
            Mat4.zero().inverse()
        self.assertIn("singular", str(ctx.exception).lower())

        # 2. Projection onto plane (rank 3, scale Z = 0)
        proj_plane = Mat4.scaling(1.0, 1.0, 0.0)
        with self.assertRaises(ValueError) as ctx:
            proj_plane.inverse()
        self.assertIn("singular", str(ctx.exception).lower())

        # 3. Two identical columns
        col_dup = Mat4((
            1, 2, 3, 4,
            1, 2, 3, 4,  # Column 1 identical to column 0
            5, 6, 7, 8,
            9, 0, 1, 2,
        ))
        with self.assertRaises(ValueError):
            col_dup.inverse()

        # 4. Two identical rows
        row_dup = Mat4((
            1, 1, 3, 4,  # Row 0: (1, 5, 9, 3), Row 1: (1, 5, 9, 3)
            5, 5, 7, 8,
            9, 9, 1, 2,
            3, 3, 4, 5,
        ))
        with self.assertRaises(ValueError):
            row_dup.inverse()

        # 5. Col linearly dependent: col 1 = 3 * col 0
        col_dep = Mat4((
            1, 2, 3, 4,
            3, 6, 9, 12,  # 3x col 0
            5, 1, 2, 3,
            7, 8, 9, 0,
        ))
        with self.assertRaises(ValueError):
            col_dep.inverse()

        # 6. All ones matrix (rank 1)
        all_ones = Mat4((1.0,) * 16)
        with self.assertRaises(ValueError):
            all_ones.inverse()

        # 7. Near-singular matrix (|det| < 1e-12)
        # sx = 1e-7, sy = 1e-7, sz = 1.0 => det = 1e-14 < 1e-12
        near_sing = Mat4.scaling(1e-7, 1e-7, 1.0)
        with self.assertRaises(ValueError) as ctx:
            near_sing.inverse()
        self.assertIn("singular", str(ctx.exception).lower())


class QuaternionNumericalLimitsTests(unittest.TestCase):
    """Testing of Quat vs Mat4 rotation equivalence and slerp boundaries."""

    def test_quat_rotation_vs_matrix_rotation_100_random_axes(self):
        """Verify ||Q.to_mat4() * v - R_axis(theta) * v|| < 1e-5 for 100 random axis-angles."""
        rng = random.Random(1337)
        max_err = 0.0

        for _ in range(100):
            # Random unit axis
            ax = rng.uniform(-1.0, 1.0)
            ay = rng.uniform(-1.0, 1.0)
            az = rng.uniform(-1.0, 1.0)
            norm = math.sqrt(ax * ax + ay * ay + az * az)
            if norm < 1e-6:
                continue
            axis = (ax / norm, ay / norm, az / norm)
            theta = rng.uniform(-2.0 * math.pi, 2.0 * math.pi)

            # Random 3D vector
            vx = rng.uniform(-10.0, 10.0)
            vy = rng.uniform(-10.0, 10.0)
            vz = rng.uniform(-10.0, 10.0)
            vec = Vec4(vx, vy, vz, 1.0)

            q = Quat.from_axis_angle(axis, theta)
            m_q = q.to_mat4()
            m_axis = Mat4.rotation_axis(axis, theta)

            res_q = m_q * vec
            res_m = m_axis * vec

            # Check component-wise difference
            err = max(
                abs(res_q.x - res_m.x),
                abs(res_q.y - res_m.y),
                abs(res_q.z - res_m.z),
                abs(res_q.w - res_m.w),
            )
            if err > max_err:
                max_err = err
            self.assertLess(err, 1e-5, f"Quat vs Mat4 rotation divergence err={err}")

            # Also check direct matrix equivalence
            for idx in range(16):
                mat_diff = abs(m_q.elements[idx] - m_axis.elements[idx])
                self.assertLess(mat_diff, 1e-5, f"Matrix element {idx} differs by {mat_diff}")

        self.assertLess(max_err, 1e-5)

    def test_slerp_boundary_conditions(self):
        """Test slerp boundary conditions: identical, antipodal, orthogonal, near-parallel."""
        # 1. Identical quaternions across t in [0.0, 1.0]
        q = Quat.from_axis_angle(Vec3(0, 1, 0), 0.75)
        for t in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
            s = q.slerp(q, t)
            self.assertAlmostEqual(s.x, q.x, places=6)
            self.assertAlmostEqual(s.y, q.y, places=6)
            self.assertAlmostEqual(s.z, q.z, places=6)
            self.assertAlmostEqual(s.w, q.w, places=6)
            self.assertAlmostEqual(s.norm(), 1.0, places=6)

        # 2. Opposite quaternions (q and -q represent same orientation in SO(3))
        # Verifies shortest-arc geodesic handling (dot < 0 negation)
        q_neg = Quat(-q.x, -q.y, -q.z, -q.w)
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            s = q.slerp(q_neg, t)
            # The resulting quaternion must represent the exact same rotation as q
            # (either +q or -q, i.e. dot(s, q)^2 == 1.0)
            dot = s.x * q.x + s.y * q.y + s.z * q.z + s.w * q.w
            self.assertAlmostEqual(abs(dot), 1.0, places=6)
            # Verify matrix representation is identical to q.to_mat4()
            self.assertEqual(s.to_mat4(), q.to_mat4())

        # 3. Orthogonal quaternions: dot(q1, q2) == 0
        # q1 = identity (0, 0, 0, 1)
        # q2 = 180 deg around X (1, 0, 0, 0)
        q1 = Quat(0.0, 0.0, 0.0, 1.0)
        q2 = Quat(1.0, 0.0, 0.0, 0.0)
        self.assertEqual(q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w, 0.0)

        # t = 0 -> q1
        self.assertEqual(q1.slerp(q2, 0.0), q1)
        # t = 1 -> q2
        self.assertEqual(q1.slerp(q2, 1.0), q2)
        # t = 0.5 -> 90 deg around X (sqrt(2)/2, 0, 0, sqrt(2)/2)
        mid = q1.slerp(q2, 0.5)
        self.assertAlmostEqual(mid.x, math.sqrt(0.5), places=6)
        self.assertAlmostEqual(mid.y, 0.0, places=6)
        self.assertAlmostEqual(mid.z, 0.0, places=6)
        self.assertAlmostEqual(mid.w, math.sqrt(0.5), places=6)
        self.assertAlmostEqual(mid.norm(), 1.0, places=6)

        # 4. Near-parallel quaternions (dot > 0.9995 triggers lerp fallback)
        q_near = Quat.from_axis_angle(Vec3(0, 1, 0), 1e-4)
        s_near = Quat.identity().slerp(q_near, 0.5)
        self.assertAlmostEqual(s_near.norm(), 1.0, places=6)
        expected_half = Quat.from_axis_angle(Vec3(0, 1, 0), 0.5e-4)
        self.assertAlmostEqual(s_near.y, expected_half.y, places=6)
        self.assertAlmostEqual(s_near.w, expected_half.w, places=6)


class VectorNumericalLimitsTests(unittest.TestCase):
    """Testing of vector numerical stability and boundary conditions."""

    def test_zero_vector_normalization(self):
        """Zero vectors must normalize to zero vectors without NaN or ZeroDivisionError."""
        v2_zero = Vec2(0.0, 0.0).normalize()
        self.assertEqual((v2_zero.x, v2_zero.y), (0.0, 0.0))
        self.assertFalse(math.isnan(v2_zero.x) or math.isnan(v2_zero.y))

        v3_zero = Vec3(0.0, 0.0, 0.0).normalize()
        self.assertEqual((v3_zero.x, v3_zero.y, v3_zero.z), (0.0, 0.0, 0.0))
        self.assertFalse(math.isnan(v3_zero.x) or math.isnan(v3_zero.y) or math.isnan(v3_zero.z))

        v4_zero = Vec4(0.0, 0.0, 0.0, 0.0).normalize()
        self.assertEqual((v4_zero.x, v4_zero.y, v4_zero.z, v4_zero.w), (0.0, 0.0, 0.0, 0.0))
        self.assertFalse(math.isnan(v4_zero.x) or math.isnan(v4_zero.w))

    def test_denormalized_subnormal_floats(self):
        """Subnormal floats smaller than 1e-12 should be safely handled as zero length."""
        tiny_v = Vec3(1e-15, -1e-16, 1e-20)
        norm_tiny = tiny_v.normalize()
        self.assertEqual((norm_tiny.x, norm_tiny.y, norm_tiny.z), (0.0, 0.0, 0.0))

    def test_high_magnitude_vectors(self):
        """High-magnitude vectors within standard float ranges should normalize correctly."""
        huge_v = Vec3(1e12, 2e12, 2e12)
        norm = huge_v.normalize()
        self.assertAlmostEqual(norm.length(), 1.0, places=6)
        self.assertAlmostEqual(norm.x, 1.0 / 3.0, places=6)
        self.assertAlmostEqual(norm.y, 2.0 / 3.0, places=6)
        self.assertAlmostEqual(norm.z, 2.0 / 3.0, places=6)

    def test_division_by_zero_and_negative_zero(self):
        """Division by 0.0 and -0.0 must raise ZeroDivisionError."""
        v = Vec3(1.0, 2.0, 3.0)
        with self.assertRaises(ZeroDivisionError):
            _ = v / 0.0
        with self.assertRaises(ZeroDivisionError):
            _ = v / -0.0

    def test_indexing_bounds_and_types(self):
        """Indexing out of range must raise IndexError; type mismatches raise TypeError."""
        v3 = Vec3(1, 2, 3)
        self.assertEqual(v3[-3], 1.0)
        with self.assertRaises(IndexError):
            _ = v3[3]
        with self.assertRaises(IndexError):
            _ = v3[-4]

        # Incompatible arithmetic types
        with self.assertRaises(TypeError):
            _ = v3 + 5.0
        with self.assertRaises(TypeError):
            _ = v3 - "foo"

    def test_std430_serialization_extremes(self):
        """Binary serialization roundtrip for subnormals and extremes in IEEE-754 float32."""
        v = Vec4(1.0e-30, -3.4e38, 0.0, 1.0)
        data = v.to_bytes()
        self.assertEqual(len(data), 16)
        unpacked = struct.unpack("<4f", data)
        self.assertEqual(unpacked[2], 0.0)
        self.assertEqual(unpacked[3], 1.0)
        self.assertAlmostEqual(unpacked[1], -3.4e38, delta=1e32)


class CameraAndControllerNumericalLimitsTests(unittest.TestCase):
    """Testing of Camera, frustum extraction, and OrbitController boundary conditions."""

    def test_look_at_near_collinear_and_extreme_distance(self):
        """Test Camera look_at with distant coordinates and near-collinear up vector."""
        # Eye at 1e6 units away
        eye = Vec3(1e6, 0.0, 0.0)
        target = Vec3(0.0, 0.0, 0.0)
        up = Vec3(0.0, 1.0, 0.0)
        cam = Camera(eye=eye, target=target, up=up)
        vm = cam.view_matrix
        # Origin mapped to camera space target should be at -Z distance 1e6
        pt = vm * Vec4(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(pt.z, -1e6, delta=1e-1)

    def test_perspective_extreme_fov_and_near_far_ratios(self):
        """Test perspective projection with very wide and narrow FOVs, large near/far ratio."""
        # Ultra wide FOV (179 degrees)
        p_wide = Mat4.perspective(math.radians(179.0), 1.0, 0.1, 1000.0)
        self.assertEqual(len(p_wide.elements), 16)

        # Ultra narrow FOV (0.1 degree)
        p_narrow = Mat4.perspective(math.radians(0.1), 1.0, 0.1, 1000.0)
        self.assertEqual(len(p_narrow.elements), 16)

        # Huge ratio near=0.001, far=100000.0
        p_ratio = Mat4.perspective(math.radians(60.0), 16.0 / 9.0, 0.001, 100000.0)
        # Near point should map to 0.0 clip Z
        near_pt = p_ratio * Vec4(0.0, 0.0, -0.001, 1.0)
        self.assertAlmostEqual(near_pt.z / near_pt.w, 0.0, places=4)
        # Far point should map to 1.0 clip Z
        far_pt = p_ratio * Vec4(0.0, 0.0, -100000.0, 1.0)
        self.assertAlmostEqual(far_pt.z / far_pt.w, 1.0, places=4)

    def test_frustum_planes_normalized_under_extreme_cams(self):
        """Frustum plane normals must always be unit length across varied configurations."""
        configs = [
            (Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0), 45.0, 1.0, 0.1, 100.0),
            (Vec3(10, 20, 30), Vec3(1, 2, 3), Vec3(0, 1, 0), 90.0, 2.35, 1.0, 5000.0),
            (Vec3(-50, 0, 0), Vec3(0, 0, 0), Vec3(0, 0, 1), 120.0, 0.5, 0.05, 50.0),
        ]
        for eye, target, up, fov, aspect, near, far in configs:
            cam = Camera(eye=eye, target=target, up=up, fov_y=fov, aspect=aspect, near=near, far=far)
            planes = cam.frustum_planes
            self.assertEqual(len(planes), 6)
            for p in planes:
                normal_len = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
                self.assertAlmostEqual(normal_len, 1.0, places=5)

    def test_orbit_controller_extreme_dt_and_damping(self):
        """OrbitController behavior under extreme dt and damping values."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, damping=0.1)

        # Extreme dt: dt = 1e6 should instantly converge without overshoot/instability
        ctrl.rotate(math.pi * 0.5, 0.0)
        ctrl.update(cam, dt=1e6)
        self.assertAlmostEqual(cam.eye.x, 5.0, places=5)
        self.assertAlmostEqual(cam.eye.z, 0.0, places=5)

    def test_orbit_controller_damping_dt_zero_preserves_position(self):
        """When damping is active (damping > 0), dt=0.0 must not advance position (alpha must be 0.0)."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, damping=0.5)
        ctrl.rotate(1.0, 0.0)

        # Before update: target_yaw is 1.0, current yaw is 0.0
        self.assertEqual(ctrl.yaw, 0.0)
        self.assertEqual(ctrl._target_yaw, 1.0)

        # Calling update with dt=0.0: zero elapsed time should produce zero motion
        ctrl.update(cam, dt=0.0)
        self.assertEqual(
            ctrl.yaw,
            0.0,
            f"OrbitController advanced yaw from 0.0 to {ctrl.yaw} during dt=0.0 frame with damping=0.5",
        )


if __name__ == "__main__":
    unittest.main()
