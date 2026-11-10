"""Comprehensive unit test suite for MiniGfx CPU math and camera controller.

Covers:
- Vector classes: Vec2, Vec3, Vec4 (arithmetic, dot, cross, norm, zero-vector handling, indexing, std430 serialization)
- Matrix class: Mat4 (identity, zero, translation, scaling, rotations, multiplication, transpose, analytic inverse, singular detection, serialization)
- Coordinate and projection conventions (right-handed look_at, [0, 1] clip Z perspective)
- Quaternion class: Quat (identity, from_axis_angle, to_mat4, multiplication, slerp, serialization)
- Camera & frustum culling (Gribb-Hartmann [0, 1] clip Z planes, sphere visibility)
- CameraParams std430 serialization (80 bytes, 16B alignment, round-trip)
- OrbitController (rotation, pitch clamping, pan, zoom, damping)
"""

from __future__ import annotations

import math
import struct
import unittest

from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.math import (
    Mat2,
    Mat2x2,
    Mat2x3,
    Mat2x4,
    Mat3,
    Mat3x2,
    Mat3x3,
    Mat3x4,
    Mat4,
    Mat4x2,
    Mat4x3,
    Mat4x4,
    Matrix,
    Quat,
    Vec2,
    Vec3,
    Vec4,
)


class TestVec2(unittest.TestCase):
    """Unit tests for Vec2."""

    def test_init_indexing_unpacking(self):
        v = Vec2(3.0, 4.0)
        self.assertEqual(v.x, 3.0)
        self.assertEqual(v.y, 4.0)
        self.assertEqual(v[0], 3.0)
        self.assertEqual(v[1], 4.0)
        self.assertEqual(v[-1], 4.0)
        self.assertEqual(v[-2], 3.0)
        with self.assertRaises(IndexError):
            _ = v[2]
        x, y = v
        self.assertEqual((x, y), (3.0, 4.0))
        self.assertEqual(len(v), 2)

    def test_arithmetic_operators(self):
        a = Vec2(1.0, 2.0)
        b = Vec2(3.0, 4.0)
        self.assertEqual(a + b, Vec2(4.0, 6.0))
        self.assertEqual(b - a, Vec2(2.0, 2.0))
        self.assertEqual(-a, Vec2(-1.0, -2.0))
        self.assertEqual(a * 2.5, Vec2(2.5, 5.0))
        self.assertEqual(2.5 * a, Vec2(2.5, 5.0))
        self.assertEqual(Vec2(5.0, 10.0) / 2.0, Vec2(2.5, 5.0))
        with self.assertRaises(ZeroDivisionError):
            _ = a / 0.0
        with self.assertRaises(TypeError):
            _ = a + 5

    def test_dot_length_length_sq(self):
        self.assertEqual(Vec2(1.0, 0.0).dot(Vec2(0.0, 1.0)), 0.0)
        self.assertEqual(Vec2(2.0, 3.0).dot(Vec2(4.0, 5.0)), 23.0)
        v = Vec2(3.0, 4.0)
        self.assertEqual(v.length_sq(), 25.0)
        self.assertEqual(v.length(), 5.0)

    def test_normalize_and_zero_safety(self):
        v = Vec2(3.0, 4.0).normalize()
        self.assertAlmostEqual(v.length(), 1.0, places=6)
        self.assertAlmostEqual(v.x, 0.6, places=6)
        self.assertAlmostEqual(v.y, 0.8, places=6)

        zero = Vec2(0.0, 0.0).normalize()
        self.assertFalse(math.isnan(zero.x))
        self.assertFalse(math.isnan(zero.y))
        self.assertEqual(zero, Vec2(0.0, 0.0))

    def test_equality_and_repr(self):
        self.assertEqual(Vec2(1.0, 2.0), Vec2(1.0, 2.0))
        self.assertEqual(Vec2(1.0, 2.0), (1.0, 2.0))
        self.assertNotEqual(Vec2(1.0, 2.0), Vec2(1.0, 2.01))
        self.assertIn("Vec2", repr(Vec2(1.0, 2.0)))

    def test_std430_serialization(self):
        data = Vec2(1.5, -2.5).to_bytes()
        self.assertEqual(len(data), 8)
        x, y = struct.unpack("<2f", data)
        self.assertEqual((x, y), (1.5, -2.5))


class TestVec3(unittest.TestCase):
    """Unit tests for Vec3."""

    def test_init_indexing_unpacking(self):
        v = Vec3(1.0, 2.0, 3.0)
        self.assertEqual((v.x, v.y, v.z), (1.0, 2.0, 3.0))
        self.assertEqual((v[0], v[1], v[2]), (1.0, 2.0, 3.0))
        self.assertEqual(v[-1], 3.0)
        with self.assertRaises(IndexError):
            _ = v[3]
        x, y, z = v
        self.assertEqual((x, y, z), (1.0, 2.0, 3.0))
        self.assertEqual(len(v), 3)

    def test_arithmetic_operators(self):
        a = Vec3(1.0, 2.0, 3.0)
        b = Vec3(4.0, 5.0, 6.0)
        self.assertEqual(a + b, Vec3(5.0, 7.0, 9.0))
        self.assertEqual(b - a, Vec3(3.0, 3.0, 3.0))
        self.assertEqual(-a, Vec3(-1.0, -2.0, -3.0))
        self.assertEqual(a * 2.0, Vec3(2.0, 4.0, 6.0))
        self.assertEqual(2.0 * a, Vec3(2.0, 4.0, 6.0))
        self.assertEqual(b / 2.0, Vec3(2.0, 2.5, 3.0))
        with self.assertRaises(ZeroDivisionError):
            _ = a / 0.0

    def test_dot_cross_product(self):
        x = Vec3(1.0, 0.0, 0.0)
        y = Vec3(0.0, 1.0, 0.0)
        z = Vec3(0.0, 0.0, 1.0)
        self.assertEqual(x.dot(y), 0.0)
        self.assertEqual(x.cross(y), z)
        self.assertEqual(y.cross(z), x)
        self.assertEqual(z.cross(x), y)
        self.assertEqual(y.cross(x), -z)
        self.assertEqual(Vec3(2, 4, 6).cross(Vec3(1, 2, 3)), Vec3(0, 0, 0))

    def test_length_and_normalize(self):
        v = Vec3(1.0, 2.0, 2.0)
        self.assertEqual(v.length_sq(), 9.0)
        self.assertEqual(v.length(), 3.0)
        norm = v.normalize()
        self.assertAlmostEqual(norm.length(), 1.0, places=6)

        zero = Vec3(0.0, 0.0, 0.0).normalize()
        self.assertFalse(math.isnan(zero.x))
        self.assertEqual(zero, Vec3(0.0, 0.0, 0.0))

    def test_std430_serialization(self):
        data = Vec3(1.0, 2.0, 3.0).to_bytes()
        self.assertEqual(len(data), 12)
        unpacked = struct.unpack("<3f", data)
        self.assertEqual(unpacked, (1.0, 2.0, 3.0))


class TestVec4(unittest.TestCase):
    """Unit tests for Vec4."""

    def test_init_indexing_unpacking(self):
        v = Vec4(1.0, 2.0, 3.0, 4.0)
        self.assertEqual((v.x, v.y, v.z, v.w), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual((v[0], v[1], v[2], v[3]), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(v[-1], 4.0)
        with self.assertRaises(IndexError):
            _ = v[4]
        x, y, z, w = v
        self.assertEqual((x, y, z, w), (1.0, 2.0, 3.0, 4.0))
        self.assertEqual(len(v), 4)

    def test_arithmetic_and_dot(self):
        a = Vec4(1.0, 2.0, 3.0, 4.0)
        b = Vec4(5.0, 6.0, 7.0, 8.0)
        self.assertEqual(a + b, Vec4(6.0, 8.0, 10.0, 12.0))
        self.assertEqual(b - a, Vec4(4.0, 4.0, 4.0, 4.0))
        self.assertEqual(a * 2.0, Vec4(2.0, 4.0, 6.0, 8.0))
        self.assertEqual(a.dot(b), 1*5 + 2*6 + 3*7 + 4*8)

    def test_length_and_normalize(self):
        v = Vec4(2.0, 2.0, 2.0, 2.0)
        self.assertEqual(v.length_sq(), 16.0)
        self.assertEqual(v.length(), 4.0)
        self.assertAlmostEqual(v.normalize().length(), 1.0, places=6)

        zero = Vec4(0.0, 0.0, 0.0, 0.0).normalize()
        self.assertFalse(math.isnan(zero.x))
        self.assertEqual(zero, Vec4(0.0, 0.0, 0.0, 0.0))

    def test_std430_serialization(self):
        data = Vec4(1.0, 2.0, 3.0, 4.0).to_bytes()
        self.assertEqual(len(data), 16)
        self.assertEqual(struct.unpack("<4f", data), (1.0, 2.0, 3.0, 4.0))


class TestMat4(unittest.TestCase):
    """Unit tests for Mat4 matrix operations."""

    def test_identity_and_zero(self):
        i = Mat4.identity()
        self.assertEqual(len(i.elements), 16)
        self.assertEqual(len(i), 16)
        for r in range(4):
            for c in range(4):
                expected = 1.0 if r == c else 0.0
                self.assertEqual(i[r, c], expected)

        z = Mat4.zero()
        self.assertTrue(all(x == 0.0 for x in z.elements))

    def test_translation(self):
        t = Mat4.translation(10.0, -5.0, 2.0)
        p = t * Vec4(0.0, 0.0, 0.0, 1.0)
        self.assertEqual(p, Vec4(10.0, -5.0, 2.0, 1.0))
        # Direction vector (w=0) unaffected by translation
        d = t * Vec4(1.0, 2.0, 3.0, 0.0)
        self.assertEqual(d, Vec4(1.0, 2.0, 3.0, 0.0))

        t2 = Mat4.from_translation(Vec3(10.0, -5.0, 2.0))
        self.assertEqual(t, t2)

    def test_scaling(self):
        s = Mat4.scaling(2.0, 3.0, 4.0)
        p = s * Vec4(1.0, 1.0, 1.0, 1.0)
        self.assertEqual(p, Vec4(2.0, 3.0, 4.0, 1.0))
        # Uniform scaling
        su = Mat4.scaling(3.0)
        self.assertEqual(su * Vec4(1.0, 2.0, 3.0, 1.0), Vec4(3.0, 6.0, 9.0, 1.0))

    def test_rotations_axis_aligned(self):
        rx = Mat4.rotation_x(math.pi * 0.5)
        p = rx * Vec4(0.0, 1.0, 0.0, 1.0)
        self.assertAlmostEqual(p.x, 0.0, places=5)
        self.assertAlmostEqual(p.y, 0.0, places=5)
        self.assertAlmostEqual(p.z, 1.0, places=5)

        ry = Mat4.rotation_y(math.pi * 0.5)
        p2 = ry * Vec4(0.0, 0.0, 1.0, 1.0)
        self.assertAlmostEqual(p2.x, 1.0, places=5)
        self.assertAlmostEqual(p2.y, 0.0, places=5)
        self.assertAlmostEqual(p2.z, 0.0, places=5)

        rz = Mat4.rotation_z(math.pi * 0.5)
        p3 = rz * Vec4(1.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(p3.x, 0.0, places=5)
        self.assertAlmostEqual(p3.y, 1.0, places=5)
        self.assertAlmostEqual(p3.z, 0.0, places=5)

    def test_rotation_axis(self):
        r_axis = Mat4.rotation_axis(Vec3(0, 1, 0), math.pi * 0.5)
        r_y = Mat4.rotation_y(math.pi * 0.5)
        self.assertEqual(r_axis, r_y)
        # Degenerate zero axis
        self.assertEqual(Mat4.rotation_axis(Vec3(0, 0, 0), 1.0), Mat4.identity())

    def test_multiplication_associativity_and_order(self):
        t = Mat4.translation(1, 2, 3)
        r = Mat4.rotation_y(math.pi * 0.5)
        s = Mat4.scaling(2, 2, 2)
        # Associativity: (T * R) * S == T * (R * S)
        self.assertEqual((t * r) * s, t * (r * s))
        # Non-commutativity: T * R != R * T
        self.assertNotEqual(t * r, r * t)

    def test_transpose(self):
        m = Mat4((
            1, 2, 3, 4,
            5, 6, 7, 8,
            9, 10, 11, 12,
            13, 14, 15, 16,
        ))
        mt = m.transpose()
        self.assertEqual(mt.transpose(), m)
        for r in range(4):
            for c in range(4):
                self.assertEqual(mt[r, c], m[c, r])

    def test_analytic_inverse(self):
        i = Mat4.identity()
        self.assertEqual(i.inverse(), i)

        # Composite SRT matrix inverse
        t = Mat4.translation(5.0, -3.0, 2.0)
        r = Mat4.rotation_z(0.7)
        s = Mat4.scaling(2.0, 0.5, 3.0)
        m = t * r * s
        inv = m.inverse()
        prod = m * inv
        for idx in range(16):
            expected = 1.0 if idx in (0, 5, 10, 15) else 0.0
            self.assertAlmostEqual(prod.elements[idx], expected, places=5)

    def test_singular_matrix_inversion(self):
        with self.assertRaises(ValueError):
            Mat4.zero().inverse()
        with self.assertRaises(ValueError):
            Mat4.scaling(0.0, 1.0, 1.0).inverse()

    def test_columns_and_elements_and_bytes(self):
        m = Mat4.identity()
        cols = m.columns
        self.assertEqual(len(cols), 4)
        self.assertIsInstance(cols[0], Vec4)
        self.assertEqual(cols[0], Vec4(1, 0, 0, 0))
        self.assertEqual(cols[1], Vec4(0, 1, 0, 0))

        raw = m.to_bytes()
        self.assertEqual(len(raw), 64)
        restored = Mat4.from_bytes(raw)
        self.assertEqual(restored, m)


class TestCoordinateAndProjections(unittest.TestCase):
    """Unit tests for right-handed coordinate frame and [0, 1] clip Z projection."""

    def test_look_at_right_handed(self):
        eye = Vec3(0, 0, 5)
        target = Vec3(0, 0, 0)
        up = Vec3(0, 1, 0)
        v = Mat4.look_at(eye, target, up)

        # Eye maps to origin of camera space
        cam_eye = v * Vec4(0, 0, 5, 1)
        self.assertAlmostEqual(cam_eye.x, 0.0, places=5)
        self.assertAlmostEqual(cam_eye.y, 0.0, places=5)
        self.assertAlmostEqual(cam_eye.z, 0.0, places=5)

        # Target maps to -Z (5 units along -Z)
        cam_target = v * Vec4(0, 0, 0, 1)
        self.assertAlmostEqual(cam_target.x, 0.0, places=5)
        self.assertAlmostEqual(cam_target.y, 0.0, places=5)
        self.assertAlmostEqual(cam_target.z, -5.0, places=5)

        # Up direction maps along camera +Y
        cam_up = v * Vec4(0, 1, 0, 0)
        self.assertAlmostEqual(cam_up.x, 0.0, places=5)
        self.assertAlmostEqual(cam_up.y, 1.0, places=5)
        self.assertAlmostEqual(cam_up.z, 0.0, places=5)

    def test_look_at_exceptions(self):
        # Coincident eye and target
        with self.assertRaises(ValueError):
            Mat4.look_at(Vec3(1, 2, 3), Vec3(1, 2, 3), Vec3(0, 1, 0))
        # Zero-length up
        with self.assertRaises(ValueError):
            Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 0, 0))
        # Collinear up and view direction
        with self.assertRaises(ValueError):
            Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 0, 1))

    def test_perspective_clip_z_zero_to_one(self):
        near = 0.5
        far = 100.0
        p = Mat4.perspective(math.radians(60.0), 16.0 / 9.0, near, far)

        # Point on near plane (z = -near) maps to clip z/w = 0.0
        near_pt = p * Vec4(0.0, 0.0, -near, 1.0)
        self.assertAlmostEqual(near_pt.z / near_pt.w, 0.0, places=5)

        # Point on far plane (z = -far) maps to clip z/w = 1.0
        far_pt = p * Vec4(0.0, 0.0, -far, 1.0)
        self.assertAlmostEqual(far_pt.z / far_pt.w, 1.0, places=5)

        # Mid-range point (z = -10) maps inside (0.0, 1.0)
        mid_pt = p * Vec4(0.0, 0.0, -10.0, 1.0)
        z_ndc = mid_pt.z / mid_pt.w
        self.assertTrue(0.0 < z_ndc < 1.0)

    def test_perspective_invalid_parameters(self):
        with self.assertRaises(ValueError):
            Mat4.perspective(0.0, 1.0, 0.1, 100.0)  # fov_y <= 0
        with self.assertRaises(ValueError):
            Mat4.perspective(math.pi, 1.0, 0.1, 100.0)  # fov_y >= pi
        with self.assertRaises(ValueError):
            Mat4.perspective(math.radians(60), -1.0, 0.1, 100.0)  # aspect <= 0
        with self.assertRaises(ValueError):
            Mat4.perspective(math.radians(60), 1.0, -0.1, 100.0)  # near <= 0
        with self.assertRaises(ValueError):
            Mat4.perspective(math.radians(60), 1.0, 10.0, 5.0)  # far <= near


class TestQuaternion(unittest.TestCase):
    """Unit tests for Quat quaternion operations."""

    def test_identity(self):
        q = Quat.identity()
        self.assertEqual((q.x, q.y, q.z, q.w), (0.0, 0.0, 0.0, 1.0))
        self.assertEqual(q.to_mat4(), Mat4.identity())

    def test_from_axis_angle_and_to_mat4(self):
        # 90 degrees around Y axis
        q = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi * 0.5)
        m_q = q.to_mat4()
        m_rot = Mat4.rotation_y(math.pi * 0.5)
        self.assertEqual(m_q, m_rot)

        # 90 degrees around X axis
        qx = Quat.from_axis_angle(Vec3(1, 0, 0), math.pi * 0.5)
        self.assertEqual(qx.to_mat4(), Mat4.rotation_x(math.pi * 0.5))

    def test_multiplication_and_conjugate(self):
        q = Quat.from_axis_angle(Vec3(0, 1, 0), 1.2)
        q_conj = q.conjugate()
        prod = q * q_conj
        self.assertEqual(prod.normalize(), Quat.identity())

    def test_slerp(self):
        q1 = Quat.identity()
        q2 = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi * 0.5)

        # t=0 returns q1
        self.assertEqual(q1.slerp(q2, 0.0), q1)
        # t=1 returns q2
        self.assertEqual(q1.slerp(q2, 1.0), q2)

        # t=0.5 returns 45 degree rotation
        mid = q1.slerp(q2, 0.5)
        expected_mid = Quat.from_axis_angle(Vec3(0, 1, 0), math.pi * 0.25)
        self.assertEqual(mid, expected_mid)

        # Shortest-path antipodal handling: -q2 should produce the same midpoint
        q2_neg = Quat(-q2.x, -q2.y, -q2.z, -q2.w)
        mid_antipodal = q1.slerp(q2_neg, 0.5)
        self.assertEqual(mid_antipodal, expected_mid)

    def test_std430_serialization(self):
        data = Quat(1, 2, 3, 4).to_bytes()
        self.assertEqual(len(data), 16)
        self.assertEqual(struct.unpack("<4f", data), (1.0, 2.0, 3.0, 4.0))


class TestCameraAndFrustum(unittest.TestCase):
    """Unit tests for Camera view_proj and frustum extraction."""

    def test_camera_initialization_and_auto_radians(self):
        # 60 degrees specified in degrees
        cam = Camera(fov_y=60.0)
        self.assertAlmostEqual(cam.fov_y, math.radians(60.0), places=5)
        # Setting in radians
        cam.fov_y = 1.0
        self.assertEqual(cam.fov_y, 1.0)
        # Setting in degrees
        cam.fov_y = 90.0
        self.assertAlmostEqual(cam.fov_y, math.radians(90.0), places=5)

    def test_camera_view_proj(self):
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0), up=Vec3(0, 1, 0))
        self.assertEqual(cam.view_proj, cam.proj_matrix * cam.view_matrix)

    def test_frustum_planes_and_sphere_culling(self):
        cam = Camera(
            eye=Vec3(0, 0, 5),
            target=Vec3(0, 0, 0),
            up=Vec3(0, 1, 0),
            fov_y=60.0,
            aspect=1.0,
            near=0.1,
            far=100.0,
        )
        planes = cam.frustum_planes
        self.assertEqual(len(planes), 6)
        # Every plane normal is unit-length
        for p in planes:
            n_len = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            self.assertAlmostEqual(n_len, 1.0, places=5)

        # Explicit frustum plane symmetry assertions verifying vertical planes have Y=0.0
        # and horizontal planes have X=0.0
        self.assertAlmostEqual(planes[0].y, 0.0, places=5)  # Left
        self.assertAlmostEqual(planes[1].y, 0.0, places=5)  # Right
        self.assertAlmostEqual(planes[2].x, 0.0, places=5)  # Bottom
        self.assertAlmostEqual(planes[3].x, 0.0, places=5)  # Top

        # Target is directly inside the frustum
        self.assertTrue(cam.is_sphere_visible(Vec3(0, 0, 0), radius=1.0))
        # Point behind the camera
        self.assertFalse(cam.is_sphere_visible(Vec3(0, 0, 10), radius=1.0))
        # Point far beyond the far plane
        self.assertFalse(cam.is_sphere_visible(Vec3(0, 0, -200), radius=1.0))
        # Sphere intersecting near plane boundary (camera at 5, near=0.1 -> near plane is at 4.9)
        self.assertTrue(cam.is_sphere_visible(Vec3(0, 0, 4.8), radius=0.2))
        # Sphere completely outside to the right
        self.assertFalse(cam.is_sphere_visible(Vec3(100, 0, 0), radius=1.0))

    def test_frustum_off_axis_quadrants(self):
        """Test points in all 4 quadrants inside and outside frustum boundaries."""
        cam = Camera(
            eye=Vec3(0, 0, 5),
            target=Vec3(0, 0, 0),
            up=Vec3(0, 1, 0),
            fov_y=60.0,
            aspect=1.0,
            near=0.1,
            far=100.0,
        )
        # At z=0 (distance 5 from eye), frustum half-extents are 5 * tan(30 deg) ~= 2.88675.
        # Test all 4 quadrants inside the frustum (|x|=2.0, |y|=2.0 < 2.88675)
        self.assertTrue(cam.is_sphere_visible(Vec3(2.0, 2.0, 0.0), radius=0.1))    # Q1: (+x, +y)
        self.assertTrue(cam.is_sphere_visible(Vec3(-2.0, 2.0, 0.0), radius=0.1))   # Q2: (-x, +y)
        self.assertTrue(cam.is_sphere_visible(Vec3(-2.0, -2.0, 0.0), radius=0.1))  # Q3: (-x, -y)
        self.assertTrue(cam.is_sphere_visible(Vec3(2.0, -2.0, 0.0), radius=0.1))   # Q4: (+x, -y)

        # Test points outside each boundary across all 4 quadrants
        # Q1 (+x, +y) boundaries:
        self.assertFalse(cam.is_sphere_visible(Vec3(4.0, 2.0, 0.0), radius=0.1))   # outside right plane
        self.assertFalse(cam.is_sphere_visible(Vec3(2.0, 4.0, 0.0), radius=0.1))   # outside top plane

        # Q2 (-x, +y) boundaries:
        self.assertFalse(cam.is_sphere_visible(Vec3(-4.0, 2.0, 0.0), radius=0.1))  # outside left plane
        self.assertFalse(cam.is_sphere_visible(Vec3(-2.0, 4.0, 0.0), radius=0.1))  # outside top plane

        # Q3 (-x, -y) boundaries:
        self.assertFalse(cam.is_sphere_visible(Vec3(-4.0, -2.0, 0.0), radius=0.1)) # outside left plane
        self.assertFalse(cam.is_sphere_visible(Vec3(-2.0, -4.0, 0.0), radius=0.1)) # outside bottom plane

        # Q4 (+x, -y) boundaries:
        self.assertFalse(cam.is_sphere_visible(Vec3(4.0, -2.0, 0.0), radius=0.1))  # outside right plane
        self.assertFalse(cam.is_sphere_visible(Vec3(2.0, -4.0, 0.0), radius=0.1))  # outside bottom plane

    def test_frustum_8_corners_unprojected_ndc(self):
        """Unproject canonical NDC corners and verify they are inside or on frustum boundary."""
        cam = Camera(
            eye=Vec3(0, 0, 5),
            target=Vec3(0, 0, 0),
            up=Vec3(0, 1, 0),
            fov_y=60.0,
            aspect=1.0,
            near=0.1,
            far=100.0,
        )
        inv_vp = cam.view_proj.inverse()
        ndc_corners = [
            (-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0),
            (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (1.0, 1.0, 1.0),
        ]
        for ndc_x, ndc_y, ndc_z in ndc_corners:
            world4 = inv_vp * Vec4(ndc_x, ndc_y, ndc_z, 1.0)
            self.assertNotEqual(world4.w, 0.0)
            inv_w = 1.0 / world4.w
            corner = Vec3(world4.x * inv_w, world4.y * inv_w, world4.z * inv_w)
            self.assertTrue(
                cam.is_sphere_visible(corner, radius=0.0),
                f"Frustum corner at NDC {(ndc_x, ndc_y, ndc_z)} was falsely culled",
            )


class TestCameraParamsStd430(unittest.TestCase):
    """Unit tests for CameraParams std430 binary packing."""

    def test_camera_params_layout_and_size(self):
        self.assertEqual(CameraParams.__struct_size__, 80)
        self.assertEqual(CameraParams.__struct_alignment__, 16)

        cam = Camera(eye=Vec3(1.0, 2.0, 3.0), target=Vec3(0, 0, 0))
        params = CameraParams.from_camera(cam, time=2.5)
        raw = params.to_bytes()
        self.assertEqual(len(raw), 80)

        # Unpack offset 0 (Mat4, 64B)
        vp_elems = struct.unpack_from("<16f", raw, 0)
        for a, b in zip(vp_elems, cam.view_proj.elements):
            self.assertAlmostEqual(a, b, places=5)

        # Unpack offset 64 (Vec3, 12B)
        eye_elems = struct.unpack_from("<3f", raw, 64)
        self.assertEqual(eye_elems, (1.0, 2.0, 3.0))

        # Unpack offset 76 (f32, 4B)
        time_val = struct.unpack_from("<f", raw, 76)[0]
        self.assertAlmostEqual(time_val, 2.5, places=5)

        # Round-trip restoration
        restored = CameraParams.from_bytes(raw)
        self.assertEqual(restored.eye_pos, cam.eye)
        self.assertEqual(restored.view_proj, cam.view_proj)
        self.assertAlmostEqual(restored.time, 2.5, places=5)


class TestOrbitController(unittest.TestCase):
    """Unit tests for OrbitController orbital navigation and damping."""

    def test_initial_orbit_from_camera(self):
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController.from_camera(cam, damping=0.1)
        self.assertAlmostEqual(ctrl.distance, 5.0, places=5)
        self.assertAlmostEqual(ctrl.yaw, 0.0, places=5)
        self.assertAlmostEqual(ctrl.pitch, 0.0, places=5)

    def test_rotate_and_update(self):
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, damping=0.0)
        # Rotate 90 degrees around Y (yaw)
        ctrl.rotate(math.pi * 0.5, 0.0)
        ctrl.update(cam, dt=0.0)
        self.assertAlmostEqual(cam.eye.x, 5.0, places=5)
        self.assertAlmostEqual(cam.eye.y, 0.0, places=5)
        self.assertAlmostEqual(cam.eye.z, 0.0, places=5)

    def test_pitch_clamping(self):
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, min_pitch=-1.5, max_pitch=1.5, damping=0.0)
        # Apply excessive pitch rotation
        ctrl.rotate(0.0, 10.0)
        self.assertAlmostEqual(ctrl.pitch, 1.5, places=5)
        ctrl.rotate(0.0, -20.0)
        self.assertAlmostEqual(ctrl.pitch, -1.5, places=5)

    def test_zoom_clamping(self):
        ctrl = OrbitController(distance=5.0, min_distance=1.0, max_distance=10.0, damping=0.0)
        ctrl.zoom(-100.0)
        self.assertEqual(ctrl.distance, 1.0)
        ctrl.zoom(+100.0)
        self.assertEqual(ctrl.distance, 10.0)

    def test_panning(self):
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0), up=Vec3(0, 1, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, damping=0.0)
        ctrl.pan(1.0, 2.0, cam)
        ctrl.update(cam, dt=0.0)
        self.assertAlmostEqual(cam.target.x, 1.0, places=5)
        self.assertAlmostEqual(cam.target.y, 2.0, places=5)

    def test_damping_dt_zero_preserves_position(self):
        """When damping > 0.0 and dt = 0.0, controller preserves camera eye and target coordinates without snapping."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, damping=0.1)

        # Apply rotation, panning, and zoom to target state
        ctrl.rotate(math.pi * 0.5, 0.2)
        ctrl.pan(1.0, 2.0, cam)
        ctrl.zoom(2.0)

        init_eye = (cam.eye.x, cam.eye.y, cam.eye.z)
        init_target = (cam.target.x, cam.target.y, cam.target.z)

        # Update with dt=0.0: zero elapsed time must preserve current state exactly
        ctrl.update(cam, dt=0.0)

        self.assertEqual(ctrl.yaw, 0.0)
        self.assertEqual(ctrl.pitch, 0.0)
        self.assertEqual(ctrl.distance, 5.0)
        self.assertEqual(ctrl.target, Vec3(0, 0, 0))
        self.assertEqual((cam.eye.x, cam.eye.y, cam.eye.z), init_eye)
        self.assertEqual((cam.target.x, cam.target.y, cam.target.z), init_target)

    def test_damping_disabled_instant_snapping(self):
        """When damping <= 0.0, update instantly snaps camera eye and target to targets."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, damping=0.0)

        ctrl.rotate(math.pi * 0.5, 0.0)
        ctrl.pan(1.0, 2.0, cam)
        ctrl.zoom(2.0)

        ctrl.update(cam, dt=0.0)
        self.assertEqual(ctrl.yaw, ctrl._target_yaw)
        self.assertEqual(ctrl.pitch, ctrl._target_pitch)
        self.assertEqual(ctrl.distance, ctrl._target_distance)
        self.assertEqual(ctrl.target, ctrl._target_pos)
        self.assertAlmostEqual(cam.target.x, 1.0, places=5)
        self.assertAlmostEqual(cam.target.y, 2.0, places=5)
        self.assertAlmostEqual(cam.eye.x, 8.0, places=5)
        self.assertAlmostEqual(cam.eye.y, 2.0, places=5)
        self.assertAlmostEqual(cam.eye.z, 0.0, places=5)


class TestCxRMatrices(unittest.TestCase):
    """Comprehensive unit and property tests for the complete set of CxR matrix types."""

    ALL_TYPES = [
        (Mat2x2, 2, 2),
        (Mat2x3, 2, 3),
        (Mat2x4, 2, 4),
        (Mat3x2, 3, 2),
        (Mat3x3, 3, 3),
        (Mat3x4, 3, 4),
        (Mat4x2, 4, 2),
        (Mat4x3, 4, 3),
        (Mat4x4, 4, 4),
    ]

    def test_instantiation_all_9_types(self):
        """Verify default identity instantiation, dimensions, len, and aliases for all 9 types."""
        self.assertIs(Mat2, Mat2x2)
        self.assertIs(Mat3, Mat3x3)
        self.assertIs(Mat4, Mat4x4)

        for cls, cols, rows in self.ALL_TYPES:
            # Default instantiation -> identity
            m = cls()
            self.assertEqual(m.cols, cols)
            self.assertEqual(m.rows, rows)
            self.assertEqual(len(m), cols * rows)
            self.assertEqual(len(m.elements), cols * rows)

            # Check identity values: 1.0 where col == row, 0.0 elsewhere
            for c in range(cols):
                for r in range(rows):
                    expected = 1.0 if c == r else 0.0
                    self.assertEqual(m[r, c], expected, f"Failed at ({r}, {c}) for {cls.__name__}")

            # Generic Matrix(cols, rows) constructor returns appropriate type
            m_gen = Matrix(cols, rows)
            self.assertIsInstance(m_gen, cls)
            self.assertEqual(m_gen, m)

            # Zero matrix constructor
            z = cls.zero()
            self.assertTrue(all(x == 0.0 for x in z))
            self.assertEqual(len(z), cols * rows)

            # Sequence constructor and flat arguments constructor
            sample_data = [float(i + 1) for i in range(cols * rows)]
            m_seq = cls(sample_data)
            m_flat = cls(*sample_data)
            self.assertEqual(m_seq, m_flat)
            self.assertEqual(list(m_seq), sample_data)

            # Invalid argument counts raise ValueError
            with self.assertRaises(ValueError):
                cls(sample_data[: cols * rows - 1])
            with self.assertRaises(ValueError):
                cls(*(sample_data + [99.0]))

    def test_indexing_1d_and_2d_and_columns(self):
        """Verify 1D, 2D (row, col), negative indexing, and column vector properties."""
        for cls, cols, rows in self.ALL_TYPES:
            sample_data = [float((c + 1) * 10 + (r + 1)) for c in range(cols) for r in range(rows)]
            m = cls(sample_data)

            # 1D indexing
            for i in range(cols * rows):
                self.assertEqual(m[i], sample_data[i])
                self.assertEqual(m[i - cols * rows], sample_data[i])

            # 2D indexing (r, c)
            for c in range(cols):
                for r in range(rows):
                    flat_idx = c * rows + r
                    self.assertEqual(m[r, c], sample_data[flat_idx])

            # Negative 2D indexing
            self.assertEqual(m[-1, -1], m[rows - 1, cols - 1])

            # Index out of bounds
            with self.assertRaises(IndexError):
                _ = m[rows, 0]
            with self.assertRaises(IndexError):
                _ = m[0, cols]
            with self.assertRaises(IndexError):
                _ = m[cols * rows]
            with self.assertRaises(IndexError):
                _ = m[-cols * rows - 1]
            with self.assertRaises(TypeError):
                _ = m["invalid"]

            # Columns property returning typed Vec2, Vec3, or Vec4
            cols_vecs = m.columns
            self.assertEqual(len(cols_vecs), cols)
            vec_cls = {2: Vec2, 3: Vec3, 4: Vec4}[rows]
            for c in range(cols):
                self.assertIsInstance(cols_vecs[c], vec_cls)
                for r in range(rows):
                    self.assertEqual(cols_vecs[c][r], m[r, c])

    def test_matmul_dimension_compatibility(self):
        """Verify matmul between compatible dimensions and ValueError for incompatible dimensions."""
        for cls_a, c_a, r_a in self.ALL_TYPES:
            for cls_b, c_b, r_b in self.ALL_TYPES:
                a_data = [(i + 1) * 0.5 for i in range(c_a * r_a)]
                b_data = [(i + 2) * 0.25 for i in range(c_b * r_b)]
                A = cls_a(a_data)
                B = cls_b(b_data)

                if c_a == r_b:
                    # Compatible: A @ B should have shape (c_b, r_a)
                    C1 = A @ B
                    C2 = A * B
                    self.assertEqual(C1, C2)
                    self.assertEqual(C1.cols, c_b)
                    self.assertEqual(C1.rows, r_a)

                    # Verify values analytically
                    for c in range(c_b):
                        for r in range(r_a):
                            expected = sum(A[r, k] * B[k, c] for k in range(c_a))
                            self.assertAlmostEqual(
                                C1[r, c], expected, places=5,
                                msg=f"Matmul mismatch at ({r}, {c}) for {cls_a.__name__} @ {cls_b.__name__}",
                            )
                else:
                    # Incompatible: must raise ValueError
                    with self.assertRaises(ValueError):
                        _ = A @ B
                    with self.assertRaises(ValueError):
                        _ = A * B

    def test_matrix_vector_multiplication(self):
        """Verify matrix-vector multiplication for compatible sizes and error handling for incompatible sizes."""
        vec_map = {2: Vec2(1.5, -2.5), 3: Vec3(1.0, 2.0, 3.0), 4: Vec4(1.0, -1.0, 2.0, -2.0)}
        vec_types = {2: Vec2, 3: Vec3, 4: Vec4}

        for cls, cols, rows in self.ALL_TYPES:
            m = cls([(i + 1) * 1.5 for i in range(cols * rows)])

            # Compatible vector multiplication
            v = vec_map[cols]
            res = m * v
            self.assertIsInstance(res, vec_types[rows])
            self.assertEqual(len(res), rows)

            for r in range(rows):
                expected = sum(m[r, c] * v[c] for c in range(cols))
                self.assertAlmostEqual(res[r], expected, places=5)

            # Incompatible vector dimensions raise ValueError
            for wrong_dim, wrong_vec in vec_map.items():
                if wrong_dim != cols:
                    with self.assertRaises(ValueError):
                        _ = m * wrong_vec

    def test_transposition_properties(self):
        """Verify transposition properties (A.T).T == A and (A @ B).T == B.T @ A.T."""
        for cls, cols, rows in self.ALL_TYPES:
            sample_data = [(i + 1) * 0.7 for i in range(cols * rows)]
            A = cls(sample_data)
            AT = A.transpose()

            self.assertEqual(AT.cols, rows)
            self.assertEqual(AT.rows, cols)
            self.assertEqual(A.T, AT)

            # Property 1: (A.T).T == A
            self.assertEqual(AT.transpose(), A)

            # Element-wise check: AT[r, c] == A[c, r]
            for r in range(AT.rows):
                for c in range(AT.cols):
                    self.assertAlmostEqual(AT[r, c], A[c, r], places=6)

        # Property 2: (A @ B).T == B.T @ A.T for all compatible dimension pairs
        for cls_a, c_a, r_a in self.ALL_TYPES:
            for cls_b, c_b, r_b in self.ALL_TYPES:
                if c_a == r_b:
                    A = cls_a([(i + 1) * 0.3 for i in range(c_a * r_a)])
                    B = cls_b([(i + 1) * 0.4 for i in range(c_b * r_b)])
                    AB_T = (A @ B).T
                    BT_AT = B.T @ A.T
                    self.assertEqual(AB_T, BT_AT)

    def test_std430_packing_roundtrip_and_strides(self):
        """Verify std430 packing byte length, column strides, 4-byte padding for 3-row matrices, and roundtrip."""
        for cls, cols, rows in self.ALL_TYPES:
            m = cls([(i + 1) * 1.25 for i in range(cols * rows)])
            raw = m.to_bytes()

            stride = 8 if rows == 2 else 16
            expected_total_bytes = cols * stride
            self.assertEqual(
                len(raw), expected_total_bytes,
                f"Unexpected std430 byte size for {cls.__name__}: got {len(raw)}, expected {expected_total_bytes}",
            )

            # If rows == 3, verify the 4th float in each 16-byte column is 0x00 padding
            if rows == 3:
                for col_idx in range(cols):
                    pad_bytes = raw[col_idx * 16 + 12 : col_idx * 16 + 16]
                    self.assertEqual(
                        pad_bytes, b"\x00\x00\x00\x00",
                        f"Column {col_idx} missing 4-byte zero padding in {cls.__name__}",
                    )

            # Roundtrip deserialization
            restored = cls.from_bytes(raw)
            self.assertEqual(restored, m)

            # Deserialization from larger buffer with offset 0
            padded_buffer = raw + b"\xcd" * 64
            restored_padded = cls.from_bytes(padded_buffer)
            self.assertEqual(restored_padded, m)

            # Deserialization with insufficient buffer raises ValueError
            with self.assertRaises(ValueError):
                cls.from_bytes(raw[: expected_total_bytes - 1])

    def test_scalar_arithmetic_and_addition(self):
        """Verify scalar multiplication, division, addition, subtraction, and negation."""
        for cls, cols, rows in self.ALL_TYPES:
            m = cls([(i + 1) * 2.0 for i in range(cols * rows)])

            # Scalar multiplication
            scaled1 = m * 3.0
            scaled2 = 3.0 * m
            self.assertEqual(scaled1, scaled2)
            self.assertEqual(list(scaled1), [(i + 1) * 6.0 for i in range(cols * rows)])

            # Scalar division
            div = m / 2.0
            self.assertEqual(list(div), [(i + 1) * 1.0 for i in range(cols * rows)])
            with self.assertRaises(ZeroDivisionError):
                _ = m / 0.0

            # Addition and subtraction with matching shape
            m2 = cls([1.0] * (cols * rows))
            added = m + m2
            subbed = m - m2
            self.assertEqual(list(added), [(i + 1) * 2.0 + 1.0 for i in range(cols * rows)])
            self.assertEqual(list(subbed), [(i + 1) * 2.0 - 1.0 for i in range(cols * rows)])

            # Negation
            neg = -m
            self.assertEqual(m + neg, cls.zero())

        # Incompatible shapes for addition raise ValueError
        with self.assertRaises(ValueError):
            _ = Mat2x3() + Mat3x2()

    def test_analytic_inverse_square_matrices(self):
        """Verify analytic matrix inverse for 2x2, 3x3, and 4x4, singular detection, and non-square rejection."""
        # 2x2 inverse
        m2 = Mat2x2(4.0, 7.0, 2.0, 6.0)
        inv2 = m2.inverse()
        self.assertEqual(m2 @ inv2, Mat2x2.identity())
        self.assertEqual(inv2 @ m2, Mat2x2.identity())

        # 3x3 inverse
        m3 = Mat3x3(
            1.0, 0.0, 5.0,
            2.0, 1.0, 6.0,
            3.0, 4.0, 0.0,
        )
        inv3 = m3.inverse()
        self.assertEqual(m3 @ inv3, Mat3x3.identity())
        self.assertEqual(inv3 @ m3, Mat3x3.identity())

        # 4x4 inverse
        m4 = Mat4x4.scaling(2.0, 3.0, 4.0)
        inv4 = m4.inverse()
        self.assertEqual(m4 @ inv4, Mat4x4.identity())

        # Singular matrix raises ValueError
        with self.assertRaises(ValueError):
            Mat2x2.zero().inverse()
        with self.assertRaises(ValueError):
            Mat3x3.zero().inverse()
        with self.assertRaises(ValueError):
            Mat4x4.zero().inverse()

        # Non-square matrices raise ValueError
        with self.assertRaises(ValueError):
            Mat2x3().inverse()
        with self.assertRaises(ValueError):
            Mat3x4().inverse()
        with self.assertRaises(ValueError):
            Mat4x2().inverse()


if __name__ == "__main__":
    unittest.main()
