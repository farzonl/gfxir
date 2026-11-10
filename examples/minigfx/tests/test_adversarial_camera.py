"""Adversarial stress-test suite for MiniGfx CPU math, camera, and orbit controller.

Adversarial coverage:
1. Perspective projection depth precision:
   - Near/far ratios ranging from 1e-4 to 1e7 (n=0.0001 to 100, f=1 to 1e6).
   - Point on near plane maps to z_c / w_c == 0.0 +- 1e-5.
   - Point on far plane maps to z_c / w_c == 1.0 +- 1e-5.
   - Monotonicity and boundary checks.
2. Frustum plane culling corner cases:
   - Spheres perfectly tangent to each of the 6 frustum planes (inside by eps and outside by eps).
   - Spheres positioned at the 8 frustum corners.
   - Very large bounding spheres enclosing the camera.
   - Spheres strictly behind the camera.
3. OrbitController numerical stability under extreme conditions:
   - 10,000 continuous update steps with varying delta times (dt in [0.0001, 10.0]).
   - Rapid consecutive rotate/pan/zoom calls with large and erratic inputs.
   - Strict pitch clamping [min_pitch, max_pitch] and distance [min_distance, max_distance].
   - Zero NaN or Inf throughout state.
   - Target collinearity and recovery.
4. std430 byte packing integrity:
   - 1,000 random CameraParams instances serialized to bytes.
   - Verification that len(b) == 80.
   - struct.unpack("<16f3ff", b) bit-for-bit and floating-point reconstruction.
"""

from __future__ import annotations

import math
import random
import struct
import unittest

from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.math import Mat4, Quat, Vec2, Vec3, Vec4


class TestPerspectiveDepthPrecision(unittest.TestCase):
    """Stress-tests for perspective projection depth precision across extreme near/far ratios."""

    def test_extreme_near_far_ratios(self):
        """Test near/far ratios from ~1 to 1e7, ensuring near plane -> 0.0 and far plane -> 1.0."""
        configs = [
            # (near, far, fov_y_deg, aspect)
            (0.001, 10000.0, 60.0, 16.0 / 9.0),      # ratio 1e7 (from prompt)
            (0.0001, 1.0, 45.0, 1.0),                # ratio 1e4
            (1.0, 1000000.0, 90.0, 4.0 / 3.0),       # ratio 1e6
            (0.1, 1000.0, 60.0, 16.0 / 9.0),         # ratio 1e4
            (10.0, 10.001, 60.0, 1.0),               # ratio ~1 (very tight)
            (0.0001, 1000.0, 120.0, 21.0 / 9.0),     # wide FOV, wide aspect
            (0.5, 50.0, 10.0, 0.5),                  # narrow FOV, portrait
            (0.01, 100000.0, 170.0, 1.0),            # extreme FOV (170 deg)
            (0.001, 100.0, 1.0, 1.0),                # extreme narrow FOV (1 deg)
        ]

        for near, far, fov_deg, aspect in configs:
            fov_rad = math.radians(fov_deg)
            p = Mat4.perspective(fov_rad, aspect, near, far)

            # Test multiple points on near plane (z = -near)
            tan_half_fov = math.tan(fov_rad * 0.5)
            h_near = near * tan_half_fov
            w_near = h_near * aspect

            near_points = [
                (0.0, 0.0, -near),
                (w_near, h_near, -near),
                (-w_near, h_near, -near),
                (w_near, -h_near, -near),
                (-w_near, -h_near, -near),
                (w_near * 0.5, -h_near * 0.3, -near),
                (12.34, -56.78, -near),  # Arbitrary lateral coordinates
            ]

            for x, y, z in near_points:
                clip = p * Vec4(x, y, z, 1.0)
                self.assertGreater(clip.w, 0.0, f"w_c must be positive, got {clip.w}")
                z_ndc = clip.z / clip.w
                self.assertAlmostEqual(
                    z_ndc,
                    0.0,
                    places=5,
                    msg=f"Near plane mapping failed for n={near}, f={far}: z_ndc={z_ndc}",
                )

            # Test multiple points on far plane (z = -far)
            h_far = far * tan_half_fov
            w_far = h_far * aspect

            far_points = [
                (0.0, 0.0, -far),
                (w_far, h_far, -far),
                (-w_far, h_far, -far),
                (w_far, -h_far, -far),
                (-w_far, -h_far, -far),
                (w_far * 0.5, -h_far * 0.3, -far),
                (-99.0, 42.0, -far),
            ]

            for x, y, z in far_points:
                clip = p * Vec4(x, y, z, 1.0)
                self.assertGreater(clip.w, 0.0, f"w_c must be positive, got {clip.w}")
                z_ndc = clip.z / clip.w
                self.assertAlmostEqual(
                    z_ndc,
                    1.0,
                    places=5,
                    msg=f"Far plane mapping failed for n={near}, f={far}: z_ndc={z_ndc}",
                )

            # Test interior depth monotonicity
            prev_z_ndc = -1.0
            for frac in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
                # Logarithmic distribution between near and far for better coverage
                z_dist = near * math.pow(far / near, frac)
                clip = p * Vec4(0.0, 0.0, -z_dist, 1.0)
                z_ndc = clip.z / clip.w
                self.assertGreaterEqual(z_ndc, 0.0)
                self.assertLessEqual(z_ndc, 1.0)
                self.assertGreater(
                    z_ndc,
                    prev_z_ndc,
                    f"Depth is not strictly monotonic: z={z_dist}, z_ndc={z_ndc}, prev={prev_z_ndc}",
                )
                prev_z_ndc = z_ndc


class TestFrustumCullingCornerCases(unittest.TestCase):
    """Stress-tests for frustum plane extraction and bounding sphere visibility."""

    def setUp(self):
        self.cam = Camera(
            eye=Vec3(0.0, 0.0, 5.0),
            target=Vec3(0.0, 0.0, 0.0),
            up=Vec3(0.0, 1.0, 0.0),
            fov_y=math.radians(60.0),
            aspect=1.0,
            near=0.1,
            far=100.0,
        )

    def _unproject_ndc(self, ndc_x: float, ndc_y: float, ndc_z: float) -> Vec3:
        """Unproject an NDC point [x, y, z in ([-1,1], [-1,1], [0,1])] to world coordinates."""
        vp = self.cam.view_proj
        inv_vp = vp.inverse()
        clip = Vec4(ndc_x, ndc_y, ndc_z, 1.0)
        world4 = inv_vp * clip
        self.assertNotEqual(world4.w, 0.0)
        inv_w = 1.0 / world4.w
        return Vec3(world4.x * inv_w, world4.y * inv_w, world4.z * inv_w)

    def test_frustum_planes_normalization(self):
        """Verify all 6 extracted planes have strictly unit-length normals."""
        planes = self.cam.frustum_planes
        self.assertEqual(len(planes), 6)
        for i, p in enumerate(planes):
            norm_len = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            self.assertAlmostEqual(
                norm_len,
                1.0,
                places=6,
                msg=f"Plane {i} normal is not unit length: {norm_len}",
            )

    def test_planes_tangent_spheres(self):
        """Spheres tangent to each plane: inside by eps -> visible; outside by eps -> culled."""
        planes = self.cam.frustum_planes
        # Face centers in NDC: Left, Right, Bottom, Top, Near, Far
        ndc_face_centers = [
            (-1.0, 0.0, 0.5),   # Left plane
            (1.0, 0.0, 0.5),    # Right plane
            (0.0, -1.0, 0.5),   # Bottom plane
            (0.0, 1.0, 0.5),    # Top plane
            (0.0, 0.0, 0.0),    # Near plane
            (0.0, 0.0, 1.0),    # Far plane
        ]

        radius = 0.5
        eps = 1e-4

        for idx, (ndc, plane) in enumerate(zip(ndc_face_centers, planes)):
            pt_on_plane = self._unproject_ndc(*ndc)
            normal = Vec3(plane.x, plane.y, plane.z)

            # Check that pt_on_plane lies exactly on this plane (distance == 0)
            dist_to_plane = normal.dot(pt_on_plane) + plane.w
            self.assertAlmostEqual(
                dist_to_plane,
                0.0,
                places=4,
                msg=f"Face {idx} point is not on plane: dist={dist_to_plane}",
            )

            # Sphere center shifted INSIDE the frustum by (radius - eps)
            # Signed distance to plane is (radius - eps) >= -radius -> visible
            # And within face boundaries, it remains inside all other 5 planes
            c_inside = pt_on_plane + normal * (radius - eps)
            self.assertTrue(
                self.cam.is_sphere_visible(c_inside, radius),
                f"Face {idx} inside tangent sphere failed visibility",
            )

            # Sphere center shifted OUTSIDE the frustum by (radius + eps)
            # Signed distance to plane is -(radius + eps) < -radius -> strictly culled
            c_outside = pt_on_plane - normal * (radius + eps)
            self.assertFalse(
                self.cam.is_sphere_visible(c_outside, radius),
                f"Face {idx} outside tangent sphere was not culled",
            )

    def test_spheres_at_8_frustum_corners(self):
        """Spheres positioned at all 8 frustum corners must be visible."""
        ndc_corners = [
            (-1.0, -1.0, 0.0),
            (1.0, -1.0, 0.0),
            (-1.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (-1.0, -1.0, 1.0),
            (1.0, -1.0, 1.0),
            (-1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0),
        ]

        for ndc in ndc_corners:
            corner_world = self._unproject_ndc(*ndc)
            # Sphere with radius 0.1 centered at frustum corner must intersect frustum
            self.assertTrue(
                self.cam.is_sphere_visible(corner_world, radius=0.1),
                f"Frustum corner sphere at NDC {ndc} (world {corner_world}) was falsely culled",
            )

    def test_large_spheres_enclosing_camera(self):
        """Very large bounding spheres enclosing the camera must always be visible."""
        test_centers = [
            Vec3(0.0, 0.0, 0.0),
            self.cam.eye,
            self.cam.target,
            Vec3(500.0, -500.0, 500.0),
            Vec3(-1000.0, 2000.0, -3000.0),
        ]
        for c in test_centers:
            # Huge sphere of radius 1e5 enclosing everything
            self.assertTrue(
                self.cam.is_sphere_visible(c, radius=1e5),
                f"Enclosing sphere centered at {c} was improperly culled",
            )

    def test_spheres_strictly_behind_camera(self):
        """Spheres placed behind the camera (outside near plane) must be culled."""
        # Camera is at (0, 0, 5) looking towards (0, 0, 0).
        # View direction is -Z. The near plane is at Z = 5 - 0.1 = 4.9.
        # Anything with Z > 5 is strictly behind the camera eye.
        behind_positions = [
            Vec3(0.0, 0.0, 6.0),
            Vec3(0.0, 0.0, 10.0),
            Vec3(5.0, 5.0, 15.0),
            Vec3(-10.0, -10.0, 20.0),
            Vec3(0.0, 0.0, 100.0),
        ]
        for pos in behind_positions:
            self.assertFalse(
                self.cam.is_sphere_visible(pos, radius=0.5),
                f"Sphere behind camera at {pos} was not culled",
            )


class TestOrbitControllerExtremeStability(unittest.TestCase):
    """Stress-tests for OrbitController numerical stability under extreme conditions."""

    def test_continuous_10000_steps_erratic_dt(self):
        """10,000 continuous update steps with varying delta times dt in [0.0001, 10.0]."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(
            target=Vec3(0, 0, 0),
            distance=5.0,
            min_distance=0.5,
            max_distance=100.0,
            min_pitch=-1.5,
            max_pitch=1.5,
            damping=0.1,
        )

        rng = random.Random(42)

        for step in range(10000):
            # Vary dt wildly between 0.0001 and 10.0
            dt = math.exp(rng.uniform(math.log(0.0001), math.log(10.0)))

            # Erratic rapid inputs
            if step % 2 == 0:
                ctrl.rotate(rng.uniform(-5.0, 5.0), rng.uniform(-5.0, 5.0))
            if step % 3 == 0:
                ctrl.pan(rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), cam)
            if step % 5 == 0:
                ctrl.zoom(rng.uniform(-10.0, 10.0))

            ctrl.update(cam, dt)

            # Assert invariants at every step
            self.assertFalse(math.isnan(ctrl.pitch), f"Step {step}: pitch is NaN")
            self.assertFalse(math.isnan(ctrl.yaw), f"Step {step}: yaw is NaN")
            self.assertFalse(math.isnan(ctrl.distance), f"Step {step}: distance is NaN")
            self.assertFalse(math.isinf(ctrl.distance), f"Step {step}: distance is Inf")

            # Pitch strictly clamped
            self.assertGreaterEqual(ctrl.pitch, ctrl.min_pitch - 1e-6)
            self.assertLessEqual(ctrl.pitch, ctrl.max_pitch + 1e-6)

            # Distance strictly clamped
            self.assertGreaterEqual(ctrl.distance, ctrl.min_distance - 1e-6)
            self.assertLessEqual(ctrl.distance, ctrl.max_distance + 1e-6)

            # Camera eye and target validity
            for coord in (cam.eye.x, cam.eye.y, cam.eye.z, cam.target.x, cam.target.y, cam.target.z):
                self.assertFalse(math.isnan(coord), f"Step {step}: camera coordinate is NaN")
                self.assertFalse(math.isinf(coord), f"Step {step}: camera coordinate is Inf")

            # Camera eye and target never coincident
            diff = cam.eye - cam.target
            self.assertGreater(diff.length(), ctrl.min_distance * 0.9)

    def test_rapid_alternating_inputs(self):
        """Rapid consecutive alternating rotate/pan/zoom calls with massive magnitudes."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0)

        for _ in range(500):
            ctrl.rotate(+1000.0, +1000.0)
            ctrl.rotate(-1000.0, -1000.0)
            ctrl.zoom(+1000.0)
            ctrl.zoom(-1000.0)
            ctrl.pan(+500.0, -500.0, cam)
            ctrl.update(cam, dt=0.016)

        self.assertGreaterEqual(ctrl.pitch, ctrl.min_pitch)
        self.assertLessEqual(ctrl.pitch, ctrl.max_pitch)
        self.assertGreaterEqual(ctrl.distance, ctrl.min_distance)
        self.assertLessEqual(ctrl.distance, ctrl.max_distance)

    def test_target_collinearity_and_recovery(self):
        """Stress-test target collinearity, gimbal limits, and recovery from extreme states."""
        cam = Camera(eye=Vec3(0, 0, 5), target=Vec3(0, 0, 0))
        ctrl = OrbitController(target=Vec3(0, 0, 0), distance=5.0, min_pitch=-1.5, max_pitch=1.5)

        # Drive pitch to exact upper bound
        ctrl.rotate(0.0, 100.0)
        ctrl.update(cam, dt=10.0)  # Large dt to settle damping
        self.assertAlmostEqual(ctrl.pitch, 1.5, places=5)

        # Verify camera can still build valid view matrix (no collinearity crash with up vector)
        vm = cam.view_matrix
        self.assertEqual(len(vm.elements), 16)
        self.assertFalse(any(math.isnan(x) for x in vm.elements))

        # Drive pitch to exact lower bound
        ctrl.rotate(0.0, -200.0)
        ctrl.update(cam, dt=10.0)
        self.assertAlmostEqual(ctrl.pitch, -1.5, places=5)
        vm2 = cam.view_matrix
        self.assertFalse(any(math.isnan(x) for x in vm2.elements))

        # Recovery back to horizontal
        ctrl.rotate(0.0, 1.5)
        ctrl.update(cam, dt=10.0)
        self.assertAlmostEqual(ctrl.pitch, 0.0, places=5)


class TestCameraParamsStd430Integrity(unittest.TestCase):
    """Stress-tests for std430 byte packing integrity of CameraParams."""

    def test_1000_random_cameraparams_serialization(self):
        """1,000 random CameraParams instances serialized, verifying len==80 and exact reconstruction."""
        rng = random.Random(1337)

        for i in range(1000):
            # Generate random transform matrix elements
            m_elems = [rng.uniform(-100.0, 100.0) for _ in range(16)]
            vp = Mat4(m_elems)
            eye = Vec3(rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0), rng.uniform(-50.0, 50.0))
            time_val = rng.uniform(0.0, 10000.0)

            params = CameraParams(view_proj=vp, eye_pos=eye, time=time_val)
            b = params.to_bytes()

            # Byte length must be exactly 80
            self.assertEqual(len(b), 80, f"Instance {i}: packed length is {len(b)} != 80")

            # std430 alignment: struct size is multiple of 16
            self.assertEqual(len(b) % 16, 0)

            # Unpack with struct.unpack("<16f3ff", b)
            unpacked = struct.unpack("<16f3ff", b)
            unpacked_mat = unpacked[0:16]
            unpacked_eye = unpacked[16:19]
            unpacked_time = unpacked[19]

            # Verify bit-for-bit or float32 precision
            # Note: struct "<16f3ff" converts Python float64 to IEEE-754 float32 and back.
            # Compare unpacked float32 values with expected float32 conversions.
            expected_mat = struct.unpack("<16f", struct.pack("<16f", *m_elems))
            expected_eye = struct.unpack("<3f", struct.pack("<3f", eye.x, eye.y, eye.z))
            expected_time = struct.unpack("<f", struct.pack("<f", time_val))[0]

            self.assertEqual(unpacked_mat, expected_mat)
            self.assertEqual(unpacked_eye, expected_eye)
            self.assertEqual(unpacked_time, expected_time)

            # Verify exact bit-for-bit equivalence with manual packing
            expected_bytes = (
                struct.pack("<16f", *expected_mat)
                + struct.pack("<3f", *expected_eye)
                + struct.pack("<f", expected_time)
            )
            self.assertEqual(b, expected_bytes)

            # Verify round-trip deserialization
            restored = CameraParams.from_bytes(b)
            self.assertEqual(restored.view_proj, Mat4(expected_mat))
            self.assertEqual(restored.eye_pos, Vec3(*expected_eye))
            self.assertAlmostEqual(restored.time, expected_time, places=5)


if __name__ == "__main__":
    unittest.main()
