"""Stress tests for Scene graph, Mesh representation, Ritter bounding sphere, and Wavefront OBJ loader.

Tests cover:
1. >70,000 vertex mesh generation, uint16/uint32 index promotion boundaries, binary vertex/index integrity.
2. Degenerate geometries (0-area triangles, collinear points, duplicate vertices, 50- and 100-sided n-gons).
3. Malformed OBJ fuzzing (v//vn, v/vt, v, negative relative indices, whitespace, 10,000 fuzz mutations).
4. Ritter bounding sphere robustness (single point, line, collinear, coplanar, extreme aspect ratios, 10k random point clouds).
5. Camera import lifecycle in clean isolated process.
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import string
import struct
import subprocess
import sys
import tempfile
import unittest

from examples.minigfx.math import Mat4x4, Quat, Vec2, Vec3, Vec4
from examples.minigfx.obj_loader import load_obj, load_obj_from_string
from examples.minigfx.scene import (
    Frustum,
    Mesh,
    Node,
    Scene,
    Vertex,
    create_cube,
    create_plane,
    create_sphere,
)


class TestLargeMeshAndIndexPromotion(unittest.TestCase):
    """Stress tests for >70,000 vertex meshes and uint16->uint32 promotion."""

    def test_sphere_over_70k_vertices(self):
        # subdivisions=200 produces (201 * 401) = 80,601 vertices and 477,600 indices
        sphere = create_sphere(radius=2.5, subdivisions=200)
        vert_count = len(sphere.vertices)
        idx_count = len(sphere.indices)
        self.assertGreater(vert_count, 70000)
        self.assertEqual(vert_count, 80601)
        self.assertEqual(idx_count, 477600)

        # Verify vertex bytes length and packing
        v_bytes = sphere.vertex_bytes()
        self.assertEqual(len(v_bytes), vert_count * 32)
        self.assertEqual(sphere.get_interleaved_vertex_bytes(), v_bytes)

        # Spot-check unpack of first, middle, 65536th, and last vertex
        for test_idx in (0, 1000, 65535, 65536, 70000, vert_count - 1):
            unpacked = Vertex.from_bytes(v_bytes[test_idx * 32 : (test_idx + 1) * 32])
            orig = sphere.vertices[test_idx]
            self.assertEqual(unpacked.position, orig.position)
            self.assertEqual(unpacked.normal, orig.normal)
            self.assertEqual(unpacked.uv, orig.uv)

        # Verify index bytes auto-promotes to uint32
        i_bytes = sphere.index_bytes()
        self.assertEqual(len(i_bytes), idx_count * 4)

        # Verify exact roundtrip unpack of all 477,600 indices
        unpacked_indices = struct.unpack(f"<{idx_count}I", i_bytes)
        self.assertEqual(list(unpacked_indices), sphere.indices)

    def test_uint16_uint32_boundary_conditions(self):
        # Case A: exactly 65536 vertices, max index 65535 -> fits in uint16 (2 bytes per index)
        m_u16_max = Mesh([Vertex() for _ in range(65536)], [0, 100, 65535], center=Vec3(), radius=0.0)
        b_u16 = m_u16_max.index_bytes()
        self.assertEqual(len(b_u16), 3 * 2)
        self.assertEqual(struct.unpack("<3H", b_u16), (0, 100, 65535))

        # Case B: exactly 65536 vertices, max index 65536 -> promotes to uint32 (4 bytes per index)
        m_u32_idx = Mesh([Vertex() for _ in range(65536)], [0, 100, 65536], center=Vec3(), radius=0.0)
        b_u32_idx = m_u32_idx.index_bytes()
        self.assertEqual(len(b_u32_idx), 3 * 4)
        self.assertEqual(struct.unpack("<3I", b_u32_idx), (0, 100, 65536))

        # Case C: 65537 vertices, max index 10 -> promotes to uint32 due to vertex count > 65536
        m_u32_vert = Mesh([Vertex() for _ in range(65537)], [0, 5, 10], center=Vec3(), radius=0.0)
        b_u32_vert = m_u32_vert.index_bytes()
        self.assertEqual(len(b_u32_vert), 3 * 4)
        self.assertEqual(struct.unpack("<3I", b_u32_vert), (0, 5, 10))

        # Case D: force_u32 flag on tiny mesh
        m_tiny = Mesh([Vertex()], [0], center=Vec3(), radius=0.0)
        b_forced = m_tiny.index_bytes(force_u32=True)
        self.assertEqual(len(b_forced), 1 * 4)
        self.assertEqual(struct.unpack("<I", b_forced), (0,))

    def test_empty_and_zero_index_mesh(self):
        m_empty = Mesh([], [])
        self.assertEqual(m_empty.vertex_bytes(), b"")
        self.assertEqual(m_empty.index_bytes(), b"")
        self.assertEqual(m_empty.center, Vec3(0.0, 0.0, 0.0))
        self.assertEqual(m_empty.radius, 0.0)


class TestDegenerateGeometryAndCollinearPolygons(unittest.TestCase):
    """Stress tests for degenerate triangles, collinear points, duplicate vertices, and n-gons."""

    def test_zero_area_triangle_identical_vertices(self):
        obj = "v 1.0 2.0 3.0\nv 1.0 2.0 3.0\nv 1.0 2.0 3.0\nf 1 2 3\n"
        for smooth in (False, True):
            with self.subTest(smooth=smooth):
                mesh = load_obj_from_string(obj, smooth_normals=smooth)
                self.assertEqual(len(mesh.vertices), 3)
                self.assertEqual(len(mesh.indices), 3)
                for v in mesh.vertices:
                    self.assertEqual(v.normal, Vec3(0.0, 1.0, 0.0))

    def test_zero_area_triangle_two_identical_vertices(self):
        obj = "v 0 0 0\nv 1 0 0\nv 1 0 0\nf 1 2 3\n"
        for smooth in (False, True):
            with self.subTest(smooth=smooth):
                mesh = load_obj_from_string(obj, smooth_normals=smooth)
                self.assertEqual(len(mesh.indices), 3)
                for v in mesh.vertices:
                    self.assertEqual(v.normal, Vec3(0.0, 1.0, 0.0))

    def test_collinear_vertices_triangle(self):
        obj = "v 0 0 0\nv 2 0 0\nv 4 0 0\nf 1 2 3\n"
        for smooth in (False, True):
            with self.subTest(smooth=smooth):
                mesh = load_obj_from_string(obj, smooth_normals=smooth)
                self.assertEqual(len(mesh.indices), 3)
                for v in mesh.vertices:
                    self.assertEqual(v.normal, Vec3(0.0, 1.0, 0.0))

    def test_duplicate_face_indices(self):
        obj = "v 0 0 0\nv 1 0 0\nf 1 1 1\n"
        mesh = load_obj_from_string(obj)
        self.assertEqual(len(mesh.indices), 3)
        self.assertEqual(len(mesh.vertices), 1)
        self.assertEqual(mesh.vertices[0].normal, Vec3(0.0, 1.0, 0.0))

    def test_50_sided_convex_ngon(self):
        lines = []
        n_sides = 50
        for i in range(n_sides):
            th = 2.0 * math.pi * i / n_sides
            lines.append(f"v {math.cos(th):.6f} {math.sin(th):.6f} 0.0")
        lines.append("f " + " ".join(str(i + 1) for i in range(n_sides)))
        obj = "\n".join(lines) + "\n"

        mesh_flat = load_obj_from_string(obj, smooth_normals=False)
        self.assertEqual(len(mesh_flat.indices), (n_sides - 2) * 3)
        self.assertEqual(len(mesh_flat.vertices), n_sides)
        for v in mesh_flat.vertices:
            self.assertAlmostEqual(v.normal.z, 1.0, places=3)

        mesh_smooth = load_obj_from_string(obj, smooth_normals=True)
        self.assertEqual(len(mesh_smooth.indices), (n_sides - 2) * 3)
        for v in mesh_smooth.vertices:
            self.assertAlmostEqual(v.normal.z, 1.0, places=3)

    def test_100_sided_convex_ngon(self):
        lines = []
        n_sides = 100
        for i in range(n_sides):
            th = 2.0 * math.pi * i / n_sides
            lines.append(f"v {math.cos(th):.6f} {math.sin(th):.6f} 0.0")
        lines.append("f " + " ".join(str(i + 1) for i in range(n_sides)))
        obj = "\n".join(lines) + "\n"

        mesh = load_obj_from_string(obj)
        self.assertEqual(len(mesh.indices), (n_sides - 2) * 3)
        self.assertEqual(len(mesh.vertices), n_sides)


class TestMalformedObjStressAndFuzzing(unittest.TestCase):
    """Stress tests and randomized fuzzing for OBJ parser robustness."""

    def test_all_face_spec_combinations(self):
        # 1. v//vn
        s1 = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 0 0 1\nf 1//1 2//1 3//1\n"
        mesh_a = load_obj_from_string(s1)
        self.assertEqual(len(mesh_a.vertices), 3)
        self.assertAlmostEqual(mesh_a.vertices[0].normal.z, 1.0)

        # 2. v/vt
        s2 = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0.1 0.2\nvt 0.3 0.4\nvt 0.5 0.6\nf 1/1 2/2 3/3\n"
        mesh_b = load_obj_from_string(s2)
        self.assertEqual(len(mesh_b.vertices), 3)
        self.assertAlmostEqual(mesh_b.vertices[0].uv.x, 0.1)

        # 3. v only
        s3 = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n"
        mesh_c = load_obj_from_string(s3)
        self.assertEqual(len(mesh_c.vertices), 3)
        self.assertAlmostEqual(mesh_c.vertices[0].normal.z, 1.0)

        # 4. v/vt/vn
        s4 = "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 1 0\nvt 0 1\nvn 0 0 1\nf 1/1/1 2/2/1 3/3/1\n"
        mesh_d = load_obj_from_string(s4)
        self.assertEqual(len(mesh_d.vertices), 3)

        # 5. Mixed formats in single OBJ
        s5 = """
        v 0 0 0
        v 1 0 0
        v 0 1 0
        v 1 1 0
        vt 0.0 0.0
        vt 1.0 1.0
        vn 0 0 1
        f 1 2 3
        f 1/1 2/2 4/1
        f 1//1 2//1 3//1
        f 1/1/1 2/2/1 4/2/1
        """
        mesh_e = load_obj_from_string(s5)
        self.assertEqual(len(mesh_e.indices), 12)

    def test_negative_relative_indexing_wrapping(self):
        # 3 vertices, valid negative indices are -1, -2, -3
        obj_valid = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -3 -2 -1\n"
        m = load_obj_from_string(obj_valid)
        self.assertEqual(len(m.indices), 3)

        # Out of bounds negative index: -4 when only 3 vertices defined
        obj_oob_neg = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -4 -2 -1\n"
        with self.assertRaises(ValueError):
            load_obj_from_string(obj_oob_neg)

        # Index 0 is invalid per OBJ spec
        obj_zero = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n"
        with self.assertRaises(ValueError):
            load_obj_from_string(obj_zero)

        # Positive out of bounds: 4 when only 3 vertices defined
        obj_oob_pos = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 4\n"
        with self.assertRaises(ValueError):
            load_obj_from_string(obj_oob_pos)

    def test_whitespace_and_crlf_variants(self):
        crlf_obj = "\r\n  v  \t 0.0  0.0  0.0   \r\n  v 1.0  0.0  0.0 \r\n  v 0.0 1.0 0.0  \r\n\r\n  f   1   2   3   \r\n\r\n"
        m = load_obj_from_string(crlf_obj)
        self.assertEqual(len(m.vertices), 3)
        self.assertEqual(len(m.indices), 3)

    def test_unsupported_commands_gracefully_ignored(self):
        obj = """
        # Comments
        mtllib material.mtl
        o TestObject
        g Group1
        s 1
        usemtl DefaultMaterial
        v 0 0 0
        v 1 0 0
        v 0 1 0
        f 1 2 3
        """
        mesh = load_obj_from_string(obj)
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.indices), 3)

    def test_face_with_less_than_three_vertices_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_obj_from_string("v 0 0 0\nv 1 0 0\nf 1 2\n")
        with self.assertRaises(ValueError):
            load_obj_from_string("v 0 0 0\nf 1\n")
        with self.assertRaises(ValueError):
            load_obj_from_string("f\n")

    def test_truncated_vertex_tokens_exception_type(self):
        with self.assertRaises(ValueError):
            load_obj_from_string("v 1.0 2.0\nf 1 1 1\n")
        with self.assertRaises(ValueError):
            load_obj_from_string("v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 1.0\nf 1 2 3\n")
        with self.assertRaises(ValueError):
            load_obj_from_string("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt\nf 1 2 3\n")

    def test_randomized_fuzzing_10000_iterations(self):
        """Randomized fuzzer validating parser stability across 10,000 malformed inputs."""
        valid_seed = """
        v 0.0 0.0 0.0
        v 1.0 0.0 0.0
        v 0.0 1.0 0.0
        vn 0.0 0.0 1.0
        vt 0.0 0.0
        f 1/1/1 2/1/1 3/1/1
        """
        successes = 0
        handled_errors = 0

        for i in range(10000):
            mode = i % 5
            if mode == 0:
                # Random printable and control strings
                s = "".join(random.choice(string.printable + "\n\r\t") for _ in range(random.randint(0, 150)))
            elif mode == 1:
                # Corrupted OBJ lines
                lines = valid_seed.strip().splitlines()
                random.shuffle(lines)
                if random.random() < 0.5:
                    lines.append(f"f {random.randint(-5, 5)} {random.randint(-5, 5)} {random.randint(-5, 5)}")
                s = "\n".join(lines)
            elif mode == 2:
                # Face token mutations
                parts = []
                for _ in range(random.randint(1, 5)):
                    tok = "/".join(str(random.choice(["", "0", "-1", "1", "99", "abc"])) for _ in range(random.randint(0, 3)))
                    parts.append(tok)
                s = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf " + " ".join(parts)
            elif mode == 3:
                # Negative indices boundaries
                idx_choices = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
                s = f"v 0 0 0\nv 1 0 0\nv 0 1 0\nf {random.choice(idx_choices)} {random.choice(idx_choices)} {random.choice(idx_choices)}"
            else:
                # Non-finite or extreme values
                val = random.choice(["nan", "-inf", "inf", "1e300", "-1e300"])
                s = f"v {val} {val} {val}\nf 1 1 1"

            try:
                mesh = load_obj_from_string(s, smooth_normals=(i % 2 == 0))
                self.assertIsInstance(mesh, Mesh)
                successes += 1
            except ValueError:
                handled_errors += 1

        self.assertEqual(successes + handled_errors, 10000)
        self.assertGreater(successes, 1000)
        self.assertGreater(handled_errors, 1000)


class TestRitterBoundingSphereRobustness(unittest.TestCase):
    """Stress tests for Ritter's bounding sphere algorithm across degenerate point clouds."""

    def test_single_point(self):
        m = Mesh([Vertex(Vec3(7.0, -3.0, 4.0))], [0])
        self.assertEqual(m.center, Vec3(7.0, -3.0, 4.0))
        self.assertEqual(m.radius, 0.0)

    def test_two_identical_points(self):
        m = Mesh([Vertex(Vec3(1.0, 1.0, 1.0)), Vertex(Vec3(1.0, 1.0, 1.0))], [0, 1])
        self.assertEqual(m.center, Vec3(1.0, 1.0, 1.0))
        self.assertEqual(m.radius, 0.0)

    def test_line_segment_two_opposite_points(self):
        m = Mesh([Vertex(Vec3(-10.0, 0.0, 0.0)), Vertex(Vec3(10.0, 0.0, 0.0))], [0, 1])
        self.assertEqual(m.center, Vec3(0.0, 0.0, 0.0))
        self.assertEqual(m.radius, 10.0)

    def test_collinear_points_containment(self):
        pts = [Vec3(i * 1.5, i * -2.0, i * 0.5) for i in range(-50, 51)]
        m = Mesh([Vertex(p) for p in pts], list(range(len(pts))))
        c, r = m.center, m.radius
        for p in pts:
            dist = (p - c).length()
            self.assertLessEqual(dist, r + 1e-5)

    def test_coplanar_points_containment(self):
        pts = []
        for i in range(36):
            th = 2.0 * math.pi * i / 36.0
            pts.append(Vec3(5.0 * math.cos(th), 0.0, 5.0 * math.sin(th)))
        m = Mesh([Vertex(p) for p in pts], list(range(len(pts))))
        c, r = m.center, m.radius
        self.assertAlmostEqual(c.x, 0.0, places=4)
        self.assertAlmostEqual(c.y, 0.0, places=4)
        self.assertAlmostEqual(c.z, 0.0, places=4)
        self.assertAlmostEqual(r, 5.0, places=4)
        for p in pts:
            dist = (p - c).length()
            self.assertLessEqual(dist, r + 1e-5)

    def test_extreme_aspect_ratio(self):
        pts = [
            Vec3(-1e6, 0.0, 0.0),
            Vec3(1e6, 0.0, 0.0),
            Vec3(0.0, 1e-4, 0.0),
            Vec3(0.0, 0.0, 1e-4),
        ]
        m = Mesh([Vertex(p) for p in pts], list(range(len(pts))))
        c, r = m.center, m.radius
        for p in pts:
            dist = (p - c).length()
            self.assertLessEqual(dist, r + 1e-3)

    def test_10000_random_point_clouds_containment(self):
        """Validate that 10,000 random point sets are 100% enclosed by the computed bounding sphere."""
        for trial in range(10000):
            n_pts = random.randint(3, 50)
            pts = [
                Vec3(random.uniform(-100.0, 100.0), random.uniform(-100.0, 100.0), random.uniform(-100.0, 100.0))
                for _ in range(n_pts)
            ]
            m = Mesh([Vertex(p) for p in pts], list(range(n_pts)))
            c, r = m.center, m.radius
            for p in pts:
                dist = (p - c).length()
                self.assertLessEqual(dist, r + 1e-5, f"Trial {trial}: point {p} outside sphere (center={c}, radius={r})")

    def test_node_bounding_sphere_transformation(self):
        mesh = create_cube(size=2.0)  # radius = 2.0 * sqrt(3) / 2 = sqrt(3) ~= 1.732
        node = Node("node", mesh=mesh, translation=(10.0, -5.0, 20.0), scale=(3.0, 3.0, 3.0))
        center, radius = node.bounding_sphere
        self.assertAlmostEqual(center.x, 10.0)
        self.assertAlmostEqual(center.y, -5.0)
        self.assertAlmostEqual(center.z, 20.0)
        self.assertAlmostEqual(radius, 3.0 * math.sqrt(3.0))

        # Reflection (negative scale) should produce positive radius
        node.scale = (-2.0, 2.0, 2.0)
        _, neg_radius = node.bounding_sphere
        self.assertAlmostEqual(neg_radius, 2.0 * math.sqrt(3.0))


class TestCameraImportLifecycle(unittest.TestCase):
    """Verifies camera module cleanly imports in a fresh Python process."""

    def test_camera_import_in_clean_process(self):
        cmd = [
            sys.executable,
            "-c",
            "import examples.minigfx.camera; assert examples.minigfx.camera.Camera is not None",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, f"Camera import failed: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
