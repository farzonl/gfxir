"""Unit tests for Scene graph, Mesh primitives, Frustum culling, and OBJ loader."""

from __future__ import annotations

import math
import struct
import sys
import tempfile
import unittest
from pathlib import Path

from examples.minigfx.camera import Camera
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
try:
    from examples.minigfx.tests.conftest import (
        MALFORMED_OBJ_CORPUS,
        SAMPLE_CUBE_OBJ,
        SAMPLE_TRI_OBJ,
    )
except ImportError:
    SAMPLE_CUBE_OBJ = """# Unit cube centered at origin
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v -0.5  0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5 -0.5
v  0.5  0.5 -0.5
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
vn  0.0  0.0  1.0
vn  0.0  1.0  0.0
vn  0.0  0.0 -1.0
vn  0.0 -1.0  0.0
vn  1.0  0.0  0.0
vn -1.0  0.0  0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
# 6 quad faces
f 1/1/1 2/2/1 4/3/1 3/4/1
f 3/1/2 4/2/2 6/3/2 5/4/2
f 5/1/3 6/2/3 8/3/3 7/4/3
f 7/1/4 8/2/4 2/3/4 1/4/4
f 2/1/5 8/2/5 6/3/5 4/4/5
f 7/1/6 1/2/6 3/3/6 5/4/6
"""

    SAMPLE_TRI_OBJ = """# Minimal single triangle
v  0.0  0.5  0.0
v -0.5 -0.5  0.0
v  0.5 -0.5  0.0
vn 0.0  0.0  1.0
vt 0.5  1.0
vt 0.0  0.0
vt 1.0  0.0
f 1/1/1 2/2/1 3/3/1
"""

    MALFORMED_OBJ_CORPUS = {
        "empty": "",
        "comments_only": "# Only comments\n# Second comment\n",
        "missing_face_indices": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n",
        "out_of_bounds_index": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 999\n",
        "zero_index": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",
        "non_numeric_vertex": "v 0.0 foo 1.0\nf 1 2 3\n",
        "negative_relative_indices": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -1 -2 -3\n",
        "quad_face": "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
        "polygon_6gon": "v 0 0 0\nv 1 0 0\nv 2 1 0\nv 1 2 0\nv 0 2 0\nv -1 1 0\nf 1 2 3 4 5 6\n",
        "truncated_vertex": "v 1.0 2.0\nf 1 1 1\n",
        "truncated_normal": "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 1.0\nf 1 2 3\n",
        "truncated_uv": "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt\nf 1 2 3\n",
    }



class TestVertex(unittest.TestCase):
    """Tests for Vertex dataclass, binary layout, and reflection."""

    def test_vertex_binary_layout(self):
        v = Vertex(Vec3(1.0, 2.0, 3.0), Vec3(0.0, 1.0, 0.0), Vec2(0.5, 0.25))
        b = v.to_bytes()
        self.assertEqual(len(b), 32)
        px, py, pz, nx, ny, nz, ux, uy = struct.unpack("<8f", b)
        self.assertAlmostEqual(px, 1.0)
        self.assertAlmostEqual(py, 2.0)
        self.assertAlmostEqual(pz, 3.0)
        self.assertAlmostEqual(nx, 0.0)
        self.assertAlmostEqual(ny, 1.0)
        self.assertAlmostEqual(nz, 0.0)
        self.assertAlmostEqual(ux, 0.5)
        self.assertAlmostEqual(uy, 0.25)

    def test_vertex_from_bytes_roundtrip(self):
        v1 = Vertex(Vec3(-4.0, 5.5, 6.25), Vec3(0.0, 0.0, -1.0), Vec2(0.125, 0.875))
        b = v1.to_bytes()
        v2 = Vertex.from_bytes(b)
        self.assertEqual(v1.position, v2.position)
        self.assertEqual(v1.normal, v2.normal)
        self.assertEqual(v1.uv, v2.uv)

    def test_vertex_buffer_too_small_raises(self):
        with self.assertRaises(ValueError):
            Vertex.from_bytes(b"\x00" * 31)

    def test_vertex_reflection_metadata(self):
        v = Vertex()
        self.assertTrue(getattr(v, "_is_gfx_struct", False))
        self.assertEqual(getattr(v, "__struct_size__", 0), 32)
        self.assertEqual(getattr(v, "__struct_alignment__", 0), 4)
        self.assertEqual(len(v.as_parameter_block()), 32)


class TestMesh(unittest.TestCase):
    """Tests for Mesh serialization and bounding sphere caching."""

    def test_empty_mesh(self):
        m = Mesh([], [])
        self.assertEqual(m.vertex_bytes(), b"")
        self.assertEqual(m.index_bytes(), b"")
        self.assertEqual(m.center, Vec3(0.0, 0.0, 0.0))
        self.assertEqual(m.radius, 0.0)

    def test_mesh_vertex_bytes(self):
        verts = [
            Vertex(Vec3(0, 0, 0), Vec3(0, 1, 0), Vec2(0, 0)),
            Vertex(Vec3(1, 0, 0), Vec3(0, 1, 0), Vec2(1, 0)),
            Vertex(Vec3(0, 1, 0), Vec3(0, 1, 0), Vec2(0, 1)),
        ]
        m = Mesh(verts, [0, 1, 2])
        self.assertEqual(len(m.vertex_bytes()), 3 * 32)
        self.assertEqual(m.get_interleaved_vertex_bytes(), m.vertex_bytes())

    def test_mesh_index_bytes_u16_vs_u32(self):
        m_small = Mesh([Vertex() for _ in range(3)], [0, 1, 2])
        b_u16 = m_small.index_bytes()
        self.assertEqual(len(b_u16), 3 * 2)
        b_forced_u32 = m_small.index_bytes(force_u32=True)
        self.assertEqual(len(b_forced_u32), 3 * 4)

        m_large = Mesh([Vertex() for _ in range(4)], [0, 1, 70000])
        b_large = m_large.index_bytes()
        self.assertEqual(len(b_large), 3 * 4)

    def test_mesh_bounding_sphere_containment(self):
        verts = [
            Vertex(Vec3(-2.0, 0.0, 0.0), Vec3(0, 1, 0), Vec2(0, 0)),
            Vertex(Vec3(3.0, 0.0, 0.0), Vec3(0, 1, 0), Vec2(1, 0)),
            Vertex(Vec3(0.0, 4.0, 0.0), Vec3(0, 1, 0), Vec2(0, 1)),
        ]
        m = Mesh(verts, [0, 1, 2])
        c, r = m.center, m.radius
        for v in verts:
            dist = (v.position - c).length()
            self.assertLessEqual(dist, r + 1e-5)


class TestProceduralPrimitives(unittest.TestCase):
    """Tests for procedural cube, UV sphere, and quad plane."""

    def test_create_cube_properties(self):
        cube = create_cube(size=2.0)
        self.assertEqual(len(cube.vertices), 24)
        self.assertEqual(len(cube.indices), 36)
        self.assertEqual(len(cube.get_interleaved_vertex_bytes()), 24 * 32)
        self.assertAlmostEqual(cube.radius, 2.0 * math.sqrt(3.0) * 0.5)

        # Check CCW winding and face normals
        for i in range(0, len(cube.indices), 3):
            i0, i1, i2 = cube.indices[i], cube.indices[i + 1], cube.indices[i + 2]
            p0, p1, p2 = cube.vertices[i0].position, cube.vertices[i1].position, cube.vertices[i2].position
            fn = (p1 - p0).cross(p2 - p0).normalize()
            vn = cube.vertices[i0].normal
            self.assertGreater(fn.dot(vn), 0.999)

    def test_create_sphere_properties(self):
        sphere = create_sphere(radius=1.5, subdivisions=8)
        self.assertGreater(len(sphere.vertices), 0)
        self.assertGreater(len(sphere.indices), 0)
        self.assertAlmostEqual(sphere.radius, 1.5)

        # Check CCW outward normals
        for i in range(0, len(sphere.indices), 3):
            i0, i1, i2 = sphere.indices[i], sphere.indices[i + 1], sphere.indices[i + 2]
            p0, p1, p2 = sphere.vertices[i0].position, sphere.vertices[i1].position, sphere.vertices[i2].position
            fn = (p1 - p0).cross(p2 - p0)
            c = (p0 + p1 + p2) * (1.0 / 3.0)
            self.assertGreater(fn.dot(c), 0.0)

    def test_create_sphere_invalid_subdivisions_raises(self):
        with self.assertRaises(ValueError):
            create_sphere(radius=1.0, subdivisions=0)

    def test_create_plane_properties(self):
        plane = create_plane(width=6.0, depth=8.0)
        self.assertEqual(len(plane.vertices), 4)
        self.assertEqual(len(plane.indices), 6)
        self.assertAlmostEqual(plane.radius, 5.0)
        for v in plane.vertices:
            self.assertEqual(v.normal, Vec3(0.0, 1.0, 0.0))

        # Check normal of both triangles
        for i in (0, 3):
            i0, i1, i2 = plane.indices[i], plane.indices[i + 1], plane.indices[i + 2]
            p0, p1, p2 = plane.vertices[i0].position, plane.vertices[i1].position, plane.vertices[i2].position
            fn = (p1 - p0).cross(p2 - p0).normalize()
            self.assertAlmostEqual(fn.y, 1.0)


class TestNodeHierarchy(unittest.TestCase):
    """Tests for Node transform composition, dirty flags, and cycle detection."""

    def test_node_local_transform_trs(self):
        node = Node("test", translation=(10, 20, 30), scale=(2, 2, 2))
        lt = node.local_transform
        p = lt * Vec4(1, 0, 0, 1)
        self.assertAlmostEqual(p.x, 12.0)
        self.assertAlmostEqual(p.y, 20.0)
        self.assertAlmostEqual(p.z, 30.0)

    def test_hierarchical_world_transform(self):
        parent = Node("parent", translation=(10, 0, 0))
        child = Node("child", translation=(0, 5, 0))
        grandchild = Node("grandchild", translation=(0, 0, 2))
        parent.add_child(child)
        child.add_child(grandchild)

        wp = grandchild.world_position
        self.assertAlmostEqual(wp.x, 10.0)
        self.assertAlmostEqual(wp.y, 5.0)
        self.assertAlmostEqual(wp.z, 2.0)

    def test_node_rotation_and_custom_transform(self):
        q = Quat.from_axis_angle((0, 1, 0), math.pi * 0.5)
        node = Node("rot_node", rotation=q)
        v = node.local_transform * Vec4(1, 0, 0, 1)
        self.assertAlmostEqual(v.x, 0.0, places=5)
        self.assertAlmostEqual(v.z, -1.0, places=5)

        node.transform = Mat4x4.translation(0, 0, 100.0)
        self.assertEqual(node.world_position, Vec3(0, 0, 100.0))

    def test_dirty_flag_invalidation_and_caching(self):
        parent = Node("parent", translation=(10, 0, 0))
        child = Node("child", translation=(5, 0, 0))
        parent.add_child(child)

        self.assertAlmostEqual(child.world_position.x, 15.0)
        self.assertFalse(child._dirty)

        parent.translation = (20, 0, 0)
        self.assertTrue(child._dirty)
        self.assertAlmostEqual(child.world_position.x, 25.0)
        self.assertFalse(child._dirty)

    def test_cycle_detection(self):
        n1 = Node("n1")
        n2 = Node("n2")
        n3 = Node("n3")
        n1.add_child(n2)
        n2.add_child(n3)

        with self.assertRaises(ValueError):
            n3.add_child(n1)

        with self.assertRaises(ValueError):
            n1.add_child(n1)

    def test_add_and_remove_child(self):
        p = Node("parent")
        c = Node("child")
        p.add_child(c)
        self.assertIn(c, p.children)
        self.assertIs(c.parent, p)

        p.remove_child(c)
        self.assertNotIn(c, p.children)
        self.assertIsNone(c.parent)

    def test_bounding_sphere_scaling(self):
        mesh = create_sphere(radius=1.0)
        node = Node("scaled_sphere", mesh=mesh, scale=(3.0, 3.0, 3.0), translation=(5.0, 0.0, 0.0))
        center, radius = node.bounding_sphere
        self.assertAlmostEqual(center.x, 5.0)
        self.assertAlmostEqual(radius, 3.0)

    def test_subtree_traversal(self):
        root = Scene("root")
        c1 = Node("c1")
        c2 = Node("c2")
        gc = Node("gc")
        root.add_child(c1)
        root.add_child(c2)
        c1.add_child(gc)

        names = [n.name for n in root.traverse()]
        self.assertEqual(names, ["root", "c1", "gc", "c2"])

    def test_deep_hierarchy_traversal_and_dirty_propagation(self):
        """1,000-node hierarchy must evaluate world_transform, dirty propagation, and traversal without RecursionError."""
        orig_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(1000)
            nodes = [
                Node(
                    name=f"node_{i}",
                    translation=(0.01, 0.02, 0.03),
                    rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.radians(0.36)),
                    scale=(1.0001, 1.0, 0.9999),
                )
                for i in range(1000)
            ]
            for i in range(999):
                nodes[i].add_child(nodes[i + 1])

            # Leaf world transform
            wt = nodes[-1].world_transform
            self.assertIsNotNone(wt)

            # Dirty propagation from root to leaf
            nodes[0].translation = (1.0, 0.0, 0.0)
            self.assertTrue(nodes[-1]._dirty)
            self.assertIsNotNone(nodes[-1].world_transform)
            self.assertFalse(nodes[-1]._dirty)

            # Full traversal
            trav = list(nodes[0].traverse())
            self.assertEqual(len(trav), 1000)
        finally:
            sys.setrecursionlimit(orig_limit)

    def test_bounding_radius_hierarchical_shear_conservatism(self):
        """Bounding radius must remain conservative under hierarchical shear to prevent false negative culling."""
        axis1 = Vec3(0.125, -0.910, -0.395).normalize()
        s1 = Vec3(0.775, 0.263, 9.496)
        axis2 = Vec3(0.311, 0.670, 0.675).normalize()
        s2 = Vec3(5.256, 3.156, 5.473)
        v_loc = Vec3(0.682, -0.132, 0.719).normalize()

        mesh = Mesh([Vertex(v_loc, Vec3(0, 1, 0))], [0], center=Vec3(0, 0, 0), radius=1.0)
        p = Node("parent", rotation=Quat.from_axis_angle(axis1, 0.516), scale=s1)
        c = Node("child", rotation=Quat.from_axis_angle(axis2, 5.166), scale=s2, mesh=mesh)
        p.add_child(c)

        wc, wr = c.bounding_sphere
        w4 = c.world_transform * Vec4(v_loc.x, v_loc.y, v_loc.z, 1.0)
        dist = (Vec3(w4.x, w4.y, w4.z) - wc).length()

        # Conservative bound: world radius must enclose the transformed vertex
        self.assertLessEqual(dist, wr)

        # Frustum visibility: camera aimed directly at vertex must see the node
        cam = Camera(eye=Vec3(w4.x, w4.y, w4.z) - Vec3(0, 0, 5), target=Vec3(w4.x, w4.y, w4.z), near=0.1, far=50.0)
        self.assertTrue(c.is_visible(cam))


class TestFrustumCulling(unittest.TestCase):
    """Tests for Frustum extraction and sphere intersection tests."""

    def test_frustum_planes_from_camera(self):
        cam = Camera(eye=Vec3(0, 0, 10), target=Vec3(0, 0, 0), near=1.0, far=50.0)
        frustum = Frustum.from_matrix(cam.view_proj)
        self.assertEqual(len(frustum.planes), 6)

        # Center of view is visible
        self.assertTrue(frustum.intersects_sphere(Vec3(0, 0, 5), 1.0))
        # Behind near plane is culled
        self.assertFalse(frustum.intersects_sphere(Vec3(0, 0, 20), 0.5))
        # Beyond far plane is culled
        self.assertFalse(frustum.intersects_sphere(Vec3(0, 0, -60), 1.0))
        # Far off to the side is culled
        self.assertFalse(frustum.intersects_sphere(Vec3(100, 0, 5), 1.0))

    def test_node_visibility_and_collection(self):
        cam = Camera(eye=Vec3(0, 0, 10), target=Vec3(0, 0, 0), near=1.0, far=50.0)
        scene = Scene("scene")
        visible_node = Node("visible", translation=(0, 0, 5), mesh=create_cube(size=1.0))
        culled_node = Node("culled", translation=(0, 0, 20), mesh=create_cube(size=1.0))
        scene.add_child(visible_node)
        scene.add_child(culled_node)

        self.assertTrue(visible_node.is_visible(cam))
        self.assertFalse(culled_node.is_visible(cam))

        collected = scene.collect_visible(cam)
        self.assertIn(visible_node, collected)
        self.assertNotIn(culled_node, collected)


class TestObjLoader(unittest.TestCase):
    """Tests for Wavefront OBJ parsing, triangulation, and error handling."""

    def test_load_sample_tri(self):
        mesh = load_obj_from_string(SAMPLE_TRI_OBJ)
        self.assertEqual(len(mesh.vertices), 3)
        self.assertEqual(len(mesh.indices), 3)

    def test_load_sample_cube(self):
        mesh = load_obj_from_string(SAMPLE_CUBE_OBJ)
        self.assertEqual(len(mesh.vertices), 24)
        self.assertEqual(len(mesh.indices), 36)

    def test_quad_triangulation(self):
        quad = MALFORMED_OBJ_CORPUS["quad_face"]
        mesh = load_obj_from_string(quad)
        self.assertEqual(len(mesh.indices), 6)
        self.assertEqual(len(mesh.vertices), 4)

    def test_polygon_6gon_triangulation(self):
        poly = MALFORMED_OBJ_CORPUS["polygon_6gon"]
        mesh = load_obj_from_string(poly)
        self.assertEqual(len(mesh.indices), 12)
        self.assertEqual(len(mesh.vertices), 6)

    def test_negative_relative_indices(self):
        neg = MALFORMED_OBJ_CORPUS["negative_relative_indices"]
        mesh = load_obj_from_string(neg)
        self.assertEqual(len(mesh.indices), 3)
        self.assertEqual(len(mesh.vertices), 3)

    def test_empty_and_comments_only(self):
        m_empty = load_obj_from_string(MALFORMED_OBJ_CORPUS["empty"])
        self.assertEqual(len(m_empty.indices), 0)
        m_comm = load_obj_from_string(MALFORMED_OBJ_CORPUS["comments_only"])
        self.assertEqual(len(m_comm.indices), 0)

    def test_malformed_corpus_errors(self):
        for key in [
            "missing_face_indices",
            "out_of_bounds_index",
            "zero_index",
            "non_numeric_vertex",
            "truncated_vertex",
            "truncated_normal",
            "truncated_uv",
        ]:
            with self.subTest(corpus_key=key):
                with self.assertRaises(ValueError):
                    load_obj_from_string(MALFORMED_OBJ_CORPUS[key])

    def test_truncated_coordinate_lines_raise_value_error(self):
        """Truncated coordinate lines (v, vn, vt) must raise ValueError rather than leaking IndexError."""
        truncated_cases = [
            "v\n",
            "v 1.0\n",
            "v 1.0 2.0\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn\nf 1 2 3\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 1.0\nf 1 2 3\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvn 1.0 2.0\nf 1 2 3\n",
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nvt\nf 1 2 3\n",
        ]
        for src in truncated_cases:
            with self.subTest(src=src.strip()):
                with self.assertRaises(ValueError):
                    load_obj_from_string(src)

    def test_load_obj_from_file_path(self):
        with tempfile.TemporaryDirectory() as td:
            fpath = Path(td) / "test.obj"
            fpath.write_text(SAMPLE_TRI_OBJ, encoding="utf-8")
            mesh = load_obj(str(fpath))
            self.assertEqual(len(mesh.indices), 3)

    def test_load_obj_from_direct_string(self):
        mesh = load_obj(SAMPLE_TRI_OBJ)
        self.assertEqual(len(mesh.indices), 3)

    def test_smooth_normals_computation(self):
        quad = MALFORMED_OBJ_CORPUS["quad_face"]
        mesh = load_obj_from_string(quad, smooth_normals=True)
        self.assertEqual(len(mesh.indices), 6)
        for v in mesh.vertices:
            self.assertAlmostEqual(v.normal.z, 1.0)


if __name__ == "__main__":
    unittest.main()
