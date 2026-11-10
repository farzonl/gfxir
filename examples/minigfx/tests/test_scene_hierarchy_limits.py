"""Stress tests for deep scene hierarchy scaling, cycle detection, non-uniform scaling bounds, and frustum culling.

Tests 4 critical dimensions:
1. Deep Hierarchy Scaling (1,000 nodes, iterative compounding without RecursionError, dirty propagation)
2. Complex Cycle Detection Stress (self, 2-node, deep ancestor, diamond re-parenting, state preservation)
3. Non-Uniform Scaling & Spectral Norm Bounding Radius (shear conservatism and frustum visibility)
4. Frustum Boundary Precision & Cross-Component Parity (tangent, eps-inside, eps-outside, 6 planes, corner tests)
"""

from __future__ import annotations

import math
from pathlib import Path
import random
import sys
import unittest

# Ensure repo root and src are in sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

from examples.minigfx.camera import Camera
from examples.minigfx.math import Mat4x4, Quat, Vec2, Vec3, Vec4
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


class TestDeepHierarchyLimits(unittest.TestCase):
    """Stress tests for deep node hierarchies (1,000 nodes)."""

    def test_deep_hierarchy_scaling_without_recursion_error(self):
        """A 1,000-node hierarchy executes world_transform, dirty propagation, and traversal without RecursionError."""
        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(1000)

            # Construct 1,000 node chain
            nodes = [
                Node(
                    name=f"node_{i}",
                    translation=Vec3(0.01, 0.02, 0.03),
                    rotation=Quat.from_axis_angle(Vec3(0, 1, 0), math.radians(0.36)),
                    scale=Vec3(1.0001, 1.0, 0.9999),
                )
                for i in range(1000)
            ]
            for i in range(999):
                nodes[i].add_child(nodes[i + 1])

            # 1. world_transform recursion check
            wt = nodes[-1].world_transform
            self.assertIsNotNone(wt)

            # 2. _mark_dirty recursion check
            nodes[0].translation = Vec3(1.0, 0.0, 0.0)
            self.assertTrue(nodes[-1]._dirty)
            self.assertIsNotNone(nodes[-1].world_transform)
            self.assertFalse(nodes[-1]._dirty)

            # 3. traverse recursion check
            trav = list(nodes[0].traverse())
            self.assertEqual(len(trav), 1000)

            # 4. collect_visible recursion check
            cam = Camera(eye=Vec3(0, 0, 10), target=Vec3(0, 0, 0), near=1.0, far=50.0)
            vis = nodes[0].collect_visible(cam)
            self.assertIsInstance(vis, list)

        finally:
            sys.setrecursionlimit(original_limit)

    def test_deep_hierarchy_compounding_numerical_stability(self):
        """Under default stack depth, test compounding 1,000 nodes."""
        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(1000)

            # 1,000 nodes each with small translation and rotation
            angle_per_node = math.radians(0.36)  # 360 degrees total across 1,000 nodes
            trans_per_node = Vec3(0.001, 0.002, -0.001)

            nodes = [
                Node(
                    name=f"node_{i}",
                    translation=trans_per_node,
                    rotation=Quat.from_axis_angle(Vec3(0, 1, 0), angle_per_node),
                    scale=Vec3(1.0, 1.0, 1.0),
                )
                for i in range(1000)
            ]
            for i in range(999):
                nodes[i].add_child(nodes[i + 1])

            # Compute world transform of leaf
            leaf_wt = nodes[-1].world_transform

            # Verify no NaNs or Infs
            for val in leaf_wt.elements:
                self.assertFalse(math.isnan(val), f"Leaf world transform contains NaN: {leaf_wt}")
                self.assertFalse(math.isinf(val), f"Leaf world transform contains Inf: {leaf_wt}")

            # Verify against iterative ground truth: M_0 * M_1 * ... * M_999
            expected_m = Mat4x4.identity()
            for node in nodes:
                expected_m = expected_m * node.local_transform

            for idx, (act, exp) in enumerate(zip(leaf_wt.elements, expected_m.elements)):
                self.assertAlmostEqual(
                    act, exp, places=4,
                    msg=f"Element {idx} mismatch: act={act}, exp={exp}"
                )

            # Test dirty propagation down 1,000 nodes
            nodes[0].translation = Vec3(10.0, 20.0, 30.0)
            # Verify leaf is marked dirty
            self.assertTrue(nodes[-1]._dirty)
            self.assertIsNone(nodes[-1]._cached_world_transform)

            # Recompute
            new_leaf_wt = nodes[-1].world_transform
            self.assertFalse(nodes[-1]._dirty)
            self.assertIsNotNone(nodes[-1]._cached_world_transform)

            # Cache hit: verify reading cached transform does not recompute
            cached_wt = nodes[-1].world_transform
            self.assertIs(cached_wt, new_leaf_wt)

        finally:
            sys.setrecursionlimit(original_limit)


class TestCycleDetectionStress(unittest.TestCase):
    """Stress tests for cycle detection across arbitrary tree topologies."""

    def test_self_parenting_raises_value_error(self):
        """Self-parenting (a.add_child(a)) must immediately raise ValueError and preserve clean state."""
        node = Node(name="solo")
        with self.assertRaises(ValueError) as ctx:
            node.add_child(node)
        self.assertIn("cycle detected", str(ctx.exception).lower())
        self.assertIsNone(node.parent)
        self.assertEqual(node.children, [])

    def test_immediate_two_node_cycle_raises_and_preserves_state(self):
        """b.add_child(a) when a.add_child(b) exists must raise ValueError and not corrupt tree."""
        a = Node(name="A")
        b = Node(name="B")
        a.add_child(b)
        self.assertIs(b.parent, a)
        self.assertEqual(a.children, [b])

        with self.assertRaises(ValueError) as ctx:
            b.add_child(a)
        self.assertIn("cycle detected", str(ctx.exception).lower())

        # State must remain uncorrupted: b is still child of a, b has no children
        self.assertIs(b.parent, a)
        self.assertEqual(b.children, [])
        self.assertEqual(a.children, [b])

    def test_deep_ancestor_cycles_across_topologies(self):
        """Deep ancestor cycle attempts across a 100-node linear chain."""
        nodes = [Node(name=f"node_{i}") for i in range(100)]
        for i in range(99):
            nodes[i].add_child(nodes[i + 1])

        # Attempt cycles from various depths
        for child_idx, parent_idx in [(99, 0), (99, 50), (50, 20), (25, 0), (1, 0)]:
            with self.assertRaises(ValueError, msg=f"Cycle {child_idx}->{parent_idx} not caught"):
                nodes[child_idx].add_child(nodes[parent_idx])

            # State check: parent must still have only 1 child (except leaf)
            self.assertEqual(len(nodes[parent_idx].children), 1)
            self.assertEqual(nodes[child_idx].children, [] if child_idx == 99 else [nodes[child_idx + 1]])

    def test_diamond_reparenting_moves_node_without_duplication(self):
        """Re-parenting across branches (DAG diamond attempt) moves the node safely."""
        root = Node(name="root")
        branch1 = Node(name="branch1")
        branch2 = Node(name="branch2")
        target = Node(name="target")

        root.add_child(branch1)
        root.add_child(branch2)
        branch1.add_child(target)

        self.assertIs(target.parent, branch1)
        self.assertEqual(branch1.children, [target])

        # Move target to branch2
        branch2.add_child(target)

        # target must be detached from branch1 and attached to branch2
        self.assertIs(target.parent, branch2)
        self.assertEqual(branch1.children, [])
        self.assertEqual(branch2.children, [target])

    def test_re_adding_same_child_is_idempotent(self):
        """Adding the same child twice must not duplicate child in children list."""
        parent = Node(name="parent")
        child = Node(name="child")

        parent.add_child(child)
        self.assertEqual(parent.children, [child])

        parent.add_child(child)
        self.assertEqual(parent.children, [child])
        self.assertEqual(len(parent.children), 1)


class TestNonUniformScalingBounds(unittest.TestCase):
    """Stress tests for extreme non-uniform scaling, aspect ratios, and bounding sphere conservatism."""

    def test_extreme_aspect_ratio_single_node_enclosure(self):
        """Single node with scale (1000.0, 0.001, 1.0) must conservatively enclose all vertices."""
        cube = create_cube(size=1.0)
        node = Node(name="stretched_cube", scale=Vec3(1000.0, 0.001, 1.0), mesh=cube)

        world_c, world_r = node.bounding_sphere
        wt = node.world_transform

        # Test every vertex in the cube
        for vert in cube.vertices:
            w_pos4 = wt * Vec4(vert.position.x, vert.position.y, vert.position.z, 1.0)
            w_pos = Vec3(w_pos4.x, w_pos4.y, w_pos4.z)
            dist_to_center = (w_pos - world_c).length()
            self.assertLessEqual(
                dist_to_center, world_r + 1e-4,
                f"Vertex {vert.position} escaped bounding sphere: dist={dist_to_center}, r={world_r}"
            )

    def test_shear_spectral_norm_conservatism(self):
        """Verify bounding radius is conservative across 500 randomized hierarchies under shearing."""
        diag = 1.0 / math.sqrt(2.0)
        test_vert = Vertex(Vec3(diag, diag, 0.0), Vec3(0, 0, 1))
        mesh = Mesh([test_vert], [0], center=Vec3(0, 0, 0), radius=1.0)

        found_violation = False
        worst_ratio = 1.0
        random.seed(42)
        for _ in range(500):
            q1 = Quat.from_axis_angle(
                Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)).normalize(),
                random.uniform(0, math.pi * 2)
            )
            s1 = Vec3(random.uniform(0.1, 10), random.uniform(0.1, 10), random.uniform(0.1, 10))
            q2 = Quat.from_axis_angle(
                Vec3(random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)).normalize(),
                random.uniform(0, math.pi * 2)
            )
            s2 = Vec3(random.uniform(0.1, 10), random.uniform(0.1, 10), random.uniform(0.1, 10))

            p_test = Node(rotation=q1, scale=s1)
            c_test = Node(rotation=q2, scale=s2, mesh=mesh)
            p_test.add_child(c_test)

            wc, wr = c_test.bounding_sphere
            c_wt = c_test.world_transform

            for _k in range(30):
                th = random.uniform(0, math.pi * 2)
                phi = random.uniform(-math.pi / 2, math.pi / 2)
                v_loc = Vec3(math.cos(phi) * math.cos(th), math.cos(phi) * math.sin(th), math.sin(phi))
                w4 = c_wt * Vec4(v_loc.x, v_loc.y, v_loc.z, 1.0)
                d = (Vec3(w4.x, w4.y, w4.z) - wc).length()
                if d > wr + 1e-4:
                    found_violation = True
                    ratio = d / wr
                    if ratio > worst_ratio:
                        worst_ratio = ratio

        self.assertFalse(
            found_violation,
            f"Expected spectral norm bound to enclose all vertices under shear, but violation found (worst ratio={worst_ratio})"
        )

    def test_frustum_visibility_under_hierarchical_shear(self):
        """Demonstrates mesh vertex in camera frustum is correctly marked visible under hierarchical shearing."""
        axis1 = Vec3(0.12489190905336965, -0.9102823653625428, -0.3946999193855744)
        angle1 = 0.5162481679103244
        s1 = Vec3(0.774763994580866, 0.26314179788115466, 9.495588663180738)

        axis2 = Vec3(0.3105136054597658, 0.6697657537103756, 0.6745332727013174)
        angle2 = 5.166257272107553
        s2 = Vec3(5.256314077003793, 3.156322449694864, 5.472965093840152)

        v_loc = Vec3(0.68216492885224, -0.13172892955936477, 0.7192346619575298)

        mesh = Mesh([Vertex(v_loc, Vec3(0, 1, 0))], [0], center=Vec3(0, 0, 0), radius=1.0)
        p = Node(name="p", rotation=Quat.from_axis_angle(axis1, angle1), scale=s1)
        c = Node(name="c", rotation=Quat.from_axis_angle(axis2, angle2), scale=s2, mesh=mesh)
        p.add_child(c)

        wc, wr = c.bounding_sphere
        p.translation = -wc
        p.invalidate()

        wc_new, wr_new = c.bounding_sphere
        w4 = c.world_transform * Vec4(v_loc.x, v_loc.y, v_loc.z, 1.0)
        v_world = Vec3(w4.x, w4.y, w4.z)
        dist_to_center = (v_world - wc_new).length()

        direction = (v_world - wc_new).normalize()
        cam = Camera(eye=v_world - direction * 5.0, target=v_world, near=0.1, far=50.0)

        w_clip = cam.view_proj * Vec4(v_world.x, v_world.y, v_world.z, 1.0)
        ndc_x = w_clip.x / w_clip.w
        ndc_y = w_clip.y / w_clip.w
        ndc_z = w_clip.z / w_clip.w
        vertex_inside_frustum = (-1.0 <= ndc_x <= 1.0) and (-1.0 <= ndc_y <= 1.0) and (0.0 <= ndc_z <= 1.0)

        node_visible = c.is_visible(cam)

        self.assertTrue(vertex_inside_frustum, f"Vertex at NDC ({ndc_x}, {ndc_y}, {ndc_z}) should be inside frustum")
        self.assertAlmostEqual(ndc_x, 0.0, places=3)
        self.assertAlmostEqual(ndc_y, 0.0, places=3)
        self.assertTrue(node_visible, "Node must be visible when vertex is inside frustum")
        self.assertLessEqual(dist_to_center, wr_new + 1e-4, "Transformed vertex must be enclosed within bounding radius")

    def test_degenerate_zero_and_negative_scales(self):
        """Zero scale collapses radius to 0; negative scale (mirroring) produces positive radius."""
        cube = create_cube(1.0)

        # Zero scale
        node_zero = Node(scale=Vec3(0.0, 0.0, 0.0), mesh=cube)
        _, r_zero = node_zero.bounding_sphere
        self.assertEqual(r_zero, 0.0)

        # Negative scale (reflection across X axis)
        node_neg = Node(scale=Vec3(-2.0, 2.0, 2.0), mesh=cube)
        _, r_neg = node_neg.bounding_sphere
        self.assertAlmostEqual(r_neg, cube.radius * 2.0, places=5)
        self.assertGreater(r_neg, 0.0)


class TestFrustumBoundaryPrecision(unittest.TestCase):
    """Stress tests for frustum boundary precision, tangent spheres, and cross-component consistency."""

    def setUp(self):
        self.cam = Camera(
            eye=Vec3(0.0, 0.0, 5.0),
            target=Vec3(0.0, 0.0, 0.0),
            up=Vec3(0.0, 1.0, 0.0),
            fov_y=math.radians(60.0),
            aspect=16.0 / 9.0,
            near=1.0,
            far=10.0,
        )
        self.frustum = Frustum.from_matrix(self.cam.view_proj)

    def test_frustum_planes_normalized(self):
        """All 6 frustum planes must have strictly unit normal vectors ||(nx, ny, nz)|| == 1.0."""
        for i, p in enumerate(self.cam.frustum_planes):
            norm_len = math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z)
            self.assertAlmostEqual(norm_len, 1.0, places=6, msg=f"Plane {i} normal not unit length: {norm_len}")

    def test_frustum_tangent_epsilon_boundary_precision(self):
        """Test spheres exactly tangent, eps-inside, and eps-outside across all 6 planes."""
        planes = self.cam.frustum_planes
        r = 1.0

        for i, p in enumerate(planes):
            n = Vec3(p.x, p.y, p.z)
            point_on_plane = n * (-p.w)

            tangent_center = point_on_plane + n * (-r)
            self.assertTrue(
                self.cam.is_sphere_visible(tangent_center, r),
                f"Plane {i}: Exactly tangent sphere should be visible within epsilon"
            )

            inside_center = point_on_plane + n * (-r + 1e-4)
            self.assertTrue(
                self.cam.is_sphere_visible(inside_center, r),
                f"Plane {i}: Epsilon-inside sphere should be visible"
            )

            outside_center = point_on_plane + n * (-r - 1e-4)
            self.assertFalse(
                self.cam.is_sphere_visible(outside_center, r),
                f"Plane {i}: Epsilon-outside sphere should be culled"
            )

    def test_cross_component_parity(self):
        """Node.is_visible, Frustum.intersects_sphere, and Camera.is_sphere_visible must be 100% consistent."""
        random.seed(42)
        mismatches = 0
        total_tests = 500

        for _ in range(total_tests):
            cx = random.uniform(-15.0, 15.0)
            cy = random.uniform(-15.0, 15.0)
            cz = random.uniform(-15.0, 15.0)
            center = Vec3(cx, cy, cz)
            radius = random.uniform(0.1, 5.0)

            node = Node(name="probe", translation=center, bounding_radius=radius)

            vis_cam = self.cam.is_sphere_visible(center, radius)
            vis_frust = self.frustum.intersects_sphere(center, radius)
            vis_node = node.is_visible(self.cam)

            if not (vis_cam == vis_frust == vis_node):
                mismatches += 1

        self.assertEqual(mismatches, 0, f"Found {mismatches}/{total_tests} consistency mismatches")

    def test_frustum_8_corners_visibility(self):
        """Spheres placed at all 8 corners of the view frustum must be visible."""
        inv_vp = self.cam.view_proj.inverse()

        ndc_corners = [
            (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0),  # Near
            (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1),  # Far
        ]

        for x, y, z in ndc_corners:
            w = inv_vp * Vec4(x, y, z, 1.0)
            corner_pt = Vec3(w.x / w.w, w.y / w.w, w.z / w.w)

            self.assertTrue(
                self.cam.is_sphere_visible(corner_pt, 0.1),
                f"Corner ({x}, {y}, {z}) at {corner_pt} must be visible"
            )


if __name__ == "__main__":
    unittest.main()
