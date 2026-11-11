"""Unit tests for ForwardRenderer, Pipeline cache, draw list state sorting, and command recording."""

from __future__ import annotations

import math
import struct
import unittest
from typing import Any, List

from examples.minigfx.camera import Camera, CameraParams
from examples.minigfx.material import LightParams, MaterialParams, ObjectParams
from examples.minigfx.math import Mat4x4, Quat, Vec3, Vec4
from examples.minigfx.renderer import (
    DrawCommand,
    ForwardRenderer,
    Pipeline,
    sort_draw_items,
)
from examples.minigfx.scene import Node, Scene, create_cube, create_sphere


class TestPipelineCache(unittest.TestCase):
    """Pipeline creation, caching identity, and mode switching tests."""

    def setUp(self) -> None:
        self.renderer = ForwardRenderer()

    def test_cache_miss_creates_new_pipeline(self) -> None:
        pipe = self.renderer.get_or_create_pipeline("dielectric")
        self.assertIsInstance(pipe, Pipeline)
        self.assertEqual(pipe.key, "dielectric")
        self.assertIn("dielectric", self.renderer.pipelines)
        self.assertIs(self.renderer.pipelines["dielectric"], pipe)

    def test_cache_hit_returns_identical_instance(self) -> None:
        p1 = self.renderer.get_or_create_pipeline("phong_opaque")
        p2 = self.renderer.get_or_create_pipeline("phong_opaque")
        self.assertIs(p1, p2, "Pipeline cache must return identical instance for matching key")

    def test_scenario_04_three_modes_exact_count(self) -> None:
        modes = ["dielectric", "metallic", "normal_debug"]
        for mode_idx in ["1", "2", "3", "1", "2", "3", "1", "1", "2", "3"]:
            mode_name = modes[int(mode_idx) - 1]
            self.renderer.get_or_create_pipeline(mode_name)

        self.assertEqual(
            len(self.renderer.pipelines),
            3,
            "ForwardRenderer must cache exactly 3 compiled pipelines after alternating queries",
        )
        self.assertEqual(set(self.renderer.pipelines.keys()), set(modes))

    def test_pipeline_shader_attributes(self) -> None:
        vs = object()
        fs = object()
        gpu_pipe = object()
        pipe = self.renderer.get_or_create_pipeline(
            "custom_pbr",
            vertex_shader=vs,
            fragment_shader=fs,
            gpu_pipeline=gpu_pipe,
        )
        self.assertIs(pipe.vertex_shader, vs)
        self.assertIs(pipe.fragment_shader, fs)
        self.assertIs(pipe.gpu_pipeline, gpu_pipe)

    def test_pipeline_string_and_named_object_keys(self) -> None:
        class NamedMode:
            def __init__(self, name: str) -> None:
                self.name = name

        mode_obj = NamedMode("wireframe_mode")
        p1 = self.renderer.get_or_create_pipeline(mode_obj)
        self.assertEqual(p1.key, "wireframe_mode")

        p2 = self.renderer.get_or_create_pipeline("wireframe_mode")
        self.assertIs(p1, p2)


class TestSceneTraversalAndFrustumCulling(unittest.TestCase):
    """Scene traversal, world transform computation, and view frustum culling."""

    def setUp(self) -> None:
        self.renderer = ForwardRenderer()
        self.camera = Camera(
            eye=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 1.0, 0.0),
            fov_y=math.radians(60.0),
            aspect=1.0,
            near=0.1,
            far=20.0,
        )

    def test_none_inputs_return_empty_list(self) -> None:
        self.assertEqual(self.renderer.build_draw_list(None, None), [])
        self.assertEqual(self.renderer.build_draw_list(Scene(), None), [])
        self.assertEqual(self.renderer.build_draw_list(None, self.camera), [])

    def test_visible_collection_finds_all_visible_nodes(self) -> None:
        scene = Scene()
        for idx in range(3):
            node = Node(f"vis_node_{idx}", mesh=create_cube(size=1.0))
            node.translation = Vec3(float(idx) - 1.0, 0.0, 0.0)
            scene.add_child(node)

        draw_items = self.renderer.build_draw_list(scene, self.camera)
        self.assertEqual(len(draw_items), 3)
        collected_names = [item["node"].name for item in draw_items]
        self.assertEqual(collected_names, ["vis_node_0", "vis_node_1", "vis_node_2"])

    def test_culling_behind_camera(self) -> None:
        scene = Scene()
        node = Node("behind_camera", mesh=create_cube(size=1.0))
        # Camera is at z=5 looking at z=0, so z=100 is far behind the camera
        node.translation = Vec3(0.0, 0.0, 100.0)
        scene.add_child(node)

        draw_items = self.renderer.build_draw_list(scene, self.camera)
        self.assertEqual(len(draw_items), 0, "Geometry behind camera must be culled")

    def test_culling_outside_lateral_frustum(self) -> None:
        scene = Scene()
        node = Node("outside_lateral", mesh=create_cube(size=1.0))
        node.translation = Vec3(500.0, 0.0, 0.0)
        scene.add_child(node)

        draw_items = self.renderer.build_draw_list(scene, self.camera)
        self.assertEqual(len(draw_items), 0, "Geometry outside lateral frustum must be culled")

    def test_partial_frustum_culling(self) -> None:
        scene = Scene()
        vis1 = Node("vis_1", mesh=create_cube())
        vis1.translation = Vec3(0.0, 0.0, 0.0)
        scene.add_child(vis1)

        vis2 = Node("vis_2", mesh=create_cube())
        vis2.translation = Vec3(0.5, 0.5, 0.0)
        scene.add_child(vis2)

        culled1 = Node("culled_behind", mesh=create_cube())
        culled1.translation = Vec3(0.0, 0.0, 50.0)
        scene.add_child(culled1)

        culled2 = Node("culled_far", mesh=create_cube())
        culled2.translation = Vec3(0.0, 0.0, -100.0)
        scene.add_child(culled2)

        draw_items = self.renderer.build_draw_list(scene, self.camera)
        self.assertEqual(len(draw_items), 2)
        names = {item["node"].name for item in draw_items}
        self.assertEqual(names, {"vis_1", "vis_2"})

    def test_hierarchical_transforms_in_culling(self) -> None:
        scene = Scene()
        parent = Node("parent")
        # Parent translation puts origin at z = 100
        parent.translation = Vec3(0.0, 0.0, 100.0)
        scene.add_child(parent)

        # Child 1 moves 100 back to z = 0 (visible)
        child_vis = Node("child_vis", mesh=create_cube())
        child_vis.translation = Vec3(0.0, 0.0, -100.0)
        parent.add_child(child_vis)

        # Child 2 stays at z = 100 (culled)
        child_culled = Node("child_culled", mesh=create_cube())
        child_culled.translation = Vec3(0.0, 0.0, 10.0)
        parent.add_child(child_culled)

        draw_items = self.renderer.build_draw_list(scene, self.camera)
        self.assertEqual(len(draw_items), 1)
        self.assertEqual(draw_items[0]["node"].name, "child_vis")

    def test_container_scene_node_without_mesh_skipped(self) -> None:
        scene = Scene("root_container")
        self.assertIsNone(scene.mesh)
        child = Node("cube_node", mesh=create_cube())
        scene.add_child(child)

        draw_items = self.renderer.build_draw_list(scene, self.camera)
        self.assertEqual(len(draw_items), 1)
        self.assertEqual(draw_items[0]["node"].name, "cube_node")


class TestDrawListSorting(unittest.TestCase):
    """Draw list state sorting and pipeline state switch reduction."""

    def test_state_switch_elimination(self) -> None:
        items = [{"pipe": "B"}, {"pipe": "A"}, {"pipe": "B"}, {"pipe": "A"}]
        sorted_items = sort_draw_items(items)
        pipes = [x["pipe"] for x in sorted_items]
        self.assertEqual(pipes, ["A", "A", "B", "B"])

    def test_sorting_stability_preserves_order(self) -> None:
        items = [
            {"pipe": "B", "id": 1},
            {"pipe": "A", "id": 2},
            {"pipe": "B", "id": 3},
            {"pipe": "A", "id": 4},
        ]
        sorted_items = sort_draw_items(items)
        self.assertEqual(
            [(x["pipe"], x["id"]) for x in sorted_items],
            [("A", 2), ("A", 4), ("B", 1), ("B", 3)],
        )

    def test_multi_pipeline_grouping(self) -> None:
        keys = ["p3", "p1", "p2", "p1", "p3", "p2", "p1"]
        items = [{"pipe": k, "idx": i} for i, k in enumerate(keys)]
        sorted_items = sort_draw_items(items)
        sorted_pipes = [x["pipe"] for x in sorted_items]
        self.assertEqual(sorted_pipes, ["p1", "p1", "p1", "p2", "p2", "p3", "p3"])

    def test_dictionary_and_object_polymorphism(self) -> None:
        class ItemObj:
            def __init__(self, pipe: str) -> None:
                self.pipe = pipe

        items: List[Any] = [
            ItemObj("pipe_z"),
            {"pipeline_id": "pipe_a"},
            {"pipeline": "pipe_m"},
            ItemObj("pipe_a"),
        ]
        sorted_items = sort_draw_items(items)

        def _get_pipe(item: Any) -> str:
            if isinstance(item, dict):
                return item.get("pipe") or item.get("pipeline") or item.get("pipeline_id") or ""
            return getattr(item, "pipe")

        extracted = [_get_pipe(x) for x in sorted_items]
        self.assertEqual(extracted, ["pipe_a", "pipe_a", "pipe_m", "pipe_z"])

    def test_empty_and_single_item_sorting(self) -> None:
        self.assertEqual(sort_draw_items([]), [])
        single = [{"pipe": "only_one"}]
        self.assertEqual(sort_draw_items(single), single)


class TestStd430ConstantPacking(unittest.TestCase):
    """std430 binary buffer serialization and alignment verification."""

    def setUp(self) -> None:
        self.renderer = ForwardRenderer()

    def test_camera_params_exact_80_bytes(self) -> None:
        cam = Camera(eye=(1.0, 2.0, 3.0), target=(0.0, 0.0, 0.0))
        buf = self.renderer.update_frame_constants(cam, time=1.5)
        self.assertEqual(len(buf), 80, "CameraParams frame constant block must be 80 bytes std430")

    def test_camera_params_struct_unpacking_offsets(self) -> None:
        cam = Camera(eye=(1.5, 2.5, 3.5), target=(0.0, 0.0, 0.0))
        test_time = 4.25
        buf = self.renderer.update_frame_constants(cam, time=test_time)

        # Offset 0..64: Mat4x4 view_proj (16 floats)
        vp_unpacked = struct.unpack_from("<16f", buf, 0)
        expected_vp = cam.view_proj.elements
        for a, b in zip(vp_unpacked, expected_vp):
            self.assertAlmostEqual(a, b, places=5)

        # Offset 64..76: Vec3 eye_pos (3 floats)
        eye_unpacked = struct.unpack_from("<3f", buf, 64)
        self.assertAlmostEqual(eye_unpacked[0], 1.5, places=5)
        self.assertAlmostEqual(eye_unpacked[1], 2.5, places=5)
        self.assertAlmostEqual(eye_unpacked[2], 3.5, places=5)

        # Offset 76..80: f32 time (1 float)
        time_unpacked = struct.unpack_from("<f", buf, 76)[0]
        self.assertAlmostEqual(time_unpacked, test_time, places=5)

    def test_object_params_greater_equal_64_bytes(self) -> None:
        node = Node("test_mesh")
        mat = MaterialParams()
        buf = self.renderer.upload_object_constants(node, mat)
        self.assertGreaterEqual(len(buf), 64, "Object constants must contain at least Mat4x4 model transform")
        self.assertEqual(len(buf), 80, "Object constants with color are 80 bytes std430")

    def test_object_params_model_matrix_verification(self) -> None:
        node = Node("transformed_node")
        node.translation = Vec3(7.0, 8.0, 9.0)
        node.scale = Vec3(2.0, 2.0, 2.0)
        buf = self.renderer.upload_object_constants(node)

        model_unpacked = struct.unpack_from("<16f", buf, 0)
        expected_model = node.world_transform.elements
        for a, b in zip(model_unpacked, expected_model):
            self.assertAlmostEqual(a, b, places=5)

    def test_object_params_color_from_material(self) -> None:
        node = Node("colored_node")
        mat = MaterialParams(albedo=Vec4(0.1, 0.2, 0.3, 0.4))
        buf = self.renderer.upload_object_constants(node, mat)

        color_unpacked = struct.unpack_from("<4f", buf, 64)
        self.assertAlmostEqual(color_unpacked[0], 0.1, places=5)
        self.assertAlmostEqual(color_unpacked[1], 0.2, places=5)
        self.assertAlmostEqual(color_unpacked[2], 0.3, places=5)
        self.assertAlmostEqual(color_unpacked[3], 0.4, places=5)


class TestCommandRecordingAndIsolation(unittest.TestCase):
    """Draw command creation, parameter binding, and push constant isolation."""

    def setUp(self) -> None:
        self.renderer = ForwardRenderer()

    def test_command_creation_attributes(self) -> None:
        cube = create_cube()
        node = Node("cube_node", mesh=cube)
        cmd = self.renderer.record_draw(
            pipeline_id="pbr_pipe",
            push_constants=b"\xab" * 80,
            mesh=cube,
            node=node,
            first_index=12,
        )
        self.assertIsInstance(cmd, DrawCommand)
        self.assertEqual(cmd.pipeline_id, "pbr_pipe")
        self.assertEqual(cmd.push_constants, b"\xab" * 80)
        self.assertIs(cmd.mesh, cube)
        self.assertIs(cmd.node, node)
        self.assertEqual(cmd.vertex_count, len(cube.vertices))
        self.assertEqual(cmd.index_count, len(cube.indices))
        self.assertEqual(cmd.first_index, 12)

    def test_push_constant_isolation(self) -> None:
        cmd0 = self.renderer.record_draw(pipeline_id="pipe_a", push_constants=b"\x01" * 80)
        cmd1 = self.renderer.record_draw(pipeline_id="pipe_b", push_constants=b"\x02" * 80)
        self.assertNotEqual(cmd0.push_constants, cmd1.push_constants)

    def test_defensive_copy_immutability(self) -> None:
        mutable_buf = bytearray(b"\x55" * 80)
        cmd = self.renderer.record_draw(pipeline_id="pipe_test", push_constants=mutable_buf)
        self.assertEqual(cmd.push_constants[0], 0x55)

        # Mutate the source buffer
        mutable_buf[0] = 0xAA
        self.assertEqual(cmd.push_constants[0], 0x55, "Recorded command must retain defensive copy of push constants")

    def test_auto_upload_constants(self) -> None:
        node = Node("auto_const_node", mesh=create_cube())
        node.translation = Vec3(10.0, 20.0, 30.0)
        draw_list = [{"pipe": "p1", "node": node, "mesh": node.mesh}]

        commands = self.renderer.record_commands(draw_list)
        self.assertEqual(len(commands), 1)
        cmd = commands[0]
        self.assertEqual(len(cmd.push_constants), 80)

        # Validate that model transform is present in auto-uploaded constants
        model_m = struct.unpack_from("<16f", cmd.push_constants, 0)
        expected = node.world_transform.elements
        for a, b in zip(model_m, expected):
            self.assertAlmostEqual(a, b, places=5)

    def test_record_commands_sorted_order(self) -> None:
        draw_list = [
            {"pipe": "pipe_b", "mesh": None, "node": None},
            {"pipe": "pipe_a", "mesh": None, "node": None},
        ]
        commands = self.renderer.record_commands(draw_list)
        self.assertEqual([cmd.pipeline_id for cmd in commands], ["pipe_a", "pipe_b"])


class TestInputValidationAndErrorHandling(unittest.TestCase):
    """Input validation and error raising contracts."""

    def setUp(self) -> None:
        self.renderer = ForwardRenderer()
        self.camera = Camera()
        self.lights = LightParams()

    def test_render_frame_camera_none_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.renderer.render_frame(camera=None, lights=self.lights)

    def test_render_frame_lights_none_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.renderer.render_frame(camera=self.camera, lights=None)

    def test_render_frame_both_none_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            self.renderer.render_frame(camera=None, lights=None)


class TestRenderLoopExecution(unittest.TestCase):
    """Full render pass execution and frame lifecycle."""

    def setUp(self) -> None:
        self.renderer = ForwardRenderer(offscreen=True)
        self.camera = Camera(
            eye=(0.0, 0.0, 5.0),
            target=(0.0, 0.0, 0.0),
            fov_y=math.radians(60.0),
            aspect=1.0,
            near=0.1,
            far=20.0,
        )
        self.lights = LightParams()

    def test_clean_headless_execution(self) -> None:
        scene = Scene()
        node = Node("center_mesh", mesh=create_cube())
        scene.add_child(node)

        commands = self.renderer.render_frame(
            camera=self.camera,
            lights=self.lights,
            scene=scene,
            time=0.5,
        )
        self.assertIsInstance(commands, list)
        self.assertEqual(len(commands), 1)
        self.assertEqual(len(self.renderer.frame_constants), 80)

    def test_empty_scene_render(self) -> None:
        scene = Scene()
        commands = self.renderer.render_frame(
            camera=self.camera,
            lights=self.lights,
            scene=scene,
            time=0.0,
        )
        self.assertEqual(len(commands), 0)

    def test_multi_mesh_scene_render(self) -> None:
        scene = Scene()
        for i in range(4):
            node = Node(f"node_{i}", mesh=create_sphere(radius=0.5))
            node.translation = Vec3(float(i) - 1.5, 0.0, 0.0)
            scene.add_child(node)

        commands = self.renderer.render_frame(
            camera=self.camera,
            lights=self.lights,
            scene=scene,
        )
        self.assertEqual(len(commands), 4)

    def test_consecutive_frames_execution(self) -> None:
        scene = Scene()
        node = Node("moving_node", mesh=create_cube())
        scene.add_child(node)

        for step in range(3):
            t = float(step) * 0.016
            commands = self.renderer.render_frame(
                camera=self.camera,
                lights=self.lights,
                scene=scene,
                time=t,
            )
            self.assertEqual(len(commands), 1)
            unpacked_time = struct.unpack_from("<f", self.renderer.frame_constants, 76)[0]
            self.assertAlmostEqual(unpacked_time, t, places=5)


if __name__ == "__main__":
    unittest.main()
