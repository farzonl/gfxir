"""Unit and integration tests for ModelViewer application."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from examples.minigfx.app import App
from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.material import LightParams, MaterialParams
from examples.minigfx.math import Mat4x4, Quat, Vec2, Vec3, Vec4
from examples.minigfx.samples.model_viewer import ModelViewerApp, ModelVertex, model_vertex_shader
from mlir_gpu.viewer import RenderedImage, ViewerError


class TestModelViewerApp(unittest.TestCase):
    """Behavioral test suite for ModelViewerApp interactive sample scene."""

    def test_scene_graph_hierarchy_initialization(self) -> None:
        """Verifies scene hierarchy contains central model, ground plane, light marker, and satellite."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        self.assertIsNotNone(app.scene)
        self.assertGreaterEqual(len(app.scene.children), 3)

        self.assertIsNotNone(app.model_node)
        self.assertIsNotNone(app.model_node.mesh)
        self.assertGreater(len(app.model_node.mesh.vertices), 0)
        self.assertGreater(len(app.model_node.mesh.indices), 0)

        self.assertIsNotNone(app.ground_node)
        self.assertIsNotNone(app.ground_node.mesh)
        self.assertEqual(len(app.ground_node.mesh.vertices), 4)

        self.assertIsNotNone(app.light_node)
        self.assertIsNotNone(app.light_node.mesh)
        self.assertGreater(len(app.light_node.mesh.vertices), 0)

        self.assertIsNotNone(app.satellite_node)
        self.assertIsNotNone(app.satellite_node.mesh)

    def test_camera_and_orbit_controller_setup(self) -> None:
        """Verifies 3D perspective camera and orbit controller parameters."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--width", "800", "--height", "600"])
        app = ModelViewerApp(args)

        self.assertIsNotNone(app.camera)
        self.assertAlmostEqual(app.camera.aspect, 800.0 / 600.0, places=4)
        self.assertAlmostEqual(app.camera.fov_y, math.radians(60.0), places=4)

        view_proj = app.camera.view_proj
        self.assertIsInstance(view_proj, Mat4x4)

        self.assertIsNotNone(app.camera_controller)
        self.assertIsInstance(app.camera_controller, OrbitController)

    def test_orbit_camera_mouse_drag_navigation(self) -> None:
        """Verifies mouse drag rotates camera yaw and updates target angles."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)
        ctrl = app.camera_controller

        yaw_before = getattr(ctrl, "azimuth", ctrl.yaw)
        app.handle_mouse_drag(dx=15.0, dy=0.0, button=1, shift=False)
        yaw_after = getattr(ctrl, "azimuth", ctrl.yaw)

        self.assertNotEqual(yaw_before, yaw_after)

    def test_orbit_camera_pitch_boundary_clamping(self) -> None:
        """Verifies extreme vertical drag is clamped strictly within safe bounds avoiding gimbal lock."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)
        ctrl = app.camera_controller

        # Extreme positive drag
        app.handle_mouse_drag(dx=0.0, dy=1e7, button=1, shift=False)
        pitch_max = getattr(ctrl, "elevation", ctrl.pitch)
        self.assertTrue(-math.pi / 2 < pitch_max < math.pi / 2)
        self.assertLessEqual(pitch_max, 1.5)
        self.assertFalse(math.isnan(pitch_max))

        # Extreme negative drag
        app.handle_mouse_drag(dx=0.0, dy=-2e7, button=1, shift=False)
        pitch_min = getattr(ctrl, "elevation", ctrl.pitch)
        self.assertTrue(-math.pi / 2 < pitch_min < math.pi / 2)
        self.assertGreaterEqual(pitch_min, -1.5)
        self.assertFalse(math.isnan(pitch_min))

    def test_camera_planar_pan_navigation(self) -> None:
        """Verifies planar panning translates camera target in local camera plane."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)
        ctrl = app.camera_controller

        pos_before = ctrl.target
        app.handle_mouse_drag(dx=20.0, dy=10.0, button=2)
        pos_after = ctrl.target

        self.assertNotEqual(pos_before, pos_after)

    def test_orbit_camera_scroll_distance_bounds(self) -> None:
        """Verifies mouse scroll wheel zooms with min/max distance clamping."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)
        ctrl = app.camera_controller

        dist_start = ctrl.distance

        # Zoom in
        app.handle_mouse_scroll(delta=5.0)
        self.assertLess(ctrl.distance, dist_start)

        # Extreme zoom in clamped to min_distance
        app.handle_mouse_scroll(delta=1e6)
        self.assertAlmostEqual(ctrl.distance, 0.5, places=3)

        # Extreme zoom out clamped to max_distance
        app.handle_mouse_scroll(delta=-1e6)
        self.assertAlmostEqual(ctrl.distance, 100.0, places=3)

    def test_material_preset_selection_and_cycling(self) -> None:
        """Verifies material switching across metallic, dielectric, and normal debug presets."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        self.assertEqual(app.current_material_name, "metallic")

        # Key 2: dielectric
        app.set_material("dielectric")
        self.assertEqual(app.current_material_name, "dielectric")

        # Key 3: normal_debug
        app.set_material("normal_debug")
        self.assertEqual(app.current_material_name, "normal_debug")

        # Cycle back to metallic
        cycled = app.cycle_material()
        self.assertEqual(cycled, "metallic")
        self.assertEqual(app.current_material_name, "metallic")

    def test_rapid_material_switching_stability(self) -> None:
        """Verifies application state remains consistent under rapid material cycling."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        for _ in range(75):
            name = app.cycle_material()
            self.assertIn(name, ["metallic", "dielectric", "normal_debug"])

        self.assertIsNotNone(app.current_material_name)

    def test_dynamic_sunlight_orbit_animation(self) -> None:
        """Verifies sunlight vector and light indicator sphere orbit over simulation updates."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        dir_initial = app.light_params.direction
        pos_initial = app.light_node.translation

        app.update(dt=0.1)

        dir_updated = app.light_params.direction
        pos_updated = app.light_node.translation

        self.assertNotEqual(dir_initial, dir_updated)
        self.assertNotEqual(pos_initial, pos_updated)

    def test_light_animation_toggling(self) -> None:
        """Verifies space toggle pauses and resumes dynamic lighting orbit."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        self.assertTrue(app.animate_light)
        app.toggle_light_animation()
        self.assertFalse(app.animate_light)

        dir_paused = app.light_params.direction
        app.update(dt=0.2)
        dir_after = app.light_params.direction

        self.assertEqual(dir_paused, dir_after)

    @patch("examples.minigfx.samples.model_viewer.render_offscreen")
    def test_zero_light_intensity_render_safety(self, mock_render: MagicMock) -> None:
        """Verifies rendering proceeds cleanly without divide-by-zero when light intensity is zero."""
        mock_render.return_value = RenderedImage(640, 480, b"\x00" * (640 * 480 * 4), "test_device")
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        app.light_params.intensity = 0.0
        commands = app.render()
        self.assertIsInstance(commands, list)

    def test_missing_model_file_raises_file_not_found(self) -> None:
        """Verifies passing a non-existent model path raises FileNotFoundError immediately."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--model", "missing_model_file.obj"])
        with self.assertRaises(FileNotFoundError):
            ModelViewerApp(args)

    def test_corrupted_model_file_raises_value_error(self) -> None:
        """Verifies corrupted OBJ file without valid geometry raises ValueError immediately."""
        with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as f:
            f.write("corrupted obj content with invalid syntax $$$$")
            corrupt_path = f.name

        try:
            args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--model", corrupt_path])
            with self.assertRaises(ValueError):
                ModelViewerApp(args)
        finally:
            if os.path.exists(corrupt_path):
                os.unlink(corrupt_path)

    def test_malformed_syntax_model_file_raises_value_error(self) -> None:
        """Verifies OBJ file with malformed syntax raises ValueError immediately."""
        with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as f:
            f.write("v 1.0 2.0\n")
            malformed_path = f.name

        try:
            args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--model", malformed_path])
            with self.assertRaises(ValueError):
                ModelViewerApp(args)
        finally:
            if os.path.exists(malformed_path):
                os.unlink(malformed_path)

    def test_empty_model_argument_raises_error(self) -> None:
        """Verifies empty --model argument string raises an error immediately."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--model", ""])
        with self.assertRaises((ValueError, FileNotFoundError)):
            ModelViewerApp(args)

    def test_omitted_model_argument_loads_default_model(self) -> None:
        """Verifies omitting --model argument loads the default procedural model mesh."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)
        self.assertIsNone(app.model_path)
        self.assertIsNotNone(app.model_node.mesh)
        self.assertEqual(len(app.model_node.mesh.vertices), 42)
        self.assertEqual(len(app.model_node.mesh.indices), 240)

    def test_custom_valid_obj_model_loading(self) -> None:
        """Verifies custom valid Wavefront OBJ file is loaded successfully into central model node."""
        valid_obj = """# Tetrahedral Mesh
v 0.0 1.0 0.0
v -1.0 -1.0 1.0
v 1.0 -1.0 1.0
v 0.0 -1.0 -1.0
f 1 2 3
f 1 3 4
f 1 4 2
f 2 4 3
"""
        with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as f:
            f.write(valid_obj)
            valid_path = f.name

        try:
            args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--model", valid_path])
            app = ModelViewerApp(args)
            self.assertIsNotNone(app.model_node.mesh)
            self.assertEqual(len(app.model_node.mesh.indices), 12)
        finally:
            if os.path.exists(valid_path):
                os.unlink(valid_path)

    def test_keyboard_event_handling(self) -> None:
        """Verifies keyboard interaction dispatches to material presets and light toggle."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        app.handle_key_press("2")
        self.assertEqual(app.current_material_name, "dielectric")

        app.handle_key_press("3")
        self.assertEqual(app.current_material_name, "normal_debug")

        app.handle_key_press("1")
        self.assertEqual(app.current_material_name, "metallic")

        self.assertTrue(app.animate_light)
        app.handle_key_press(" ")
        self.assertFalse(app.animate_light)

    def test_headless_execution_with_zero_frames(self) -> None:
        """Verifies zero-frame execution exits cleanly with code 0 without advancing frames."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)
        exit_code = app.run()

        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 0)

    @patch("examples.minigfx.samples.model_viewer.render_offscreen")
    def test_headless_execution_with_finite_frames(self, mock_render: MagicMock) -> None:
        """Verifies finite-frame execution renders requested number of frames and terminates."""
        mock_render.return_value = RenderedImage(640, 480, b"\x00" * (640 * 480 * 4), "test_device")
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "3"])
        app = ModelViewerApp(args)
        exit_code = app.run()

        self.assertEqual(exit_code, 0)
        self.assertEqual(app.frame_count, 3)
        self.assertEqual(mock_render.call_count, 3)

    def test_command_line_argument_parsing_validation(self) -> None:
        """Verifies CLI parser validation rules for dimensions, frame counts, and backend targets."""
        parser = ModelViewerApp.build_parser()

        # Valid options
        parsed = parser.parse_args(["--target", "metal", "--frames", "10", "--width", "1920", "--height", "1080"])
        self.assertEqual(parsed.target, "metal")
        self.assertEqual(parsed.frames, 10)
        self.assertEqual(parsed.width, 1920)
        self.assertEqual(parsed.height, 1080)

        # Invalid target choice
        with self.assertRaises(SystemExit):
            parser.parse_args(["--target", "unsupported_backend"])

        # Negative frames
        with self.assertRaises(SystemExit):
            parser.parse_args(["--frames", "-5"])

        # Non-positive dimension
        with self.assertRaises(SystemExit):
            parser.parse_args(["--width", "0"])

    def test_strict_matrix_shape_types_usage(self) -> None:
        """Verifies that Mat4x4 shape-named type is used throughout and deprecated aliases are absent."""
        import examples.minigfx.math as mgfx_math
        from examples.minigfx.samples.model_viewer import Mat4x4 as SampleMat4x4

        self.assertEqual(SampleMat4x4, mgfx_math.Mat4x4)
        self.assertFalse(hasattr(mgfx_math, "Mat4"), "Legacy Mat4 alias must remain removed")

    def test_window_mesh_and_hud_vertex_packing(self) -> None:
        """Verifies triangle storage buffer packing for scene meshes and HUD glyph quads."""
        from examples.minigfx.window import build_hud_overlay_vertices, pack_mesh_triangles

        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0"])
        app = ModelViewerApp(args)

        mesh_bytes, mesh_vert_count = pack_mesh_triangles(app.model_node.mesh)
        self.assertGreater(mesh_vert_count, 0)
        self.assertEqual(mesh_vert_count % 3, 0)
        self.assertEqual(len(mesh_bytes), mesh_vert_count * 32)

        hud_bytes, hud_vert_count = build_hud_overlay_vertices(
            [("FPS: 60.0 | Backend: metal", (1.0, 1.0, 1.0, 1.0))],
            1280,
            720,
        )
        self.assertGreater(hud_vert_count, 0)
        self.assertEqual(hud_vert_count % 6, 0)
        self.assertEqual(len(hud_bytes), hud_vert_count * 32)

    def test_window_embedded_shader_compilation(self) -> None:
        """Verifies embedded window shaders compile to SPIR-V when GfxIR toolchain is available."""
        import shutil
        from examples.minigfx.window import _ensure_toolchain_paths, compile_minigfx_shaders

        _ensure_toolchain_paths()
        if shutil.which("gfx-compile") is None or shutil.which("spirv-val") is None:
            self.skipTest("gfx-compile or spirv-val not available in PATH")

        shaders = compile_minigfx_shaders("vulkan")
        for stage_key in ("mesh_vs", "pbr_fs", "normal_fs", "overlay_vs", "overlay_fs"):
            self.assertIn(stage_key, shaders)
            self.assertTrue(shaders[stage_key].is_file())
            self.assertGreater(shaders[stage_key].stat().st_size, 0)

    def test_offscreen_render_uses_gpu_viewer(self) -> None:
        """Verifies that offscreen rendering dispatches through mlir_gpu.viewer.render_offscreen."""
        import examples.minigfx.samples.model_viewer as mv_module

        source = inspect.getsource(mv_module.ModelViewerApp.render_offscreen_frame)
        self.assertIn("render_offscreen", source, "render_offscreen must be invoked directly")

    def test_deterministic_offscreen_timing(self) -> None:
        """Verifies that offscreen mode enforces deterministic 60.0 FPS / 16.7 ms timing."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "1"])
        app = ModelViewerApp(args)

        app.update(dt=0.000001)  # Simulate microsecond spike
        self.assertEqual(app.fps, 60.0)
        self.assertAlmostEqual(app.dt, 1.0 / 60.0, places=5)

    def test_embedded_pipeline_compilation(self) -> None:
        """Verifies genuine shader compilation produces valid pipeline manifest and input layout."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--target", "vulkan"])
        app = ModelViewerApp(args)

        manifest_path = app.compile_embedded_pipeline("vulkan")
        self.assertTrue(manifest_path.exists())
        manifest = json.loads(manifest_path.read_text())

        self.assertIn("input_layout", manifest)
        self.assertIn("entries", manifest)
        self.assertEqual(len(manifest["entries"]), 2)
        self.assertEqual(manifest["target"], "vulkan")

    def test_clean_architecture_zero_base_class_monkeypatching(self) -> None:
        """Verifies that importing model_viewer does not mutate base App, OrbitController, or DebugOverlay."""
        from examples.minigfx.app import App as BaseApp
        from examples.minigfx.camera import OrbitController as BaseOrbitController
        import examples.minigfx.samples.model_viewer as mv_module

        # Base App parser must NOT be contaminated with model_viewer-specific flags
        base_parser = BaseApp.build_parser()
        base_options = [opt for action in base_parser._actions for opt in action.option_strings]
        self.assertNotIn("--model", base_options, "Base App.build_parser must not contain --model")
        self.assertNotIn("--output", base_options, "Base App.build_parser must not contain --output")

        # OrbitController class must not have monkeypatched property aliases
        self.assertNotIn("azimuth", BaseOrbitController.__dict__, "OrbitController must not be monkeypatched with azimuth")
        self.assertNotIn("elevation", BaseOrbitController.__dict__, "OrbitController must not be monkeypatched with elevation")

    def test_dedicated_overlay_pipeline_compilation_and_draw_routing(self) -> None:
        """Verifies dedicated overlay pipeline is compiled with depth disabled and routed to text quads."""
        args = ModelViewerApp.parse_args(["--offscreen", "--frames", "0", "--target", "vulkan"])
        app = ModelViewerApp(args)

        pbr_manifest_path = app.compile_embedded_pipeline("vulkan")
        self.assertTrue(pbr_manifest_path.exists())

        overlay_manifest_path = pbr_manifest_path.parent.parent / "overlay" / "pipeline.json"
        self.assertTrue(overlay_manifest_path.exists(), "Dedicated overlay pipeline.json must exist")

        overlay_manifest = json.loads(overlay_manifest_path.read_text())
        self.assertEqual(overlay_manifest["target"], "vulkan")
        self.assertEqual(len(overlay_manifest["entries"]), 2)

        depth_stencil = overlay_manifest.get("depth_stencil", {})
        self.assertFalse(depth_stencil.get("depth_test_enable"), "Overlay pipeline must have depth test disabled")
        self.assertFalse(depth_stencil.get("depth_write_enable"), "Overlay pipeline must have depth write disabled")
        self.assertEqual(depth_stencil.get("depth_compare_op"), "always", "Overlay depth compare op must be always")

        with patch("examples.minigfx.samples.model_viewer.render_offscreen") as mock_render:
            mock_render.return_value = RenderedImage(640, 480, b"\x00" * (640 * 480 * 4), "test_device")
            app.render_offscreen_frame()

            multi_manifest_path = pbr_manifest_path.parent.parent / "multi_manifest.json"
            self.assertTrue(multi_manifest_path.exists(), "multi_manifest.json must be generated")
            multi_manifest = json.loads(multi_manifest_path.read_text())

            self.assertIn("pbr_forward", multi_manifest["pipelines"])
            self.assertIn("overlay", multi_manifest["pipelines"])

            draws = multi_manifest["draws"]
            overlay_draws = [d for d in draws if d["pipeline"] == "overlay"]
            self.assertGreater(len(overlay_draws), 0, "Text quads must produce at least one overlay draw call")

            for od in overlay_draws:
                self.assertEqual(od["constant_size"], 48, "Overlay draw call must submit 48-byte OverlayParams constants")


if __name__ == "__main__":
    unittest.main()
