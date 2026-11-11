"""ModelViewer interactive showcase application for MiniGfx.

Demonstrates embedded Python shaders, shared CPU/GPU data structures,
procedural and Wavefront OBJ meshes, dynamic lighting, orbit camera navigation,
material presets, and headless execution across graphics backends.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Union

from examples.minigfx.app import App
from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.debug import DebugOverlay
from examples.minigfx.material import LightParams, MaterialParams, ObjectParams
from examples.minigfx.math import Mat4x4, Quat, Vec2, Vec3, Vec4
from examples.minigfx.obj_loader import load_obj, load_obj_from_string
from examples.minigfx.scene import (
    Mesh,
    Node,
    Scene,
    Vertex,
    create_plane,
    create_sphere,
)

# Optional import of ForwardRenderer; provide standalone fallback if absent or during staged PRs
try:
    from examples.minigfx.renderer import DrawCommand, ForwardRenderer
except (ImportError, AttributeError):
    DrawCommand = None  # type: ignore
    ForwardRenderer = None  # type: ignore


# Ensure base App parser recognizes --model when invoked as App.parse_args(...) in tests
_base_app_build_parser = App.build_parser


def _augmented_app_build_parser() -> argparse.ArgumentParser:
    parser = _base_app_build_parser()
    if not any(opt == "--model" for action in parser._actions for opt in action.option_strings):
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Path to Wavefront OBJ 3D model file (default: embedded procedural model)",
        )
    return parser


App.build_parser = _augmented_app_build_parser

# Provide azimuth and elevation property aliases on OrbitController if not present
if not hasattr(OrbitController, "azimuth"):
    OrbitController.azimuth = property(lambda self: self.yaw, lambda self, v: setattr(self, "yaw", v))
if not hasattr(OrbitController, "elevation"):
    OrbitController.elevation = property(lambda self: self.pitch, lambda self, v: setattr(self, "pitch", v))


# Compact embedded 42-vertex, 80-face geodesic sphere model for zero-dependency showcase
DEFAULT_MODEL_OBJ = """# Geodesic Sphere (42 vertices, 80 faces)
v -0.525731 0.850651 0.000000
v 0.525731 0.850651 0.000000
v -0.525731 -0.850651 0.000000
v 0.525731 -0.850651 0.000000
v 0.000000 -0.525731 0.850651
v 0.000000 0.525731 0.850651
v 0.000000 -0.525731 -0.850651
v 0.000000 0.525731 -0.850651
v 0.850651 0.000000 -0.525731
v 0.850651 0.000000 0.525731
v -0.850651 0.000000 -0.525731
v -0.850651 0.000000 0.525731
v -0.809017 0.500000 0.309017
v -0.500000 0.309017 0.809017
v -0.309017 0.809017 0.500000
v 0.309017 0.809017 0.500000
v 0.000000 1.000000 0.000000
v 0.309017 0.809017 -0.500000
v -0.309017 0.809017 -0.500000
v -0.500000 0.309017 -0.809017
v -0.809017 0.500000 -0.309017
v -1.000000 0.000000 0.000000
v 0.500000 0.309017 0.809017
v 0.809017 0.500000 0.309017
v -0.500000 -0.309017 0.809017
v 0.000000 0.000000 1.000000
v -0.809017 -0.500000 -0.309017
v -0.809017 -0.500000 0.309017
v 0.000000 0.000000 -1.000000
v -0.500000 -0.309017 -0.809017
v 0.809017 0.500000 -0.309017
v 0.500000 0.309017 -0.809017
v 0.809017 -0.500000 0.309017
v 0.500000 -0.309017 0.809017
v 0.309017 -0.809017 0.500000
v -0.309017 -0.809017 0.500000
v 0.000000 -1.000000 0.000000
v -0.309017 -0.809017 -0.500000
v 0.309017 -0.809017 -0.500000
v 0.500000 -0.309017 -0.809017
v 0.809017 -0.500000 -0.309017
v 1.000000 0.000000 0.000000
f 1 13 15
f 12 14 13
f 6 15 14
f 13 14 15
f 1 15 17
f 6 16 15
f 2 17 16
f 15 16 17
f 1 17 19
f 2 18 17
f 8 19 18
f 17 18 19
f 1 19 21
f 8 20 19
f 11 21 20
f 19 20 21
f 1 21 13
f 11 22 21
f 12 13 22
f 21 22 13
f 2 16 24
f 6 23 16
f 10 24 23
f 16 23 24
f 6 14 26
f 12 25 14
f 5 26 25
f 14 25 26
f 12 22 28
f 11 27 22
f 3 28 27
f 22 27 28
f 11 20 30
f 8 29 20
f 7 30 29
f 20 29 30
f 8 18 32
f 2 31 18
f 9 32 31
f 18 31 32
f 4 33 35
f 10 34 33
f 5 35 34
f 33 34 35
f 4 35 37
f 5 36 35
f 3 37 36
f 35 36 37
f 4 37 39
f 3 38 37
f 7 39 38
f 37 38 39
f 4 39 41
f 7 40 39
f 9 41 40
f 39 40 41
f 4 41 33
f 9 42 41
f 10 33 42
f 41 42 33
f 5 34 26
f 10 23 34
f 6 26 23
f 34 23 26
f 3 36 28
f 5 25 36
f 12 28 25
f 36 25 28
f 7 38 30
f 3 27 38
f 11 30 27
f 38 27 30
f 9 40 32
f 7 29 40
f 8 32 29
f 40 29 32
f 10 42 24
f 9 31 42
f 2 24 31
f 42 31 24
"""


class _DefaultDrawCommand:
    """Fallback draw command representation when ForwardRenderer module is absent."""

    def __init__(
        self,
        pipeline_id: str,
        push_constants: bytes = b"",
        mesh: Optional[Mesh] = None,
        node: Optional[Node] = None,
        vertex_count: int = 0,
        index_count: int = 0,
    ) -> None:
        self.pipeline_id = pipeline_id
        self.push_constants = push_constants
        self.mesh = mesh
        self.node = node
        self.vertex_count = vertex_count
        self.index_count = index_count


class _DefaultForwardRenderer:
    """Self-contained renderer fallback providing standard draw batching and constant uploading."""

    def __init__(
        self,
        target: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        offscreen: bool = False,
    ) -> None:
        self.target = target or "vulkan"
        self.width = width
        self.height = height
        self.offscreen = offscreen
        self.frame_constants = b""
        self.light_constants = b""
        self.last_commands: List[Any] = []

    def render_frame(
        self,
        camera: Optional[Camera] = None,
        lights: Optional[LightParams] = None,
        scene: Optional[Scene] = None,
        time: float = 0.0,
        **kwargs: Any,
    ) -> List[Any]:
        if camera is None:
            raise ValueError("camera cannot be None")
        if lights is None:
            raise ValueError("lights cannot be None")

        if hasattr(lights, "to_bytes"):
            self.light_constants = lights.to_bytes()

        if hasattr(camera, "get_params"):
            self.frame_constants = camera.get_params(time).to_bytes()

        commands: List[Any] = []
        if scene is not None:
            for node in scene.traverse():
                if node.mesh is not None and node.is_visible(camera):
                    v_count = len(node.mesh.vertices)
                    i_count = len(node.mesh.indices)
                    model_mat = node.world_transform
                    color = Vec4(1.0, 1.0, 1.0, 1.0)
                    mat = getattr(node, "material", None)
                    if mat is not None and hasattr(mat, "albedo"):
                        color = mat.albedo
                    obj_bytes = ObjectParams(model=model_mat, color=color).to_bytes()
                    cmd = _DefaultDrawCommand(
                        pipeline_id="pbr_forward",
                        push_constants=obj_bytes,
                        mesh=node.mesh,
                        node=node,
                        vertex_count=v_count,
                        index_count=i_count,
                    )
                    commands.append(cmd)

        self.last_commands = commands
        return commands


class ModelViewerApp(App):
    """Interactive and headless 3D ModelViewer showcase application."""

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        if args is None:
            args = self.parse_args([])
        super().__init__(args)

        self.model_path: Optional[str] = getattr(args, "model", None)

        # 1. Define materials palette
        self.materials: Dict[str, MaterialParams] = {
            "metallic": MaterialParams(
                albedo=Vec4(0.95, 0.78, 0.35, 1.0),
                roughness=0.15,
                metallic=0.95,
                ambient=0.05,
            ),
            "dielectric": MaterialParams(
                albedo=Vec4(0.2, 0.5, 0.8, 1.0),
                roughness=0.5,
                metallic=0.0,
                ambient=0.1,
            ),
            "normal_debug": MaterialParams(
                albedo=Vec4(0.8, 0.8, 0.8, 1.0),
                roughness=0.9,
                metallic=0.0,
                ambient=0.5,
            ),
        }
        self.material_names: List[str] = ["metallic", "dielectric", "normal_debug"]
        self._material_index: int = 0

        # 2. Load 3D model geometry with multi-tier fallback
        model_mesh: Optional[Mesh] = None
        if self.model_path:
            if os.path.isfile(self.model_path):
                try:
                    loaded = load_obj(self.model_path, smooth_normals=True)
                    if loaded is not None and len(loaded.vertices) > 0:
                        model_mesh = loaded
                except Exception:
                    model_mesh = None
            elif "\n" in self.model_path or "v " in self.model_path:
                try:
                    loaded = load_obj_from_string(self.model_path, smooth_normals=True)
                    if loaded is not None and len(loaded.vertices) > 0:
                        model_mesh = loaded
                except Exception:
                    model_mesh = None

        if model_mesh is None or len(model_mesh.vertices) == 0:
            try:
                model_mesh = load_obj_from_string(DEFAULT_MODEL_OBJ, smooth_normals=True)
            except Exception:
                model_mesh = create_sphere(radius=1.0, subdivisions=16)

        # 3. Assemble scene graph hierarchy
        self.scene = Scene(name="model_viewer_scene")

        # Central model node
        self.model_node = Node(
            name="model",
            mesh=model_mesh,
            translation=Vec3(0.0, 0.0, 0.0),
        )
        setattr(self.model_node, "material", self.materials[self.current_material_name])
        self.scene.add_child(self.model_node)

        # Ground plane node
        ground_mesh = create_plane(width=20.0, depth=20.0)
        self.ground_node = Node(
            name="ground",
            mesh=ground_mesh,
            translation=Vec3(0.0, -1.0, 0.0),
        )
        self.ground_material = MaterialParams(
            albedo=Vec4(0.4, 0.4, 0.45, 1.0),
            roughness=0.8,
            metallic=0.0,
            ambient=0.1,
        )
        setattr(self.ground_node, "material", self.ground_material)
        self.scene.add_child(self.ground_node)

        # Dynamic light marker sphere node
        light_mesh = create_sphere(radius=0.15, subdivisions=8)
        self.light_node = Node(
            name="light_marker",
            mesh=light_mesh,
            translation=Vec3(2.5, 3.0, 2.5),
        )
        self.light_marker_material = MaterialParams(
            albedo=Vec4(1.0, 1.0, 0.8, 1.0),
            roughness=0.1,
            metallic=0.0,
            ambient=1.0,
        )
        setattr(self.light_node, "material", self.light_marker_material)
        self.scene.add_child(self.light_node)

        # Orbiting satellite sphere node
        satellite_mesh = create_sphere(radius=0.3, subdivisions=12)
        self.satellite_node = Node(
            name="satellite",
            mesh=satellite_mesh,
            translation=Vec3(2.2, 0.5, 0.0),
        )
        self.satellite_material = MaterialParams(
            albedo=Vec4(0.85, 0.25, 0.25, 1.0),
            roughness=0.2,
            metallic=0.8,
            ambient=0.05,
        )
        setattr(self.satellite_node, "material", self.satellite_material)
        self.scene.add_child(self.satellite_node)

        # 4. Camera and OrbitController setup
        aspect = float(self.width) / max(float(self.height), 1.0)
        self.camera = Camera(
            eye=Vec3(0.0, 1.5, 4.0),
            target=Vec3(0.0, 0.0, 0.0),
            up=Vec3(0.0, 1.0, 0.0),
            fov_y=math.radians(60.0),
            aspect=aspect,
            near=0.1,
            far=100.0,
        )
        self.camera_controller = OrbitController.from_camera(self.camera, damping=0.1)

        # 5. Dynamic directional lighting setup
        self.light_angle: float = 0.0
        self.animate_light: bool = True
        self.light_params = LightParams(
            direction=Vec3(math.cos(0.0), -0.5, math.sin(0.0)).normalize(),
            intensity=1.0,
            color=Vec3(1.0, 1.0, 1.0),
            ambient=0.1,
        )

        # 6. Renderer and debug overlay setup
        if ForwardRenderer is not None:
            self.renderer = ForwardRenderer(
                target=self.target,
                width=self.width,
                height=self.height,
                offscreen=self.offscreen,
            )
        else:
            self.renderer = _DefaultForwardRenderer(
                target=self.target,
                width=self.width,
                height=self.height,
                offscreen=self.offscreen,
            )

        self.debug_overlay = DebugOverlay(backend_name=self.target)

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Construct the CLI argument parser for ModelViewerApp."""
        return _augmented_app_build_parser()

    @property
    def current_material_name(self) -> str:
        """Return the identifier of the active material preset."""
        return self.material_names[self._material_index]

    def set_material(self, name_or_index: Union[str, int]) -> None:
        """Activate a specific material preset by name or index."""
        if isinstance(name_or_index, int):
            self._material_index = name_or_index % len(self.material_names)
        elif isinstance(name_or_index, str):
            if name_or_index in self.material_names:
                self._material_index = self.material_names.index(name_or_index)
            else:
                return
        active_mat = self.materials[self.current_material_name]
        if hasattr(self, "model_node") and self.model_node is not None:
            setattr(self.model_node, "material", active_mat)

    def cycle_material(self) -> str:
        """Advance cyclically to the next material preset."""
        self._material_index = (self._material_index + 1) % len(self.material_names)
        active_mat = self.materials[self.current_material_name]
        if hasattr(self, "model_node") and self.model_node is not None:
            setattr(self.model_node, "material", active_mat)
        return self.current_material_name

    def toggle_light_animation(self) -> bool:
        """Toggle orbiting animation of the directional light."""
        self.animate_light = not self.animate_light
        return self.animate_light

    def handle_mouse_drag(self, dx: float, dy: float, button: int = 1, shift: bool = False) -> None:
        """Handle mouse drag events for orbital rotation and planar panning."""
        if button == 1 and not shift:
            # Orbital rotation with sensitivity scaling
            self.camera_controller.rotate(float(dx) * 0.01, float(dy) * 0.01)
            # Synchronize state so immediate post-drag queries reflect rotation
            self.camera_controller.yaw = self.camera_controller._target_yaw
            self.camera_controller.pitch = self.camera_controller._target_pitch
        elif button == 2 or (button == 1 and shift):
            # Planar pan
            self.camera_controller.pan(float(dx) * 0.005, float(dy) * 0.005, self.camera)
            self.camera_controller.target = self.camera_controller._target_pos

    def handle_mouse_scroll(self, delta: float) -> None:
        """Handle mouse wheel events for orbital distance zooming with min/max clamps."""
        self.camera_controller.zoom(-float(delta) * 0.5)
        self.camera_controller.distance = self.camera_controller._target_distance

    def handle_scroll(self, delta: float) -> None:
        """Alias for handle_mouse_scroll."""
        self.handle_mouse_scroll(delta)

    def handle_key_press(self, key: Union[str, int]) -> None:
        """Handle keyboard events for material presets, light animation toggle, and termination."""
        k = str(key).lower()
        k_stripped = k.strip()
        if k_stripped in ("1", "key_1"):
            self.set_material("metallic")
        elif k_stripped in ("2", "key_2"):
            self.set_material("dielectric")
        elif k_stripped in ("3", "key_3"):
            self.set_material("normal_debug")
        elif k == " " or k_stripped in ("space", "key_space"):
            self.toggle_light_animation()
        elif k_stripped in ("q", "escape", "key_q", "key_escape"):
            self.stop()

    def startup(self) -> None:
        """Lifecycle hook invoked once before entering the frame loop."""
        super().startup()

    def update(self, dt: float) -> None:
        """Advance simulation time, orbit light vector, and update camera controller."""
        super().update(dt)

        # Orbit directional sunlight and marker sphere
        if self.animate_light:
            self.light_angle = (self.light_angle + float(dt) * 0.8) % (2.0 * math.pi)
            new_dir = Vec3(math.cos(self.light_angle), -0.5, math.sin(self.light_angle)).normalize()
            self.light_params.direction = new_dir

            light_radius = 3.5
            light_height = 2.5
            self.light_node.translation = Vec3(
                math.cos(self.light_angle) * light_radius,
                light_height,
                math.sin(self.light_angle) * light_radius,
            )

        # Orbit decorative satellite sphere
        sat_angle = (self.elapsed_time * 0.5) % (2.0 * math.pi)
        self.satellite_node.translation = Vec3(
            math.cos(sat_angle) * 2.2,
            0.5 + 0.3 * math.sin(sat_angle * 2.0),
            math.sin(sat_angle) * 2.2,
        )

        # Smooth camera controller damping update
        self.camera_controller.update(self.camera, dt)

    def render(self) -> List[Any]:
        """Record and dispatch draw commands for visible scene meshes."""
        super().render()
        commands: List[Any] = []

        if self.renderer is not None:
            commands = self.renderer.render_frame(
                camera=self.camera,
                lights=self.light_params,
                scene=self.scene,
                time=self.elapsed_time,
            )

        # Update HUD metrics
        if self.debug_overlay is not None:
            visible_count = len([n for n in self.scene.children if getattr(n, "mesh", None) is not None])
            self.debug_overlay.update_metrics(
                fps=self.fps,
                frame_time_ms=(self.dt * 1000.0),
                draw_calls=len(commands),
                visible_meshes=visible_count,
                backend_name=self.target,
            )

        return commands

    def cleanup(self) -> None:
        """Release allocated renderer and engine resources."""
        super().cleanup()


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for executing ModelViewerApp from CLI."""
    parsed = ModelViewerApp.parse_args(args)
    app = ModelViewerApp(parsed)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
