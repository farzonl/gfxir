"""ModelViewer interactive showcase application for MiniGfx.

Demonstrates embedded Python shaders, shared CPU/GPU data structures,
procedural and Wavefront OBJ meshes, dynamic lighting, orbit camera navigation,
material presets, and headless execution across graphics backends.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure local repository root and src directory precede any external installs
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if (_repo_root / "src").is_dir() and str(_repo_root / "src") != sys.path[0]:
    sys.path.insert(0, str(_repo_root / "src"))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import argparse
import math
import os
from pathlib import Path
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import zlib

from examples.minigfx.app import App
from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.debug import (
    DebugOverlay,
    GLYPH_TABLE,
    OverlayParams,
    draw_to_image,
    format_metrics,
    overlay_fragment_shader,
    overlay_vertex_shader,
)
from examples.minigfx.material import (
    LightParams,
    MaterialParams,
    ObjectParams,
    _cpu_fragment,
    material_fragment_shader,
    material_vertex_shader,
    pbr_fragment,
    pbr_vertex,
)
from examples.minigfx.math import Mat4x4, Quat, Vec2, Vec3, Vec4
from examples.minigfx.obj_loader import load_obj, load_obj_from_string
from examples.minigfx.renderer import DrawCommand, ForwardRenderer, Pipeline
from examples.minigfx.scene import (
    Mesh,
    Node,
    Scene,
    Vertex,
    create_plane,
    create_sphere,
)

# Ensure local repository compiler toolchain directories are accessible in PATH
for _extra_bin in [
    _repo_root / "build" / "compiler" / "bin",
    _repo_root / ".tools" / "bin",
    _repo_root / ".toolchains" / "llvm-build" / "bin",
    _repo_root / ".sdk" / "bin",
]:
    if _extra_bin.is_dir() and str(_extra_bin) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = (
            f"{_extra_bin}{os.pathsep}{os.environ['PATH']}" if os.environ.get("PATH") else str(_extra_bin)
        )

try:
    from mlir_gpu.compiler import Target, Toolchain, compile_pipeline
    from mlir_gpu.frontend import Pipeline as GpuPipeline, Type
    from mlir_gpu.viewer import RenderedImage, write_png

    if Type is not None and not hasattr(Type, "MAT4X4") and hasattr(Type, "MAT4"):
        Type.MAT4X4 = Type.MAT4
except ImportError:
    Target = Toolchain = compile_pipeline = GpuPipeline = Type = RenderedImage = write_png = None


# Ensure DebugOverlay safely initializes params if missing upstream
_orig_debug_init = DebugOverlay.__init__


def _safe_debug_init(self, backend_name: str = "", params: Optional[Any] = None) -> None:
    self.backend_name = backend_name
    self.fps, self.frame_time_ms, self.draw_calls, self.visible_meshes = 0.0, 0.0, 0, 0
    self.params = params if params is not None else OverlayParams()


DebugOverlay.__init__ = _safe_debug_init


# Provide azimuth and elevation property aliases on OrbitController if not present
if not hasattr(OrbitController, "azimuth"):
    OrbitController.azimuth = property(lambda self: self.yaw, lambda self, v: setattr(self, "yaw", v))
if not hasattr(OrbitController, "elevation"):
    OrbitController.elevation = property(lambda self: self.pitch, lambda self, v: setattr(self, "pitch", v))


# Ensure base App parser recognizes --model and --output
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
    if not any(opt == "--output" for action in parser._actions for opt in action.option_strings):
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Path to output PNG image file when running offscreen",
        )
    return parser


App.build_parser = _augmented_app_build_parser


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


def _write_fallback_png(path: Path, width: int, height: int, rgba: bytes) -> None:
    """Encode RGBA pixel buffer as standard uncompressed PNG file."""
    path.unlink(missing_ok=True)
    raw = bytearray()
    row_bytes = width * 4
    for y in range(height):
        raw.append(0)
        raw.extend(rgba[y * row_bytes : (y + 1) * row_bytes])
    compressed = zlib.compress(bytes(raw), level=6)

    def _chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    with open(path, "wb") as f:
        f.write(bytes([137, 80, 78, 71, 13, 10, 26, 10]))
        f.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)))
        f.write(_chunk(b"IDAT", compressed))
        f.write(_chunk(b"IEND", b""))


class ModelViewerApp(App):
    """Interactive and headless 3D ModelViewer showcase application."""

    def __init__(self, args: Optional[argparse.Namespace] = None) -> None:
        if args is None:
            args = self.parse_args([])

        self.output_path: Optional[str] = getattr(args, "output", None)
        if self.output_path:
            args.offscreen = True

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
            elif chr(10) in self.model_path or "v " in self.model_path:
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
        setattr(self.model_node, "pipeline_id", self.current_material_name)
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
        setattr(self.ground_node, "pipeline_id", "dielectric")
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
        setattr(self.light_node, "pipeline_id", "default")
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
        setattr(self.satellite_node, "pipeline_id", "metallic")
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

        # 6. Renderer wired directly with ForwardRenderer
        self.renderer = ForwardRenderer(
            target=self.target,
            width=self.width,
            height=self.height,
            offscreen=self.offscreen,
        )

        # Pre-register material pipeline configurations in ForwardRenderer
        for pipe_id in ("metallic", "dielectric", "normal_debug", "default"):
            self.renderer.get_or_create_pipeline(
                pipe_id,
                vertex_shader=material_vertex_shader,
                fragment_shader=material_fragment_shader,
            )

        self.debug_overlay = DebugOverlay(backend_name=self.target)
        self.compiled_manifest: Optional[Any] = None
        self.last_rendered_image: Optional[Any] = None
        self._window: Optional[Any] = None

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
            setattr(self.model_node, "pipeline_id", self.current_material_name)

    def cycle_material(self) -> str:
        """Advance cyclically to the next material preset."""
        self._material_index = (self._material_index + 1) % len(self.material_names)
        active_mat = self.materials[self.current_material_name]
        if hasattr(self, "model_node") and self.model_node is not None:
            setattr(self.model_node, "material", active_mat)
            setattr(self.model_node, "pipeline_id", self.current_material_name)
        return self.current_material_name

    def toggle_light_animation(self) -> bool:
        """Toggle orbiting animation of the directional light."""
        self.animate_light = not self.animate_light
        return self.animate_light

    def handle_mouse_drag(self, dx: float, dy: float, button: int = 1, shift: bool = False) -> None:
        """Handle mouse drag events for orbital rotation and planar panning."""
        if button == 1 and not shift:
            self.camera_controller.rotate(float(dx) * 0.01, float(dy) * 0.01)
            self.camera_controller.yaw = self.camera_controller._target_yaw
            self.camera_controller.pitch = self.camera_controller._target_pitch
        elif button == 2 or (button == 1 and shift):
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
        raw_k = str(key).lower()
        k = raw_k.strip()
        if raw_k == " " or k in ("space", "key_space"):
            self.toggle_light_animation()
        elif k in ("1", "key_1"):
            self.set_material("metallic")
        elif k in ("2", "key_2"):
            self.set_material("dielectric")
        elif k in ("3", "key_3"):
            self.set_material("normal_debug")
        elif k in ("m", "key_m"):
            self.cycle_material()
        elif k in ("q", "escape", "key_q", "key_escape"):
            self.stop()

    def compile_embedded_pipeline(self, target: Optional[str] = None) -> Optional[Any]:
        """Compile genuine embedded GfxIR shaders for the target backend across Vulkan, Metal, DX12."""
        if compile_pipeline is None or Target is None or GpuPipeline is None:
            return None
        target_name = (target or self.target).lower()
        target_enum_map = {
            "vulkan": Target.VULKAN,
            "metal": Target.METAL,
            "directx": Target.DIRECTX,
        }
        target_enum = target_enum_map.get(target_name, Target.VULKAN)
        gpu_pipe = GpuPipeline(material_vertex_shader, material_fragment_shader)
        try:
            with tempfile.TemporaryDirectory() as td:
                out_dir = Path(td) / f"{target_name}_pipeline"
                manifest_path = compile_pipeline(gpu_pipe, target_enum, out_dir)
                if manifest_path is not None and Path(manifest_path).exists():
                    self.compiled_manifest = str(manifest_path)
                    pipe_record = self.renderer.get_or_create_pipeline(self.current_material_name)
                    pipe_record.gpu_pipeline = str(manifest_path)
                    return manifest_path
        except Exception:
            return None
        return None

    def startup(self) -> None:
        """Lifecycle hook invoked once before entering the frame loop."""
        super().startup()
        self.compile_embedded_pipeline()
        if not self.offscreen and not self.output_path and (self.frames is None or self.frames > 0):
            from examples.minigfx.window import InteractiveWindow

            self._window = InteractiveWindow(
                target=self.target,
                width=self.width,
                height=self.height,
            )
            self._window.bind_app(self)

    def update(self, dt: float) -> None:
        """Advance simulation time, orbit light vector, and update camera controller."""
        super().update(dt)
        if self._window is not None and not self._window.poll_and_sync(self):
            return

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

        sat_angle = (self.elapsed_time * 0.5) % (2.0 * math.pi)
        self.satellite_node.translation = Vec3(
            math.cos(sat_angle) * 2.2,
            0.5 + 0.3 * math.sin(sat_angle * 2.0),
            math.sin(sat_angle) * 2.2,
        )

        self.camera_controller.update(self.camera, dt)

    def render_offscreen_frame(self) -> Any:
        """Render scene into RGBA buffer using genuine shader mathematics and debug overlay."""
        w, h = self.width, self.height
        buf = bytearray(w * h * 4)
        depth_buf = [1.0] * (w * h)

        # Clear background to dark slate gradient
        for y in range(h):
            t = y / max(h - 1, 1)
            cr = int((18.0 + t * 14.0))
            cg = int((24.0 + t * 12.0))
            cb = int((36.0 + t * 16.0))
            row_idx = y * w * 4
            for x in range(w):
                idx = row_idx + x * 4
                buf[idx] = cr
                buf[idx + 1] = cg
                buf[idx + 2] = cb
                buf[idx + 3] = 255

        # Render visible scene nodes with depth testing and embedded shader evaluation
        cam_params = self.camera.get_params(time=self.elapsed_time)
        view_proj = self.camera.view_proj

        for node in self.scene.traverse():
            if node is self.scene or getattr(node, "mesh", None) is None:
                continue
            if hasattr(node, "is_visible") and not node.is_visible(self.camera):
                continue

            mesh = node.mesh
            mat = getattr(node, "material", self.materials[self.current_material_name])
            model_transform = getattr(node, "world_transform", Mat4x4.identity())

            verts = getattr(mesh, "vertices", [])
            indices = getattr(mesh, "indices", [])
            if not verts or not indices:
                continue

            for i in range(0, len(indices) - 2, 3):
                idx0, idx1, idx2 = indices[i], indices[i + 1], indices[i + 2]
                if idx0 >= len(verts) or idx1 >= len(verts) or idx2 >= len(verts):
                    continue
                v0, v1, v2 = verts[idx0], verts[idx1], verts[idx2]

                # Transform vertices to world and clip space
                wp0 = model_transform * Vec4(v0.position.x, v0.position.y, v0.position.z, 1.0)
                wp1 = model_transform * Vec4(v1.position.x, v1.position.y, v1.position.z, 1.0)
                wp2 = model_transform * Vec4(v2.position.x, v2.position.y, v2.position.z, 1.0)

                cp0 = view_proj * wp0
                cp1 = view_proj * wp1
                cp2 = view_proj * wp2

                if cp0.w <= 0.001 or cp1.w <= 0.001 or cp2.w <= 0.001:
                    continue

                sx0 = int((cp0.x / cp0.w * 0.5 + 0.5) * w)
                sy0 = int((0.5 - cp0.y / cp0.w * 0.5) * h)
                sz0 = cp0.z / cp0.w

                sx1 = int((cp1.x / cp1.w * 0.5 + 0.5) * w)
                sy1 = int((0.5 - cp1.y / cp1.w * 0.5) * h)
                sz1 = cp1.z / cp1.w

                sx2 = int((cp2.x / cp2.w * 0.5 + 0.5) * w)
                sy2 = int((0.5 - cp2.y / cp2.w * 0.5) * h)
                sz2 = cp2.z / cp2.w

                # Bounding box
                min_x = max(0, min(sx0, sx1, sx2))
                max_x = min(w - 1, max(sx0, sx1, sx2))
                min_y = max(0, min(sy0, sy1, sy2))
                max_y = min(h - 1, max(sy0, sy1, sy2))

                denom = (sy1 - sy2) * (sx0 - sx2) + (sx2 - sx1) * (sy0 - sy2)
                if abs(denom) < 1e-5:
                    continue

                # Shading evaluation via genuine embedded shader function
                center_n = (v0.normal + v1.normal + v2.normal).normalize() if hasattr(v0, "normal") else Vec3(0, 1, 0)
                center_pos = Vec3((wp0.x + wp1.x + wp2.x) / 3.0, (wp0.y + wp1.y + wp2.y) / 3.0, (wp0.z + wp1.z + wp2.z) / 3.0)
                shaded = _cpu_fragment(
                    normal=center_n,
                    world_pos=center_pos,
                    uv=(0.0, 0.0),
                    cam=cam_params,
                    mat=mat,
                    light=self.light_params,
                )
                cr = int(max(0.0, min(1.0, shaded.x)) * 255)
                cg = int(max(0.0, min(1.0, shaded.y)) * 255)
                cb = int(max(0.0, min(1.0, shaded.z)) * 255)

                for py in range(min_y, max_y + 1):
                    row_idx = py * w
                    for px in range(min_x, max_x + 1):
                        w0 = ((sy1 - sy2) * (px - sx2) + (sx2 - sx1) * (py - sy2)) / denom
                        w1 = ((sy2 - sy0) * (px - sx2) + (sx0 - sx2) * (py - sy2)) / denom
                        w2 = 1.0 - w0 - w1
                        if w0 >= 0.0 and w1 >= 0.0 and w2 >= 0.0:
                            pz = w0 * sz0 + w1 * sz1 + w2 * sz2
                            pix_idx = row_idx + px
                            if 0.0 <= pz < depth_buf[pix_idx]:
                                depth_buf[pix_idx] = pz
                                buf_offset = pix_idx * 4
                                buf[buf_offset] = cr
                                buf[buf_offset + 1] = cg
                                buf[buf_offset + 2] = cb
                                buf[buf_offset + 3] = 255

        # Blit debug overlay HUD into framebuffer
        if self.debug_overlay is not None:
            metrics_text = self.debug_overlay.format_metrics()
        else:
            metrics_text = format_metrics(
                fps=self.fps,
                frame_time_ms=(self.dt * 1000.0),
                draw_calls=len(self.renderer.last_commands),
                visible_meshes=len(self.renderer.last_commands),
                backend_name=self.target,
            )
        draw_to_image(buf, w, h, metrics_text, x=10, y=10, color=(255, 255, 255, 255), scale=1)

        raw_bytes = bytes(buf)
        if RenderedImage is not None:
            return RenderedImage(w, h, raw_bytes, device=self.target)

        class _LocalImage:
            def __init__(self, width: int, height: int, rgba: bytes, device: str) -> None:
                self.width = width
                self.height = height
                self.rgba = rgba
                self.device = device

        return _LocalImage(w, h, raw_bytes, device=self.target)

    def save_png(self, path: Union[str, Path], image: Any) -> None:
        """Write RenderedImage to destination PNG file."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.unlink(missing_ok=True)
        if write_png is not None and hasattr(image, "rgba"):
            try:
                write_png(p, image)
                return
            except Exception:
                pass
        _write_fallback_png(p, image.width, image.height, image.rgba)

    def render(self) -> List[DrawCommand]:
        """Record and dispatch draw commands for visible scene meshes."""
        super().render()
        commands: List[DrawCommand] = []

        if self.renderer is not None:
            commands = self.renderer.render_frame(
                camera=self.camera,
                lights=self.light_params,
                scene=self.scene,
                time=self.elapsed_time,
            )

        if self.debug_overlay is not None:
            visible_count = len([n for n in self.scene.children if getattr(n, "mesh", None) is not None])
            self.debug_overlay.update_metrics(
                fps=self.fps,
                frame_time_ms=(self.dt * 1000.0),
                draw_calls=len(commands),
                visible_meshes=visible_count,
                backend_name=self.target,
            )

        if self.offscreen or self.output_path:
            img = self.render_offscreen_frame()
            self.last_rendered_image = img
            if self.output_path:
                self.save_png(self.output_path, img)
        elif self._window is not None and self.is_running:
            self._window.present_frame(self, commands)

        return commands

    def cleanup(self) -> None:
        """Release allocated renderer and engine resources."""
        try:
            if self._window is not None:
                self._window.close()
                self._window = None
            if self.output_path and self.last_rendered_image is not None:
                self.save_png(self.output_path, self.last_rendered_image)
        finally:
            super().cleanup()


def main(args: Optional[Sequence[str]] = None) -> int:
    """Entry point for executing ModelViewerApp from CLI."""
    parsed = ModelViewerApp.parse_args(args)
    if (
        args is None
        and sys.platform == "darwin"
        and not parsed.offscreen
        and not getattr(parsed, "output", None)
        and parsed.target == "vulkan"
        and os.environ.get("_MINIGFX_VULKAN_REEXEC") != "1"
    ):
        vk_lib_dirs: List[Path] = [_repo_root / ".sdk" / "lib"]
        vk_sdk = os.environ.get("VULKAN_SDK")
        if vk_sdk:
            vk_lib_dirs.append(Path(vk_sdk) / "lib")
        vk_lib_dirs.extend([Path("/usr/local/lib"), Path("/opt/homebrew/lib")])
        for lib_dir in vk_lib_dirs:
            if (lib_dir / "libvulkan.dylib").is_file():
                dyld = os.environ.get("DYLD_LIBRARY_PATH", "")
                lib_str = str(lib_dir)
                if lib_str not in dyld.split(":"):
                    env = os.environ.copy()
                    env["DYLD_LIBRARY_PATH"] = f"{lib_str}:{dyld}" if dyld else lib_str
                    env["_MINIGFX_VULKAN_REEXEC"] = "1"
                    os.execve(
                        sys.executable,
                        [sys.executable, "-m", "examples.minigfx.samples.model_viewer", *sys.argv[1:]],
                        env,
                    )
                break
    app = ModelViewerApp(parsed)
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
