"""ModelViewer interactive showcase application for MiniGfx.

Demonstrates embedded Python shaders, shared CPU/GPU data structures,
procedural and Wavefront OBJ meshes, dynamic lighting, orbit camera navigation,
material presets, and genuine GPU offscreen execution across graphics backends.
"""


import argparse
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Union

# Ensure local repository root and src directory precede any external installs
_repo_root = Path(__file__).resolve().parent.parent.parent.parent
if (_repo_root / "src").is_dir() and str(_repo_root / "src") != sys.path[0]:
    sys.path.insert(0, str(_repo_root / "src"))
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from examples.minigfx.app import App
from examples.minigfx.camera import Camera, CameraParams, OrbitController
from examples.minigfx.debug import (
    DebugOverlay,
    OverlayParams,
    generate_text_quads,
)
from examples.minigfx.material import (
    LightParams,
    Material,
    MaterialParams,
    ObjectParams,
    Sampler,
    Texture,
    material_fragment_shader,
    material_vertex_shader,
    linear_pbr_fragment,
    linear_textured_pbr_fragment,
    textured_pbr_fragment,
)
from examples.minigfx.math import Mat4x4, Vec2, Vec3, Vec4
from examples.minigfx.obj_loader import load_obj, load_obj_from_string
from examples.minigfx.renderer import DrawCommand, ForwardRenderer
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
    _repo_root / "build" / "native",
    _repo_root / "build" / "native" / "Release",
    _repo_root / ".tools" / "bin",
    _repo_root / ".toolchains" / "llvm-build" / "bin",
    _repo_root / ".sdk" / "bin",
    _repo_root / ".sdk" / "bin" / "x64",
]:
    if _extra_bin.is_dir() and str(_extra_bin) not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = (
            f"{_extra_bin}{os.pathsep}{os.environ['PATH']}" if os.environ.get("PATH") else str(_extra_bin)
        )

from mlir_gpu.compiler import (
    MULTI_DRAW_SCHEMA_VERSION,
    RASTER_CONVENTION,
    Target,
    TargetProfile,
    compile_pipeline,
)
from mlir_gpu.frontend import (
    DepthStencil,
    Pipeline as GpuPipeline,
    Sampler as GpuSampler,
    Texture2D,
    Type,
    Vertex as GpuVertex,
    fragment,
    select,
    vec4,
    vertex,
)
from mlir_gpu.parameters import struct as gfx_struct
from mlir_gpu.viewer import (
    DrawCall,
    RenderedImage,
    load_program,
    render_offscreen,
    write_png,
    with_post_process,
)
from examples.minigfx.hdr import display_to_scene, make_post_pipeline, validate_exposure


@gfx_struct(layout="vertex")
class ModelVertex:
    """Standard 32-byte interleaved vertex layout for 3D meshes."""

    position: Type.VEC3
    normal: Type.VEC3
    uv: Type.VEC2


@vertex
def model_vertex_shader(vert: ModelVertex, cam: CameraParams, obj: ObjectParams) -> GpuVertex:
    """Embedded vertex shader executing genuine GPU transformation."""
    world_pos = obj.model * vec4(vert.position[0], vert.position[1], vert.position[2], 1.0)
    clip_pos = cam.view_proj * world_pos
    return GpuVertex(
        clip_pos,
        varyings={
            "normal": vert.normal,
            "world_pos": world_pos.xyz,
            "uv": vert.uv,
        },
    )


@vertex
def overlay_vertex_shader(vert: ModelVertex, params: OverlayParams) -> GpuVertex:
    """Embedded overlay vertex shader passing 2D NDC coordinates to clip space."""
    clip_pos = vec4(vert.position[0], vert.position[1], 0.0, 1.0)
    return GpuVertex(
        clip_pos,
        varyings={
            "uv": vert.uv,
            "clip_x": clip_pos.x,
        },
    )


@fragment
def overlay_fragment_shader(uv: Type.VEC2, params: OverlayParams) -> vec4:
    """Embedded overlay fragment shader with procedural glyph cell shaping."""
    edge = select((uv.x > 0.08) & (uv.x < 0.92) & (uv.y > 0.08) & (uv.y < 0.92), 1.0, 0.0)
    return select(edge > 0.5, params.text_color, params.bg_color)


def make_hdr_overlay_shader(exposure, comparison=False):
    overlay_fragment = overlay_fragment_shader.function

    @fragment
    def overlay_hdr_fragment(uv: Type.VEC2, clip_x: Type.F32, params: OverlayParams):
        color = overlay_fragment(uv, params)
        hdr_color = display_to_scene(color, exposure)
        if comparison:
            return select(clip_x < 0.0, color, hdr_color)
        return hdr_color

    return overlay_hdr_fragment


def make_comparison_fragment():
    ldr_fragment = material_fragment_shader.function
    hdr_fragment = linear_pbr_fragment.function

    @fragment
    def comparison_fragment(normal: Type.VEC3, world_pos: Type.VEC3, uv: Type.VEC2,
                            cam: CameraParams, mat: MaterialParams, light: LightParams):
        clip_pos = cam.view_proj * vec4(world_pos, 1.0)
        ldr = ldr_fragment(normal, world_pos, uv, cam, mat, light)
        hdr = hdr_fragment(normal, world_pos, uv, cam, mat, light)
        return select(clip_pos.x < 0.0, ldr, hdr)

    return comparison_fragment


def make_textured_comparison_fragment():
    ldr_fragment = textured_pbr_fragment.function
    hdr_fragment = linear_textured_pbr_fragment.function

    @fragment
    def comparison_fragment(normal: Type.VEC3, world_pos: Type.VEC3, uv: Type.VEC2,
                            albedo_tex: Texture2D, albedo_sampler: GpuSampler,
                            cam: CameraParams, mat: MaterialParams, light: LightParams):
        clip_pos = cam.view_proj * vec4(world_pos, 1.0)
        ldr = ldr_fragment(normal, world_pos, uv, albedo_tex, albedo_sampler, cam, mat, light)
        hdr = hdr_fragment(normal, world_pos, uv, albedo_tex, albedo_sampler, cam, mat, light)
        return select(clip_pos.x < 0.0, ldr, hdr)

    return comparison_fragment


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
        self.model_paths: List[str] = list(getattr(args, "models", None) or (
            [self.model_path] if self.model_path is not None else []
        ))
        self._model_index: int = 0
        self._pipeline_dir: Optional[tempfile.TemporaryDirectory] = None

        # 1. Define materials palette
        self.materials: Dict[str, Union[MaterialParams, Material]] = {
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
        self.albedo_texture: Optional[Texture] = None
        texture_path = getattr(args, "albedo_texture", None)
        if texture_path is not None:
            try:
                from PIL import Image
            except ImportError as error:
                raise RuntimeError("Image textures require Pillow; install mlir-gpu[viewer] or Pillow") from error
            with Image.open(texture_path) as image:
                rgba = image.convert("RGBA").transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                self.albedo_texture = Texture.from_rgba8(rgba.width, rgba.height, rgba.tobytes())
            sampler = Sampler(
                mag_filter=getattr(args, "texture_filter", "linear"),
                min_filter=getattr(args, "texture_filter", "linear"),
                address_mode_u=getattr(args, "texture_wrap", "repeat"),
                address_mode_v=getattr(args, "texture_wrap", "repeat"),
            )
            for name in ("metallic", "dielectric"):
                params = self.materials[name]
                self.materials[name] = Material(
                    albedo=Vec4(1.0, 1.0, 1.0, 1.0),
                    roughness=params.roughness, metallic=params.metallic, ambient=params.ambient,
                    albedo_texture=self.albedo_texture, sampler=sampler,
                )

        # 2. Load 3D model geometry
        model_mesh: Optional[Mesh] = None
        if self.model_paths:
            self._model_meshes = [self._load_model(path) for path in self.model_paths]
            self.model_path = self.model_paths[0]
            model_mesh = self._model_meshes[0]
        else:
            try:
                model_mesh = load_obj_from_string(DEFAULT_MODEL_OBJ, smooth_normals=True)
            except (ValueError, OSError):
                model_mesh = create_sphere(radius=1.0, subdivisions=16)
            self._model_meshes = [model_mesh]

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
            hdr=getattr(args, "hdr", False) or getattr(args, "hdr_comparison", False),
            exposure=getattr(args, "exposure", 1.0),
            comparison=getattr(args, "hdr_comparison", False),
        )

        # Pre-register material pipeline configurations in ForwardRenderer
        for pipe_id in ("metallic", "dielectric", "normal_debug", "default"):
            self.renderer.get_or_create_pipeline(
                pipe_id,
                vertex_shader=material_vertex_shader,
                fragment_shader=material_fragment_shader,
            )

        self.debug_overlay = DebugOverlay(backend_name=self.target)
        self.compiled_manifest: Optional[str] = None
        self.last_rendered_image: Optional[RenderedImage] = None
        self._window: Optional[Any] = None

    @classmethod
    def build_parser(cls) -> argparse.ArgumentParser:
        """Construct the CLI argument parser for ModelViewerApp."""
        parser = super().build_parser()
        parser.description = "ModelViewer interactive showcase application for MiniGfx"
        models = parser.add_mutually_exclusive_group()
        models.add_argument(
            "--model",
            type=str,
            default=None,
            help="Path to Wavefront OBJ 3D model file (default: embedded procedural model)",
        )
        models.add_argument(
            "--models",
            nargs="+",
            metavar="OBJ",
            help="OBJ files to preload and cycle through with N (starts with the first)",
        )
        parser.add_argument("--albedo-texture", type=Path, help="Albedo image for the central model (requires Pillow and mesh UVs)")
        parser.add_argument("--texture-filter", choices=("nearest", "linear"), default="linear", help="Albedo texture filtering (default: linear)")
        parser.add_argument("--texture-wrap", choices=("repeat", "clamp_to_edge", "mirror_repeat"), default="repeat", help="Albedo texture U/V addressing (default: repeat)")
        parser.add_argument(
            "--output",
            type=str,
            default=None,
            help="Path to output PNG image file when running offscreen",
        )
        parser.add_argument("--hdr", action="store_true", help="Render linear HDR through a separate ACES post pass")
        parser.add_argument("--exposure", type=validate_exposure, default=1.0, help="HDR exposure multiplier (0.01 to 64)")
        parser.add_argument("--hdr-comparison", action="store_true", help="Compare HDR-off rendering on the left with HDR plus exposure on the right")
        return parser

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

    @staticmethod
    def _load_model(path: str) -> Mesh:
        if not path.strip():
            raise ValueError("Model path cannot be empty")
        if os.path.isfile(path):
            mesh = load_obj(path, smooth_normals=True)
        elif "\n" in path:
            mesh = load_obj_from_string(path, smooth_normals=True)
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
        if not mesh.vertices or not mesh.indices:
            raise ValueError(f"Invalid OBJ model: no valid geometry found in {path}")
        return mesh

    def cycle_model(self) -> Optional[str]:
        """Switch meshes without changing scene, camera, or material state."""
        self._model_index = (self._model_index + 1) % len(self._model_meshes)
        self.model_node.mesh = self._model_meshes[self._model_index]
        if self.model_paths:
            self.model_path = self.model_paths[self._model_index]
        return self.model_path

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
        """Handle keyboard events for models, materials, lighting, and termination."""
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
        elif k in ("n", "key_n"):
            self.cycle_model()
        elif k in ("h", "key_h"):
            self.renderer.hdr = not self.renderer.hdr
        elif k in ("c", "key_c"):
            self.renderer.comparison = not self.renderer.comparison
            self.renderer.hdr = True
        elif k in ("[", "key_left_bracket"):
            self.renderer.exposure = max(0.01, self.renderer.exposure / 2.0)
        elif k in ("]", "key_right_bracket"):
            self.renderer.exposure = min(64.0, self.renderer.exposure * 2.0)
        elif k in ("q", "escape", "key_q", "key_escape"):
            self.stop()

    def compile_embedded_pipeline(self, target: Optional[str] = None) -> Path:
        """Compile genuine embedded GfxIR shaders for 3D PBR and 2D overlay across Vulkan, Metal, DX12."""
        target_name = (target or self.target).lower()
        target_enum_map = {
            "vulkan": Target.VULKAN,
            "metal": Target.METAL,
            "directx": Target.DIRECTX,
        }
        if target_name not in target_enum_map:
            raise ValueError(f"Unsupported graphics target: {target_name!r}")
        target_enum = target_enum_map[target_name]

        if not hasattr(self, "_pipeline_dir") or self._pipeline_dir is None:
            self._pipeline_dir = tempfile.TemporaryDirectory(prefix="minigfx_viewer_")

        settings = (target_name, self.renderer.hdr, self.renderer.exposure, self.renderer.comparison)
        suffix = f"_hdr_{self.renderer.exposure}_{int(self.renderer.comparison)}" if self.renderer.hdr else ""
        out_dir = Path(self._pipeline_dir.name) / f"{target_name}_pipeline{suffix}"
        pbr_dir = out_dir / "pbr_forward"
        overlay_dir = out_dir / "overlay"
        manifest_file = pbr_dir / "pipeline.json"

        if manifest_file.exists() and (overlay_dir / "pipeline.json").exists():
            self.compiled_manifest = str(manifest_file)
            self._pipeline_settings = settings
            pipe_record = self.renderer.get_or_create_pipeline(self.current_material_name)
            pipe_record.gpu_pipeline = str(manifest_file)
            return manifest_file

        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)

        # 1. Primary 3D Scene PBR Pipeline (depth test & depth write enabled)
        pbr_gpu_pipe = GpuPipeline(
            model_vertex_shader,
            (make_comparison_fragment() if self.renderer.comparison else linear_pbr_fragment)
            if self.renderer.hdr else material_fragment_shader,
            depth_stencil=DepthStencil(
                format="d32_sfloat",
                depth_test_enable=True,
                depth_write_enable=True,
                depth_compare_op="less_equal",
            ),
        )
        pbr_manifest_path = compile_pipeline(pbr_gpu_pipe, target_enum, pbr_dir)
        if pbr_manifest_path is None or not Path(pbr_manifest_path).exists():
            raise RuntimeError(f"PBR pipeline compilation failed for target {target_name}")

        if self.albedo_texture is not None:
            textured_fragment = (
                make_textured_comparison_fragment() if self.renderer.comparison else linear_textured_pbr_fragment
            ) if self.renderer.hdr else textured_pbr_fragment
            compile_pipeline(
                GpuPipeline(model_vertex_shader, textured_fragment, depth_stencil=pbr_gpu_pipe.depth_stencil),
                target_enum, out_dir / "textured_pbr",
            )

        # 2. Dedicated 2D Overlay Pipeline (depth test & depth write disabled)
        overlay_gpu_pipe = GpuPipeline(
            overlay_vertex_shader,
            make_hdr_overlay_shader(self.renderer.exposure, self.renderer.comparison)
            if self.renderer.hdr else overlay_fragment_shader,
            depth_stencil=DepthStencil(
                format="d32_sfloat",
                depth_test_enable=False,
                depth_write_enable=False,
                depth_compare_op="always",
            ),
        )
        overlay_manifest_path = compile_pipeline(overlay_gpu_pipe, target_enum, overlay_dir)
        if overlay_manifest_path is None or not Path(overlay_manifest_path).exists():
            raise RuntimeError(f"Overlay pipeline compilation failed for target {target_name}")
        if self.renderer.hdr:
            compile_pipeline(make_post_pipeline(self.renderer.exposure, self.renderer.comparison),
                             target_enum, out_dir / "post")

        self.compiled_manifest = str(pbr_manifest_path)
        self._pipeline_settings = settings
        pipe_record = self.renderer.get_or_create_pipeline(self.current_material_name)
        pipe_record.gpu_pipeline = str(pbr_manifest_path)
        return Path(pbr_manifest_path)

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

        # Enforce deterministic fixed time-step for headless offscreen rendering
        if self.offscreen:
            dt = 1.0 / 60.0
            self.dt = dt
            self.fps = 60.0
            self.elapsed_time = self.frame_count * dt
        else:
            if self.frame_count == 0 and dt < 0.001:
                dt = 1.0 / 60.0
            if self.fps <= 0.0 or self.fps > 5000.0:
                self.fps = 60.0

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

    def render_offscreen_frame(self) -> RenderedImage:
        """Render scene into RenderedImage using genuine GPU execution via mlir_gpu.viewer."""
        settings = (self.target, self.renderer.hdr, self.renderer.exposure, self.renderer.comparison)
        if (self.compiled_manifest is None or not Path(self.compiled_manifest).exists()
                or getattr(self, "_pipeline_settings", settings) != settings):
            self.compile_embedded_pipeline()

        pbr_manifest_path = Path(self.compiled_manifest)
        pbr_manifest_data = json.loads(pbr_manifest_path.read_text())
        if (pbr_manifest_path.parent / "overlay" / "pipeline.json").exists():
            out_dir = pbr_manifest_path.parent
            overlay_manifest_path = out_dir / "overlay" / "pipeline.json"
        elif (pbr_manifest_path.parent.parent / "overlay" / "pipeline.json").exists():
            out_dir = pbr_manifest_path.parent.parent
            overlay_manifest_path = out_dir / "overlay" / "pipeline.json"
        else:
            out_dir = pbr_manifest_path.parent
            overlay_manifest_path = pbr_manifest_path
        overlay_manifest_data = json.loads(overlay_manifest_path.read_text())

        consolidated_vb = bytearray()
        consolidated_ib = bytearray()
        draw_calls: List[DrawCall] = []
        v_offset = 0
        first_index = 0

        cam_params = self.camera.get_params(time=self.elapsed_time)

        for node in self.scene.traverse():
            if node is self.scene or getattr(node, "mesh", None) is None:
                continue
            if hasattr(node, "is_visible") and not node.is_visible(self.camera):
                continue

            mesh = node.mesh
            verts = getattr(mesh, "vertices", [])
            indices = getattr(mesh, "indices", [])
            if not verts or not indices:
                continue

            mat = getattr(node, "material", self.materials[self.current_material_name])
            world_transform = getattr(node, "world_transform", Mat4x4.identity())
            obj_params = ObjectParams(model=world_transform, color=Vec4(1.0, 1.0, 1.0, 1.0))

            # Pack push constants: cam (80B) + obj (80B) + mat (32B) + light (32B) = 224B
            constants = (
                cam_params.to_bytes()
                + obj_params.to_bytes()
                + mat.to_bytes()
                + self.light_params.to_bytes()
            )

            draw_calls.append(
                DrawCall(
                    pipeline="textured_pbr" if getattr(mat, "is_textured", False) else "pbr_forward",
                    type="indexed",
                    topology="triangle-list",
                    index_count=len(indices),
                    first_index=first_index,
                    vertex_offset=v_offset,
                    instance_count=1,
                    first_instance=0,
                    index_type="uint16",
                    constant_offset=0,
                    constant_size=len(constants),
                    parameters={"data": list(constants)},
                )
            )

            consolidated_vb.extend(mesh.vertex_bytes())
            consolidated_ib.extend(mesh.index_bytes())
            v_offset += len(verts)
            first_index += len(indices)

        # Embedded debug overlay rendered via dedicated overlay pipeline
        if self.debug_overlay is not None:
            metrics_text = self.debug_overlay.format_metrics()
            quads = generate_text_quads(metrics_text, start_x=10.0, start_y=10.0, scale=1.0)
            quad_verts, quad_indices = quads.to_buffers()
            if quad_verts and quad_indices:
                w_f, h_f = float(self.width), float(self.height)
                overlay_vb = bytearray()
                overlay_ib = bytearray()
                for qv in quad_verts:
                    px, py, u, v = qv[0], qv[1], qv[2], qv[3]
                    ndc_x = (px / w_f) * 2.0 - 1.0
                    ndc_y = 1.0 - (py / h_f) * 2.0
                    overlay_vb.extend(
                        ModelVertex(
                            position=Vec3(ndc_x, ndc_y, 0.0),
                            normal=Vec3(0.0, 0.0, 1.0),
                            uv=Vec2(u, v),
                        ).to_bytes()
                    )
                for qi in quad_indices:
                    overlay_ib.extend(struct.pack("<H", qi))

                overlay_params = OverlayParams(
                    text_color=getattr(self.debug_overlay, "params", OverlayParams()).text_color,
                    bg_color=getattr(self.debug_overlay, "params", OverlayParams()).bg_color,
                    viewport=Vec2(w_f, h_f),
                    scale=getattr(self.debug_overlay, "params", OverlayParams()).scale,
                )
                overlay_constants = overlay_params.to_bytes()

                draw_calls.append(
                    DrawCall(
                        pipeline="overlay",
                        type="indexed",
                        topology="triangle-list",
                        index_count=len(quad_indices),
                        first_index=first_index,
                        vertex_offset=v_offset,
                        instance_count=1,
                        first_instance=0,
                        index_type="uint16",
                        constant_offset=0,
                        constant_size=len(overlay_constants),
                        parameters={"data": list(overlay_constants)},
                    )
                )

                consolidated_vb.extend(overlay_vb)
                consolidated_ib.extend(overlay_ib)
                v_offset += len(quad_verts)
                first_index += len(quad_indices)

        pbr_entries = [
            {"stage": e["stage"], "entry_point": e["entry_point"], "binary": f"pbr_forward/{e['binary']}" if (out_dir / "pbr_forward").is_dir() else e["binary"]}
            for e in pbr_manifest_data["entries"]
        ]
        overlay_entries = [
            {"stage": e["stage"], "entry_point": e["entry_point"], "binary": f"overlay/{e['binary']}" if (out_dir / "overlay").is_dir() else e["binary"]}
            for e in overlay_manifest_data["entries"]
        ]

        multi_manifest = {
            "schema_version": MULTI_DRAW_SCHEMA_VERSION,
            "target": self.target,
            "target_profile": TargetProfile.default_for_target(self.target).to_dict(),
            "raster_convention": RASTER_CONVENTION,
            "runtime_verified": False,
            "backend_status": "validated-binary",
            "pipelines": {
                "pbr_forward": {
                    "entries": pbr_entries,
                    "parameters": pbr_manifest_data.get("parameters"),
                    "interface": pbr_manifest_data.get("interface"),
                    "depth_stencil": pbr_manifest_data.get("depth_stencil"),
                    "input_layout": pbr_manifest_data.get("input_layout"),
                },
                "overlay": {
                    "entries": overlay_entries,
                    "parameters": overlay_manifest_data.get("parameters"),
                    "interface": overlay_manifest_data.get("interface"),
                    "depth_stencil": overlay_manifest_data.get("depth_stencil", {
                        "format": "d32_sfloat",
                        "depth_test_enable": False,
                        "depth_write_enable": False,
                        "depth_compare_op": "always",
                    }),
                    "input_layout": overlay_manifest_data.get("input_layout"),
                },
            },
            "draws": [d.to_dict() for d in draw_calls],
            "depth_stencil": DepthStencil().to_dict(),
            "input_layout": pbr_manifest_data.get("input_layout"),
        }

        multi_manifest_path = out_dir / "multi_manifest.json"
        textures, samplers = {}, {}
        if self.albedo_texture is not None:
            textured_data = json.loads((out_dir / "textured_pbr" / "pipeline.json").read_text())
            multi_manifest["pipelines"]["textured_pbr"] = {
                key: textured_data[key]
                for key in ("entries", "parameters", "interface", "depth_stencil", "input_layout")
                if key in textured_data
            }
            multi_manifest["pipelines"]["textured_pbr"]["entries"] = [
                {**entry, "binary": f"textured_pbr/{entry['binary']}"}
                for entry in textured_data["entries"]
            ]
            textured_material = self.materials["metallic"]
            if not isinstance(textured_material, Material):
                raise TypeError("The textured metallic preset must be a Material")
            textures, samplers = textured_material.gpu_resources()
        multi_manifest_path.write_text(json.dumps(multi_manifest, indent=2))

        program = load_program(
            multi_manifest_path,
            vertex_buffer=bytes(consolidated_vb),
            index_buffer=bytes(consolidated_ib),
            textures=textures,
            samplers=samplers,
        )
        if self.renderer.hdr:
            program = with_post_process(program, load_program(out_dir / "post" / "pipeline.json"))

        image = render_offscreen(
            program,
            width=self.width,
            height=self.height,
            time=self.elapsed_time,
        )
        if not isinstance(image, RenderedImage):
            raise RuntimeError(f"render_offscreen returned unexpected type {type(image)}, expected RenderedImage")
        return image

    def save_png(self, path: Union[str, Path], image: Any) -> None:
        """Write RenderedImage to destination PNG file via mlir_gpu.viewer.write_png."""
        p = Path(path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.unlink(missing_ok=True)
        write_png(p, image)

    def render(self) -> List[DrawCommand]:
        """Record and dispatch draw commands for visible scene meshes."""
        super().render()
        commands: List[DrawCommand] = []

        if self.offscreen:
            self.fps = 60.0
            self.dt = 1.0 / 60.0

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

    def resize(self, width: int, height: int) -> None:
        """Keep the projection aspect ratio in sync with drawable pixels."""
        super().resize(width, height)
        self.camera.aspect = float(width) / float(height)

    def cleanup(self) -> None:
        """Release allocated renderer and engine resources."""
        try:
            if self.output_path and self.last_rendered_image is not None:
                self.save_png(self.output_path, self.last_rendered_image)
        finally:
            super().cleanup()
            if hasattr(self, "_pipeline_dir") and self._pipeline_dir is not None:
                self._pipeline_dir.cleanup()
                self._pipeline_dir = None


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
