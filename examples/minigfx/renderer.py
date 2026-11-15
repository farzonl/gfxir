"""Forward renderer, pipeline cache, draw list state sorting, and command recording."""

from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import Any, Dict, List, Optional, Sequence, Union

from .camera import Camera, CameraParams
from .math import Mat4x4, Vec3, Vec4


@dataclass
class Pipeline:
    """Compiled or cached graphics pipeline state."""

    key: str
    vertex_shader: Any = None
    fragment_shader: Any = None
    gpu_pipeline: Any = None


@dataclass
class DrawCommand:
    """Recorded graphics draw command with isolated push constants and optional bound textures."""

    pipeline_id: str
    push_constants: bytes = b""
    mesh: Any = None
    node: Any = None
    vertex_count: int = 0
    index_count: int = 0
    first_index: int = 0
    vertex_offset: int = 0
    textures: Dict[str, Any] = field(default_factory=dict)
    samplers: Dict[str, Any] = field(default_factory=dict)


def sort_draw_items(items: Sequence[Any]) -> List[Any]:
    """Sort draw items stably by pipeline identifier to eliminate state switches."""

    def _get_key(item: Any) -> str:
        if isinstance(item, dict):
            k = item.get("pipe") or item.get("pipeline") or item.get("pipeline_id") or ""
        else:
            k = getattr(item, "pipe", getattr(item, "pipeline", getattr(item, "pipeline_id", "")))
        return str(k)

    return sorted(items, key=_get_key)


class ForwardRenderer:
    """Forward rendering pipeline manager, draw list sorter, and command recorder."""

    def __init__(
        self,
        target: Optional[str] = None,
        width: int = 1280,
        height: int = 720,
        offscreen: bool = False,
        **kwargs: Any,
    ) -> None:
        try:
            from .app import get_default_target

            default_target = get_default_target()
        except Exception:
            default_target = "vulkan"

        self.target: str = str(target or default_target)
        self.width: int = int(width)
        self.height: int = int(height)
        self.offscreen: bool = bool(offscreen)
        self.pipelines: dict[str, Pipeline] = {}
        self.frame_constants: bytes = b""
        self.light_constants: bytes = b""
        self.last_commands: List[DrawCommand] = []

    def get_or_create_pipeline(
        self,
        key_or_mode: Any,
        vertex_shader: Optional[Any] = None,
        fragment_shader: Optional[Any] = None,
        gpu_pipeline: Optional[Any] = None,
    ) -> Pipeline:
        """Retrieve cached pipeline or compile, store, and return it."""
        key_str = key_or_mode if isinstance(key_or_mode, str) else getattr(key_or_mode, "name", str(key_or_mode))
        if key_str in self.pipelines:
            return self.pipelines[key_str]

        pipe = Pipeline(
            key=key_str,
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader,
            gpu_pipeline=gpu_pipeline,
        )
        self.pipelines[key_str] = pipe
        return pipe

    @classmethod
    def sort_draw_items(cls, items: Sequence[Any]) -> List[Any]:
        """Class method stably sorting draw items by pipeline ID."""
        return sort_draw_items(items)

    def update_frame_constants(self, camera: Any, time: float = 0.0) -> bytes:
        """Pack std430 CameraParams (exactly 80 bytes: 64B view_proj + 12B eye_pos + 4B time)."""
        if hasattr(camera, "get_params"):
            buf = camera.get_params(time=time).to_bytes()
        elif hasattr(camera, "to_bytes"):
            buf = camera.to_bytes()
        else:
            vp = getattr(camera, "view_proj", Mat4x4.identity())
            eye = getattr(camera, "eye", getattr(camera, "position", Vec3(0.0, 0.0, 0.0)))
            if not isinstance(eye, Vec3):
                eye = Vec3(eye[0], eye[1], eye[2])
            buf = CameraParams(view_proj=vp, eye_pos=eye, time=float(time)).to_bytes()
        self.frame_constants = buf
        return buf

    def upload_object_constants(self, node: Any, material: Any = None) -> bytes:
        """Pack std430 ObjectParams (80 bytes >= 64B: 64B model transform + 16B color)."""
        from .material import ObjectParams

        model = getattr(node, "world_transform", Mat4x4.identity()) if node is not None else Mat4x4.identity()
        color = Vec4(1.0, 1.0, 1.0, 1.0)
        if material is not None:
            c = getattr(material, "base_color", getattr(material, "albedo", getattr(material, "color", None)))
            if c is not None:
                color = (
                    Vec4(c.x, c.y, c.z, getattr(c, "w", 1.0))
                    if hasattr(c, "x")
                    else Vec4(c[0], c[1], c[2], c[3] if len(c) > 3 else 1.0)
                )
        return ObjectParams(model=model, color=color).to_bytes()

    def record_draw(
        self,
        pipeline_id: Any,
        push_constants: Any = b"",
        mesh: Any = None,
        node: Any = None,
        **kwargs: Any,
    ) -> DrawCommand:
        """Record a single draw command with isolated push constants."""
        if hasattr(push_constants, "to_bytes"):
            pc_bytes = push_constants.to_bytes()
        elif isinstance(push_constants, (bytes, bytearray, memoryview)):
            pc_bytes = bytes(push_constants)
        elif push_constants:
            pc_bytes = bytes(push_constants)
        else:
            pc_bytes = b""

        vertex_count = kwargs.get("vertex_count")
        if vertex_count is None and mesh is not None and hasattr(mesh, "vertices"):
            vertex_count = len(mesh.vertices)
        elif vertex_count is None:
            vertex_count = 0

        index_count = kwargs.get("index_count")
        if index_count is None and mesh is not None and hasattr(mesh, "indices"):
            index_count = len(mesh.indices)
        elif index_count is None:
            index_count = 0

        raw_textures = kwargs.get("textures")
        raw_samplers = kwargs.get("samplers")
        mat = kwargs.get("material")
        if mat is None and node is not None:
            mat = getattr(node, "material", None)
        textures: Dict[str, Any] = dict(raw_textures) if raw_textures else {}
        samplers: Dict[str, Any] = dict(raw_samplers) if raw_samplers else {}
        if not textures and mat is not None and getattr(mat, "albedo_texture", None) is not None:
            from .material import Sampler as MgfxSampler

            textures["albedo_tex"] = mat.albedo_texture
            samplers["albedo_sampler"] = getattr(mat, "sampler", None) or MgfxSampler()

        return DrawCommand(
            pipeline_id=str(pipeline_id),
            push_constants=pc_bytes,
            mesh=mesh,
            node=node,
            vertex_count=int(vertex_count),
            index_count=int(index_count),
            first_index=int(kwargs.get("first_index", 0)),
            vertex_offset=int(kwargs.get("vertex_offset", 0)),
            textures=textures,
            samplers=samplers,
        )

    def record_commands(self, draw_list: Sequence[Any]) -> List[DrawCommand]:
        """Record sequential draw commands from draw list, sorting and auto-uploading constants."""
        sorted_items = self.sort_draw_items(draw_list)
        commands: List[DrawCommand] = []

        for item in sorted_items:
            if isinstance(item, dict):
                pipe_id = item.get("pipe") or item.get("pipeline") or item.get("pipeline_id") or "default"
                mesh = item.get("mesh")
                node = item.get("node")
                mat = item.get("material") or (getattr(node, "material", None) if node is not None else None)
                pc = item.get("push_constants")
                if pc is None and node is not None:
                    pc = self.upload_object_constants(node, mat)
                elif pc is None:
                    pc = b""
                cmd = self.record_draw(
                    pipeline_id=pipe_id,
                    push_constants=pc,
                    mesh=mesh,
                    node=node,
                    material=mat,
                    textures=item.get("textures"),
                    samplers=item.get("samplers"),
                    vertex_count=item.get("vertex_count"),
                    index_count=item.get("index_count"),
                    first_index=item.get("first_index", 0),
                    vertex_offset=item.get("vertex_offset", 0),
                )
            elif isinstance(item, DrawCommand):
                cmd = item
            else:
                pipe_id = getattr(item, "pipe", getattr(item, "pipeline", getattr(item, "pipeline_id", "default")))
                mesh = getattr(item, "mesh", None)
                node = getattr(item, "node", None)
                mat = getattr(item, "material", getattr(node, "material", None) if node is not None else None)
                pc = getattr(item, "push_constants", None)
                if pc is None and node is not None:
                    pc = self.upload_object_constants(node, mat)
                elif pc is None:
                    pc = b""
                cmd = self.record_draw(
                    pipeline_id=pipe_id,
                    push_constants=pc,
                    mesh=mesh,
                    node=node,
                    material=mat,
                    textures=getattr(item, "textures", None),
                    samplers=getattr(item, "samplers", None),
                    vertex_count=getattr(item, "vertex_count", None),
                    index_count=getattr(item, "index_count", None),
                    first_index=getattr(item, "first_index", 0),
                    vertex_offset=getattr(item, "vertex_offset", 0),
                )
            commands.append(cmd)

        self.last_commands = commands
        return commands

    def build_draw_list(
        self,
        scene: Optional[Any] = None,
        camera: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        """Traverse scene graph, evaluate frustum culling against camera, and collect draw items."""
        if scene is None or camera is None:
            return []

        draw_items: List[Dict[str, Any]] = []
        if hasattr(scene, "traverse"):
            nodes = list(scene.traverse())
        elif hasattr(scene, "is_visible"):
            nodes = [scene]
        else:
            nodes = []

        for node in nodes:
            # Skip root scene container node if it has no mesh
            if node is scene and getattr(node, "mesh", None) is None:
                continue
            if hasattr(node, "is_visible") and not node.is_visible(camera):
                continue

            mat = getattr(node, "material", None)
            pipe_key = getattr(node, "pipeline_id", None) or getattr(node, "pipeline", None)
            if pipe_key is None:
                if mat is not None:
                    mat_name = getattr(mat, "name", "")
                    if mat_name:
                        pipe_key = mat_name
                    elif getattr(mat, "albedo_texture", None) is not None:
                        pipe_key = "textured_pbr"
                    else:
                        pipe_key = mat
                else:
                    pipe_key = "default"
            if not isinstance(pipe_key, str):
                pipe_key = getattr(pipe_key, "name", None) or str(pipe_key)

            draw_items.append({
                "pipe": pipe_key,
                "node": node,
                "mesh": getattr(node, "mesh", None),
                "material": getattr(node, "material", None),
            })

        return draw_items

    def render_frame(
        self,
        camera: Optional[Any] = None,
        lights: Optional[Any] = None,
        scene: Optional[Any] = None,
        time: float = 0.0,
        **kwargs: Any,
    ) -> List[DrawCommand]:
        """Execute forward render pass; raises ValueError if camera or lights is None."""
        if camera is None:
            raise ValueError("camera cannot be None")
        if lights is None:
            raise ValueError("lights cannot be None")

        if hasattr(lights, "to_bytes"):
            self.light_constants = lights.to_bytes()

        self.update_frame_constants(camera, time=time)
        draw_list = self.build_draw_list(scene, camera)
        commands = self.record_commands(draw_list)
        return commands


__all__ = [
    "DrawCommand",
    "ForwardRenderer",
    "Pipeline",
    "sort_draw_items",
]
