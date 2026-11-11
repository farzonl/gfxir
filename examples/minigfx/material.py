"""Embedded-shader material system and PBR-lite lighting for MiniGfx."""

from __future__ import annotations

import math
import struct
from typing import Any, Callable

from examples.minigfx.math import Mat4, Vec2, Vec3, Vec4
from examples.minigfx.tonemap import tonemap_and_gamma

try:
    from mlir_gpu.frontend import Type, Value, Vertex, fragment, vec2, vec3, vec4, vertex
    import mlir_gpu.shader_math as sm
except ImportError:
    Type = Value = Vertex = fragment = vec2 = vec3 = vec4 = vertex = sm = None


def _to_vec3(v: Any) -> Vec3:
    if isinstance(v, Vec3): return v
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"): return Vec3(v.x, v.y, v.z)
    if isinstance(v, (tuple, list)) and len(v) >= 3: return Vec3(v[0], v[1], v[2])
    raise TypeError(f"Cannot convert {v!r} to Vec3")


class _Vec4(Vec4):
    def __getitem__(self, idx: Any) -> Any:
        return tuple(self)[idx] if isinstance(idx, slice) else super().__getitem__(idx)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (Vec4, tuple, list)) and len(o) == self._dim:
            return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(self, o))
        return False


def _to_vec4(v: Any) -> Vec4:
    if isinstance(v, _Vec4): return v
    if isinstance(v, Vec4): return _Vec4(v.x, v.y, v.z, v.w)
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"): return _Vec4(v.x, v.y, v.z, getattr(v, "w", 1.0))
    if isinstance(v, (tuple, list)) and len(v) in (3, 4): return _Vec4(v[0], v[1], v[2], 1.0 if len(v) == 3 else v[3])
    raise TypeError(f"Cannot convert {v!r} to Vec4")


class MaterialParams:
    """Shared host/device material parameters (std430, 32 bytes)."""
    __struct_size__, __struct_alignment__, _is_gfx_struct = 32, 16, True

    def __init__(self, albedo: Any = None, roughness: float = 0.5, metallic: float = 0.0,
                 ambient: float = 0.0, base_color: Any = None, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"MaterialParams unexpected kwargs: {list(kwargs.keys())}")
        col = albedo if albedo is not None else (base_color if base_color is not None else (1.0, 1.0, 1.0, 1.0))
        self._albedo = _to_vec4(col)
        r, m, amb = float(roughness), float(metallic), float(ambient)
        if not (0.0 <= r <= 1.0): raise ValueError(f"Roughness {r} outside [0, 1]")
        if not (0.0 <= m <= 1.0): raise ValueError(f"Metallic {m} outside [0, 1]")
        if amb < 0.0 or not math.isfinite(amb): raise ValueError(f"Ambient {amb} must be non-negative and finite")
        self.roughness, self.metallic, self.ambient = r, m, amb

    @property
    def albedo(self) -> Vec4: return self._albedo
    @albedo.setter
    def albedo(self, v: Any) -> None: self._albedo = _to_vec4(v)
    @property
    def base_color(self) -> Vec4: return self._albedo
    @base_color.setter
    def base_color(self, v: Any) -> None: self._albedo = _to_vec4(v)

    def to_bytes(self) -> bytes:
        a = self._albedo
        return struct.pack("<8f", a.x, a.y, a.z, a.w, self.roughness, self.metallic, self.ambient, 0.0)

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> MaterialParams:
        if len(buf) < 32: raise ValueError(f"Buffer < 32B for MaterialParams")
        r, g, b, a, ro, me, am = struct.unpack_from("<7f", buf, 0)
        return cls(albedo=Vec4(r, g, b, a), roughness=ro, metallic=me, ambient=am)

    @classmethod
    def as_parameter_block(cls, name: str | None = None) -> Any:
        if Type is None: return None
        from mlir_gpu.parameters import ParameterBlock
        return ParameterBlock(name or "material", (("albedo", Type.VEC4), ("roughness", Type.F32), ("metallic", Type.F32), ("ambient", Type.F32)))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, MaterialParams)
            and self._albedo == o._albedo
            and math.isclose(self.roughness, o.roughness, rel_tol=1e-5, abs_tol=1e-6)
            and math.isclose(self.metallic, o.metallic, rel_tol=1e-5, abs_tol=1e-6)
            and math.isclose(self.ambient, o.ambient, rel_tol=1e-5, abs_tol=1e-6)
        )


class LightParams:
    """Shared host/device directional light parameters (std430, 32 bytes)."""
    __struct_size__, __struct_alignment__, _is_gfx_struct = 32, 16, True

    def __init__(self, direction: Any = (0.0, -1.0, 0.0), intensity: float = 1.0,
                 color: Any = (1.0, 1.0, 1.0), ambient: float = 0.1, **kwargs: Any) -> None:
        if kwargs:
            raise TypeError(f"LightParams unexpected kwargs: {list(kwargs.keys())}")
        d, f_int, f_amb = _to_vec3(direction), float(intensity), float(ambient)
        if d.length_sq() < 1e-12: raise ValueError("Light direction must not be zero-length")
        if f_int < 0.0 or not math.isfinite(f_int): raise ValueError("Light intensity must be non-negative")
        if f_amb < 0.0 or not math.isfinite(f_amb): raise ValueError("Ambient intensity must be non-negative")
        self.direction, self.intensity, self.color, self.ambient = d.normalize(), f_int, _to_vec3(color), f_amb

    def to_bytes(self) -> bytes:
        d, c = self.direction, self.color
        return struct.pack("<3ff3ff", d.x, d.y, d.z, self.intensity, c.x, c.y, c.z, self.ambient)

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> LightParams:
        if len(buf) < 32: raise ValueError(f"Buffer < 32B for LightParams")
        dx, dy, dz, i, cx, cy, cz, a = struct.unpack_from("<8f", buf, 0)
        return cls(direction=Vec3(dx, dy, dz), intensity=i, color=Vec3(cx, cy, cz), ambient=a)

    @classmethod
    def as_parameter_block(cls, name: str | None = None) -> Any:
        if Type is None: return None
        from mlir_gpu.parameters import ParameterBlock
        return ParameterBlock(name or "light", (("direction", Type.VEC3), ("intensity", Type.F32), ("color", Type.VEC3), ("ambient", Type.F32)))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, LightParams)
            and self.direction == o.direction
            and math.isclose(self.intensity, o.intensity, rel_tol=1e-5, abs_tol=1e-6)
            and self.color == o.color
            and math.isclose(self.ambient, o.ambient, rel_tol=1e-5, abs_tol=1e-6)
        )


class ObjectParams:
    """Shared host/device per-draw parameters (std430, 80 bytes)."""
    __struct_size__, __struct_alignment__, _is_gfx_struct = 80, 16, True

    def __init__(self, model: Any = None, color: Any = (1.0, 1.0, 1.0, 1.0), **kwargs: Any) -> None:
        if kwargs: raise TypeError(f"ObjectParams unexpected kwargs: {list(kwargs.keys())}")
        self.model = model if isinstance(model, Mat4) else (Mat4(model) if model is not None else Mat4.identity())
        self.color = _to_vec4(color)

    def to_bytes(self) -> bytes:
        c = self.color
        return self.model.to_bytes() + struct.pack("<4f", c.x, c.y, c.z, c.w)

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> ObjectParams:
        if len(buf) < 80: raise ValueError(f"Buffer < 80B for ObjectParams")
        return cls(model=Mat4.from_bytes(buf[0:64]), color=Vec4(*struct.unpack_from("<4f", buf, 64)))

    @classmethod
    def as_parameter_block(cls, name: str | None = None) -> Any:
        if Type is None: return None
        from mlir_gpu.parameters import ParameterBlock
        return ParameterBlock(name or "obj", (("model", Type.MAT4X4), ("color", Type.VEC4)))

    def __eq__(self, o: object) -> bool:
        return isinstance(o, ObjectParams) and self.model == o.model and self.color == o.color


class CameraParams:
    """Shared host/device camera parameters (std430, 80 bytes)."""
    __struct_size__, __struct_alignment__, _is_gfx_struct = 80, 16, True

    def __init__(self, view_proj: Any = None, eye_pos: Any = (0.0, 0.0, 0.0), time: float = 0.0, **kwargs: Any) -> None:
        if kwargs: raise TypeError(f"CameraParams unexpected kwargs: {list(kwargs.keys())}")
        self.view_proj = view_proj if isinstance(view_proj, Mat4) else (Mat4(view_proj) if view_proj is not None else Mat4.identity())
        self.eye_pos = _to_vec3(eye_pos)
        f_time = float(time)
        if not math.isfinite(f_time): raise ValueError(f"Camera time {f_time} must be finite")
        self.time = f_time

    def to_bytes(self) -> bytes:
        e = self.eye_pos
        return self.view_proj.to_bytes() + struct.pack("<4f", e.x, e.y, e.z, self.time)

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> CameraParams:
        if len(buf) < 80: raise ValueError(f"Buffer < 80B for CameraParams")
        return cls(view_proj=Mat4.from_bytes(buf[0:64]), eye_pos=Vec3(*struct.unpack_from("<3f", buf, 64)), time=struct.unpack_from("<f", buf, 76)[0])

    @classmethod
    def as_parameter_block(cls, name: str | None = None) -> Any:
        if Type is None: return None
        from mlir_gpu.parameters import ParameterBlock
        return ParameterBlock(name or "cam", (("view_proj", Type.MAT4X4), ("eye_pos", Type.VEC3), ("time", Type.F32)))

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, CameraParams)
            and self.view_proj == o.view_proj
            and self.eye_pos == o.eye_pos
            and math.isclose(self.time, o.time, rel_tol=1e-5, abs_tol=1e-6)
        )


def evaluate_diffuse_lighting(normal: Any, light_dir: Any, albedo: Any, metallic: float = 0.0) -> tuple[float, float, float]:
    n, l = _to_vec3(normal), _to_vec3(light_dir)
    if n.length_sq() < 1e-12 or l.length_sq() < 1e-12: return (0.0, 0.0, 0.0)
    n_dot_l = n.normalize().dot(l.normalize())
    if n_dot_l <= 0.0: return (0.0, 0.0, 0.0)
    factor = n_dot_l * (1.0 - float(metallic))
    a = _to_vec4(albedo)
    return (a.x * factor, a.y * factor, a.z * factor)


def evaluate_specular(normal: Any, view_dir: Any, light_dir: Any, roughness: float = 0.5) -> float:
    n, v, l = _to_vec3(normal), _to_vec3(view_dir), _to_vec3(light_dir)
    if n.length_sq() < 1e-12 or v.length_sq() < 1e-12 or l.length_sq() < 1e-12: return 0.0
    n, v, l = n.normalize(), v.normalize(), l.normalize()
    if n.dot(l) <= 0.0 or n.dot(v) <= 0.0: return 0.0
    h_sum = Vec3(l.x + v.x, l.y + v.y, l.z + v.z)
    if h_sum.length_sq() < 1e-12: return 0.0
    n_dot_h = max(0.0, n.dot(h_sum.normalize()))
    alpha = 2.0 / max(float(roughness) ** 2, 0.001)
    return float(n_dot_h ** alpha)


def evaluate_lighting(normal: Any, view_dir: Any, light: LightParams, material: MaterialParams) -> tuple[float, float, float]:
    amb_factor = light.ambient + material.ambient
    amb = (material.albedo.x * amb_factor, material.albedo.y * amb_factor, material.albedo.z * amb_factor)
    diff = evaluate_diffuse_lighting(normal, light.direction, material.albedo, material.metallic)
    spec = evaluate_specular(normal, view_dir, light.direction, material.roughness) * light.intensity
    return (amb[0] + diff[0] * light.intensity + spec, amb[1] + diff[1] * light.intensity + spec, amb[2] + diff[2] * light.intensity + spec)


class ShaderFunction:
    """Callable shader wrapper supporting CPU evaluation and GPU AST pipeline tracing."""
    def __init__(self, entry: Any, kind: str, cpu_fn: Callable | None = None) -> None:
        self._entry, self.kind, self._cpu_fn = entry, kind, cpu_fn
        self.name = getattr(entry, "name", kind)
        self.function = getattr(entry, "function", entry)
        self.workgroup_size = getattr(entry, "workgroup_size", (1, 1, 1))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._cpu_fn is not None: return self._cpu_fn(*args, **kwargs)
        if callable(self.function): return self.function(*args, **kwargs)
        raise TypeError(f"Shader {self.name} is not callable on CPU")

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        if self._entry is not None and hasattr(self._entry, "trace"): return self._entry.trace(*args, **kwargs)
        raise RuntimeError("Compiler unavailable for shader tracing")


def _cpu_vertex(pos: Any = (0.0, 0.0, 0.0), cam: Any = None, obj: Any = None) -> Vec4:
    model = getattr(obj, "model", Mat4.identity())
    vp = getattr(cam, "view_proj", Mat4.identity())
    return vp * (model * _to_vec4(pos))


def _cpu_fragment(normal: Any = (0.0, 1.0, 0.0), world_pos: Any = (0.0, 0.0, 0.0), uv: Any = (0.0, 0.0),
                  cam: Any = None, mat: Any = None, light: Any = None) -> Vec4:
    c = cam if cam is not None else CameraParams()
    m = mat if mat is not None else MaterialParams()
    l = light if light is not None else LightParams()
    lit = evaluate_lighting(normal, c.eye_pos - _to_vec3(world_pos), l, m)
    final = tonemap_and_gamma(Vec3(*lit))
    return Vec4(final.x, final.y, final.z, m.albedo.w)


if vertex is not None and fragment is not None:
    @vertex
    def _gpu_vertex(vertex_index: Type.U32, cam: CameraParams, obj: ObjectParams) -> Vertex:
        pos = vec4(0.0, 0.0, 0.0, 1.0)
        world_pos = obj.model * pos
        clip_pos = cam.view_proj * world_pos
        return Vertex(clip_pos, varyings={"normal": vec3(0.0, 1.0, 0.0), "world_pos": world_pos.xyz, "uv": vec2(0.0, 0.0)})

    @fragment
    def _gpu_fragment(normal: Value, world_pos: Value, uv: Value, cam: CameraParams, mat: MaterialParams, light: LightParams) -> Value:
        n, l = sm.normalize(normal), sm.normalize(light.direction)
        n_dot_l = sm.clamp(sm.dot(n, l), 0.0, 1.0)
        diffuse = mat.albedo.xyz * n_dot_l * light.intensity * (1.0 - mat.metallic)
        ambient = mat.albedo.xyz * (light.ambient + mat.ambient)
        v = sm.normalize(cam.eye_pos - world_pos)
        h = sm.normalize(l + v)
        n_dot_h = sm.clamp(sm.dot(n, h), 0.0, 1.0)
        specular = sm.pow(n_dot_h, 2.0 / (mat.roughness * mat.roughness + 0.001)) * light.intensity
        hdr = ambient + diffuse + vec3(specular, specular, specular)
        ldr = sm.clamp((hdr * (hdr * 2.51 + 0.03)) / (hdr * (hdr * 2.43 + 0.59) + 0.14), 0.0, 1.0)
        srgb = sm.pow(ldr, 1.0 / 2.2)
        return vec4(srgb, mat.albedo.w)

    pbr_vertex = ShaderFunction(_gpu_vertex, "vertex", _cpu_vertex)
    pbr_fragment = ShaderFunction(_gpu_fragment, "fragment", _cpu_fragment)
else:
    pbr_vertex = ShaderFunction(_cpu_vertex, "vertex", _cpu_vertex)
    pbr_fragment = ShaderFunction(_cpu_fragment, "fragment", _cpu_fragment)

material_vertex_shader, material_fragment_shader = pbr_vertex, pbr_fragment
default_vs, default_ps = pbr_vertex, pbr_fragment

__all__ = [
    "CameraParams", "LightParams", "MaterialParams", "ObjectParams", "ShaderFunction",
    "default_ps", "default_vs", "evaluate_diffuse_lighting", "evaluate_lighting",
    "evaluate_specular", "material_fragment_shader", "material_vertex_shader",
    "pbr_fragment", "pbr_vertex",
]
