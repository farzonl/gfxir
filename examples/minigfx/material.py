"""Embedded-shader material system and PBR-lite lighting for MiniGfx."""

from __future__ import annotations

import math
import struct
from typing import Any, Callable, Sequence

from examples.minigfx.math import Mat4x4, Vec2, Vec3, Vec4
from examples.minigfx.tonemap import tonemap_and_gamma

try:
    from mlir_gpu.frontend import (
        Sampler as GpuSampler,
        SamplerDesc,
        Texture2D as GpuTexture2D,
        TextureData,
        Type,
        Value,
        Vertex,
        fragment,
        load_texture,
        sample_texture,
        sample_texture_level,
        vec2,
        vec3,
        vec4,
        vertex,
    )
    import mlir_gpu.shader_math as sm
except ImportError:
    GpuSampler = SamplerDesc = GpuTexture2D = TextureData = None
    Type = Value = Vertex = fragment = load_texture = sample_texture = sample_texture_level = None
    vec2 = vec3 = vec4 = vertex = sm = None


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
        self.model = model if isinstance(model, Mat4x4) else (Mat4x4(model) if model is not None else Mat4x4.identity())
        self.color = _to_vec4(color)

    def to_bytes(self) -> bytes:
        c = self.color
        return self.model.to_bytes() + struct.pack("<4f", c.x, c.y, c.z, c.w)

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> ObjectParams:
        if len(buf) < 80: raise ValueError(f"Buffer < 80B for ObjectParams")
        return cls(model=Mat4x4.from_bytes(buf[0:64]), color=Vec4(*struct.unpack_from("<4f", buf, 64)))

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
        self.view_proj = view_proj if isinstance(view_proj, Mat4x4) else (Mat4x4(view_proj) if view_proj is not None else Mat4x4.identity())
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
        return cls(view_proj=Mat4x4.from_bytes(buf[0:64]), eye_pos=Vec3(*struct.unpack_from("<3f", buf, 64)), time=struct.unpack_from("<f", buf, 76)[0])

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


_VALID_FILTERS = frozenset({"nearest", "linear"})
_VALID_ADDRESS_MODES = frozenset({"repeat", "clamp_to_edge", "mirror_repeat"})
_VALID_TEXTURE_FORMATS: dict[str, int] = {"rgba8_unorm": 4, "rgba32_sfloat": 16}


def _wrap_texel_index(index: int, size: int, mode: str) -> int:
    if mode == "repeat":
        return index % size
    if mode == "clamp_to_edge":
        return max(0, min(size - 1, index))
    if mode == "mirror_repeat":
        period = 2 * size
        m = index % period
        return m if m < size else (period - 1 - m)
    raise ValueError(f"Unsupported sampler address mode: {mode!r}")


class Sampler:
    """MiniGfx 2D texture sampler configuration shared between CPU and GPU."""

    def __init__(
        self,
        mag_filter: str = "linear",
        min_filter: str = "linear",
        address_mode_u: str = "repeat",
        address_mode_v: str = "repeat",
    ) -> None:
        if mag_filter not in _VALID_FILTERS:
            raise ValueError(f"Invalid mag_filter {mag_filter!r}; expected one of {sorted(_VALID_FILTERS)}")
        if min_filter not in _VALID_FILTERS:
            raise ValueError(f"Invalid min_filter {min_filter!r}; expected one of {sorted(_VALID_FILTERS)}")
        if address_mode_u not in _VALID_ADDRESS_MODES:
            raise ValueError(f"Invalid address_mode_u {address_mode_u!r}; expected one of {sorted(_VALID_ADDRESS_MODES)}")
        if address_mode_v not in _VALID_ADDRESS_MODES:
            raise ValueError(f"Invalid address_mode_v {address_mode_v!r}; expected one of {sorted(_VALID_ADDRESS_MODES)}")
        self.mag_filter = mag_filter
        self.min_filter = min_filter
        self.address_mode_u = address_mode_u
        self.address_mode_v = address_mode_v

    @classmethod
    def linear(cls, address_mode: str = "repeat") -> Sampler:
        return cls(mag_filter="linear", min_filter="linear", address_mode_u=address_mode, address_mode_v=address_mode)

    @classmethod
    def nearest(cls, address_mode: str = "repeat") -> Sampler:
        return cls(mag_filter="nearest", min_filter="nearest", address_mode_u=address_mode, address_mode_v=address_mode)

    @classmethod
    def clamp(cls, filter_mode: str = "linear") -> Sampler:
        return cls(
            mag_filter=filter_mode,
            min_filter=filter_mode,
            address_mode_u="clamp_to_edge",
            address_mode_v="clamp_to_edge",
        )

    def to_sampler_desc(self) -> Any:
        if SamplerDesc is None:
            raise RuntimeError("mlir_gpu is required to convert Sampler to SamplerDesc")
        return SamplerDesc(
            mag_filter=self.mag_filter,
            min_filter=self.min_filter,
            address_mode_u=self.address_mode_u,
            address_mode_v=self.address_mode_v,
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Sampler)
            and self.mag_filter == other.mag_filter
            and self.min_filter == other.min_filter
            and self.address_mode_u == other.address_mode_u
            and self.address_mode_v == other.address_mode_v
        )


class Texture:
    """MiniGfx 2D RGBA texture resource supporting CPU sampling and GPU upload."""

    def __init__(self, width: int, height: int, data: bytes | bytearray | memoryview, format: str = "rgba8_unorm") -> None:
        if type(width) is not int or width < 1:
            raise ValueError(f"Texture width must be a positive integer, got {width!r}")
        if type(height) is not int or height < 1:
            raise ValueError(f"Texture height must be a positive integer, got {height!r}")
        if format not in _VALID_TEXTURE_FORMATS:
            raise ValueError(f"Unsupported texture format {format!r}; expected one of {sorted(_VALID_TEXTURE_FORMATS)}")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError(f"Texture data must be bytes-like, got {type(data).__name__}")
        raw = bytes(data)
        expected = width * height * _VALID_TEXTURE_FORMATS[format]
        if len(raw) != expected:
            raise ValueError(f"Texture ({width}x{height}, {format}) expects {expected} bytes, got {len(raw)}")
        self.width = width
        self.height = height
        self.data = raw
        self.format = format

    @classmethod
    def from_texture_data(cls, tex_data: Any) -> Texture:
        return cls(width=tex_data.width, height=tex_data.height, data=tex_data.data, format=tex_data.format)

    @classmethod
    def from_rgba8(cls, width: int, height: int, pixels: Sequence[Sequence[int]] | bytes | bytearray | memoryview) -> Texture:
        if isinstance(pixels, (bytes, bytearray, memoryview)):
            return cls(width=width, height=height, data=bytes(pixels), format="rgba8_unorm")
        buf = bytearray()
        for rgba in pixels:
            if len(rgba) != 4:
                raise ValueError("Each RGBA8 pixel must contain 4 channel values")
            buf.extend(max(0, min(255, int(c))) for c in rgba)
        return cls(width=width, height=height, data=bytes(buf), format="rgba8_unorm")

    @classmethod
    def from_rgba_f32(cls, width: int, height: int, pixels: Sequence[Sequence[float]]) -> Texture:
        buf = bytearray()
        for rgba in pixels:
            if len(rgba) != 4:
                raise ValueError("Each RGBA32F pixel must contain 4 channel values")
            buf.extend(struct.pack("<4f", *(float(c) for c in rgba)))
        return cls(width=width, height=height, data=bytes(buf), format="rgba32_sfloat")

    @classmethod
    def solid(cls, color: Any = (1.0, 1.0, 1.0, 1.0), width: int = 1, height: int = 1, format: str = "rgba8_unorm") -> Texture:
        c = _to_vec4(color)
        count = width * height
        if format == "rgba8_unorm":
            pixel = bytes(max(0, min(255, int(round(ch * 255.0)))) for ch in (c.x, c.y, c.z, c.w))
            return cls(width=width, height=height, data=pixel * count, format="rgba8_unorm")
        if format == "rgba32_sfloat":
            pixel = struct.pack("<4f", c.x, c.y, c.z, c.w)
            return cls(width=width, height=height, data=pixel * count, format="rgba32_sfloat")
        raise ValueError(f"Unsupported texture format: {format!r}")

    @classmethod
    def checkerboard(
        cls,
        width: int = 8,
        height: int = 8,
        color_a: tuple[int, int, int, int] = (255, 255, 255, 255),
        color_b: tuple[int, int, int, int] = (0, 0, 0, 255),
        tile_size: int = 1,
    ) -> Texture:
        if type(tile_size) is not int or tile_size < 1:
            raise ValueError(f"tile_size must be a positive integer, got {tile_size!r}")
        buf = bytearray()
        for y in range(height):
            for x in range(width):
                even = ((x // tile_size) + (y // tile_size)) % 2 == 0
                buf.extend(color_a if even else color_b)
        return cls(width=width, height=height, data=bytes(buf), format="rgba8_unorm")

    def to_texture_data(self) -> Any:
        if TextureData is None:
            raise RuntimeError("mlir_gpu is required to convert Texture to TextureData")
        return TextureData(width=self.width, height=self.height, data=self.data, format=self.format)

    def _read_texel(self, x: int, y: int) -> _Vec4:
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return _Vec4(0.0, 0.0, 0.0, 0.0)
        idx = y * self.width + x
        if self.format == "rgba8_unorm":
            off = idx * 4
            r, g, b, a = self.data[off], self.data[off + 1], self.data[off + 2], self.data[off + 3]
            return _Vec4(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        off = idx * 16
        r, g, b, a = struct.unpack_from("<4f", self.data, off)
        return _Vec4(r, g, b, a)

    def load(self, x: int, y: int, level: int = 0) -> Vec4:
        if int(level) != 0:
            return _Vec4(0.0, 0.0, 0.0, 0.0)
        return self._read_texel(int(x), int(y))

    def load_cpu(self, coord: Any, level: Any = 0) -> Vec4:
        if hasattr(coord, "x") and hasattr(coord, "y"):
            cx, cy = int(coord.x), int(coord.y)
        elif isinstance(coord, (tuple, list)) and len(coord) == 2:
            cx, cy = int(coord[0]), int(coord[1])
        else:
            raise TypeError(f"Texture.load_cpu expects 2D integer coord, got {coord!r}")
        return self.load(cx, cy, int(level))

    def sample(self, sampler_or_uv: Sampler | Any | None = None, uv_or_sampler: Any = (0.0, 0.0)) -> Vec4:
        if (
            sampler_or_uv is not None
            and not isinstance(sampler_or_uv, Sampler)
            and not hasattr(sampler_or_uv, "mag_filter")
            and ((hasattr(sampler_or_uv, "x") and hasattr(sampler_or_uv, "y")) or isinstance(sampler_or_uv, (tuple, list)))
        ):
            uv = sampler_or_uv
            sampler = None if uv_or_sampler == (0.0, 0.0) else uv_or_sampler
        else:
            sampler = sampler_or_uv
            uv = uv_or_sampler

        if sampler is None:
            samp = Sampler()
        elif isinstance(sampler, Sampler):
            samp = sampler
        else:
            samp = Sampler(
                mag_filter=getattr(sampler, "mag_filter", "linear"),
                min_filter=getattr(sampler, "min_filter", "linear"),
                address_mode_u=getattr(sampler, "address_mode_u", "repeat"),
                address_mode_v=getattr(sampler, "address_mode_v", "repeat"),
            )

        if hasattr(uv, "x") and hasattr(uv, "y"):
            u, v = float(uv.x), float(uv.y)
        elif isinstance(uv, (tuple, list)) and len(uv) >= 2:
            u, v = float(uv[0]), float(uv[1])
        else:
            raise TypeError(f"Texture.sample expects 2D UV coordinates, got {uv!r}")

        if samp.mag_filter == "nearest":
            ix = _wrap_texel_index(int(math.floor(u * self.width)), self.width, samp.address_mode_u)
            iy = _wrap_texel_index(int(math.floor(v * self.height)), self.height, samp.address_mode_v)
            return self._read_texel(ix, iy)

        tx = u * self.width - 0.5
        ty = v * self.height - 0.5
        bx = int(math.floor(tx))
        by = int(math.floor(ty))
        fx = tx - bx
        fy = ty - by

        x0 = _wrap_texel_index(bx, self.width, samp.address_mode_u)
        x1 = _wrap_texel_index(bx + 1, self.width, samp.address_mode_u)
        y0 = _wrap_texel_index(by, self.height, samp.address_mode_v)
        y1 = _wrap_texel_index(by + 1, self.height, samp.address_mode_v)

        c00 = self._read_texel(x0, y0)
        c10 = self._read_texel(x1, y0)
        c01 = self._read_texel(x0, y1)
        c11 = self._read_texel(x1, y1)

        w00 = (1.0 - fx) * (1.0 - fy)
        w10 = fx * (1.0 - fy)
        w01 = (1.0 - fx) * fy
        w11 = fx * fy
        return _Vec4(
            c00.x * w00 + c10.x * w10 + c01.x * w01 + c11.x * w11,
            c00.y * w00 + c10.y * w10 + c01.y * w01 + c11.y * w11,
            c00.z * w00 + c10.z * w10 + c01.z * w01 + c11.z * w11,
            c00.w * w00 + c10.w * w10 + c01.w * w01 + c11.w * w11,
        )

    def sample_level(self, sampler: Sampler | Any | None = None, uv: Any = (0.0, 0.0), lod: float = 0.0) -> Vec4:
        _ = float(lod)
        return self.sample(sampler, uv)

    sample_cpu = sample
    sample_level_cpu = sample_level


class Material:
    """High-level MiniGfx material combining PBR parameters with optional 2D albedo texture & sampler."""

    def __init__(
        self,
        params: MaterialParams | None = None,
        *,
        albedo: Any = None,
        base_color: Any = None,
        roughness: float = 0.5,
        metallic: float = 0.0,
        ambient: float = 0.0,
        albedo_texture: Texture | None = None,
        sampler: Sampler | None = None,
        name: str = "",
    ) -> None:
        if params is not None:
            if not isinstance(params, MaterialParams):
                raise TypeError(f"params must be a MaterialParams instance, got {type(params).__name__}")
            self.params = params
        else:
            self.params = MaterialParams(
                albedo=albedo,
                base_color=base_color,
                roughness=roughness,
                metallic=metallic,
                ambient=ambient,
            )
        if albedo_texture is not None and not isinstance(albedo_texture, Texture):
            raise TypeError(f"albedo_texture must be a Texture instance, got {type(albedo_texture).__name__}")
        if sampler is not None and not isinstance(sampler, Sampler):
            raise TypeError(f"sampler must be a Sampler instance, got {type(sampler).__name__}")
        self.albedo_texture = albedo_texture
        self.sampler = sampler if sampler is not None else (Sampler() if albedo_texture is not None else None)
        self.name = str(name)

    @property
    def is_textured(self) -> bool:
        return self.albedo_texture is not None

    @property
    def albedo(self) -> Vec4:
        return self.params.albedo

    @albedo.setter
    def albedo(self, v: Any) -> None:
        self.params.albedo = v

    @property
    def base_color(self) -> Vec4:
        return self.params.base_color

    @base_color.setter
    def base_color(self, v: Any) -> None:
        self.params.base_color = v

    @property
    def roughness(self) -> float:
        return self.params.roughness

    @property
    def metallic(self) -> float:
        return self.params.metallic

    @property
    def ambient(self) -> float:
        return self.params.ambient

    def sample_albedo(self, uv: Any = (0.0, 0.0)) -> Vec4:
        base = self.params.albedo
        if self.albedo_texture is None:
            return base
        texel = self.albedo_texture.sample(self.sampler, uv)
        return _Vec4(base.x * texel.x, base.y * texel.y, base.z * texel.z, base.w * texel.w)

    def to_bytes(self) -> bytes:
        return self.params.to_bytes()

    def gpu_resources(
        self,
        texture_binding_name: str = "albedo_tex",
        sampler_binding_name: str = "albedo_sampler",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if self.albedo_texture is None:
            return {}, {}
        samp = self.sampler if self.sampler is not None else Sampler()
        return (
            {texture_binding_name: self.albedo_texture.to_texture_data()},
            {sampler_binding_name: samp.to_sampler_desc()},
        )


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

    def _next_set0_binding_offset(self, start: int = 0) -> int:
        if self._entry is not None and hasattr(self._entry, "_next_set0_binding_offset"):
            return self._entry._next_set0_binding_offset(start)
        return start

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        if self._entry is not None and hasattr(self._entry, "trace"): return self._entry.trace(*args, **kwargs)
        raise RuntimeError("Compiler unavailable for shader tracing")


def _cpu_vertex(pos: Any = (0.0, 0.0, 0.0), cam: Any = None, obj: Any = None) -> Vec4:
    model = getattr(obj, "model", Mat4x4.identity())
    vp = getattr(cam, "view_proj", Mat4x4.identity())
    return vp * (model * _to_vec4(pos))


def _cpu_fragment(normal: Any = (0.0, 1.0, 0.0), world_pos: Any = (0.0, 0.0, 0.0), uv: Any = (0.0, 0.0),
                  cam: Any = None, mat: Any = None, light: Any = None) -> Vec4:
    c = cam if cam is not None else CameraParams()
    m = mat if mat is not None else MaterialParams()
    l = light if light is not None else LightParams()
    if isinstance(m, Material) and m.albedo_texture is not None:
        sampled = m.sample_albedo(uv)
        effective_mat = MaterialParams(albedo=sampled, roughness=m.roughness, metallic=m.metallic, ambient=m.ambient)
    elif isinstance(m, Material):
        effective_mat = m.params
    else:
        effective_mat = m
    lit = evaluate_lighting(normal, c.eye_pos - _to_vec3(world_pos), l, effective_mat)
    final = tonemap_and_gamma(Vec3(*lit))
    return Vec4(final.x, final.y, final.z, effective_mat.albedo.w)


def _cpu_textured_fragment(
    normal: Any = (0.0, 1.0, 0.0),
    world_pos: Any = (0.0, 0.0, 0.0),
    uv: Any = (0.0, 0.0),
    albedo_tex: Texture | None = None,
    albedo_sampler: Sampler | None = None,
    cam: Any = None,
    mat: Any = None,
    light: Any = None,
) -> Vec4:
    c = cam if cam is not None else CameraParams()
    raw_mat = mat if mat is not None else MaterialParams()
    base_params = raw_mat.params if isinstance(raw_mat, Material) else raw_mat
    tex = albedo_tex if albedo_tex is not None else getattr(raw_mat, "albedo_texture", None)
    samp = albedo_sampler if albedo_sampler is not None else getattr(raw_mat, "sampler", None)
    if tex is not None:
        t_col = tex.sample(samp, uv)
        mod_albedo = Vec4(
            base_params.albedo.x * t_col.x,
            base_params.albedo.y * t_col.y,
            base_params.albedo.z * t_col.z,
            base_params.albedo.w * t_col.w,
        )
        effective_mat = MaterialParams(
            albedo=mod_albedo,
            roughness=base_params.roughness,
            metallic=base_params.metallic,
            ambient=base_params.ambient,
        )
    else:
        effective_mat = base_params
    l = light if light is not None else LightParams()
    lit = evaluate_lighting(normal, c.eye_pos - _to_vec3(world_pos), l, effective_mat)
    final = tonemap_and_gamma(Vec3(*lit))
    return Vec4(final.x, final.y, final.z, effective_mat.albedo.w)


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

    @fragment
    def _gpu_textured_fragment(
        normal: Value,
        world_pos: Value,
        uv: Value,
        albedo_tex: GpuTexture2D,
        albedo_sampler: GpuSampler,
        cam: CameraParams,
        mat: MaterialParams,
        light: LightParams,
    ) -> Value:
        texel = sample_texture(albedo_tex, albedo_sampler, uv)
        base_albedo = mat.albedo * texel
        n, l = sm.normalize(normal), sm.normalize(light.direction)
        n_dot_l = sm.clamp(sm.dot(n, l), 0.0, 1.0)
        diffuse = base_albedo.xyz * n_dot_l * light.intensity * (1.0 - mat.metallic)
        ambient = base_albedo.xyz * (light.ambient + mat.ambient)
        v = sm.normalize(cam.eye_pos - world_pos)
        h = sm.normalize(l + v)
        n_dot_h = sm.clamp(sm.dot(n, h), 0.0, 1.0)
        specular = sm.pow(n_dot_h, 2.0 / (mat.roughness * mat.roughness + 0.001)) * light.intensity
        hdr = ambient + diffuse + vec3(specular, specular, specular)
        ldr = sm.clamp((hdr * (hdr * 2.51 + 0.03)) / (hdr * (hdr * 2.43 + 0.59) + 0.14), 0.0, 1.0)
        srgb = sm.pow(ldr, 1.0 / 2.2)
        return vec4(srgb, base_albedo.w)

    pbr_vertex = ShaderFunction(_gpu_vertex, "vertex", _cpu_vertex)
    pbr_fragment = ShaderFunction(_gpu_fragment, "fragment", _cpu_fragment)
    textured_pbr_fragment = ShaderFunction(_gpu_textured_fragment, "fragment", _cpu_textured_fragment)
else:
    pbr_vertex = ShaderFunction(_cpu_vertex, "vertex", _cpu_vertex)
    pbr_fragment = ShaderFunction(_cpu_fragment, "fragment", _cpu_fragment)
    textured_pbr_fragment = ShaderFunction(_cpu_textured_fragment, "fragment", _cpu_textured_fragment)

material_vertex_shader, material_fragment_shader = pbr_vertex, pbr_fragment
default_vs, default_ps = pbr_vertex, pbr_fragment

__all__ = [
    "CameraParams", "LightParams", "Material", "MaterialParams", "ObjectParams", "Sampler",
    "ShaderFunction", "Texture", "default_ps", "default_vs", "evaluate_diffuse_lighting",
    "evaluate_lighting", "evaluate_specular", "material_fragment_shader", "material_vertex_shader",
    "pbr_fragment", "pbr_vertex", "textured_pbr_fragment",
]
