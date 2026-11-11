"""Procedural bitmap glyph debug overlay for the MiniGfx engine."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from dataclasses import field
import math
from typing import Any, Optional, Union
import struct
from typing import Callable

from examples.minigfx.math import Vec2, Vec4

try:
    from mlir_gpu.frontend import Type, Value, Vertex, fragment, vec2, vec3, vec4, vertex
    import mlir_gpu.shader_math as sm
except ImportError:
    Type = Value = Vertex = fragment = vec2 = vec3 = vec4 = vertex = sm = None

if Vertex is None:
    @dataclass(frozen=True)
    class Vertex:
        position: Any
        color: Any = None
        varyings: dict[str, Any] = field(default_factory=dict)


def _to_vec2(v: Any) -> Vec2:
    if isinstance(v, Vec2):
        return v
    if hasattr(v, "x") and hasattr(v, "y"):
        return Vec2(v.x, v.y)
    if isinstance(v, (tuple, list)) and len(v) >= 2:
        return Vec2(v[0], v[1])
    raise TypeError(f"Cannot convert {v!r} to Vec2")


def _to_vec4(v: Any) -> Vec4:
    if isinstance(v, Vec4):
        return v
    if hasattr(v, "x") and hasattr(v, "y") and hasattr(v, "z"):
        return Vec4(v.x, v.y, v.z, getattr(v, "w", 1.0))
    if isinstance(v, (tuple, list)) and len(v) in (3, 4):
        return Vec4(v[0], v[1], v[2], 1.0 if len(v) == 3 else v[3])
    raise TypeError(f"Cannot convert {v!r} to Vec4")


class OverlayParams:
    """Shared host/device overlay parameters (std430, 48 bytes)."""

    __struct_size__, __struct_alignment__, _is_gfx_struct = 48, 16, True

    def __init__(
        self,
        text_color: Any = (1.0, 1.0, 1.0, 1.0),
        bg_color: Any = (0.0, 0.0, 0.0, 0.0),
        viewport: Any = (1280.0, 720.0),
        scale: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise TypeError(f"OverlayParams unexpected kwargs: {list(kwargs.keys())}")
        self._text_color = _to_vec4(text_color)
        self._bg_color = _to_vec4(bg_color)
        self._viewport = _to_vec2(viewport)
        self.scale = float(scale)

    @property
    def text_color(self) -> Vec4:
        return self._text_color

    @text_color.setter
    def text_color(self, v: Any) -> None:
        self._text_color = _to_vec4(v)

    @property
    def bg_color(self) -> Vec4:
        return self._bg_color

    @bg_color.setter
    def bg_color(self, v: Any) -> None:
        self._bg_color = _to_vec4(v)

    @property
    def viewport(self) -> Vec2:
        return self._viewport

    @viewport.setter
    def viewport(self, v: Any) -> None:
        self._viewport = _to_vec2(v)

    def to_bytes(self) -> bytes:
        tc, bg, vp = self._text_color, self._bg_color, self._viewport
        return struct.pack(
            "<12f",
            tc.x, tc.y, tc.z, tc.w,
            bg.x, bg.y, bg.z, bg.w,
            vp.x, vp.y,
            self.scale, 0.0,
        )

    @classmethod
    def from_bytes(cls, buf: bytes | bytearray | memoryview) -> OverlayParams:
        if len(buf) < 48:
            raise ValueError(f"Buffer < 48B for OverlayParams (got {len(buf)})")
        tc0, tc1, tc2, tc3, bg0, bg1, bg2, bg3, vx, vy, sc, _pad = struct.unpack_from("<12f", buf, 0)
        return cls(
            text_color=Vec4(tc0, tc1, tc2, tc3),
            bg_color=Vec4(bg0, bg1, bg2, bg3),
            viewport=Vec2(vx, vy),
            scale=sc,
        )

    @classmethod
    def as_parameter_block(cls, name: str | None = None) -> Any:
        if Type is None:
            return None
        from mlir_gpu.parameters import ParameterBlock
        return ParameterBlock(
            name or "overlay",
            (
                ("text_color", Type.VEC4),
                ("bg_color", Type.VEC4),
                ("viewport", Type.VEC2),
                ("scale", Type.F32),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text_color": [self._text_color.x, self._text_color.y, self._text_color.z, self._text_color.w],
            "bg_color": [self._bg_color.x, self._bg_color.y, self._bg_color.z, self._bg_color.w],
            "viewport": [self._viewport.x, self._viewport.y],
            "scale": self.scale,
        }

    def __eq__(self, o: object) -> bool:
        return (
            isinstance(o, OverlayParams)
            and self._text_color == o._text_color
            and self._bg_color == o._bg_color
            and self._viewport == o._viewport
            and math.isclose(self.scale, o.scale, rel_tol=1e-5, abs_tol=1e-6)
        )


class ShaderFunction:
    """Callable shader wrapper supporting CPU evaluation and GPU AST pipeline tracing."""

    def __init__(self, entry: Any, kind: str, cpu_fn: Callable | None = None) -> None:
        self._entry, self.kind, self._cpu_fn = entry, kind, cpu_fn
        self.name = getattr(entry, "name", getattr(entry, "__name__", kind))
        self.function = getattr(entry, "function", entry)
        self.workgroup_size = getattr(entry, "workgroup_size", (1, 1, 1))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self._cpu_fn is not None:
            return self._cpu_fn(*args, **kwargs)
        if callable(self.function):
            return self.function(*args, **kwargs)
        raise TypeError(f"Shader {self.name} is not callable on CPU")

    def trace(self, *args: Any, **kwargs: Any) -> Any:
        if self._entry is not None and hasattr(self._entry, "trace"):
            return self._entry.trace(*args, **kwargs)
        raise RuntimeError("Compiler unavailable for shader tracing")


class OverlayVertex(Vertex):
    """CPU vertex representation providing coordinate properties."""

    @property
    def x(self) -> float:
        return float(self.position.x)

    @property
    def y(self) -> float:
        return float(self.position.y)

    @property
    def z(self) -> float:
        return float(self.position.z)

    @property
    def w(self) -> float:
        return float(self.position.w)

    @property
    def uv(self) -> Any:
        return self.varyings.get("uv", Vec2(0.0, 0.0))


def _cpu_overlay_vs(vert: Any = (0.0, 0.0), params: Any = None, **kwargs: Any) -> OverlayVertex:
    p = params if isinstance(params, OverlayParams) else (OverlayParams(**params) if isinstance(params, dict) else (params or OverlayParams()))
    vp = getattr(p, "viewport", Vec2(1280.0, 720.0))
    vw = float(vp.x) if float(vp.x) > 0.0 else 1.0
    vh = float(vp.y) if float(vp.y) > 0.0 else 1.0

    pos_in = kwargs.get("pos", vert)
    uv_in = kwargs.get("uv", None)

    if hasattr(pos_in, "position"):
        inner_pos = pos_in.position
        px = getattr(inner_pos, "x", inner_pos[0] if hasattr(inner_pos, "__getitem__") else 0.0)
        py = getattr(inner_pos, "y", inner_pos[1] if hasattr(inner_pos, "__getitem__") else 0.0)
        if uv_in is None and hasattr(pos_in, "varyings") and "uv" in pos_in.varyings:
            uv_in = pos_in.varyings["uv"]
    elif hasattr(pos_in, "x") and hasattr(pos_in, "y"):
        px, py = float(pos_in.x), float(pos_in.y)
    elif isinstance(pos_in, (tuple, list)) and len(pos_in) >= 2:
        px, py = float(pos_in[0]), float(pos_in[1])
        if len(pos_in) >= 4 and uv_in is None:
            uv_in = (float(pos_in[2]), float(pos_in[3]))
    else:
        px, py = 0.0, 0.0

    if uv_in is not None:
        if hasattr(uv_in, "x") and hasattr(uv_in, "y"):
            u, v = float(uv_in.x), float(uv_in.y)
        elif isinstance(uv_in, (tuple, list)) and len(uv_in) >= 2:
            u, v = float(uv_in[0]), float(uv_in[1])
        else:
            u, v = 0.0, 0.0
    else:
        u, v = 0.0, 0.0

    ndc_x = (px / vw) * 2.0 - 1.0
    ndc_y = 1.0 - (py / vh) * 2.0
    clip_pos = Vec4(ndc_x, ndc_y, 0.0, 1.0)
    return OverlayVertex(clip_pos, varyings={"uv": Vec2(u, v)})


def _cpu_overlay_ps(uv: Any = (0.0, 0.0), params: Any = None, char: str | None = None, **kwargs: Any) -> Vec4:
    p = params if isinstance(params, OverlayParams) else (OverlayParams(**params) if isinstance(params, dict) else (params or OverlayParams()))
    tc = _to_vec4(getattr(p, "text_color", Vec4(1.0, 1.0, 1.0, 1.0)))
    bg = _to_vec4(getattr(p, "bg_color", Vec4(0.0, 0.0, 0.0, 0.0)))
    if char is not None and char in GLYPH_TABLE:
        u_val = getattr(uv, "x", uv[0] if hasattr(uv, "__getitem__") else 0.0)
        v_val = getattr(uv, "y", uv[1] if hasattr(uv, "__getitem__") else 0.0)
        col_idx = int(u_val * 5.0)
        row_idx = int(v_val * 7.0)
        if 0 <= col_idx < 5 and 0 <= row_idx < 7:
            glyph = GLYPH_TABLE[char]
            if (glyph[col_idx] >> row_idx) & 1:
                return tc
            return bg
    return tc


# 5x7 column-major bitmap font spanning ASCII 32..95 (64 glyphs, 320 bytes).
_FONT_B64 = (
    "AAAAAAAAAF8AAAAHAAcAFH8UfxQkKn8qEiMTCGRiNklVIlAABQMAAAAcIkEAAEEiHAAUCD4IFAgIKggIAFAwAAAICAgICAB"
    "gYAAAIBAIBAI+UUlFPgBCf0AAQmFRSUYhQUVLMRgUEn8QJ0VFRTk8SklJMAFxCQUDNklJSTYGSUkpHgA2NgAAAFY2AAAIFCJ"
    "BABQUFBQUAEEiFAgCAVEJBjJJeUE+fhEREX5/SUlJNj5BQUEif0FBIhx/SUlJQX8JCQkBPkFJSXp/CAgIfwBBf0EAIEBBPwF"
    "/CBQiQX9AQEBAfwIMAn9/BAgQfz5BQUE+fwkJCQY+QVEhXn8JGSlGRklJSTEBAX8BAT9AQEA/HyBAIB8/QDhAP2MUCBRjBwh"
    "wCAdhUUlFQwB/QUEAAgQIECAAQUF/AAQCAQIEQEBAQEA="
)
_FONT_DATA = base64.b64decode(_FONT_B64)
GLYPH_TABLE: dict[str, tuple[int, ...]] = {
    chr(32 + i): tuple(_FONT_DATA[i * 5 : (i + 1) * 5]) for i in range(64)
}
GLYPH_TABLE["|"] = (0, 0, 127, 0, 0)
for _c in range(ord("a"), ord("z") + 1):
    GLYPH_TABLE[chr(_c)] = GLYPH_TABLE[chr(_c).upper()]


if vertex is not None and fragment is not None:
    @vertex
    def _gpu_overlay_vs(pos: Type.VEC2, uv: Type.VEC2, params: OverlayParams) -> Vertex:
        ndc_x = (pos.x / params.viewport.x) * 2.0 - 1.0
        ndc_y = 1.0 - (pos.y / params.viewport.y) * 2.0
        return Vertex(vec4(ndc_x, ndc_y, 0.0, 1.0), varyings={"uv": uv})

    @fragment
    def _gpu_overlay_ps(uv: Type.VEC2, params: OverlayParams) -> vec4:
        return params.text_color

    overlay_vertex_shader = ShaderFunction(_gpu_overlay_vs, "vertex", _cpu_overlay_vs)
    overlay_fragment_shader = ShaderFunction(_gpu_overlay_ps, "fragment", _cpu_overlay_ps)
else:
    overlay_vertex_shader = ShaderFunction(_cpu_overlay_vs, "vertex", _cpu_overlay_vs)
    overlay_fragment_shader = ShaderFunction(_cpu_overlay_ps, "fragment", _cpu_overlay_ps)

overlay_vs = overlay_vertex_shader
overlay_ps = overlay_fragment_shader


@dataclass(slots=True)
class TextQuad:
    """Screen-space 2D quad geometry for a single rendered character."""

    char: str
    positions: tuple[tuple[float, float], ...]
    uvs: tuple[tuple[float, float], ...]
    indices: tuple[int, ...]
    vertices: list[list[float]]


class TextQuadList(list):
    """List of TextQuad instances with helpers for GPU buffer extraction."""

    @property
    def vertices(self) -> list[list[float]]:
        return [v for q in self for v in q.vertices]

    @property
    def indices(self) -> list[int]:
        return [i for q in self for i in q.indices]

    def to_buffers(self) -> tuple[list[list[float]], list[int]]:
        return self.vertices, self.indices

    def as_mesh(self) -> tuple[list[list[float]], list[int]]:
        return self.to_buffers()


def get_overlay_pipeline_state() -> dict[str, Any]:
    """Return pipeline state overrides for rendering 2D debug overlay on top."""
    return {"depth_write": False, "depth_test": False, "cull_mode": "none"}


def compute_fps(dt: float) -> float:
    """Compute frames-per-second safely guarding against zero, negative, or NaN."""
    try:
        f = float(dt)
        return 1.0 / f if f > 1e-6 and not math.isnan(f) and not math.isinf(f) else 0.0
    except (ValueError, TypeError):
        return 0.0


def format_metrics(
    fps: float = 0.0, dt: float = 0.0, draw_calls: int = 0,
    visible_meshes: int = 0, backend_name: str = "", *, frame_time_ms: Optional[float] = None,
) -> str:
    """Format frame performance metrics into a readable status line."""
    dt_ms = frame_time_ms if frame_time_ms is not None else dt * 1000.0
    return (
        f"FPS: {fps:.1f} | Frame: {dt_ms:.1f}ms | Backend: {backend_name} | "
        f"Draws: {draw_calls} | Meshes: {visible_meshes}"
    )


def generate_text_quads(
    text: str, start_x: float = 0.0, start_y: float = 0.0, scale: float = 1.0,
    *, x: Optional[float] = None, y: Optional[float] = None,
    viewport_w: Optional[float] = None, viewport_h: Optional[float] = None, as_buffers: bool = False,
) -> Union[TextQuadList, tuple[list[list[float]], list[int]]]:
    """Generate 2D screen-space quads or vertex/index buffers for given text."""
    start_x = float(x) if x is not None else float(start_x)
    start_y = float(y) if y is not None else float(start_y)
    if not text or scale <= 0.0 or (
        viewport_w is not None and viewport_h is not None and (viewport_w <= 0 or viewport_h <= 0)
    ):
        return ([], []) if as_buffers else TextQuadList()

    w, h, adv_x, line_h = 5.0 * scale, 7.0 * scale, 6.0 * scale, 9.0 * scale
    cur_x, cur_y = start_x, start_y
    quads = TextQuadList()

    for ch in text:
        if ch == "\n":
            cur_x, cur_y = start_x, cur_y + line_h
            continue
        if ch == "\r":
            continue
        px0, px1 = cur_x, cur_x + w
        py0, py1 = cur_y, cur_y + h
        if viewport_w is not None and viewport_h is not None:
            vw, vh = float(viewport_w), float(viewport_h)
            px0, px1 = max(0.0, min(vw, px0)), max(0.0, min(vw, px1))
            py0, py1 = max(0.0, min(vh, py0)), max(0.0, min(vh, py1))

        base = 4 * len(quads)
        quads.append(
            TextQuad(
                char=ch,
                positions=((px0, py0), (px1, py0), (px1, py1), (px0, py1)),
                uvs=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
                indices=(base, base + 1, base + 2, base + 2, base + 3, base),
                vertices=[[px0, py0, 0.0, 0.0], [px1, py0, 1.0, 0.0], [px1, py1, 1.0, 1.0], [px0, py1, 0.0, 1.0]],
            )
        )
        cur_x += adv_x

    return quads.to_buffers() if as_buffers else quads


def draw_to_image(
    buffer: bytearray, width: int, height: int, text: str,
    x: int = 0, y: int = 0, color: tuple[int, ...] = (255, 255, 255, 255), scale: int = 1,
) -> None:
    """Software rasterizer blitting text into RGBA bytearray buffer with safe clipping."""
    if width <= 0 or height <= 0 or scale <= 0 or not text:
        return
    c_bytes, buf_len = bytes(color[:4]), len(buffer)
    cur_x, cur_y, adv_x, line_h = int(x), int(y), 6 * scale, 9 * scale

    for ch in text:
        if ch == "\n":
            cur_x, cur_y = int(x), cur_y + line_h
            continue
        if ch == "\r":
            continue
        glyph = GLYPH_TABLE.get(ch, GLYPH_TABLE["?"])
        for cx, col_mask in enumerate(glyph):
            for cy in range(7):
                if (col_mask >> cy) & 1:
                    for sy in range(scale):
                        py = cur_y + cy * scale + sy
                        if 0 <= py < height:
                            row_start = py * width * 4
                            for sx in range(scale):
                                px = cur_x + cx * scale + sx
                                if 0 <= px < width:
                                    idx = row_start + px * 4
                                    if idx + 4 <= buf_len:
                                        buffer[idx : idx + 4] = c_bytes
        cur_x += adv_x


class DebugOverlay:
    """Debug HUD tracking performance metrics and rendering text overlays."""

    def __init__(self, backend_name: str = "", params: Optional[OverlayParams] = None) -> None:
        self.backend_name = backend_name
        self.fps, self.frame_time_ms, self.draw_calls, self.visible_meshes = 0.0, 0.0, 0, 0
        self.params = params if params is not None else OverlayParams()

    def update_metrics(
        self, fps: float = 0.0, frame_time_ms: float = 0.0,
        draw_calls: int = 0, visible_meshes: int = 0, backend_name: Optional[str] = None,
    ) -> None:
        self.fps, self.frame_time_ms = float(fps), float(frame_time_ms)
        self.draw_calls, self.visible_meshes = int(draw_calls), int(visible_meshes)
        if backend_name is not None:
            self.backend_name = backend_name

    def format_metrics(self) -> str:
        return format_metrics(
            fps=self.fps, draw_calls=self.draw_calls, visible_meshes=self.visible_meshes,
            backend_name=self.backend_name, frame_time_ms=self.frame_time_ms,
        )

    def draw_to_image(
        self, buffer: bytearray, width: int, height: int, text: Optional[str] = None,
        x: int = 10, y: int = 10, color: tuple[int, ...] = (255, 255, 255, 255), scale: int = 1,
    ) -> None:
        msg = self.format_metrics() if text is None else text
        draw_to_image(buffer, width, height, msg, x=x, y=y, color=color, scale=scale)

    def generate_text_quads(
        self, text: Optional[str] = None, start_x: float = 10.0, start_y: float = 10.0, scale: float = 1.0,
        *, x: Optional[float] = None, y: Optional[float] = None,
        viewport_w: Optional[float] = None, viewport_h: Optional[float] = None, as_buffers: bool = False,
    ):
        msg = self.format_metrics() if text is None else text
        return generate_text_quads(
            msg, start_x=start_x, start_y=start_y, scale=scale,
            x=x, y=y, viewport_w=viewport_w, viewport_h=viewport_h, as_buffers=as_buffers,
        )


__all__ = [
    "GLYPH_TABLE",
    "DebugOverlay",
    "OverlayParams",
    "OverlayVertex",
    "ShaderFunction",
    "TextQuad",
    "TextQuadList",
    "compute_fps",
    "draw_to_image",
    "format_metrics",
    "generate_text_quads",
    "get_overlay_pipeline_state",
    "overlay_fragment_shader",
    "overlay_ps",
    "overlay_vertex_shader",
    "overlay_vs",
]
