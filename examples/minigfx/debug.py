"""Procedural bitmap glyph debug overlay for the MiniGfx engine."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from typing import Any, Optional, Union

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

    def __init__(self, backend_name: str = "") -> None:
        self.backend_name = backend_name
        self.fps, self.frame_time_ms, self.draw_calls, self.visible_meshes = 0.0, 0.0, 0, 0

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
