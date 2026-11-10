"""Inlined ACES filmic tonemapper and sRGB gamma transfer functions.

Eliminates intermediate HDR render targets and compute passes by inlining
tonemapping and perceptual gamma conversion directly into fragment shaders.
Polymorphic across float, int, Vec3, and Vec4 (preserving alpha unchanged).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Union

try:
    from .math import Vec3, Vec4
except (ImportError, ValueError):
    try:
        from examples.minigfx.math import Vec3, Vec4
    except (ImportError, ValueError):
        # Fallback minimal vector types when minigfx.math is not yet installed
        class _FallbackVec:  # type: ignore[no-redef]
            __slots__ = ()

            def __len__(self) -> int:
                return len(self.__slots__)

            def __iter__(self):
                for s in self.__slots__:
                    yield getattr(self, s)

            def __eq__(self, o: object) -> bool:
                if isinstance(o, (type(self), tuple, list)) and len(o) == len(self.__slots__):
                    return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(self, o))
                return False

        class Vec3(_FallbackVec):  # type: ignore[no-redef]
            __slots__ = ("x", "y", "z")

            def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
                self.x, self.y, self.z = float(x), float(y), float(z)

            def __repr__(self) -> str:
                return f"Vec3({self.x}, {self.y}, {self.z})"

        class Vec4(_FallbackVec):  # type: ignore[no-redef]
            __slots__ = ("x", "y", "z", "w")

            def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
                self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)

            def __repr__(self) -> str:
                return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"

ColorType = Union[float, int, Vec3, Vec4]


def _scalar_aces(x: float) -> float:
    """Narkowicz ACES filmic curve with pre-clamping and NaN/overflow guards."""
    if math.isnan(x) or x <= 0.0:
        return 0.0
    if x > 1e4:
        return 1.0
    val = (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14)
    return 1.0 if val > 1.0 else (0.0 if val < 0.0 else val)


def _scalar_linear_to_srgb(x: float, precise: bool = True) -> float:
    """Convert linear luminance to sRGB gamma space."""
    if math.isnan(x) or x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if not precise:
        return x ** (1.0 / 2.2)
    return 12.92 * x if x <= 0.0031308 else min(1.0, 1.055 * (x ** (1.0 / 2.4)) - 0.055)


def _scalar_srgb_to_linear(x: float, precise: bool = True) -> float:
    """Convert sRGB gamma space to linear luminance."""
    if math.isnan(x) or x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if not precise:
        return x ** 2.2
    return x / 12.92 if x <= 0.04045 else min(1.0, ((x + 0.055) / 1.055) ** 2.4)


def _map_rgb(fn: Callable[..., float], color: Any, *args: Any, **kwargs: Any) -> Any:
    """Higher-order polymorphic dispatch helper over scalar, Vec3, and Vec4."""
    if isinstance(color, bool):
        raise TypeError("Unsupported color type bool; expected float, int, Vec3, or Vec4")
    if isinstance(color, (int, float)):
        return fn(float(color), *args, **kwargs)
    if isinstance(color, Vec3):
        return Vec3(fn(color.x, *args, **kwargs), fn(color.y, *args, **kwargs), fn(color.z, *args, **kwargs))
    if isinstance(color, Vec4):
        return Vec4(fn(color.x, *args, **kwargs), fn(color.y, *args, **kwargs), fn(color.z, *args, **kwargs), color.w)
    raise TypeError(f"Unsupported color type {type(color).__name__}; expected float, int, Vec3, or Vec4")


def aces_tonemap(color: ColorType) -> ColorType:
    """Apply Krzysztof Narkowicz ACES filmic tonemapping curve clamped to [0.0, 1.0]."""
    return _map_rgb(_scalar_aces, color)


def linear_to_srgb(color: ColorType, precise: bool = True) -> ColorType:
    """Convert linear RGB to sRGB gamma space (IEC 61966-2-1 piecewise or power-law)."""
    return _map_rgb(_scalar_linear_to_srgb, color, precise=precise)


def srgb_to_linear(color: ColorType, precise: bool = True) -> ColorType:
    """Convert sRGB gamma space to linear RGB."""
    return _map_rgb(_scalar_srgb_to_linear, color, precise=precise)


def srgb_gamma(color: ColorType) -> ColorType:
    """Fast sRGB gamma approximation (color ** (1.0 / 2.2))."""
    return linear_to_srgb(color, precise=False)


def tonemap_and_gamma(color: ColorType, precise: bool = True) -> ColorType:
    """Apply ACES filmic tonemapping followed by sRGB gamma correction."""
    return linear_to_srgb(aces_tonemap(color), precise=precise)
