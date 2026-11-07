"""Small, pure distance-field helpers inlined into device expressions."""

from collections.abc import Callable

from . import shader_math as sm
from .frontend import Type, Value, maximum, minimum, select, vec3


def sphere(point: Value, radius: Value | float) -> Value:
    sm._vector(point, 3)
    size = sm._floating(radius)
    if size.node.type != Type.F32:
        raise TypeError("Sphere radius must be an f32 scalar.")
    _nonnegative(size, "Sphere radius")
    return sm.length(point) - size


def _nonnegative(value: Value, label: str) -> None:
    for component in sm._known_components(value.node):
        if component is not None and component < 0:
            raise ValueError(f"{label} must be nonnegative.")


def box(point: Value, half_size: Value) -> Value:
    sm._vector(point, 3)
    sm._vector(half_size, 3)
    _nonnegative(half_size, "Box half-size")
    q = abs(point) - half_size
    return sm.length(maximum(q, 0.0)) + minimum(maximum(q.x, maximum(q.y, q.z)), 0.0)


def box2(point: Value, half_size: Value) -> Value:
    sm._vector(point, 2)
    sm._vector(half_size, 2)
    _nonnegative(half_size, "Box half-size")
    q = abs(point) - half_size
    return sm.length(maximum(q, 0.0)) + minimum(maximum(q.x, q.y), 0.0)


def repeat(point: Value, period: Value | float) -> Value:
    """Center each positive-period cell around zero, including negative coordinates."""
    size = sm._floating(period)
    return sm.mod(sm._floating(point) + size * 0.5, size) - size * 0.5


def fold_square(point: Value) -> Value:
    p = abs(sm._vector(point, 2))
    return select(p.y > p.x, p.yx, p)


def normal(distance: Callable[[Value], Value], point: Value, epsilon: Value | float = 0.002) -> Value:
    sm._vector(point, 3)
    h = sm._floating(epsilon)
    if h.node.type != Type.F32:
        raise TypeError("Normal epsilon must be an f32 scalar.")
    for component in sm._known_components(h.node):
        if component is not None and component <= 0:
            raise ValueError("Normal epsilon must be positive.")
    gradient = vec3(0.0)
    for direction in (vec3(1, -1, -1), vec3(-1, -1, 1), vec3(-1, 1, -1), vec3(1)):
        sample = distance(point + direction * h)
        if not isinstance(sample, Value) or sample.node.type != Type.F32:
            raise TypeError("Normal estimation requires an f32 scalar distance function.")
        gradient = gradient + direction * sample
    return sm.normalize(gradient)
