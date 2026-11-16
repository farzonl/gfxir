"""Numerical precision, indexing, and backend resource limit contracts for gfxpy."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from .targets import IndexingLimitError

I32_MAX: int = (1 << 31) - 1
U32_MAX: int = (1 << 32) - 1
FLOAT32_MAX: float = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
FLOAT32_MIN_NORMAL: float = struct.unpack("<f", b"\x00\x00\x80\x00")[0]
FLOAT32_EPSILON: float = 2.0**-23

_D3D12_MAX_WORKGROUP_COUNT = (65535, 65535, 65535)
_VULKAN_CONSERVATIVE_WORKGROUP_COUNT = (65535, 65535, 65535)
_METAL_DEFAULT_WORKGROUP_COUNT = (U32_MAX, U32_MAX, U32_MAX)


@dataclass(frozen=True)
class ComputeLimits:
    """Supported element count, buffer byte, and dispatch grid limits for a backend."""

    max_elements: int = I32_MAX
    max_buffer_bytes: int | None = None
    max_workgroup_count: tuple[int, int, int] = (U32_MAX, U32_MAX, U32_MAX)


def default_limits_for_target(target: str) -> ComputeLimits:
    """Return conservative default limits for a target backend."""
    if target == "directx":
        return ComputeLimits(
            max_elements=I32_MAX,
            max_buffer_bytes=None,
            max_workgroup_count=_D3D12_MAX_WORKGROUP_COUNT,
        )
    if target == "vulkan":
        return ComputeLimits(
            max_elements=I32_MAX,
            max_buffer_bytes=None,
            max_workgroup_count=_VULKAN_CONSERVATIVE_WORKGROUP_COUNT,
        )
    if target == "metal":
        return ComputeLimits(
            max_elements=I32_MAX,
            max_buffer_bytes=None,
            max_workgroup_count=_METAL_DEFAULT_WORKGROUP_COUNT,
        )
    return ComputeLimits()


def get_backend_limits(backend: object) -> ComputeLimits:
    """Inspect backend-reported ComputeLimits or fall back to target defaults."""
    limits = getattr(backend, "limits", None)
    if isinstance(limits, ComputeLimits):
        return limits
    target = getattr(backend, "target", None)
    if isinstance(target, str):
        return default_limits_for_target(target)
    return ComputeLimits()


def coerce_float32(value: int | float) -> float:
    """Convert a Python int/float to IEEE 754 float32, rejecting finite overflow.

    Finite values whose magnitude exceeds ``FLOAT32_MAX`` raise ``OverflowError``.
    Non-finite values (``inf``, ``-inf``, ``nan``) are preserved, and sub-minimum
    magnitudes underflow toward ``0.0`` according to IEEE 754 ``float32`` rounding.
    """
    try:
        as_float = float(value)
    except OverflowError as exc:
        raise OverflowError(
            f"Scalar value {value!r} overflows IEEE 754 float32 range "
            f"(max finite {FLOAT32_MAX:.7e})."
        ) from exc
    try:
        packed = struct.pack("<f", as_float)
    except OverflowError as exc:
        raise OverflowError(
            f"Scalar value {value!r} overflows IEEE 754 float32 range "
            f"(max finite {FLOAT32_MAX:.7e})."
        ) from exc
    return struct.unpack("<f", packed)[0]


def validate_array_allocation(
    shape: tuple[int, ...],
    itemsize: int,
    limits: ComputeLimits,
) -> int:
    """Validate array shape, i32 element count, and buffer byte limits before allocation."""
    for dim in shape:
        if dim > I32_MAX:
            raise IndexingLimitError(
                f"Array dimension {dim} in shape {shape} exceeds signed 32-bit limit ({I32_MAX})."
            )
    size = math.prod(shape)
    effective_max_elements = min(I32_MAX, limits.max_elements)
    if size > effective_max_elements:
        raise IndexingLimitError(
            f"Array element count {size} for shape {shape} exceeds supported limit "
            f"({effective_max_elements})."
        )
    nbytes = size * itemsize
    if limits.max_buffer_bytes is not None and nbytes > limits.max_buffer_bytes:
        raise IndexingLimitError(
            f"Array allocation {nbytes} bytes for shape {shape} exceeds backend "
            f"max_buffer_bytes ({limits.max_buffer_bytes})."
        )
    return size


def validate_1d_dispatch(
    length: int,
    workgroup_size_x: int,
    limits: ComputeLimits,
    *,
    op: str = "dispatch",
) -> tuple[int, int, int]:
    """Validate 1D element count, rounded-up i32 invocation index, and X grid limit."""
    if length <= 0:
        raise ValueError(f"{op} requires positive length, got {length}.")
    effective_max_elements = min(I32_MAX, limits.max_elements)
    if length > effective_max_elements:
        raise IndexingLimitError(
            f"{op} length {length} exceeds supported limit ({effective_max_elements})."
        )
    grid_x = (length + workgroup_size_x - 1) // workgroup_size_x
    max_gid_x = grid_x * workgroup_size_x - 1
    if max_gid_x > I32_MAX:
        raise IndexingLimitError(
            f"{op} rounded-up invocation index {max_gid_x} (grid_x={grid_x}, "
            f"workgroup_size_x={workgroup_size_x}) exceeds signed 32-bit limit ({I32_MAX})."
        )
    max_wg_x = limits.max_workgroup_count[0]
    if grid_x > max_wg_x:
        raise IndexingLimitError(
            f"{op} 1D dispatch grid_x={grid_x} (length={length}, wg_x={workgroup_size_x}) "
            f"exceeds backend max_workgroup_count[0] ({max_wg_x})."
        )
    return (grid_x, 1, 1)


def validate_matmul_dispatch(
    m: int,
    n: int,
    k: int,
    itemsize: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    limits: ComputeLimits,
) -> tuple[int, int, int]:
    """Validate 2D matmul dimensions, linear i32 offsets, output bytes, and 2D grid limits."""
    for name, dim in (("m", m), ("n", n), ("k", k)):
        if dim > I32_MAX:
            raise IndexingLimitError(
                f"matmul dimension {name}={dim} exceeds signed 32-bit limit ({I32_MAX})."
            )

    effective_max_elements = min(I32_MAX, limits.max_elements)
    for label, count in (("A", m * k), ("B", k * n), ("C", m * n)):
        if count > effective_max_elements:
            raise IndexingLimitError(
                f"matmul operand/output {label} element count {count} for ({m}, {k}) @ ({k}, {n}) "
                f"exceeds supported limit ({effective_max_elements})."
            )

    out_bytes = m * n * itemsize
    if limits.max_buffer_bytes is not None and out_bytes > limits.max_buffer_bytes:
        raise IndexingLimitError(
            f"matmul output allocation {out_bytes} bytes for shape ({m}, {n}) exceeds "
            f"backend max_buffer_bytes ({limits.max_buffer_bytes})."
        )

    if m == 0 or n == 0 or k == 0:
        return (0, 0, 0)

    # Verify max linear element offsets fit in signed i32 without wraparound.
    for label, max_offset in (
        ("A", (m - 1) * k + (k - 1)),
        ("B", (k - 1) * n + (n - 1)),
        ("C", (m - 1) * n + (n - 1)),
    ):
        if max_offset > I32_MAX:
            raise IndexingLimitError(
                f"matmul max linear offset {max_offset} in {label} exceeds signed 32-bit limit ({I32_MAX})."
            )

    num_tiles = (k + tile_k - 1) // tile_k
    max_k_idx = num_tiles * tile_k - 1
    if num_tiles > I32_MAX or max_k_idx > I32_MAX:
        raise IndexingLimitError(
            f"matmul K-tile index {max_k_idx} (k={k}, tile_k={tile_k}) exceeds "
            f"signed 32-bit limit ({I32_MAX})."
        )

    grid_x = (n + tile_n - 1) // tile_n
    grid_y = (m + tile_m - 1) // tile_m
    if grid_x * tile_n - 1 > I32_MAX or grid_y * tile_m - 1 > I32_MAX:
        raise IndexingLimitError(
            f"matmul rounded-up tile coordinate exceeds signed 32-bit limit ({I32_MAX}) "
            f"for grid ({grid_x}, {grid_y}) and tile ({tile_n}, {tile_m})."
        )

    max_wg_x, max_wg_y, _ = limits.max_workgroup_count
    if grid_x > max_wg_x or grid_y > max_wg_y:
        raise IndexingLimitError(
            f"matmul dispatch grid ({grid_x}, {grid_y}, 1) for ({m}, {n}) exceeds "
            f"backend max_workgroup_count ({limits.max_workgroup_count})."
        )
    return (grid_x, grid_y, 1)
