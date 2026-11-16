"""NumPy-free portable GPU array example package on GfxIR."""

from __future__ import annotations

from .array import GfxArray, array, empty, matmul, sum  # noqa: A004
from .dtypes import DType, float32, resolve_dtype
from .fusion import fuse
from .limits import (
    FLOAT32_EPSILON,
    FLOAT32_MAX,
    FLOAT32_MIN_NORMAL,
    I32_MAX,
    ComputeLimits,
    coerce_float32,
)
from .runtime import (
    ComputeBackend,
    ComputeBuffer,
    Device,
    close_default_device,
    current_device,
    device,
)
from .targets import (
    ArrayClosedError,
    BackendInitializationError,
    BackendNotImplementedError,
    CrossDeviceError,
    DeviceClosedError,
    GfxPyError,
    IndexingLimitError,
    ShapeMismatchError,
    UnsupportedDTypeError,
    UnsupportedFusionError,
    UnsupportedTargetError,
    default_target,
    normalize_target,
)

__all__ = [
    "ArrayClosedError",
    "BackendInitializationError",
    "BackendNotImplementedError",
    "ComputeBackend",
    "ComputeBuffer",
    "ComputeLimits",
    "CrossDeviceError",
    "DType",
    "Device",
    "DeviceClosedError",
    "FLOAT32_EPSILON",
    "FLOAT32_MAX",
    "FLOAT32_MIN_NORMAL",
    "GfxArray",
    "GfxPyError",
    "I32_MAX",
    "IndexingLimitError",
    "ShapeMismatchError",
    "UnsupportedDTypeError",
    "UnsupportedFusionError",
    "UnsupportedTargetError",
    "array",
    "close_default_device",
    "coerce_float32",
    "current_device",
    "default_target",
    "device",
    "empty",
    "float32",
    "fuse",
    "matmul",
    "normalize_target",
    "resolve_dtype",
    "sum",
]
