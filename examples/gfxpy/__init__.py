"""NumPy-free portable GPU array example package on GfxIR."""

from __future__ import annotations

from .array import GfxArray, array, empty, matmul, sum  # noqa: A004
from .dtypes import DType, float32, resolve_dtype
from .fusion import fuse
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
    "CrossDeviceError",
    "DType",
    "Device",
    "DeviceClosedError",
    "GfxArray",
    "GfxPyError",
    "ShapeMismatchError",
    "UnsupportedDTypeError",
    "UnsupportedFusionError",
    "UnsupportedTargetError",
    "array",
    "close_default_device",
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
