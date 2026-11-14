"""NumPy-free portable GPU array example package on GfxIR."""

from __future__ import annotations

from .array import GfxArray, array, empty
from .dtypes import DType, float32, resolve_dtype
from .runtime import ComputeBackend, ComputeBuffer, Device, current_device, device
from .targets import (
    ArrayClosedError,
    BackendInitializationError,
    BackendNotImplementedError,
    CrossDeviceError,
    DeviceClosedError,
    GfxPyError,
    ShapeMismatchError,
    UnsupportedDTypeError,
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
    "UnsupportedTargetError",
    "array",
    "current_device",
    "default_target",
    "device",
    "empty",
    "float32",
    "normalize_target",
    "resolve_dtype",
]
