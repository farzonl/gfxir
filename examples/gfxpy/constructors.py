"""GPU-backed float32 array construction and copying."""

from __future__ import annotations

from .array import GfxArray, _is_scalar_number, _normalize_shape, array, empty
from .dtypes import DType, float32, resolve_dtype
from .kernels import WORKGROUP_SIZE_1D
from .limits import coerce_float32, get_backend_limits, validate_1d_dispatch, validate_array_allocation
from .runtime import Device, current_device
from .targets import CrossDeviceError


def _output(
    shape: int | tuple[int, ...], dtype: DType | str | None, device: Device | None,
) -> GfxArray:
    resolved_shape = _normalize_shape(shape)
    resolved_dtype = resolve_dtype(dtype)
    dev = current_device() if device is None else device
    dev.ensure_open()
    limits = get_backend_limits(dev.backend)
    size = validate_array_allocation(resolved_shape, resolved_dtype.itemsize, limits)
    if size:
        validate_1d_dispatch(size, WORKGROUP_SIZE_1D[0], limits, op="constructor")
    return empty(resolved_shape, resolved_dtype, device=dev)


def full(
    shape: int | tuple[int, ...],
    fill_value: int | float,
    dtype: DType | str | None = float32,
    *,
    device: Device | None = None,
) -> GfxArray:
    """Create a float32 array without materializing a host-sized fill payload."""
    if not _is_scalar_number(fill_value):
        raise TypeError("full() fill_value must be a real int or float, not a bool.")
    value = coerce_float32(fill_value)
    result = _output(shape, dtype, device)
    if result.size:
        assert result._buffer is not None
        try:
            result.device.backend.fill(result._buffer, value, result.size)
        except Exception:
            result.close()
            raise
    return result


def zeros(
    shape: int | tuple[int, ...], dtype: DType | str | None = float32, *, device: Device | None = None,
) -> GfxArray:
    """Create a device array filled with zeros."""
    return full(shape, 0.0, dtype, device=device)


def ones(
    shape: int | tuple[int, ...], dtype: DType | str | None = float32, *, device: Device | None = None,
) -> GfxArray:
    """Create a device array filled with ones."""
    return full(shape, 1.0, dtype, device=device)


def _require_array(a: GfxArray) -> None:
    if not isinstance(a, GfxArray):
        raise TypeError(f"Expected a GfxArray, got {type(a).__name__}.")
    a._ensure_open()


def full_like(
    a: GfxArray, fill_value: int | float, dtype: DType | str | None = None, *, device: Device | None = None,
) -> GfxArray:
    """Fill an independent array with the input shape and, by default, device."""
    _require_array(a)
    return full(
        a.shape, fill_value, a.dtype if dtype is None else dtype,
        device=a.device if device is None else device,
    )


def zeros_like(
    a: GfxArray, dtype: DType | str | None = None, *, device: Device | None = None,
) -> GfxArray:
    """Create zeros with a device array's shape."""
    return full_like(a, 0.0, dtype, device=device)


def ones_like(
    a: GfxArray, dtype: DType | str | None = None, *, device: Device | None = None,
) -> GfxArray:
    """Create ones with a device array's shape."""
    return full_like(a, 1.0, dtype, device=device)


def asarray(
    a: object, dtype: DType | str | None = None, *, device: Device | None = None,
) -> GfxArray:
    """Reuse a compatible GfxArray, or upload an explicit Python sequence."""
    if isinstance(a, GfxArray):
        a._ensure_open()
        resolve_dtype(a.dtype if dtype is None else dtype)
        if device is not None:
            device.ensure_open()
            if device is not a.device:
                raise CrossDeviceError("asarray() cannot transfer between devices; no host fallback is performed.")
        return a
    return array(a, dtype, device=device)


def copy(a: object, *, device: Device | None = None) -> GfxArray:
    """Create an independent device copy, or upload a Python sequence."""
    if not isinstance(a, GfxArray):
        return array(a, device=device)
    source = asarray(a, device=device)
    result = _output(source.shape, source.dtype, source.device)
    if source.size:
        assert source._buffer is not None and result._buffer is not None
        try:
            source.device.backend.copy_buffer(source._buffer, result._buffer, source.size)
        except Exception:
            result.close()
            raise
    return result
