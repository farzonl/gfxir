"""1D contiguous float32 device-resident array contract for gfxpy."""

from __future__ import annotations

from collections.abc import Sequence
import struct
from typing import Any

from .dtypes import DType, float32, resolve_dtype
from .runtime import ComputeBuffer, Device, current_device
from .targets import ArrayClosedError, CrossDeviceError, ShapeMismatchError


def _is_scalar_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_1d_float_sequence(data: object) -> list[float]:
    """Validate that data is a 1D Python sequence of real numbers."""
    if isinstance(data, (str, bytes, bytearray, memoryview)):
        raise TypeError(
            f"Expected a 1D sequence of numbers, not {type(data).__name__}."
        )
    if not isinstance(data, Sequence):
        raise TypeError(
            f"Expected a 1D Python sequence (list, tuple, range), got {type(data).__name__}."
        )
    values: list[float] = []
    for idx, item in enumerate(data):
        if isinstance(item, (list, tuple, Sequence)) and not isinstance(item, (str, bytes)):
            raise ShapeMismatchError(
                f"Only 1D contiguous sequences are supported; found nested sequence at index {idx}."
            )
        if not _is_scalar_number(item):
            raise TypeError(
                f"Expected numeric element at index {idx}, got {type(item).__name__}."
            )
        values.append(float(item))
    return values


class GfxArray:
    """Contiguous 1D float32 array backed by a persistent device buffer."""

    def __init__(
        self,
        *,
        size: int,
        dtype: DType,
        device: Device,
        buffer: ComputeBuffer | None,
    ) -> None:
        self._size = size
        self._dtype = dtype
        self._device = device
        self._buffer = buffer
        self._closed = False

    @property
    def shape(self) -> tuple[int]:
        return (self._size,)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def size(self) -> int:
        return self._size

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def device(self) -> Device:
        return self._device

    @property
    def nbytes(self) -> int:
        return self._size * self._dtype.itemsize

    @property
    def closed(self) -> bool:
        return self._closed or self._device.closed or (
            self._buffer is not None and self._buffer.closed
        )

    def _ensure_open(self) -> None:
        if self._closed:
            raise ArrayClosedError("GfxArray has been closed.")
        self._device.ensure_open()
        if self._buffer is not None and self._buffer.closed:
            raise ArrayClosedError("GfxArray backing buffer has been closed.")

    def close(self) -> None:
        """Release the underlying device buffer."""
        if not self._closed:
            self._closed = True
            if self._buffer is not None and not self._buffer.closed:
                self._buffer.close()

    def __enter__(self) -> GfxArray:
        self._ensure_open()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def tolist(self) -> list[float]:
        """Synchronize and download the array contents to a Python list of floats."""
        self._ensure_open()
        if self._size == 0:
            return []
        assert self._buffer is not None
        raw = self._device.backend.download(self._buffer)
        expected_bytes = self._size * self._dtype.itemsize
        if len(raw) != expected_bytes:
            raise RuntimeError(
                f"Backend '{self._device.target}' returned {len(raw)} bytes, expected {expected_bytes}."
            )
        return list(struct.unpack(f"<{self._size}{self._dtype.struct_code}", raw))

    def _binary_or_scalar(self, op: str, other: object) -> Any:
        self._ensure_open()
        if isinstance(other, GfxArray):
            other._ensure_open()
            if self._device is not other._device:
                raise CrossDeviceError(
                    f"Cannot combine arrays on different devices: {self._device!r} vs {other._device!r}."
                )
            if self._size != other._size:
                raise ShapeMismatchError(
                    f"Incompatible 1D lengths for '{op}': {self._size} vs {other._size}."
                )
            if self._size == 0:
                return GfxArray(size=0, dtype=self._dtype, device=self._device, buffer=None)
            assert self._buffer is not None and other._buffer is not None
            backend = self._device.backend
            out_buf = backend.allocate(self.nbytes)
            try:
                backend.binary_op(op, self._buffer, other._buffer, out_buf, self._size)
            except Exception:
                out_buf.close()
                raise
            return GfxArray(size=self._size, dtype=self._dtype, device=self._device, buffer=out_buf)

        if _is_scalar_number(other):
            if self._size == 0:
                return GfxArray(size=0, dtype=self._dtype, device=self._device, buffer=None)
            assert self._buffer is not None
            backend = self._device.backend
            out_buf = backend.allocate(self.nbytes)
            try:
                backend.scalar_op(op, self._buffer, float(other), out_buf, self._size)
            except Exception:
                out_buf.close()
                raise
            return GfxArray(size=self._size, dtype=self._dtype, device=self._device, buffer=out_buf)

        return NotImplemented

    def __add__(self, other: object) -> Any:
        return self._binary_or_scalar("add", other)

    def __radd__(self, other: object) -> Any:
        return self._binary_or_scalar("add", other)

    def __mul__(self, other: object) -> Any:
        return self._binary_or_scalar("mul", other)

    def __rmul__(self, other: object) -> Any:
        return self._binary_or_scalar("mul", other)

    def __repr__(self) -> str:
        return (
            f"GfxArray(shape={self.shape}, dtype={self._dtype.name}, "
            f"target={self._device.target!r}, closed={self.closed})"
        )


def array(
    data: Sequence[float | int],
    dtype: DType | str | None = float32,
    *,
    device: Device | None = None,
) -> GfxArray:
    """Create a 1D contiguous float32 GfxArray from a Python sequence."""
    resolved_dtype = resolve_dtype(dtype)
    values = _validate_1d_float_sequence(data)
    target_device = current_device() if device is None else device
    target_device.ensure_open()

    n = len(values)
    if n == 0:
        return GfxArray(size=0, dtype=resolved_dtype, device=target_device, buffer=None)

    raw = struct.pack(f"<{n}{resolved_dtype.struct_code}", *values)
    backend = target_device.backend
    buf = backend.allocate(len(raw))
    try:
        backend.upload(buf, raw)
    except Exception:
        buf.close()
        raise
    return GfxArray(size=n, dtype=resolved_dtype, device=target_device, buffer=buf)


def empty(
    length: int,
    dtype: DType | str | None = float32,
    *,
    device: Device | None = None,
) -> GfxArray:
    """Allocate an uninitialized 1D contiguous float32 GfxArray of the given length."""
    if isinstance(length, bool) or not isinstance(length, int):
        raise TypeError(f"Array length must be an int, got {type(length).__name__}.")
    if length < 0:
        raise ValueError(f"Array length must be non-negative, got {length}.")

    resolved_dtype = resolve_dtype(dtype)
    target_device = current_device() if device is None else device
    target_device.ensure_open()

    if length == 0:
        return GfxArray(size=0, dtype=resolved_dtype, device=target_device, buffer=None)

    buf = target_device.backend.allocate(length * resolved_dtype.itemsize)
    return GfxArray(size=length, dtype=resolved_dtype, device=target_device, buffer=buf)
