"""1D and 2D contiguous float32 device-resident array contract for gfxpy."""

from __future__ import annotations

import array as _stdlib_array
from collections.abc import Sequence
import math
import struct
import sys
from typing import Any

from .dtypes import DType, float32, resolve_dtype
from .runtime import ComputeBuffer, Device, current_device
from .targets import ArrayClosedError, CrossDeviceError, ShapeMismatchError

_REDUCE_WORKGROUP_SIZE = 64


def _is_scalar_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_sequence_element(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray, memoryview))


def _validate_float_sequence(data: object) -> tuple[tuple[int, ...], list[float]]:
    """Validate that data is a 1D or rectangular 2D Python sequence of real numbers."""
    if isinstance(data, (str, bytes, bytearray, memoryview)) or not isinstance(data, Sequence):
        raise TypeError(
            f"Expected a 1D or 2D Python sequence of numbers, got {type(data).__name__}."
        )
    if len(data) == 0:
        return (0,), []

    first = data[0]
    if _is_sequence_element(first):
        rows = len(data)
        cols = len(first)
        values: list[float] = []
        for r, row in enumerate(data):
            if not _is_sequence_element(row):
                raise ShapeMismatchError(
                    f"Ragged 2D sequence: row {r} is {type(row).__name__}, expected sequence."
                )
            if len(row) != cols:
                raise ShapeMismatchError(
                    f"Ragged 2D sequence: row {r} has length {len(row)}, expected {cols}."
                )
            for c, item in enumerate(row):
                if _is_sequence_element(item):
                    raise ShapeMismatchError(
                        f"Only 1D and 2D sequences are supported; found nested sequence at ({r}, {c})."
                    )
                if not _is_scalar_number(item):
                    raise TypeError(
                        f"Expected numeric element at ({r}, {c}), got {type(item).__name__}."
                    )
                values.append(float(item))
        return (rows, cols), values

    values = []
    for idx, item in enumerate(data):
        if _is_sequence_element(item):
            raise ShapeMismatchError(
                f"Ragged sequence: found nested sequence at index {idx}."
            )
        if not _is_scalar_number(item):
            raise TypeError(
                f"Expected numeric element at index {idx}, got {type(item).__name__}."
            )
        values.append(float(item))
    return (len(values),), values


def _normalize_shape(shape_or_len: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(shape_or_len, bool):
        raise TypeError("Array length/shape cannot be a bool.")
    if isinstance(shape_or_len, int):
        if shape_or_len < 0:
            raise ValueError(f"Array length must be non-negative, got {shape_or_len}.")
        return (shape_or_len,)
    if isinstance(shape_or_len, tuple):
        if len(shape_or_len) not in (1, 2):
            raise ShapeMismatchError(
                f"Only 1D and 2D shapes are supported, got {shape_or_len!r}."
            )
        dims: list[int] = []
        for d in shape_or_len:
            if isinstance(d, bool) or not isinstance(d, int):
                raise TypeError(f"Shape dimensions must be ints, got {type(d).__name__}.")
            if d < 0:
                raise ValueError(f"Shape dimensions must be non-negative, got {shape_or_len!r}.")
            dims.append(d)
        return tuple(dims)
    raise TypeError(f"Expected int length or 1D/2D shape tuple, got {type(shape_or_len).__name__}.")


class GfxArray:
    """Contiguous 1D or 2D float32 array backed by a persistent device buffer."""

    def __init__(
        self,
        *,
        shape: tuple[int, ...],
        dtype: DType,
        device: Device,
        buffer: ComputeBuffer | None,
    ) -> None:
        self._shape = shape
        self._size = math.prod(shape)
        self._dtype = dtype
        self._device = device
        self._buffer = buffer
        self._closed = False

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

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

    def download_into(self, dst: bytearray | memoryview | _stdlib_array.array) -> int:
        """Synchronize and copy device bytes directly into a pre-allocated writable buffer."""
        self._ensure_open()
        if self._size == 0:
            return 0
        expected_bytes = self._size * self._dtype.itemsize
        dst_view = memoryview(dst).cast("B")
        if dst_view.readonly:
            raise TypeError("download_into destination buffer must be writable.")
        if len(dst_view) < expected_bytes:
            raise ValueError(
                f"Destination buffer ({len(dst_view)} bytes) is smaller than {expected_bytes} bytes."
            )
        assert self._buffer is not None
        backend = self._device.backend
        if hasattr(backend, "download_into"):
            read_bytes = backend.download_into(self._buffer, dst_view[:expected_bytes])
            if read_bytes != expected_bytes:
                raise RuntimeError(
                    f"Backend '{self._device.target}' returned {read_bytes} bytes, expected {expected_bytes}."
                )
            return read_bytes
        raw = backend.download(self._buffer)
        if len(raw) != expected_bytes:
            raise RuntimeError(
                f"Backend '{self._device.target}' returned {len(raw)} bytes, expected {expected_bytes}."
            )
        dst_view[:expected_bytes] = raw
        return expected_bytes

    def tobytes(self) -> bytes:
        """Synchronize and download the raw little-endian device buffer bytes."""
        self._ensure_open()
        if self._size == 0:
            return b""
        assert self._buffer is not None
        raw = self._device.backend.download(self._buffer)
        expected_bytes = self._size * self._dtype.itemsize
        if len(raw) != expected_bytes:
            raise RuntimeError(
                f"Backend '{self._device.target}' returned {len(raw)} bytes, expected {expected_bytes}."
            )
        return raw

    def toarray(self) -> _stdlib_array.array:
        """Synchronize and download the array into a stdlib array.array without float boxing."""
        self._ensure_open()
        out = _stdlib_array.array(self._dtype.struct_code)
        if self._size == 0:
            return out
        expected_bytes = self._size * self._dtype.itemsize
        out.frombytes(b"\x00" * expected_bytes)
        self.download_into(out)
        if sys.byteorder != "little":
            out.byteswap()
        return out

    def tomemoryview(self) -> memoryview:
        """Synchronize and return a typed memoryview over the downloaded array without float boxing."""
        arr = self.toarray()
        mv = memoryview(arr)
        if self.ndim == 2:
            return mv.cast("B").cast(self._dtype.struct_code, shape=self._shape)
        return mv

    def tolist(self) -> list[Any]:
        """Synchronize and download the array contents to a Python list (or nested 2D list)."""
        self._ensure_open()
        if self._size == 0:
            if self.ndim == 2:
                return [[] for _ in range(self._shape[0])]
            return []
        raw = self.tobytes()
        if sys.byteorder == "little":
            mv = memoryview(raw).cast(self._dtype.struct_code)
            if self.ndim == 2:
                rows, cols = self._shape
                return [mv[r * cols : (r + 1) * cols].tolist() for r in range(rows)]
            return mv.tolist()
        flat = list(struct.unpack(f"<{self._size}{self._dtype.struct_code}", raw))
        if self.ndim == 2:
            rows, cols = self._shape
            return [flat[r * cols : (r + 1) * cols] for r in range(rows)]
        return flat

    def _binary_or_scalar(self, op: str, other: object) -> Any:
        self._ensure_open()
        if isinstance(other, GfxArray):
            other._ensure_open()
            if self._device is not other._device:
                raise CrossDeviceError(
                    f"Cannot combine arrays on different devices: {self._device!r} vs {other._device!r}."
                )
            if self._shape != other._shape:
                raise ShapeMismatchError(
                    f"Incompatible shapes for '{op}': {self._shape} vs {other._shape}."
                )
            if self._size == 0:
                return GfxArray(shape=self._shape, dtype=self._dtype, device=self._device, buffer=None)
            assert self._buffer is not None and other._buffer is not None
            backend = self._device.backend
            out_buf = _allocate_output(backend, self.nbytes)
            try:
                backend.binary_op(op, self._buffer, other._buffer, out_buf, self._size)
            except Exception:
                out_buf.close()
                raise
            return GfxArray(shape=self._shape, dtype=self._dtype, device=self._device, buffer=out_buf)

        if _is_scalar_number(other):
            if self._size == 0:
                return GfxArray(shape=self._shape, dtype=self._dtype, device=self._device, buffer=None)
            assert self._buffer is not None
            backend = self._device.backend
            out_buf = _allocate_output(backend, self.nbytes)
            try:
                backend.scalar_op(op, self._buffer, float(other), out_buf, self._size)
            except Exception:
                out_buf.close()
                raise
            return GfxArray(shape=self._shape, dtype=self._dtype, device=self._device, buffer=out_buf)

        return NotImplemented

    def __add__(self, other: object) -> Any:
        return self._binary_or_scalar("add", other)

    def __radd__(self, other: object) -> Any:
        return self._binary_or_scalar("add", other)

    def __mul__(self, other: object) -> Any:
        return self._binary_or_scalar("mul", other)

    def __rmul__(self, other: object) -> Any:
        return self._binary_or_scalar("mul", other)

    def __matmul__(self, other: object) -> Any:
        if not isinstance(other, GfxArray):
            return NotImplemented
        return matmul(self, other)

    def sum(self) -> GfxArray:
        """Compute bounded sum reduction over all elements, returning a shape (1,) GfxArray."""
        return sum(self)

    def __repr__(self) -> str:
        return (
            f"GfxArray(shape={self.shape}, dtype={self._dtype.name}, "
            f"target={self._device.target!r}, closed={self.closed})"
        )


def array(
    data: Sequence[Any],
    dtype: DType | str | None = float32,
    *,
    device: Device | None = None,
) -> GfxArray:
    """Create a 1D or 2D contiguous float32 GfxArray from a Python sequence."""
    resolved_dtype = resolve_dtype(dtype)
    shape, values = _validate_float_sequence(data)
    target_device = current_device() if device is None else device
    target_device.ensure_open()

    n = len(values)
    if n == 0:
        return GfxArray(shape=shape, dtype=resolved_dtype, device=target_device, buffer=None)

    raw = struct.pack(f"<{n}{resolved_dtype.struct_code}", *values)
    backend = target_device.backend
    buf = backend.allocate(len(raw))
    try:
        backend.upload(buf, raw)
    except Exception:
        buf.close()
        raise
    return GfxArray(shape=shape, dtype=resolved_dtype, device=target_device, buffer=buf)


def _allocate_output(backend: ComputeBackend, size_bytes: int) -> ComputeBuffer:
    if hasattr(backend, "allocate_uninitialized"):
        return backend.allocate_uninitialized(size_bytes)
    return backend.allocate(size_bytes)


def empty(
    shape: int | tuple[int, ...],
    dtype: DType | str | None = float32,
    *,
    device: Device | None = None,
) -> GfxArray:
    """Allocate an uninitialized 1D or 2D contiguous float32 GfxArray."""
    resolved_shape = _normalize_shape(shape)
    resolved_dtype = resolve_dtype(dtype)
    target_device = current_device() if device is None else device
    target_device.ensure_open()

    n = math.prod(resolved_shape)
    if n == 0:
        return GfxArray(shape=resolved_shape, dtype=resolved_dtype, device=target_device, buffer=None)

    buf = _allocate_output(target_device.backend, n * resolved_dtype.itemsize)
    return GfxArray(shape=resolved_shape, dtype=resolved_dtype, device=target_device, buffer=buf)


def sum(x: GfxArray) -> GfxArray:  # noqa: A001
    """Compute a device-resident float32 sum reduction returning shape (1,)."""
    if not isinstance(x, GfxArray):
        raise TypeError(f"sum() expects a GfxArray, got {type(x).__name__}.")
    x._ensure_open()
    dev = x.device
    backend = dev.backend

    if x.size == 0:
        buf = backend.allocate(x.dtype.itemsize)
        try:
            backend.upload(buf, struct.pack("<f", 0.0))
        except Exception:
            buf.close()
            raise
        return GfxArray(shape=(1,), dtype=x.dtype, device=dev, buffer=buf)

    assert x._buffer is not None
    cur_buf = x._buffer
    owns_cur = False
    cur_len = x.size

    try:
        while cur_len > 1:
            num_groups = (cur_len + _REDUCE_WORKGROUP_SIZE - 1) // _REDUCE_WORKGROUP_SIZE
            next_buf = _allocate_output(backend, num_groups * x.dtype.itemsize)
            try:
                backend.reduce_sum_pass(cur_buf, next_buf, cur_len)
            except Exception:
                next_buf.close()
                raise
            if owns_cur:
                cur_buf.close()
            cur_buf = next_buf
            owns_cur = True
            cur_len = num_groups

        if not owns_cur:
            # Single-element input: copy via a 1-workgroup pass into an owned output buffer.
            out_buf = _allocate_output(backend, x.dtype.itemsize)
            try:
                backend.reduce_sum_pass(cur_buf, out_buf, 1)
            except Exception:
                out_buf.close()
                raise
            cur_buf = out_buf

        return GfxArray(shape=(1,), dtype=x.dtype, device=dev, buffer=cur_buf)
    except Exception:
        if owns_cur:
            cur_buf.close()
        raise


def matmul(a: GfxArray, b: GfxArray) -> GfxArray:
    """Compute 2D float32 matrix multiplication (M, K) @ (K, N) -> (M, N) on GPU."""
    if not isinstance(a, GfxArray) or not isinstance(b, GfxArray):
        raise TypeError("matmul() expects two GfxArray operands.")
    a._ensure_open()
    b._ensure_open()
    if a.device is not b.device:
        raise CrossDeviceError(
            f"Cannot multiply arrays on different devices: {a.device!r} vs {b.device!r}."
        )
    if a.ndim != 2 or b.ndim != 2:
        raise ShapeMismatchError(
            f"matmul requires 2D arrays, got shapes {a.shape} and {b.shape}."
        )
    m, k_a = a.shape
    k_b, n = b.shape
    if k_a != k_b:
        raise ShapeMismatchError(
            f"Incompatible matrix shapes for matmul: {a.shape} and {b.shape}."
        )

    out_shape = (m, n)
    if m == 0 or n == 0:
        return GfxArray(shape=out_shape, dtype=a.dtype, device=a.device, buffer=None)

    backend = a.device.backend
    out_buf = _allocate_output(backend, m * n * a.dtype.itemsize)
    if k_a == 0:
        try:
            backend.upload(out_buf, b"\x00" * (m * n * a.dtype.itemsize))
        except Exception:
            out_buf.close()
            raise
        return GfxArray(shape=out_shape, dtype=a.dtype, device=a.device, buffer=out_buf)

    assert a._buffer is not None and b._buffer is not None
    try:
        backend.matmul(a._buffer, b._buffer, out_buf, m, n, k_a)
    except Exception:
        out_buf.close()
        raise
    return GfxArray(shape=out_shape, dtype=a.dtype, device=a.device, buffer=out_buf)
