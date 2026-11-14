"""Data type definitions and validation for gfxpy."""

from __future__ import annotations

from dataclasses import dataclass

from .targets import UnsupportedDTypeError


@dataclass(frozen=True)
class DType:
    """Descriptor for an array element data type."""

    name: str
    itemsize: int
    struct_code: str

    def __str__(self) -> str:
        return self.name


float32 = DType(name="float32", itemsize=4, struct_code="f")


def resolve_dtype(dtype: DType | str | None) -> DType:
    """Validate and normalize a dtype to a supported DType."""
    if dtype is None or dtype is float32:
        return float32
    if isinstance(dtype, DType):
        if dtype == float32:
            return float32
        raise UnsupportedDTypeError(
            f"Unsupported dtype '{dtype.name}'. Only 'float32' is supported."
        )
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in {"float32", "f32"}:
            return float32
        raise UnsupportedDTypeError(
            f"Unsupported dtype '{dtype}'. Only 'float32' is supported."
        )
    raise UnsupportedDTypeError(
        f"Unsupported dtype {dtype!r}. Only 'float32' is supported."
    )
