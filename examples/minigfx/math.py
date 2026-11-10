"""Pure Python linear algebra library for MiniGfx (zero external dependencies).

Storage and Coordinate Conventions:
- Right-handed Cartesian coordinates (+X right, +Y up, +Z towards viewer; view along -Z).
- Column-major matrix layout: flat tuple of cols * rows floats, indexed as m[r, c] or m[i].
- Complete set of CxR matrix types: Mat2x2 (Mat2), Mat2x3, Mat2x4, Mat3x2, Mat3x3 (Mat3),
  Mat3x4, Mat4x2, Mat4x3, Mat4x4 (Mat4).
- Column-vector transform convention: v' = M * v.
- Clip-space Z range: [0.0, 1.0] (DirectX, Metal, Vulkan standard).
- std430 binary packing via .to_bytes() using IEEE-754 little-endian encoding with column padding.
"""

from __future__ import annotations

import math
import struct
from typing import Any, Iterator, Sequence, Union


class _Vec:
    """Base class providing shared vector arithmetic and serialization."""

    __slots__ = ()
    _dim: int
    _fmt: str

    def __len__(self) -> int:
        return self._dim

    def __iter__(self) -> Iterator[float]:
        for i in range(self._dim):
            yield self[i]

    def __add__(self, o: _Vec) -> _Vec:
        if isinstance(o, type(self)):
            return type(self)(*(a + b for a, b in zip(self, o)))
        raise TypeError(f"+ unsupported for {type(self)} and {type(o)}")

    def __sub__(self, o: _Vec) -> _Vec:
        if isinstance(o, type(self)):
            return type(self)(*(a - b for a, b in zip(self, o)))
        raise TypeError(f"- unsupported for {type(self)} and {type(o)}")

    def __neg__(self) -> _Vec:
        return type(self)(*(-a for a in self))

    def __mul__(self, s: float) -> _Vec:
        if isinstance(s, (int, float)):
            return type(self)(*(a * s for a in self))
        return NotImplemented

    def __rmul__(self, s: float) -> _Vec:
        return self.__mul__(s)

    def __truediv__(self, s: float) -> _Vec:
        if isinstance(s, (int, float)):
            if s == 0.0:
                raise ZeroDivisionError("Vector division by zero")
            return self * (1.0 / s)
        return NotImplemented

    def dot(self, o: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(self, o))

    def length_sq(self) -> float:
        return self.dot(self)

    def length(self) -> float:
        return math.sqrt(self.length_sq())

    def normalize(self) -> _Vec:
        d = self.length()
        return type(self)(*(0.0 for _ in range(self._dim))) if d < 1e-12 else self * (1.0 / d)

    def to_bytes(self) -> bytes:
        return struct.pack(self._fmt, *self)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, (type(self), tuple, list)) and len(o) == self._dim:
            return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(self, o))
        return False


class Vec2(_Vec):
    """2D single-precision float vector."""

    __slots__ = ("x", "y")
    _dim, _fmt = 2, "<2f"

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        self.x, self.y = float(x), float(y)

    def __getitem__(self, i: int) -> float:
        if i in (0, -2): return self.x
        if i in (1, -1): return self.y
        raise IndexError(f"Vec2 index out of range: {i}")

    def __repr__(self) -> str:
        return f"Vec2({self.x}, {self.y})"


class Vec3(_Vec):
    """3D single-precision float vector."""

    __slots__ = ("x", "y", "z")
    _dim, _fmt = 3, "<3f"

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0) -> None:
        self.x, self.y, self.z = float(x), float(y), float(z)

    def __getitem__(self, i: int) -> float:
        if i in (0, -3): return self.x
        if i in (1, -2): return self.y
        if i in (2, -1): return self.z
        raise IndexError(f"Vec3 index out of range: {i}")

    def cross(self, o: Vec3) -> Vec3:
        return Vec3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )

    def __repr__(self) -> str:
        return f"Vec3({self.x}, {self.y}, {self.z})"


class Vec4(_Vec):
    """4D single-precision float vector."""

    __slots__ = ("x", "y", "z", "w")
    _dim, _fmt = 4, "<4f"

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0) -> None:
        self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)

    def __getitem__(self, i: int) -> float:
        if i in (0, -4): return self.x
        if i in (1, -3): return self.y
        if i in (2, -2): return self.z
        if i in (3, -1): return self.w
        raise IndexError(f"Vec4 index out of range: {i}")

    def __repr__(self) -> str:
        return f"Vec4({self.x}, {self.y}, {self.z}, {self.w})"


class Matrix:
    """Generic CxR column-major matrix obeying std430 packing and layout rules."""

    __slots__ = ("_data", "_cols", "_rows")
    cols: int = 4
    rows: int = 4
    _data: tuple[float, ...]
    _cols: int
    _rows: int

    def __new__(cls, *args: Any, **kwargs: Any) -> Matrix:
        if cls is Matrix:
            if len(args) < 2:
                raise ValueError("Matrix requires at least cols and rows: Matrix(cols, rows, ...)")
            cols, rows = int(args[0]), int(args[1])
            target_cls = _MATRIX_CLASSES.get((cols, rows))
            if target_cls is not None:
                return target_cls(*args[2:])
            inst = super().__new__(cls)
            inst._cols = cols
            inst._rows = rows
            return inst
        return super().__new__(cls)

    def __init__(self, *args: Any) -> None:
        if self.__class__ is Matrix:
            cols, rows = int(args[0]), int(args[1])
            rem = args[2:]
            self._cols = cols
            self._rows = rows
        else:
            cols, rows = self.__class__.cols, self.__class__.rows
            if len(args) >= 2 and args[0] == cols and args[1] == rows:
                rem = args[2:]
            else:
                rem = args
        total = cols * rows
        name = "Mat4" if self.__class__.__name__ in ("Mat4", "Mat4x4") else self.__class__.__name__
        if len(rem) == 0:
            self._data = tuple(1.0 if c == r else 0.0 for c in range(cols) for r in range(rows))
        elif len(rem) == 1:
            t = tuple(float(x) for x in rem[0])
            if len(t) != total:
                raise ValueError(f"{name} requires {total} elements, got {len(t)}")
            self._data = t
        elif len(rem) == total:
            self._data = tuple(float(x) for x in rem)
        else:
            raise ValueError(f"{name} requires 0, 1, or {total} arguments, got {len(rem)}")

    @property
    def cols(self) -> int:
        return getattr(self, "_cols", self.__class__.cols)

    @property
    def rows(self) -> int:
        return getattr(self, "_rows", self.__class__.rows)

    @property
    def elements(self) -> tuple[float, ...]:
        return self._data

    @property
    def _elements(self) -> tuple[float, ...]:
        return self._data

    @property
    def columns(self) -> tuple[_Vec, ...]:
        cols_list = []
        for c in range(self.cols):
            col_data = [self._data[c * self.rows + r] for r in range(self.rows)]
            if self.rows == 2:
                cols_list.append(Vec2(col_data[0], col_data[1]))
            elif self.rows == 3:
                cols_list.append(Vec3(col_data[0], col_data[1], col_data[2]))
            elif self.rows == 4:
                cols_list.append(Vec4(col_data[0], col_data[1], col_data[2], col_data[3]))
            else:
                cols_list.append(tuple(col_data))
        return tuple(cols_list)

    def __getitem__(self, idx: Union[int, tuple[int, int]]) -> float:
        if isinstance(idx, tuple):
            if len(idx) != 2:
                raise IndexError(f"Matrix 2D index requires (row, col), got {idx}")
            r, c = idx
            r_norm = r + self.rows if r < 0 else r
            c_norm = c + self.cols if c < 0 else c
            if 0 <= r_norm < self.rows and 0 <= c_norm < self.cols:
                return self._data[c_norm * self.rows + r_norm]
            raise IndexError(f"Matrix index {idx} out of range for shape ({self.cols} cols, {self.rows} rows)")
        if isinstance(idx, int):
            total = self.cols * self.rows
            norm_idx = idx + total if idx < 0 else idx
            if not (0 <= norm_idx < total):
                raise IndexError(f"Matrix 1D index {idx} out of range for size {total}")
            return self._data[norm_idx]
        raise TypeError(f"Matrix index must be int or tuple of 2 ints, got {type(idx).__name__}")

    def __iter__(self) -> Iterator[float]:
        return iter(self._data)

    def __len__(self) -> int:
        return self.cols * self.rows

    @classmethod
    def identity(cls, cols: int | None = None, rows: int | None = None) -> Matrix:
        if cls is Matrix:
            c = cols if cols is not None else 4
            r = rows if rows is not None else c
            target_cls = _get_matrix_class(c, r)
            return target_cls.identity()
        data = tuple(1.0 if c == r else 0.0 for c in range(cls.cols) for r in range(cls.rows))
        return cls(data)

    @classmethod
    def zero(cls, cols: int | None = None, rows: int | None = None) -> Matrix:
        if cls is Matrix:
            c = cols if cols is not None else 4
            r = rows if rows is not None else c
            target_cls = _get_matrix_class(c, r)
            return target_cls.zero()
        return cls((0.0,) * (cls.cols * cls.rows))

    def transpose(self) -> Matrix:
        new_cols = self.rows
        new_rows = self.cols
        transposed = [0.0] * (new_cols * new_rows)
        for c in range(self.cols):
            for r in range(self.rows):
                val = self._data[c * self.rows + r]
                transposed[r * new_rows + c] = val
        target_cls = _get_matrix_class(new_cols, new_rows)
        return target_cls(transposed)

    @property
    def T(self) -> Matrix:
        return self.transpose()

    def __mul__(self, o: Union[Matrix, _Vec, float, int]) -> Union[Matrix, _Vec]:
        if isinstance(o, Matrix):
            if self.cols != o.rows:
                raise ValueError(
                    f"Cannot multiply matrix with cols={self.cols} by matrix with rows={o.rows}"
                )
            res_cols = o.cols
            res_rows = self.rows
            res_data = [0.0] * (res_cols * res_rows)
            for c in range(res_cols):
                for r in range(res_rows):
                    s = 0.0
                    for k in range(self.cols):
                        s += self._data[k * self.rows + r] * o._data[c * o.rows + k]
                    res_data[c * res_rows + r] = s
            target_cls = _get_matrix_class(res_cols, res_rows)
            return target_cls(res_data)

        if isinstance(o, _Vec):
            if self.cols != len(o):
                raise ValueError(
                    f"Cannot multiply matrix with cols={self.cols} by vector of length {len(o)}"
                )
            v_elems = list(o)
            res_elems = [0.0] * self.rows
            for r in range(self.rows):
                s = 0.0
                for c in range(self.cols):
                    s += self._data[c * self.rows + r] * v_elems[c]
                res_elems[r] = s
            if self.rows == 2:
                return Vec2(res_elems[0], res_elems[1])
            elif self.rows == 3:
                return Vec3(res_elems[0], res_elems[1], res_elems[2])
            elif self.rows == 4:
                return Vec4(res_elems[0], res_elems[1], res_elems[2], res_elems[3])
            raise NotImplementedError(f"Vector result unsupported for rows={self.rows}")

        if isinstance(o, (int, float)):
            s = float(o)
            return type(self)([x * s for x in self._data])

        return NotImplemented

    def __rmul__(self, s: Any) -> Any:
        if isinstance(s, (int, float)):
            return self * s
        return NotImplemented

    def __matmul__(self, o: Any) -> Any:
        return self.__mul__(o)

    def __rmatmul__(self, o: Any) -> Any:
        if isinstance(o, _Vec):
            if len(o) != self.rows:
                raise ValueError(
                    f"Cannot multiply vector of length {len(o)} by matrix with rows={self.rows}"
                )
            v_elems = list(o)
            res_elems = [0.0] * self.cols
            for c in range(self.cols):
                s = 0.0
                for r in range(self.rows):
                    s += v_elems[r] * self._data[c * self.rows + r]
                res_elems[c] = s
            if self.cols == 2:
                return Vec2(res_elems[0], res_elems[1])
            elif self.cols == 3:
                return Vec3(res_elems[0], res_elems[1], res_elems[2])
            elif self.cols == 4:
                return Vec4(res_elems[0], res_elems[1], res_elems[2], res_elems[3])
        return NotImplemented

    def __truediv__(self, s: float) -> Matrix:
        if isinstance(s, (int, float)):
            if s == 0.0:
                raise ZeroDivisionError("Matrix division by zero")
            return self * (1.0 / s)
        return NotImplemented

    def __add__(self, o: Matrix) -> Matrix:
        if isinstance(o, Matrix):
            if self.cols != o.cols or self.rows != o.rows:
                raise ValueError(
                    f"Cannot add matrices with shapes ({self.cols}, {self.rows}) and ({o.cols}, {o.rows})"
                )
            return type(self)([a + b for a, b in zip(self._data, o._data)])
        return NotImplemented

    def __sub__(self, o: Matrix) -> Matrix:
        if isinstance(o, Matrix):
            if self.cols != o.cols or self.rows != o.rows:
                raise ValueError(
                    f"Cannot subtract matrices with shapes ({self.cols}, {self.rows}) and ({o.cols}, {o.rows})"
                )
            return type(self)([a - b for a, b in zip(self._data, o._data)])
        return NotImplemented

    def __neg__(self) -> Matrix:
        return type(self)([-a for a in self._data])

    def inverse(self) -> Matrix:
        if self.cols != self.rows:
            raise ValueError(f"Cannot invert non-square matrix of shape ({self.cols} cols, {self.rows} rows)")
        if self.cols == 2:
            m = self._data
            det = m[0] * m[3] - m[2] * m[1]
            if abs(det) < 1e-12:
                raise ValueError("Matrix is singular and cannot be inverted.")
            inv = 1.0 / det
            return Mat2x2((m[3] * inv, -m[1] * inv, -m[2] * inv, m[0] * inv))
        if self.cols == 3:
            m = self._data
            c00 = m[4] * m[8] - m[7] * m[5]
            c10 = -(m[1] * m[8] - m[7] * m[2])
            c20 = m[1] * m[5] - m[4] * m[2]
            det = m[0] * c00 + m[3] * c10 + m[6] * c20
            if abs(det) < 1e-12:
                raise ValueError("Matrix is singular and cannot be inverted.")
            inv = 1.0 / det
            c01 = -(m[3] * m[8] - m[6] * m[5])
            c11 = m[0] * m[8] - m[6] * m[2]
            c21 = -(m[0] * m[5] - m[3] * m[2])
            c02 = m[3] * m[7] - m[6] * m[4]
            c12 = -(m[0] * m[7] - m[6] * m[1])
            c22 = m[0] * m[4] - m[3] * m[1]
            return Mat3x3((
                c00 * inv, c10 * inv, c20 * inv,
                c01 * inv, c11 * inv, c21 * inv,
                c02 * inv, c12 * inv, c22 * inv,
            ))
        if self.cols == 4:
            m = self._data
            a00, a10, a20, a30 = m[0], m[1], m[2], m[3]
            a01, a11, a21, a31 = m[4], m[5], m[6], m[7]
            a02, a12, a22, a32 = m[8], m[9], m[10], m[11]
            a03, a13, a23, a33 = m[12], m[13], m[14], m[15]

            b00, b01 = a00 * a11 - a01 * a10, a00 * a12 - a02 * a10
            b02, b03 = a00 * a13 - a03 * a10, a01 * a12 - a02 * a11
            b04, b05 = a01 * a13 - a03 * a11, a02 * a13 - a03 * a12
            b06, b07 = a20 * a31 - a21 * a30, a20 * a32 - a22 * a30
            b08, b09 = a20 * a33 - a23 * a30, a21 * a32 - a22 * a31
            b10, b11 = a21 * a33 - a23 * a31, a22 * a33 - a23 * a32

            det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06
            if abs(det) < 1e-12:
                raise ValueError("Matrix is singular and cannot be inverted.")

            inv = 1.0 / det
            return Mat4x4((
                (a11 * b11 - a12 * b10 + a13 * b09) * inv,
                (-a10 * b11 + a12 * b08 - a13 * b07) * inv,
                (a10 * b10 - a11 * b08 + a13 * b06) * inv,
                (-a10 * b09 + a11 * b07 - a12 * b06) * inv,
                (-a01 * b11 + a02 * b10 - a03 * b09) * inv,
                (a00 * b11 - a02 * b08 + a03 * b07) * inv,
                (-a00 * b10 + a01 * b08 - a03 * b06) * inv,
                (a00 * b09 - a01 * b07 + a02 * b06) * inv,
                (a31 * b05 - a32 * b04 + a33 * b03) * inv,
                (-a30 * b05 + a32 * b02 - a33 * b01) * inv,
                (a30 * b04 - a31 * b02 + a33 * b00) * inv,
                (-a30 * b03 + a31 * b01 - a32 * b00) * inv,
                (-a21 * b05 + a22 * b04 - a23 * b03) * inv,
                (a20 * b05 - a22 * b02 + a23 * b01) * inv,
                (-a20 * b04 + a21 * b02 - a23 * b00) * inv,
                (a20 * b03 - a21 * b01 + a22 * b00) * inv,
            ))
        raise NotImplementedError(f"Inverse not implemented for matrix with cols={self.cols}")

    def to_bytes(self) -> bytes:
        out = bytearray()
        for c in range(self.cols):
            col_floats = [self._data[c * self.rows + r] for r in range(self.rows)]
            if self.rows == 2:
                out.extend(struct.pack("<2f", *col_floats))
            elif self.rows == 3:
                out.extend(struct.pack("<3ff", *col_floats, 0.0))
            elif self.rows == 4:
                out.extend(struct.pack("<4f", *col_floats))
            else:
                raise NotImplementedError(f"std430 packing unsupported for rows={self.rows}")
        return bytes(out)

    @classmethod
    def from_bytes(
        cls,
        buffer: bytes | bytearray | memoryview,
        cols: int | None = None,
        rows: int | None = None,
    ) -> Matrix:
        if cls is Matrix:
            c = cols if cols is not None else 4
            r = rows if rows is not None else c
            target_cls = _get_matrix_class(c, r)
            return target_cls.from_bytes(buffer)
        c = cls.cols
        r = cls.rows
        stride = 8 if r == 2 else 16
        min_bytes = c * stride
        name = "Mat4" if cls.__name__ in ("Mat4", "Mat4x4") else cls.__name__
        if len(buffer) < min_bytes:
            raise ValueError(f"Buffer size {len(buffer)} < {min_bytes} bytes for {name}")
        flat: list[float] = []
        for col_idx in range(c):
            offset = col_idx * stride
            if r == 2:
                flat.extend(struct.unpack_from("<2f", buffer, offset))
            elif r == 3:
                flat.extend(struct.unpack_from("<3f", buffer, offset))
            elif r == 4:
                flat.extend(struct.unpack_from("<4f", buffer, offset))
            else:
                raise NotImplementedError(f"std430 unpacking unsupported for rows={r}")
        return cls(flat)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Matrix):
            if self.cols != o.cols or self.rows != o.rows:
                return False
            return all(
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(self._data, o._data)
            )
        if isinstance(o, (tuple, list)) and len(o) == self.cols * self.rows:
            return all(
                math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                for a, b in zip(self._data, o)
            )
        return False

    def __repr__(self) -> str:
        name = "Mat4" if self.__class__.__name__ in ("Mat4", "Mat4x4") else self.__class__.__name__
        return f"{name}({self._data})"


class Mat2x2(Matrix):
    """2x2 column-major transformation matrix."""

    cols, rows = 2, 2
    __slots__ = ()


Mat2 = Mat2x2


class Mat2x3(Matrix):
    """2x3 column-major matrix (2 columns, 3 rows)."""

    cols, rows = 2, 3
    __slots__ = ()


class Mat2x4(Matrix):
    """2x4 column-major matrix (2 columns, 4 rows)."""

    cols, rows = 2, 4
    __slots__ = ()


class Mat3x2(Matrix):
    """3x2 column-major matrix (3 columns, 2 rows)."""

    cols, rows = 3, 2
    __slots__ = ()


class Mat3x3(Matrix):
    """3x3 column-major transformation matrix."""

    cols, rows = 3, 3
    __slots__ = ()


Mat3 = Mat3x3


class Mat3x4(Matrix):
    """3x4 column-major matrix (3 columns, 4 rows)."""

    cols, rows = 3, 4
    __slots__ = ()


class Mat4x2(Matrix):
    """4x2 column-major matrix (4 columns, 2 rows)."""

    cols, rows = 4, 2
    __slots__ = ()


class Mat4x3(Matrix):
    """4x3 column-major matrix (4 columns, 3 rows)."""

    cols, rows = 4, 3
    __slots__ = ()


class Mat4(Matrix):
    """4x4 column-major transformation matrix."""

    cols, rows = 4, 4
    __slots__ = ()

    @classmethod
    def translation(cls, x: float | Sequence[float], y: float = 0.0, z: float = 0.0) -> Mat4:
        if hasattr(x, "__len__") and not isinstance(x, (int, float)):
            tx, ty, tz = float(x[0]), float(x[1]), float(x[2])
        else:
            tx, ty, tz = float(x), float(y), float(z)
        return cls((
            1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,  tx,  ty,  tz,  1.0,
        ))

    @classmethod
    def from_translation(cls, v: Sequence[float]) -> Mat4:
        return cls.translation(v)

    @classmethod
    def scaling(cls, x: float | Sequence[float], y: float | None = None, z: float | None = None) -> Mat4:
        if hasattr(x, "__len__") and not isinstance(x, (int, float)):
            sx, sy, sz = float(x[0]), float(x[1]), float(x[2])
        elif y is None and z is None:
            sx = sy = sz = float(x)
        else:
            sx = float(x)
            sy = float(y if y is not None else 1.0)
            sz = float(z if z is not None else 1.0)
        return cls((
            sx,  0.0, 0.0, 0.0,  0.0, sy,  0.0, 0.0,
            0.0, 0.0, sz,  0.0,  0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def from_scale(cls, v: Sequence[float]) -> Mat4:
        return cls.scaling(v)

    @classmethod
    def rotation_x(cls, a: float) -> Mat4:
        c, s = math.cos(a), math.sin(a)
        return cls((
            1.0, 0.0, 0.0, 0.0,  0.0, c, s, 0.0,
            0.0, -s,  c,   0.0,  0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_y(cls, a: float) -> Mat4:
        c, s = math.cos(a), math.sin(a)
        return cls((
            c,   0.0, -s,  0.0,  0.0, 1.0, 0.0, 0.0,
            s,   0.0, c,   0.0,  0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_z(cls, a: float) -> Mat4:
        c, s = math.cos(a), math.sin(a)
        return cls((
            c,   s,   0.0, 0.0,  -s,  c,   0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0,
        ))

    @classmethod
    def rotation_axis(cls, axis: Sequence[float], angle: float) -> Mat4:
        ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        mag_sq = ax * ax + ay * ay + az * az
        if mag_sq < 1e-12:
            return cls.identity()
        inv = 1.0 / math.sqrt(mag_sq)
        x, y, z = ax * inv, ay * inv, az * inv
        c, s, t = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
        return cls((
            t * x * x + c,     t * x * y + s * z, t * x * z - s * y, 0.0,
            t * x * y - s * z, t * y * y + c,     t * y * z + s * x, 0.0,
            t * x * z + s * y, t * y * z - s * x, t * z * z + c,     0.0,
            0.0,               0.0,               0.0,               1.0,
        ))

    @classmethod
    def look_at(cls, eye: Sequence[float], target: Sequence[float], up: Sequence[float]) -> Mat4:
        ex, ey, ez = float(eye[0]), float(eye[1]), float(eye[2])
        zx, zy, zz = ex - float(target[0]), ey - float(target[1]), ez - float(target[2])
        z_sq = zx * zx + zy * zy + zz * zz
        if z_sq < 1e-12:
            raise ValueError("Eye and target positions must not be coincident.")
        inv_z = 1.0 / math.sqrt(z_sq)
        zx, zy, zz = zx * inv_z, zy * inv_z, zz * inv_z

        ux, uy, uz = float(up[0]), float(up[1]), float(up[2])
        if ux * ux + uy * uy + uz * uz < 1e-12:
            raise ValueError("Up vector must not be zero-length.")

        xx, xy, xz = uy * zz - uz * zy, uz * zx - ux * zz, ux * zy - uy * zx
        x_sq = xx * xx + xy * xy + xz * xz
        if x_sq < 1e-12:
            raise ValueError("Up vector must not be collinear with view direction.")
        inv_x = 1.0 / math.sqrt(x_sq)
        xx, xy, xz = xx * inv_x, xy * inv_x, xz * inv_x

        yx, yy, yz = zy * xz - zz * xy, zz * xx - zx * xz, zx * xy - zy * xx
        return cls((
            xx, yx, zx, 0.0,
            xy, yy, zy, 0.0,
            xz, yz, zz, 0.0,
            -(xx * ex + xy * ey + xz * ez),
            -(yx * ex + yy * ey + yz * ez),
            -(zx * ex + zy * ey + zz * ez),
            1.0,
        ))

    @classmethod
    def perspective(cls, fov_y: float, aspect: float, near: float, far: float) -> Mat4:
        if not (0.0 < fov_y < math.pi):
            raise ValueError(f"fov_y must be between 0 and pi radians, got {fov_y}")
        if aspect <= 0.0:
            raise ValueError(f"aspect ratio must be positive, got {aspect}")
        if near <= 0.0:
            raise ValueError(f"near plane must be positive, got {near}")
        if far <= near:
            raise ValueError(f"far plane must be greater than near plane, got near={near}, far={far}")

        f = 1.0 / math.tan(fov_y * 0.5)
        nf = 1.0 / (near - far)
        return cls((
            f / aspect, 0.0, 0.0,             0.0,
            0.0,        f,   0.0,             0.0,
            0.0,        0.0, far * nf,        -1.0,
            0.0,        0.0, near * far * nf, 0.0,
        ))


Mat4x4 = Mat4

_MATRIX_CLASSES: dict[tuple[int, int], type[Matrix]] = {
    (2, 2): Mat2x2,
    (2, 3): Mat2x3,
    (2, 4): Mat2x4,
    (3, 2): Mat3x2,
    (3, 3): Mat3x3,
    (3, 4): Mat3x4,
    (4, 2): Mat4x2,
    (4, 3): Mat4x3,
    (4, 4): Mat4x4,
}


def _get_matrix_class(cols: int, rows: int) -> type[Matrix]:
    return _MATRIX_CLASSES.get((cols, rows), Matrix)


class Quat:
    """Unit quaternion representing 3D rotation."""

    __slots__ = ("x", "y", "z", "w")

    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 1.0) -> None:
        self.x, self.y, self.z, self.w = float(x), float(y), float(z), float(w)

    @classmethod
    def identity(cls) -> Quat:
        return cls(0.0, 0.0, 0.0, 1.0)

    @classmethod
    def from_axis_angle(cls, axis: Sequence[float], angle: float) -> Quat:
        ax, ay, az = float(axis[0]), float(axis[1]), float(axis[2])
        sq = ax * ax + ay * ay + az * az
        if sq < 1e-12:
            return cls.identity()
        s = math.sin(angle * 0.5) / math.sqrt(sq)
        return cls(ax * s, ay * s, az * s, math.cos(angle * 0.5))

    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalize(self) -> Quat:
        n = self.norm()
        return Quat.identity() if n < 1e-12 else Quat(self.x / n, self.y / n, self.z / n, self.w / n)

    def conjugate(self) -> Quat:
        return Quat(-self.x, -self.y, -self.z, self.w)

    def mul(self, o: Quat) -> Quat:
        return self.__mul__(o)

    def __mul__(self, o: Quat) -> Quat:
        if isinstance(o, Quat):
            return Quat(
                self.w * o.x + self.x * o.w + self.y * o.z - self.z * o.y,
                self.w * o.y - self.x * o.z + self.y * o.w + self.z * o.x,
                self.w * o.z + self.x * o.y - self.y * o.x + self.z * o.w,
                self.w * o.w - self.x * o.x - self.y * o.y - self.z * o.z,
            )
        return NotImplemented

    def to_mat4(self) -> Mat4:
        q = self.normalize()
        x, y, z, w = q.x, q.y, q.z, q.w
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return Mat4((
            1.0 - 2.0 * (yy + zz), 2.0 * (xy + wz),       2.0 * (xz - wy),       0.0,
            2.0 * (xy - wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz + wx),       0.0,
            2.0 * (xz + wy),       2.0 * (yz - wx),       1.0 - 2.0 * (xx + yy), 0.0,
            0.0,                   0.0,                   0.0,                   1.0,
        ))

    def slerp(self, o: Quat, t: float) -> Quat:
        q1, q2 = self.normalize(), o.normalize()
        dot = q1.x * q2.x + q1.y * q2.y + q1.z * q2.z + q1.w * q2.w
        if dot < 0.0:
            q2, dot = Quat(-q2.x, -q2.y, -q2.z, -q2.w), -dot
        if dot > 0.9995:
            return Quat(
                q1.x + t * (q2.x - q1.x), q1.y + t * (q2.y - q1.y),
                q1.z + t * (q2.z - q1.z), q1.w + t * (q2.w - q1.w),
            ).normalize()
        theta = math.acos(max(-1.0, min(1.0, dot)))
        s = math.sin(theta)
        w1, w2 = math.sin((1.0 - t) * theta) / s, math.sin(t * theta) / s
        return Quat(
            w1 * q1.x + w2 * q2.x, w1 * q1.y + w2 * q2.y,
            w1 * q1.z + w2 * q2.z, w1 * q1.w + w2 * q2.w,
        ).normalize()

    def to_bytes(self) -> bytes:
        return struct.pack("<4f", self.x, self.y, self.z, self.w)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Quat) and all(
            math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
            for a, b in zip((self.x, self.y, self.z, self.w), (o.x, o.y, o.z, o.w))
        )

    def __repr__(self) -> str:
        return f"Quat({self.x}, {self.y}, {self.z}, {self.w})"


__all__ = [
    "Vec2",
    "Vec3",
    "Vec4",
    "Matrix",
    "Mat2x2",
    "Mat2",
    "Mat2x3",
    "Mat2x4",
    "Mat3x2",
    "Mat3x3",
    "Mat3",
    "Mat3x4",
    "Mat4x2",
    "Mat4x3",
    "Mat4x4",
    "Mat4",
    "Quat",
]

