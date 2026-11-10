"""Pure Python linear algebra library for MiniGfx (zero external dependencies).

Storage and Coordinate Conventions:
- Right-handed Cartesian coordinates (+X right, +Y up, +Z towards viewer; view along -Z).
- Column-major matrix layout: m[col * 4 + row] stored as a flat tuple of 16 floats.
- Column-vector transform convention: v' = M * v.
- Clip-space Z range: [0.0, 1.0] (DirectX, Metal, Vulkan standard).
- std430 binary packing via .to_bytes() using IEEE-754 little-endian encoding.
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


class Mat4:
    """4x4 column-major transformation matrix."""

    __slots__ = ("_elements",)

    def __init__(self, *args: Any) -> None:
        if len(args) == 0:
            self._elements: tuple[float, ...] = (
                1.0, 0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 1.0,
            )
        elif len(args) == 1:
            t = tuple(float(x) for x in args[0])
            if len(t) != 16:
                raise ValueError(f"Mat4 requires 16 elements, got {len(t)}")
            self._elements = t
        elif len(args) == 16:
            self._elements = tuple(float(x) for x in args)
        else:
            raise ValueError(f"Mat4 requires 0, 1, or 16 arguments, got {len(args)}")

    @property
    def elements(self) -> tuple[float, ...]:
        return self._elements

    @property
    def columns(self) -> tuple[Vec4, Vec4, Vec4, Vec4]:
        e = self._elements
        return (
            Vec4(e[0], e[1], e[2], e[3]), Vec4(e[4], e[5], e[6], e[7]),
            Vec4(e[8], e[9], e[10], e[11]), Vec4(e[12], e[13], e[14], e[15]),
        )

    def __getitem__(self, idx: Union[int, tuple[int, int]]) -> float:
        return self._elements[idx[1] * 4 + idx[0]] if isinstance(idx, tuple) else self._elements[idx]

    def __iter__(self) -> Iterator[float]:
        return iter(self._elements)

    def __len__(self) -> int:
        return 16

    @classmethod
    def identity(cls) -> Mat4:
        return cls()

    @classmethod
    def zero(cls) -> Mat4:
        return cls((0.0,) * 16)

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

    def __mul__(self, o: Union[Mat4, Vec4]) -> Union[Mat4, Vec4]:
        a = self._elements
        if isinstance(o, Mat4):
            b, res = o._elements, [0.0] * 16
            for c in range(4):
                b0, b1, b2, b3 = b[c * 4], b[c * 4 + 1], b[c * 4 + 2], b[c * 4 + 3]
                for r in range(4):
                    res[c * 4 + r] = a[r] * b0 + a[4 + r] * b1 + a[8 + r] * b2 + a[12 + r] * b3
            return Mat4(res)
        if isinstance(o, Vec4):
            x, y, z, w = o.x, o.y, o.z, o.w
            return Vec4(
                a[0] * x + a[4] * y + a[8] * z + a[12] * w,
                a[1] * x + a[5] * y + a[9] * z + a[13] * w,
                a[2] * x + a[6] * y + a[10] * z + a[14] * w,
                a[3] * x + a[7] * y + a[11] * z + a[15] * w,
            )
        return NotImplemented

    def transpose(self) -> Mat4:
        m = self._elements
        return Mat4((
            m[0], m[4], m[8],  m[12],  m[1], m[5], m[9],  m[13],
            m[2], m[6], m[10], m[14],  m[3], m[7], m[11], m[15],
        ))

    def inverse(self) -> Mat4:
        m = self._elements
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
        return Mat4((
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

    def to_bytes(self) -> bytes:
        return struct.pack("<16f", *self._elements)

    @classmethod
    def from_bytes(cls, buffer: bytes | bytearray | memoryview) -> Mat4:
        if len(buffer) < 64:
            raise ValueError(f"Buffer size {len(buffer)} < 64 bytes for Mat4")
        return cls(struct.unpack_from("<16f", buffer, 0))

    def __eq__(self, o: object) -> bool:
        if isinstance(o, Mat4):
            return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(self._elements, o._elements))
        if isinstance(o, (tuple, list)) and len(o) == 16:
            return all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7) for a, b in zip(self._elements, o))
        return False

    def __repr__(self) -> str:
        return f"Mat4({self._elements})"


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
