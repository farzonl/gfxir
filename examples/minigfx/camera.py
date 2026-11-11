"""Camera system, frustum extraction, shared params, and orbit controller."""

import math
import struct
from typing import Optional, Sequence, Tuple

from .math import Mat4x4, Vec3, Vec4

try:
    from mlir_gpu.parameters import struct as _gfx_struct
except ImportError:
    _gfx_struct = None


class Camera:
    """3D perspective camera producing view, projection, and view-projection matrices."""

    def __init__(
        self,
        eye: Optional[Vec3] = None,
        target: Optional[Vec3] = None,
        up: Optional[Vec3] = None,
        fov_y: float = math.radians(60.0),
        aspect: float = 16.0 / 9.0,
        near: float = 0.1,
        far: float = 100.0,
    ) -> None:
        self.eye = eye if eye is not None else Vec3(0.0, 0.0, 5.0)
        self.target = target if target is not None else Vec3(0.0, 0.0, 0.0)
        self.up = up if up is not None else Vec3(0.0, 1.0, 0.0)
        self._fov_y = math.radians(fov_y) if fov_y > math.pi else float(fov_y)
        self.aspect, self.near, self.far = float(aspect), float(near), float(far)

    @property
    def fov_y(self) -> float:
        return self._fov_y

    @fov_y.setter
    def fov_y(self, val: float) -> None:
        fval = float(val)
        self._fov_y = math.radians(fval) if fval > math.pi else fval

    @property
    def view_matrix(self) -> Mat4x4:
        return Mat4x4.look_at(self.eye, self.target, self.up)

    @property
    def proj_matrix(self) -> Mat4x4:
        return Mat4x4.perspective(self._fov_y, self.aspect, self.near, self.far)

    @property
    def view_proj(self) -> Mat4x4:
        return self.proj_matrix * self.view_matrix

    @property
    def frustum_planes(self) -> Tuple[Vec4, Vec4, Vec4, Vec4, Vec4, Vec4]:
        """Extract 6 frustum planes (Left, Right, Bottom, Top, Near, Far) via Gribb-Hartmann."""
        m = self.view_proj.elements
        r0 = (m[0], m[4], m[8], m[12])
        r1 = (m[1], m[5], m[9], m[13])
        r2 = (m[2], m[6], m[10], m[14])
        r3 = (m[3], m[7], m[11], m[15])

        def _norm(a: float, b: float, c: float, d: float) -> Vec4:
            inv = 1.0 / (math.sqrt(a * a + b * b + c * c) or 1.0)
            return Vec4(a * inv, b * inv, c * inv, d * inv)

        return (
            _norm(r3[0] + r0[0], r3[1] + r0[1], r3[2] + r0[2], r3[3] + r0[3]),  # Left
            _norm(r3[0] - r0[0], r3[1] - r0[1], r3[2] - r0[2], r3[3] - r0[3]),  # Right
            _norm(r3[0] + r1[0], r3[1] + r1[1], r3[2] + r1[2], r3[3] + r1[3]),  # Bottom
            _norm(r3[0] - r1[0], r3[1] - r1[1], r3[2] - r1[2], r3[3] - r1[3]),  # Top
            _norm(r2[0], r2[1], r2[2], r2[3]),                                  # Near ([0, 1] Z)
            _norm(r3[0] - r2[0], r3[1] - r2[1], r3[2] - r2[2], r3[3] - r2[3]),  # Far ([0, 1] Z)
        )

    def extract_frustum_planes(self) -> Tuple[Vec4, Vec4, Vec4, Vec4, Vec4, Vec4]:
        return self.frustum_planes

    def is_sphere_visible(self, center: Sequence[float] | Vec3, radius: float) -> bool:
        """Return True if bounding sphere intersects or is inside the frustum."""
        cx = getattr(center, "x", center[0])
        cy = getattr(center, "y", center[1])
        cz = getattr(center, "z", center[2])
        r = float(radius)
        return all(p.x * cx + p.y * cy + p.z * cz + p.w >= -r - 1e-7 for p in self.frustum_planes)

    def get_params(self, time: float = 0.0) -> CameraParams:
        return CameraParams.from_camera(self, time=time)


if _gfx_struct is not None:
    @_gfx_struct
    class CameraParams:
        view_proj: Mat4x4
        eye_pos: Vec3
        time: float = 0.0

    @classmethod
    def _from_camera(cls, camera: Camera, time: float = 0.0) -> CameraParams:
        return cls(camera.view_proj, camera.eye, time)
    CameraParams.from_camera = _from_camera

    _orig_from_bytes = CameraParams.from_bytes
    @classmethod
    def _from_bytes(cls, buffer: bytes | bytearray | memoryview) -> CameraParams:
        inst = _orig_from_bytes(buffer)
        inst.view_proj = Mat4x4(inst.view_proj) if not isinstance(inst.view_proj, Mat4x4) else inst.view_proj
        inst.eye_pos = Vec3(*inst.eye_pos) if not isinstance(inst.eye_pos, Vec3) else inst.eye_pos
        return inst
    CameraParams.from_bytes = _from_bytes

else:
    class CameraParams:
        """Shared host/device camera parameter block (std430, 80 bytes)."""

        __struct_size__ = 80
        __struct_alignment__ = 16
        _is_gfx_struct = True

        def __init__(
            self,
            view_proj: Mat4x4 | Sequence[float],
            eye_pos: Vec3 | Sequence[float],
            time: float = 0.0,
        ) -> None:
            self.view_proj = view_proj if isinstance(view_proj, Mat4x4) else Mat4x4(view_proj)
            self.eye_pos = eye_pos if isinstance(eye_pos, Vec3) else Vec3(*eye_pos)
            self.time = float(time)

        def to_bytes(self) -> bytes:
            data = bytearray(80)
            data[0:64] = self.view_proj.to_bytes()
            struct.pack_into("<3f", data, 64, self.eye_pos.x, self.eye_pos.y, self.eye_pos.z)
            struct.pack_into("<f", data, 76, self.time)
            return bytes(data)

        @classmethod
        def from_bytes(cls, buffer: bytes | bytearray | memoryview) -> CameraParams:
            if len(buffer) < 80:
                raise ValueError(f"Buffer size {len(buffer)} < 80 bytes")
            vp = Mat4x4.from_bytes(buffer[0:64])
            eye = Vec3(*struct.unpack_from("<3f", buffer, 64))
            return cls(vp, eye, struct.unpack_from("<f", buffer, 76)[0])

        @classmethod
        def from_camera(cls, camera: Camera, time: float = 0.0) -> CameraParams:
            return cls(camera.view_proj, camera.eye, time)

        def __repr__(self) -> str:
            return f"CameraParams(view_proj={self.view_proj!r}, eye_pos={self.eye_pos!r}, time={self.time!r})"

        def __eq__(self, other: object) -> bool:
            if not isinstance(other, CameraParams):
                return False
            return self.view_proj == other.view_proj and self.eye_pos == other.eye_pos and math.isclose(self.time, other.time)


class OrbitController:
    """Smooth orbital camera controller with damping, rotation, panning, and zoom."""

    def __init__(
        self,
        target: Optional[Vec3] = None,
        distance: float = 5.0,
        yaw: float = 0.0,
        pitch: float = 0.0,
        min_distance: float = 0.5,
        max_distance: float = 100.0,
        min_pitch: float = -1.5,
        max_pitch: float = 1.5,
        damping: float = 0.1,
    ) -> None:
        self.target = target if target is not None else Vec3(0.0, 0.0, 0.0)
        self.distance, self.yaw, self.pitch = float(distance), float(yaw), float(pitch)
        self.min_distance, self.max_distance = float(min_distance), float(max_distance)
        self.min_pitch, self.max_pitch, self.damping = float(min_pitch), float(max_pitch), float(damping)

        self._target_pos: Vec3 = self.target
        self._target_distance: float = max(self.min_distance, min(self.max_distance, self.distance))
        self._target_yaw: float = self.yaw
        self._target_pitch: float = max(self.min_pitch, min(self.max_pitch, self.pitch))

    @classmethod
    def from_camera(cls, camera: Camera, damping: float = 0.1) -> OrbitController:
        offset = camera.eye - camera.target
        dist = offset.length()
        if dist < 1e-6:
            return cls(target=camera.target, distance=1.0, damping=damping)
        pitch = math.asin(max(-1.0, min(1.0, offset.y / dist)))
        yaw = math.atan2(offset.x, offset.z)
        return cls(target=camera.target, distance=dist, yaw=yaw, pitch=pitch, damping=damping)

    def rotate(self, delta_x: float, delta_y: float) -> None:
        """Adjust target yaw and pitch angles."""
        self._target_yaw += float(delta_x)
        self._target_pitch = max(self.min_pitch, min(self.max_pitch, self._target_pitch + float(delta_y)))
        if self.damping <= 0.0:
            self.yaw, self.pitch = self._target_yaw, self._target_pitch

    def pan(self, delta_x: float, delta_y: float, camera: Optional[Camera] = None) -> None:
        """Translate target in camera local X/Y plane."""
        if camera is not None:
            fwd = camera.target - camera.eye
            f_dir = fwd.normalize()
            r_dir = f_dir.cross(camera.up).normalize()
            u_dir = r_dir.cross(f_dir).normalize()
            delta_p = r_dir * float(delta_x) + u_dir * float(delta_y)
        else:
            delta_p = Vec3(float(delta_x), float(delta_y), 0.0)

        self._target_pos = self._target_pos + delta_p
        if self.damping <= 0.0:
            self.target = self._target_pos

    def zoom(self, delta: float) -> None:
        """Adjust target distance with min/max clamps."""
        self._target_distance = max(self.min_distance, min(self.max_distance, self._target_distance + float(delta)))
        if self.damping <= 0.0:
            self.distance = self._target_distance

    def update(self, camera: Camera, dt: float) -> None:
        """Update controller state with smooth damping and update camera eye and target."""
        if self.damping <= 0.0:
            alpha = 1.0
        elif dt <= 0.0:
            alpha = 0.0
        else:
            alpha = max(0.0, min(1.0, 1.0 - math.exp(-float(dt) / self.damping)))

        self.yaw += (self._target_yaw - self.yaw) * alpha
        self.pitch += (self._target_pitch - self.pitch) * alpha
        self.distance += (self._target_distance - self.distance) * alpha
        self.target = Vec3(
            self.target.x + (self._target_pos.x - self.target.x) * alpha,
            self.target.y + (self._target_pos.y - self.target.y) * alpha,
            self.target.z + (self._target_pos.z - self.target.z) * alpha,
        )

        cp, sp, cy, sy = math.cos(self.pitch), math.sin(self.pitch), math.cos(self.yaw), math.sin(self.yaw)
        camera.eye = Vec3(
            self.target.x + self.distance * cp * sy,
            self.target.y + self.distance * sp,
            self.target.z + self.distance * cp * cy,
        )
        camera.target = self.target
