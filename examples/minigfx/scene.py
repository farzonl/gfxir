"""Scene graph, mesh representation, procedural primitives, and frustum culling."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct
from typing import Any, Iterator, List, Optional, Sequence, Tuple

from .camera import Camera
from .math import Mat4x4, Quat, Vec2, Vec3, Vec4


@dataclass(slots=True)
class Vertex:
    """Interleaved vertex struct (pos: Vec3, normal: Vec3, uv: Vec2) in 32 bytes."""

    position: Vec3 = field(default_factory=Vec3)
    normal: Vec3 = field(default_factory=lambda: Vec3(0.0, 1.0, 0.0))
    uv: Vec2 = field(default_factory=Vec2)

    _is_gfx_struct: bool = True
    __struct_size__: int = 32
    __struct_alignment__: int = 4

    def to_bytes(self) -> bytes:
        p, n, u = self.position, self.normal, self.uv
        return struct.pack(
            "<8f",
            getattr(p, "x", p[0]), getattr(p, "y", p[1]), getattr(p, "z", p[2]),
            getattr(n, "x", n[0]), getattr(n, "y", n[1]), getattr(n, "z", n[2]),
            getattr(u, "x", u[0]), getattr(u, "y", u[1]),
        )

    @classmethod
    def from_bytes(cls, data: bytes | bytearray | memoryview) -> Vertex:
        if len(data) < 32:
            raise ValueError(f"Buffer size {len(data)} < 32 bytes for Vertex")
        px, py, pz, nx, ny, nz, ux, uy = struct.unpack_from("<8f", data, 0)
        return cls(Vec3(px, py, pz), Vec3(nx, ny, nz), Vec2(ux, uy))

    def as_parameter_block(self) -> bytes:
        return self.to_bytes()


class Mesh:
    """Indexed triangle mesh with vertex/index buffers and cached bounding sphere."""

    def __init__(
        self,
        vertices: Sequence[Vertex],
        indices: Sequence[int],
        center: Optional[Vec3] = None,
        radius: Optional[float] = None,
    ) -> None:
        self.vertices: List[Vertex] = list(vertices)
        self.indices: List[int] = [int(i) for i in indices]
        if center is not None and radius is not None:
            self._center, self._radius = center, float(radius)
        else:
            self._center, self._radius = self._compute_bounding_sphere()

    @property
    def center(self) -> Vec3:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def vertex_bytes(self) -> bytes:
        return b"".join(v.to_bytes() for v in self.vertices)

    def get_interleaved_vertex_bytes(self) -> bytes:
        return self.vertex_bytes()

    def index_bytes(self, force_u32: bool = False) -> bytes:
        if not self.indices:
            return b""
        if not force_u32 and max(self.indices) < 65536 and len(self.vertices) <= 65536:
            return struct.pack(f"<{len(self.indices)}H", *self.indices)
        return struct.pack(f"<{len(self.indices)}I", *self.indices)

    def _compute_bounding_sphere(self) -> Tuple[Vec3, float]:
        if not self.vertices:
            return Vec3(0.0, 0.0, 0.0), 0.0
        p0 = self.vertices[0].position
        p1 = max(self.vertices, key=lambda v: (v.position - p0).length_sq()).position
        p2 = max(self.vertices, key=lambda v: (v.position - p1).length_sq()).position
        center = (p1 + p2) * 0.5
        radius = (p2 - p1).length() * 0.5
        for v in self.vertices:
            d = (v.position - center).length()
            if d > radius:
                new_r = (radius + d) * 0.5
                center = center + (v.position - center) * ((new_r - radius) / d)
                radius = new_r
        return center, radius


def create_cube(size: float = 1.0) -> Mesh:
    """Generate cube with 24 vertices, 36 indices, and distinct face normals."""
    h = size * 0.5
    faces = [
        (Vec3(0, 0, 1), [Vec3(-h, -h, h), Vec3(h, -h, h), Vec3(h, h, h), Vec3(-h, h, h)]),
        (Vec3(0, 0, -1), [Vec3(h, -h, -h), Vec3(-h, -h, -h), Vec3(-h, h, -h), Vec3(h, h, -h)]),
        (Vec3(1, 0, 0), [Vec3(h, -h, h), Vec3(h, -h, -h), Vec3(h, h, -h), Vec3(h, h, h)]),
        (Vec3(-1, 0, 0), [Vec3(-h, -h, -h), Vec3(-h, -h, h), Vec3(-h, h, h), Vec3(-h, h, -h)]),
        (Vec3(0, 1, 0), [Vec3(-h, h, h), Vec3(h, h, h), Vec3(h, h, -h), Vec3(-h, h, -h)]),
        (Vec3(0, -1, 0), [Vec3(-h, -h, -h), Vec3(h, -h, -h), Vec3(h, -h, h), Vec3(-h, -h, h)]),
    ]
    uvs = (Vec2(0.0, 0.0), Vec2(1.0, 0.0), Vec2(1.0, 1.0), Vec2(0.0, 1.0))
    vertices, indices = [], []
    for norm, quad in faces:
        b = len(vertices)
        vertices.extend(Vertex(p, norm, uv) for p, uv in zip(quad, uvs))
        indices.extend([b, b + 1, b + 2, b, b + 2, b + 3])
    return Mesh(vertices, indices, center=Vec3(0.0, 0.0, 0.0), radius=size * math.sqrt(3.0) * 0.5)


def create_sphere(radius: float = 1.0, subdivisions: int = 16) -> Mesh:
    """Generate UV sphere with normalized outward normals and CCW winding."""
    if subdivisions < 1:
        raise ValueError(f"Sphere subdivisions must be >= 1, got {subdivisions}")
    lat_count, lon_count = subdivisions, max(3, subdivisions * 2)
    vertices = []
    for i in range(lat_count + 1):
        th = math.pi * i / lat_count
        y, r_xz = radius * math.cos(th), radius * math.sin(th)
        for j in range(lon_count + 1):
            phi = 2.0 * math.pi * j / lon_count
            x, z = r_xz * math.sin(phi), r_xz * math.cos(phi)
            pos = Vec3(x, y, z)
            norm = Vec3(x / radius, y / radius, z / radius) if radius > 0 else Vec3(0, 1, 0)
            vertices.append(Vertex(pos, norm, Vec2(float(j) / lon_count, float(i) / lat_count)))
    indices, stride = [], lon_count + 1
    for i in range(lat_count):
        for j in range(lon_count):
            first, second = i * stride + j, i * stride + j + stride
            if i == 0:
                indices.extend([first, second, second + 1])
            elif i == lat_count - 1:
                indices.extend([first, second, first + 1])
            else:
                indices.extend([first, second, second + 1, first, second + 1, first + 1])
    return Mesh(vertices, indices, center=Vec3(0.0, 0.0, 0.0), radius=float(radius))


def create_plane(width: float = 10.0, depth: float = 10.0) -> Mesh:
    """Generate quad plane on XZ plane with +Y normal."""
    hw, hd = width * 0.5, depth * 0.5
    norm = Vec3(0.0, 1.0, 0.0)
    vertices = [
        Vertex(Vec3(-hw, 0.0, hd), norm, Vec2(0.0, 0.0)),
        Vertex(Vec3(hw, 0.0, hd), norm, Vec2(1.0, 0.0)),
        Vertex(Vec3(hw, 0.0, -hd), norm, Vec2(1.0, 1.0)),
        Vertex(Vec3(-hw, 0.0, -hd), norm, Vec2(0.0, 1.0)),
    ]
    indices = [0, 1, 2, 0, 2, 3]
    return Mesh(vertices, indices, center=Vec3(0.0, 0.0, 0.0), radius=0.5 * math.hypot(width, depth))


class Frustum:
    """View frustum with 6 bounding planes (Left, Right, Bottom, Top, Near, Far)."""

    def __init__(self, planes: Optional[Sequence[Vec4]] = None) -> None:
        self.planes: Tuple[Vec4, ...] = tuple(planes) if planes is not None else Camera().frustum_planes

    @classmethod
    def from_matrix(cls, mat4: Mat4x4) -> Frustum:
        m = mat4.elements
        r = [(m[i], m[4 + i], m[8 + i], m[12 + i]) for i in range(4)]
        def _norm(a: float, b: float, c: float, d: float) -> Vec4:
            inv = 1.0 / (math.sqrt(a * a + b * b + c * c) or 1.0)
            return Vec4(a * inv, b * inv, c * inv, d * inv)
        return cls((
            _norm(r[3][0] + r[0][0], r[3][1] + r[0][1], r[3][2] + r[0][2], r[3][3] + r[0][3]),
            _norm(r[3][0] - r[0][0], r[3][1] - r[0][1], r[3][2] - r[0][2], r[3][3] - r[0][3]),
            _norm(r[3][0] + r[1][0], r[3][1] + r[1][1], r[3][2] + r[1][2], r[3][3] + r[1][3]),
            _norm(r[3][0] - r[1][0], r[3][1] - r[1][1], r[3][2] - r[1][2], r[3][3] - r[1][3]),
            _norm(r[2][0], r[2][1], r[2][2], r[2][3]),
            _norm(r[3][0] - r[2][0], r[3][1] - r[2][1], r[3][2] - r[2][2], r[3][3] - r[2][3]),
        ))

    def intersects_sphere(self, center: Sequence[float] | Vec3, radius: float) -> bool:
        cx, cy, cz = getattr(center, "x", center[0]), getattr(center, "y", center[1]), getattr(center, "z", center[2])
        r = float(radius)
        return all(p.x * cx + p.y * cy + p.z * cz + p.w >= -r - 1e-7 for p in self.planes)


def _matrix3x3_spectral_norm(wt: Sequence[float]) -> float:
    """Computes the exact operator 2-norm (largest singular value) of the 3x3 block of a 4x4 matrix.

    Uses a fast path for orthogonal columns (un-sheared scale/rotation), falling back to
    Smith's exact analytical trigonometric eigenvalue algorithm on A^T * A for sheared matrices.
    """
    a00, a10, a20 = wt[0], wt[1], wt[2]
    a01, a11, a21 = wt[4], wt[5], wt[6]
    a02, a12, a22 = wt[8], wt[9], wt[10]

    b00 = a00 * a00 + a10 * a10 + a20 * a20
    b11 = a01 * a01 + a11 * a11 + a21 * a21
    b22 = a02 * a02 + a12 * a12 + a22 * a22

    b01 = a00 * a01 + a10 * a11 + a20 * a21
    b02 = a00 * a02 + a10 * a12 + a20 * a22
    b12 = a01 * a02 + a11 * a12 + a21 * a22

    off_sq = b01 * b01 + b02 * b02 + b12 * b12
    max_diag = max(b00, b11, b22)
    if off_sq <= 1e-14 * max_diag * max_diag:
        return math.sqrt(max_diag)

    m = (b00 + b11 + b22) * (1.0 / 3.0)
    k00 = b00 - m
    k11 = b11 - m
    k22 = b22 - m

    q = (k00 * k00 + k11 * k11 + k22 * k22) * 0.5 + off_sq
    p = math.sqrt(q * (1.0 / 3.0))

    if p <= m * 1e-12:
        return math.sqrt(m)

    detK = (
        k00 * (k11 * k22 - b12 * b12)
        - b01 * (b01 * k22 - b02 * b12)
        + b02 * (b01 * b12 - b02 * k11)
    )

    r = detK / (2.0 * p * p * p)
    if r <= -1.0:
        phi = math.pi * (1.0 / 3.0)
    elif r >= 1.0:
        phi = 0.0
    else:
        phi = math.acos(r) * (1.0 / 3.0)

    eig_max = m + 2.0 * p * math.cos(phi)
    return math.sqrt(max(0.0, eig_max))


class Node:
    """Scene graph node managing hierarchy, local/world transforms, and visibility."""

    def __init__(
        self,
        name: str = "node",
        translation: Optional[Sequence[float] | Vec3] = None,
        rotation: Optional[Quat] = None,
        scale: Optional[Sequence[float] | Vec3] = None,
        mesh: Optional[Mesh] = None,
        *,
        position: Optional[Sequence[float] | Vec3] = None,
        bounding_radius: Optional[float] = None,
        transform: Optional[Mat4x4] = None,
        **kwargs: Any,
    ) -> None:
        self.name = str(name)
        pos = translation if translation is not None else (position if position is not None else kwargs.get("position"))
        self._translation = self._to_vec3(pos) if pos is not None else Vec3(0.0, 0.0, 0.0)
        self._rotation = rotation if rotation is not None else kwargs.get("rotation", Quat.identity())
        sc = scale if scale is not None else kwargs.get("scale")
        self._scale = self._to_vec3(sc) if sc is not None else Vec3(1.0, 1.0, 1.0)
        self.mesh: Optional[Mesh] = mesh if mesh is not None else kwargs.get("mesh")
        br = bounding_radius if bounding_radius is not None else kwargs.get("bounding_radius")
        self._bounding_radius: Optional[float] = float(br) if br is not None else None
        self._custom_transform: Optional[Mat4x4] = transform if transform is not None else kwargs.get("transform")
        self.parent: Optional[Node] = None
        self.children: List[Node] = []
        self._dirty: bool = True
        self._cached_world_transform: Optional[Mat4x4] = None

    @staticmethod
    def _to_vec3(v: Sequence[float] | Vec3) -> Vec3:
        return v if isinstance(v, Vec3) else Vec3(float(v[0]), float(v[1]), float(v[2]))

    @property
    def translation(self) -> Vec3:
        return self._translation

    @translation.setter
    def translation(self, val: Sequence[float] | Vec3) -> None:
        self._translation = self._to_vec3(val)
        self._mark_dirty()

    @property
    def position(self) -> Vec3:
        return self._translation

    @position.setter
    def position(self, val: Sequence[float] | Vec3) -> None:
        self.translation = val

    @property
    def rotation(self) -> Quat:
        return self._rotation

    @rotation.setter
    def rotation(self, val: Quat) -> None:
        self._rotation = val
        self._mark_dirty()

    @property
    def scale(self) -> Vec3:
        return self._scale

    @scale.setter
    def scale(self, val: Sequence[float] | Vec3) -> None:
        self._scale = self._to_vec3(val)
        self._mark_dirty()

    @property
    def transform(self) -> Mat4x4:
        return self._custom_transform if self._custom_transform is not None else self.local_transform

    @transform.setter
    def transform(self, val: Optional[Mat4x4]) -> None:
        self._custom_transform = val
        self._mark_dirty()

    @property
    def local_transform(self) -> Mat4x4:
        if self._custom_transform is not None:
            return self._custom_transform
        return Mat4x4.translation(self._translation) * self._rotation.to_mat4x4() * Mat4x4.scaling(self._scale)

    @property
    def world_transform(self) -> Mat4x4:
        if not self._dirty and self._cached_world_transform is not None:
            return self._cached_world_transform

        chain: List[Node] = []
        curr: Optional[Node] = self
        while curr is not None and (curr._dirty or curr._cached_world_transform is None):
            chain.append(curr)
            curr = curr.parent

        parent_transform = curr._cached_world_transform if curr is not None else None
        for node in reversed(chain):
            local = node.local_transform
            parent_transform = parent_transform * local if parent_transform is not None else local
            node._cached_world_transform = parent_transform
            node._dirty = False

        return self._cached_world_transform  # type: ignore[return-value]

    @property
    def world_position(self) -> Vec3:
        wt = self.world_transform.elements
        return Vec3(wt[12], wt[13], wt[14])

    @property
    def bounding_radius(self) -> float:
        r = self._bounding_radius if self._bounding_radius is not None else (self.mesh.radius if self.mesh else 0.0)
        return r * _matrix3x3_spectral_norm(self.world_transform.elements)

    @bounding_radius.setter
    def bounding_radius(self, val: float) -> None:
        self._bounding_radius = float(val)

    @property
    def bounding_sphere(self) -> Tuple[Vec3, float]:
        local_c = self.mesh.center if self.mesh is not None else Vec3(0.0, 0.0, 0.0)
        c4 = self.world_transform * Vec4(local_c.x, local_c.y, local_c.z, 1.0)
        return Vec3(c4.x, c4.y, c4.z), self.bounding_radius

    def _mark_dirty(self) -> None:
        stack = [self]
        while stack:
            node = stack.pop()
            node._dirty = True
            node._cached_world_transform = None
            stack.extend(node.children)

    def invalidate(self) -> None:
        self._mark_dirty()

    def add_child(self, child: Node) -> None:
        curr: Optional[Node] = self
        while curr is not None:
            if curr is child:
                raise ValueError(f"Cannot add ancestor '{child.name}' as child to '{self.name}' (cycle detected)")
            curr = curr.parent
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        self.children.append(child)
        child._mark_dirty()

    def remove_child(self, child: Node) -> None:
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            child._mark_dirty()

    def is_visible(self, camera: Camera) -> bool:
        center, radius = self.bounding_sphere
        return camera.is_sphere_visible(center, radius)

    def traverse(self) -> Iterator[Node]:
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            stack.extend(reversed(node.children))

    def collect_visible(self, camera: Camera) -> List[Node]:
        return [node for node in self.traverse() if node.is_visible(camera)]


class Scene(Node):
    """Root node of a scene graph hierarchy."""

    def __init__(self, name: str = "scene") -> None:
        super().__init__(name=name)
