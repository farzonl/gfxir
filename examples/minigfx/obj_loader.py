"""Lightweight Wavefront OBJ parser generating interleaved Mesh instances."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

from .math import Vec2, Vec3
from .scene import Mesh, Vertex


def _resolve_index(raw_idx: int, count: int) -> int:
    if raw_idx == 0:
        raise ValueError("OBJ index 0 is invalid")
    idx = raw_idx - 1 if raw_idx > 0 else count + raw_idx
    if not (0 <= idx < count):
        raise ValueError(f"OBJ index out of bounds: {raw_idx} (count={count})")
    return idx


def load_obj_from_string(text: str, *, smooth_normals: bool = False) -> Mesh:
    """Parse Wavefront OBJ string into a Mesh with triangulated faces and deduplicated vertices."""
    positions: List[Vec3] = []
    normals: List[Vec3] = []
    uvs: List[Vec2] = []
    triangles: List[Tuple[Tuple[int, int, int], Tuple[int, int, int], Tuple[int, int, int]]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        tokens = line.split()
        cmd = tokens[0]
        if cmd == "v":
            if len(tokens) < 4:
                raise ValueError(f"Malformed vertex line: {line}")
            positions.append(Vec3(float(tokens[1]), float(tokens[2]), float(tokens[3])))
        elif cmd == "vn":
            if len(tokens) < 4:
                raise ValueError(f"Malformed normal line: {line}")
            normals.append(Vec3(float(tokens[1]), float(tokens[2]), float(tokens[3])).normalize())
        elif cmd == "vt":
            if len(tokens) < 2:
                raise ValueError(f"Malformed uv line: {line}")
            uvs.append(Vec2(float(tokens[1]), float(tokens[2]) if len(tokens) > 2 else 0.0))
        elif cmd == "f":
            specs = tokens[1:]
            if len(specs) < 3:
                raise ValueError(f"Face requires at least 3 vertices: {line}")
            face: List[Tuple[int, int, int]] = []
            for spec in specs:
                parts = spec.split("/")
                v_idx = _resolve_index(int(parts[0]), len(positions))
                vt_idx = _resolve_index(int(parts[1]), len(uvs)) if len(parts) > 1 and parts[1] else -1
                vn_idx = _resolve_index(int(parts[2]), len(normals)) if len(parts) > 2 and parts[2] else -1
                face.append((v_idx, vt_idx, vn_idx))
            for i in range(1, len(face) - 1):
                triangles.append((face[0], face[i], face[i + 1]))

    smooth_dict: Dict[int, Vec3] = {}
    if smooth_normals:
        for tri in triangles:
            p0, p1, p2 = positions[tri[0][0]], positions[tri[1][0]], positions[tri[2][0]]
            fn = (p1 - p0).cross(p2 - p0)
            for v_idx, _, vn_idx in tri:
                if vn_idx == -1:
                    smooth_dict[v_idx] = smooth_dict.get(v_idx, Vec3(0.0, 0.0, 0.0)) + fn
        smooth_dict = {
            k: v.normalize() if v.length_sq() > 1e-12 else Vec3(0.0, 1.0, 0.0)
            for k, v in smooth_dict.items()
        }

    mesh_vertices: List[Vertex] = []
    mesh_indices: List[int] = []
    vert_cache: Dict[Tuple[int, int, Any], int] = {}

    for tri in triangles:
        p0, p1, p2 = positions[tri[0][0]], positions[tri[1][0]], positions[tri[2][0]]
        fn = (p1 - p0).cross(p2 - p0)
        fn = fn.normalize() if fn.length_sq() > 1e-12 else Vec3(0.0, 1.0, 0.0)
        for v_idx, vt_idx, vn_idx in tri:
            if vn_idx != -1:
                norm_key: Any = ("idx", vn_idx)
                norm = normals[vn_idx]
            elif smooth_normals:
                norm_key = ("smooth", v_idx)
                norm = smooth_dict.get(v_idx, Vec3(0.0, 1.0, 0.0))
            else:
                norm_key = ("flat", round(fn.x, 5), round(fn.y, 5), round(fn.z, 5))
                norm = fn
            key = (v_idx, vt_idx, norm_key)
            if key not in vert_cache:
                vert_cache[key] = len(mesh_vertices)
                pos = positions[v_idx]
                uv = uvs[vt_idx] if vt_idx != -1 else Vec2(0.0, 0.0)
                mesh_vertices.append(Vertex(pos, norm, uv))
            mesh_indices.append(vert_cache[key])

    return Mesh(mesh_vertices, mesh_indices)


def load_obj(path_or_str: str, *, smooth_normals: bool = False) -> Mesh:
    """Load OBJ mesh from a file path or in-memory string."""
    if "\n" not in path_or_str and len(path_or_str) < 512 and os.path.isfile(path_or_str):
        with open(path_or_str, "r", encoding="utf-8") as f:
            return load_obj_from_string(f.read(), smooth_normals=smooth_normals)
    return load_obj_from_string(path_or_str, smooth_normals=smooth_normals)
