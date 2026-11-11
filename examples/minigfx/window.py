"""Interactive GPU window presenter for MiniGfx (Metal and Vulkan via GfxIR)."""

from __future__ import annotations

import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from examples.minigfx.debug import GLYPH_TABLE
from examples.minigfx.material import CameraParams, LightParams, MaterialParams, ObjectParams
from examples.minigfx.math import Mat4x4, Vec3, Vec4
from examples.minigfx.scene import Mesh

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def _ensure_toolchain_paths() -> None:
    """Ensure local GfxIR compiler, LLVM, and helper binary directories are on PATH."""
    extra_dirs: List[Path] = []
    for env_key in ("LLVM_BIN", "GPU_TOOLS_BIN"):
        val = os.environ.get(env_key)
        if val:
            extra_dirs.append(Path(val))
    vk_sdk = os.environ.get("VULKAN_SDK")
    if vk_sdk:
        extra_dirs.append(Path(vk_sdk) / "bin")
    extra_dirs.extend(
        [
            _REPO_ROOT / "build" / "compiler" / "bin",
            _REPO_ROOT / ".toolchains" / "llvm-build" / "bin",
            _REPO_ROOT / ".tools" / "bin",
            _REPO_ROOT / ".sdk" / "bin",
            Path("/usr/local/bin"),
            Path("/opt/homebrew/bin"),
        ]
    )
    current = os.environ.get("PATH", "")
    parts = current.split(os.pathsep) if current else []
    added = [str(d) for d in extra_dirs if d.is_dir() and str(d) not in parts]
    if added:
        os.environ["PATH"] = os.pathsep.join(added + parts)


def pack_mesh_triangles(mesh: Mesh) -> Tuple[bytes, int]:
    """Expand an indexed Mesh into a flat storage buffer of 2x vec4 per triangle vertex.

    Layout per vertex (32 bytes = 2 x vec4<f32>):
      vec4[2*i + 0] = (pos.x, pos.y, pos.z, uv.x)
      vec4[2*i + 1] = (norm.x, norm.y, norm.z, uv.y)
    Returns (raw_bytes, vertex_count).
    """
    verts = mesh.vertices
    indices = mesh.indices
    if not verts:
        return b"", 0

    seq = indices if indices else list(range(len(verts)))
    usable = (len(seq) // 3) * 3
    if usable == 0:
        return b"", 0

    out = bytearray(usable * 32)
    pack_into = struct.pack_into
    offset = 0
    for idx in seq[:usable]:
        v = verts[idx]
        p, n, u = v.position, v.normal, v.uv
        pack_into(
            "<8f",
            out,
            offset,
            float(p.x),
            float(p.y),
            float(p.z),
            float(u.x),
            float(n.x),
            float(n.y),
            float(n.z),
            float(u.y),
        )
        offset += 32
    return bytes(out), usable


def build_hud_overlay_vertices(
    lines: Sequence[Tuple[str, Tuple[float, float, float, float]]],
    fb_width: int,
    fb_height: int,
    pixel_scale: int = 2,
) -> Tuple[bytes, int]:
    """Build 2D clip-space triangle vertices (pos vec4 + color vec4) for bitmap HUD text."""
    if fb_width <= 0 or fb_height <= 0 or not lines:
        return b"", 0

    s = max(1, int(pixel_scale))
    pad = 8 * s
    start_x = 10 * s
    start_y = 10 * s
    adv_x = 6 * s
    line_h = 10 * s

    max_chars = max((len(text) for text, _ in lines), default=0)
    if max_chars == 0:
        return b"", 0

    box_x0 = start_x - pad
    box_y0 = start_y - pad
    box_x1 = start_x + max_chars * adv_x + pad
    box_y1 = start_y + len(lines) * line_h - (3 * s) + pad

    inv_w = 2.0 / float(fb_width)
    inv_h = 2.0 / float(fb_height)

    quads: List[Tuple[float, float, float, float, float, float, float, float]] = []
    # Translucent dark slate backdrop panel
    quads.append((float(box_x0), float(box_y0), float(box_x1), float(box_y1), 0.04, 0.05, 0.09, 0.80))

    cur_y = start_y
    for text, color in lines:
        cr, cg, cb, ca = color
        cur_x = start_x
        for ch in text:
            if ch == "\n":
                cur_x = start_x
                cur_y += line_h
                continue
            if ch == " ":
                cur_x += adv_x
                continue
            glyph = GLYPH_TABLE.get(ch, GLYPH_TABLE.get("?", (0, 0, 0, 0, 0)))
            for row in range(7):
                col = 0
                while col < 5:
                    if (glyph[col] >> row) & 1:
                        run_end = col + 1
                        while run_end < 5 and ((glyph[run_end] >> row) & 1):
                            run_end += 1
                        px0 = float(cur_x + col * s)
                        py0 = float(cur_y + row * s)
                        px1 = float(cur_x + run_end * s)
                        py1 = float(py0 + s)
                        quads.append((px0, py0, px1, py1, cr, cg, cb, ca))
                        col = run_end
                    else:
                        col += 1
            cur_x += adv_x
        cur_y += line_h

    vert_count = len(quads) * 6
    buf = bytearray(vert_count * 32)
    pack_into = struct.pack_into
    off = 0
    for x0, y0, x1, y1, r, g, b, a in quads:
        nx0 = x0 * inv_w - 1.0
        nx1 = x1 * inv_w - 1.0
        ny0 = 1.0 - y0 * inv_h
        ny1 = 1.0 - y1 * inv_h
        # Two CCW triangles: (nx0, ny0)-(nx0, ny1)-(nx1, ny1) and (nx0, ny0)-(nx1, ny1)-(nx1, ny0)
        for vx, vy in ((nx0, ny0), (nx0, ny1), (nx1, ny1), (nx0, ny0), (nx1, ny1), (nx1, ny0)):
            pack_into("<8f", buf, off, vx, vy, 0.0, 1.0, r, g, b, a)
            off += 32

    return bytes(buf), vert_count


def compile_minigfx_shaders(target_name: str) -> Dict[str, Path]:
    """Compile embedded MiniGfx Python shaders via GfxIR to native GPU binaries (.metallib or .spv)."""
    _ensure_toolchain_paths()
    from mlir_gpu.ast_frontend import ASTLowering
    from mlir_gpu.compiler import Target, TargetProfile, Toolchain, _compile_stage
    from mlir_gpu.frontend import Buffer, Node, ShaderIO, Stage, Type, Value, Vertex, _node, fragment, u32, vec2, vec3, vec4
    import mlir_gpu.shader_math as sm

    target = Target(target_name)
    ext = "spv" if target == Target.VULKAN else "metallib"

    cache_key_src = f"minigfx_window_shaders_v2_{target_name}"
    digest = hashlib.sha256(cache_key_src.encode("utf-8")).hexdigest()[:12]
    cache_dir = _REPO_ROOT / "build" / "minigfx_cache" / f"shaders_{target_name}_{digest}"

    expected = {
        "mesh_vs": cache_dir / f"mesh_vs.{ext}",
        "pbr_fs": cache_dir / f"pbr_fs.{ext}",
        "normal_fs": cache_dir / f"normal_fs.{ext}",
        "overlay_vs": cache_dir / f"overlay_vs.{ext}",
        "overlay_fs": cache_dir / f"overlay_fs.{ext}",
    }
    if all(p.is_file() and p.stat().st_size > 0 for p in expected.values()):
        return expected

    def _mesh_vs(
        vertex_index: u32,
        verts: Buffer[Type.VEC4, "read_only"],
        cam: CameraParams,
        obj: ObjectParams,
    ) -> Vertex:
        p = verts[vertex_index * u32(2)]
        n = verts[vertex_index * u32(2) + u32(1)]
        pos = vec4(p.xyz, 1.0)
        world_pos = obj.model * pos
        world_norm = (obj.model * vec4(n.xyz, 0.0)).xyz
        clip_pos = cam.view_proj * world_pos
        return Vertex(
            clip_pos,
            varyings={"normal": world_norm, "world_pos": world_pos.xyz, "uv": vec2(p.w, n.w)},
        )

    @fragment
    def _pbr_fs(
        normal: Value,
        world_pos: Value,
        uv: Value,
        cam: CameraParams,
        mat: MaterialParams,
        light: LightParams,
    ) -> Value:
        n = sm.normalize(normal)
        l = sm.normalize(light.direction)
        v = sm.normalize(cam.eye_pos - world_pos)

        n_dot_l = sm.clamp(sm.dot(n, l), 0.0, 1.0)
        diffuse = mat.albedo.xyz * n_dot_l * light.intensity * (1.0 - mat.metallic) * light.color

        hemi = n.y * 0.35 + 0.65
        ambient = mat.albedo.xyz * (light.ambient * hemi + mat.ambient)

        h = sm.normalize(l + v)
        n_dot_h = sm.clamp(sm.dot(n, h), 0.0, 1.0)
        horizon = sm.clamp(n_dot_l * 4.0, 0.0, 1.0)
        spec_pow = sm.pow(n_dot_h, 2.0 / (mat.roughness * mat.roughness + 0.01)) * light.intensity * horizon
        spec_tint = mat.albedo.xyz * mat.metallic + vec3(1.0, 1.0, 1.0) * (1.0 - mat.metallic)
        specular = spec_tint * spec_pow * light.color

        n_dot_v = sm.clamp(sm.dot(n, v), 0.0, 1.0)
        fresnel = sm.pow(1.0 - n_dot_v, 4.0) * (0.04 + 0.25 * mat.metallic)
        rim = mat.albedo.xyz * fresnel * (light.ambient * 2.0 + 0.08)

        hdr = ambient + diffuse + specular + rim
        ldr = sm.clamp((hdr * (hdr * 2.51 + 0.03)) / (hdr * (hdr * 2.43 + 0.59) + 0.14), 0.0, 1.0)
        srgb = sm.pow(ldr, 1.0 / 2.2)
        return vec4(srgb, mat.albedo.w)

    @fragment
    def _normal_fs(normal: Value, world_pos: Value, uv: Value) -> Value:
        n = sm.normalize(normal)
        rgb = sm.clamp(n * 0.5 + 0.5, 0.0, 1.0)
        return vec4(sm.pow(rgb, 1.0 / 2.2), 1.0)

    def _overlay_vs(vertex_index: u32, verts: Buffer[Type.VEC4, "read_only"]) -> Vertex:
        pos = verts[vertex_index * u32(2)]
        col = verts[vertex_index * u32(2) + u32(1)]
        return Vertex(pos, varyings={"color": col})

    @fragment
    def _overlay_fs(color: Value) -> Value:
        return color

    cam_blk = CameraParams.as_parameter_block("cam")
    obj_blk = ObjectParams.as_parameter_block("obj")
    mat_blk = MaterialParams.as_parameter_block("mat")
    light_blk = LightParams.as_parameter_block("light")

    # Build Stage objects
    vs_in_iface = (ShaderIO("vertex_index", Type.U32, builtin="vertex_index"),)
    vs_in_nodes = (Node("input", Type.U32, literal="vertex_index"),)
    mesh_low = ASTLowering(_mesh_vs, vs_in_nodes, vs_in_iface, {"cam": cam_blk, "obj": obj_blk})
    mesh_res = mesh_low.result
    mesh_outputs = (_node(mesh_res.position, Type.VEC4), *(_node(val) for val in mesh_res.varyings.values()))
    mesh_out_iface = (
        ShaderIO("position", Type.VEC4, builtin="position"),
        *(ShaderIO(k, node.type, location=i) for i, (k, node) in enumerate(zip(mesh_res.varyings, mesh_outputs[1:]))),
    )
    mesh_vs_stage = Stage(
        "_mesh_vs",
        "vertex",
        vs_in_nodes,
        mesh_outputs,
        vs_in_iface,
        mesh_out_iface,
        __file__,
        1,
        mesh_low.module("_mesh_vs", "vertex", mesh_outputs, mesh_out_iface),
    )

    mesh_varyings = tuple(io for io in mesh_out_iface if io.builtin is None)
    pbr_fs_stage = _pbr_fs.trace(mesh_varyings)
    normal_fs_stage = _normal_fs.trace(mesh_varyings)

    ov_low = ASTLowering(_overlay_vs, vs_in_nodes, vs_in_iface, {})
    ov_res = ov_low.result
    ov_outputs = (_node(ov_res.position, Type.VEC4), _node(ov_res.varyings["color"]))
    ov_out_iface = (
        ShaderIO("position", Type.VEC4, builtin="position"),
        ShaderIO("color", Type.VEC4, location=0),
    )
    overlay_vs_stage = Stage(
        "_overlay_vs",
        "vertex",
        vs_in_nodes,
        ov_outputs,
        vs_in_iface,
        ov_out_iface,
        __file__,
        1,
        ov_low.module("_overlay_vs", "vertex", ov_outputs, ov_out_iface),
    )
    overlay_fs_stage = _overlay_fs.trace((ShaderIO("color", Type.VEC4, location=0),))

    full_params = [b.to_dict() for b in (cam_blk, obj_blk, mat_blk, light_blk)]
    tools = Toolchain().preflight(target)
    profile = TargetProfile.default_for_target(target)

    cache_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("mesh_vs", mesh_vs_stage, full_params),
        ("pbr_fs", pbr_fs_stage, full_params),
        ("normal_fs", normal_fs_stage, None),
        ("overlay_vs", overlay_vs_stage, None),
        ("overlay_fs", overlay_fs_stage, None),
    ]
    for label, stg, params in jobs:
        sub = cache_dir / f"_build_{label}"
        if sub.exists():
            shutil.rmtree(sub)
        sub.mkdir(parents=True)
        produced = _compile_stage(stg, target, sub, tools, target_profile=profile, pipeline_parameters=params)
        shutil.copyfile(sub / produced, expected[label])
        shutil.rmtree(sub, ignore_errors=True)

    return expected


def _build_metal_bridge_dylib() -> Path:
    """Compile _metal_bridge.mm into a cached shared library on macOS."""
    src_mm = Path(__file__).resolve().parent / "_metal_bridge.mm"
    src_hash = hashlib.sha256(src_mm.read_bytes()).hexdigest()[:12]
    out_dir = _REPO_ROOT / "build" / "minigfx_cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    dylib = out_dir / f"libminigfx_metal_{src_hash}.dylib"
    if dylib.is_file():
        return dylib

    inc_candidates: List[Path] = [_REPO_ROOT / ".sdk" / "include"]
    msc_inc = os.environ.get("METAL_SHADER_CONVERTER_INCLUDE")
    if msc_inc:
        inc_candidates.append(Path(msc_inc))
    inc_candidates.extend([Path("/usr/local/include"), Path("/opt/homebrew/include")])
    inc_flags = [f"-I{p}" for p in inc_candidates if p.is_dir()]

    tmp_dylib = out_dir / f".tmp_{os.getpid()}_{dylib.name}"
    cmd = [
        "clang++",
        "-ObjC++",
        "-fobjc-arc",
        "-std=c++17",
        "-O2",
        "-shared",
        *inc_flags,
        str(src_mm),
        "-framework",
        "Cocoa",
        "-framework",
        "Metal",
        "-framework",
        "QuartzCore",
        "-o",
        str(tmp_dylib),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to build MiniGfx Metal bridge:\n{res.stderr}")
    tmp_dylib.replace(dylib)
    return dylib


class _MetalWindowBackend:
    """Native Metal GPU presenter driving a GLFW Cocoa NSWindow."""

    OVERLAY_BUF_ID = 0

    def __init__(self, glfw_mod: Any, window: Any, fb_w: int, fb_h: int, shaders: Dict[str, Path]) -> None:
        dylib_path = _build_metal_bridge_dylib()
        lib = ctypes.CDLL(str(dylib_path))

        lib.mgfx_metal_create.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
        lib.mgfx_metal_create.restype = ctypes.c_void_p

        lib.mgfx_metal_last_error.argtypes = [ctypes.c_void_p]
        lib.mgfx_metal_last_error.restype = ctypes.c_char_p

        lib.mgfx_metal_device_name.argtypes = [ctypes.c_void_p]
        lib.mgfx_metal_device_name.restype = ctypes.c_char_p

        lib.mgfx_metal_create_pipeline.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.mgfx_metal_create_pipeline.restype = ctypes.c_int

        lib.mgfx_metal_upload_buffer.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        lib.mgfx_metal_upload_buffer.restype = ctypes.c_int

        lib.mgfx_metal_begin_frame.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.c_float,
        ]
        lib.mgfx_metal_begin_frame.restype = ctypes.c_int

        lib.mgfx_metal_draw.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint32,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
            ctypes.c_size_t,
        ]
        lib.mgfx_metal_draw.restype = ctypes.c_int

        lib.mgfx_metal_end_frame.argtypes = [ctypes.c_void_p]
        lib.mgfx_metal_end_frame.restype = ctypes.c_int

        lib.mgfx_metal_destroy.argtypes = [ctypes.c_void_p]
        lib.mgfx_metal_destroy.restype = None

        self._lib = lib
        cocoa_win = glfw_mod.get_cocoa_window(window)
        err_buf = ctypes.create_string_buffer(512)
        self._ctx = lib.mgfx_metal_create(ctypes.c_void_p(cocoa_win), fb_w, fb_h, err_buf, len(err_buf))
        if not self._ctx:
            raise RuntimeError(f"Metal initialization failed: {err_buf.value.decode('utf-8', errors='replace')}")

        dev_raw = lib.mgfx_metal_device_name(self._ctx)
        self.device_name = dev_raw.decode("utf-8", errors="replace") if dev_raw else "Metal"

        self.pipe_pbr = lib.mgfx_metal_create_pipeline(
            self._ctx,
            str(shaders["mesh_vs"]).encode("utf-8"),
            b"_mesh_vs",
            str(shaders["pbr_fs"]).encode("utf-8"),
            b"_pbr_fs",
            1,
            0,
        )
        self.pipe_normal = lib.mgfx_metal_create_pipeline(
            self._ctx,
            str(shaders["mesh_vs"]).encode("utf-8"),
            b"_mesh_vs",
            str(shaders["normal_fs"]).encode("utf-8"),
            b"_normal_fs",
            1,
            0,
        )
        self.pipe_overlay = lib.mgfx_metal_create_pipeline(
            self._ctx,
            str(shaders["overlay_vs"]).encode("utf-8"),
            b"_overlay_vs",
            str(shaders["overlay_fs"]).encode("utf-8"),
            b"_overlay_fs",
            0,
            1,
        )
        if min(self.pipe_pbr, self.pipe_normal, self.pipe_overlay) < 0:
            err = lib.mgfx_metal_last_error(self._ctx)
            msg = err.decode("utf-8", errors="replace") if err else "unknown pipeline error"
            self.destroy()
            raise RuntimeError(f"Metal pipeline creation failed: {msg}")

        self._mesh_cache: Dict[int, Tuple[int, int]] = {}
        self._next_buf_id = 1

    def _ensure_mesh(self, mesh: Mesh) -> Tuple[int, int]:
        key = id(mesh)
        cached = self._mesh_cache.get(key)
        if cached is not None:
            return cached
        raw, vcount = pack_mesh_triangles(mesh)
        buf_id = self._next_buf_id
        self._next_buf_id += 1
        if vcount > 0:
            rc = self._lib.mgfx_metal_upload_buffer(self._ctx, buf_id, raw, len(raw))
            if rc != 0:
                raise RuntimeError("Metal vertex buffer upload failed")
        self._mesh_cache[key] = (buf_id, vcount)
        return buf_id, vcount

    def render(
        self,
        fb_w: int,
        fb_h: int,
        cam_bytes: bytes,
        light_bytes: bytes,
        draws: Sequence[Tuple[Mesh, bytes, bytes, bool]],
        hud_bytes: bytes,
        hud_vert_count: int,
    ) -> None:
        if not self._ctx:
            return
        rc = self._lib.mgfx_metal_begin_frame(self._ctx, fb_w, fb_h, 0.035, 0.04, 0.065, 1.0)
        if rc == 1:
            return  # Window minimized
        if rc != 0:
            err = self._lib.mgfx_metal_last_error(self._ctx)
            raise RuntimeError(f"Metal begin_frame failed: {err.decode('utf-8', errors='replace') if err else rc}")

        for mesh, obj_bytes, mat_bytes, is_normal_debug in draws:
            buf_id, vcount = self._ensure_mesh(mesh)
            if vcount == 0:
                continue
            vs_consts = cam_bytes + obj_bytes
            if is_normal_debug:
                self._lib.mgfx_metal_draw(
                    self._ctx,
                    self.pipe_normal,
                    buf_id,
                    vcount,
                    vs_consts,
                    len(vs_consts),
                    None,
                    0,
                )
            else:
                fs_consts = cam_bytes + mat_bytes + light_bytes
                self._lib.mgfx_metal_draw(
                    self._ctx,
                    self.pipe_pbr,
                    buf_id,
                    vcount,
                    vs_consts,
                    len(vs_consts),
                    fs_consts,
                    len(fs_consts),
                )

        if hud_vert_count > 0 and hud_bytes:
            self._lib.mgfx_metal_upload_buffer(self._ctx, self.OVERLAY_BUF_ID, hud_bytes, len(hud_bytes))
            self._lib.mgfx_metal_draw(
                self._ctx,
                self.pipe_overlay,
                self.OVERLAY_BUF_ID,
                hud_vert_count,
                None,
                0,
                None,
                0,
            )

        rc = self._lib.mgfx_metal_end_frame(self._ctx)
        if rc != 0:
            err = self._lib.mgfx_metal_last_error(self._ctx)
            raise RuntimeError(f"Metal end_frame failed: {err.decode('utf-8', errors='replace') if err else rc}")

    def destroy(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.mgfx_metal_destroy(self._ctx)
            self._ctx = None


class _VulkanWindowBackend:
    """Pure-Python Vulkan GPU presenter driving a GLFW window."""

    def __init__(self, glfw_mod: Any, window: Any, fb_w: int, fb_h: int, shaders: Dict[str, Path]) -> None:
        import vulkan as vk

        self.vk = vk
        self.glfw = glfw_mod
        self.window = window
        self.shaders = {k: p.read_bytes() for k, p in shaders.items()}
        self._cleanup_callbacks: List[Any] = []
        self._swapchain_cleanup: List[Any] = []
        self._mesh_buffers: Dict[int, Tuple[Any, Any, Any, int]] = {}
        self._hud_buffer: Optional[Tuple[Any, Any, Any, int]] = None

        req_exts = list(glfw_mod.get_required_instance_extensions() or [])
        avail_exts = {e.extensionName for e in vk.vkEnumerateInstanceExtensionProperties(None)}
        flags = 0
        if "VK_KHR_portability_enumeration" in avail_exts:
            req_exts.append("VK_KHR_portability_enumeration")
            flags = vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        if "VK_KHR_get_physical_device_properties2" in avail_exts:
            req_exts.append("VK_KHR_get_physical_device_properties2")

        app_info = vk.VkApplicationInfo(
            pApplicationName="MiniGfx ModelViewer",
            apiVersion=vk.VK_API_VERSION_1_0,
        )
        self.instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(
                flags=flags,
                pApplicationInfo=app_info,
                ppEnabledExtensionNames=req_exts,
                ppEnabledLayerNames=[],
            ),
            None,
        )
        self._cleanup_callbacks.append(lambda: vk.vkDestroyInstance(self.instance, None))

        surf_ptr = vk.ffi.new("VkSurfaceKHR*")
        if glfw_mod.create_window_surface(self.instance, window, None, surf_ptr) != vk.VK_SUCCESS:
            raise RuntimeError("GLFW failed to create Vulkan window surface")
        self.surface = surf_ptr[0]
        destroy_surf = vk.vkGetInstanceProcAddr(self.instance, "vkDestroySurfaceKHR")
        self._cleanup_callbacks.append(lambda: destroy_surf(self.instance, self.surface, None))

        get_surf_support = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfaceSupportKHR")
        chosen = None
        for phys in vk.vkEnumeratePhysicalDevices(self.instance):
            props = vk.vkGetPhysicalDeviceProperties(phys)
            dev_exts = {e.extensionName for e in vk.vkEnumerateDeviceExtensionProperties(phys, None)}
            if "VK_KHR_swapchain" not in dev_exts:
                continue
            for q_idx, q_prop in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(phys)):
                if q_prop.queueCount > 0 and (q_prop.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT):
                    if get_surf_support(phys, q_idx, self.surface):
                        chosen = (phys, q_idx, props, dev_exts)
                        break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError("No Vulkan graphics+presentation device found")

        self.physical, self.queue_family, props, dev_exts = chosen
        raw_name = props.deviceName
        self.device_name = (
            raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
        ).rstrip("\x00")
        self.mem_props = vk.vkGetPhysicalDeviceMemoryProperties(self.physical)

        enabled_dev_exts = ["VK_KHR_swapchain"]
        if "VK_KHR_portability_subset" in dev_exts:
            enabled_dev_exts.append("VK_KHR_portability_subset")
        if "VK_KHR_storage_buffer_storage_class" in dev_exts:
            enabled_dev_exts.append("VK_KHR_storage_buffer_storage_class")

        q_info = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.queue_family, pQueuePriorities=[1.0])
        self.device = vk.vkCreateDevice(
            self.physical,
            vk.VkDeviceCreateInfo(pQueueCreateInfos=[q_info], ppEnabledExtensionNames=enabled_dev_exts),
            None,
        )
        self._cleanup_callbacks.append(lambda: vk.vkDestroyDevice(self.device, None))
        self.queue = vk.vkGetDeviceQueue(self.device, self.queue_family, 0)

        self.cmd_pool = vk.vkCreateCommandPool(
            self.device,
            vk.VkCommandPoolCreateInfo(
                queueFamilyIndex=self.queue_family,
                flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            ),
            None,
        )
        self._cleanup_callbacks.append(lambda: vk.vkDestroyCommandPool(self.device, self.cmd_pool, None))
        self.cmd_buf = vk.vkAllocateCommandBuffers(
            self.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=self.cmd_pool,
                level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                commandBufferCount=1,
            ),
        )[0]

        self.desc_set_layout = vk.vkCreateDescriptorSetLayout(
            self.device,
            vk.VkDescriptorSetLayoutCreateInfo(
                pBindings=[
                    vk.VkDescriptorSetLayoutBinding(
                        binding=0,
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT,
                    )
                ]
            ),
            None,
        )
        self._cleanup_callbacks.append(lambda: vk.vkDestroyDescriptorSetLayout(self.device, self.desc_set_layout, None))

        self.desc_pool = vk.vkCreateDescriptorPool(
            self.device,
            vk.VkDescriptorPoolCreateInfo(
                flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                maxSets=256,
                pPoolSizes=[vk.VkDescriptorPoolSize(type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, descriptorCount=256)],
            ),
            None,
        )
        self._cleanup_callbacks.append(lambda: vk.vkDestroyDescriptorPool(self.device, self.desc_pool, None))

        self.img_avail = vk.vkCreateSemaphore(self.device, vk.VkSemaphoreCreateInfo(), None)
        self.render_done = vk.vkCreateSemaphore(self.device, vk.VkSemaphoreCreateInfo(), None)
        self.fence = vk.vkCreateFence(
            self.device, vk.VkFenceCreateInfo(flags=vk.VK_FENCE_CREATE_SIGNALED_BIT), None
        )
        self._cleanup_callbacks.append(lambda: vk.vkDestroySemaphore(self.device, self.img_avail, None))
        self._cleanup_callbacks.append(lambda: vk.vkDestroySemaphore(self.device, self.render_done, None))
        self._cleanup_callbacks.append(lambda: vk.vkDestroyFence(self.device, self.fence, None))

        self._create_swapchain_and_pipelines(fb_w, fb_h)

    def _find_mem_type(self, type_bits: int, req_flags: int) -> int:
        for i in range(self.mem_props.memoryTypeCount):
            if (type_bits & (1 << i)) and ((self.mem_props.memoryTypes[i].propertyFlags & req_flags) == req_flags):
                return i
        raise RuntimeError("No compatible Vulkan memory type found")

    def _alloc_ssbo(self, raw: bytes) -> Tuple[Any, Any, Any, int]:
        vk = self.vk
        cap = max(len(raw), 64)
        buf = vk.vkCreateBuffer(
            self.device,
            vk.VkBufferCreateInfo(
                size=cap,
                usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
            ),
            None,
        )
        req = vk.vkGetBufferMemoryRequirements(self.device, buf)
        mem_idx = self._find_mem_type(
            req.memoryTypeBits,
            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        )
        mem = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(allocationSize=req.size, memoryTypeIndex=mem_idx),
            None,
        )
        vk.vkBindBufferMemory(self.device, buf, mem, 0)
        ptr = vk.vkMapMemory(self.device, mem, 0, len(raw), 0)
        vk.ffi.memmove(ptr, raw, len(raw))
        vk.vkUnmapMemory(self.device, mem)

        dset = vk.vkAllocateDescriptorSets(
            self.device,
            vk.VkDescriptorSetAllocateInfo(descriptorPool=self.desc_pool, pSetLayouts=[self.desc_set_layout]),
        )[0]
        buf_info = [vk.VkDescriptorBufferInfo(buffer=buf, offset=0, range=cap)]
        writes = [
            vk.VkWriteDescriptorSet(
                dstSet=dset,
                dstBinding=0,
                descriptorCount=1,
                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                pBufferInfo=buf_info,
            )
        ]
        vk.vkUpdateDescriptorSets(self.device, 1, writes, 0, None)
        return buf, mem, dset, cap

    def _destroy_swapchain(self) -> None:
        self.vk.vkDeviceWaitIdle(self.device)
        while self._swapchain_cleanup:
            fn = self._swapchain_cleanup.pop()
            try:
                fn()
            except Exception:
                pass

    def _create_swapchain_and_pipelines(self, req_w: int, req_h: int) -> None:
        vk = self.vk
        caps = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR")(
            self.physical, self.surface
        )
        fmts = vk.vkGetInstanceProcAddr(self.instance, "vkGetPhysicalDeviceSurfaceFormatsKHR")(
            self.physical, self.surface
        )
        color_fmt, color_space = fmts[0].format, fmts[0].colorSpace
        for f in fmts:
            if f.format in (vk.VK_FORMAT_B8G8R8A8_UNORM, vk.VK_FORMAT_R8G8B8A8_UNORM):
                color_fmt, color_space = f.format, f.colorSpace
                break

        if caps.currentExtent.width != 0xFFFFFFFF:
            w, h = caps.currentExtent.width, caps.currentExtent.height
        else:
            w = max(caps.minImageExtent.width, min(req_w, caps.maxImageExtent.width))
            h = max(caps.minImageExtent.height, min(req_h, caps.maxImageExtent.height))
        w, h = max(1, w), max(1, h)
        self.sc_w, self.sc_h = w, h

        img_count = caps.minImageCount + 1
        if caps.maxImageCount > 0:
            img_count = min(img_count, caps.maxImageCount)

        create_sc = vk.vkGetDeviceProcAddr(self.device, "vkCreateSwapchainKHR")
        destroy_sc = vk.vkGetDeviceProcAddr(self.device, "vkDestroySwapchainKHR")
        get_sc_imgs = vk.vkGetDeviceProcAddr(self.device, "vkGetSwapchainImagesKHR")

        self.swapchain = create_sc(
            self.device,
            vk.VkSwapchainCreateInfoKHR(
                surface=self.surface,
                minImageCount=img_count,
                imageFormat=color_fmt,
                imageColorSpace=color_space,
                imageExtent=vk.VkExtent2D(width=w, height=h),
                imageArrayLayers=1,
                imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                preTransform=caps.currentTransform,
                compositeAlpha=vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
                presentMode=vk.VK_PRESENT_MODE_FIFO_KHR,
                clipped=vk.VK_TRUE,
            ),
            None,
        )
        self._swapchain_cleanup.append(lambda: destroy_sc(self.device, self.swapchain, None))
        sc_images = get_sc_imgs(self.device, self.swapchain)

        # Depth attachment image (D32_SFLOAT)
        depth_fmt = vk.VK_FORMAT_D32_SFLOAT
        depth_img = vk.vkCreateImage(
            self.device,
            vk.VkImageCreateInfo(
                imageType=vk.VK_IMAGE_TYPE_2D,
                format=depth_fmt,
                extent=vk.VkExtent3D(width=w, height=h, depth=1),
                mipLevels=1,
                arrayLayers=1,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                usage=vk.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
            ),
            None,
        )
        self._swapchain_cleanup.append(lambda: vk.vkDestroyImage(self.device, depth_img, None))
        d_req = vk.vkGetImageMemoryRequirements(self.device, depth_img)
        d_mem = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(
                allocationSize=d_req.size,
                memoryTypeIndex=self._find_mem_type(d_req.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
            ),
            None,
        )
        self._swapchain_cleanup.append(lambda: vk.vkFreeMemory(self.device, d_mem, None))
        vk.vkBindImageMemory(self.device, depth_img, d_mem, 0)
        depth_view = vk.vkCreateImageView(
            self.device,
            vk.VkImageViewCreateInfo(
                image=depth_img,
                viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                format=depth_fmt,
                subresourceRange=vk.VkImageSubresourceRange(
                    aspectMask=vk.VK_IMAGE_ASPECT_DEPTH_BIT, levelCount=1, layerCount=1
                ),
            ),
            None,
        )
        self._swapchain_cleanup.append(lambda: vk.vkDestroyImageView(self.device, depth_view, None))

        rp_attachments = [
            vk.VkAttachmentDescription(
                format=color_fmt,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
            ),
            vk.VkAttachmentDescription(
                format=depth_fmt,
                samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR,
                storeOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                finalLayout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ),
        ]
        color_refs = [vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)]
        depth_ref = vk.VkAttachmentReference(
            attachment=1, layout=vk.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        )
        rp_subpasses = [
            vk.VkSubpassDescription(
                pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                pColorAttachments=color_refs,
                pDepthStencilAttachment=depth_ref,
            )
        ]
        rp_deps = [
            vk.VkSubpassDependency(
                srcSubpass=vk.VK_SUBPASS_EXTERNAL,
                dstSubpass=0,
                srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                | vk.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT,
                dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                | vk.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
            )
        ]
        self.render_pass = vk.vkCreateRenderPass(
            self.device,
            vk.VkRenderPassCreateInfo(
                pAttachments=rp_attachments,
                pSubpasses=rp_subpasses,
                pDependencies=rp_deps,
            ),
            None,
        )
        self._swapchain_cleanup.append(lambda: vk.vkDestroyRenderPass(self.device, self.render_pass, None))

        self.framebuffers = []
        for img in sc_images:
            iv = vk.vkCreateImageView(
                self.device,
                vk.VkImageViewCreateInfo(
                    image=img,
                    viewType=vk.VK_IMAGE_VIEW_TYPE_2D,
                    format=color_fmt,
                    subresourceRange=vk.VkImageSubresourceRange(
                        aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1
                    ),
                ),
                None,
            )
            self._swapchain_cleanup.append(lambda v=iv: vk.vkDestroyImageView(self.device, v, None))
            fb = vk.vkCreateFramebuffer(
                self.device,
                vk.VkFramebufferCreateInfo(
                    renderPass=self.render_pass,
                    pAttachments=[iv, depth_view],
                    width=w,
                    height=h,
                    layers=1,
                ),
                None,
            )
            self._swapchain_cleanup.append(lambda f=fb: vk.vkDestroyFramebuffer(self.device, f, None))
            self.framebuffers.append(fb)

        self.layout_mesh = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(
                pSetLayouts=[self.desc_set_layout],
                pPushConstantRanges=[
                    vk.VkPushConstantRange(
                        stageFlags=vk.VK_SHADER_STAGE_VERTEX_BIT | vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                        offset=0,
                        size=224,
                    )
                ],
            ),
            None,
        )
        self._swapchain_cleanup.append(lambda: vk.vkDestroyPipelineLayout(self.device, self.layout_mesh, None))

        self.layout_overlay = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(pSetLayouts=[self.desc_set_layout], pPushConstantRanges=[]),
            None,
        )
        self._swapchain_cleanup.append(lambda: vk.vkDestroyPipelineLayout(self.device, self.layout_overlay, None))

        def _make_pipe(vs_key: str, vs_ep: str, fs_key: str, fs_ep: str, layout: Any, depth_on: bool, blend_on: bool):
            vs_mod = vk.vkCreateShaderModule(
                self.device,
                vk.VkShaderModuleCreateInfo(codeSize=len(self.shaders[vs_key]), pCode=self.shaders[vs_key]),
                None,
            )
            self._swapchain_cleanup.append(lambda m=vs_mod: vk.vkDestroyShaderModule(self.device, m, None))
            fs_mod = vk.vkCreateShaderModule(
                self.device,
                vk.VkShaderModuleCreateInfo(codeSize=len(self.shaders[fs_key]), pCode=self.shaders[fs_key]),
                None,
            )
            self._swapchain_cleanup.append(lambda m=fs_mod: vk.vkDestroyShaderModule(self.device, m, None))

            stages = [
                vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_VERTEX_BIT, module=vs_mod, pName=vs_ep
                ),
                vk.VkPipelineShaderStageCreateInfo(
                    stage=vk.VK_SHADER_STAGE_FRAGMENT_BIT, module=fs_mod, pName=fs_ep
                ),
            ]
            vert_input = vk.VkPipelineVertexInputStateCreateInfo()
            input_asm = vk.VkPipelineInputAssemblyStateCreateInfo(
                topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST
            )
            viewports = [vk.VkViewport(x=0, y=0, width=float(w), height=float(h), minDepth=0.0, maxDepth=1.0)]
            scissors = [vk.VkRect2D(offset=vk.VkOffset2D(0, 0), extent=vk.VkExtent2D(w, h))]
            vp_state = vk.VkPipelineViewportStateCreateInfo(pViewports=viewports, pScissors=scissors)
            raster_state = vk.VkPipelineRasterizationStateCreateInfo(
                polygonMode=vk.VK_POLYGON_MODE_FILL,
                cullMode=vk.VK_CULL_MODE_NONE,
                frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE,
                lineWidth=1.0,
            )
            ms_state = vk.VkPipelineMultisampleStateCreateInfo(
                rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT
            )
            ds_state = vk.VkPipelineDepthStencilStateCreateInfo(
                depthTestEnable=vk.VK_TRUE if depth_on else vk.VK_FALSE,
                depthWriteEnable=vk.VK_TRUE if depth_on else vk.VK_FALSE,
                depthCompareOp=vk.VK_COMPARE_OP_LESS_OR_EQUAL,
            )
            blend_atts = [
                vk.VkPipelineColorBlendAttachmentState(
                    blendEnable=vk.VK_TRUE if blend_on else vk.VK_FALSE,
                    srcColorBlendFactor=vk.VK_BLEND_FACTOR_SRC_ALPHA,
                    dstColorBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    colorBlendOp=vk.VK_BLEND_OP_ADD,
                    srcAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE,
                    dstAlphaBlendFactor=vk.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
                    alphaBlendOp=vk.VK_BLEND_OP_ADD,
                    colorWriteMask=0xF,
                )
            ]
            blend_state = vk.VkPipelineColorBlendStateCreateInfo(pAttachments=blend_atts)
            create_infos = [
                vk.VkGraphicsPipelineCreateInfo(
                    pStages=stages,
                    pVertexInputState=vert_input,
                    pInputAssemblyState=input_asm,
                    pViewportState=vp_state,
                    pRasterizationState=raster_state,
                    pMultisampleState=ms_state,
                    pDepthStencilState=ds_state,
                    pColorBlendState=blend_state,
                    layout=layout,
                    renderPass=self.render_pass,
                    subpass=0,
                )
            ]
            pipe = vk.vkCreateGraphicsPipelines(
                self.device,
                vk.VK_NULL_HANDLE,
                1,
                create_infos,
                None,
            )[0]
            self._swapchain_cleanup.append(lambda p=pipe: vk.vkDestroyPipeline(self.device, p, None))
            return pipe

        self.pipe_pbr = _make_pipe("mesh_vs", "_mesh_vs", "pbr_fs", "_pbr_fs", self.layout_mesh, True, False)
        self.pipe_normal = _make_pipe("mesh_vs", "_mesh_vs", "normal_fs", "_normal_fs", self.layout_mesh, True, False)
        self.pipe_overlay = _make_pipe("overlay_vs", "_overlay_vs", "overlay_fs", "_overlay_fs", self.layout_overlay, False, True)

    def _ensure_mesh(self, mesh: Mesh) -> Tuple[Any, int]:
        key = id(mesh)
        cached = self._mesh_buffers.get(key)
        if cached is not None:
            return cached[2], cached[3]
        raw, vcount = pack_mesh_triangles(mesh)
        if vcount == 0:
            return None, 0
        buf, mem, dset, _ = self._alloc_ssbo(raw)
        self._mesh_buffers[key] = (buf, mem, dset, vcount)
        return dset, vcount

    def _update_hud(self, raw: bytes) -> Optional[Any]:
        vk = self.vk
        if not raw:
            return None
        if self._hud_buffer is None or len(raw) > self._hud_buffer[3]:
            if self._hud_buffer is not None:
                old_buf, old_mem, old_dset, _ = self._hud_buffer
                vk.vkFreeDescriptorSets(self.device, self.desc_pool, 1, [old_dset])
                vk.vkDestroyBuffer(self.device, old_buf, None)
                vk.vkFreeMemory(self.device, old_mem, None)
            buf, mem, dset, cap = self._alloc_ssbo(bytes(max(len(raw) * 2, 4096)))
            self._hud_buffer = (buf, mem, dset, cap)

        _, mem, dset, _ = self._hud_buffer
        ptr = vk.vkMapMemory(self.device, mem, 0, len(raw), 0)
        vk.ffi.memmove(ptr, raw, len(raw))
        vk.vkUnmapMemory(self.device, mem)
        return dset

    def render(
        self,
        fb_w: int,
        fb_h: int,
        cam_bytes: bytes,
        light_bytes: bytes,
        draws: Sequence[Tuple[Mesh, bytes, bytes, bool]],
        hud_bytes: bytes,
        hud_vert_count: int,
    ) -> None:
        vk = self.vk
        if fb_w <= 0 or fb_h <= 0:
            return
        if fb_w != self.sc_w or fb_h != self.sc_h:
            self._destroy_swapchain()
            self._create_swapchain_and_pipelines(fb_w, fb_h)

        vk.vkWaitForFences(self.device, 1, [self.fence], vk.VK_TRUE, 0xFFFFFFFFFFFFFFFF)
        acquire = vk.vkGetDeviceProcAddr(self.device, "vkAcquireNextImageKHR")
        present = vk.vkGetDeviceProcAddr(self.device, "vkQueuePresentKHR")

        try:
            img_idx = acquire(self.device, self.swapchain, 0xFFFFFFFFFFFFFFFF, self.img_avail, vk.VK_NULL_HANDLE)
        except vk.VkErrorOutOfDateKhr:
            self._destroy_swapchain()
            self._create_swapchain_and_pipelines(fb_w, fb_h)
            return

        vk.vkResetFences(self.device, 1, [self.fence])
        vk.vkResetCommandBuffer(self.cmd_buf, 0)
        vk.vkBeginCommandBuffer(self.cmd_buf, vk.VkCommandBufferBeginInfo())

        clear_color = vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.035, 0.04, 0.065, 1.0]))
        clear_depth = vk.VkClearValue(depthStencil=vk.VkClearDepthStencilValue(depth=1.0, stencil=0))
        vk.vkCmdBeginRenderPass(
            self.cmd_buf,
            vk.VkRenderPassBeginInfo(
                renderPass=self.render_pass,
                framebuffer=self.framebuffers[img_idx],
                renderArea=vk.VkRect2D(offset=vk.VkOffset2D(0, 0), extent=vk.VkExtent2D(self.sc_w, self.sc_h)),
                pClearValues=[clear_color, clear_depth],
            ),
            vk.VK_SUBPASS_CONTENTS_INLINE,
        )

        for mesh, obj_bytes, mat_bytes, is_normal_debug in draws:
            dset, vcount = self._ensure_mesh(mesh)
            if vcount == 0 or dset is None:
                continue
            vk.vkCmdBindPipeline(
                self.cmd_buf,
                vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
                self.pipe_normal if is_normal_debug else self.pipe_pbr,
            )
            vk.vkCmdBindDescriptorSets(
                self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.layout_mesh, 0, 1, [dset], 0, None
            )
            pc = cam_bytes + obj_bytes + mat_bytes + light_bytes
            vk.vkCmdPushConstants(
                self.cmd_buf,
                self.layout_mesh,
                vk.VK_SHADER_STAGE_VERTEX_BIT | vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                0,
                len(pc),
                vk.ffi.from_buffer(pc),
            )
            vk.vkCmdDraw(self.cmd_buf, vcount, 1, 0, 0)

        if hud_vert_count > 0 and hud_bytes:
            hud_dset = self._update_hud(hud_bytes)
            if hud_dset is not None:
                vk.vkCmdBindPipeline(self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipe_overlay)
                vk.vkCmdBindDescriptorSets(
                    self.cmd_buf, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, self.layout_overlay, 0, 1, [hud_dset], 0, None
                )
                vk.vkCmdDraw(self.cmd_buf, hud_vert_count, 1, 0, 0)

        vk.vkCmdEndRenderPass(self.cmd_buf)
        vk.vkEndCommandBuffer(self.cmd_buf)

        submits = [
            vk.VkSubmitInfo(
                pWaitSemaphores=[self.img_avail],
                pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                pCommandBuffers=[self.cmd_buf],
                pSignalSemaphores=[self.render_done],
            )
        ]
        vk.vkQueueSubmit(self.queue, 1, submits, self.fence)
        try:
            present_info = vk.VkPresentInfoKHR(
                pWaitSemaphores=[self.render_done],
                pSwapchains=[self.swapchain],
                pImageIndices=[img_idx],
            )
            present(self.queue, present_info)
        except vk.VkErrorOutOfDateKhr:
            self._destroy_swapchain()
            self._create_swapchain_and_pipelines(fb_w, fb_h)

    def destroy(self) -> None:
        vk = getattr(self, "vk", None)
        if vk is None:
            return
        if hasattr(self, "device"):
            vk.vkDeviceWaitIdle(self.device)
        self._destroy_swapchain()
        if self._hud_buffer is not None:
            buf, mem, _, _ = self._hud_buffer
            vk.vkDestroyBuffer(self.device, buf, None)
            vk.vkFreeMemory(self.device, mem, None)
            self._hud_buffer = None
        for buf, mem, _, _ in self._mesh_buffers.values():
            vk.vkDestroyBuffer(self.device, buf, None)
            vk.vkFreeMemory(self.device, mem, None)
        self._mesh_buffers.clear()
        while self._cleanup_callbacks:
            fn = self._cleanup_callbacks.pop()
            try:
                fn()
            except Exception:
                pass
        self.vk = None


class InteractiveWindow:
    """GLFW window + GPU presenter wiring interactive input & rendering to ModelViewerApp."""

    def __init__(self, target: str, width: int, height: int, title: str = "MiniGfx ModelViewer") -> None:
        import glfw

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW (an interactive desktop session is required).")

        self.glfw = glfw
        self.target = target
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        self._app_ref: Optional[Any] = None
        self._dragging_button: Optional[int] = None
        self._last_cursor: Optional[Tuple[float, float]] = None
        self._shift_down: bool = False

        glfw.set_key_callback(self.window, self._on_key)
        glfw.set_mouse_button_callback(self.window, self._on_mouse_button)
        glfw.set_cursor_pos_callback(self.window, self._on_cursor_pos)
        glfw.set_scroll_callback(self.window, self._on_scroll)

        fb_w, fb_h = glfw.get_framebuffer_size(self.window)
        shaders = compile_minigfx_shaders("metal" if target == "metal" else "vulkan")

        try:
            if target == "metal":
                if sys.platform != "darwin":
                    raise RuntimeError("Metal window backend is only supported on macOS.")
                self.backend: Any = _MetalWindowBackend(glfw, self.window, fb_w, fb_h, shaders)
            elif target == "vulkan":
                self.backend = _VulkanWindowBackend(glfw, self.window, fb_w, fb_h, shaders)
            else:
                raise RuntimeError(f"Interactive window on {sys.platform} supports 'metal' or 'vulkan' (got {target!r}).")
        except Exception:
            glfw.destroy_window(self.window)
            self.window = None
            glfw.terminate()
            raise

    def bind_app(self, app: Any) -> None:
        self._app_ref = app

    def _on_key(self, window: Any, key: int, scancode: int, action: int, mods: int) -> None:
        glfw = self.glfw
        self._shift_down = bool(mods & glfw.MOD_SHIFT)
        if action != glfw.PRESS or self._app_ref is None:
            return
        key_map = {
            glfw.KEY_1: "1",
            glfw.KEY_2: "2",
            glfw.KEY_3: "3",
            glfw.KEY_SPACE: " ",
            glfw.KEY_Q: "q",
            glfw.KEY_ESCAPE: "escape",
        }
        mapped = key_map.get(key)
        if mapped is not None:
            self._app_ref.handle_key_press(mapped)

    def _on_mouse_button(self, window: Any, button: int, action: int, mods: int) -> None:
        glfw = self.glfw
        self._shift_down = bool(mods & glfw.MOD_SHIFT)
        if action == glfw.PRESS:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self._dragging_button = 1
                self._last_cursor = glfw.get_cursor_pos(window)
            elif button in (glfw.MOUSE_BUTTON_MIDDLE, glfw.MOUSE_BUTTON_RIGHT):
                self._dragging_button = 2
                self._last_cursor = glfw.get_cursor_pos(window)
        elif action == glfw.RELEASE:
            self._dragging_button = None
            self._last_cursor = None

    def _on_cursor_pos(self, window: Any, xpos: float, ypos: float) -> None:
        if self._dragging_button is None or self._app_ref is None:
            return
        if self._last_cursor is not None:
            dx = xpos - self._last_cursor[0]
            dy = ypos - self._last_cursor[1]
            self._app_ref.handle_mouse_drag(dx, dy, button=self._dragging_button, shift=self._shift_down)
        self._last_cursor = (xpos, ypos)

    def _on_scroll(self, window: Any, xoffset: float, yoffset: float) -> None:
        if self._app_ref is not None and abs(yoffset) > 1e-6:
            self._app_ref.handle_mouse_scroll(yoffset)

    def poll_and_sync(self, app: Any) -> bool:
        """Poll OS events and update camera aspect ratio. Returns False if window should close."""
        if self.window is None:
            return False
        self.glfw.poll_events()
        if self.glfw.window_should_close(self.window):
            app.stop()
            return False
        fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        if fb_w > 0 and fb_h > 0:
            app.width, app.height = fb_w, fb_h
            app.camera.aspect = float(fb_w) / float(fb_h)
        return True

    def present_frame(self, app: Any, commands: Sequence[Any]) -> None:
        """Submit recorded scene DrawCommands and HUD overlay to the GPU swapchain."""
        if self.window is None:
            return
        fb_w, fb_h = self.glfw.get_framebuffer_size(self.window)
        win_w, _ = self.glfw.get_window_size(self.window)
        if fb_w <= 0 or fb_h <= 0:
            return

        cam_bytes = app.camera.get_params(time=app.elapsed_time).to_bytes()

        # Point GPU light vector toward the sun light source so diffuse/specular track the sun sphere
        lp = app.light_params
        if hasattr(app, "light_node") and app.light_node is not None:
            sun_vec = app.light_node.translation
            gpu_dir = sun_vec.normalize() if sun_vec.length_sq() > 1e-8 else Vec3(0.5, 1.0, 0.5).normalize()
        else:
            d = lp.direction
            gpu_dir = Vec3(-d.x, -d.y, -d.z) if d.y < 0.0 else d
        gpu_light_bytes = LightParams(
            direction=gpu_dir,
            intensity=lp.intensity,
            color=lp.color,
            ambient=lp.ambient,
        ).to_bytes()

        mat_mode = getattr(app, "current_material_name", "metallic")
        draws: List[Tuple[Mesh, bytes, bytes, bool]] = []
        for cmd in commands:
            mesh = getattr(cmd, "mesh", None)
            node = getattr(cmd, "node", None)
            if mesh is None:
                continue
            model_mat = getattr(node, "world_transform", Mat4x4.identity()) if node is not None else Mat4x4.identity()
            mat = getattr(node, "material", None)
            if mat is None:
                mat = MaterialParams()
            obj_bytes = ObjectParams(model=model_mat, color=mat.albedo).to_bytes()
            is_normal_debug = (mat_mode == "normal_debug") and (node is getattr(app, "model_node", None))
            draws.append((mesh, obj_bytes, mat.to_bytes(), is_normal_debug))

        dpi_scale = max(1, round(fb_w / max(win_w, 1)))
        hud_text_1 = (
            app.debug_overlay.format_metrics() + f" | Mat: {mat_mode}"
            if getattr(app, "debug_overlay", None) is not None
            else f"FPS: {app.fps:.1f} | Backend: {self.target} | Mat: {mat_mode}"
        )
        sun_state = "ON" if getattr(app, "animate_light", True) else "PAUSED"
        hud_text_2 = f"[1] METAL  [2] DIELECTRIC  [3] NORMALS  [SPACE] SUN:{sun_state}  [DRAG/SCROLL] CAM  [ESC] QUIT"
        hud_bytes, hud_vcount = build_hud_overlay_vertices(
            [
                (hud_text_1, (0.92, 0.96, 1.0, 1.0)),
                (hud_text_2, (0.95, 0.80, 0.40, 0.95)),
            ],
            fb_w,
            fb_h,
            pixel_scale=2 * dpi_scale,
        )

        self.backend.render(fb_w, fb_h, cam_bytes, gpu_light_bytes, draws, hud_bytes, hud_vcount)

        if app.frame_count % 15 == 0:
            self.glfw.set_window_title(
                self.window,
                f"MiniGfx ModelViewer — {self.backend.device_name} ({self.target.upper()}) | {app.fps:.0f} FPS",
            )

    def close(self) -> None:
        if getattr(self, "backend", None) is not None:
            try:
                self.backend.destroy()
            finally:
                self.backend = None
        if getattr(self, "window", None) is not None:
            self.glfw.destroy_window(self.window)
            self.window = None
            self.glfw.terminate()
