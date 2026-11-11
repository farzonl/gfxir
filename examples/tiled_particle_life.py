"""Shared-memory tiled Particle Life N-body simulation and RGBA8 visualization compute shaders."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import struct
import tempfile
import time

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    GlobalInvocationId,
    LocalInvocationId,
    ParameterBlock,
    RenderedImage,
    Target,
    Toolchain,
    Type,
    Value,
    compile_pipeline,
    compute,
    compute_session,
    i32,
    maximum,
    render_compute_rgba,
    select,
    shader_math as sm,
    shared_array,
    u32,
    vec2,
    write_png,
    workgroup_barrier,
)

TILE_SIZE = 64

TiledParticleLifeParams = ParameterBlock(
    "tiled_particle_life_params",
    (
        ("count", Type.U32),
        ("num_species", Type.U32),
        ("dt", Type.F32),
        ("friction", Type.F32),
        ("r_max", Type.F32),
        ("r_min", Type.F32),
        ("world_width", Type.F32),
        ("world_height", Type.F32),
        ("force_scale", Type.F32),
    ),
)

TiledParticleRenderParams = ParameterBlock(
    "tiled_particle_render_params",
    (
        ("width", Type.U32),
        ("height", Type.U32),
        ("count", Type.U32),
        ("world_width", Type.F32),
        ("world_height", Type.F32),
        ("particle_radius", Type.F32),
    ),
)


def _pack_rgba8(r: Value, g: Value, b: Value) -> Value:
    ru = u32(int(sm.clamp(r * 255.0 + 0.5, 0.0, 255.0)))
    gu = u32(int(sm.clamp(g * 255.0 + 0.5, 0.0, 255.0)))
    bu = u32(int(sm.clamp(b * 255.0 + 0.5, 0.0, 255.0)))
    return ru + gu * u32(256) + bu * u32(65536) + u32(255) * u32(16777216)


@compute(workgroup_size=(TILE_SIZE, 1, 1))
def tiled_particle_life_step(
    gid: GlobalInvocationId,
    lid: LocalInvocationId,
    in_pos: Buffer[Type.VEC2, "read_only"],
    in_vel: Buffer[Type.VEC2, "read_only"],
    in_species: Buffer[Type.I32, "read_only"],
    out_pos: Buffer[Type.VEC2, "write_only"],
    out_vel: Buffer[Type.VEC2, "write_only"],
    force_matrix: Buffer[Type.F32, "read_only"],
    params: TiledParticleLifeParams,
):
    tile_pos = shared_array(vec2, TILE_SIZE, name="tile_pos")
    tile_species = shared_array(Type.I32, TILE_SIZE, name="tile_species")

    i = i32(gid.x)
    lx = i32(lid.x)
    cnt = i32(params.count)
    active = i < cnt

    p_i = vec2(0.0, 0.0)
    v_i = vec2(0.0, 0.0)
    s_i = 0
    if active:
        p_i = in_pos[i]
        v_i = in_vel[i]
        s_i = in_species[i]

    fx = 0.0
    fy = 0.0
    half_w = params.world_width * 0.5
    half_h = params.world_height * 0.5
    num_tiles = (cnt + (TILE_SIZE - 1)) // TILE_SIZE

    for tile_idx in range(num_tiles):
        load_idx = tile_idx * TILE_SIZE + lx
        tp = vec2(0.0, 0.0)
        ts = 0
        if load_idx < cnt:
            tp = in_pos[load_idx]
            ts = in_species[load_idx]
        tile_pos[lx] = tp
        tile_species[lx] = ts
        workgroup_barrier()

        if active:
            for k in range(TILE_SIZE):
                j = tile_idx * TILE_SIZE + k
                if (j < cnt) and (j != i):
                    p_j = tile_pos[k]
                    s_j = tile_species[k]

                    raw_dx = p_j.x - p_i.x
                    raw_dy = p_j.y - p_i.y

                    dx = select(
                        raw_dx < -half_w,
                        raw_dx + params.world_width,
                        select(raw_dx > half_w, raw_dx - params.world_width, raw_dx),
                    )
                    dy = select(
                        raw_dy < -half_h,
                        raw_dy + params.world_height,
                        select(raw_dy > half_h, raw_dy - params.world_height, raw_dy),
                    )

                    r2 = dx * dx + dy * dy
                    r = sm.sqrt(r2)
                    if (r > 0.0001) and (r < params.r_max):
                        inv_r = sm.rsqrt(r2)
                        ux = dx * inv_r
                        uy = dy * inv_r

                        if r < params.r_min:
                            f_mag = r / params.r_min - 1.0
                        else:
                            xi_dist = (r - params.r_min) / (params.r_max - params.r_min)
                            tent = 1.0 - abs(2.0 * xi_dist - 1.0)
                            mat_idx = s_i * i32(params.num_species) + s_j
                            coeff = force_matrix[mat_idx]
                            f_mag = coeff * tent

                        f_scaled = f_mag * params.force_scale
                        fx = fx + ux * f_scaled
                        fy = fy + uy * f_scaled

        workgroup_barrier()

    if active:
        damp = maximum(1.0 - params.friction * params.dt, 0.0)
        vx_next = (v_i.x + fx * params.dt) * damp
        vy_next = (v_i.y + fy * params.dt) * damp
        v_next = vec2(vx_next, vy_next)

        x_raw = p_i.x + vx_next * params.dt
        y_raw = p_i.y + vy_next * params.dt

        x_rem = x_raw - sm.floor(x_raw / params.world_width) * params.world_width
        x_pos = maximum(x_rem, 0.0)
        x_wrapped = select(x_pos >= params.world_width, 0.0, x_pos)

        y_rem = y_raw - sm.floor(y_raw / params.world_height) * params.world_height
        y_pos = maximum(y_rem, 0.0)
        y_wrapped = select(y_pos >= params.world_height, 0.0, y_pos)

        out_pos[i] = vec2(x_wrapped, y_wrapped)
        out_vel[i] = v_next


@compute(workgroup_size=(16, 16, 1))
def tiled_particle_life_render_kernel(
    gid: GlobalInvocationId,
    lid: LocalInvocationId,
    positions: Buffer[Type.VEC2, "read_only"],
    species: Buffer[Type.I32, "read_only"],
    palette: Buffer[Type.VEC4, "read_only"],
    framebuffer: Buffer[Type.U32, "write_only"],
    params: TiledParticleRenderParams,
):
    tile_pos = shared_array(vec2, TILE_SIZE, name="render_pos")
    tile_sp = shared_array(Type.I32, TILE_SIZE, name="render_sp")

    px = gid.x
    py = gid.y
    tid = i32(lid.y) * 16 + i32(lid.x)
    cnt = i32(params.count)
    in_bounds = (px < params.width) and (py < params.height)

    wx = (float(px) + 0.5) / float(params.width) * params.world_width
    wy = (float(py) + 0.5) / float(params.height) * params.world_height
    half_w = params.world_width * 0.5
    half_h = params.world_height * 0.5
    bg_v = (float(py) + 0.5) / float(params.height)

    r_acc = 0.035 + 0.02 * bg_v
    g_acc = 0.045 + 0.025 * bg_v
    b_acc = 0.085 + 0.04 * bg_v
    rad2 = params.particle_radius * params.particle_radius
    num_tiles = (cnt + (TILE_SIZE - 1)) // TILE_SIZE

    for tile_idx in range(num_tiles):
        if tid < TILE_SIZE:
            load_idx = tile_idx * TILE_SIZE + tid
            tp = vec2(0.0, 0.0)
            ts = 0
            if load_idx < cnt:
                tp = positions[load_idx]
                ts = species[load_idx]
            tile_pos[tid] = tp
            tile_sp[tid] = ts
        workgroup_barrier()

        if in_bounds:
            for k in range(TILE_SIZE):
                idx = tile_idx * TILE_SIZE + k
                if idx < cnt:
                    pos = tile_pos[k]
                    sp = tile_sp[k]

                    raw_dx = pos.x - wx
                    raw_dy = pos.y - wy
                    dx = select(
                        raw_dx < -half_w,
                        raw_dx + params.world_width,
                        select(raw_dx > half_w, raw_dx - params.world_width, raw_dx),
                    )
                    dy = select(
                        raw_dy < -half_h,
                        raw_dy + params.world_height,
                        select(raw_dy > half_h, raw_dy - params.world_height, raw_dy),
                    )

                    r2 = dx * dx + dy * dy
                    if r2 < rad2:
                        dist = sm.sqrt(r2)
                        norm = 1.0 - dist / params.particle_radius
                        glow = norm * norm
                        core = select(dist < params.particle_radius * 0.35, 0.35, 0.0)
                        col = palette[sp]
                        r_acc = r_acc + col.x * glow + core
                        g_acc = g_acc + col.y * glow + core
                        b_acc = b_acc + col.z * glow + core

        workgroup_barrier()

    if in_bounds:
        framebuffer[py * params.width + px] = _pack_rgba8(r_acc, g_acc, b_acc)


pipeline = ComputePipeline(tiled_particle_life_step, dispatch=Dispatch(grid=(1, 1, 1)))
render_pipeline = ComputePipeline(tiled_particle_life_render_kernel, dispatch=Dispatch(grid=(8, 8, 1)))

DEFAULT_SIMULATION_PARAMS: dict[str, int | float] = {
    "count": 64,
    "num_species": 4,
    "dt": 0.04,
    "friction": 2.5,
    "r_max": 32.0,
    "r_min": 6.0,
    "world_width": 128.0,
    "world_height": 128.0,
    "force_scale": 18.0,
}

SPECIES_PALETTE: tuple[tuple[float, float, float, float], ...] = (
    (0.15, 0.85, 1.00, 1.0),
    (1.00, 0.32, 0.24, 1.0),
    (0.22, 0.95, 0.48, 1.0),
    (1.00, 0.82, 0.18, 1.0),
)

DEFAULT_FORCE_MATRIX: tuple[float, ...] = (
    0.85, -0.45, 0.30, -0.20,
    0.35, 0.75, -0.50, 0.25,
    -0.30, 0.40, 0.80, -0.35,
    0.25, -0.30, 0.45, 0.70,
)


def create_initial_particle_state(
    count: int = 64,
    num_species: int = 4,
    world_width: float = 128.0,
    world_height: float = 128.0,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], list[int]]:
    """Generate the deterministic multi-cluster initial particle distribution."""
    positions: list[tuple[float, float]] = []
    velocities: list[tuple[float, float]] = []
    species: list[int] = []
    centers = (
        (world_width * 0.28, world_height * 0.30),
        (world_width * 0.72, world_height * 0.30),
        (world_width * 0.30, world_height * 0.72),
        (world_width * 0.70, world_height * 0.70),
    )
    state = 0x9E3779B9
    for idx in range(count):
        sp = idx % num_species
        cx, cy = centers[sp % len(centers)]
        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        angle = ((state & 0xFFFF) / 65536.0) * (2.0 * math.pi)
        radius = 4.0 + (((state >> 16) & 0xFFFF) / 65536.0) * 18.0
        px = (cx + math.cos(angle) * radius) % world_width
        py = (cy + math.sin(angle) * radius) % world_height

        state = (1664525 * state + 1013904223) & 0xFFFFFFFF
        vx = (((state & 0xFFFF) / 65536.0) - 0.5) * 3.0
        vy = ((((state >> 16) & 0xFFFF) / 65536.0) - 0.5) * 3.0
        positions.append((px, py))
        velocities.append((vx, vy))
        species.append(sp)
    return positions, velocities, species


def pack_vec2_buffer(items: list[tuple[float, float]]) -> bytes:
    return b"".join(struct.pack("<2f", x, y) for x, y in items)


def unpack_vec2_buffer(raw: bytes) -> list[tuple[float, float]]:
    count = len(raw) // 8
    return [struct.unpack_from("<2f", raw, i * 8) for i in range(count)]


def pack_i32_buffer(items: list[int]) -> bytes:
    return struct.pack(f"<{len(items)}i", *items)


def pack_f32_buffer(items: tuple[float, ...] | list[float]) -> bytes:
    return struct.pack(f"<{len(items)}f", *items)


def pack_vec4_buffer(items: tuple[tuple[float, float, float, float], ...]) -> bytes:
    return b"".join(struct.pack("<4f", *item) for item in items)


@dataclass(frozen=True)
class TiledParticleLifeRunResult:
    positions: list[tuple[float, float]]
    velocities: list[tuple[float, float]]
    image: RenderedImage
    sim_elapsed_ms: float
    render_elapsed_ms: float


def simulate_and_render_tiled_particle_life(
    target: Target,
    *,
    steps: int = 8,
    width: int = 128,
    height: int = 128,
    particle_radius: float = 5.5,
    params: dict[str, int | float] | None = None,
    validation: bool = False,
    provider: str | None = None,
    toolchain: Toolchain | None = None,
) -> TiledParticleLifeRunResult:
    """Run multi-step shared-memory tiled Particle Life physics and render RGBA8 frame on the GPU."""
    sim_params = dict(DEFAULT_SIMULATION_PARAMS if params is None else params)
    count = int(sim_params["count"])
    pos_init, vel_init, sp_init = create_initial_particle_state(
        count=count,
        num_species=int(sim_params["num_species"]),
        world_width=float(sim_params["world_width"]),
        world_height=float(sim_params["world_height"]),
    )
    pos_bytes = pack_vec2_buffer(pos_init)
    vel_bytes = pack_vec2_buffer(vel_init)
    sp_bytes = pack_i32_buffer(sp_init)
    force_bytes = pack_f32_buffer(DEFAULT_FORCE_MATRIX)
    palette_bytes = pack_vec4_buffer(SPECIES_PALETTE)
    sim_grid = ((count + TILE_SIZE - 1) // TILE_SIZE, 1, 1)

    with tempfile.TemporaryDirectory() as td:
        step_manifest = compile_pipeline(pipeline, target, Path(td) / "tiled_step", toolchain)
        render_manifest = compile_pipeline(render_pipeline, target, Path(td) / "tiled_render", toolchain)

        with compute_session(
            step_manifest,
            validation=validation,
            provider=provider,
            toolchain=toolchain,
        ) as run_step:
            t_sim0 = time.perf_counter()
            for _ in range(steps):
                res = run_step(
                    buffers={
                        "in_pos": pos_bytes,
                        "in_vel": vel_bytes,
                        "in_species": sp_bytes,
                        "out_pos": len(pos_bytes),
                        "out_vel": len(vel_bytes),
                        "force_matrix": force_bytes,
                    },
                    parameters=sim_params,
                    grid=sim_grid,
                )
                pos_bytes = res.buffers["out_pos"]
                vel_bytes = res.buffers["out_vel"]
            sim_elapsed_ms = (time.perf_counter() - t_sim0) * 1000.0

        t_ren0 = time.perf_counter()
        image = render_compute_rgba(
            render_manifest,
            width,
            height,
            output_buffer="framebuffer",
            buffers={
                "positions": pos_bytes,
                "species": sp_bytes,
                "palette": palette_bytes,
            },
            parameters={
                "width": width,
                "height": height,
                "count": count,
                "world_width": float(sim_params["world_width"]),
                "world_height": float(sim_params["world_height"]),
                "particle_radius": particle_radius,
            },
            validation=validation,
            provider=provider,
            toolchain=toolchain,
        )
        render_elapsed_ms = (time.perf_counter() - t_ren0) * 1000.0

    return TiledParticleLifeRunResult(
        positions=unpack_vec2_buffer(pos_bytes),
        velocities=unpack_vec2_buffer(vel_bytes),
        image=image,
        sim_elapsed_ms=sim_elapsed_ms,
        render_elapsed_ms=render_elapsed_ms,
    )


def _default_cli_toolchain(llvm_bin: Path | None, tools_bin: Path | None) -> Toolchain:
    repo_root = Path(__file__).resolve().parents[1]
    for bin_dir in (
        repo_root / ".toolchains" / "llvm-build" / "bin",
        repo_root / ".tools" / "bin",
        repo_root / ".sdk" / "bin",
    ):
        if bin_dir.is_dir():
            entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
            if str(bin_dir) not in entries:
                os.environ["PATH"] = os.pathsep.join([*entries, str(bin_dir)])
    resolved_llvm = llvm_bin or (Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)
    resolved_tools = tools_bin or (Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None)
    return Toolchain(llvm_bin=resolved_llvm, tools_bin=resolved_tools)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simulate and render shared-memory tiled Particle Life compute shader output to PNG."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output PNG path")
    parser.add_argument("--target", choices=[t.value for t in Target], default="vulkan")
    parser.add_argument("--provider", default=None, help="Optional runtime provider (warp, lavapipe, swiftshader, hardware)")
    parser.add_argument("--count", type=int, default=64)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--llvm-bin", type=Path, default=None, help="Optional directory containing gfx-compile and llc")
    parser.add_argument("--tools-bin", type=Path, default=None, help="Optional directory containing spirv-val / dxv / viewers")
    args = parser.parse_args()

    toolchain = _default_cli_toolchain(args.llvm_bin, args.tools_bin)
    sim_params = dict(DEFAULT_SIMULATION_PARAMS)
    sim_params["count"] = args.count
    result = simulate_and_render_tiled_particle_life(
        Target(args.target),
        steps=args.steps,
        width=args.width,
        height=args.height,
        params=sim_params,
        provider=args.provider,
        toolchain=toolchain,
    )
    args.output.unlink(missing_ok=True)
    write_png(args.output, result.image)
    print(
        f"Rendered Tiled Particle Life ({args.count} particles, {args.steps} steps, "
        f"{result.image.width}x{result.image.height}) on {result.image.device}: "
        f"sim={result.sim_elapsed_ms:.2f} ms, render={result.render_elapsed_ms:.2f} ms -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
