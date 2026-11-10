"""Multi-buffer Particle Life simulation step compute shader authored in Python."""

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    GlobalInvocationId,
    ParameterBlock,
    Type,
    compute,
    i32,
    maximum,
    select,
    shader_math as sm,
    u32,
    vec2,
)

ParticleLifeParams = ParameterBlock(
    "particle_life_params",
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


@compute(workgroup_size=(64, 1, 1))
def particle_life_step(
    gid: GlobalInvocationId,
    in_pos: Buffer[Type.VEC2, "read_only"],
    in_vel: Buffer[Type.VEC2, "read_only"],
    in_species: Buffer[Type.I32, "read_only"],
    out_pos: Buffer[Type.VEC2, "write_only"],
    out_vel: Buffer[Type.VEC2, "write_only"],
    force_matrix: Buffer[Type.F32, "read_only"],
    params: ParticleLifeParams,
):
    i = gid.x
    if i < params.count:
        p_i = in_pos[i]
        v_i = in_vel[i]
        s_i = in_species[i]

        fx = 0.0
        fy = 0.0
        half_w = params.world_width * 0.5
        half_h = params.world_height * 0.5

        for j in range(params.count):
            if u32(j) != i:
                p_j = in_pos[j]
                s_j = in_species[j]

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


pipeline = ComputePipeline(particle_life_step, dispatch=Dispatch(grid=(16, 1, 1)))
