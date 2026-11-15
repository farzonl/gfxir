"""GfxIR @compute kernels for 1D elementwise, bounded reduction, and 2D tiled matmul."""

from __future__ import annotations

from mlir_gpu import (
    Buffer,
    GlobalInvocationId,
    LocalInvocationId,
    ParameterBlock,
    Type,
    WorkgroupId,
    compute,
    global_invocation_id,
    i32,
    shared_array,
    vec4,
    workgroup_barrier,
)

WORKGROUP_SIZE_1D = (256, 1, 1)
TILE_DIM = 16
TILE_M = 64
TILE_N = 64
TILE_K = 16
MATMUL_WG_X = 16
MATMUL_WG_Y = 16
UNROLLED_TILE_STEPS = tuple(range(TILE_K))
UNROLLED_4 = (0, 1, 2, 3)

BinaryElementwiseParams = ParameterBlock(
    "gfxpy_binary_params",
    (("n", Type.I32),),
)

ScalarElementwiseParams = ParameterBlock(
    "gfxpy_scalar_params",
    (("n", Type.I32), ("scalar", Type.F32)),
)

MatMulParams = ParameterBlock(
    "gfxpy_matmul_params",
    (
        ("m", Type.I32),
        ("n", Type.I32),
        ("k", Type.I32),
        ("num_tiles", Type.I32),
    ),
)


@compute(workgroup_size=WORKGROUP_SIZE_1D)
def add_f32_kernel(
    lhs: Buffer[Type.F32, "read_only"],
    rhs: Buffer[Type.F32, "read_only"],
    out: Buffer[Type.F32, "write_only"],
    params: BinaryElementwiseParams,
):
    idx = i32(global_invocation_id().x)
    if idx < params.n:
        out[idx] = lhs[idx] + rhs[idx]


@compute(workgroup_size=WORKGROUP_SIZE_1D)
def mul_f32_kernel(
    lhs: Buffer[Type.F32, "read_only"],
    rhs: Buffer[Type.F32, "read_only"],
    out: Buffer[Type.F32, "write_only"],
    params: BinaryElementwiseParams,
):
    idx = i32(global_invocation_id().x)
    if idx < params.n:
        out[idx] = lhs[idx] * rhs[idx]


@compute(workgroup_size=WORKGROUP_SIZE_1D)
def scalar_add_f32_kernel(
    lhs: Buffer[Type.F32, "read_only"],
    out: Buffer[Type.F32, "write_only"],
    params: ScalarElementwiseParams,
):
    idx = i32(global_invocation_id().x)
    if idx < params.n:
        out[idx] = lhs[idx] + params.scalar


@compute(workgroup_size=WORKGROUP_SIZE_1D)
def scalar_mul_f32_kernel(
    lhs: Buffer[Type.F32, "read_only"],
    out: Buffer[Type.F32, "write_only"],
    params: ScalarElementwiseParams,
):
    idx = i32(global_invocation_id().x)
    if idx < params.n:
        out[idx] = lhs[idx] * params.scalar


@compute(workgroup_size=WORKGROUP_SIZE_1D)
def reduce_sum_f32_kernel(
    gid: GlobalInvocationId,
    lid: LocalInvocationId,
    wid: WorkgroupId,
    inp: Buffer[Type.F32, "read_only"],
    out: Buffer[Type.F32, "write_only"],
    params: BinaryElementwiseParams,
):
    shared_sum = shared_array(Type.F32, 256, name="reduce_sum")
    lx = i32(lid.x)
    gx = i32(gid.x)
    wx = i32(wid.x)

    val = 0.0
    if gx < params.n:
        val = inp[gx]
    shared_sum[lx] = val
    workgroup_barrier()

    stride = 128
    while stride > 0:
        if lx < stride:
            shared_sum[lx] = shared_sum[lx] + shared_sum[lx + stride]
        workgroup_barrier()
        stride = stride // 2

    if lx == 0:
        out[wx] = shared_sum[0]


@compute(workgroup_size=(MATMUL_WG_X, MATMUL_WG_Y, 1))
def tiled_matmul_f32_kernel(
    lid: LocalInvocationId,
    wid: WorkgroupId,
    mat_a: Buffer[Type.F32, "read_only"],
    mat_b: Buffer[Type.F32, "read_only"],
    mat_c: Buffer[Type.F32, "write_only"],
    params: MatMulParams,
):
    tile_a = shared_array(Type.F32, (TILE_M, TILE_K + 4), name="tile_a")
    tile_b = shared_array(Type.F32, (TILE_K, TILE_N + 4), name="tile_b")

    lx = i32(lid.x)
    ly = i32(lid.y)
    tid = ly * MATMUL_WG_X + lx

    a_tile_row = tid // 4
    a_tile_col_base = (tid % 4) * 4
    a_global_row = i32(wid.y) * TILE_M + a_tile_row

    b_tile_row = tid // 16
    b_tile_col_base = (tid % 16) * 4
    b_global_col_base = i32(wid.x) * TILE_N + b_tile_col_base

    row_base = i32(wid.y) * TILE_M + ly * 4
    col_base = i32(wid.x) * TILE_N + lx * 4

    acc0 = vec4(0.0, 0.0, 0.0, 0.0)
    acc1 = vec4(0.0, 0.0, 0.0, 0.0)
    acc2 = vec4(0.0, 0.0, 0.0, 0.0)
    acc3 = vec4(0.0, 0.0, 0.0, 0.0)

    for t in range(params.num_tiles):
        k_base = t * TILE_K
        for u in UNROLLED_4:
            ac = k_base + a_tile_col_base + u
            av = 0.0
            if (a_global_row < params.m) and (ac < params.k):
                av = mat_a[a_global_row * params.k + ac]
            tile_a[a_tile_row, a_tile_col_base + u] = av

            br = k_base + b_tile_row
            bc = b_global_col_base + u
            bv = 0.0
            if (br < params.k) and (bc < params.n):
                bv = mat_b[br * params.n + bc]
            tile_b[b_tile_row, b_tile_col_base + u] = bv

        workgroup_barrier()

        for p in UNROLLED_TILE_STEPS:
            a0 = tile_a[ly * 4 + 0, p]
            a1 = tile_a[ly * 4 + 1, p]
            a2 = tile_a[ly * 4 + 2, p]
            a3 = tile_a[ly * 4 + 3, p]
            b_vec = vec4(
                tile_b[p, lx * 4 + 0],
                tile_b[p, lx * 4 + 1],
                tile_b[p, lx * 4 + 2],
                tile_b[p, lx * 4 + 3],
            )
            acc0 = acc0 + vec4(a0, a0, a0, a0) * b_vec
            acc1 = acc1 + vec4(a1, a1, a1, a1) * b_vec
            acc2 = acc2 + vec4(a2, a2, a2, a2) * b_vec
            acc3 = acc3 + vec4(a3, a3, a3, a3) * b_vec

        workgroup_barrier()

    if ((row_base + 0) < params.m) and ((col_base + 0) < params.n):
        mat_c[(row_base + 0) * params.n + col_base + 0] = acc0.x
    if ((row_base + 0) < params.m) and ((col_base + 1) < params.n):
        mat_c[(row_base + 0) * params.n + col_base + 1] = acc0.y
    if ((row_base + 0) < params.m) and ((col_base + 2) < params.n):
        mat_c[(row_base + 0) * params.n + col_base + 2] = acc0.z
    if ((row_base + 0) < params.m) and ((col_base + 3) < params.n):
        mat_c[(row_base + 0) * params.n + col_base + 3] = acc0.w
    if ((row_base + 1) < params.m) and ((col_base + 0) < params.n):
        mat_c[(row_base + 1) * params.n + col_base + 0] = acc1.x
    if ((row_base + 1) < params.m) and ((col_base + 1) < params.n):
        mat_c[(row_base + 1) * params.n + col_base + 1] = acc1.y
    if ((row_base + 1) < params.m) and ((col_base + 2) < params.n):
        mat_c[(row_base + 1) * params.n + col_base + 2] = acc1.z
    if ((row_base + 1) < params.m) and ((col_base + 3) < params.n):
        mat_c[(row_base + 1) * params.n + col_base + 3] = acc1.w
    if ((row_base + 2) < params.m) and ((col_base + 0) < params.n):
        mat_c[(row_base + 2) * params.n + col_base + 0] = acc2.x
    if ((row_base + 2) < params.m) and ((col_base + 1) < params.n):
        mat_c[(row_base + 2) * params.n + col_base + 1] = acc2.y
    if ((row_base + 2) < params.m) and ((col_base + 2) < params.n):
        mat_c[(row_base + 2) * params.n + col_base + 2] = acc2.z
    if ((row_base + 2) < params.m) and ((col_base + 3) < params.n):
        mat_c[(row_base + 2) * params.n + col_base + 3] = acc2.w
    if ((row_base + 3) < params.m) and ((col_base + 0) < params.n):
        mat_c[(row_base + 3) * params.n + col_base + 0] = acc3.x
    if ((row_base + 3) < params.m) and ((col_base + 1) < params.n):
        mat_c[(row_base + 3) * params.n + col_base + 1] = acc3.y
    if ((row_base + 3) < params.m) and ((col_base + 2) < params.n):
        mat_c[(row_base + 3) * params.n + col_base + 2] = acc3.z
    if ((row_base + 3) < params.m) and ((col_base + 3) < params.n):
        mat_c[(row_base + 3) * params.n + col_base + 3] = acc3.w


def compute_grid_1d(length: int) -> tuple[int, int, int]:
    """Compute 1D workgroup dispatch grid for a non-empty length."""
    if length <= 0:
        raise ValueError(f"compute_grid_1d requires positive length, got {length}.")
    wg_x = WORKGROUP_SIZE_1D[0]
    return ((length + wg_x - 1) // wg_x, 1, 1)


def compute_matmul_grid(m: int, n: int) -> tuple[int, int, int]:
    """Compute 2D workgroup dispatch grid for tiled matrix multiplication."""
    if m <= 0 or n <= 0:
        raise ValueError(f"compute_matmul_grid requires positive dimensions, got ({m}, {n}).")
    return ((n + TILE_N - 1) // TILE_N, (m + TILE_M - 1) // TILE_M, 1)


BINARY_KERNELS = {
    "add": add_f32_kernel,
    "mul": mul_f32_kernel,
}

SCALAR_KERNELS = {
    "add": scalar_add_f32_kernel,
    "mul": scalar_mul_f32_kernel,
}
