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
    coop_matrix_load,
    coop_matrix_mul_add,
    coop_matrix_store,
    coop_matrix_zero,
    global_invocation_id,
    i32,
    shared_array,
    workgroup_barrier,
)

WORKGROUP_SIZE_1D = (64, 1, 1)
TILE_DIM = 16
UNROLLED_TILE_STEPS = tuple(range(TILE_DIM))

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
    shared_sum = shared_array(Type.F32, 64, name="reduce_sum")
    lx = i32(lid.x)
    gx = i32(gid.x)
    wx = i32(wid.x)

    val = 0.0
    if gx < params.n:
        val = inp[gx]
    shared_sum[lx] = val
    workgroup_barrier()

    stride = 32
    while stride > 0:
        if lx < stride:
            shared_sum[lx] = shared_sum[lx] + shared_sum[lx + stride]
        workgroup_barrier()
        stride = stride // 2

    if lx == 0:
        out[wx] = shared_sum[0]


@compute(workgroup_size=(TILE_DIM, TILE_DIM, 1))
def tiled_matmul_f32_kernel(
    lid: LocalInvocationId,
    wid: WorkgroupId,
    mat_a: Buffer[Type.F32, "read_only"],
    mat_b: Buffer[Type.F32, "read_only"],
    mat_c: Buffer[Type.F32, "write_only"],
    params: MatMulParams,
):
    tile_a = shared_array(Type.F32, (TILE_DIM, TILE_DIM), name="tile_a")
    tile_b = shared_array(Type.F32, (TILE_DIM, TILE_DIM), name="tile_b")

    lx = i32(lid.x)
    ly = i32(lid.y)
    row = i32(wid.y) * TILE_DIM + ly
    col = i32(wid.x) * TILE_DIM + lx

    acc = 0.0
    for t in range(params.num_tiles):
        a_col = t * TILE_DIM + lx
        a_val = 0.0
        if (row < params.m) and (a_col < params.k):
            a_val = mat_a[row * params.k + a_col]
        tile_a[ly, lx] = a_val

        b_row = t * TILE_DIM + ly
        b_val = 0.0
        if (b_row < params.k) and (col < params.n):
            b_val = mat_b[b_row * params.n + col]
        tile_b[ly, lx] = b_val

        workgroup_barrier()

        for p in UNROLLED_TILE_STEPS:
            acc = acc + tile_a[ly, p] * tile_b[p, lx]

        workgroup_barrier()

    if (row < params.m) and (col < params.n):
        mat_c[row * params.n + col] = acc


def make_coop_matmul_f32_kernel(
    tile_m: int,
    tile_n: int,
    tile_k: int,
    subgroup_size: int = 32,
):
    """Build a subgroup cooperative-matrix MatMul kernel for aligned M/N/K tiles."""

    @compute(workgroup_size=(subgroup_size, 1, 1))
    def coop_matmul_f32_kernel(
        wid: WorkgroupId,
        mat_a: Buffer[Type.F32, "read_only"],
        mat_b: Buffer[Type.F32, "read_only"],
        mat_c: Buffer[Type.F32, "write_only"],
        params: MatMulParams,
    ):
        row = i32(wid.y) * tile_m
        col = i32(wid.x) * tile_n
        acc = coop_matrix_zero(tile_m, tile_n, Type.F32)
        for t in range(params.num_tiles):
            k_off = t * tile_k
            a = coop_matrix_load(mat_a, row * params.k + k_off, params.k, tile_m, tile_k, "a")
            b = coop_matrix_load(mat_b, k_off * params.n + col, params.n, tile_k, tile_n, "b")
            acc = coop_matrix_mul_add(a, b, acc)
        coop_matrix_store(mat_c, row * params.n + col, acc, params.n)

    return coop_matmul_f32_kernel


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
    return ((n + TILE_DIM - 1) // TILE_DIM, (m + TILE_DIM - 1) // TILE_DIM, 1)


BINARY_KERNELS = {
    "add": add_f32_kernel,
    "mul": mul_f32_kernel,
}

SCALAR_KERNELS = {
    "add": scalar_add_f32_kernel,
    "mul": scalar_mul_f32_kernel,
}
