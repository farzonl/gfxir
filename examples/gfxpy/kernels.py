"""GfxIR @compute kernels for 1D contiguous float32 elementwise operations."""

from __future__ import annotations

from mlir_gpu import (
    Buffer,
    ParameterBlock,
    Type,
    compute,
    global_invocation_id,
    i32,
)

WORKGROUP_SIZE_1D = (64, 1, 1)

BinaryElementwiseParams = ParameterBlock(
    "gfxpy_binary_params",
    (("n", Type.I32),),
)

ScalarElementwiseParams = ParameterBlock(
    "gfxpy_scalar_params",
    (("n", Type.I32), ("scalar", Type.F32)),
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


def compute_grid_1d(length: int) -> tuple[int, int, int]:
    """Compute 1D workgroup dispatch grid for a non-empty length."""
    if length <= 0:
        raise ValueError(f"compute_grid_1d requires positive length, got {length}.")
    wg_x = WORKGROUP_SIZE_1D[0]
    return ((length + wg_x - 1) // wg_x, 1, 1)


BINARY_KERNELS = {
    "add": add_f32_kernel,
    "mul": mul_f32_kernel,
}

SCALAR_KERNELS = {
    "add": scalar_add_f32_kernel,
    "mul": scalar_mul_f32_kernel,
}
