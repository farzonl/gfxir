"""Tiled shared-memory matrix multiplication compute benchmark without tensor-core assumptions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import struct
import tempfile
import time

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    LocalInvocationId,
    ParameterBlock,
    Target,
    Toolchain,
    Type,
    WorkgroupId,
    compile_pipeline,
    compute,
    compute_session,
    i32,
    shared_array,
    workgroup_barrier,
)

TILE_DIM = 16
UNROLLED_TILE_STEPS = tuple(range(TILE_DIM))

MatMulParams = ParameterBlock(
    "matmul_params",
    (
        ("m", Type.I32),
        ("n", Type.I32),
        ("k", Type.I32),
        ("num_tiles", Type.I32),
    ),
)


@compute(workgroup_size=(TILE_DIM, TILE_DIM, 1))
def tiled_matmul_kernel(
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


pipeline = ComputePipeline(tiled_matmul_kernel, dispatch=Dispatch(grid=(4, 4, 1)))


@dataclass(frozen=True)
class MatMulRunResult:
    c: list[float]
    m: int
    n: int
    k: int
    elapsed_ms: float
    gflops: float
    device: str


def create_deterministic_matrices(m: int, n: int, k: int) -> tuple[list[float], list[float]]:
    """Create bounded deterministic row-major f32 matrices A (m x k) and B (k x n)."""
    a = [(((idx * 37 + 11) % 101) - 50) * 0.05 for idx in range(m * k)]
    b = [(((idx * 53 + 23) % 97) - 48) * 0.05 for idx in range(k * n)]
    return a, b


def cpu_matmul(a: list[float], b: list[float], m: int, n: int, k: int) -> list[float]:
    """Reference CPU row-major matrix multiplication C = A @ B."""
    c = [0.0] * (m * n)
    for r in range(m):
        a_row_base = r * k
        c_row_base = r * n
        for p in range(k):
            a_val = a[a_row_base + p]
            b_row_base = p * n
            for col in range(n):
                c[c_row_base + col] += a_val * b[b_row_base + col]
    return c


def run_tiled_matmul(
    a: list[float],
    b: list[float],
    m: int,
    n: int,
    k: int,
    target: Target,
    *,
    validation: bool = False,
    provider: str | None = None,
    toolchain: Toolchain | None = None,
) -> MatMulRunResult:
    """Compile and execute tiled shared-memory C = A @ B on the target GPU."""
    if len(a) != m * k:
        raise ValueError(f"Expected {m * k} elements in A, got {len(a)}")
    if len(b) != k * n:
        raise ValueError(f"Expected {k * n} elements in B, got {len(b)}")

    grid_x = (n + TILE_DIM - 1) // TILE_DIM
    grid_y = (m + TILE_DIM - 1) // TILE_DIM
    num_tiles = (k + TILE_DIM - 1) // TILE_DIM

    a_bytes = struct.pack(f"<{m * k}f", *a)
    b_bytes = struct.pack(f"<{k * n}f", *b)

    with tempfile.TemporaryDirectory() as td:
        manifest = compile_pipeline(pipeline, target, Path(td) / "matmul", toolchain)
        with compute_session(
            manifest,
            validation=validation,
            provider=provider,
            toolchain=toolchain,
        ) as run:
            t0 = time.perf_counter()
            res = run(
                buffers={
                    "mat_a": a_bytes,
                    "mat_b": b_bytes,
                    "mat_c": m * n * 4,
                },
                parameters={
                    "m": m,
                    "n": n,
                    "k": k,
                    "num_tiles": num_tiles,
                },
                grid=(grid_x, grid_y, 1),
            )
            elapsed_s = time.perf_counter() - t0

    c_vals = list(struct.unpack(f"<{m * n}f", res.buffers["mat_c"]))
    elapsed_ms = elapsed_s * 1000.0
    flops = 2.0 * float(m) * float(n) * float(k)
    gflops = (flops / elapsed_s / 1e9) if elapsed_s > 0.0 else 0.0
    return MatMulRunResult(
        c=c_vals,
        m=m,
        n=n,
        k=k,
        elapsed_ms=elapsed_ms,
        gflops=gflops,
        device=res.device,
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
    parser = argparse.ArgumentParser(description="Run tiled shared-memory matrix multiplication benchmark.")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--k", type=int, default=128)
    parser.add_argument("--target", choices=[t.value for t in Target], default="vulkan")
    parser.add_argument("--provider", default=None, help="Optional runtime provider (warp, lavapipe, swiftshader, hardware)")
    parser.add_argument("--llvm-bin", type=Path, default=None, help="Optional directory containing gfx-compile and llc")
    parser.add_argument("--tools-bin", type=Path, default=None, help="Optional directory containing spirv-val / dxv / viewers")
    args = parser.parse_args()

    toolchain = _default_cli_toolchain(args.llvm_bin, args.tools_bin)
    a, b = create_deterministic_matrices(args.m, args.n, args.k)
    expected = cpu_matmul(a, b, args.m, args.n, args.k)
    res = run_tiled_matmul(
        a,
        b,
        args.m,
        args.n,
        args.k,
        Target(args.target),
        provider=args.provider,
        toolchain=toolchain,
    )
    max_err = max(abs(x - y) for x, y in zip(res.c, expected))
    if max_err > 1e-3:
        raise RuntimeError(f"Tiled matmul max error {max_err} exceeded tolerance 1e-3")
    print(
        f"Tiled MatMul ({args.m}x{args.k} @ {args.k}x{args.n}) verified on {res.device}: "
        f"{res.elapsed_ms:.2f} ms ({res.gflops:.3f} GFLOP/s, max_err={max_err:.2e})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
