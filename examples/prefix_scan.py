"""Bounded workgroup-shared parallel prefix scan compute benchmark authored in Python."""

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
    GlobalInvocationId,
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
    select,
    shared_array,
    workgroup_barrier,
)

BLOCK_SIZE = 256
MAX_ELEMENTS = BLOCK_SIZE * BLOCK_SIZE

PrefixScanParams = ParameterBlock(
    "prefix_scan_params",
    (
        ("count", Type.I32),
        ("exclusive", Type.I32),
    ),
)

BlockOffsetParams = ParameterBlock(
    "block_offset_params",
    (
        ("count", Type.I32),
    ),
)


@compute(workgroup_size=(BLOCK_SIZE, 1, 1))
def workgroup_prefix_scan(
    gid: GlobalInvocationId,
    lid: LocalInvocationId,
    wid: WorkgroupId,
    inp: Buffer[Type.I32, "read_only"],
    out: Buffer[Type.I32, "write_only"],
    block_sums: Buffer[Type.I32, "write_only"],
    params: PrefixScanParams,
):
    buf_a = shared_array(Type.I32, BLOCK_SIZE, name="scan_a")
    buf_b = shared_array(Type.I32, BLOCK_SIZE, name="scan_b")

    lx = i32(lid.x)
    gx = i32(gid.x)
    wx = i32(wid.x)

    val = 0
    if gx < params.count:
        val = inp[gx]
    buf_a[lx] = val
    workgroup_barrier()

    stride = 1
    while stride < BLOCK_SIZE:
        v1 = buf_a[lx]
        if lx >= stride:
            v1 = v1 + buf_a[lx - stride]
        buf_b[lx] = v1
        workgroup_barrier()

        stride2 = stride * 2
        v2 = buf_b[lx]
        if lx >= stride2:
            v2 = v2 + buf_b[lx - stride2]
        buf_a[lx] = v2
        workgroup_barrier()

        stride = stride * 4

    if lx == (BLOCK_SIZE - 1):
        block_sums[wx] = buf_a[BLOCK_SIZE - 1]

    if gx < params.count:
        inc = buf_a[lx]
        out[gx] = select(params.exclusive != 0, inc - val, inc)


@compute(workgroup_size=(BLOCK_SIZE, 1, 1))
def add_block_offsets(
    gid: GlobalInvocationId,
    wid: WorkgroupId,
    block_offsets: Buffer[Type.I32, "read_only"],
    data: Buffer[Type.I32, "read_write"],
    params: BlockOffsetParams,
):
    gx = i32(gid.x)
    wx = i32(wid.x)
    if (wx > 0) and (gx < params.count):
        data[gx] = data[gx] + block_offsets[wx]


scan_pipeline = ComputePipeline(workgroup_prefix_scan, dispatch=Dispatch(grid=(1, 1, 1)))
offset_pipeline = ComputePipeline(add_block_offsets, dispatch=Dispatch(grid=(1, 1, 1)))


@dataclass(frozen=True)
class PrefixScanRunResult:
    values: list[int]
    elapsed_ms: float
    device: str


def cpu_prefix_scan(values: list[int], *, exclusive: bool = False) -> list[int]:
    """Exact 32-bit signed integer CPU reference prefix sum."""
    result: list[int] = []
    running = 0
    for item in values:
        if exclusive:
            result.append(running)
        running = (running + item) & 0xFFFF_FFFF
        if running >= 2**31:
            running -= 2**32
        if not exclusive:
            result.append(running)
    return result


def run_prefix_scan(
    values: list[int],
    target: Target,
    *,
    exclusive: bool = False,
    validation: bool = False,
    provider: str | None = None,
    toolchain: Toolchain | None = None,
) -> PrefixScanRunResult:
    """Execute bounded single- or two-level parallel prefix scan on the target GPU."""
    count = len(values)
    if count == 0:
        return PrefixScanRunResult(values=[], elapsed_ms=0.0, device="empty")
    if count > MAX_ELEMENTS:
        raise ValueError(f"Input length {count} exceeds bounded prefix scan capacity {MAX_ELEMENTS}.")

    num_blocks = (count + BLOCK_SIZE - 1) // BLOCK_SIZE
    inp_bytes = struct.pack(f"<{count}i", *values)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        scan_manifest = compile_pipeline(scan_pipeline, target, td_path / "scan", toolchain)
        offset_manifest = (
            compile_pipeline(offset_pipeline, target, td_path / "offset", toolchain)
            if num_blocks > 1
            else None
        )

        with compute_session(
            scan_manifest,
            validation=validation,
            provider=provider,
            toolchain=toolchain,
        ) as exec_scan:
            t0 = time.perf_counter()
            step1 = exec_scan(
                buffers={
                    "inp": inp_bytes,
                    "out": count * 4,
                    "block_sums": num_blocks * 4,
                },
                parameters={"count": count, "exclusive": 1 if exclusive else 0},
                grid=(num_blocks, 1, 1),
            )
            out_bytes = step1.buffers["out"]
            device_name = step1.device

            if num_blocks > 1:
                step2 = exec_scan(
                    buffers={
                        "inp": step1.buffers["block_sums"],
                        "out": num_blocks * 4,
                        "block_sums": 4,
                    },
                    parameters={"count": num_blocks, "exclusive": 1},
                    grid=(1, 1, 1),
                )
                assert offset_manifest is not None
                with compute_session(
                    offset_manifest,
                    validation=validation,
                    provider=provider,
                    toolchain=toolchain,
                ) as exec_offset:
                    step3 = exec_offset(
                        buffers={
                            "block_offsets": step2.buffers["out"],
                            "data": out_bytes,
                        },
                        parameters={"count": count},
                        grid=(num_blocks, 1, 1),
                    )
                    out_bytes = step3.buffers["data"]

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

    scanned = list(struct.unpack(f"<{count}i", out_bytes))
    return PrefixScanRunResult(values=scanned, elapsed_ms=elapsed_ms, device=device_name)


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
    parser = argparse.ArgumentParser(description="Run bounded GPU parallel prefix scan benchmark.")
    parser.add_argument("--count", type=int, default=4096, help="Number of 32-bit integers to scan")
    parser.add_argument("--exclusive", action="store_true", help="Compute exclusive scan instead of inclusive")
    parser.add_argument("--target", choices=[t.value for t in Target], default="vulkan")
    parser.add_argument("--provider", default=None, help="Optional runtime provider (warp, lavapipe, swiftshader, hardware)")
    parser.add_argument("--llvm-bin", type=Path, default=None, help="Optional directory containing gfx-compile and llc")
    parser.add_argument("--tools-bin", type=Path, default=None, help="Optional directory containing spirv-val / dxv / viewers")
    args = parser.parse_args()

    toolchain = _default_cli_toolchain(args.llvm_bin, args.tools_bin)
    inputs = [((i * 19 + 7) % 31) - 12 for i in range(args.count)]
    expected = cpu_prefix_scan(inputs, exclusive=args.exclusive)
    res = run_prefix_scan(
        inputs,
        Target(args.target),
        exclusive=args.exclusive,
        provider=args.provider,
        toolchain=toolchain,
    )
    if res.values != expected:
        raise RuntimeError("GPU prefix scan output diverged from CPU reference")
    mode = "exclusive" if args.exclusive else "inclusive"
    print(
        f"Prefix scan ({mode}, {args.count} i32 elements) verified on {res.device} "
        f"in {res.elapsed_ms:.2f} ms"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
