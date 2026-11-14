#!/usr/bin/env python3
"""Cross-platform benchmark runner comparing NumPy (float32) against GfxIR GPU compute."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import statistics
import struct
import sys
import tempfile
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

_TILE_DIM = 16
_UNROLLED = tuple(range(_TILE_DIM))
try:
    from mlir_gpu import ParameterBlock as _ParameterBlock, Type as _Type

    SaxpyParams = _ParameterBlock(
        "saxpy_params",
        (("n", _Type.I32), ("alpha", _Type.F32)),
    )
    MatMulParams = _ParameterBlock(
        "matmul_params",
        (("m", _Type.I32), ("n", _Type.I32), ("k", _Type.I32), ("num_tiles", _Type.I32)),
    )
except Exception:
    SaxpyParams = None  # type: ignore[assignment]
    MatMulParams = None  # type: ignore[assignment]


def _get_rss_mb() -> float:
    """Return current/peak resident set size in MiB across POSIX and Windows."""
    if sys.platform == "win32":
        try:
            import ctypes
            from ctypes import wintypes

            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]

            counters = PROCESS_MEMORY_COUNTERS()
            counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                return float(counters.WorkingSetSize) / (1024.0 * 1024.0)
        except Exception:
            return 0.0
        return 0.0

    try:
        import resource

        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes; Linux reports KiB.
        divisor = (1024.0 * 1024.0) if sys.platform == "darwin" else 1024.0
        return float(ru.ru_maxrss) / divisor
    except Exception:
        return 0.0


def default_target_name() -> str:
    """Select the default GPU backend for the host operating system."""
    if sys.platform == "darwin":
        return "metal"
    if sys.platform == "win32":
        return "directx"
    if sys.platform.startswith("linux"):
        return "vulkan"
    raise RuntimeError(f"Unsupported platform for default GPU target: {sys.platform}")


def resolve_target(target_arg: str) -> str:
    """Resolve 'auto' or validate an explicit target name."""
    name = default_target_name() if target_arg == "auto" else target_arg.lower()
    if name not in {"metal", "vulkan", "directx"}:
        raise ValueError(f"Unsupported target '{target_arg}'. Expected auto, metal, vulkan, or directx.")
    return name


def _summarize_timings_ms(samples_ms: list[float]) -> dict[str, float]:
    """Compute mean, median, min, and p95 latency in milliseconds."""
    if not samples_ms:
        return {"meanMs": 0.0, "medianMs": 0.0, "minMs": 0.0, "p95Ms": 0.0}
    ordered = sorted(samples_ms)
    p95_idx = min(len(ordered) - 1, int(math.ceil(0.95 * len(ordered))) - 1)
    return {
        "meanMs": round(statistics.fmean(ordered), 4),
        "medianMs": round(statistics.median(ordered), 4),
        "minMs": round(ordered[0], 4),
        "p95Ms": round(ordered[p95_idx], 4),
    }


def _geomean(values: list[float]) -> float:
    """Return geometric mean of positive numbers, or 0.0 if empty."""
    positives = [v for v in values if v > 0.0]
    if not positives:
        return 0.0
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


@dataclass(frozen=True)
class SuiteSpec:
    """Definition of a single benchmark suite."""

    bench_id: str
    bench_name: str
    category: str
    kind: str  # "saxpy_chain" | "matmul"
    size_label: str
    n: int
    m: int = 0
    k: int = 0
    chain_depth: int = 1


def build_suites(preset: str) -> list[SuiteSpec]:
    """Return the benchmark suites for 'quick' or 'full' presets."""
    if preset == "quick":
        return [
            SuiteSpec("saxpy_16k_k1", "SAXPY 16K (1x)", "Elementwise 1D", "saxpy_chain", "16K x 1", n=16_384, chain_depth=1),
            SuiteSpec("saxpy_64k_k5", "SAXPY 64K (5x)", "Elementwise 1D", "saxpy_chain", "64K x 5", n=65_536, chain_depth=5),
            SuiteSpec("matmul_64", "MatMul 64x64", "Matrix Multiply", "matmul", "64x64", n=64, m=64, k=64),
            SuiteSpec("matmul_128", "MatMul 128x128", "Matrix Multiply", "matmul", "128x128", n=128, m=128, k=128),
        ]
    return [
        SuiteSpec("saxpy_64k_k1", "SAXPY 64K (1x)", "Elementwise 1D", "saxpy_chain", "64K x 1", n=65_536, chain_depth=1),
        SuiteSpec("saxpy_1m_k1", "SAXPY 1M (1x)", "Elementwise 1D", "saxpy_chain", "1M x 1", n=1_048_576, chain_depth=1),
        SuiteSpec("saxpy_1m_k10", "SAXPY 1M (10x)", "Elementwise 1D", "saxpy_chain", "1M x 10", n=1_048_576, chain_depth=10),
        SuiteSpec("saxpy_4m_k10", "SAXPY 4M (10x)", "Elementwise 1D", "saxpy_chain", "4M x 10", n=4_194_304, chain_depth=10),
        SuiteSpec("matmul_64", "MatMul 64x64", "Matrix Multiply", "matmul", "64x64", n=64, m=64, k=64),
        SuiteSpec("matmul_256", "MatMul 256x256", "Matrix Multiply", "matmul", "256x256", n=256, m=256, k=256),
        SuiteSpec("matmul_512", "MatMul 512x512", "Matrix Multiply", "matmul", "512x512", n=512, m=512, k=512),
    ]


def _make_record(
    spec: SuiteSpec,
    backend: str,
    *,
    ran: bool,
    status: str,
    compile_ms: float = 0.0,
    cold_ms: float = 0.0,
    samples_ms: list[float] | None = None,
    rss_before_mb: float = 0.0,
    rss_after_run_mb: float = 0.0,
    max_abs_error: float = 0.0,
    runtime_mode: str = "",
) -> dict[str, Any]:
    stats = _summarize_timings_ms(samples_ms or [])
    return {
        "benchId": spec.bench_id,
        "benchName": spec.bench_name,
        "category": spec.category,
        "sizeLabel": spec.size_label,
        "backend": backend,
        "runtimeMode": runtime_mode,
        "ran": ran,
        "compileMs": round(compile_ms, 4),
        "coldMs": round(cold_ms, 4),
        "meanMs": stats["meanMs"],
        "medianMs": stats["medianMs"],
        "minMs": stats["minMs"],
        "p95Ms": stats["p95Ms"],
        "rssBeforeMb": round(rss_before_mb, 2),
        "rssAfterRunMb": round(rss_after_run_mb, 2),
        "peakRssMb": round(max(rss_before_mb, rss_after_run_mb), 2),
        "maxAbsError": float(f"{max_abs_error:.3e}"),
        "status": status,
    }


def _benchmark_numpy_suite(spec: SuiteSpec, np: Any, warmup: int, iters: int) -> tuple[dict[str, Any], Any]:
    """Run a suite on NumPy float32 and return (record, reference_output)."""
    rss_before = _get_rss_mb()
    rng = np.random.default_rng(42)

    if spec.kind == "saxpy_chain":
        x = rng.standard_normal(spec.n, dtype=np.float32)
        y = rng.standard_normal(spec.n, dtype=np.float32)
        alpha = np.float32(1.25)

        def op() -> Any:
            cur = x
            for _ in range(spec.chain_depth):
                cur = cur * alpha + y
            return cur
    else:
        a = rng.standard_normal((spec.m, spec.k), dtype=np.float32)
        b = rng.standard_normal((spec.k, spec.n), dtype=np.float32)

        def op() -> Any:
            return a @ b

    t0 = time.perf_counter()
    ref = op()
    cold_ms = (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        op()

    samples: list[float] = []
    for _ in range(iters):
        t_start = time.perf_counter()
        _ = op()
        samples.append((time.perf_counter() - t_start) * 1000.0)

    rss_after = _get_rss_mb()
    rec = _make_record(
        spec,
        "numpy-cpu-f32",
        ran=True,
        status="ok",
        compile_ms=0.0,
        cold_ms=cold_ms,
        samples_ms=samples,
        rss_before_mb=rss_before,
        rss_after_run_mb=rss_after,
        max_abs_error=0.0,
        runtime_mode=f"numpy {np.__version__}",
    )
    return rec, ref


def _benchmark_gfx_suite(
    spec: SuiteSpec,
    target_name: str,
    provider: str | None,
    warmup: int,
    iters: int,
    np: Any | None,
    ref_output: Any | None,
) -> dict[str, Any]:
    """Run a suite on GfxIR GPU compute and return its result record."""
    backend_label = f"gfxir-{target_name}"
    rss_before = _get_rss_mb()

    try:
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
            global_invocation_id,
            i32,
            shared_array,
            workgroup_barrier,
        )
    except Exception as exc:
        return _make_record(spec, backend_label, ran=False, status=f"unavailable: {exc}")

    target_map = {
        "metal": Target.METAL,
        "vulkan": Target.VULKAN,
        "directx": Target.DIRECTX,
    }
    target = target_map[target_name]
    toolchain = Toolchain(
        llvm_bin=Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None,
        tools_bin=Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None,
    )

    try:
        with tempfile.TemporaryDirectory() as td:
            t_compile0 = time.perf_counter()
            if spec.kind == "saxpy_chain":
                @compute(workgroup_size=(64, 1, 1))
                def saxpy_kernel(
                    x: Buffer[Type.F32, "read_only"],
                    y: Buffer[Type.F32, "read_only"],
                    out: Buffer[Type.F32, "write_only"],
                    params: SaxpyParams,
                ):
                    idx = i32(global_invocation_id().x)
                    if idx < params.n:
                        out[idx] = x[idx] * params.alpha + y[idx]

                pipeline = ComputePipeline(saxpy_kernel, Dispatch(grid=(1, 1, 1)))
                manifest = compile_pipeline(pipeline, target, Path(td) / "saxpy", toolchain)
                compile_ms = (time.perf_counter() - t_compile0) * 1000.0

                if np is not None:
                    rng = np.random.default_rng(42)
                    x_bytes = rng.standard_normal(spec.n, dtype=np.float32).tobytes()
                    y_bytes = rng.standard_normal(spec.n, dtype=np.float32).tobytes()
                else:
                    x_bytes = struct.pack(f"<{spec.n}f", *([1.0] * spec.n))
                    y_bytes = struct.pack(f"<{spec.n}f", *([2.0] * spec.n))
                grid = ((spec.n + 63) // 64, 1, 1)

                def run_once(run_fn: Any) -> bytes:
                    cur = x_bytes
                    for _ in range(spec.chain_depth):
                        res = run_fn(
                            buffers={"x": cur, "y": y_bytes, "out": spec.n * 4},
                            parameters={"n": spec.n, "alpha": 1.25},
                            grid=grid,
                        )
                        cur = res.buffers["out"]
                    return cur
            else:
                tile_dim = _TILE_DIM

                @compute(workgroup_size=(_TILE_DIM, _TILE_DIM, 1))
                def tiled_matmul_kernel(
                    lid: LocalInvocationId,
                    wid: WorkgroupId,
                    mat_a: Buffer[Type.F32, "read_only"],
                    mat_b: Buffer[Type.F32, "read_only"],
                    mat_c: Buffer[Type.F32, "write_only"],
                    params: MatMulParams,
                ):
                    tile_a = shared_array(Type.F32, (_TILE_DIM, _TILE_DIM), name="tile_a")
                    tile_b = shared_array(Type.F32, (_TILE_DIM, _TILE_DIM), name="tile_b")
                    lx = i32(lid.x)
                    ly = i32(lid.y)
                    row = i32(wid.y) * _TILE_DIM + ly
                    col = i32(wid.x) * _TILE_DIM + lx
                    acc = 0.0
                    for t in range(params.num_tiles):
                        a_col = t * _TILE_DIM + lx
                        a_val = 0.0
                        if (row < params.m) and (a_col < params.k):
                            a_val = mat_a[row * params.k + a_col]
                        tile_a[ly, lx] = a_val

                        b_row = t * _TILE_DIM + ly
                        b_val = 0.0
                        if (b_row < params.k) and (col < params.n):
                            b_val = mat_b[b_row * params.n + col]
                        tile_b[ly, lx] = b_val
                        workgroup_barrier()
                        for p in _UNROLLED:
                            acc = acc + tile_a[ly, p] * tile_b[p, lx]
                        workgroup_barrier()
                    if (row < params.m) and (col < params.n):
                        mat_c[row * params.n + col] = acc

                pipeline = ComputePipeline(tiled_matmul_kernel, Dispatch(grid=(1, 1, 1)))
                manifest = compile_pipeline(pipeline, target, Path(td) / "matmul", toolchain)
                compile_ms = (time.perf_counter() - t_compile0) * 1000.0

                if np is not None:
                    rng = np.random.default_rng(42)
                    a_bytes = rng.standard_normal((spec.m, spec.k), dtype=np.float32).tobytes()
                    b_bytes = rng.standard_normal((spec.k, spec.n), dtype=np.float32).tobytes()
                else:
                    a_bytes = struct.pack(f"<{spec.m * spec.k}f", *([1.0] * (spec.m * spec.k)))
                    b_bytes = struct.pack(f"<{spec.k * spec.n}f", *([1.0] * (spec.k * spec.n)))
                grid = ((spec.n + tile_dim - 1) // tile_dim, (spec.m + tile_dim - 1) // tile_dim, 1)
                num_tiles = (spec.k + tile_dim - 1) // tile_dim

                def run_once(run_fn: Any) -> bytes:
                    res = run_fn(
                        buffers={"mat_a": a_bytes, "mat_b": b_bytes, "mat_c": spec.m * spec.n * 4},
                        parameters={"m": spec.m, "n": spec.n, "k": spec.k, "num_tiles": num_tiles},
                        grid=grid,
                    )
                    return res.buffers["mat_c"]

            with compute_session(manifest, validation=False, provider=provider, toolchain=toolchain) as run_fn:
                t_cold0 = time.perf_counter()
                out_bytes = run_once(run_fn)
                cold_ms = compile_ms + (time.perf_counter() - t_cold0) * 1000.0

                for _ in range(warmup):
                    run_once(run_fn)

                samples: list[float] = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    _ = run_once(run_fn)
                    samples.append((time.perf_counter() - t0) * 1000.0)

        max_err = 0.0
        if np is not None and ref_output is not None:
            got = np.frombuffer(out_bytes, dtype=np.float32).reshape(ref_output.shape)
            max_err = float(np.max(np.abs(got - ref_output)))

        rss_after = _get_rss_mb()
        return _make_record(
            spec,
            backend_label,
            ran=True,
            status="ok",
            compile_ms=compile_ms,
            cold_ms=cold_ms,
            samples_ms=samples,
            rss_before_mb=rss_before,
            rss_after_run_mb=rss_after,
            max_abs_error=max_err,
            runtime_mode="mlir_gpu.compute_session",
        )
    except Exception as exc:
        return _make_record(
            spec,
            backend_label,
            ran=False,
            status=f"unavailable: {exc}",
            rss_before_mb=rss_before,
            rss_after_run_mb=_get_rss_mb(),
        )


def run_benchmarks(
    *,
    target_arg: str,
    preset: str,
    warmup: int,
    iters: int,
    provider: str | None = None,
    skip_gpu: bool = False,
) -> dict[str, Any]:
    """Run all benchmark suites and return the structured JSON report."""
    target_name = resolve_target(target_arg)
    suites = build_suites(preset)

    try:
        import numpy as np
    except ImportError:
        np = None

    results: list[dict[str, Any]] = []
    numpy_medians: list[float] = []
    gfx_medians: list[float] = []

    for spec in suites:
        ref_output = None
        if np is not None:
            np_rec, ref_output = _benchmark_numpy_suite(spec, np, warmup=warmup, iters=iters)
            results.append(np_rec)
            if np_rec["ran"] and np_rec["medianMs"] > 0:
                numpy_medians.append(np_rec["medianMs"])
        else:
            results.append(_make_record(spec, "numpy-cpu-f32", ran=False, status="unavailable: numpy not installed"))

        if skip_gpu:
            results.append(_make_record(spec, f"gfxir-{target_name}", ran=False, status="skipped: --skip-gpu"))
        else:
            gfx_rec = _benchmark_gfx_suite(spec, target_name, provider, warmup, iters, np, ref_output)
            results.append(gfx_rec)
            if gfx_rec["ran"] and gfx_rec["medianMs"] > 0:
                gfx_medians.append(gfx_rec["medianMs"])

    np_geo = round(_geomean(numpy_medians), 4)
    gfx_geo = round(_geomean(gfx_medians), 4)
    overall_winner = ""
    if np_geo > 0 and gfx_geo > 0:
        overall_winner = f"gfxir-{target_name}" if gfx_geo < np_geo else "numpy-cpu-f32"
    elif np_geo > 0:
        overall_winner = "numpy-cpu-f32"
    elif gfx_geo > 0:
        overall_winner = f"gfxir-{target_name}"

    return {
        "version": 1,
        "os": platform.system(),
        "arch": platform.machine(),
        "python": platform.python_version(),
        "numpyVersion": np.__version__ if np is not None else None,
        "target": target_name,
        "provider": provider or "default",
        "preset": preset,
        "overallWinner": overall_winner,
        "overallGeomeanMs": {
            "numpy-cpu-f32": np_geo,
            f"gfxir-{target_name}": gfx_geo,
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run NumPy vs GfxIR GPU compute benchmarks.")
    parser.add_argument("--target", default="auto", choices=["auto", "metal", "vulkan", "directx"])
    parser.add_argument("--preset", default="quick", choices=["quick", "full"])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--provider", default=None)
    parser.add_argument("--skip-gpu", action="store_true", help="Run only the NumPy baseline (for host-only testing).")
    parser.add_argument("--require-gpu", action="store_true", help="Exit non-zero if no GPU benchmark executed.")
    parser.add_argument("--list-suites", action="store_true", help="Print suite IDs for the chosen preset and exit.")
    parser.add_argument("--out", type=Path, default=Path("bench-results.json"))
    args = parser.parse_args()

    if args.list_suites:
        for spec in build_suites(args.preset):
            print(f"{spec.bench_id}: {spec.bench_name} ({spec.category})")
        return 0

    report = run_benchmarks(
        target_arg=args.target,
        preset=args.preset,
        warmup=args.warmup,
        iters=args.iters,
        provider=args.provider,
        skip_gpu=args.skip_gpu,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote benchmark report to {args.out}")

    gpu_ran = any(r["ran"] and r["backend"].startswith("gfxir-") for r in report["results"])
    if args.require_gpu and not gpu_ran:
        print("Error: --require-gpu set, but no GPU benchmark executed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
