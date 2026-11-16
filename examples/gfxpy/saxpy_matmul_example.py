"""NumPy-style 4M SAXPY and 4096x4096 MatMul example using gfxpy."""

from __future__ import annotations

from pathlib import Path
import sys
import time

# Allow running directly from an uninstalled checkout without PYTHONPATH.
_EXAMPLES_DIR = str(Path(__file__).resolve().parents[1])
_SELF_DIR = str(Path(__file__).resolve().parent)
if _SELF_DIR in sys.path:
    sys.path.remove(_SELF_DIR)
if _EXAMPLES_DIR not in sys.path:
    sys.path.insert(0, _EXAMPLES_DIR)

import gfxpy as np  # noqa: E402

SAXPY_N = 4_194_304  # 4M
MATMUL_N = 4096      # 4096 x 4096

# SAXPY stands for Single-Precision A * X + Y
def run_saxpy(n: int = SAXPY_N, alpha: float = 1.25) -> None:
    x = np.array([1.0] * n, dtype=np.float32)
    y = np.array([2.0] * n, dtype=np.float32)

    t0 = time.perf_counter()
    z = alpha * x + y
    res = z.tomemoryview()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert res[0] == 3.25 and res[-1] == 3.25
    gflops = (2.0 * n) / (elapsed_ms / 1000.0) / 1e9
    print(f"SAXPY ({n}): {elapsed_ms:.2f} ms ({gflops:.2f} GFLOP/s), z[0]={res[0]}")


def run_matmul(n: int = MATMUL_N) -> None:
    a = np.array([[0.5] * n for _ in range(n)], dtype=np.float32)
    b = np.array([[0.25] * n for _ in range(n)], dtype=np.float32)

    t0 = time.perf_counter()
    c = a @ b
    res = c.tomemoryview()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    expected = n * 0.5 * 0.25
    assert res[0, 0] == expected and res[n - 1, n - 1] == expected
    gflops = (2.0 * n * n * n) / (elapsed_ms / 1000.0) / 1e9
    print(f"MatMul ({n}x{n}): {elapsed_ms:.2f} ms ({gflops:.2f} GFLOP/s), c[0,0]={res[0, 0]}")


if __name__ == "__main__":
    run_saxpy()
    run_matmul()
