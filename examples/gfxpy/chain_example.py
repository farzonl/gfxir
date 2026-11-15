"""Cross-backend NumPy-free gfxpy chained GPU array example.

Run directly from a repository checkout:
    python examples/gfxpy/chain_example.py --target auto
    python examples/gfxpy/chain_example.py --target directx --provider warp
    python examples/gfxpy/chain_example.py --target vulkan --provider swiftshader
    python examples/gfxpy/chain_example.py --target metal
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_EXAMPLES_DIR, _SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.append(str(_p))

import gfxpy as gp  # noqa: E402
from gfxpy.gpu_backend import GfxIRComputeBackend  # noqa: E402
from mlir_gpu import Toolchain  # noqa: E402


def _default_toolchain() -> Toolchain:
    llvm_env = os.environ.get("LLVM_BIN")
    tools_env = os.environ.get("GPU_TOOLS_BIN")
    llvm_bin = Path(llvm_env) if llvm_env else (_REPO_ROOT / ".toolchains" / "llvm-build" / "bin")
    return Toolchain(
        llvm_bin=llvm_bin if llvm_bin.is_dir() else None,
        tools_bin=Path(tools_env) if tools_env else None,
    )


class _CountingBackend:
    """Thin proxy around GfxIRComputeBackend to record host transfers and GPU dispatches."""

    def __init__(self, inner: GfxIRComputeBackend) -> None:
        self._inner = inner
        self.uploads = 0
        self.downloads = 0
        self.dispatches: list[str] = []

    @property
    def target(self) -> str:
        return self._inner.target

    @property
    def closed(self) -> bool:
        return self._inner.closed

    @property
    def device_name(self) -> str:
        return self._inner.device_name

    @property
    def inner(self) -> GfxIRComputeBackend:
        return self._inner

    def allocate(self, size_bytes: int):
        return self._inner.allocate(size_bytes)

    def upload(self, buf, data: bytes) -> None:
        self._inner.upload(buf, data)
        self.uploads += 1

    def download(self, buf) -> bytes:
        res = self._inner.download(buf)
        self.downloads += 1
        return res

    def binary_op(self, op: str, lhs, rhs, out, length: int) -> None:
        self._inner.binary_op(op, lhs, rhs, out, length)
        self.dispatches.append(f"binary:{op}")

    def scalar_op(self, op: str, lhs, scalar: float, out, length: int) -> None:
        self._inner.scalar_op(op, lhs, scalar, out, length)
        self.dispatches.append(f"scalar:{op}")

    def fused_op(self, spec, inputs, scalars, out, length: int) -> None:
        self._inner.fused_op(spec, inputs, scalars, out, length)
        self.dispatches.append(f"fused:{spec.signature}")

    def close(self) -> None:
        self._inner.close()


def _python_reference(
    a_vals: list[float],
    b_vals: list[float],
    c_vals: list[float],
) -> list[float]:
    return [((a * b + 1.5) * 2.0 + c) for a, b, c in zip(a_vals, b_vals, c_vals)]


def format_provider_label(ctx: object) -> str:
    """Return the context's provider label without assuming 'hardware' when metadata is absent."""
    provider = getattr(ctx, "provider", None)
    if provider:
        return str(provider)
    is_software = getattr(ctx, "is_software", None)
    if is_software is True:
        return "software"
    if is_software is False:
        return "hardware"
    return "unknown"


def run_example(
    *,
    target: str = "auto",
    provider: str | None = None,
    validation: bool = False,
    length: int = 137,
) -> int:
    a_vals = [float(i) * 0.25 - 5.0 for i in range(length)]
    b_vals = [float((i % 7) + 1) * 0.5 for i in range(length)]
    c_vals = [float(i % 11) * -0.75 for i in range(length)]
    expected = _python_reference(a_vals, b_vals, c_vals)

    resolved_target = gp.normalize_target(target)
    try:
        raw_backend = GfxIRComputeBackend(
            resolved_target,
            provider=provider,
            validation=validation,
            toolchain=_default_toolchain(),
        )
        counted = _CountingBackend(raw_backend)
        dev = gp.device(resolved_target, backend=counted)
    except gp.GfxPyError as exc:
        print(f"SKIPPED/UNAVAILABLE [{resolved_target}]: {exc}")
        return 2

    try:
        with dev:
            a = gp.array(a_vals, dtype=gp.float32)
            b = gp.array(b_vals, dtype=gp.float32)
            c = gp.array(c_vals, dtype=gp.float32)

            # Chained GPU expression: intermediate tensors remain device-resident.
            y = (a * b + 1.5) * 2.0 + c
            downloads_before = counted.downloads
            actual = y.tolist()
            downloads_after = counted.downloads

            # Empty array handling avoids zero-byte GPU allocations or zero-grid dispatches.
            empty_res = (gp.array([]) * 2.0 + 1.0).tolist()

            max_err = max((abs(g - e) for g, e in zip(actual, expected)), default=0.0)
            provider_label = format_provider_label(raw_backend._ctx)

            print(f"Target:      {dev.target}")
            print(f"Device:      {raw_backend.device_name}")
            print(f"Provider:    {provider_label}")
            print(f"Length:      {length} (non-workgroup-aligned)")
            print(f"Dispatches:  {len(counted.dispatches)} ({', '.join(counted.dispatches)})")
            print(f"Uploads:     {counted.uploads}")
            print(f"Downloads:   before tolist()={downloads_before}, after tolist()={downloads_after}")
            print(f"Max error:   {max_err:.3e}")
            print(f"Empty check: {empty_res}")

            if downloads_before != 0 or downloads_after != 1:
                raise AssertionError(
                    f"Expected 0 intermediate downloads and 1 final download, got {downloads_before} -> {downloads_after}"
                )
            if max_err > 1e-4:
                raise AssertionError(f"Numerical mismatch vs Python reference: max_err={max_err}")
            if empty_res != []:
                raise AssertionError(f"Expected empty list for empty array chain, got {empty_res}")
    finally:
        dev.close()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run NumPy-free gfxpy chained GPU array example.")
    parser.add_argument("--target", default="auto", help="auto, vulkan, directx, or metal")
    parser.add_argument("--provider", default=None, help="Backend provider override (e.g. warp, swiftshader, hardware)")
    parser.add_argument("--validation", action="store_true", help="Enable GPU API validation layers")
    parser.add_argument("--length", type=int, default=137, help="Input array length (default: 137)")
    args = parser.parse_args(argv)
    return run_example(
        target=args.target,
        provider=args.provider,
        validation=args.validation,
        length=args.length,
    )


if __name__ == "__main__":
    raise SystemExit(main())
