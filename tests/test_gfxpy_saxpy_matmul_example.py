"""Host tests for examples/gfxpy/saxpy_matmul_example.py."""

from __future__ import annotations

from pathlib import Path
import struct
import sys
import unittest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = ROOT / "examples"
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

import gfxpy as gp  # noqa: E402
import gfxpy.saxpy_matmul_example as saxpy_matmul_example  # noqa: E402


class _FakeBuffer:
    def __init__(self, size_bytes: int) -> None:
        self.data = bytearray(size_bytes)
        self.closed = False

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    def close(self) -> None:
        self.closed = True


class _FakeComputeBackend:
    def __init__(self, target: str = "metal") -> None:
        self.target = target
        self.device_name = "FakeGPU"
        self.closed = False
        self.dispatches: list[str] = []

    def allocate(self, size_bytes: int) -> _FakeBuffer:
        return _FakeBuffer(size_bytes)

    def upload(self, buf: _FakeBuffer, data: bytes) -> None:
        buf.data[: len(data)] = data

    def download(self, buf: _FakeBuffer) -> bytes:
        return bytes(buf.data)

    def binary_op(self, op: str, lhs: _FakeBuffer, rhs: _FakeBuffer, out: _FakeBuffer, length: int) -> None:
        self.dispatches.append(f"binary:{op}:{length}")
        lv = struct.unpack(f"<{length}f", lhs.data[: length * 4])
        rv = struct.unpack(f"<{length}f", rhs.data[: length * 4])
        res = [a + b if op == "add" else a * b for a, b in zip(lv, rv)]
        out.data[: length * 4] = struct.pack(f"<{length}f", *res)

    def scalar_op(self, op: str, lhs: _FakeBuffer, scalar: float, out: _FakeBuffer, length: int) -> None:
        self.dispatches.append(f"scalar:{op}:{length}")
        lv = struct.unpack(f"<{length}f", lhs.data[: length * 4])
        res = [a + scalar if op == "add" else a * scalar for a in lv]
        out.data[: length * 4] = struct.pack(f"<{length}f", *res)

    def matmul(
        self,
        a: _FakeBuffer,
        b: _FakeBuffer,
        out: _FakeBuffer,
        m: int,
        n: int,
        k: int,
    ) -> None:
        self.dispatches.append(f"matmul:{m}x{k}x{n}")
        av = struct.unpack(f"<{m * k}f", a.data[: m * k * 4])
        bv = struct.unpack(f"<{k * n}f", b.data[: k * n * 4])
        cv = [0.0] * (m * n)
        for r in range(m):
            for p in range(k):
                a_rp = av[r * k + p]
                for col in range(n):
                    cv[r * n + col] += a_rp * bv[p * n + col]
        out.data[: m * n * 4] = struct.pack(f"<{m * n}f", *cv)

    def close(self) -> None:
        self.closed = True


class TestSaxpyMatmulExample(unittest.TestCase):
    def test_defaults_match_4m_saxpy_and_4096_matmul(self) -> None:
        self.assertEqual(saxpy_matmul_example.SAXPY_N, 4_194_304)
        self.assertEqual(saxpy_matmul_example.MATMUL_N, 4096)

    def test_saxpy_and_matmul_end_to_end(self) -> None:
        backend = _FakeComputeBackend()
        with gp.device("metal", backend=backend) as dev:
            saxpy_matmul_example.run_saxpy(n=137)
            saxpy_matmul_example.run_matmul(n=19)
        dev.close()

        self.assertEqual(
            backend.dispatches,
            ["scalar:mul:137", "binary:add:137", "matmul:19x19x19"],
        )
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
