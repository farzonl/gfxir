"""Shared-memory allocation, workgroup barrier, and integer atomic compilation and GPU execution tests."""

from __future__ import annotations

import functools
import json
import operator
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from mlir_gpu import (
     HIGH_END_COMPUTE,
    Buffer,
    CompileError,
    ComputePipeline,
    Dispatch,
    GlobalInvocationId,
    LocalInvocationId,
    Target,
    Type,
    WorkgroupId,
    atomic_add,
    atomic_and,
    atomic_cas,
    atomic_compare_exchange,
    atomic_exchange,
    atomic_max,
    atomic_min,
    atomic_or,
    atomic_sub,
    atomic_umax,
    atomic_umin,
    atomic_xor,
    compile_pipeline,
    compute,
    dispatch_compute,
    full_barrier,
    i32,
    shared_array,
    storage_barrier,
    vec2,
    workgroup_barrier,
)
from test_support import TOOLS, available


def _build_reduction_kernel():
    @compute(workgroup_size=(64, 1, 1))
    def shared_tree_reduce(
        gid: GlobalInvocationId,
        lid: LocalInvocationId,
        inp: Buffer[Type.I32, "read_only"],
        out: Buffer[Type.I32, "read_write"],
    ):
        tile = shared_array(Type.I32, 64, name="reduce_tile")
        lx = i32(lid.x)
        tile[lx] = inp[gid.x]
        workgroup_barrier()

        stride = 32
        while stride > 0:
            if lx < stride:
                tile[lx] = tile[lx] + tile[lx + stride]
            workgroup_barrier()
            stride = stride // 2

        if lx == 0:
            atomic_add(out, 0, tile[0])

    return shared_tree_reduce


def _build_histogram_and_atomics_kernel():
    @compute(workgroup_size=(64, 1, 1))
    def histogram_and_atomics(
        gid: GlobalInvocationId,
        lid: LocalInvocationId,
        inp: Buffer[Type.I32, "read_only"],
        masks: Buffer[Type.I32, "read_only"],
        bins: Buffer[Type.I32, "read_write"],
        stats: Buffer[Type.I32, "read_write"],
    ):
        local_bins = shared_array(Type.I32, 16, name="local_bins")
        lx = i32(lid.x)
        gx = i32(gid.x)

        if lx < 16:
            local_bins[lx] = 0
        workgroup_barrier()

        val = inp[gx]
        mask = masks[gx]
        bin_idx = val & 15
        atomic_add(local_bins, bin_idx, 1)
        workgroup_barrier()

        if lx < 16:
            atomic_add(bins, lx, local_bins[lx])

        atomic_sub(stats, 0, val)
        atomic_min(stats, 1, val)
        atomic_max(stats, 2, val)
        atomic_or(stats, 3, mask)
        atomic_and(stats, 4, mask)
        atomic_xor(stats, 5, val)
        prev = atomic_compare_exchange(stats, 6, -1, gx)
        if prev == -1:
            atomic_add(stats, 7, 1)

    return histogram_and_atomics


def _build_shared_transpose_kernel():
    @compute(workgroup_size=(8, 8, 1))
    def shared_transpose_16x16(
        lid: LocalInvocationId,
        wid: WorkgroupId,
        inp: Buffer[Type.I32, "read_only"],
        out: Buffer[Type.I32, "write_only"],
    ):
        tile = shared_array(Type.I32, (8, 8), name="transpose_tile")
        lx = lid.x
        ly = lid.y
        gx = wid.x * 8 + lx
        gy = wid.y * 8 + ly
        tile[ly, lx] = inp[gy * 16 + gx]
        workgroup_barrier()

        out_x = wid.y * 8 + lx
        out_y = wid.x * 8 + ly
        out[out_y * 16 + out_x] = tile[lx, ly]

    return shared_transpose_16x16


@unittest.skipUnless(available("gfx-compile"), "Requires native gfx-compile toolchain")
class TestComputeSharedMemoryAndAtomicsAuthoring(unittest.TestCase):
    """Python authoring, reflection, binary validation, and negative verifier diagnostics."""

    def test_shared_memory_reflection_and_pipeline_manifest(self):
        @compute(workgroup_size=(16, 16, 1))
        def tiled_kernel(
            lid: LocalInvocationId,
            buf: Buffer[Type.U32, "read_write"],
        ):
            tile_a = shared_array(Type.F32, (16, 16), name="tile_a")
            tile_v = shared_array(vec2, 32, name="tile_v")
            lx = lid.x
            ly = lid.y
            tile_a[ly, lx] = 1.5
            if lx == 0 and ly == 0:
                tile_v[0] = vec2(2.0, 3.0)
            workgroup_barrier()
            storage_barrier()
            full_barrier()
            if lx == 0 and ly == 0:
                atomic_umin(buf, 0, 7)
                atomic_umax(buf, 1, 9)
                atomic_exchange(buf, 2, 42)
                atomic_cas(buf, 3, 0, 11)

        # tile_a = 16 * 16 * 4 = 1024 bytes; tile_v = 32 * 8 = 256 bytes -> 1280 bytes total.
        pipe = ComputePipeline(tiled_kernel, dispatch=Dispatch(grid=(1, 1, 1)))
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            vk_refl = td_path / "vk_refl.json"
            vk_manifest_path = compile_pipeline(
                pipe, Target.VULKAN, td_path / "vk_pipe", TOOLS, reflection_path=vk_refl
            )
            manifest = json.loads(vk_manifest_path.read_text(encoding="utf-8"))
            refl = json.loads(vk_refl.read_text(encoding="utf-8"))
            self.assertEqual(refl["shared_memory_bytes"], 1280)
            self.assertEqual(manifest["shared_memory_bytes"], 1280)

            if available("llc") and available("dxv"):
                dx_refl = td_path / "dx_refl.json"
                dx_manifest_path = compile_pipeline(
                    pipe, Target.DIRECTX, td_path / "dx_pipe", TOOLS, reflection_path=dx_refl
                )
                dx_manifest = json.loads(dx_manifest_path.read_text(encoding="utf-8"))
                self.assertEqual(dx_manifest["shared_memory_bytes"], 1280)

    def test_divergent_barrier_is_rejected_with_clear_diagnostic(self):
        @compute(workgroup_size=(64, 1, 1))
        def bad_divergent_barrier(
            lid: LocalInvocationId,
            out: Buffer[Type.I32, "read_write"],
        ):
            tile = shared_array(Type.I32, 64)
            if lid.x == 0:
                tile[0] = 1
                workgroup_barrier()
            out[lid.x] = tile[0]

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(CompileError) as ctx:
                compile_pipeline(bad_divergent_barrier, Target.VULKAN, Path(td) / "bad.spv", TOOLS)
            self.assertIn("barrier cannot be placed inside thread-divergent control flow", str(ctx.exception))

    def test_shared_memory_budget_enforced_by_default_and_configurable_by_profile(self):
        @compute(workgroup_size=(64, 1, 1))
        def large_shared_kernel(
            lid: LocalInvocationId,
            out: Buffer[Type.F32, "read_write"],
        ):
            big = shared_array(Type.F32, 12288, name="big_tile")
            big[lid.x] = 3.25
            workgroup_barrier()
            out[lid.x] = big[lid.x]

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            with self.assertRaises(CompileError) as ctx:
                compile_pipeline(large_shared_kernel, Target.VULKAN, td_path / "default_fail.spv", TOOLS)
            self.assertIn("exceeds target limit of 32768 bytes", str(ctx.exception))

            ok_out = td_path / "high_end_ok.spv"
            compile_pipeline(
                large_shared_kernel,
                Target.VULKAN,
                ok_out,
                TOOLS,
                target_profile=HIGH_END_COMPUTE,
            )
            self.assertTrue(ok_out.is_file())

    def test_atomic_on_read_only_buffer_is_rejected(self):
        @compute(workgroup_size=(64, 1, 1))
        def bad_ro_atomic(
            gid: GlobalInvocationId,
            inp: Buffer[Type.I32, "read_only"],
        ):
            atomic_add(inp, gid.x, 1)

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises((TypeError, CompileError)) as ctx:
                compile_pipeline(bad_ro_atomic, Target.VULKAN, Path(td) / "bad_ro.spv", TOOLS)
            self.assertIn("read_write", str(ctx.exception))


class _SharedAtomicsExecutionMixin:
    """Shared execution verification suite across Vulkan, DirectX 12 WARP, and Metal."""

    target: Target
    provider: str | None = None
    validation: bool = False

    def _dispatch(self, pipe: ComputePipeline, buffers: dict[str, bytes]):
        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(pipe, self.target, Path(td) / "pipe", TOOLS)
            kwargs = {
                "buffers": buffers,
                "validation": self.validation,
                "toolchain": TOOLS,
            }
            if self.provider is not None:
                kwargs["provider"] = self.provider
            return dispatch_compute(manifest, **kwargs)

    def test_workgroup_shared_tree_reduction_and_global_atomic_add(self):
        kernel = _build_reduction_kernel()
        pipe = ComputePipeline(kernel, dispatch=Dispatch(grid=(4, 1, 1)))
        values = [((i * 37) - 1200) for i in range(256)]
        expected_sum = sum(values)

        inp_bytes = struct.pack("<256i", *values)
        out_bytes = struct.pack("<1i", 100)
        res = self._dispatch(pipe, {"inp": inp_bytes, "out": out_bytes})
        (actual_sum,) = struct.unpack("<1i", res.buffers["out"])
        self.assertEqual(actual_sum, 100 + expected_sum)

    def test_privatized_shared_histogram_and_all_integer_atomics(self):
        kernel = _build_histogram_and_atomics_kernel()
        pipe = ComputePipeline(kernel, dispatch=Dispatch(grid=(4, 1, 1)))

        values = [((i * 73 + 19) % 97) - 40 for i in range(256)]
        masks = [(1 << (i % 12)) | (1 << ((i * 5) % 12)) | 0x0010_0000 for i in range(256)]

        expected_bins = [0] * 16
        for v in values:
            expected_bins[v & 15] += 1

        init_sub = 100_000
        expected_sub = init_sub - sum(values)
        expected_min = min(values)
        expected_max = max(values)
        expected_or = functools.reduce(operator.or_, masks, 0)
        expected_and = functools.reduce(operator.and_, masks, 0x7FFF_FFFF)
        expected_xor = functools.reduce(operator.xor, values, 0)

        inp_bytes = struct.pack("<256i", *values)
        mask_bytes = struct.pack("<256i", *masks)
        bins_bytes = struct.pack("<16i", *([0] * 16))
        stats_bytes = struct.pack(
            "<8i",
            init_sub,
            999_999,
            -999_999,
            0,
            0x7FFF_FFFF,
            0,
            -1,
            0,
        )

        res = self._dispatch(
            pipe,
            {
                "inp": inp_bytes,
                "masks": mask_bytes,
                "bins": bins_bytes,
                "stats": stats_bytes,
            },
        )
        actual_bins = list(struct.unpack("<16i", res.buffers["bins"]))
        actual_stats = list(struct.unpack("<8i", res.buffers["stats"]))

        self.assertEqual(actual_bins, expected_bins)
        self.assertEqual(actual_stats[0], expected_sub)
        self.assertEqual(actual_stats[1], expected_min)
        self.assertEqual(actual_stats[2], expected_max)
        self.assertEqual(actual_stats[3], expected_or)
        self.assertEqual(actual_stats[4], expected_and)
        self.assertEqual(actual_stats[5], expected_xor)
        self.assertGreaterEqual(actual_stats[6], 0)
        self.assertLess(actual_stats[6], 256)
        self.assertEqual(actual_stats[7], 1, "Exactly one thread must win atomic_compare_exchange")

    def test_2d_shared_array_matrix_transpose(self):
        kernel = _build_shared_transpose_kernel()
        pipe = ComputePipeline(kernel, dispatch=Dispatch(grid=(2, 2, 1)))
        inp = [r * 100 + c for r in range(16) for c in range(16)]
        expected = [c * 100 + r for r in range(16) for c in range(16)]

        res = self._dispatch(
            pipe,
            {
                "inp": struct.pack("<256i", *inp),
                "out": bytes(256 * 4),
            },
        )
        actual = list(struct.unpack("<256i", res.buffers["out"]))
        self.assertEqual(actual, expected)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan shared-memory and atomic tests",
)
class TestComputeAtomicsVulkanExecution(_SharedAtomicsExecutionMixin, unittest.TestCase):
    target = Target.VULKAN

    @classmethod
    def setUpClass(cls):
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX 12 WARP shared-memory and atomic tests",
)
class TestComputeAtomicsDirectXExecution(_SharedAtomicsExecutionMixin, unittest.TestCase):
    target = Target.DIRECTX
    provider = "warp"

    @classmethod
    def setUpClass(cls):
        if not (available("llc") and available("dxv") and available("mlir-gpu-directx-view")):
            raise unittest.SkipTest("DirectX 12 compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal shared-memory and atomic tests",
)
class TestComputeAtomicsMetalExecution(_SharedAtomicsExecutionMixin, unittest.TestCase):
    target = Target.METAL

    @classmethod
    def setUpClass(cls):
        required = ("llc", "dxv", "metal-shaderconverter", "mlir-gpu-metal-view")
        if not all(available(tool) for tool in required):
            raise unittest.SkipTest("Metal compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"


if __name__ == "__main__":
    unittest.main()
