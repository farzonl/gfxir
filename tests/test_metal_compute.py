"""Tests for persistent macOS Metal compute resources."""

from __future__ import annotations

import ctypes
from dataclasses import replace
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    ParameterBlock,
    PersistentComputeError,
    Target,
    Type,
    compile_pipeline,
    compute,
    global_invocation_id,
    i32,
    load_compute_program,
    open_compute_context,
)
from mlir_gpu.metal_compute import MetalComputeContext
from mlir_gpu.viewer import ComputeProgram, Shader
from test_support import TOOLS

MetalNoopParams = ParameterBlock("mtl_noop_p", (("n", Type.I32),))
MetalVecParams = ParameterBlock("mtl_vec_p", (("n", Type.I32), ("k", Type.F32)))


class _FakeMetalBridge:
    def __init__(self, *, fail_create: str | None = None) -> None:
        self.fail_create = fail_create
        self.next_id = 1
        self.buffers: dict[int, bytearray] = {}
        self.pipelines: set[int] = set()
        self.dispatches: list[tuple[int, int, tuple[int, int, int]]] = []
        self.sync_calls = 0
        self.destroyed_contexts = 0

    def _alloc_id(self) -> int:
        val = self.next_id
        self.next_id += 1
        return val

    def gfx_mtl_create_context(self, err_buf, err_cap):
        if self.fail_create:
            err_buf.value = self.fail_create.encode("utf-8")
            return 0
        return self._alloc_id()

    def gfx_mtl_destroy_context(self, ctx):
        self.destroyed_contexts += 1

    def gfx_mtl_context_device_name(self, ctx):
        return b"FakeAppleMetalGPU"

    def gfx_mtl_create_buffer(self, ctx, size_bytes, initial_ptr, initial_size, err_buf, err_cap):
        handle = self._alloc_id()
        buf = bytearray(size_bytes)
        if initial_ptr and initial_size > 0:
            raw = ctypes.string_at(initial_ptr, initial_size)
            buf[:initial_size] = raw
        self.buffers[handle] = buf
        return handle

    def gfx_mtl_destroy_buffer(self, handle):
        self.buffers.pop(handle, None)

    def gfx_mtl_upload_buffer(self, ctx, handle, offset, data_ptr, data_size, err_buf, err_cap):
        raw = ctypes.string_at(data_ptr, data_size)
        self.buffers[handle][offset : offset + data_size] = raw
        return 0

    def gfx_mtl_download_buffer(self, ctx, handle, offset, out_ptr, read_size, err_buf, err_cap):
        raw = bytes(self.buffers[handle][offset : offset + read_size])
        ctypes.memmove(out_ptr, raw, read_size)
        return 0

    def gfx_mtl_buffer_contents(self, ctx, handle, out_size_ptr, err_buf, err_cap):
        buf = self.buffers[handle]
        if not hasattr(self, "_backing"):
            self._backing = {}
        c_arr = (ctypes.c_uint8 * len(buf)).from_buffer(buf)
        self._backing[handle] = c_arr
        ctypes.cast(out_size_ptr, ctypes.POINTER(ctypes.c_uint64))[0] = len(buf)
        return ctypes.addressof(c_arr)

    def gfx_mtl_create_pipeline(self, ctx, path, entry, wg_x, wg_y, wg_z, use_root_descriptors, err_buf, err_cap):
        handle = self._alloc_id()
        self.pipelines.add(handle)
        return handle

    def gfx_mtl_destroy_pipeline(self, handle):
        self.pipelines.discard(handle)
        self.destroyed_pipelines = getattr(self, "destroyed_pipelines", 0) + 1

    def gfx_mtl_dispatch(self, ctx, pipe, bindings, num_bindings, rc_ptr, rc_size, gx, gy, gz, err_buf, err_cap):
        self.dispatches.append((pipe, num_bindings, (gx, gy, gz)))
        return 0

    def gfx_mtl_begin_batch(self, ctx, err_buf, err_cap):
        self.begin_batch_calls = getattr(self, "begin_batch_calls", 0) + 1
        return 0

    def gfx_mtl_end_batch(self, ctx, err_buf, err_cap):
        self.end_batch_calls = getattr(self, "end_batch_calls", 0) + 1
        return 0

    def gfx_mtl_synchronize(self, ctx, err_buf, err_cap):
        self.sync_calls += 1
        return 0


class TestMetalComputeContract(unittest.TestCase):
    def test_non_macos_rejected_without_cpu_fallback(self) -> None:
        with mock.patch("mlir_gpu.metal_compute.sys.platform", "linux"):
            with self.assertRaises(PersistentComputeError) as ctx:
                with open_compute_context("metal"):
                    pass
            self.assertIn("macOS", str(ctx.exception))

    def test_missing_device_or_tier2_surfaces_error(self) -> None:
        fake = _FakeMetalBridge(fail_create="Converted DXIL Metal compute execution requires Argument Buffers Tier 2")
        with (
            mock.patch("mlir_gpu.metal_compute.sys.platform", "darwin"),
            mock.patch("mlir_gpu.metal_compute._load_bridge", return_value=fake),
        ):
            with self.assertRaises(PersistentComputeError) as ctx:
                with open_compute_context("metal"):
                    pass
            self.assertIn("Argument Buffers Tier 2", str(ctx.exception))

    def test_buffer_pipeline_and_lifecycle_validation(self) -> None:
        fake = _FakeMetalBridge()

        @compute(workgroup_size=(64, 1, 1))
        def noop_kernel(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: MetalNoopParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx]

        with tempfile.TemporaryDirectory() as td:
            # Build a minimal valid Metal compute manifest without invoking metal-shaderconverter.
            metallib_path = Path(td) / "noop.metallib"
            metallib_path.write_bytes(b"MTLB")
            manifest = compile_pipeline(
                ComputePipeline(noop_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "noop_vk",
                TOOLS,
            )
            vk_prog = load_compute_program(manifest)
            mtl_prog = replace(
                vk_prog,
                target="metal",
                compute=replace(vk_prog.compute, code=b"MTLB"),
            )

            with (
                mock.patch("mlir_gpu.metal_compute.sys.platform", "darwin"),
                mock.patch("mlir_gpu.metal_compute._load_bridge", return_value=fake),
            ):
                with open_compute_context("metal") as ctx1, open_compute_context("metal") as ctx2:
                    self.assertEqual(ctx1.target, "metal")
                    self.assertEqual(ctx1.device_name, "FakeAppleMetalGPU")

                    with self.assertRaises(PersistentComputeError):
                        ctx1.create_buffer(0)
                    with self.assertRaises(PersistentComputeError):
                        ctx1.create_buffer(4, b"12345678")

                    b1 = ctx1.create_buffer(16, b"abcd")
                    self.assertEqual(b1.download(4), b"abcd")
                    b1.upload(b"efgh", offset=4)
                    self.assertEqual(b1.download(8), b"abcdefgh")

                    with self.assertRaises(PersistentComputeError):
                        b1.upload(b"x", offset=16)
                    with self.assertRaises(PersistentComputeError):
                        b1.download(20)

                    pipe = ctx1.compile_pipeline(mtl_prog)
                    b2 = ctx1.create_buffer(16)
                    other = ctx2.create_buffer(16)

                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": b1, "dst": other}, parameters={"n": 4}, grid=(1, 1, 1))
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": b1}, parameters={"n": 4}, grid=(1, 1, 1))
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": b1, "dst": b2}, parameters={"n": 4}, grid=(0, 1, 1))
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": b1, "dst": b2}, parameters={"n": 4}, grid=(0x1_0000_0000, 1, 1))

                    pipe.dispatch({"src": b1, "dst": b2}, parameters={"n": 4}, grid=(1, 1, 1))
                    pipe.dispatch({"src": b1, "dst": b2}, parameters={"n": 4}, grid=(65536, 1, 1))
                    self.assertEqual(len(fake.dispatches), 2)

                    pipe.close()
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": b1, "dst": b2}, parameters={"n": 4}, grid=(1, 1, 1))

                    b1.close()
                    with self.assertRaises(PersistentComputeError):
                        b1.download()

                self.assertTrue(ctx1.closed)
                with self.assertRaises(PersistentComputeError):
                    ctx1.create_buffer(16)

    def test_metal_buffer_pool_reuse_in_flight_safety_and_release(self) -> None:
        fake = _FakeMetalBridge()

        @compute(workgroup_size=(64, 1, 1))
        def copy_kernel(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: MetalNoopParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx]

        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(
                ComputePipeline(copy_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "copy_pool_vk",
                TOOLS,
            )
            vk_prog = load_compute_program(manifest)
            mtl_prog = replace(
                vk_prog,
                target="metal",
                compute=replace(vk_prog.compute, code=b"MTLB"),
            )
            with (
                mock.patch("mlir_gpu.metal_compute.sys.platform", "darwin"),
                mock.patch("mlir_gpu.metal_compute._load_bridge", return_value=fake),
            ):
                with open_compute_context("metal") as ctx:
                    pipe = ctx.compile_pipeline(mtl_prog)
                    b1 = ctx.create_buffer(16, b"abcd" * 4)
                    b1.close()
                    b2 = ctx.create_buffer(16, zero_init=False)
                    self.assertEqual(ctx.stats["allocations"], 1)
                    self.assertEqual(ctx.stats["pool_hits"], 1)

                    b_out = ctx.create_buffer(16, zero_init=False)
                    pipe.dispatch({"src": b2, "dst": b_out}, parameters={"n": 4}, grid=(1, 1, 1))
                    b2.close()
                    # In-flight buffer must not be reused before synchronize().
                    b3 = ctx.create_buffer(16, zero_init=False)
                    self.assertEqual(ctx.stats["allocations"], 3)
                    ctx.synchronize()
                    self.assertEqual(ctx.stats["pooled_buffers"], 1)
                    b_out.close()
                    b3.close()
                    self.assertEqual(ctx.stats["pooled_buffers"], 3)
                    freed = ctx.release_pool()
                    self.assertEqual(freed, 48)
                    self.assertEqual(ctx.stats["pooled_buffers"], 0)
                    self.assertEqual(len(fake.buffers), 0)

    def test_deferred_cleanup_and_stats_without_intermediate_waits(self) -> None:
        fake = _FakeMetalBridge()

        @compute(workgroup_size=(64, 1, 1))
        def copy_kernel(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: MetalNoopParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx]

        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(
                ComputePipeline(copy_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "copy_vk",
                TOOLS,
            )
            vk_prog = load_compute_program(manifest)
            mtl_prog = replace(
                vk_prog,
                target="metal",
                compute=replace(vk_prog.compute, code=b"MTLB"),
            )
            with (
                mock.patch("mlir_gpu.metal_compute.sys.platform", "darwin"),
                mock.patch("mlir_gpu.metal_compute._load_bridge", return_value=fake),
            ):
                with open_compute_context("metal") as ctx:
                    b1 = ctx.create_buffer(16, b"abcd" * 4)
                    b_tmp = ctx.create_buffer(16)
                    b2 = ctx.create_buffer(16)
                    pipe = ctx.compile_pipeline(mtl_prog)

                    ctx.reset_stats()
                    sync_before = fake.sync_calls
                    with ctx.batch():
                        pipe.dispatch({"src": b1, "dst": b_tmp}, parameters={"n": 4}, grid=(1, 1, 1))
                        pipe.dispatch({"src": b_tmp, "dst": b2}, parameters={"n": 4}, grid=(1, 1, 1))
                        b_tmp.close()
                        pipe.close()
                        self.assertEqual(fake.sync_calls, sync_before)
                        self.assertEqual(ctx.stats["waits"], 0)
                        self.assertEqual(ctx.stats["deferred_cleanups"], 2)
                        self.assertGreater(len(fake.buffers), 0)

                    _ = b2.download()
                    self.assertEqual(ctx.stats["dispatches"], 2)
                    self.assertEqual(ctx.stats["submissions"], 1)
                    self.assertEqual(getattr(fake, "begin_batch_calls", 0), 1)
                    self.assertEqual(getattr(fake, "end_batch_calls", 0), 1)
                    self.assertEqual(ctx.stats["waits"], 1)
                    self.assertEqual(fake.sync_calls, sync_before + 1)
                    self.assertEqual(getattr(fake, "destroyed_pipelines", 0), 1)
                    with b2.map_view() as view:
                        self.assertEqual(len(view), 16)
                        view[:4] = b"WXYZ"
                    self.assertEqual(b2.download()[:4], b"WXYZ")

    def test_dropped_buffers_and_pipelines_release_native_allocations_and_keep_tracking_bounded(self) -> None:
        import gc

        fake = _FakeMetalBridge()
        prog = ComputeProgram(
            compute=Shader(entry_point="main", code=b"metallib"),
            target="metal",
            workgroup_size=(64, 1, 1),
            buffers=[],
        )
        with mock.patch("sys.platform", "darwin"), mock.patch(
            "mlir_gpu.metal_compute._load_bridge", return_value=fake
        ):
            ctx = MetalComputeContext(pool_max_bytes=0)
            for _ in range(50):
                b = ctx.create_buffer(16)
                b.close()
            self.assertEqual(len(ctx._buffers), 0)
            self.assertEqual(len(fake.buffers), 0)
            self.assertEqual(ctx.stats["live_buffers"], 0)

            dropped_buf = ctx.create_buffer(32)
            dropped_pipe = ctx.compile_pipeline(prog)
            self.assertEqual(len(fake.buffers), 1)
            self.assertEqual(len(fake.pipelines), 1)
            del dropped_buf, dropped_pipe
            gc.collect()

            self.assertEqual(len(ctx._buffers), 0)
            self.assertEqual(len(ctx._pipelines), 0)
            self.assertEqual(len(fake.buffers), 0)
            self.assertEqual(len(fake.pipelines), 0)

            # Dropping a buffer while work is pending defers cleanup and still frees on close().
            pending_buf = ctx.create_buffer(16)
            ctx._pending_work = True
            del pending_buf
            gc.collect()
            self.assertEqual(len(fake.buffers), 1)
            ctx.close()
            self.assertEqual(len(fake.buffers), 0)
            self.assertEqual(fake.destroyed_contexts, 1)

    def test_close_attempts_remaining_resources_on_sync_and_destroy_failures(self) -> None:
        fake = _FakeMetalBridge()
        fail_first_destroy = {"done": False}
        orig_destroy_buf = fake.gfx_mtl_destroy_buffer

        def failing_destroy_buffer(handle: int) -> None:
            if not fail_first_destroy["done"]:
                fail_first_destroy["done"] = True
                raise RuntimeError("injected destroy_buffer failure")
            orig_destroy_buf(handle)

        def failing_sync(ctx, err_buf, err_cap) -> int:
            err_buf.value = b"injected sync failure"
            return 1

        fake.gfx_mtl_destroy_buffer = failing_destroy_buffer  # type: ignore[method-assign]
        fake.gfx_mtl_synchronize = failing_sync  # type: ignore[method-assign]

        prog = ComputeProgram(
            compute=Shader(entry_point="main", code=b"metallib"),
            target="metal",
            workgroup_size=(64, 1, 1),
            buffers=[],
        )
        with mock.patch("sys.platform", "darwin"), mock.patch(
            "mlir_gpu.metal_compute._load_bridge", return_value=fake
        ):
            ctx = MetalComputeContext(pool_max_bytes=0)
            _b1 = ctx.create_buffer(16)
            _b2 = ctx.create_buffer(16)
            _pipe = ctx.compile_pipeline(prog)
            ctx._pending_work = True

            with self.assertRaises(PersistentComputeError) as err_ctx:
                ctx.close()
            self.assertIn("injected sync failure", str(err_ctx.exception))
            self.assertTrue(ctx.closed)
            self.assertEqual(len(ctx._buffers), 0)
            self.assertEqual(len(ctx._pipelines), 0)
            self.assertEqual(len(fake.buffers), 1)  # Only the one whose destroy raised remains
            self.assertEqual(len(fake.pipelines), 0)
            self.assertEqual(fake.destroyed_contexts, 1)

            # Second close is idempotent.
            ctx.close()
            self.assertEqual(fake.destroyed_contexts, 1)


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live persistent Metal compute tests",
)
class TestPersistentMetalComputeExecution(unittest.TestCase):
    def test_two_dependent_dispatches_without_intermediate_host_readback(self) -> None:
        length = 137

        @compute(workgroup_size=(64, 1, 1))
        def scale_add(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: MetalVecParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx] * params.k + 1.0

        @compute(workgroup_size=(64, 1, 1))
        def square_shift(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: MetalVecParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                v = src[idx]
                dst[idx] = v * v - params.k

        host_in = [float(i) * 0.25 - 4.0 for i in range(length)]
        expected = [((x * 2.0 + 1.0) ** 2) - 0.5 for x in host_in]
        in_bytes = struct.pack(f"<{length}f", *host_in)
        grid = ((length + 63) // 64, 1, 1)

        with tempfile.TemporaryDirectory() as td:
            m1 = compile_pipeline(
                ComputePipeline(scale_add, Dispatch(grid=(1, 1, 1))),
                Target.METAL,
                Path(td) / "k1",
                TOOLS,
            )
            m2 = compile_pipeline(
                ComputePipeline(square_shift, Dispatch(grid=(1, 1, 1))),
                Target.METAL,
                Path(td) / "k2",
                TOOLS,
            )
            with open_compute_context("metal") as ctx:
                buf_in = ctx.create_buffer(len(in_bytes), initial_data=in_bytes)
                buf_mid = ctx.create_buffer(len(in_bytes))
                buf_out = ctx.create_buffer(len(in_bytes))
                pipe1 = ctx.compile_pipeline(m1)
                pipe2 = ctx.compile_pipeline(m2)

                pipe1.dispatch({"src": buf_in, "dst": buf_mid}, parameters={"n": length, "k": 2.0}, grid=grid)
                pipe2.dispatch({"src": buf_mid, "dst": buf_out}, parameters={"n": length, "k": 0.5}, grid=grid)

                out_vals = struct.unpack(f"<{length}f", buf_out.download())
                for got, exp in zip(out_vals, expected):
                    self.assertAlmostEqual(got, exp, delta=1e-3)

    def test_target_profile_apple9_o3_and_root_descriptors(self) -> None:
        import json
        from mlir_gpu import TargetProfile

        with self.assertRaises(ValueError):
            TargetProfile(metal_gpu_family="InvalidFamily")
        with self.assertRaises(ValueError):
            TargetProfile(dxil_opt_level=4)

        @compute(workgroup_size=(64, 1, 1))
        def add_one(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
        ):
            idx = i32(global_invocation_id().x)
            if idx < 16:
                dst[idx] = src[idx] + 1.0

        profile = TargetProfile(metal_gpu_family="Apple9", dxil_opt_level=3)
        with tempfile.TemporaryDirectory() as td:
            m = compile_pipeline(
                ComputePipeline(add_one, Dispatch(grid=(1, 1, 1))),
                Target.METAL,
                Path(td) / "apple9_o3",
                TOOLS,
                target_profile=profile,
            )
            manifest = json.loads(Path(m).read_text())
            self.assertTrue(manifest.get("metal_root_descriptors"))
            self.assertEqual(manifest["target_profile"]["metal_gpu_family"], "Apple9")
            self.assertEqual(manifest["target_profile"]["dxil_opt_level"], 3)

            host_in = [float(i) for i in range(16)]
            in_bytes = struct.pack("<16f", *host_in)
            with open_compute_context("metal") as ctx:
                buf_in = ctx.create_buffer(len(in_bytes), initial_data=in_bytes)
                buf_out = ctx.create_buffer(len(in_bytes))
                pipe = ctx.compile_pipeline(m)
                pipe.dispatch({"src": buf_in, "dst": buf_out}, grid=(1, 1, 1))
                out_vals = struct.unpack("<16f", buf_out.download())
                for got, exp in zip(out_vals, [v + 1.0 for v in host_in]):
                    self.assertAlmostEqual(got, exp, delta=1e-5)


if __name__ == "__main__":
    unittest.main()
