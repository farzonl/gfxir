"""Tests for the persistent compute runtime contract and Vulkan implementation."""

from __future__ import annotations

from contextlib import nullcontext
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
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
    open_compute_context,
)
from mlir_gpu import compute_runtime
from test_support import TOOLS

TestNoopParams = ParameterBlock("p", (("n", Type.I32),))
TestVecParams = ParameterBlock("vec_params", (("n", Type.I32), ("k", Type.F32)))


class _FakeVk:
    class _FakeFFI:
        @staticmethod
        def from_buffer(data):
            return data

    ffi = _FakeFFI()
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT = 1
    VK_SHARING_MODE_EXCLUSIVE = 0
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT = 1
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT = 2
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER = 7
    VK_COMMAND_BUFFER_LEVEL_PRIMARY = 0
    VK_PIPELINE_BIND_POINT_COMPUTE = 1
    VK_ACCESS_HOST_WRITE_BIT = 1
    VK_ACCESS_HOST_READ_BIT = 2
    VK_ACCESS_SHADER_READ_BIT = 4
    VK_ACCESS_SHADER_WRITE_BIT = 8
    VK_PIPELINE_STAGE_HOST_BIT = 1
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 2
    VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT = 1
    VK_TRUE = 1

    def __init__(self) -> None:
        self.mem_store: dict[int, bytearray] = {}
        self._next_id = 1
        self.wait_fences_calls = 0
        self.queue_submit_calls = 0
        self.dispatch_calls = 0
        self.destroyed_buffers: list[int] = []
        self.destroyed_descriptor_pools: list[int] = []

    def _alloc_id(self) -> int:
        val = self._next_id
        self._next_id += 1
        return val

    def VkFenceCreateInfo(self):
        return object()

    def VkCommandBufferAllocateInfo(self, **kwargs):
        return kwargs

    def VkBufferCreateInfo(self, **kwargs):
        return kwargs

    def VkMemoryAllocateInfo(self, **kwargs):
        return kwargs

    def VkDescriptorPoolSize(self, **kwargs):
        return kwargs

    def VkDescriptorPoolCreateInfo(self, **kwargs):
        return kwargs

    def VkDescriptorSetAllocateInfo(self, **kwargs):
        return kwargs

    def VkDescriptorBufferInfo(self, **kwargs):
        return kwargs

    def VkWriteDescriptorSet(self, **kwargs):
        return kwargs

    def VkCommandBufferBeginInfo(self):
        return object()

    def VkMemoryBarrier(self, **kwargs):
        return kwargs

    def VkSubmitInfo(self, **kwargs):
        return kwargs

    def vkCreateFence(self, device, info, alloc):
        return self._alloc_id()

    def vkDestroyFence(self, device, fence, alloc):
        pass

    def vkAllocateCommandBuffers(self, device, info):
        return [self._alloc_id()]

    def vkFreeCommandBuffers(self, device, pool, count, cmds):
        pass

    def vkDeviceWaitIdle(self, device):
        pass

    def vkCreateBuffer(self, device, info, alloc):
        return self._alloc_id()

    def vkDestroyBuffer(self, device, buf, alloc):
        self.destroyed_buffers.append(buf)

    def vkGetBufferMemoryRequirements(self, device, buf):
        class Reqs:
            size = 4096
            memoryTypeBits = 1

        return Reqs()

    def vkAllocateMemory(self, device, info, alloc):
        mem_id = self._alloc_id()
        self.mem_store[mem_id] = bytearray(info["allocationSize"])
        return mem_id

    def vkFreeMemory(self, device, mem, alloc):
        self.mem_store.pop(mem, None)

    def vkBindBufferMemory(self, device, buf, mem, offset):
        pass

    def vkMapMemory(self, device, mem, offset, size, flags):
        return memoryview(self.mem_store[mem])[offset : offset + size]

    def vkUnmapMemory(self, device, mem):
        pass

    def vkCreateDescriptorPool(self, device, info, alloc):
        return self._alloc_id()

    def vkDestroyDescriptorPool(self, device, pool, alloc):
        pass

    def vkAllocateDescriptorSets(self, device, info):
        return [self._alloc_id() for _ in range(info["descriptorSetCount"])]

    def vkUpdateDescriptorSets(self, device, count, writes, copy_count, copies):
        pass

    def vkResetCommandBuffer(self, cmd, flags):
        pass

    def vkBeginCommandBuffer(self, cmd, info):
        pass

    def vkCmdPipelineBarrier(self, *args):
        pass

    def vkCmdBindPipeline(self, *args):
        pass

    def vkCmdBindDescriptorSets(self, *args):
        pass

    def vkCmdPushConstants(self, *args):
        pass

    def vkCmdDispatch(self, *args):
        self.dispatch_calls += 1

    def vkEndCommandBuffer(self, cmd):
        pass

    def vkResetFences(self, device, count, fences):
        pass

    def vkQueueSubmit(self, queue, count, submits, fence):
        self.queue_submit_calls += 1

    def vkFreeDescriptorSets(self, device, pool, count, sets):
        pass

    def vkDestroyDescriptorPool(self, device, pool, alloc):
        self.destroyed_descriptor_pools.append(pool)

    def vkWaitForFences(self, device, count, fences, wait_all, timeout):
        self.wait_fences_calls += 1


class _FakeVulkan:
    def __init__(self, resources, validation=False, compute_only=True, **kwargs) -> None:
        self.vk = _FakeVk()
        self.device = 1
        self.queue = 2
        self.pool = 3
        self.name = "FakeVulkanDevice"
        self.max_compute_workgroup_count = (65535, 65535, 65535)

    def memory_type(self, bits, flags):
        return 0

    def compute_pipeline(self, resources, program):
        return 10, 11, [12]

    def parameter_stages(self, parameters):
        return 1

    def check_validation(self):
        pass


class TestPersistentComputeValidation(unittest.TestCase):
    def test_unsupported_target_rejected_without_cpu_fallback(self) -> None:
        with self.assertRaises(PersistentComputeError):
            with open_compute_context("cuda"):
                pass

    def test_buffer_and_lifecycle_validation_with_mock_vulkan(self) -> None:
        with (
            mock.patch.object(compute_runtime, "_Vulkan", _FakeVulkan),
            mock.patch.object(compute_runtime, "_api_errors", nullcontext),
        ):
            with open_compute_context("vulkan") as ctx1, open_compute_context("vulkan") as ctx2:
                self.assertEqual(ctx1.target, "vulkan")
                self.assertEqual(ctx1.device_name, "FakeVulkanDevice")

                with self.assertRaises(PersistentComputeError):
                    ctx1.create_buffer(0)
                with self.assertRaises(PersistentComputeError):
                    ctx1.create_buffer(-8)
                with self.assertRaises(PersistentComputeError):
                    ctx1.create_buffer(True)  # type: ignore[arg-type]
                with self.assertRaises(PersistentComputeError):
                    ctx1.create_buffer(4, b"12345678")

                buf1 = ctx1.create_buffer(16, b"abcd")
                self.assertEqual(buf1.download(4), b"abcd")
                buf1.upload(b"efgh", offset=4)
                self.assertEqual(buf1.download(8), b"abcdefgh")

                with self.assertRaises(PersistentComputeError):
                    buf1.upload(b"x", offset=16)
                with self.assertRaises(PersistentComputeError):
                    buf1.download(20)
                with self.assertRaises(PersistentComputeError):
                    buf1.upload("not-bytes")  # type: ignore[arg-type]

                @compute(workgroup_size=(64, 1, 1))
                def noop_kernel(
                    src: Buffer[Type.F32, "read_only"],
                    dst: Buffer[Type.F32, "write_only"],
                    params: TestNoopParams,
                ):
                    idx = i32(global_invocation_id().x)
                    if idx < params.n:
                        dst[idx] = src[idx]

                with tempfile.TemporaryDirectory() as td:
                    manifest = compile_pipeline(
                        ComputePipeline(noop_kernel, Dispatch(grid=(1, 1, 1))),
                        Target.VULKAN,
                        Path(td) / "noop",
                        TOOLS,
                    )
                    pipe = ctx1.compile_pipeline(manifest)
                    buf2 = ctx1.create_buffer(16)
                    other_buf = ctx2.create_buffer(16)

                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": buf1, "dst": other_buf}, parameters={"n": 4}, grid=(1, 1, 1))
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": buf1}, parameters={"n": 4}, grid=(1, 1, 1))
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": buf1, "dst": buf2}, parameters={"n": 4}, grid=(0, 1, 1))

                    pipe.dispatch({"src": buf1, "dst": buf2}, parameters={"n": 4}, grid=(1, 1, 1))
                    pipe.close()
                    with self.assertRaises(PersistentComputeError):
                        pipe.dispatch({"src": buf1, "dst": buf2}, parameters={"n": 4}, grid=(1, 1, 1))

                buf1.close()
                self.assertTrue(buf1.closed)
                with self.assertRaises(PersistentComputeError):
                    buf1.download()

            self.assertTrue(ctx1.closed)
            with self.assertRaises(PersistentComputeError):
                ctx1.create_buffer(16)

    def test_batched_dispatches_defer_waits_and_cleanup(self) -> None:
        @compute(workgroup_size=(64, 1, 1))
        def noop_kernel(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: TestNoopParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx]

        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(
                ComputePipeline(noop_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "noop_batch",
                TOOLS,
            )
            with (
                mock.patch.object(compute_runtime, "_Vulkan", _FakeVulkan),
                mock.patch.object(compute_runtime, "_api_errors", nullcontext),
            ):
                with open_compute_context("vulkan") as ctx:
                    vk = ctx._gpu.vk
                    buf_a = ctx.create_buffer(16, initial_data=b"\x01" * 16)
                    buf_b = ctx.create_buffer(16)
                    buf_tmp = ctx.create_buffer(16)
                    pipe = ctx.compile_pipeline(manifest)

                    ctx.reset_stats()
                    with ctx.batch():
                        pipe.dispatch({"src": buf_a, "dst": buf_tmp}, parameters={"n": 4}, grid=(1, 1, 1))
                        pipe.dispatch({"src": buf_tmp, "dst": buf_b}, parameters={"n": 4}, grid=(1, 1, 1))
                        pipe.dispatch({"src": buf_b, "dst": buf_a}, parameters={"n": 4}, grid=(1, 1, 1))
                        # Closing an intermediate buffer and pipeline mid-batch must defer destruction
                        # without forcing a host wait.
                        buf_tmp.close()
                        pipe.close()
                        self.assertEqual(ctx.stats["waits"], 0)
                        self.assertEqual(vk.wait_fences_calls, 0)
                        self.assertEqual(len(vk.destroyed_buffers), 0)
                        self.assertEqual(ctx.stats["deferred_cleanups"], 2)

                    # Batch exit submits once without waiting.
                    self.assertEqual(ctx.stats["dispatches"], 3)
                    self.assertEqual(ctx.stats["submissions"], 1)
                    self.assertEqual(ctx.stats["waits"], 0)
                    self.assertEqual(vk.queue_submit_calls, 1)
                    self.assertEqual(vk.wait_fences_calls, 0)
                    self.assertEqual(vk.dispatch_calls, 3)

                    # Readback establishes completion and flushes deferred cleanups.
                    _ = buf_a.download()
                    self.assertEqual(ctx.stats["waits"], 1)
                    self.assertEqual(vk.wait_fences_calls, 1)
                    self.assertEqual(len(vk.destroyed_buffers), 1)
                    self.assertEqual(len(vk.destroyed_descriptor_pools), 1)

    def test_repeated_batches_and_partial_failure_recovery(self) -> None:
        @compute(workgroup_size=(64, 1, 1))
        def noop_kernel(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: TestNoopParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx]

        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(
                ComputePipeline(noop_kernel, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "noop_repeat",
                TOOLS,
            )
            with (
                mock.patch.object(compute_runtime, "_Vulkan", _FakeVulkan),
                mock.patch.object(compute_runtime, "_api_errors", nullcontext),
            ):
                with open_compute_context("vulkan") as ctx:
                    buf_a = ctx.create_buffer(16, initial_data=b"\x01" * 16)
                    buf_b = ctx.create_buffer(16)
                    pipe = ctx.compile_pipeline(manifest)

                    # Repeated execution across multiple batches (exceeding descriptor cache size).
                    for i in range(40):
                        with ctx.batch():
                            src, dst = (buf_a, buf_b) if i % 2 == 0 else (buf_b, buf_a)
                            pipe.dispatch({"src": src, "dst": dst}, parameters={"n": 4}, grid=(1, 1, 1))
                    ctx.synchronize()
                    self.assertEqual(ctx.stats["dispatches"], 40)

                    # Partial failure mid-batch leaves context usable for subsequent dispatches.
                    with self.assertRaises(PersistentComputeError):
                        with ctx.batch():
                            pipe.dispatch({"src": buf_a, "dst": buf_b}, parameters={"n": 4}, grid=(1, 1, 1))
                            pipe.dispatch({"src": buf_a}, parameters={"n": 4}, grid=(1, 1, 1))

                    ctx.synchronize()
                    ctx.reset_stats()
                    with ctx.batch():
                        pipe.dispatch({"src": buf_a, "dst": buf_b}, parameters={"n": 4}, grid=(1, 1, 1))
                    _ = buf_b.download()
                    self.assertEqual(ctx.stats["dispatches"], 1)
                    self.assertEqual(ctx.stats["submissions"], 1)
                    self.assertEqual(ctx.stats["waits"], 1)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live persistent Vulkan compute tests",
)
class TestPersistentVulkanComputeExecution(unittest.TestCase):
    def test_two_dependent_dispatches_without_intermediate_host_readback(self) -> None:
        length = 137  # Non-workgroup-aligned for workgroup_size=(64, 1, 1)

        @compute(workgroup_size=(64, 1, 1))
        def scale_add(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: TestVecParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                dst[idx] = src[idx] * params.k + 1.0

        @compute(workgroup_size=(64, 1, 1))
        def square_shift(
            src: Buffer[Type.F32, "read_only"],
            dst: Buffer[Type.F32, "write_only"],
            params: TestVecParams,
        ):
            idx = i32(global_invocation_id().x)
            if idx < params.n:
                v = src[idx]
                dst[idx] = v * v - params.k

        host_in = [float(i) * 0.25 - 4.0 for i in range(length)]
        expected = [((x * 2.0 + 1.0) ** 2) - 0.5 for x in host_in]
        in_bytes = struct.pack(f"<{length}f", *host_in)
        grid = ((length + 63) // 64, 1, 1)

        validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        with tempfile.TemporaryDirectory() as td:
            m1 = compile_pipeline(
                ComputePipeline(scale_add, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "k1",
                TOOLS,
            )
            m2 = compile_pipeline(
                ComputePipeline(square_shift, Dispatch(grid=(1, 1, 1))),
                Target.VULKAN,
                Path(td) / "k2",
                TOOLS,
            )
            with open_compute_context("vulkan", validation=validation) as ctx:
                buf_in = ctx.create_buffer(len(in_bytes), initial_data=in_bytes)
                buf_mid = ctx.create_buffer(len(in_bytes))
                buf_out = ctx.create_buffer(len(in_bytes))
                pipe1 = ctx.compile_pipeline(m1)
                pipe2 = ctx.compile_pipeline(m2)

                # Two dependent dispatches with zero intermediate host readback.
                pipe1.dispatch({"src": buf_in, "dst": buf_mid}, parameters={"n": length, "k": 2.0}, grid=grid)
                pipe2.dispatch({"src": buf_mid, "dst": buf_out}, parameters={"n": length, "k": 0.5}, grid=grid)

                out_vals = struct.unpack(f"<{length}f", buf_out.download())
                for got, exp in zip(out_vals, expected):
                    self.assertAlmostEqual(got, exp, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
