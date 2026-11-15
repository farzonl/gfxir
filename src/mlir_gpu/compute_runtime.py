"""Persistent compute resource contract and Vulkan implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import weakref

from .parameters import pack_parameters
from .viewer import (
    ComputeProgram,
    ViewerError,
    _Vulkan,
    _api_errors,
    _validate_provider,
    load_compute_program,
)


class PersistentComputeError(ViewerError):
    """Raised on invalid persistent compute resource usage, binding, or lifecycle."""


_DEFAULT_POOL_MAX_BYTES = 64 * 1024 * 1024


def _resolve_pool_max_bytes(pool_max_bytes: int | None) -> int:
    if pool_max_bytes is not None:
        if type(pool_max_bytes) is not int or pool_max_bytes < 0:
            raise PersistentComputeError(
                f"pool_max_bytes must be a non-negative integer, got {pool_max_bytes!r}."
            )
        return pool_max_bytes
    env_val = os.environ.get("MLIR_GPU_POOL_MAX_BYTES")
    if env_val is not None:
        try:
            parsed = int(env_val)
            if parsed >= 0:
                return parsed
        except ValueError:
            pass
    return _DEFAULT_POOL_MAX_BYTES


class _BoundedBufferPool:
    """Per-context bounded buffer pool with FIFO eviction and memory high-water stats."""

    def __init__(
        self,
        *,
        max_bytes: int = _DEFAULT_POOL_MAX_BYTES,
        max_per_bucket: int = 32,
        max_entries: int = 64,
    ) -> None:
        self._max_bytes = max_bytes
        self._max_per_bucket = max_per_bucket
        self._max_entries = max_entries
        self._buckets: dict[int, list[object]] = {}
        self._fifo: deque[int] = deque()
        self._pooled_bytes = 0
        self._pooled_buffers = 0
        self._live_bytes = 0
        self._live_buffers = 0
        self._peak_live_bytes = 0
        self._peak_total_bytes = 0
        self._allocations = 0
        self._pool_hits = 0
        self._pool_misses = 0
        self._pool_returns = 0
        self._pool_evictions = 0

    def acquire(self, bucket_size: int) -> object | None:
        if self._max_bytes <= 0:
            self._pool_misses += 1
            return None
        bucket = self._buckets.get(bucket_size)
        if bucket:
            handle = bucket.pop()
            if not bucket:
                del self._buckets[bucket_size]
            self._pooled_bytes -= bucket_size
            self._pooled_buffers -= 1
            self._pool_hits += 1
            return handle
        self._pool_misses += 1
        return None

    def record_allocation(self) -> None:
        self._allocations += 1

    def on_live_alloc(self, size: int) -> None:
        self._live_buffers += 1
        self._live_bytes += size
        if self._live_bytes > self._peak_live_bytes:
            self._peak_live_bytes = self._live_bytes
        total = self._live_bytes + self._pooled_bytes
        if total > self._peak_total_bytes:
            self._peak_total_bytes = total

    def on_live_free(self, size: int) -> None:
        self._live_buffers = max(0, self._live_buffers - 1)
        self._live_bytes = max(0, self._live_bytes - size)

    def release(self, bucket_size: int, handle: object, destroy_fn: Callable[[object], None]) -> bool:
        if self._max_bytes <= 0 or bucket_size > self._max_bytes:
            destroy_fn(handle)
            self._pool_evictions += 1
            return False
        bucket = self._buckets.get(bucket_size)
        if bucket is not None and len(bucket) >= self._max_per_bucket:
            destroy_fn(handle)
            self._pool_evictions += 1
            return False
        while self._fifo and (
            self._pooled_bytes + bucket_size > self._max_bytes
            or self._pooled_buffers >= self._max_entries
        ):
            evict_size = self._fifo.popleft()
            evict_bucket = self._buckets.get(evict_size)
            if not evict_bucket:
                continue
            evict_handle = evict_bucket.pop(0)
            if not evict_bucket:
                del self._buckets[evict_size]
            self._pooled_bytes -= evict_size
            self._pooled_buffers -= 1
            self._pool_evictions += 1
            destroy_fn(evict_handle)

        if bucket is None:
            bucket = []
            self._buckets[bucket_size] = bucket
        bucket.append(handle)
        self._fifo.append(bucket_size)
        self._pooled_bytes += bucket_size
        self._pooled_buffers += 1
        self._pool_returns += 1
        total = self._live_bytes + self._pooled_bytes
        if total > self._peak_total_bytes:
            self._peak_total_bytes = total
        return True

    def clear(self, destroy_fn: Callable[[object], None]) -> int:
        freed_bytes = 0
        items = list(self._buckets.items())
        self._buckets.clear()
        self._fifo.clear()
        self._pooled_bytes = 0
        self._pooled_buffers = 0
        errors: list[BaseException] = []
        for bucket_size, bucket in items:
            for handle in bucket:
                try:
                    destroy_fn(handle)
                except BaseException as exc:
                    errors.append(exc)
                    self._buckets.setdefault(bucket_size, []).append(handle)
                    self._fifo.append(bucket_size)
                    self._pooled_bytes += bucket_size
                    self._pooled_buffers += 1
                else:
                    freed_bytes += bucket_size
        if errors:
            raise errors[0]
        return freed_bytes

    def reset_counters(self) -> None:
        self._allocations = 0
        self._pool_hits = 0
        self._pool_misses = 0
        self._pool_returns = 0
        self._pool_evictions = 0
        self._peak_live_bytes = self._live_bytes
        self._peak_total_bytes = self._live_bytes + self._pooled_bytes

    def stats(self) -> dict[str, int]:
        return {
            "allocations": self._allocations,
            "pool_hits": self._pool_hits,
            "pool_misses": self._pool_misses,
            "pool_returns": self._pool_returns,
            "pool_evictions": self._pool_evictions,
            "live_buffers": self._live_buffers,
            "live_bytes": self._live_bytes,
            "pooled_buffers": self._pooled_buffers,
            "pooled_bytes": self._pooled_bytes,
            "peak_live_bytes": self._peak_live_bytes,
            "peak_total_bytes": self._peak_total_bytes,
        }


def _validate_positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise PersistentComputeError(f"{name} must be a positive integer, got {value!r}.")
    return value


def _validate_non_negative_int(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PersistentComputeError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


def _pack_dispatch_parameters(
    program: ComputeProgram,
    parameters: Mapping[str, object] | bytes | bytearray | memoryview | None,
) -> bytes | None:
    if program.parameters is None:
        if parameters is not None:
            raise PersistentComputeError(
                "The compute pipeline declares no parameter block, so parameters must be None."
            )
        return None
    if parameters is None:
        raise PersistentComputeError(
            f"Compute parameter block {program.parameters['name']!r} requires host values."
        )
    if isinstance(parameters, (bytes, bytearray, memoryview)):
        packed = bytes(parameters)
        if len(packed) != program.parameters["size"]:
            raise PersistentComputeError(
                f"Packed parameter bytes length ({len(packed)}) does not match "
                f"reflected size ({program.parameters['size']})."
            )
        return packed
    if isinstance(parameters, Mapping):
        try:
            return pack_parameters(program.parameters, dict(parameters))
        except (TypeError, ValueError) as exc:
            raise PersistentComputeError(f"Invalid compute parameter values: {exc}") from exc
    raise PersistentComputeError("Compute parameters must be a field mapping or packed byte buffer.")


class PersistentBuffer(ABC):
    """Device-owned persistent storage buffer with explicit host upload/download."""

    @property
    @abstractmethod
    def size_bytes(self) -> int:
        """Logical size of the buffer in bytes."""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether this buffer or its owning context has been closed."""

    @property
    @abstractmethod
    def context(self) -> ComputeContext:
        """Owning ComputeContext for cross-device validation."""

    @abstractmethod
    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        """Upload host bytes into the device buffer. Blocks until prior GPU work completes."""

    @abstractmethod
    @abstractmethod
    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        """Synchronize GPU execution and read back device buffer bytes to the host."""

    def download_into(
        self,
        dst: bytearray | memoryview,
        size_bytes: int | None = None,
        offset: int = 0,
    ) -> int:
        """Synchronize GPU execution and copy device bytes directly into a writable buffer."""
        raw = self.download(size_bytes=size_bytes, offset=offset)
        read_size = len(raw)
        if read_size == 0:
            return 0
        dst_view = memoryview(dst).cast("B")
        if dst_view.readonly:
            raise PersistentComputeError("download_into destination buffer must be writable.")
        if len(dst_view) < read_size:
            raise PersistentComputeError(
                f"download_into destination buffer ({len(dst_view)} bytes) is smaller than {read_size} bytes."
            )
        dst_view[:read_size] = raw
        return read_size

    @abstractmethod
    def close(self) -> None:
        """Release the underlying device buffer (idempotent)."""

    def __enter__(self) -> PersistentBuffer:
        if self.closed:
            raise PersistentComputeError("Cannot enter a closed PersistentBuffer.")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class PersistentPipeline(ABC):
    """Compiled compute pipeline bound to a ComputeContext for repeated dispatches."""

    @property
    @abstractmethod
    def program(self) -> ComputeProgram:
        """Reflected ComputeProgram metadata."""

    @property
    @abstractmethod
    def context(self) -> ComputeContext:
        """Owning ComputeContext."""

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether this pipeline or its owning context has been closed."""

    @abstractmethod
    def dispatch(
        self,
        buffers: Mapping[str, PersistentBuffer],
        parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        """Record and submit a compute dispatch against persistent device buffers.

        Device writes are synchronized with subsequent dispatches via GPU memory barriers
        without intermediate host readback. Call ``download()`` or ``context.synchronize()``
        to block the host until GPU execution finishes.
        """

    @abstractmethod
    def close(self) -> None:
        """Release pipeline resources (idempotent)."""

    def __enter__(self) -> PersistentPipeline:
        if self.closed:
            raise PersistentComputeError("Cannot enter a closed PersistentPipeline.")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class ComputeContext(ABC):
    """Owns a GPU compute device/queue and creates persistent buffers and pipelines.

    Threading policy: a ComputeContext and its child buffers/pipelines have
    single-thread affinity (callers must not concurrently dispatch or close
    resources on the same context from multiple threads without external locking).
    """

    @property
    def stats(self) -> dict[str, int]:
        """Return instrumentation counters for dispatches, submissions, host waits, and deferred cleanups."""
        return {"dispatches": 0, "submissions": 0, "waits": 0, "deferred_cleanups": 0}

    def reset_stats(self) -> None:
        """Reset instrumentation counters to zero."""

    def release_pool(self) -> int:
        """Synchronize if needed and release all pooled idle buffers, returning bytes freed."""
        return 0

    @contextmanager
    def batch(self):
        """Group dependent dispatches without intermediate host waits."""
        yield
        self.synchronize()

    @property
    @abstractmethod
    def target(self) -> str:
        """Backend target name (e.g. 'vulkan')."""

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Reported GPU or software provider device name."""

    @property
    def provider(self) -> str | None:
        """Resolved execution provider (e.g. 'hardware', 'warp', 'lavapipe', 'swiftshader'), or None if unknown."""
        return None

    @property
    def is_software(self) -> bool | None:
        """Whether execution uses a software/CPU adapter, or None if unknown."""
        return None

    def supports_capability(self, capability: str) -> bool:
        """Return whether the runtime device supports the given named capability."""
        return False

    @property
    def cooperative_matrix_configs(self) -> tuple[tuple[int, int, int], ...]:
        """Supported (M, N, K) f32 subgroup cooperative matrix shapes, if any."""
        return ()

    @property
    @abstractmethod
    def closed(self) -> bool:
        """Whether this context has been closed."""

    @abstractmethod
    def create_buffer(
        self,
        size_bytes: int,
        initial_data: bytes | bytearray | memoryview | None = None,
    ) -> PersistentBuffer:
        """Allocate a persistent device buffer of ``size_bytes`` bytes."""

    @abstractmethod
    def compile_pipeline(self, program: ComputeProgram | Path | str) -> PersistentPipeline:
        """Create a reusable compute pipeline from a ComputeProgram or manifest path."""

    @abstractmethod
    def synchronize(self) -> None:
        """Block the host until all previously submitted dispatches have completed."""

    @abstractmethod
    def close(self) -> None:
        """Synchronize and release all buffers, pipelines, and device resources."""

    def __enter__(self) -> ComputeContext:
        if self.closed:
            raise PersistentComputeError("Cannot enter a closed ComputeContext.")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


class VulkanBuffer(PersistentBuffer):
    """Vulkan storage buffer backed by host-visible, host-coherent device memory."""

    def __init__(
        self,
        context: VulkanComputeContext,
        size_bytes: int,
        vk_buffer: object,
        vk_memory: object,
        alloc_size: int,
        mapped: object | None = None,
    ) -> None:
        self._context = context
        self._size_bytes = size_bytes
        self._vk_buffer = vk_buffer
        self._vk_memory = vk_memory
        self._alloc_size = alloc_size
        self._mapped: object | None = mapped
        context._next_buffer_id += 1
        self._buffer_id = context._next_buffer_id
        self._closed = False

    @property
    def size_bytes(self) -> int:
        return self._size_bytes

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    @property
    def context(self) -> VulkanComputeContext:
        return self._context

    def _ensure_open(self) -> None:
        self._context._ensure_open()
        if self._closed:
            raise PersistentComputeError("PersistentBuffer has been closed.")

    def _ensure_mapped(self) -> object:
        if self._mapped is None:
            gpu = self._context._gpu
            self._mapped = gpu.vk.vkMapMemory(gpu.device, self._vk_memory, 0, self._alloc_size, 0)
        return self._mapped

    def upload(self, data: bytes | bytearray | memoryview, offset: int = 0) -> None:
        self._ensure_open()
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise PersistentComputeError("upload data must be bytes, bytearray, or memoryview.")
        _validate_non_negative_int(offset, "upload offset")
        payload = bytes(data)
        if offset + len(payload) > self._size_bytes:
            raise PersistentComputeError(
                f"Upload range [{offset}, {offset + len(payload)}) exceeds buffer size {self._size_bytes}."
            )
        if not payload:
            return
        self._context.synchronize()
        with _api_errors():
            mapped = self._ensure_mapped()
            mapped[offset : offset + len(payload)] = payload

    def download_into(
        self,
        dst: bytearray | memoryview,
        size_bytes: int | None = None,
        offset: int = 0,
    ) -> int:
        self._ensure_open()
        _validate_non_negative_int(offset, "download offset")
        if offset > self._size_bytes:
            raise PersistentComputeError(
                f"Download offset {offset} exceeds buffer size {self._size_bytes}."
            )
        read_size = (self._size_bytes - offset) if size_bytes is None else _validate_non_negative_int(
            size_bytes, "download size_bytes"
        )
        if offset + read_size > self._size_bytes:
            raise PersistentComputeError(
                f"Download range [{offset}, {offset + read_size}) exceeds buffer size {self._size_bytes}."
            )
        dst_view = memoryview(dst).cast("B")
        if dst_view.readonly:
            raise PersistentComputeError("download_into destination buffer must be writable.")
        if len(dst_view) < read_size:
            raise PersistentComputeError(
                f"download_into destination buffer ({len(dst_view)} bytes) is smaller than {read_size} bytes."
            )
        self._context.synchronize()
        if read_size == 0:
            return 0
        with _api_errors():
            mapped = self._ensure_mapped()
            dst_view[:read_size] = mapped[offset : offset + read_size]
            return read_size

    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        self._ensure_open()
        _validate_non_negative_int(offset, "download offset")
        if offset > self._size_bytes:
            raise PersistentComputeError(
                f"Download offset {offset} exceeds buffer size {self._size_bytes}."
            )
        read_size = (self._size_bytes - offset) if size_bytes is None else _validate_non_negative_int(
            size_bytes, "download size_bytes"
        )
        if offset + read_size > self._size_bytes:
            raise PersistentComputeError(
                f"Download range [{offset}, {offset + read_size}) exceeds buffer size {self._size_bytes}."
            )
        self._context.synchronize()
        if read_size == 0:
            return b""
        with _api_errors():
            mapped = self._ensure_mapped()
            return bytes(mapped[offset : offset + read_size])

    @contextmanager
    def map_view(self):
        """Synchronize pending GPU work and yield a zero-copy byte memoryview over mapped memory."""
        self._ensure_open()
        self._context.synchronize()
        with _api_errors():
            mapped = self._ensure_mapped()
            view = memoryview(mapped).cast("B")[: self._size_bytes]
        try:
            yield view
        finally:
            view.release()

    def _destroy_vk_resources(self) -> None:
        gpu = self._context._gpu
        vk = gpu.vk
        if self._mapped is not None:
            vk.vkUnmapMemory(gpu.device, self._vk_memory)
            self._mapped = None
        vk.vkDestroyBuffer(gpu.device, self._vk_buffer, None)
        vk.vkFreeMemory(gpu.device, self._vk_memory, None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context._buffers.discard(self)
        self._context._pool.on_live_free(self._alloc_size)
        handle = (self._vk_buffer, self._vk_memory, self._alloc_size, self._mapped)
        self._mapped = None
        if self._context.closed:
            self._context._destroy_vk_buffer(handle)
        elif self._context._recording or self._context._submitted:
            self._context._defer_cleanup(lambda: self._context._return_buffer_to_pool(handle))
        else:
            self._context._return_buffer_to_pool(handle)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


_MAX_CACHED_BINDINGS = 32


class VulkanPipeline(PersistentPipeline):
    """Reusable Vulkan compute pipeline with reflected descriptor set bindings."""

    def __init__(
        self,
        context: VulkanComputeContext,
        program: ComputeProgram,
        pipeline: object,
        layout: object,
        set_layouts: list[object],
        desc_pool: object | None,
        descriptor_sets: list[object],
        resources: ExitStack,
    ) -> None:
        self._context = context
        self._program = program
        self._pipeline = pipeline
        self._layout = layout
        self._set_layouts = set_layouts
        self._desc_pool = desc_pool
        self._descriptor_sets = descriptor_sets
        self._resources = resources
        self._closed = False
        self._last_bindings: tuple[tuple[str, int, int], ...] | None = None
        self._binding_cache: dict[tuple[tuple[str, int, int], ...], list[object]] = {}
        self._unused_initial_sets = bool(descriptor_sets)

    @property
    def program(self) -> ComputeProgram:
        return self._program

    @property
    def context(self) -> VulkanComputeContext:
        return self._context

    @property
    def closed(self) -> bool:
        return self._closed or self._context.closed

    def _ensure_open(self) -> None:
        self._context._ensure_open()
        if self._closed:
            raise PersistentComputeError("PersistentPipeline has been closed.")

    def dispatch(
        self,
        buffers: Mapping[str, PersistentBuffer],
        parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
        grid: tuple[int, int, int] | None = None,
    ) -> None:
        self._ensure_open()
        active_grid = self._program.dispatch.grid if grid is None else tuple(grid)
        if len(active_grid) != 3 or any(type(dim) is not int or dim < 1 for dim in active_grid):
            raise PersistentComputeError("Compute dispatch grid must contain 3 positive integers.")
        gx, gy, gz = active_grid
        gpu = self._context._gpu
        if (
            gx > gpu.max_compute_workgroup_count[0]
            or gy > gpu.max_compute_workgroup_count[1]
            or gz > gpu.max_compute_workgroup_count[2]
        ):
            raise PersistentComputeError(
                "Compute dispatch grid exceeds the Vulkan device maxComputeWorkGroupCount limits."
            )

        packed_params = _pack_dispatch_parameters(self._program, parameters)

        provided = dict(buffers or {})
        reflected_names = {buf["name"] for buf in self._program.buffers}
        extra = sorted(set(provided) - reflected_names)
        if extra:
            raise PersistentComputeError(f"Unknown compute buffer name(s): {', '.join(extra)}.")
        missing = sorted(reflected_names - set(provided))
        if missing:
            raise PersistentComputeError(f"Missing required compute buffer binding(s): {', '.join(missing)}.")

        bound_buffers: list[tuple[dict, VulkanBuffer]] = []
        binding_key: list[tuple[str, int, int]] = []
        for buf_meta in self._program.buffers:
            name = buf_meta["name"]
            buf = provided[name]
            if not isinstance(buf, VulkanBuffer):
                raise PersistentComputeError(
                    f"Buffer {name!r} must be a VulkanBuffer created by this context."
                )
            if buf.context is not self._context:
                raise PersistentComputeError(
                    f"Cross-device binding rejected for buffer {name!r}: "
                    f"buffer belongs to a different ComputeContext."
                )
            buf._ensure_open()
            if buf.size_bytes <= 0:
                raise PersistentComputeError(f"Buffer {name!r} must have positive size.")
            bound_buffers.append((buf_meta, buf))
            binding_key.append((name, buf._buffer_id, buf._alloc_size))

        vk = gpu.vk
        with _api_errors():
            if self._context._submitted:
                # Wait on prior submitted batch before resetting/re-recording the command buffer.
                self._context.synchronize()

            active_sets = self._descriptor_sets
            if self._descriptor_sets:
                key_tuple = tuple(binding_key)
                cached_sets = self._binding_cache.get(key_tuple)
                if cached_sets is not None:
                    active_sets = cached_sets
                else:
                    if self._unused_initial_sets:
                        active_sets = self._descriptor_sets
                        self._unused_initial_sets = False
                    else:
                        if len(self._binding_cache) >= _MAX_CACHED_BINDINGS:
                            self._context.synchronize()
                            vk.vkResetDescriptorPool(gpu.device, self._desc_pool, 0)
                            self._binding_cache.clear()
                        active_sets = list(
                            vk.vkAllocateDescriptorSets(
                                gpu.device,
                                vk.VkDescriptorSetAllocateInfo(
                                    descriptorPool=self._desc_pool,
                                    descriptorSetCount=len(self._set_layouts),
                                    pSetLayouts=self._set_layouts,
                                ),
                            )
                        )
                    writes = []
                    for buf_meta, buf in bound_buffers:
                        info = vk.VkDescriptorBufferInfo(
                            buffer=buf._vk_buffer,
                            offset=0,
                            range=buf._alloc_size,
                        )
                        writes.append(
                            vk.VkWriteDescriptorSet(
                                dstSet=active_sets[buf_meta["set"]],
                                dstBinding=buf_meta["binding"],
                                dstArrayElement=0,
                                descriptorCount=1,
                                descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                                pBufferInfo=[info],
                            )
                        )
                    vk.vkUpdateDescriptorSets(gpu.device, len(writes), writes, 0, None)
                    self._binding_cache[key_tuple] = active_sets
                self._last_bindings = key_tuple

            cmd = self._context._command_buffer
            if not self._context._recording:
                vk.vkResetCommandBuffer(cmd, 0)
                vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo())
                self._context._recording = True
                if bound_buffers:
                    pre_barrier = vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_HOST_WRITE_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                    )
                    vk.vkCmdPipelineBarrier(
                        cmd,
                        vk.VK_PIPELINE_STAGE_HOST_BIT | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        1,
                        [pre_barrier],
                        0,
                        None,
                        0,
                        None,
                    )
            elif bound_buffers:
                chain_barrier = vk.VkMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                )
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    [chain_barrier],
                    0,
                    None,
                    0,
                    None,
                )

            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
            if active_sets:
                vk.vkCmdBindDescriptorSets(
                    cmd,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self._layout,
                    0,
                    len(active_sets),
                    active_sets,
                    0,
                    None,
                )
            if packed_params is not None and self._program.parameters is not None:
                vk.vkCmdPushConstants(
                    cmd,
                    self._layout,
                    gpu.parameter_stages(self._program.parameters),
                    self._program.parameters["bindings"]["vulkan"]["offset"],
                    len(packed_params),
                    vk.ffi.from_buffer(packed_params),
                )
            vk.vkCmdDispatch(cmd, gx, gy, gz)
            self._context._stats["dispatches"] += 1
            self._context._has_buffer_writes = self._context._has_buffer_writes or bool(bound_buffers)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._context._pipelines.discard(self)
        if not self._context.closed:
            if self._context._recording or self._context._submitted:
                self._context._defer_cleanup(self._resources.close)
                return
        self._resources.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


class VulkanComputeContext(ComputeContext):
    """Persistent Vulkan compute context owning device, queue, buffers, and pipelines."""

    def __init__(
        self,
        *,
        validation: bool = False,
        provider: str | None = None,
        pool_max_bytes: int | None = None,
    ) -> None:
        _validate_provider(provider, "vulkan")
        self._resources = ExitStack()
        self._closed = False
        self._recording = False
        self._submitted = False
        self._has_buffer_writes = False
        self._batch_depth = 0
        self._retired_resources: list[object] = []
        self._stats = {"dispatches": 0, "submissions": 0, "waits": 0, "deferred_cleanups": 0}
        self._pool = _BoundedBufferPool(max_bytes=_resolve_pool_max_bytes(pool_max_bytes))
        self._next_buffer_id = 0
        self._buffers: weakref.WeakSet[VulkanBuffer] = weakref.WeakSet()
        self._pipelines: weakref.WeakSet[VulkanPipeline] = weakref.WeakSet()

        try:
            with _api_errors():
                vulkan_kwargs = {"provider": provider} if provider is not None else {}
                self._gpu = _Vulkan(
                    self._resources,
                    validation=validation,
                    compute_only=True,
                    **vulkan_kwargs,
                )
                vk = self._gpu.vk
                self._resources.callback(vk.vkDeviceWaitIdle, self._gpu.device)
                self._fence = vk.vkCreateFence(self._gpu.device, vk.VkFenceCreateInfo(), None)
                self._resources.callback(vk.vkDestroyFence, self._gpu.device, self._fence, None)
                self._command_buffer = vk.vkAllocateCommandBuffers(
                    self._gpu.device,
                    vk.VkCommandBufferAllocateInfo(
                        commandPool=self._gpu.pool,
                        level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                        commandBufferCount=1,
                    ),
                )[0]
                self._resources.callback(
                    vk.vkFreeCommandBuffers,
                    self._gpu.device,
                    self._gpu.pool,
                    1,
                    [self._command_buffer],
                )
        except Exception:
            self._closed = True
            self._resources.close()
            raise

    @property
    def target(self) -> str:
        return "vulkan"

    @property
    def device_name(self) -> str:
        return self._gpu.name

    @property
    def provider(self) -> str:
        return self._gpu.provider

    @property
    def is_software(self) -> bool:
        return self._gpu.is_software

    def supports_capability(self, capability: str) -> bool:
        if capability == "cooperative_matrix":
            return bool(self._gpu.cooperative_matrix_configs)
        return False

    @property
    def cooperative_matrix_configs(self) -> tuple[tuple[int, int, int], ...]:
        return self._gpu.cooperative_matrix_configs

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise PersistentComputeError("ComputeContext has been closed.")

    def _destroy_vk_buffer(self, handle: object) -> None:
        vk_buf, vk_mem, _alloc_size, mapped = handle  # type: ignore[misc]
        gpu = self._gpu
        vk = gpu.vk
        errors: list[BaseException] = []
        if mapped is not None:
            try:
                vk.vkUnmapMemory(gpu.device, vk_mem)
            except BaseException as exc:
                errors.append(exc)
        try:
            vk.vkDestroyBuffer(gpu.device, vk_buf, None)
        except BaseException as exc:
            errors.append(exc)
        try:
            vk.vkFreeMemory(gpu.device, vk_mem, None)
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]

    def _return_buffer_to_pool(self, handle: object) -> None:
        _vk_buf, _vk_mem, alloc_size, _mapped = handle  # type: ignore[misc]
        if self._closed:
            self._destroy_vk_buffer(handle)
            return
        self._pool.release(alloc_size, handle, self._destroy_vk_buffer)

    def create_buffer(
        self,
        size_bytes: int,
        initial_data: bytes | bytearray | memoryview | None = None,
        *,
        zero_init: bool = True,
    ) -> VulkanBuffer:
        self._ensure_open()
        _validate_positive_int(size_bytes, "buffer size_bytes")
        payload: bytes | None = None
        if initial_data is not None:
            if not isinstance(initial_data, (bytes, bytearray, memoryview)):
                raise PersistentComputeError("initial_data must be bytes, bytearray, or memoryview.")
            payload = bytes(initial_data)
            if len(payload) > size_bytes:
                raise PersistentComputeError(
                    f"initial_data length ({len(payload)}) exceeds buffer size ({size_bytes})."
                )

        gpu = self._gpu
        vk = gpu.vk
        alloc_size = max(4, (size_bytes + 3) & ~3)
        pooled = self._pool.acquire(alloc_size)
        if pooled is not None:
            vk_buf, vk_mem, _pooled_alloc_size, mapped = pooled  # type: ignore[misc]
            try:
                if payload is not None or zero_init:
                    with _api_errors():
                        if mapped is None:
                            mapped = vk.vkMapMemory(gpu.device, vk_mem, 0, alloc_size, 0)
                        if payload:
                            mapped[0 : len(payload)] = payload
                            if zero_init and len(payload) < alloc_size:
                                mapped[len(payload) : alloc_size] = b"\x00" * (alloc_size - len(payload))
                        elif zero_init:
                            mapped[0:alloc_size] = b"\x00" * alloc_size
            except Exception:
                self._destroy_vk_buffer((vk_buf, vk_mem, alloc_size, mapped))
                raise
            buf = VulkanBuffer(self, size_bytes, vk_buf, vk_mem, alloc_size, mapped=mapped)
            self._pool.on_live_alloc(alloc_size)
            self._buffers.add(buf)
            return buf

        with _api_errors():
            vk_buf = vk.vkCreateBuffer(
                gpu.device,
                vk.VkBufferCreateInfo(
                    size=alloc_size,
                    usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                ),
                None,
            )
            vk_mem = None
            mapped = None
            try:
                reqs = vk.vkGetBufferMemoryRequirements(gpu.device, vk_buf)
                vk_mem = vk.vkAllocateMemory(
                    gpu.device,
                    vk.VkMemoryAllocateInfo(
                        allocationSize=reqs.size,
                        memoryTypeIndex=gpu.memory_type(
                            reqs.memoryTypeBits,
                            vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        ),
                    ),
                    None,
                )
                vk.vkBindBufferMemory(gpu.device, vk_buf, vk_mem, 0)
                if payload is not None or zero_init:
                    mapped = vk.vkMapMemory(gpu.device, vk_mem, 0, alloc_size, 0)
                    if payload:
                        mapped[0 : len(payload)] = payload
                        if zero_init and len(payload) < alloc_size:
                            mapped[len(payload) : alloc_size] = b"\x00" * (alloc_size - len(payload))
                    elif zero_init:
                        mapped[0:alloc_size] = b"\x00" * alloc_size
            except Exception:
                if mapped is not None:
                    vk.vkUnmapMemory(gpu.device, vk_mem)
                vk.vkDestroyBuffer(gpu.device, vk_buf, None)
                if vk_mem is not None:
                    vk.vkFreeMemory(gpu.device, vk_mem, None)
                raise

        self._pool.record_allocation()
        self._pool.on_live_alloc(alloc_size)
        buf = VulkanBuffer(self, size_bytes, vk_buf, vk_mem, alloc_size, mapped=mapped)
        self._buffers.add(buf)
        return buf

    def compile_pipeline(self, program: ComputeProgram | Path | str) -> VulkanPipeline:
        self._ensure_open()
        loaded = load_compute_program(program) if isinstance(program, (Path, str)) else program
        if not isinstance(loaded, ComputeProgram):
            raise PersistentComputeError("compile_pipeline requires a ComputeProgram or manifest Path.")
        if loaded.target != "vulkan":
            raise PersistentComputeError(
                f"VulkanComputeContext requires 'vulkan' target artifacts, got {loaded.target!r}."
            )

        pipe_resources = ExitStack()
        try:
            with _api_errors():
                pipeline, layout, set_layouts = self._gpu.compute_pipeline(pipe_resources, loaded)
                desc_pool = None
                descriptor_sets: list[object] = []
                if set_layouts:
                    vk = self._gpu.vk
                    buf_count = max(1, len(loaded.buffers)) * _MAX_CACHED_BINDINGS
                    pool_sizes = [
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=buf_count,
                        )
                    ]
                    desc_pool = vk.vkCreateDescriptorPool(
                        self._gpu.device,
                        vk.VkDescriptorPoolCreateInfo(
                            maxSets=len(set_layouts) * _MAX_CACHED_BINDINGS,
                            pPoolSizes=pool_sizes,
                            flags=vk.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
                        ),
                        None,
                    )
                    pipe_resources.callback(vk.vkDestroyDescriptorPool, self._gpu.device, desc_pool, None)
                    descriptor_sets = list(
                        vk.vkAllocateDescriptorSets(
                            self._gpu.device,
                            vk.VkDescriptorSetAllocateInfo(
                                descriptorPool=desc_pool,
                                descriptorSetCount=len(set_layouts),
                                pSetLayouts=set_layouts,
                            ),
                        )
                    )
        except Exception:
            pipe_resources.close()
            raise

        pipe = VulkanPipeline(
            self,
            loaded,
            pipeline,
            layout,
            set_layouts,
            desc_pool,
            descriptor_sets,
            pipe_resources,
        )
        self._pipelines.add(pipe)
        return pipe

    @property
    def stats(self) -> dict[str, int]:
        merged = dict(self._stats)
        merged.update(self._pool.stats())
        return merged

    def reset_stats(self) -> None:
        for k in self._stats:
            self._stats[k] = 0
        self._pool.reset_counters()

    def release_pool(self) -> int:
        self._ensure_open()
        if self._recording or self._submitted:
            self.synchronize()
        return self._pool.clear(self._destroy_vk_buffer)

    def _defer_cleanup(self, cleanup_fn: object) -> None:
        self._retired_resources.append(cleanup_fn)
        self._stats["deferred_cleanups"] += 1

    def _flush_retired(self) -> None:
        if not self._retired_resources:
            return
        pending = list(self._retired_resources)
        self._retired_resources.clear()
        remaining: list[object] = []
        errors: list[BaseException] = []
        for fn in pending:
            try:
                fn()  # type: ignore[operator]
            except BaseException as exc:
                errors.append(exc)
                remaining.append(fn)
        if remaining:
            self._retired_resources = remaining + self._retired_resources
        if errors:
            raise errors[0]

    def _submit_recording(self) -> None:
        if not self._recording:
            return
        vk = self._gpu.vk
        cmd = self._command_buffer
        with _api_errors():
            if self._has_buffer_writes:
                post_barrier = vk.VkMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=(
                        vk.VK_ACCESS_HOST_READ_BIT
                        | vk.VK_ACCESS_SHADER_READ_BIT
                        | vk.VK_ACCESS_SHADER_WRITE_BIT
                    ),
                )
                vk.vkCmdPipelineBarrier(
                    cmd,
                    vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    vk.VK_PIPELINE_STAGE_HOST_BIT | vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                    0,
                    1,
                    [post_barrier],
                    0,
                    None,
                    0,
                    None,
                )
            vk.vkEndCommandBuffer(cmd)
            vk.vkResetFences(self._gpu.device, 1, [self._fence])
            submit_info = vk.VkSubmitInfo(pCommandBuffers=[cmd])
            vk.vkQueueSubmit(self._gpu.queue, 1, [submit_info], self._fence)
            self._recording = False
            self._has_buffer_writes = False
            self._submitted = True
            self._stats["submissions"] += 1

    @contextmanager
    def batch(self):
        self._ensure_open()
        self._batch_depth += 1
        try:
            yield
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and not self._closed:
                self._submit_recording()

    def synchronize(self) -> None:
        self._ensure_open()
        if self._recording:
            self._submit_recording()
        if not self._submitted:
            self._flush_retired()
            return
        vk = self._gpu.vk
        with _api_errors():
            vk.vkWaitForFences(self._gpu.device, 1, [self._fence], vk.VK_TRUE, 30_000_000_000)
            self._submitted = False
            self._stats["waits"] += 1
            self._flush_retired()
            self._gpu.check_validation()

    def close(self) -> None:
        if self._closed:
            return
        errors: list[BaseException] = []
        if self._recording or self._submitted:
            try:
                self.synchronize()
            except BaseException as exc:
                errors.append(exc)
        self._recording = False
        self._submitted = False
        try:
            self._flush_retired()
        except BaseException as exc:
            errors.append(exc)
        for buf in list(self._buffers):
            try:
                buf.close()
            except BaseException as exc:
                errors.append(exc)
        for pipe in list(self._pipelines):
            try:
                pipe.close()
            except BaseException as exc:
                errors.append(exc)
        try:
            self._pool.clear(self._destroy_vk_buffer)
        except BaseException as exc:
            errors.append(exc)
        self._buffers.clear()
        self._pipelines.clear()
        self._closed = True
        try:
            self._resources.close()
        except BaseException as exc:
            errors.append(exc)
        if errors:
            raise errors[0]

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


@contextmanager
def open_compute_context(
    target: str = "vulkan",
    *,
    validation: bool = False,
    provider: str | None = None,
):
    """Open a persistent ComputeContext for the specified GPU backend."""
    normalized = target.strip().lower() if isinstance(target, str) else ""
    if normalized == "vulkan":
        ctx: ComputeContext = VulkanComputeContext(validation=validation, provider=provider)
    elif normalized == "directx":
        from .directx_compute import DirectXComputeContext

        ctx = DirectXComputeContext(validation=validation, provider=provider)
    elif normalized == "metal":
        from .metal_compute import MetalComputeContext

        ctx = MetalComputeContext(validation=validation, provider=provider)
    else:
        raise PersistentComputeError(
            f"Persistent ComputeContext is currently implemented for 'vulkan', 'directx', and 'metal', got {target!r}."
        )
    try:
        yield ctx
    finally:
        ctx.close()
