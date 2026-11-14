"""Persistent compute resource contract and Vulkan implementation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from pathlib import Path

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
    def download(self, size_bytes: int | None = None, offset: int = 0) -> bytes:
        """Synchronize GPU execution and read back device buffer bytes to the host."""

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
    """Owns a GPU compute device/queue and creates persistent buffers and pipelines."""

    @property
    @abstractmethod
    def target(self) -> str:
        """Backend target name (e.g. 'vulkan')."""

    @property
    @abstractmethod
    def device_name(self) -> str:
        """Reported GPU or software provider device name."""

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
    ) -> None:
        self._context = context
        self._size_bytes = size_bytes
        self._vk_buffer = vk_buffer
        self._vk_memory = vk_memory
        self._alloc_size = alloc_size
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
        gpu = self._context._gpu
        vk = gpu.vk
        with _api_errors():
            mapped = vk.vkMapMemory(gpu.device, self._vk_memory, 0, self._alloc_size, 0)
            try:
                mapped[offset : offset + len(payload)] = payload
            finally:
                vk.vkUnmapMemory(gpu.device, self._vk_memory)

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
        gpu = self._context._gpu
        vk = gpu.vk
        with _api_errors():
            mapped = vk.vkMapMemory(gpu.device, self._vk_memory, 0, self._alloc_size, 0)
            try:
                return bytes(mapped[offset : offset + read_size])
            finally:
                vk.vkUnmapMemory(gpu.device, self._vk_memory)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._context.closed:
            self._context.synchronize()
            gpu = self._context._gpu
            vk = gpu.vk
            vk.vkDestroyBuffer(gpu.device, self._vk_buffer, None)
            vk.vkFreeMemory(gpu.device, self._vk_memory, None)


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
            binding_key.append((name, id(buf), buf._alloc_size))

        vk = gpu.vk
        with _api_errors():
            # Wait on prior submission before rewriting descriptor sets or reusing command buffer.
            self._context.synchronize()

            key_tuple = tuple(binding_key)
            if self._descriptor_sets and key_tuple != self._last_bindings:
                writes = []
                for buf_meta, buf in bound_buffers:
                    info = vk.VkDescriptorBufferInfo(
                        buffer=buf._vk_buffer,
                        offset=0,
                        range=buf._alloc_size,
                    )
                    writes.append(
                        vk.VkWriteDescriptorSet(
                            dstSet=self._descriptor_sets[buf_meta["set"]],
                            dstBinding=buf_meta["binding"],
                            dstArrayElement=0,
                            descriptorCount=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            pBufferInfo=[info],
                        )
                    )
                vk.vkUpdateDescriptorSets(gpu.device, len(writes), writes, 0, None)
                self._last_bindings = key_tuple

            cmd = self._context._command_buffer
            vk.vkResetCommandBuffer(cmd, 0)
            vk.vkBeginCommandBuffer(cmd, vk.VkCommandBufferBeginInfo())
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
            vk.vkCmdBindPipeline(cmd, vk.VK_PIPELINE_BIND_POINT_COMPUTE, self._pipeline)
            if self._descriptor_sets:
                vk.vkCmdBindDescriptorSets(
                    cmd,
                    vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                    self._layout,
                    0,
                    len(self._descriptor_sets),
                    self._descriptor_sets,
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
            if bound_buffers:
                post_barrier = vk.VkMemoryBarrier(
                    srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                    dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT | vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
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

            vk.vkResetFences(gpu.device, 1, [self._context._fence])
            submit_info = vk.VkSubmitInfo(pCommandBuffers=[cmd])
            vk.vkQueueSubmit(gpu.queue, 1, [submit_info], self._context._fence)
            self._context._submitted = True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._context.closed:
            self._context.synchronize()
        self._resources.close()


class VulkanComputeContext(ComputeContext):
    """Persistent Vulkan compute context owning device, queue, buffers, and pipelines."""

    def __init__(self, *, validation: bool = False, provider: str | None = None) -> None:
        _validate_provider(provider, "vulkan")
        self._resources = ExitStack()
        self._closed = False
        self._submitted = False
        self._buffers: list[VulkanBuffer] = []
        self._pipelines: list[VulkanPipeline] = []

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
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise PersistentComputeError("ComputeContext has been closed.")

    def create_buffer(
        self,
        size_bytes: int,
        initial_data: bytes | bytearray | memoryview | None = None,
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
                mapped = vk.vkMapMemory(gpu.device, vk_mem, 0, alloc_size, 0)
                try:
                    if payload:
                        mapped[0 : len(payload)] = payload
                        if len(payload) < alloc_size:
                            mapped[len(payload) : alloc_size] = b"\x00" * (alloc_size - len(payload))
                    else:
                        mapped[0:alloc_size] = b"\x00" * alloc_size
                finally:
                    vk.vkUnmapMemory(gpu.device, vk_mem)
            except Exception:
                vk.vkDestroyBuffer(gpu.device, vk_buf, None)
                if vk_mem is not None:
                    vk.vkFreeMemory(gpu.device, vk_mem, None)
                raise

        buf = VulkanBuffer(self, size_bytes, vk_buf, vk_mem, alloc_size)
        self._buffers.append(buf)
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
                    buf_count = max(1, len(loaded.buffers))
                    pool_sizes = [
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=buf_count,
                        )
                    ]
                    desc_pool = vk.vkCreateDescriptorPool(
                        self._gpu.device,
                        vk.VkDescriptorPoolCreateInfo(
                            maxSets=len(set_layouts),
                            pPoolSizes=pool_sizes,
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
        self._pipelines.append(pipe)
        return pipe

    def synchronize(self) -> None:
        self._ensure_open()
        if not self._submitted:
            return
        vk = self._gpu.vk
        with _api_errors():
            vk.vkWaitForFences(self._gpu.device, 1, [self._fence], vk.VK_TRUE, 30_000_000_000)
            self._submitted = False
            self._gpu.check_validation()

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._submitted:
                self.synchronize()
        finally:
            for buf in self._buffers:
                buf.close()
            for pipe in self._pipelines:
                pipe.close()
            self._closed = True
            self._resources.close()


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
    else:
        raise PersistentComputeError(
            f"Persistent ComputeContext is currently implemented for 'vulkan' and 'directx', got {target!r}."
        )
    try:
        yield ctx
    finally:
        ctx.close()
