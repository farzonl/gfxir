from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time as clock
import zlib
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .compiler import CompileError, PIPELINE_SCHEMA_VERSION, RASTER_CONVENTION, Toolchain, _check_binary, _run, convert_dxil_to_metallib
from .frontend import Draw
from .interfaces import reflect_pipeline_parameters, verify_graphics_interface
from .parameters import FrameParameters, pack_parameters, verify_parameters


class ViewerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Shader:
    entry_point: str
    code: bytes


@dataclass(frozen=True)
class GraphicsProgram:
    vertex: Shader
    fragment: Shader
    target: str = "vulkan"
    draw: Draw = Draw()
    parameters: dict | None = None


@dataclass(frozen=True)
class RenderedImage:
    width: int
    height: int
    rgba: bytes
    device: str


def load_program(manifest: Path) -> GraphicsProgram:
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if isinstance(data, dict) and type(data.get("schema_version")) is int and data.get("schema_version") == 1:
        raise ViewerError(
            "Schema-version 1 shaders use target-dependent coordinates. "
            "Recompile the Python program into a new output directory with the current compiler."
        )
    if (
        not isinstance(data, dict) or type(data.get("schema_version")) is not int
        or data.get("schema_version") not in (2, 3, 4) or data.get("target") not in ("vulkan", "directx", "metal")
    ):
        raise ViewerError(f"The viewer requires a schema-version {PIPELINE_SCHEMA_VERSION} Vulkan, DirectX, or Metal pipeline manifest.")
    parameters = None
    if data["schema_version"] == 4:
        try:
            verify_graphics_interface(data.get("interface"), preview=True)
            parameters = reflect_pipeline_parameters(data["interface"])
            verify_parameters(data.get("parameters"))
            if parameters is None or data.get("parameters") != parameters:
                raise ValueError("Manifest parameters must match complete stage reflection.")
        except ValueError as error:
            raise ViewerError(f"Invalid parameter resource manifest: {error}") from error
        if data["target"] != "vulkan":
            raise ViewerError("Metal parameter resources require the runtime binding adapters in #10.")
        _frame_parameters(parameters)
    reflected = data.get("interface")
    has_parameters = isinstance(reflected, dict) and any(
        isinstance(stage, dict) and "parameters" in stage for stage in reflected.values()
    )
    if data["schema_version"] != 4 and ("parameters" in data or has_parameters):
        raise ViewerError("Parameter resources require schema version 4; recompile this stale manifest.")
    if data["schema_version"] == 2:
        legacy_interface = {
            "vertex_input": "vertex_index:u32",
            "vertex_output": {"position": "clip_space:f32x4", "color": "location0:f32x4"},
            "fragment_input": {"color": "location0:f32x4"},
            "fragment_output": {"color": "target0:f32x4"},
        }
        if data.get("interface") != legacy_interface or data.get("draw") != Draw().to_dict():
            raise ViewerError("Unsupported legacy version-2 interface; recompile with the native gfx compiler.")
    else:
        try:
            verify_graphics_interface(data.get("interface"), preview=True)
        except ValueError as error:
            raise ViewerError(f"Unsupported graphics interface: {error}") from error
    draw_data = data.get("draw")
    if not isinstance(draw_data, dict) or set(draw_data) != {"topology", "vertex_count"}:
        raise ViewerError("A separate topology/vertex_count draw description is required.")
    try:
        draw = Draw(vertex_count=draw_data["vertex_count"], topology=draw_data["topology"])
    except ValueError as error:
        raise ViewerError(f"Invalid draw description: {error}") from error
    if data.get("raster_convention") != RASTER_CONVENTION:
        raise ViewerError("The pipeline does not declare the compiler's canonical raster convention; recompile it.")
    entries = data.get("entries")
    if not isinstance(entries, list) or len(entries) != 2:
        raise ViewerError("The manifest must contain exactly one vertex and one fragment entry.")
    shaders: dict[str, Shader] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ViewerError("Invalid shader entry in manifest.")
        stage, name, filename = entry.get("stage"), entry.get("entry_point"), entry.get("binary")
        if stage not in ("vertex", "fragment") or stage in shaders:
            raise ViewerError("The manifest must contain unique vertex and fragment stages.")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ViewerError("Invalid shader entry-point name.")
        if data["schema_version"] in (3, 4) and data["interface"][stage]["name"] != name:
            raise ViewerError("Shader entry-point names must match the compiler-generated reflection.")
        if not isinstance(filename, str) or not filename:
            raise ViewerError("A shader binary path is required.")
        binary = (manifest.parent / filename).resolve()
        if Path(filename).is_absolute() or not binary.is_relative_to(manifest.parent):
            raise ViewerError("Shader binaries must be inside the pipeline directory.")
        magic, minimum = {
            "vulkan": (b"\x03\x02\x23\x07", 20), "directx": (b"DXBC", 32), "metal": (b"MTLB", 16),
        }[data["target"]]
        _check_binary(binary, magic, minimum)
        code = binary.read_bytes()
        if data["target"] == "vulkan" and len(code) % 4:
            raise ViewerError(f"SPIR-V binary is not word-aligned: {filename}.")
        shaders[stage] = Shader(name, code)
    return GraphicsProgram(shaders["vertex"], shaders["fragment"], data["target"], draw, parameters)


def _frame_parameters(parameters: dict | None) -> None:
    if parameters is None:
        return
    try:
        verify_parameters(parameters)
    except ValueError as error:
        raise ViewerError(f"Invalid frame parameter resource: {error}") from error
    fields = [{"name": field["name"], "type": field["type"]} for field in parameters["fields"]]
    if ({"name": parameters["name"], "fields": fields} != FrameParameters.to_dict()
            or any(stage not in ("vertex", "fragment") for stage in parameters["stages"])):
        raise ViewerError("The preview supplies FrameParameters only; custom parameter blocks require explicit host values.")


def _time_value(value: float) -> float:
    if type(value) not in (int, float) or value < 0 or value > 3.4028234663852886e38 or not math.isfinite(value):
        raise ViewerError("Time must be finite, nonnegative f32 seconds.")
    return float(value)


def _frame_data(parameters: dict, width: int, height: int, elapsed: float) -> bytes:
    _frame_parameters(parameters)
    if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
        raise ViewerError("Frame parameters require positive framebuffer-pixel dimensions.")
    return pack_parameters(parameters, {"time": _time_value(elapsed), "resolution": (width, height)})


def _dimensions(width: int, height: int) -> None:
    if type(width) is not int or type(height) is not int or not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ViewerError("Preview dimensions must be integers between 1 and 4096.")


def write_png(path: Path, image: RenderedImage) -> None:
    _dimensions(image.width, image.height)
    stride = image.width * 4
    if len(image.rgba) != stride * image.height:
        raise ViewerError("RGBA readback size does not match the image dimensions.")

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))

    rows = b"".join(
        b"\0" + image.rgba[offset:offset + stride]
        for offset in range(0, len(image.rgba), stride)
    )
    header = struct.pack(">IIBBBBB", image.width, image.height, 8, 6, 0, 0, 0)
    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")
    with Path(path).open("xb") as output:
        output.write(encoded)


@contextmanager
def _metal_libraries(program: GraphicsProgram, toolchain: Toolchain):
    if sys.platform != "darwin":
        raise ViewerError("DXIL-to-Metal preview requires macOS. This is not a DirectX runtime renderer.")
    if program.target not in ("directx", "metal"):
        raise ViewerError("The Metal viewer requires DirectX or Metal artifacts.")
    if program.parameters is not None:
        raise ViewerError("Metal parameter resources require the runtime binding adapters in #10.")
    with tempfile.TemporaryDirectory(prefix="mlir-gpu-metal-") as work:
        directory = Path(work)
        libraries = []
        for stage, shader in (("vertex", program.vertex), ("fragment", program.fragment)):
            library = directory / f"{stage}.metallib"
            if program.target == "directx":
                source, validated = directory / f"{stage}.dxil", directory / f"{stage}.validated.dxil"
                source.write_bytes(shader.code)
                _run([toolchain.find("dxv"), str(source), "-o", str(validated)])
                _check_binary(validated, b"DXBC", 32)
                convert_dxil_to_metallib(validated, library, shader.entry_point, toolchain.find("metal-shaderconverter"))
            else:
                library.write_bytes(shader.code)
            libraries.extend([f"--{stage}", str(library), f"--{stage}-entry", shader.entry_point])
        yield directory, [toolchain.find("mlir-gpu-metal-view"), *libraries,
                          "--vertex-count", str(program.draw.vertex_count)]


def _metal_run(arguments: list[str], validation: bool, capture: bool) -> str:
    environment = os.environ.copy()
    if validation:
        environment.update(MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1")
    try:
        result = subprocess.run(arguments, text=True, capture_output=capture, env=environment, check=False)
    except OSError as error:
        raise ViewerError(f"Cannot start the native Metal preview: {error}") from error
    if result.returncode:
        diagnostics = (result.stderr or result.stdout or "").strip() if capture else "See native renderer diagnostics above."
        raise ViewerError(f"Native Metal preview failed ({result.returncode}): {diagnostics}")
    return result.stdout or ""


def _preview_toolchain() -> Toolchain:
    directory = os.environ.get("GPU_TOOLS_BIN")
    return Toolchain(tools_bin=Path(directory) if directory else None)


def render_metal_offscreen(
    program: GraphicsProgram, width: int = 640, height: int = 480,
    validation: bool = False, toolchain: Toolchain | None = None,
) -> RenderedImage:
    _dimensions(width, height)
    with _metal_libraries(program, toolchain or _preview_toolchain()) as (directory, arguments):
        output = directory / "readback.rgba"
        text = _metal_run(
            [*arguments, "--width", str(width), "--height", str(height), "--readback", str(output)],
            validation, True,
        )
        report = json.loads(text)
        if (
            not isinstance(report, dict) or report.get("width") != width or report.get("height") != height
            or not isinstance(report.get("device"), str) or not report["device"]
        ):
            raise ViewerError("The native Metal renderer returned an invalid render report.")
        rgba = output.read_bytes()
        if len(rgba) != width * height * 4:
            raise ViewerError("Metal RGBA readback size does not match the requested dimensions.")
        return RenderedImage(width, height, rgba, report["device"])


def show_metal_window(
    program: GraphicsProgram, width: int = 800, height: int = 600, frames: int = 0,
    validation: bool = False, toolchain: Toolchain | None = None,
) -> None:
    _dimensions(width, height)
    if type(frames) is not int or frames < 0:
        raise ViewerError("Frame count must be a nonnegative integer (0 means until closed).")
    with _metal_libraries(program, toolchain or _preview_toolchain()) as (_, arguments):
        _metal_run(
            [*arguments, "--width", str(width), "--height", str(height), "--frames", str(frames)],
            validation, False,
        )


def _binding(name: str):
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as error:
        raise ViewerError(
            f"Cannot load {name}: {error}. Install the viewer extra with "
            'python -m pip install -e ".[viewer]". A system Vulkan loader/driver is also required; '
            "on macOS, set DYLD_LIBRARY_PATH to the Vulkan SDK lib directory "
            "(often /usr/local/lib) before starting Python."
        ) from error


@contextmanager
def _api_errors():
    vk = _binding("vulkan")
    try:
        yield
    except (vk.VkError, vk.VkException, vk.ProcedureNotFoundError, vk.ExtensionNotSupportedError) as error:
        raise ViewerError(f"Vulkan operation failed: {type(error).__name__}. {error}") from error


class _Vulkan:
    def __init__(self, resources: ExitStack, window=None, glfw=None, validation: bool = False):
        vk = self.vk = _binding("vulkan")
        self.resources = resources
        self.surface = None
        self.validation_errors: list[str] = []
        resources.callback(self.check_validation)
        extensions = list(glfw.get_required_instance_extensions() or []) if window else []
        available = {extension.extensionName for extension in vk.vkEnumerateInstanceExtensionProperties(None)}
        flags = 0
        if "VK_KHR_portability_enumeration" in available:
            extensions.append("VK_KHR_portability_enumeration")
            flags = vk.VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR
        if "VK_KHR_get_physical_device_properties2" in available:
            extensions.append("VK_KHR_get_physical_device_properties2")
        if missing := set(extensions) - available:
            raise ViewerError(f"Required Vulkan instance extensions are unavailable: {', '.join(sorted(missing))}.")
        layers = []
        debug_info = None
        if validation:
            available_layers = {layer.layerName for layer in vk.vkEnumerateInstanceLayerProperties()}
            if "VK_LAYER_KHRONOS_validation" not in available_layers:
                raise ViewerError("--validation requires VK_LAYER_KHRONOS_validation from the Vulkan SDK.")
            if "VK_EXT_debug_utils" not in available:
                raise ViewerError("--validation requires the VK_EXT_debug_utils extension.")
            layers.append("VK_LAYER_KHRONOS_validation")
            extensions.append("VK_EXT_debug_utils")

            def report(severity, message_type, data, user):
                message = vk.ffi.string(data.pMessage).decode("utf-8", errors="replace")
                if severity & vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
                    self.validation_errors.append(message)
                print(f"Vulkan validation: {message}", file=sys.stderr)
                return vk.VK_FALSE

            debug_info = vk.VkDebugUtilsMessengerCreateInfoEXT(
                messageSeverity=vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | vk.VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT,
                messageType=vk.VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | vk.VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
                pfnUserCallback=report,
            )
        self.instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            pNext=debug_info,
            flags=flags,
            pApplicationInfo=vk.VkApplicationInfo(pApplicationName="mlir-gpu triangle", apiVersion=vk.VK_API_VERSION_1_0),
            ppEnabledExtensionNames=extensions,
            ppEnabledLayerNames=layers,
        ), None)
        resources.callback(vk.vkDestroyInstance, self.instance, None)
        if debug_info is not None:
            messenger = self.instance_function("vkCreateDebugUtilsMessengerEXT")(self.instance, debug_info, None)
            resources.callback(self.instance_function("vkDestroyDebugUtilsMessengerEXT"), self.instance, messenger, None)
        if window:
            surface = vk.ffi.new("VkSurfaceKHR*")
            result = glfw.create_window_surface(self.instance, window, None, surface)
            if result != vk.VK_SUCCESS:
                raise ViewerError(f"GLFW could not create a Vulkan surface (VkResult {result}).")
            self.surface = surface[0]
            resources.callback(self.instance_function("vkDestroySurfaceKHR"), self.instance, self.surface, None)

        candidates = []
        for physical in vk.vkEnumeratePhysicalDevices(self.instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            for index, queue in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)):
                if not queue.queueCount or not queue.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                    continue
                device_extensions = {item.extensionName for item in vk.vkEnumerateDeviceExtensionProperties(physical, None)}
                if self.surface is not None:
                    if "VK_KHR_swapchain" not in device_extensions:
                        continue
                    if not self.instance_function("vkGetPhysicalDeviceSurfaceSupportKHR")(physical, index, self.surface):
                        continue
                candidates.append((properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_CPU, physical, index, properties, device_extensions))
                break
        if not candidates:
            raise ViewerError("No Vulkan device has a suitable graphics/presentation queue.")
        _, self.physical, self.family, properties, device_extensions = sorted(candidates, key=lambda candidate: candidate[0])[0]
        self.name = properties.deviceName
        self.max_width = properties.limits.maxFramebufferWidth
        self.max_height = properties.limits.maxFramebufferHeight
        self.max_push_constants = properties.limits.maxPushConstantsSize
        enabled = ["VK_KHR_swapchain"] if self.surface is not None else []
        if "VK_KHR_portability_subset" in device_extensions:
            enabled.append("VK_KHR_portability_subset")
        queue_info = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.family, pQueuePriorities=[1.0])
        self.device = vk.vkCreateDevice(self.physical, vk.VkDeviceCreateInfo(
            pQueueCreateInfos=[queue_info],
            ppEnabledExtensionNames=enabled,
        ), None)
        resources.callback(vk.vkDestroyDevice, self.device, None)
        self.queue = vk.vkGetDeviceQueue(self.device, self.family, 0)
        self.pool = vk.vkCreateCommandPool(self.device, vk.VkCommandPoolCreateInfo(
            queueFamilyIndex=self.family, flags=vk.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        ), None)
        resources.callback(vk.vkDestroyCommandPool, self.device, self.pool, None)

    def check_validation(self):
        if self.validation_errors:
            raise ViewerError(f"Vulkan validation reported {len(self.validation_errors)} error(s): {self.validation_errors[0]}")

    def instance_function(self, name):
        return self.vk.vkGetInstanceProcAddr(self.instance, name)

    def device_function(self, name):
        return self.vk.vkGetDeviceProcAddr(self.device, name)

    def memory_type(self, bits, flags):
        properties = self.vk.vkGetPhysicalDeviceMemoryProperties(self.physical)
        for index in range(properties.memoryTypeCount):
            if bits & (1 << index) and properties.memoryTypes[index].propertyFlags & flags == flags:
                return index
        raise ViewerError(f"No Vulkan memory type supports required flags 0x{flags:x}.")

    def view(self, resources, image, format):
        vk = self.vk
        view = vk.vkCreateImageView(self.device, vk.VkImageViewCreateInfo(
            image=image, viewType=vk.VK_IMAGE_VIEW_TYPE_2D, format=format,
            subresourceRange=vk.VkImageSubresourceRange(
                aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, levelCount=1, layerCount=1,
            ),
        ), None)
        resources.callback(vk.vkDestroyImageView, self.device, view, None)
        return view

    def image(self, resources, width, height, format):
        vk = self.vk
        properties = vk.vkGetPhysicalDeviceFormatProperties(self.physical, format)
        if not properties.optimalTilingFeatures & vk.VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT:
            raise ViewerError("The Vulkan device does not support the offscreen color format.")
        image = vk.vkCreateImage(self.device, vk.VkImageCreateInfo(
            imageType=vk.VK_IMAGE_TYPE_2D, format=format,
            extent=vk.VkExtent3D(width=width, height=height, depth=1),
            mipLevels=1, arrayLayers=1, samples=vk.VK_SAMPLE_COUNT_1_BIT,
            tiling=vk.VK_IMAGE_TILING_OPTIMAL,
            usage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE, initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
        ), None)
        memory = None

        def destroy():
            vk.vkDestroyImage(self.device, image, None)
            if memory is not None:
                vk.vkFreeMemory(self.device, memory, None)

        resources.callback(destroy)
        requirements = vk.vkGetImageMemoryRequirements(self.device, image)
        memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(
            allocationSize=requirements.size,
            memoryTypeIndex=self.memory_type(requirements.memoryTypeBits, vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT),
        ), None)
        vk.vkBindImageMemory(self.device, image, memory, 0)
        return image, self.view(resources, image, format)

    def readback_buffer(self, resources, size):
        vk = self.vk
        buffer = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            size=size, usage=vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT, sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        ), None)
        memory = None

        def destroy():
            vk.vkDestroyBuffer(self.device, buffer, None)
            if memory is not None:
                vk.vkFreeMemory(self.device, memory, None)

        resources.callback(destroy)
        requirements = vk.vkGetBufferMemoryRequirements(self.device, buffer)
        memory = vk.vkAllocateMemory(self.device, vk.VkMemoryAllocateInfo(
            allocationSize=requirements.size,
            memoryTypeIndex=self.memory_type(
                requirements.memoryTypeBits,
                vk.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            ),
        ), None)
        vk.vkBindBufferMemory(self.device, buffer, memory, 0)
        return buffer, memory

    def pipeline(self, resources, program, format, width, height, present):
        vk = self.vk
        _frame_parameters(program.parameters)
        if width > self.max_width or height > self.max_height:
            raise ViewerError("The requested preview exceeds the device's framebuffer limits.")
        final_layout = vk.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR if present else vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL
        # Keep pointer-bearing array elements alive through the binding's struct copies.
        subpass = vk.VkSubpassDescription(
            pipelineBindPoint=vk.VK_PIPELINE_BIND_POINT_GRAPHICS,
            pColorAttachments=[vk.VkAttachmentReference(attachment=0, layout=vk.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL)],
        )
        render_pass = vk.vkCreateRenderPass(self.device, vk.VkRenderPassCreateInfo(
            pAttachments=[vk.VkAttachmentDescription(
                format=format, samples=vk.VK_SAMPLE_COUNT_1_BIT,
                loadOp=vk.VK_ATTACHMENT_LOAD_OP_CLEAR, storeOp=vk.VK_ATTACHMENT_STORE_OP_STORE,
                stencilLoadOp=vk.VK_ATTACHMENT_LOAD_OP_DONT_CARE, stencilStoreOp=vk.VK_ATTACHMENT_STORE_OP_DONT_CARE,
                initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED, finalLayout=final_layout,
            )],
            pSubpasses=[subpass],
            pDependencies=[
                vk.VkSubpassDependency(
                    srcSubpass=vk.VK_SUBPASS_EXTERNAL, dstSubpass=0,
                    srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    dstStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    dstAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                ),
                vk.VkSubpassDependency(
                    srcSubpass=0, dstSubpass=vk.VK_SUBPASS_EXTERNAL,
                    srcStageMask=vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                    srcAccessMask=vk.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                    dstStageMask=vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT if present else vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    dstAccessMask=0 if present else vk.VK_ACCESS_TRANSFER_READ_BIT,
                ),
            ],
        ), None)
        resources.callback(vk.vkDestroyRenderPass, self.device, render_pass, None)
        ranges = []
        if program.parameters is not None:
            if program.parameters["size"] > self.max_push_constants:
                raise ViewerError("The reflected frame block exceeds the device's push-constant limit.")
            ranges.append(vk.VkPushConstantRange(
                stageFlags=self.parameter_stages(program.parameters),
                offset=program.parameters["bindings"]["vulkan"]["offset"],
                size=program.parameters["size"],
            ))
        layout = vk.vkCreatePipelineLayout(self.device, vk.VkPipelineLayoutCreateInfo(pPushConstantRanges=ranges), None)
        resources.callback(vk.vkDestroyPipelineLayout, self.device, layout, None)
        stages = []
        for shader, stage in ((program.vertex, vk.VK_SHADER_STAGE_VERTEX_BIT), (program.fragment, vk.VK_SHADER_STAGE_FRAGMENT_BIT)):
            module = vk.vkCreateShaderModule(self.device, vk.VkShaderModuleCreateInfo(codeSize=len(shader.code), pCode=shader.code), None)
            resources.callback(vk.vkDestroyShaderModule, self.device, module, None)
            stages.append(vk.VkPipelineShaderStageCreateInfo(stage=stage, module=module, pName=shader.entry_point))
        pipeline = vk.vkCreateGraphicsPipelines(self.device, vk.VK_NULL_HANDLE, 1, [vk.VkGraphicsPipelineCreateInfo(
            pStages=stages,
            pVertexInputState=vk.VkPipelineVertexInputStateCreateInfo(),
            pInputAssemblyState=vk.VkPipelineInputAssemblyStateCreateInfo(topology=vk.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST),
            pViewportState=vk.VkPipelineViewportStateCreateInfo(
                pViewports=[vk.VkViewport(width=float(width), height=float(height), minDepth=0.0, maxDepth=1.0)],
                pScissors=[vk.VkRect2D(extent=vk.VkExtent2D(width=width, height=height))],
            ),
            pRasterizationState=vk.VkPipelineRasterizationStateCreateInfo(
                polygonMode=vk.VK_POLYGON_MODE_FILL, cullMode=vk.VK_CULL_MODE_NONE,
                frontFace=vk.VK_FRONT_FACE_COUNTER_CLOCKWISE, lineWidth=1.0,
            ),
            pMultisampleState=vk.VkPipelineMultisampleStateCreateInfo(rasterizationSamples=vk.VK_SAMPLE_COUNT_1_BIT),
            pColorBlendState=vk.VkPipelineColorBlendStateCreateInfo(pAttachments=[vk.VkPipelineColorBlendAttachmentState(
                colorWriteMask=vk.VK_COLOR_COMPONENT_R_BIT | vk.VK_COLOR_COMPONENT_G_BIT | vk.VK_COLOR_COMPONENT_B_BIT | vk.VK_COLOR_COMPONENT_A_BIT,
            )]),
            layout=layout, renderPass=render_pass, subpass=0,
        )], None)[0]
        resources.callback(vk.vkDestroyPipeline, self.device, pipeline, None)
        return render_pass, pipeline, layout

    def parameter_stages(self, parameters):
        flags = 0
        for stage in parameters["stages"]:
            flags |= {"vertex": self.vk.VK_SHADER_STAGE_VERTEX_BIT,
                      "fragment": self.vk.VK_SHADER_STAGE_FRAGMENT_BIT}[stage]
        return flags

    def framebuffer(self, resources, render_pass, view, width, height):
        vk = self.vk
        framebuffer = vk.vkCreateFramebuffer(self.device, vk.VkFramebufferCreateInfo(
            renderPass=render_pass, pAttachments=[view], width=width, height=height, layers=1,
        ), None)
        resources.callback(vk.vkDestroyFramebuffer, self.device, framebuffer, None)
        return framebuffer

    def command(self, resources, render_pass, pipeline, framebuffer, width, height, draw: Draw, copy=None,
                *, layout=None, parameters=None, elapsed=0.0, command=None):
        vk = self.vk
        data = _frame_data(parameters, width, height, elapsed) if parameters is not None else None
        if data is not None and layout is None:
            raise ViewerError("Frame parameters require the reflected pipeline layout.")
        if command is None:
            command = vk.vkAllocateCommandBuffers(self.device, vk.VkCommandBufferAllocateInfo(
                commandPool=self.pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=1,
            ))[0]
            resources.callback(vk.vkFreeCommandBuffers, self.device, self.pool, 1, [command])
        else:
            vk.vkResetCommandBuffer(command, 0)
        vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo())
        vk.vkCmdBeginRenderPass(command, vk.VkRenderPassBeginInfo(
            renderPass=render_pass, framebuffer=framebuffer,
            renderArea=vk.VkRect2D(extent=vk.VkExtent2D(width=width, height=height)),
            pClearValues=[vk.VkClearValue(color=vk.VkClearColorValue(float32=[0.02, 0.02, 0.04, 1.0]))],
        ), vk.VK_SUBPASS_CONTENTS_INLINE)
        vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline)
        if data is not None:
            vk.vkCmdPushConstants(command, layout, self.parameter_stages(parameters),
                                  parameters["bindings"]["vulkan"]["offset"], len(data), vk.ffi.from_buffer(data))
        vk.vkCmdDraw(command, draw.vertex_count, 1, 0, 0)
        vk.vkCmdEndRenderPass(command)
        if copy is not None:
            image, buffer = copy
            vk.vkCmdCopyImageToBuffer(command, image, vk.VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, 1, [
                vk.VkBufferImageCopy(
                    imageSubresource=vk.VkImageSubresourceLayers(aspectMask=vk.VK_IMAGE_ASPECT_COLOR_BIT, layerCount=1),
                    imageExtent=vk.VkExtent3D(width=width, height=height, depth=1),
                ),
            ])
            vk.vkCmdPipelineBarrier(
                command, vk.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.VK_PIPELINE_STAGE_HOST_BIT, 0,
                1, [vk.VkMemoryBarrier(srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT, dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT)],
                0, None, 0, None,
            )
        vk.vkEndCommandBuffer(command)
        return command


def render_offscreen(program: GraphicsProgram, width: int = 640, height: int = 480,
                     validation: bool = False, time: float = 0.0) -> RenderedImage:
    elapsed = _time_value(time)
    with _offscreen_renderer(program, width, height, validation) as render:
        return render(elapsed)


@contextmanager
def _offscreen_renderer(program: GraphicsProgram, width: int, height: int, validation: bool):
    _dimensions(width, height)
    _frame_parameters(program.parameters)
    if program.target != "vulkan":
        raise ViewerError("Vulkan rendering requires Vulkan artifacts; use the Metal preview for DXIL.")
    with _api_errors(), ExitStack() as resources:
        gpu = _Vulkan(resources, validation=validation)
        vk = gpu.vk
        format = vk.VK_FORMAT_R8G8B8A8_UNORM
        image, view = gpu.image(resources, width, height, format)
        buffer, memory = gpu.readback_buffer(resources, width * height * 4)
        render_pass, pipeline, layout = gpu.pipeline(resources, program, format, width, height, False)
        framebuffer = gpu.framebuffer(resources, render_pass, view, width, height)
        fence = vk.vkCreateFence(gpu.device, vk.VkFenceCreateInfo(), None)
        resources.callback(vk.vkDestroyFence, gpu.device, fence, None)
        command = None

        def render(elapsed):
            nonlocal command
            first = command is None
            command = gpu.command(resources, render_pass, pipeline, framebuffer, width, height, program.draw, (image, buffer),
                                  layout=layout, parameters=program.parameters, elapsed=_time_value(elapsed), command=command)
            if first:
                resources.callback(vk.vkDeviceWaitIdle, gpu.device)
            vk.vkResetFences(gpu.device, 1, [fence])
            vk.vkQueueSubmit(gpu.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[command])], fence)
            vk.vkWaitForFences(gpu.device, 1, [fence], vk.VK_TRUE, 10_000_000_000)
            mapped = vk.vkMapMemory(gpu.device, memory, 0, width * height * 4, 0)
            try:
                rgba = bytes(mapped)
            finally:
                vk.vkUnmapMemory(gpu.device, memory)
            return RenderedImage(width, height, rgba, gpu.name)

        yield render


def _video_encoder() -> str:
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise ViewerError("MP4 export requires FFmpeg on PATH or the video extra: "
                          "python -m pip install -e '.[video]'.") from error
    try:
        return imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as error:
        raise ViewerError(f"Cannot find the MP4 encoder: {error}") from error


def export_mp4(program: GraphicsProgram, path: Path, width: int = 640, height: int = 480,
               duration: float = 4.0, fps: int = 30, start_time: float = 0.0,
               validation: bool = False) -> int:
    _dimensions(width, height)
    if width % 2 or height % 2:
        raise ViewerError("H.264 MP4 dimensions must be even for yuv420p.")
    if type(fps) is not int or not 1 <= fps <= 240:
        raise ViewerError("Video FPS must be an integer between 1 and 240.")
    seconds = _time_value(duration)
    if seconds == 0:
        raise ViewerError("Video duration must be positive.")
    start = _time_value(start_time)
    count = math.ceil(Fraction(str(seconds)) * fps)
    _time_value(start + (count - 1) / fps)
    if program.target != "vulkan":
        raise ViewerError("MP4 export currently requires Vulkan artifacts; Metal frame resources remain #10.")
    _frame_parameters(program.parameters)
    path = Path(path).absolute()
    if path.suffix.lower() != ".mp4":
        raise ViewerError("Video output must have an .mp4 extension.")
    if path.exists():
        raise ViewerError(f"Output already exists; choose a new MP4 path: {path}")
    executable = _video_encoder()
    with tempfile.TemporaryDirectory(prefix=".mlir-gpu-video-", dir=path.parent) as work, tempfile.TemporaryFile() as diagnostics:
        output = Path(work) / "animation.mp4"
        arguments = [
            executable, "-hide_banner", "-loglevel", "error", "-n",
            "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", f"{width}x{height}",
            "-framerate", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
            "-crf", "18", "-pix_fmt", "yuv420p", "-frames:v", str(count),
            "-movflags", "+faststart", str(output),
        ]

        def encoder_error(status):
            diagnostics.seek(0)
            message = diagnostics.read(65536).decode("utf-8", errors="replace").strip()
            return ViewerError(f"MP4 encoder failed ({status}): {message or '(no diagnostics)'}")

        try:
            encoder = subprocess.Popen(arguments, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                       stderr=diagnostics, bufsize=0)
        except OSError as error:
            raise ViewerError(f"Cannot start the MP4 encoder: {error}") from error
        with encoder:
            try:
                with _offscreen_renderer(program, width, height, validation) as render:
                    for index in range(count):
                        image = render(start + index / fps)
                        if (image.width, image.height, len(image.rgba)) != (width, height, width * height * 4):
                            raise ViewerError("Video RGBA frame does not match the requested dimensions.")
                        data = memoryview(image.rgba)
                        while data:
                            written = encoder.stdin.write(data)
                            if not written:
                                raise ViewerError("MP4 encoder stopped accepting frame data.")
                            data = data[written:]
                encoder.stdin.close()
                status = encoder.wait()
                if status:
                    raise encoder_error(status)
            except BrokenPipeError as error:
                raise encoder_error(encoder.wait()) from error
            finally:
                if not encoder.stdin.closed:
                    encoder.stdin.close()
                if encoder.poll() is None:
                    encoder.terminate()
                    try:
                        encoder.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        encoder.kill()
                        encoder.wait()
        if not output.is_file() or output.stat().st_size < 32:
            raise ViewerError("MP4 encoder did not produce a complete MP4 container.")
        with output.open("rb") as binary:
            header = binary.read(12)
        if len(header) != 12 or header[4:8] != b"ftyp":
            raise ViewerError("MP4 encoder did not produce a valid MP4 container.")
        try:
            os.link(output, path)
        except FileExistsError as error:
            raise ViewerError(f"Output already exists; choose a new MP4 path: {path}") from error
    return count


def _swapchain(gpu, resources, program, width, height):
    vk = gpu.vk
    capabilities = gpu.instance_function("vkGetPhysicalDeviceSurfaceCapabilitiesKHR")(gpu.physical, gpu.surface)
    formats = gpu.instance_function("vkGetPhysicalDeviceSurfaceFormatsKHR")(gpu.physical, gpu.surface)
    if not formats:
        raise ViewerError("The surface exposes no Vulkan color formats.")
    format, color_space = None, None
    for candidate in formats:
        if (
            candidate.format in (vk.VK_FORMAT_B8G8R8A8_UNORM, vk.VK_FORMAT_R8G8B8A8_UNORM)
            and candidate.colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
        ):
            format, color_space = candidate.format, candidate.colorSpace
            break
    if (
        format is None and len(formats) == 1 and formats[0].format == vk.VK_FORMAT_UNDEFINED
        and formats[0].colorSpace == vk.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
    ):
        format, color_space = vk.VK_FORMAT_B8G8R8A8_UNORM, formats[0].colorSpace
    if format is None:
        raise ViewerError("The surface requires a linear RGBA/BGRA8 UNORM attachment with sRGB presentation for target parity.")
    if not capabilities.supportedUsageFlags & vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT:
        raise ViewerError("The surface does not support color attachments.")
    if capabilities.currentExtent.width != 0xFFFFFFFF:
        width, height = capabilities.currentExtent.width, capabilities.currentExtent.height
    else:
        width = max(capabilities.minImageExtent.width, min(width, capabilities.maxImageExtent.width))
        height = max(capabilities.minImageExtent.height, min(height, capabilities.maxImageExtent.height))
    if width == 0 or height == 0:
        raise vk.VkErrorOutOfDateKhr()
    count = capabilities.minImageCount + 1
    if capabilities.maxImageCount:
        count = min(count, capabilities.maxImageCount)
    alpha = next(
        bit for bit in (vk.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, vk.VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
                        vk.VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR, vk.VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR)
        if capabilities.supportedCompositeAlpha & bit
    )
    swapchain = gpu.device_function("vkCreateSwapchainKHR")(gpu.device, vk.VkSwapchainCreateInfoKHR(
        surface=gpu.surface, minImageCount=count, imageFormat=format, imageColorSpace=color_space,
        imageExtent=vk.VkExtent2D(width=width, height=height), imageArrayLayers=1,
        imageUsage=vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, imageSharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
        preTransform=capabilities.currentTransform, compositeAlpha=alpha,
        presentMode=vk.VK_PRESENT_MODE_FIFO_KHR, clipped=vk.VK_TRUE,
    ), None)
    resources.callback(gpu.device_function("vkDestroySwapchainKHR"), gpu.device, swapchain, None)
    render_pass, pipeline, layout = gpu.pipeline(resources, program, format, width, height, True)
    commands, finished, framebuffers = [], [], []
    for image in gpu.device_function("vkGetSwapchainImagesKHR")(gpu.device, swapchain):
        view = gpu.view(resources, image, format)
        framebuffer = gpu.framebuffer(resources, render_pass, view, width, height)
        framebuffers.append(framebuffer)
        commands.append(gpu.command(resources, render_pass, pipeline, framebuffer, width, height, program.draw,
                                    layout=layout, parameters=program.parameters))
        semaphore = vk.vkCreateSemaphore(gpu.device, vk.VkSemaphoreCreateInfo(), None)
        resources.callback(vk.vkDestroySemaphore, gpu.device, semaphore, None)
        finished.append(semaphore)
    acquired = vk.vkCreateSemaphore(gpu.device, vk.VkSemaphoreCreateInfo(), None)
    resources.callback(vk.vkDestroySemaphore, gpu.device, acquired, None)
    resources.callback(vk.vkDeviceWaitIdle, gpu.device)
    def record_frame(index, elapsed):
        if program.parameters is not None:
            gpu.command(resources, render_pass, pipeline, framebuffers[index], width, height, program.draw,
                        layout=layout, parameters=program.parameters, elapsed=elapsed, command=commands[index])

    return swapchain, commands, acquired, finished, record_frame


def show_window(program: GraphicsProgram, width: int = 800, height: int = 600, frames: int = 0,
                validation: bool = False, time: float | None = None) -> int:
    _dimensions(width, height)
    _frame_parameters(program.parameters)
    frozen_time = _time_value(time) if time is not None else None
    if program.target != "vulkan":
        raise ViewerError("Vulkan rendering requires Vulkan artifacts; use the Metal preview for DXIL.")
    if type(frames) is not int or frames < 0:
        raise ViewerError("Frame count must be a nonnegative integer (0 means until closed).")
    glfw = _binding("glfw")
    if not glfw.init():
        raise ViewerError(f"GLFW initialization failed: {glfw.get_error()}. A desktop session is required.")
    with ExitStack() as resources:
        resources.callback(glfw.terminate)
        resources.enter_context(_api_errors())
        if not glfw.vulkan_supported():
            raise ViewerError("GLFW cannot find a Vulkan loader and driver.")
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        window = glfw.create_window(width, height, "mlir-gpu | Vulkan triangle | Esc to close", None, None)
        if not window:
            raise ViewerError(f"GLFW window creation failed: {glfw.get_error()}.")
        resources.callback(glfw.destroy_window, window)
        gpu = _Vulkan(resources, window, glfw, validation)
        started = clock.monotonic()
        vk = gpu.vk
        presented = 0
        announced = False
        while not glfw.window_should_close(window) and (frames == 0 or presented < frames):
            glfw.poll_events()
            size = glfw.get_framebuffer_size(window)
            if not all(size) or glfw.get_window_attrib(window, glfw.ICONIFIED):
                glfw.wait_events_timeout(0.1)
                continue
            with ExitStack() as swap_resources:
                try:
                    swapchain, commands, acquired, finished, record_frame = _swapchain(gpu, swap_resources, program, *size)
                except vk.VkErrorOutOfDateKhr:
                    glfw.wait_events_timeout(0.1)
                    continue
                if not announced:
                    print(f"Vulkan viewer ready on {gpu.name}. Close the window or press Esc to exit.", flush=True)
                    announced = True
                recreate = False
                while not recreate and not glfw.window_should_close(window) and (frames == 0 or presented < frames):
                    glfw.poll_events()
                    if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                        glfw.set_window_should_close(window, True)
                    if (
                        glfw.window_should_close(window) or glfw.get_framebuffer_size(window) != size
                        or glfw.get_window_attrib(window, glfw.ICONIFIED)
                    ):
                        break
                    index = vk.ffi.new("uint32_t*")
                    try:
                        gpu.device_function("vkAcquireNextImageKHR")(gpu.device, swapchain, 100_000_000, acquired, vk.VK_NULL_HANDLE, index)
                    except (vk.VkTimeout, vk.VkNotReady):
                        continue
                    except vk.VkSuboptimalKhr:
                        recreate = True
                    except vk.VkErrorOutOfDateKhr:
                        break
                    image_index = index[0]
                    record_frame(image_index, frozen_time if frozen_time is not None else clock.monotonic() - started)
                    vk.vkQueueSubmit(gpu.queue, 1, [vk.VkSubmitInfo(
                        pWaitSemaphores=[acquired], pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT],
                        pCommandBuffers=[commands[image_index]], pSignalSemaphores=[finished[image_index]],
                    )], vk.VK_NULL_HANDLE)
                    try:
                        gpu.device_function("vkQueuePresentKHR")(gpu.queue, vk.VkPresentInfoKHR(
                            pWaitSemaphores=[finished[image_index]], pSwapchains=[swapchain], pImageIndices=[image_index],
                        ))
                        presented += 1
                    except vk.VkSuboptimalKhr:
                        presented += 1
                        recreate = True
                    except vk.VkErrorOutOfDateKhr:
                        recreate = True
                    # A single frame in flight keeps this preview's resource lifetime simple.
                    vk.vkQueueWaitIdle(gpu.queue)
        return presented


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a live Vulkan or DXIL-to-Metal preview window by default.",
        epilog="Use -frames 0 to keep the window open, --output for PNG, or --video for MP4.",
    )
    parser.add_argument("manifest", type=Path, help="Compiled Vulkan, DirectX, or Metal pipeline.json")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="Render an offscreen PNG instead of opening a window (new file)")
    mode.add_argument("--video", type=Path, help="Export a fixed-time Vulkan animation to MP4 (new file)")
    mode.add_argument("--frames", "-frames", type=int, default=0, help="Live window: close after N frames; 0 stays open")
    parser.add_argument("--validation", action="store_true", help="Enable Vulkan validation or Metal API/GPU validation")
    parser.add_argument("--time", type=float, help="Vulkan: fixed snapshot/live time or MP4 start time (seconds)")
    parser.add_argument("--duration", type=float, help="MP4 duration in seconds (default 4)")
    parser.add_argument("--fps", type=int, help="MP4 frames per second, 1-240 (default 30)")
    args = parser.parse_args()
    if args.video is None and (args.duration is not None or args.fps is not None):
        parser.error("--duration and --fps require --video")
    try:
        program = load_program(args.manifest)
        timing = {"time": args.time} if args.time is not None else {}
        if timing and program.target != "vulkan":
            raise ViewerError("--time for Metal previews requires the runtime adapter in #10.")
        if args.video:
            fps = args.fps if args.fps is not None else 30
            count = export_mp4(program, args.video, args.width, args.height,
                               args.duration if args.duration is not None else 4.0,
                               fps, args.time if args.time is not None else 0.0, args.validation)
            print(f"Exported {count} frames at {fps} fps: {args.video}")
        elif args.output:
            if args.output.exists():
                raise ViewerError(f"Output already exists; choose a new PNG path: {args.output}")
            render = render_offscreen if program.target == "vulkan" else render_metal_offscreen
            image = render(program, args.width, args.height, args.validation, **timing)
            write_png(args.output, image)
            print(f"Rendered {image.width}x{image.height} on {image.device}: {args.output}")
        elif program.target == "vulkan":
            count = show_window(program, args.width, args.height, args.frames, args.validation, **timing)
            print(f"Presented {count} Vulkan frames.")
        else:
            show_metal_window(program, args.width, args.height, args.frames, args.validation)
    except (ViewerError, CompileError, OSError, ValueError) as error:
        print(f"mlir-gpu-view: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
