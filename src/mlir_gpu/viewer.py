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
from collections.abc import Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .compiler import (
    COMPUTE_SCHEMA_VERSION,
    CompileError,
    PIPELINE_SCHEMA_VERSION,
    RASTER_CONVENTION,
    TargetProfile,
    Toolchain,
    _check_binary,
    _run,
    convert_dxil_to_metallib,
)
from .frontend import Dispatch, Draw
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
class ComputeProgram:
    compute: Shader
    target: str = "vulkan"
    workgroup_size: tuple[int, int, int] = (1, 1, 1)
    buffers: tuple[dict, ...] = ()
    dispatch: Dispatch = Dispatch()
    parameters: dict | None = None


@dataclass(frozen=True)
class ComputeDispatchResult:
    buffers: dict[str, bytes]
    device: str


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


def load_compute_program(manifest: Path) -> ComputeProgram:
    manifest = Path(manifest).resolve()
    data = json.loads(manifest.read_text())
    if (
        not isinstance(data, dict)
        or type(data.get("schema_version")) is not int
        or data.get("schema_version") != COMPUTE_SCHEMA_VERSION
        or data.get("target") not in ("vulkan", "directx", "metal")
        or data.get("stage") != "compute"
    ):
        raise ViewerError(
            f"Compute execution requires a schema-version {COMPUTE_SCHEMA_VERSION} "
            "Vulkan, DirectX, or Metal compute pipeline manifest."
        )

    wg = data.get("workgroup_size")
    if (
        not isinstance(wg, list)
        or len(wg) != 3
        or any(type(dim) is not int or dim < 1 for dim in wg)
    ):
        raise ViewerError("Compute manifest must declare a valid 3D positive workgroup_size.")
    workgroup_size = (wg[0], wg[1], wg[2])

    dispatch_data = data.get("dispatch")
    if not isinstance(dispatch_data, dict) or set(dispatch_data) != {"grid"}:
        raise ViewerError("Compute manifest must declare a dispatch grid.")
    grid_data = dispatch_data.get("grid")
    if (
        not isinstance(grid_data, list)
        or len(grid_data) != 3
        or any(type(dim) is not int or dim < 1 for dim in grid_data)
    ):
        raise ViewerError("Compute dispatch grid dimensions must be 3 positive integers.")
    dispatch = Dispatch(grid=(grid_data[0], grid_data[1], grid_data[2]))

    interface = data.get("interface")
    if not isinstance(interface, dict) or not isinstance(interface.get("compute"), dict):
        raise ViewerError("Compute manifest must include compute stage reflection.")
    compute_refl = interface["compute"]

    raw_buffers = data.get("buffers")
    if not isinstance(raw_buffers, list):
        raise ViewerError("Compute manifest buffers field must be a list.")
    seen_names: set[str] = set()
    seen_slots: set[tuple[int, int]] = set()
    validated_buffers: list[dict] = []
    for buf in raw_buffers:
        if not isinstance(buf, dict):
            raise ViewerError("Invalid buffer binding entry in compute manifest.")
        name = buf.get("name")
        set_idx = buf.get("set")
        binding_idx = buf.get("binding")
        access = buf.get("access")
        elem_type = buf.get("element_type")
        if not isinstance(name, str) or not name or name in seen_names:
            raise ViewerError(f"Invalid or duplicate buffer name in compute manifest: {name!r}.")
        if type(set_idx) is not int or set_idx < 0 or type(binding_idx) is not int or binding_idx < 0:
            raise ViewerError(f"Buffer {name!r} has invalid set/binding indices.")
        if (set_idx, binding_idx) in seen_slots:
            raise ViewerError(f"Duplicate buffer slot ({set_idx}, {binding_idx}) in compute manifest.")
        if access not in ("read_only", "write_only", "read_write"):
            raise ViewerError(f"Buffer {name!r} has unsupported access mode {access!r}.")
        if not isinstance(elem_type, str) or not elem_type:
            raise ViewerError(f"Buffer {name!r} is missing element_type.")
        seen_names.add(name)
        seen_slots.add((set_idx, binding_idx))
        validated_buffers.append({
            "name": name,
            "set": set_idx,
            "binding": binding_idx,
            "access": access,
            "element_type": elem_type,
        })
    validated_buffers.sort(key=lambda item: (item["set"], item["binding"]))

    parameters = None
    if "parameters" in data or "parameters" in compute_refl:
        parameters = data.get("parameters")
        try:
            verify_parameters(parameters)
            if compute_refl.get("parameters") != parameters:
                raise ValueError("Manifest parameters must match compute stage reflection.")
        except ValueError as error:
            raise ViewerError(f"Invalid compute parameter resource manifest: {error}") from error

    entries = data.get("entries")
    if not isinstance(entries, list) or len(entries) != 1 or not isinstance(entries[0], dict):
        raise ViewerError("Compute manifest must contain exactly one compute entry.")
    entry = entries[0]
    stage, entry_name, filename = entry.get("stage"), entry.get("entry_point"), entry.get("binary")
    if stage != "compute":
        raise ViewerError("Compute manifest entry must have stage 'compute'.")
    if not isinstance(entry_name, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", entry_name):
        raise ViewerError("Invalid compute shader entry-point name.")
    if compute_refl.get("name") != entry_name:
        raise ViewerError("Compute entry-point name must match compiler reflection.")
    if not isinstance(filename, str) or not filename:
        raise ViewerError("A compute shader binary path is required.")
    binary = (manifest.parent / filename).resolve()
    if Path(filename).is_absolute() or not binary.is_relative_to(manifest.parent):
        raise ViewerError("Shader binaries must be inside the pipeline directory.")
    magic, minimum = {
        "vulkan": (b"\x03\x02\x23\x07", 20),
        "directx": (b"DXBC", 32),
        "metal": (b"MTLB", 16),
    }[data["target"]]
    _check_binary(binary, magic, minimum)
    code = binary.read_bytes()
    if data["target"] == "vulkan" and len(code) % 4:
        raise ViewerError(f"SPIR-V binary is not word-aligned: {filename}.")

    return ComputeProgram(
        compute=Shader(entry_name, code),
        target=data["target"],
        workgroup_size=workgroup_size,
        buffers=tuple(validated_buffers),
        dispatch=dispatch,
        parameters=parameters,
    )


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
    _frame_parameters(program.parameters)
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
        arguments = [toolchain.find("mlir-gpu-metal-view"), *libraries, "--vertex-count", str(program.draw.vertex_count)]
        if program.parameters is not None:
            layout = directory / "frame-layout.json"
            layout.write_text(json.dumps(program.parameters, allow_nan=False))
            arguments.extend(["--frame-layout", str(layout)])
        yield directory, arguments


def _native_run(arguments: list[str], capture: bool, backend: str, environment: dict[str, str] | None = None) -> str:
    try:
        result = subprocess.run(arguments, text=True, capture_output=capture, env=environment, check=False)
    except OSError as error:
        raise ViewerError(f"Cannot start the native {backend} preview: {error}") from error
    if result.returncode:
        diagnostics = (result.stderr or result.stdout or "").strip() if capture else "See native renderer diagnostics above."
        raise ViewerError(f"Native {backend} preview failed ({result.returncode}): {diagnostics}")
    return result.stdout or ""


def _metal_run(arguments: list[str], validation: bool, capture: bool) -> str:
    environment = os.environ.copy()
    if validation:
        environment.update(MTL_DEBUG_LAYER="1", MTL_SHADER_VALIDATION="1")
    return _native_run(arguments, capture, "Metal", environment)


VALID_PROVIDERS = {"hardware", "warp", "lavapipe", "swiftshader"}


def _validate_provider(provider: str | None, backend: str) -> None:
    if provider is None:
        return
    if not isinstance(provider, str) or provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unknown provider '{provider}'. Supported providers are: {', '.join(sorted(VALID_PROVIDERS))}."
        )
    if backend == "directx" and provider in ("lavapipe", "swiftshader"):
        raise ValueError(
            f"Provider '{provider}' is not supported by the DirectX 12 backend. Use 'hardware' or 'warp'."
        )
    if backend == "vulkan" and provider == "warp":
        raise ValueError(
            "Provider 'warp' is not supported by the Vulkan backend. Use 'lavapipe', 'swiftshader', or 'hardware'."
        )
    if backend == "metal":
        raise ValueError(f"Provider '{provider}' is not supported by the Metal backend.")


def _match_vulkan_device(properties, provider: str | None) -> bool:
    if provider is None:
        return True
    raw_name = getattr(properties, "deviceName", b"")
    name = raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)
    name_lower = name.lower()
    vendor_id = getattr(properties, "vendorID", 0)
    device_id = getattr(properties, "deviceID", 0)
    device_type = getattr(properties, "deviceType", 0)

    if provider == "lavapipe":
        return vendor_id == 0x10005 or "llvmpipe" in name_lower or "lavapipe" in name_lower
    elif provider == "swiftshader":
        return vendor_id == 0x1AE0 or device_id == 0xC0DE or "swiftshader" in name_lower
    elif provider == "hardware":
        is_cpu = device_type == 4  # VK_PHYSICAL_DEVICE_TYPE_CPU
        is_gpu = device_type in (1, 2)  # INTEGRATED_GPU, DISCRETE_GPU
        is_sw_vendor = vendor_id in (0x10005, 0x1AE0) or device_id == 0xC0DE
        is_sw_name = "llvmpipe" in name_lower or "lavapipe" in name_lower or "swiftshader" in name_lower
        return is_gpu and not is_cpu and not is_sw_vendor and not is_sw_name
    return False


def _directx_run(arguments: list[str], validation: bool, capture: bool) -> str:
    return _native_run([*arguments, *(["--validation"] if validation else [])], capture, "DirectX 12")


def _preview_toolchain() -> Toolchain:
    directory = os.environ.get("GPU_TOOLS_BIN")
    return Toolchain(tools_bin=Path(directory) if directory else None)


@contextmanager
def _directx_shaders(program: GraphicsProgram, toolchain: Toolchain):
    if sys.platform != "win32":
        raise ViewerError("Native DirectX 12 preview requires Windows.")
    if program.target != "directx":
        raise ViewerError("The DirectX 12 viewer requires DirectX artifacts.")
    _frame_parameters(program.parameters)
    with tempfile.TemporaryDirectory(prefix="mlir-gpu-directx-") as work:
        directory = Path(work)
        arguments = [toolchain.find("mlir-gpu-directx-view")]
        for stage, shader in (("vertex", program.vertex), ("fragment", program.fragment)):
            path = directory / f"{stage}.dxil"
            path.write_bytes(shader.code)
            arguments.extend([f"--{stage}", str(path)])
        arguments.extend(["--vertex-count", str(program.draw.vertex_count)])
        yield directory, arguments


def _readback_result(text: str, output: Path, width: int, height: int, backend: str) -> RenderedImage:
    report = json.loads(text)
    if (
        not isinstance(report, dict) or type(report.get("width")) is not int
        or type(report.get("height")) is not int or report["width"] != width or report["height"] != height
        or not isinstance(report.get("device"), str) or not report["device"]
    ):
        raise ViewerError(f"The native {backend} renderer returned an invalid render report.")
    rgba = output.read_bytes()
    if len(rgba) != width * height * 4:
        raise ViewerError(f"{backend} RGBA readback size does not match the requested dimensions.")
    return RenderedImage(width, height, rgba, report["device"])


@contextmanager
def _directx_offscreen_renderer(
    program: GraphicsProgram, width: int, height: int, validation: bool, toolchain: Toolchain | None = None,
    provider: str | None = None,
):
    _dimensions(width, height)
    explicit_warp = provider == "warp"
    if provider is None and os.environ.get("MLIR_GPU_TEST_WARP") == "1":
        provider = "warp"
    _validate_provider(provider, "directx")
    with _directx_shaders(program, toolchain or _preview_toolchain()) as (directory, arguments):
        output = directory / "readback.rgba"
        warp_args = ["--warp"] if provider == "warp" else []
        def render(elapsed: float) -> RenderedImage:
            timing = ["--time", str(_time_value(elapsed))]
            text = _directx_run(
                [*arguments, *warp_args, "--width", str(width), "--height", str(height), "--readback", str(output), *timing],
                validation, True,
            )
            image = _readback_result(text, output, width, height, "DirectX 12")
            if explicit_warp and "basic render driver" not in image.device.lower():
                raise ViewerError(
                    "Requested provider 'warp' but returned device is not a software adapter. "
                    "Accidental hardware/driver fallback rejected."
                )
            if provider == "hardware" and "basic render driver" in image.device.lower():
                raise ViewerError(
                    "Requested provider 'hardware' but returned device is a software adapter. "
                    "Accidental hardware/driver fallback rejected."
                )
            return image
        yield render


def render_directx_offscreen(
    program: GraphicsProgram, width: int = 640, height: int = 480,
    validation: bool = False, toolchain: Toolchain | None = None, time: float = 0.0,
    provider: str | None = None,
) -> RenderedImage:
    elapsed = _time_value(time)
    kwargs = {"provider": provider} if provider is not None else {}
    with _directx_offscreen_renderer(program, width, height, validation, toolchain, **kwargs) as render:
        return render(elapsed)


def show_directx_window(
    program: GraphicsProgram, width: int = 800, height: int = 600, frames: int = 0,
    validation: bool = False, toolchain: Toolchain | None = None, time: float | None = None,
    provider: str | None = None,
) -> None:
    _dimensions(width, height)
    _validate_provider(provider, "directx")
    if type(frames) is not int or frames < 0:
        raise ViewerError("Frame count must be a nonnegative integer (0 means until closed).")
    timing = ["--time", str(_time_value(time))] if time is not None else []
    warp_args = ["--warp"] if provider == "warp" else []
    with _directx_shaders(program, toolchain or _preview_toolchain()) as (_, arguments):
        _directx_run(
            [*arguments, *warp_args, "--width", str(width), "--height", str(height), "--frames", str(frames), *timing],
            validation, False,
        )


def render_metal_offscreen(
    program: GraphicsProgram, width: int = 640, height: int = 480,
    validation: bool = False, toolchain: Toolchain | None = None, time: float = 0.0,
) -> RenderedImage:
    elapsed = _time_value(time)
    with _metal_offscreen_renderer(program, width, height, validation, toolchain) as render:
        return render(elapsed)


@contextmanager
def _metal_offscreen_renderer(
    program: GraphicsProgram, width: int, height: int, validation: bool, toolchain: Toolchain | None = None,
):
    _dimensions(width, height)
    with _metal_libraries(program, toolchain or _preview_toolchain()) as (directory, arguments):
        output = directory / "readback.rgba"
        def render(elapsed: float) -> RenderedImage:
            timing = ["--time", str(_time_value(elapsed))] if elapsed != 0 else []
            text = _metal_run(
                [*arguments, "--width", str(width), "--height", str(height), "--readback", str(output), *timing],
                validation, True,
            )
            return _readback_result(text, output, width, height, "Metal")
        yield render


def show_metal_window(
    program: GraphicsProgram, width: int = 800, height: int = 600, frames: int = 0,
    validation: bool = False, toolchain: Toolchain | None = None, time: float | None = None,
) -> None:
    _dimensions(width, height)
    if type(frames) is not int or frames < 0:
        raise ViewerError("Frame count must be a nonnegative integer (0 means until closed).")
    timing = ["--time", str(_time_value(time))] if time is not None else []
    with _metal_libraries(program, toolchain or _preview_toolchain()) as (_, arguments):
        _metal_run(
            [*arguments, "--width", str(width), "--height", str(height), "--frames", str(frames), *timing],
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
    def __init__(
        self,
        resources: ExitStack,
        window=None,
        glfw=None,
        validation: bool = False,
        provider: str | None = None,
        compute_only: bool = False,
    ):
        if provider is None and os.environ.get("MLIR_GPU_TEST_SWIFTSHADER") == "1":
            provider = "swiftshader"
        _validate_provider(provider, "vulkan")
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
            swiftshader_env = (
                provider == "swiftshader"
                or os.environ.get("MLIR_GPU_TEST_SWIFTSHADER") == "1"
                or "swiftshader" in (os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES") or "").lower()
            )
            if "VK_LAYER_KHRONOS_validation" not in available_layers:
                if not swiftshader_env:
                    raise ViewerError("--validation requires VK_LAYER_KHRONOS_validation from the Vulkan SDK.")
            elif "VK_EXT_debug_utils" not in available:
                if not swiftshader_env:
                    raise ViewerError("--validation requires the VK_EXT_debug_utils extension.")
            else:
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

        required_queue = vk.VK_QUEUE_COMPUTE_BIT if compute_only else vk.VK_QUEUE_GRAPHICS_BIT
        candidates = []
        for physical in vk.vkEnumeratePhysicalDevices(self.instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            for index, queue in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)):
                if not queue.queueCount or not queue.queueFlags & required_queue:
                    continue
                device_extensions = {item.extensionName for item in vk.vkEnumerateDeviceExtensionProperties(physical, None)}
                if self.surface is not None:
                    if "VK_KHR_swapchain" not in device_extensions:
                        continue
                    if not self.instance_function("vkGetPhysicalDeviceSurfaceSupportKHR")(physical, index, self.surface):
                        continue
                is_cpu = 1 if properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_CPU else 0
                is_not_discrete = 0 if properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1
                candidates.append((is_cpu, is_not_discrete, physical, index, properties, device_extensions))
                break

        if not candidates:
            if provider is not None:
                raise ViewerError(
                    f"Requested provider '{provider}' was not found among available Vulkan physical devices. "
                    "Accidental hardware/driver fallback rejected."
                )
            queue_kind = "compute" if compute_only else "graphics/presentation"
            raise ViewerError(f"No Vulkan device has a suitable {queue_kind} queue.")

        if provider is not None:
            matching = [c for c in candidates if _match_vulkan_device(c[4], provider)]
            if not matching:
                raise ViewerError(
                    f"Requested provider '{provider}' was not found among available Vulkan physical devices. "
                    "Accidental hardware/driver fallback rejected."
                )
            selected = sorted(matching, key=lambda c: (c[0], c[1]))[0]
        else:
            selected = sorted(candidates, key=lambda c: (c[0], c[1]))[0]

        _, _, self.physical, self.family, properties, device_extensions = selected
        raw_name = properties.deviceName
        self.name = (raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)).rstrip("\x00")
        self.max_width = properties.limits.maxFramebufferWidth
        self.max_height = properties.limits.maxFramebufferHeight
        self.max_push_constants = properties.limits.maxPushConstantsSize
        self.max_compute_workgroup_count = tuple(int(v) for v in properties.limits.maxComputeWorkGroupCount)
        self.max_compute_workgroup_size = tuple(int(v) for v in properties.limits.maxComputeWorkGroupSize)
        self.max_compute_workgroup_invocations = int(properties.limits.maxComputeWorkGroupInvocations)
        enabled = ["VK_KHR_swapchain"] if self.surface is not None else []
        if "VK_KHR_portability_subset" in device_extensions:
            enabled.append("VK_KHR_portability_subset")
        if "VK_KHR_storage_buffer_storage_class" in device_extensions:
            enabled.append("VK_KHR_storage_buffer_storage_class")
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

    def storage_buffer(self, resources, size: int, initial_data: bytes | None = None):
        vk = self.vk
        alloc_size = max(4, (size + 3) & ~3)
        buffer = vk.vkCreateBuffer(self.device, vk.VkBufferCreateInfo(
            size=alloc_size,
            usage=vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
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
        mapped = vk.vkMapMemory(self.device, memory, 0, alloc_size, 0)
        try:
            if initial_data:
                data_len = len(initial_data)
                mapped[0:data_len] = initial_data
                if data_len < alloc_size:
                    mapped[data_len:alloc_size] = b"\x00" * (alloc_size - data_len)
            else:
                mapped[0:alloc_size] = b"\x00" * alloc_size
        finally:
            vk.vkUnmapMemory(self.device, memory)
        return buffer, memory, alloc_size

    def compute_pipeline(self, resources, program: ComputeProgram):
        vk = self.vk
        wgx, wgy, wgz = program.workgroup_size
        if (
            wgx > self.max_compute_workgroup_size[0]
            or wgy > self.max_compute_workgroup_size[1]
            or wgz > self.max_compute_workgroup_size[2]
            or wgx * wgy * wgz > self.max_compute_workgroup_invocations
        ):
            raise ViewerError("The compute kernel's workgroup_size exceeds the Vulkan device limits.")

        set_layouts = []
        if program.buffers:
            max_set = max(buf["set"] for buf in program.buffers)
            for set_idx in range(max_set + 1):
                bindings = [
                    vk.VkDescriptorSetLayoutBinding(
                        binding=buf["binding"],
                        descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                        descriptorCount=1,
                        stageFlags=vk.VK_SHADER_STAGE_COMPUTE_BIT,
                    )
                    for buf in program.buffers
                    if buf["set"] == set_idx
                ]
                set_layout = vk.vkCreateDescriptorSetLayout(
                    self.device,
                    vk.VkDescriptorSetLayoutCreateInfo(pBindings=bindings),
                    None,
                )
                resources.callback(vk.vkDestroyDescriptorSetLayout, self.device, set_layout, None)
                set_layouts.append(set_layout)

        ranges = []
        if program.parameters is not None:
            if program.parameters["size"] > self.max_push_constants:
                raise ViewerError("The reflected compute parameter block exceeds the device's push-constant limit.")
            ranges.append(vk.VkPushConstantRange(
                stageFlags=self.parameter_stages(program.parameters),
                offset=program.parameters["bindings"]["vulkan"]["offset"],
                size=program.parameters["size"],
            ))

        layout = vk.vkCreatePipelineLayout(
            self.device,
            vk.VkPipelineLayoutCreateInfo(pSetLayouts=set_layouts, pPushConstantRanges=ranges),
            None,
        )
        resources.callback(vk.vkDestroyPipelineLayout, self.device, layout, None)

        module = vk.vkCreateShaderModule(
            self.device,
            vk.VkShaderModuleCreateInfo(codeSize=len(program.compute.code), pCode=program.compute.code),
            None,
        )
        resources.callback(vk.vkDestroyShaderModule, self.device, module, None)
        stage_info = vk.VkPipelineShaderStageCreateInfo(
            stage=vk.VK_SHADER_STAGE_COMPUTE_BIT,
            module=module,
            pName=program.compute.entry_point,
        )
        create_info = vk.VkComputePipelineCreateInfo(stage=stage_info, layout=layout)
        pipeline = vk.vkCreateComputePipelines(self.device, vk.VK_NULL_HANDLE, 1, [create_info], None)[0]
        resources.callback(vk.vkDestroyPipeline, self.device, pipeline, None)
        return pipeline, layout, set_layouts

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
            flags |= {
                "vertex": self.vk.VK_SHADER_STAGE_VERTEX_BIT,
                "fragment": self.vk.VK_SHADER_STAGE_FRAGMENT_BIT,
                "compute": self.vk.VK_SHADER_STAGE_COMPUTE_BIT,
            }[stage]
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


def render_offscreen(
    program: GraphicsProgram, width: int = 640, height: int = 480,
    validation: bool = False, time: float = 0.0, provider: str | None = None,
) -> RenderedImage:
    elapsed = _time_value(time)
    kwargs = {"provider": provider} if provider is not None else {}
    with _offscreen_renderer(program, width, height, validation, **kwargs) as render:
        return render(elapsed)


@contextmanager
def _offscreen_renderer(
    program: GraphicsProgram, width: int, height: int, validation: bool, provider: str | None = None,
):
    _dimensions(width, height)
    _frame_parameters(program.parameters)
    if program.target == "directx" and sys.platform != "darwin":
        kwargs = {"provider": provider} if provider is not None else {}
        with _directx_offscreen_renderer(program, width, height, validation, **kwargs) as render:
            yield render
        return
    if program.target == "metal":
        if provider is not None:
            raise ValueError(f"Provider '{provider}' is not supported by the Metal backend.")
        with _metal_offscreen_renderer(program, width, height, validation) as render:
            yield render
        return
    if program.target != "vulkan":
        raise ViewerError("Vulkan rendering requires Vulkan artifacts; use the Metal preview for DXIL.")
    vulkan_kwargs = {"provider": provider} if provider is not None else {}
    with _api_errors(), ExitStack() as resources:
        gpu = _Vulkan(resources, validation=validation, **vulkan_kwargs)
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


def render_vulkan_offscreen(
    program: GraphicsProgram, width: int = 640, height: int = 480,
    validation: bool = False, time: float = 0.0, provider: str | None = None,
) -> RenderedImage:
    _validate_provider(provider, "vulkan")
    if program.target != "vulkan":
        raise ViewerError("Vulkan rendering requires Vulkan artifacts; use the Metal preview for DXIL.")
    return render_offscreen(program, width, height, validation, time=time, provider=provider)


@contextmanager
def _vulkan_offscreen_renderer(
    program: GraphicsProgram, width: int, height: int, validation: bool, provider: str | None = None,
):
    _validate_provider(provider, "vulkan")
    if program.target != "vulkan":
        raise ViewerError("Vulkan rendering requires Vulkan artifacts; use the Metal preview for DXIL.")
    with _offscreen_renderer(program, width, height, validation, provider=provider) as render:
        yield render


def _prepare_compute_inputs(
    program: ComputeProgram,
    buffers: Mapping[str, bytes | bytearray | memoryview | int] | None,
    parameters: Mapping[str, object] | bytes | bytearray | memoryview | None,
    grid: tuple[int, int, int] | None,
) -> tuple[tuple[int, int, int], bytes | None, list[tuple[dict, int, bytes | None]]]:
    active_grid = program.dispatch.grid if grid is None else tuple(grid)
    if (
        len(active_grid) != 3
        or any(type(dim) is not int or dim < 1 for dim in active_grid)
    ):
        raise ViewerError("Compute dispatch grid must contain 3 positive integers.")

    packed_params: bytes | None = None
    if program.parameters is None:
        if parameters is not None:
            raise ViewerError("The compute pipeline declares no parameter block, so parameters must be None.")
    else:
        if parameters is None:
            raise ViewerError(f"Compute parameter block {program.parameters['name']!r} requires host values.")
        if isinstance(parameters, (bytes, bytearray, memoryview)):
            packed_params = bytes(parameters)
            if len(packed_params) != program.parameters["size"]:
                raise ViewerError(
                    f"Packed parameter bytes length ({len(packed_params)}) does not match "
                    f"reflected size ({program.parameters['size']})."
                )
        elif isinstance(parameters, Mapping):
            try:
                packed_params = pack_parameters(program.parameters, dict(parameters))
            except (TypeError, ValueError) as error:
                raise ViewerError(f"Invalid compute parameter values: {error}") from error
        else:
            raise ViewerError("Compute parameters must be a field mapping or packed byte buffer.")

    provided = dict(buffers or {})
    reflected_names = {buf["name"] for buf in program.buffers}
    extra = sorted(set(provided) - reflected_names)
    if extra:
        raise ViewerError(f"Unknown compute buffer name(s): {', '.join(extra)}.")
    missing = sorted(reflected_names - set(provided))
    if missing:
        raise ViewerError(f"Missing required compute buffer binding(s): {', '.join(missing)}.")

    prepared: list[tuple[dict, int, bytes | None]] = []
    for buf in program.buffers:
        name = buf["name"]
        access = buf["access"]
        spec = provided[name]
        if type(spec) is int:
            if spec <= 0:
                raise ViewerError(f"Compute buffer {name!r} byte size must be positive.")
            prepared.append((buf, spec, None))
        elif isinstance(spec, (bytes, bytearray, memoryview)):
            raw = bytes(spec)
            if len(raw) == 0:
                raise ViewerError(f"Compute buffer {name!r} byte buffer must be non-empty.")
            initial = None if access == "write_only" else raw
            prepared.append((buf, len(raw), initial))
        else:
            raise ViewerError(
                f"Compute buffer {name!r} must be a byte buffer (bytes/bytearray/memoryview) or positive int byte size."
            )

    return (active_grid[0], active_grid[1], active_grid[2]), packed_params, prepared


@contextmanager
def _vulkan_compute_session(
    program: ComputeProgram,
    validation: bool = False,
    provider: str | None = None,
):
    _validate_provider(provider, "vulkan")
    if program.target != "vulkan":
        raise ViewerError("Vulkan compute execution requires Vulkan SPIR-V artifacts.")
    vulkan_kwargs = {"provider": provider} if provider is not None else {}
    with _api_errors(), ExitStack() as resources:
        gpu = _Vulkan(resources, validation=validation, compute_only=True, **vulkan_kwargs)
        vk = gpu.vk
        resources.callback(vk.vkDeviceWaitIdle, gpu.device)
        pipeline, layout, set_layouts = gpu.compute_pipeline(resources, program)
        fence = vk.vkCreateFence(gpu.device, vk.VkFenceCreateInfo(), None)
        resources.callback(vk.vkDestroyFence, gpu.device, fence, None)

        def run(
            *,
            buffers: Mapping[str, bytes | bytearray | memoryview | int] | None = None,
            parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
            grid: tuple[int, int, int] | None = None,
        ) -> ComputeDispatchResult:
            active_grid, packed_params, prepared = _prepare_compute_inputs(program, buffers, parameters, grid)
            gx, gy, gz = active_grid
            if (
                gx > gpu.max_compute_workgroup_count[0]
                or gy > gpu.max_compute_workgroup_count[1]
                or gz > gpu.max_compute_workgroup_count[2]
            ):
                raise ViewerError("Compute dispatch grid exceeds the Vulkan device maxComputeWorkGroupCount limits.")

            with ExitStack() as step_resources:
                step_resources.callback(vk.vkDeviceWaitIdle, gpu.device)
                allocated: list[tuple[dict, int, object, object, int]] = []
                for buf_meta, byte_size, initial_data in prepared:
                    vk_buf, vk_mem, alloc_size = gpu.storage_buffer(step_resources, byte_size, initial_data)
                    allocated.append((buf_meta, byte_size, vk_buf, vk_mem, alloc_size))

                descriptor_sets = []
                if set_layouts:
                    pool_sizes = [
                        vk.VkDescriptorPoolSize(
                            type=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            descriptorCount=len(allocated),
                        )
                    ]
                    desc_pool = vk.vkCreateDescriptorPool(
                        gpu.device,
                        vk.VkDescriptorPoolCreateInfo(
                            maxSets=len(set_layouts),
                            pPoolSizes=pool_sizes,
                        ),
                        None,
                    )
                    step_resources.callback(vk.vkDestroyDescriptorPool, gpu.device, desc_pool, None)
                    alloc_info = vk.VkDescriptorSetAllocateInfo(
                        descriptorPool=desc_pool,
                        descriptorSetCount=len(set_layouts),
                        pSetLayouts=set_layouts,
                    )
                    descriptor_sets = list(vk.vkAllocateDescriptorSets(gpu.device, alloc_info))

                    buf_infos = []
                    writes = []
                    for buf_meta, _, vk_buf, _, alloc_size in allocated:
                        info = vk.VkDescriptorBufferInfo(buffer=vk_buf, offset=0, range=alloc_size)
                        buf_infos.append(info)
                        writes.append(vk.VkWriteDescriptorSet(
                            dstSet=descriptor_sets[buf_meta["set"]],
                            dstBinding=buf_meta["binding"],
                            dstArrayElement=0,
                            descriptorCount=1,
                            descriptorType=vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                            pBufferInfo=[info],
                        ))
                    vk.vkUpdateDescriptorSets(gpu.device, len(writes), writes, 0, None)

                command = vk.vkAllocateCommandBuffers(gpu.device, vk.VkCommandBufferAllocateInfo(
                    commandPool=gpu.pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ))[0]
                step_resources.callback(vk.vkFreeCommandBuffers, gpu.device, gpu.pool, 1, [command])

                vk.vkBeginCommandBuffer(command, vk.VkCommandBufferBeginInfo())
                if allocated:
                    host_barrier = vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_HOST_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_SHADER_READ_BIT | vk.VK_ACCESS_SHADER_WRITE_BIT,
                    )
                    vk.vkCmdPipelineBarrier(
                        command,
                        vk.VK_PIPELINE_STAGE_HOST_BIT,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        0,
                        1,
                        [host_barrier],
                        0,
                        None,
                        0,
                        None,
                    )
                vk.vkCmdBindPipeline(command, vk.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline)
                if descriptor_sets:
                    vk.vkCmdBindDescriptorSets(
                        command,
                        vk.VK_PIPELINE_BIND_POINT_COMPUTE,
                        layout,
                        0,
                        len(descriptor_sets),
                        descriptor_sets,
                        0,
                        None,
                    )
                if packed_params is not None and program.parameters is not None:
                    vk.vkCmdPushConstants(
                        command,
                        layout,
                        gpu.parameter_stages(program.parameters),
                        program.parameters["bindings"]["vulkan"]["offset"],
                        len(packed_params),
                        vk.ffi.from_buffer(packed_params),
                    )
                vk.vkCmdDispatch(command, gx, gy, gz)
                if allocated:
                    readback_barrier = vk.VkMemoryBarrier(
                        srcAccessMask=vk.VK_ACCESS_SHADER_WRITE_BIT,
                        dstAccessMask=vk.VK_ACCESS_HOST_READ_BIT,
                    )
                    vk.vkCmdPipelineBarrier(
                        command,
                        vk.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                        vk.VK_PIPELINE_STAGE_HOST_BIT,
                        0,
                        1,
                        [readback_barrier],
                        0,
                        None,
                        0,
                        None,
                    )
                vk.vkEndCommandBuffer(command)

                vk.vkResetFences(gpu.device, 1, [fence])
                submit_info = vk.VkSubmitInfo(pCommandBuffers=[command])
                vk.vkQueueSubmit(gpu.queue, 1, [submit_info], fence)
                vk.vkWaitForFences(gpu.device, 1, [fence], vk.VK_TRUE, 30_000_000_000)

                outputs: dict[str, bytes] = {}
                for buf_meta, byte_size, _, vk_mem, alloc_size in allocated:
                    if buf_meta["access"] in ("write_only", "read_write"):
                        mapped = vk.vkMapMemory(gpu.device, vk_mem, 0, alloc_size, 0)
                        try:
                            outputs[buf_meta["name"]] = bytes(mapped[:byte_size])
                        finally:
                            vk.vkUnmapMemory(gpu.device, vk_mem)
                gpu.check_validation()
                return ComputeDispatchResult(buffers=outputs, device=gpu.name)

        yield run


@contextmanager
def compute_session(
    program: ComputeProgram | Path,
    *,
    validation: bool = False,
    provider: str | None = None,
    toolchain: Toolchain | None = None,
):
    loaded = load_compute_program(program) if isinstance(program, (Path, str)) else program
    if not isinstance(loaded, ComputeProgram):
        raise TypeError("compute_session requires a ComputeProgram or manifest Path.")
    if loaded.target == "vulkan":
        with _vulkan_compute_session(loaded, validation=validation, provider=provider) as run:
            yield run
        return
    raise ViewerError(f"Compute dispatch is not yet supported for target {loaded.target!r}.")


def dispatch_compute(
    program: ComputeProgram | Path,
    *,
    buffers: Mapping[str, bytes | bytearray | memoryview | int] | None = None,
    parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
    grid: tuple[int, int, int] | None = None,
    validation: bool = False,
    provider: str | None = None,
    toolchain: Toolchain | None = None,
) -> ComputeDispatchResult:
    with compute_session(program, validation=validation, provider=provider, toolchain=toolchain) as run:
        return run(buffers=buffers, parameters=parameters, grid=grid)


def render_compute_rgba(
    program: ComputeProgram | Path,
    width: int,
    height: int,
    *,
    output_buffer: str = "output_buffer",
    buffers: Mapping[str, bytes | bytearray | memoryview | int] | None = None,
    parameters: Mapping[str, object] | bytes | bytearray | memoryview | None = None,
    grid: tuple[int, int, int] | None = None,
    validation: bool = False,
    provider: str | None = None,
    toolchain: Toolchain | None = None,
) -> RenderedImage:
    _dimensions(width, height)
    loaded = load_compute_program(program) if isinstance(program, (Path, str)) else program
    if not isinstance(loaded, ComputeProgram):
        raise TypeError("render_compute_rgba requires a ComputeProgram or manifest Path.")
    expected_bytes = width * height * 4
    merged_buffers = dict(buffers or {})
    merged_buffers[output_buffer] = expected_bytes
    if grid is None:
        wgx, wgy, _ = loaded.workgroup_size
        active_grid = ((width + wgx - 1) // wgx, (height + wgy - 1) // wgy, 1)
    else:
        active_grid = grid
    result = dispatch_compute(
        loaded,
        buffers=merged_buffers,
        parameters=parameters,
        grid=active_grid,
        validation=validation,
        provider=provider,
        toolchain=toolchain,
    )
    if output_buffer not in result.buffers:
        raise ViewerError(f"Output buffer {output_buffer!r} is not a writable compute buffer.")
    rgba = result.buffers[output_buffer]
    if len(rgba) != expected_bytes:
        raise ViewerError("Compute RGBA output buffer size does not match requested image dimensions.")
    return RenderedImage(width=width, height=height, rgba=rgba, device=result.device)


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
               validation: bool = False, provider: str | None = None) -> int:
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
    if program.target not in ("vulkan", "directx", "metal"):
        raise ViewerError("MP4 export requires Vulkan, DirectX, or Metal artifacts.")
    _validate_provider(provider, program.target)
    if program.target == "directx" and sys.platform not in ("win32", "darwin"):
        raise ViewerError(
            "Native DirectX 12 MP4 export requires Windows; "
            "use Vulkan or DXIL-to-Metal on macOS for video export."
        )
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
                renderer = (
                    _offscreen_renderer if program.target == "vulkan" else
                    _directx_offscreen_renderer if program.target == "directx" and sys.platform != "darwin" else
                    _metal_offscreen_renderer
                )
                renderer_kwargs = {"provider": provider} if provider is not None else {}
                with renderer(program, width, height, validation, **renderer_kwargs) as render:
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
        except (PermissionError, OSError):
            if path.exists():
                raise ViewerError(f"Output already exists; choose a new MP4 path: {path}")
            moved = False
            if sys.platform == "win32":
                try:
                    os.rename(output, path)
                    moved = True
                except FileExistsError as error:
                    raise ViewerError(f"Output already exists; choose a new MP4 path: {path}") from error
                except OSError:
                    moved = False
            if not moved:
                try:
                    with open(path, "xb") as out_file, output.open("rb") as in_file:
                        shutil.copyfileobj(in_file, out_file)
                except FileExistsError as error:
                    raise ViewerError(f"Output already exists; choose a new MP4 path: {path}") from error
                except Exception as error:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass
                    raise ViewerError(f"Failed to publish MP4 output to {path}: {error}") from error
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
                validation: bool = False, time: float | None = None, provider: str | None = None) -> int:
    _dimensions(width, height)
    _frame_parameters(program.parameters)
    _validate_provider(provider, "vulkan")
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
        vulkan_kwargs = {"provider": provider} if provider is not None else {}
        gpu = _Vulkan(resources, window, glfw, validation, **vulkan_kwargs)
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


def _parse_version_tuple(val: str) -> tuple[int, ...]:
    cleaned = val.replace("feature_level_", "").replace("shader_model_", "").replace("_", ".").replace(":", ".")
    parts = []
    for part in cleaned.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            pass
    return tuple(parts)


def get_device_capabilities(
    backend: str = "directx",
    provider: str | None = None,
    toolchain: Toolchain | None = None,
    required_capabilities: list[str] | set[str] | None = None,
    target_profile: TargetProfile | dict | None = None,
) -> dict:
    if backend not in ("directx", "vulkan", "metal"):
        raise ValueError(f"Unknown backend '{backend}'. Supported backends are: 'directx', 'vulkan', 'metal'.")
    _validate_provider(provider, backend)

    profile_obj: TargetProfile | None = None
    if target_profile is not None:
        profile_obj = (
            target_profile
            if isinstance(target_profile, TargetProfile)
            else TargetProfile.from_dict(target_profile)
        )
        req_list = list(required_capabilities or [])
        for cap in profile_obj.required_capabilities:
            if cap not in req_list:
                req_list.append(cap)
        required_capabilities = req_list

    if backend == "metal":
        if sys.platform != "darwin":
            return {
                "target": "metal",
                "backend": "metal",
                "provider": "hardware",
                "status": "UNSUPPORTED",
                "is_software": False,
                "device_name": "Metal (unsupported on this platform)",
                "missing_capabilities": ["darwin_platform"],
                "unverified_capabilities": [],
            }

    if backend == "directx":
        if sys.platform != "win32":
            raise ViewerError("Native DirectX 12 preview requires Windows.")
        if provider is None and os.environ.get("MLIR_GPU_TEST_WARP") == "1":
            provider = "warp"
        tc = toolchain or _preview_toolchain()
        viewer_bin = tc.find("mlir-gpu-directx-view")
        arguments = [str(viewer_bin), "--report-capabilities"]
        if provider == "warp":
            arguments.append("--warp")
        try:
            result = subprocess.run(arguments, capture_output=True, text=True, check=False)
        except OSError as error:
            raise ViewerError(f"Cannot start DirectX 12 capability query: {error}") from error
        if result.returncode != 0:
            diagnostics = (result.stderr or result.stdout or "").strip()
            raise ViewerError(f"DirectX 12 capability query failed ({result.returncode}): {diagnostics}")
        try:
            data = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise ViewerError(f"Invalid JSON returned by DirectX 12 capability query: {result.stdout}") from error

        device_name = str(data.get("device_name") or data.get("adapter_name", ""))
        is_software = (
            bool(data.get("is_software"))
            or bool(data.get("software_adapter"))
            or int(data.get("vendor_id", 0)) == 5140
            or "basic render driver" in device_name.lower()
        )
        if provider == "warp" and not is_software:
            raise ViewerError(
                "Requested provider 'warp' but returned device is not a software adapter. "
                "Accidental hardware/driver fallback rejected."
            )
        if provider == "hardware" and is_software:
            raise ViewerError(
                "Requested provider 'hardware' but returned device is a software adapter. "
                "Accidental hardware/driver fallback rejected."
            )

        feature_level = str(data.get("d3d12_feature_level") or data.get("feature_level", ""))
        shader_model = str(data.get("shader_model") or data.get("highest_shader_model", ""))
        missing_dx = []
        unverified_dx = []
        if required_capabilities:
            device_fl_tuple = _parse_version_tuple(feature_level)
            device_sm_tuple = _parse_version_tuple(shader_model)
            for cap in required_capabilities:
                if cap in (feature_level, shader_model):
                    continue
                if "feature_level" in cap or cap in ("11_0", "11_1", "12_0", "12_1", "12_2"):
                    req_fl = _parse_version_tuple(cap)
                    if req_fl and device_fl_tuple >= req_fl:
                        continue
                    missing_dx.append(cap)
                elif "shader_model" in cap or (len(cap) == 3 and cap[1] == "_" and cap[0] == "6"):
                    req_sm = _parse_version_tuple(cap)
                    if req_sm and device_sm_tuple >= req_sm:
                        continue
                    missing_dx.append(cap)
                elif cap in ("warp", "software") and not is_software:
                    missing_dx.append(cap)
                elif cap == "hardware" and is_software:
                    missing_dx.append(cap)
                else:
                    unverified_dx.append(cap)
        if missing_dx:
            status_dx = "UNSUPPORTED"
        elif unverified_dx:
            status_dx = "UNVERIFIED"
        else:
            status_dx = "EXECUTED"
        ret_dx = {
            "target": "directx",
            "backend": "directx",
            "provider": "warp" if is_software else "hardware",
            "device_name": device_name,
            "adapter_name": device_name,
            "vendor_id": int(data.get("vendor_id", 0)),
            "device_id": int(data.get("device_id", 0)),
            "subsystem_id": int(data.get("subsystem_id", 0)),
            "revision": int(data.get("revision", 0)),
            "dedicated_video_memory": int(data.get("dedicated_video_memory", 0)),
            "dedicated_memory": int(data.get("dedicated_video_memory", 0)),
            "dedicated_system_memory": int(data.get("dedicated_system_memory", 0)),
            "shared_system_memory": int(data.get("shared_system_memory", 0)),
            "is_software": is_software,
            "software_adapter": is_software,
            "d3d12_feature_level": feature_level,
            "feature_level": feature_level,
            "shader_model": shader_model,
            "highest_shader_model": str(data.get("highest_shader_model") or shader_model),
            "max_workgroup_size": list(data.get("max_workgroup_size", [1024, 1024, 64])),
            "max_workgroup_invocations": int(data.get("max_workgroup_invocations", 1024)),
            "max_shared_memory_bytes": int(data.get("max_shared_memory_bytes", 32768)),
            "status": status_dx,
            "missing_capabilities": missing_dx,
            "unverified_capabilities": unverified_dx,
        }
        if profile_obj is not None:
            matched, extra_missing = match_device_profile(ret_dx, profile_obj)
            for m in extra_missing:
                if m not in ret_dx["missing_capabilities"]:
                    ret_dx["missing_capabilities"].append(m)
            if not matched:
                ret_dx["status"] = "UNSUPPORTED"
        return ret_dx

    # Vulkan backend
    if provider is None and os.environ.get("MLIR_GPU_TEST_SWIFTSHADER") == "1":
        provider = "swiftshader"
    with _api_errors(), ExitStack() as resources:
        vk = _binding("vulkan")
        extensions = []
        available = {extension.extensionName for extension in vk.vkEnumerateInstanceExtensionProperties(None)}
        flags = 0
        if "VK_KHR_portability_enumeration" in available:
            extensions.append("VK_KHR_portability_enumeration")
            flags = getattr(vk, "VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR", 1)
        if "VK_KHR_get_physical_device_properties2" in available:
            extensions.append("VK_KHR_get_physical_device_properties2")
        instance = vk.vkCreateInstance(vk.VkInstanceCreateInfo(
            flags=flags,
            pApplicationInfo=vk.VkApplicationInfo(pApplicationName="mlir-gpu capabilities", apiVersion=vk.VK_API_VERSION_1_0),
            ppEnabledExtensionNames=extensions,
            ppEnabledLayerNames=[],
        ), None)
        resources.callback(vk.vkDestroyInstance, instance, None)

        candidates = []
        for physical in vk.vkEnumeratePhysicalDevices(instance):
            properties = vk.vkGetPhysicalDeviceProperties(physical)
            for index, queue in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical)):
                if not queue.queueCount or not queue.queueFlags & vk.VK_QUEUE_GRAPHICS_BIT:
                    continue
                device_extensions = {item.extensionName for item in vk.vkEnumerateDeviceExtensionProperties(physical, None)}
                is_cpu = 1 if properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_CPU else 0
                is_not_discrete = 0 if properties.deviceType == vk.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU else 1
                candidates.append((is_cpu, is_not_discrete, physical, index, properties, device_extensions))
                break

        if not candidates:
            if provider is not None:
                raise ViewerError(
                    f"Requested provider '{provider}' was not found among available Vulkan physical devices. "
                    "Accidental hardware/driver fallback rejected."
                )
            raise ViewerError("No Vulkan device has a suitable graphics/presentation queue.")

        if provider is not None:
            matching = [c for c in candidates if _match_vulkan_device(c[4], provider)]
            if not matching:
                raise ViewerError(
                    f"Requested provider '{provider}' was not found among available Vulkan physical devices. "
                    "Accidental hardware/driver fallback rejected."
                )
            selected = sorted(matching, key=lambda c: (c[0], c[1]))[0]
        else:
            selected = sorted(candidates, key=lambda c: (c[0], c[1]))[0]

        _, _, physical, family, properties, device_extensions = selected
        raw_name = properties.deviceName
        name = (raw_name.decode("utf-8", errors="replace") if isinstance(raw_name, bytes) else str(raw_name)).rstrip("\x00")
        type_names = {0: "other", 1: "integrated_gpu", 2: "discrete_gpu", 3: "virtual", 4: "cpu"}
        device_id = getattr(properties, "deviceID", 0)
        vendor_id = getattr(properties, "vendorID", 0)
        device_type = getattr(properties, "deviceType", 0)
        is_software = (
            device_type == vk.VK_PHYSICAL_DEVICE_TYPE_CPU
            or vendor_id in (0x10005, 0x1AE0)
            or device_id == 0xC0DE
            or "llvmpipe" in name.lower() or "lavapipe" in name.lower() or "swiftshader" in name.lower()
        )
        resolved_provider = (
            "lavapipe" if (vendor_id == 0x10005 or "llvmpipe" in name.lower() or "lavapipe" in name.lower())
            else "swiftshader" if (vendor_id == 0x1AE0 or device_id == 0xC0DE or "swiftshader" in name.lower())
            else "hardware"
        )
        vk_wg = getattr(properties.limits, "maxComputeWorkGroupSize", [1024, 1024, 64])
        max_workgroup_size = [int(vk_wg[0]), int(vk_wg[1]), int(vk_wg[2])]
        max_workgroup_invocations = int(getattr(properties.limits, "maxComputeWorkGroupInvocations", 1024))
        max_shared_memory_bytes = int(getattr(properties.limits, "maxComputeSharedMemorySize", 32768))
        limits_dict = {
            "max_push_constants_size": properties.limits.maxPushConstantsSize,
            "max_framebuffer_width": properties.limits.maxFramebufferWidth,
            "max_framebuffer_height": properties.limits.maxFramebufferHeight,
            "max_workgroup_size": max_workgroup_size,
            "max_workgroup_invocations": max_workgroup_invocations,
            "max_shared_memory_bytes": max_shared_memory_bytes,
        }

        missing = []
        unverified = []
        if required_capabilities:
            cap_aliases = {
                "ray_tracing": {"VK_KHR_ray_tracing_pipeline", "VK_NV_ray_tracing", "VK_KHR_ray_query"},
                "mesh_shaders": {"VK_EXT_mesh_shader", "VK_NV_mesh_shader"},
                "mesh_shader": {"VK_EXT_mesh_shader", "VK_NV_mesh_shader"},
                "variable_rate_shading": {"VK_KHR_fragment_shading_rate", "VK_NV_shading_rate_image"},
                "dynamic_rendering": {"VK_KHR_dynamic_rendering"},
            }
            features = None
            if hasattr(vk, "vkGetPhysicalDeviceFeatures"):
                try:
                    features = vk.vkGetPhysicalDeviceFeatures(physical)
                except Exception:
                    features = None

            for cap in required_capabilities:
                if cap in device_extensions:
                    continue
                if cap in cap_aliases:
                    if any(ext in device_extensions for ext in cap_aliases[cap]):
                        continue
                    missing.append(cap)
                    continue
                if cap.startswith("max_push_constants_size:"):
                    try:
                        val = int(cap.split(":", 1)[1])
                        if properties.limits.maxPushConstantsSize >= val:
                            continue
                        else:
                            missing.append(cap)
                            continue
                    except ValueError:
                        pass
                matched_feature = False
                if features is not None:
                    camel = "".join(part.capitalize() if i > 0 else part for i, part in enumerate(cap.split("_")))
                    val = getattr(features, cap, None)
                    if val is None:
                        val = getattr(features, camel, None)
                    if type(val) in (bool, int):
                        matched_feature = True
                        if val:
                            continue
                        else:
                            missing.append(cap)
                            continue
                if cap.startswith("VK_") or cap in cap_aliases or matched_feature:
                    missing.append(cap)
                elif cap in ("lavapipe", "swiftshader", "software") and not is_software:
                    missing.append(cap)
                elif cap == "hardware" and is_software:
                    missing.append(cap)
                else:
                    unverified.append(cap)

        if missing:
            status = "UNSUPPORTED"
        elif unverified:
            status = "UNVERIFIED"
        else:
            status = "EXECUTED"
        ret_vk = {
            "target": "vulkan",
            "backend": "vulkan",
            "provider": resolved_provider,
            "device_name": name,
            "vendor_id": int(vendor_id),
            "device_id": int(device_id),
            "device_type": type_names.get(device_type, "unknown"),
            "driver_version": str(getattr(properties, "driverVersion", 0)),
            "api_version": f"{properties.apiVersion >> 22}.{(properties.apiVersion >> 12) & 0x3FF}.{properties.apiVersion & 0xFFF}",
            "is_software": is_software,
            "limits": limits_dict,
            "max_push_constants_size": properties.limits.maxPushConstantsSize,
            "max_framebuffer_width": properties.limits.maxFramebufferWidth,
            "max_framebuffer_height": properties.limits.maxFramebufferHeight,
            "max_workgroup_size": max_workgroup_size,
            "max_workgroup_invocations": max_workgroup_invocations,
            "max_shared_memory_bytes": max_shared_memory_bytes,
            "extensions": sorted(device_extensions),
            "status": status,
            "missing_capabilities": missing,
            "unverified_capabilities": unverified,
        }
        if profile_obj is not None:
            matched, extra_missing = match_device_profile(ret_vk, profile_obj)
            for m in extra_missing:
                if m not in ret_vk["missing_capabilities"]:
                    ret_vk["missing_capabilities"].append(m)
            if not matched:
                ret_vk["status"] = "UNSUPPORTED"
        return ret_vk


def match_device_profile(device_caps: dict[str, Any], profile: TargetProfile) -> tuple[bool, list[str]]:
    """Match device capabilities against a TargetProfile and return (satisfied, missing_features)."""
    missing: list[str] = []
    dev_wg = device_caps.get("max_workgroup_size", [0, 0, 0])
    for axis, dev_dim, req_dim in zip(("X", "Y", "Z"), dev_wg, profile.max_workgroup_size):
        if dev_dim < req_dim:
            missing.append(f"max_workgroup_size_{axis}: required {req_dim}, device has {dev_dim}")
    dev_inv = int(device_caps.get("max_workgroup_invocations", 0))
    if dev_inv < profile.max_workgroup_invocations:
        missing.append(f"max_workgroup_invocations: required {profile.max_workgroup_invocations}, device has {dev_inv}")
    dev_shmem = int(device_caps.get("max_shared_memory_bytes", 0))
    if dev_shmem < profile.max_shared_memory_bytes:
        missing.append(f"max_shared_memory_bytes: required {profile.max_shared_memory_bytes}, device has {dev_shmem}")
    missing.extend(profile.check_capabilities(device_caps))
    return (len(missing) == 0, missing)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Open a live Vulkan, DirectX 12 (Windows), or Metal preview window by default.",
        epilog="Use -frames 0 to keep the window open, --output for PNG, or --video for MP4 on Vulkan/Metal.",
    )
    parser.add_argument("manifest", type=Path, nargs="?", default=None, help="Compiled Vulkan, DirectX, or Metal pipeline.json")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=600)
    parser.add_argument("--provider", choices=["hardware", "warp", "lavapipe", "swiftshader"],
                        help="Explicitly select rendering provider (hardware, warp, lavapipe, swiftshader)")
    parser.add_argument("--report-capabilities", action="store_true",
                        help="Report device capabilities as JSON and exit")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--output", type=Path, help="Render an offscreen PNG instead of opening a window (new file)")
    mode.add_argument("--video", type=Path, help="Export a fixed-time GPU animation to MP4 (new file)")
    mode.add_argument("--frames", "-frames", type=int, default=0, help="Live window: close after N frames; 0 stays open")
    parser.add_argument("--validation", action="store_true", help="Enable Vulkan, D3D12 debug/GPU, or Metal API/GPU validation")
    parser.add_argument("--time", type=float, help="Fixed snapshot/live time or MP4 start time (seconds)")
    parser.add_argument("--duration", type=float, help="MP4 duration in seconds (default 4)")
    parser.add_argument("--fps", type=int, help="MP4 frames per second, 1-240 (default 30)")
    args = parser.parse_args()
    if args.video is None and (args.duration is not None or args.fps is not None):
        parser.error("--duration and --fps require --video")
    if args.manifest is None:
        if args.report_capabilities:
            backend = "vulkan" if args.provider in ("lavapipe", "swiftshader") else ("directx" if sys.platform == "win32" else "vulkan")
            caps = get_device_capabilities(backend=backend, provider=args.provider)
            print(json.dumps(caps, indent=2))
            return 0
        parser.error("the following arguments are required: manifest")
    try:
        program = load_program(args.manifest)
        if args.report_capabilities:
            caps = get_device_capabilities(backend=program.target, provider=args.provider)
            print(json.dumps(caps, indent=2))
            return 0
        timing = {"time": args.time} if args.time is not None else {}
        provider_kwargs = {"provider": args.provider} if args.provider is not None else {}
        if args.video:
            fps = args.fps if args.fps is not None else 30
            count = export_mp4(program, args.video, args.width, args.height,
                               args.duration if args.duration is not None else 4.0,
                               fps, args.time if args.time is not None else 0.0, args.validation,
                               **provider_kwargs)
            print(f"Exported {count} frames at {fps} fps: {args.video}")
        elif args.output:
            if args.output.exists():
                raise ViewerError(f"Output already exists; choose a new PNG path: {args.output}")
            render = (
                render_offscreen if program.target == "vulkan" else
                render_directx_offscreen if program.target == "directx" and sys.platform != "darwin" else
                render_metal_offscreen
            )
            image = render(program, args.width, args.height, args.validation, **provider_kwargs, **timing)
            write_png(args.output, image)
            print(f"Rendered {image.width}x{image.height} on {image.device}: {args.output}")
        elif program.target == "vulkan":
            count = show_window(program, args.width, args.height, args.frames, args.validation, **provider_kwargs, **timing)
            print(f"Presented {count} Vulkan frames.")
        elif program.target == "directx" and sys.platform != "darwin":
            show_directx_window(program, args.width, args.height, args.frames, args.validation, **provider_kwargs, **timing)
        else:
            if args.provider is not None:
                raise ValueError(f"Provider '{args.provider}' is not supported by the Metal backend.")
            show_metal_window(program, args.width, args.height, args.frames, args.validation, **timing)
    except (ViewerError, CompileError, OSError, ValueError) as error:
        print(f"mlir-gpu-view: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
