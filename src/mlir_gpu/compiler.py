from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

from .frontend import Draw, Pipeline, Stage
from .interfaces import reflect_pipeline_parameters, verify_graphics_interface


class CompileError(RuntimeError):
    pass


class Target(Enum):
    VULKAN = "vulkan"
    DIRECTX = "directx"
    METAL = "metal"


@dataclass(frozen=True)
class TargetProfile:
    """Target capability profile defining hardware limits and required features."""

    spirv_version: str = "1.0"
    shader_model: str = "6.0"
    required_capabilities: tuple[str, ...] = ()
    max_workgroup_size: tuple[int, int, int] = (1024, 1024, 64)
    max_workgroup_invocations: int = 1024
    max_shared_memory_bytes: int = 32768

    def __post_init__(self) -> None:
        if not isinstance(self.required_capabilities, tuple):
            object.__setattr__(
                self,
                "required_capabilities",
                tuple(str(c) for c in self.required_capabilities),
            )
        if not isinstance(self.max_workgroup_size, tuple) or len(self.max_workgroup_size) != 3:
            size_list = list(self.max_workgroup_size)
            if len(size_list) != 3:
                raise ValueError(f"max_workgroup_size must have 3 dimensions, got {len(size_list)}")
            object.__setattr__(
                self,
                "max_workgroup_size",
                (int(size_list[0]), int(size_list[1]), int(size_list[2])),
            )
        for i, dim in enumerate(self.max_workgroup_size):
            if type(dim) is not int or isinstance(dim, bool) or dim <= 0:
                raise ValueError(f"max_workgroup_size dimension {i} must be positive, got {dim}")
        if type(self.max_workgroup_invocations) is not int or isinstance(self.max_workgroup_invocations, bool) or self.max_workgroup_invocations <= 0:
            raise ValueError(f"max_workgroup_invocations must be positive, got {self.max_workgroup_invocations}")
        if type(self.max_shared_memory_bytes) is not int or isinstance(self.max_shared_memory_bytes, bool) or self.max_shared_memory_bytes < 0:
            raise ValueError(f"max_shared_memory_bytes cannot be negative, got {self.max_shared_memory_bytes}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "spirv_version": self.spirv_version,
            "shader_model": self.shader_model,
            "required_capabilities": list(self.required_capabilities),
            "max_workgroup_size": list(self.max_workgroup_size),
            "max_workgroup_invocations": self.max_workgroup_invocations,
            "max_shared_memory_bytes": self.max_shared_memory_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TargetProfile:
        if not isinstance(data, Mapping):
            raise TypeError(f"TargetProfile data must be a mapping, got {type(data).__name__}")
        return cls(
            spirv_version=str(data.get("spirv_version", "1.0")),
            shader_model=str(data.get("shader_model", "6.0")),
            required_capabilities=tuple(data.get("required_capabilities", ())),
            max_workgroup_size=tuple(data.get("max_workgroup_size", (1024, 1024, 64))),
            max_workgroup_invocations=int(data.get("max_workgroup_invocations", 1024)),
            max_shared_memory_bytes=int(data.get("max_shared_memory_bytes", 32768)),
        )

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> TargetProfile:
        return cls.from_dict(json.loads(text))

    def verify_workgroup_size(self, dims: Sequence[int]) -> None:
        """Verify that workgroup dimensions satisfy profile constraints."""
        if not isinstance(dims, (list, tuple)):
            raise TypeError(f"Workgroup size must be a list or tuple, got {type(dims).__name__}")
        if len(dims) != 3:
            raise ValueError(f"Workgroup size requires exactly 3 dimensions [X, Y, Z], got {len(dims)}")
        for i, val in enumerate(dims):
            if type(val) is not int or isinstance(val, bool):
                raise TypeError(f"Workgroup dimension {i} must be an int, got {type(val).__name__}")
            if val <= 0:
                raise ValueError(f"Workgroup dimension {i} must be positive, got {val}")

        for i, (dim, max_dim) in enumerate(zip(dims, self.max_workgroup_size)):
            if dim > max_dim:
                axis = ("X", "Y", "Z")[i]
                raise ValueError(
                    f"Workgroup {axis} dimension ({dim}) exceeds target profile limit ({max_dim})"
                )

        total = dims[0] * dims[1] * dims[2]
        if total > self.max_workgroup_invocations:
            raise ValueError(
                f"Total workgroup invocations ({total} = {dims[0]}x{dims[1]}x{dims[2]}) "
                f"exceeds target profile limit ({self.max_workgroup_invocations})"
            )

    def verify_shared_memory_bytes(self, size_bytes: int) -> None:
        """Verify that shared memory allocation does not exceed profile limit."""
        if type(size_bytes) is not int or isinstance(size_bytes, bool):
            raise TypeError(f"Shared memory size must be an integer, got {type(size_bytes).__name__}")
        if size_bytes < 0:
            raise ValueError(f"Shared memory size cannot be negative, got {size_bytes}")
        if size_bytes > self.max_shared_memory_bytes:
            raise ValueError(
                f"Shared memory allocation of {size_bytes} bytes exceeds target profile limit "
                f"of {self.max_shared_memory_bytes} bytes"
            )

    def supports_capability(self, capability: str) -> bool:
        """Check if this profile requires or supports the given capability string."""
        return capability in self.required_capabilities

    def check_capabilities(
        self, available_capabilities: Sequence[str] | set[str] | Mapping[str, Any]
    ) -> list[str]:
        """Check if all profile required capabilities are present in available capabilities."""
        if isinstance(available_capabilities, Mapping):
            available = set(available_capabilities.get("extensions", []))
            for k in ("shader_model", "d3d12_feature_level", "feature_level", "provider", "target"):
                if k in available_capabilities:
                    available.add(str(available_capabilities[k]))
        elif isinstance(available_capabilities, (list, tuple, set)):
            available = set(available_capabilities)
        else:
            raise TypeError(
                f"available_capabilities must be a collection or mapping, got {type(available_capabilities).__name__}"
            )
        return [cap for cap in self.required_capabilities if cap not in available]

    @classmethod
    def default_for_target(cls, target: Target | str) -> TargetProfile:
        name = target.value if isinstance(target, Target) else str(target).lower()
        if name == "vulkan":
            return VULKAN_PORTABLE_1_0
        elif name in ("directx", "metal"):
            return DIRECTX_SM60
        return TargetProfile()

    @classmethod
    def preset(cls, name: str) -> TargetProfile:
        key = name.strip()
        if key in PRESETS:
            return PRESETS[key]
        raise ValueError(
            f"Unknown preset profile: {name!r}. Available: {sorted(set(k.upper() for k in PRESETS))}"
        )


VULKAN_PORTABLE_1_0 = TargetProfile(
    spirv_version="1.0",
    shader_model="6.0",
    required_capabilities=(),
    max_workgroup_size=(1024, 1024, 64),
    max_workgroup_invocations=1024,
    max_shared_memory_bytes=32768,
)

DIRECTX_SM60 = TargetProfile(
    spirv_version="1.0",
    shader_model="6.0",
    required_capabilities=(),
    max_workgroup_size=(1024, 1024, 64),
    max_workgroup_invocations=1024,
    max_shared_memory_bytes=32768,
)

DIRECTX_SM66 = TargetProfile(
    spirv_version="1.4",
    shader_model="6.6",
    required_capabilities=(),
    max_workgroup_size=(1024, 1024, 64),
    max_workgroup_invocations=1024,
    max_shared_memory_bytes=32768,
)

HIGH_END_COMPUTE = TargetProfile(
    spirv_version="1.6",
    shader_model="6.6",
    required_capabilities=("subgroup_operations", "int64_atomics"),
    max_workgroup_size=(1024, 1024, 64),
    max_workgroup_invocations=1024,
    max_shared_memory_bytes=65536,
)

PRESETS: dict[str, TargetProfile] = {
    "VULKAN_PORTABLE_1_0": VULKAN_PORTABLE_1_0,
    "vulkan_portable_1_0": VULKAN_PORTABLE_1_0,
    "DIRECTX_SM60": DIRECTX_SM60,
    "directx_sm60": DIRECTX_SM60,
    "DIRECTX_SM66": DIRECTX_SM66,
    "directx_sm66": DIRECTX_SM66,
    "HIGH_END_COMPUTE": HIGH_END_COMPUTE,
    "high_end_compute": HIGH_END_COMPUTE,
}


@dataclass(frozen=True)
class _MLIRStage:
    kind: str
    name: str
    source: Path


@dataclass(frozen=True)
class MLIRPipeline:
    """Textual MLIR or structured JSON using the same artifact/viewer contract."""

    vertex: Path
    fragment: Path
    vertex_entry: str
    fragment_entry: str
    draw: Draw = field(default_factory=Draw)

    def trace(self) -> tuple[_MLIRStage, _MLIRStage]:
        if not isinstance(self.draw, Draw):
            raise TypeError("MLIRPipeline draw must be a Draw object.")
        return (
            _MLIRStage("vertex", self.vertex_entry, Path(self.vertex)),
            _MLIRStage("fragment", self.fragment_entry, Path(self.fragment)),
        )


PIPELINE_SCHEMA_VERSION = 3
PARAMETER_SCHEMA_VERSION = 4
RASTER_CONVENTION = {
    "clip_x": "right",
    "clip_y": "up",
    "clip_z": "zero-to-one",
    "framebuffer_origin": "top-left",
    "pixel_centers": "half-integer",
    "color": "linear-rgba8-unorm",
    "presentation_color_space": "srgb",
    "samples": 1,
    "culling": "none",
}


@dataclass(frozen=True)
class Toolchain:
    llvm_bin: Path | None = None
    tools_bin: Path | None = None

    def find(self, name: str) -> str:
        directories = (
            [self.llvm_bin] if name in {"gfx-compile", "gfx-opt", "mlir-opt", "mlir-translate", "llc", "llvm-as", "llvm-config"}
            else [self.tools_bin]
        )
        for directory in directories:
            if directory is not None:
                candidate = directory / (name + (".exe" if os.name == "nt" else ""))
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return str(candidate.resolve())
                raise CompileError(
                    f"Required tool {name!r} is missing or not executable in {directory}. "
                    "Build tools/build_toolchain.py and select .toolchains/llvm-build/bin for the native gfx compiler."
                )
        tool = shutil.which(name)
        if tool is None:
            raise CompileError(
                f"Required tool {name!r} was not found. Configure --llvm-bin/--tools-bin or PATH; "
                "there is no MSL/HLSL fallback."
            )
        return tool

    def preflight(self, target: Target) -> dict[str, str]:
        names = ["gfx-compile"]
        if target == Target.VULKAN:
            names.append("spirv-val")
        else:
            names.extend(["llc", "dxv"])
            if target == Target.METAL:
                names.append("metal-shaderconverter")
        tools = {name: self.find(name) for name in names}
        return tools


def _run(arguments: Sequence[str]) -> str:
    try:
        result = subprocess.run(arguments, text=True, capture_output=True, check=False)
    except OSError as error:
        raise CompileError(f"Could not execute {shlex.join(arguments)}: {error}") from error
    if result.returncode:
        diagnostics = result.stderr.strip() or result.stdout.strip() or "(no diagnostics)"
        raise CompileError(
            f"Command failed ({result.returncode}): {shlex.join(arguments)}\n{diagnostics}"
        )
    return result.stdout


def _check_binary(path: Path, magic: bytes, minimum: int) -> None:
    if not path.is_file():
        raise CompileError(f"Compiler did not produce the expected binary: {path.name}.")
    data = path.read_bytes()
    if len(data) < minimum or not data.startswith(magic):
        raise CompileError(f"Compiler produced an invalid {path.suffix} binary: {path.name}.")


def convert_dxil_to_metallib(dxil: Path, metallib: Path, entry_point: str, converter: str) -> None:
    _run([
        converter, str(dxil),
        f"--entry-point={entry_point}", f"--rename-entry-point={entry_point}",
        "-o", str(metallib),
    ])
    _check_binary(metallib, b"MTLB", 16)


def _compile_stage(
    stage: Stage | _MLIRStage,
    target: Target,
    directory: Path,
    tools: dict[str, str],
    target_profile: TargetProfile | None = None,
) -> str:
    stem = stage.kind
    native = isinstance(stage, _MLIRStage)
    input_format = "json" if not native or stage.source.suffix.lower() == ".json" else "mlir"
    source = directory / f"{stem}.{input_format}"
    if native:
        try:
            shutil.copyfile(stage.source, source)
        except OSError as error:
            raise CompileError(f"Cannot read native shader {stage.source}: {error}") from error
    else:
        source.write_text(json.dumps(stage.to_dict(), allow_nan=False, indent=2) + "\n")
    reflection = directory / f"{stem}.reflection.json"
    output = directory / f"{stem}.{'spv' if target == Target.VULKAN else 'll'}"
    profile_args: list[str] = []
    if target_profile is not None:
        profile_path = directory / f"{stem}.target_profile.json"
        profile_path.write_text(target_profile.to_json() + "\n")
        profile_args = ["--target-profile", str(profile_path)]
    _run([
        tools["gfx-compile"], str(source), "--input-format", input_format,
        "--target", "vulkan" if target == Target.VULKAN else "directx",
        *profile_args,
        "--output", str(output), "--ir-output", str(directory / f"{stem}.mlir"),
        "--lowered-output", str(directory / f"{stem}.lowered.mlir"), "--reflection", str(reflection),
    ])
    if target == Target.VULKAN:
        _check_binary(output, b"\x03\x02\x23\x07", 20)
        _run([tools["spirv-val"], "--target-env", "vulkan1.0", str(output)])
        return output.name
    unsigned = directory / f"{stem}.unsigned.dxil"
    dxil = directory / f"{stem}.dxil"
    _run([tools["llc"], "-O0", "-filetype=obj", str(output), "-o", str(unsigned)])
    _check_binary(unsigned, b"DXBC", 32)
    _run([tools["dxv"], str(unsigned), "-o", str(dxil)])
    _check_binary(dxil, b"DXBC", 32)
    if target == Target.DIRECTX:
        return dxil.name
    metallib = directory / f"{stem}.metallib"
    convert_dxil_to_metallib(dxil, metallib, stage.name, tools["metal-shaderconverter"])
    return metallib.name


def compile_pipeline(
    pipeline: Pipeline | MLIRPipeline | Any,
    target: Target,
    output: Path,
    toolchain: Toolchain | None = None,
    target_profile: TargetProfile | str | None = None,
    reflection_path: Path | None = None,
) -> Path | dict:
    if not isinstance(target, Target):
        raise TypeError("target must be a Target.")
    if target_profile is None:
        profile = TargetProfile.default_for_target(target)
    elif isinstance(target_profile, str):
        profile = TargetProfile.preset(target_profile)
    elif isinstance(target_profile, TargetProfile):
        profile = target_profile
    else:
        raise TypeError(
            f"target_profile must be a TargetProfile, preset name, or None, got {type(target_profile).__name__}"
        )
    output = Path(output).absolute()
    if output.exists():
        raise CompileError(f"Output already exists; choose a new directory: {output}")

    is_compute_entry = hasattr(pipeline, "kind") and getattr(pipeline, "kind") == "compute"
    if is_compute_entry:
        stages = [pipeline.trace()]
    else:
        stages = pipeline.trace()

    tools = (toolchain or Toolchain()).preflight(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mlir-gpu-", dir=output.parent) as scratch:
        directory = Path(scratch) / "pipeline"
        directory.mkdir()

        if is_compute_entry:
            # Validate output suffix against target
            if target == Target.VULKAN:
                if output.suffix != ".spv":
                    raise CompileError(
                        f"Target.VULKAN compute output must have '.spv' extension, got: {output.name}"
                    )
            elif target == Target.DIRECTX:
                if output.suffix not in (".dxil", ".ll"):
                    raise CompileError(
                        f"Target.DIRECTX compute output must have '.dxil' or '.ll' extension, got: {output.name}"
                    )
            elif target == Target.METAL:
                if output.suffix != ".metallib":
                    raise CompileError(
                        f"Target.METAL compute output must have '.metallib' extension, got: {output.name}"
                    )

            stage = stages[0]
            try:
                binary = _compile_stage(stage, target, directory, tools, target_profile=profile)
            except TypeError as err:
                if "unexpected keyword argument 'target_profile'" in str(err):
                    binary = _compile_stage(stage, target, directory, tools)
                else:
                    raise

            if target == Target.VULKAN:
                shutil.copyfile(directory / f"{stage.kind}.spv", output)
            elif target == Target.DIRECTX:
                if output.suffix == ".ll":
                    shutil.copyfile(directory / f"{stage.kind}.ll", output)
                else:
                    shutil.copyfile(directory / f"{stage.kind}.dxil", output)
            elif target == Target.METAL:
                shutil.copyfile(directory / f"{stage.kind}.metallib", output)

            if reflection_path is not None:
                reflection_path = Path(reflection_path).absolute()
                reflection_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / f"{stage.kind}.reflection.json", reflection_path)

            return {
                "target": target.value,
                "target_profile": profile.to_dict(),
                "stage": stage.kind,
                "entry_point": stage.name,
                "binary": str(output),
            }

        entries = []
        for stage in stages:
            try:
                binary = _compile_stage(stage, target, directory, tools, target_profile=profile)
            except TypeError as err:
                if "unexpected keyword argument 'target_profile'" in str(err):
                    binary = _compile_stage(stage, target, directory, tools)
                else:
                    raise
            entries.append({"stage": stage.kind, "entry_point": stage.name, "binary": binary})
        interface = {
            stage.kind: json.loads((directory / f"{stage.kind}.reflection.json").read_text())
            for stage in stages
        }
        verify_graphics_interface(interface)
        parameters = reflect_pipeline_parameters(interface)
        for stage in stages:
            if interface[stage.kind]["name"] != stage.name or interface[stage.kind]["stage"] != stage.kind:
                raise CompileError(f"Reflected entry does not match the declared {stage.kind} stage {stage.name!r}.")
        manifest = {
            "schema_version": PARAMETER_SCHEMA_VERSION if parameters else PIPELINE_SCHEMA_VERSION,
            "target": target.value,
            "target_profile": profile.to_dict(),
            "entries": entries,
            "interface": interface,
            "draw": pipeline.draw.to_dict(),
            "raster_convention": RASTER_CONVENTION,
            "runtime_verified": False,
            "backend_status": "converted-binary" if target == Target.METAL else "validated-binary",
        }
        if parameters:
            manifest["parameters"] = parameters
        (directory / "pipeline.json").write_text(json.dumps(manifest, indent=2) + "\n")
        directory.rename(output)
    return output / "pipeline.json"
