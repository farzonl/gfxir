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

from .frontend import ComputePipeline, Dispatch, Draw, Pipeline, Stage
from .interfaces import reflect_pipeline_parameters, verify_graphics_interface
from .parameters import verify_parameters


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
COMPUTE_SCHEMA_VERSION = 5
MULTI_DRAW_SCHEMA_VERSION = 6


@dataclass(frozen=True)
class DrawCall:
    """Represents a single draw command within a multi-draw sequence."""

    pipeline: str
    type: str = "standard"  # "standard" | "indexed"
    topology: str = "triangle-list"
    vertex_count: int = 0
    index_count: int = 0
    instance_count: int = 1
    first_vertex: int = 0
    first_index: int = 0
    vertex_offset: int = 0
    first_instance: int = 0
    index_type: str = "uint16"  # "uint16" | "uint32"
    constant_offset: int = 0
    constant_size: int = 0
    parameters: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.type not in ("standard", "indexed"):
            raise ValueError(f"Invalid draw type: {self.type!r}. Must be 'standard' or 'indexed'.")
        if self.topology != "triangle-list":
            raise ValueError(f"Unsupported topology: {self.topology!r}. Only 'triangle-list' is supported.")
        if self.vertex_count < 0 or self.index_count < 0:
            raise ValueError("vertex_count and index_count must be nonnegative.")
        if self.instance_count < 1:
            raise ValueError("instance_count must be at least 1.")
        if self.first_vertex < 0 or self.first_index < 0:
            raise ValueError("first_vertex and first_index must be nonnegative.")
        if self.constant_offset < 0 or self.constant_size < 0:
            raise ValueError("constant_offset and constant_size must be nonnegative.")
        if self.type == "standard" and self.vertex_count % 3 != 0:
            raise ValueError("Standard triangle-list vertex_count must be a multiple of 3.")
        if self.type == "indexed" and self.index_count % 3 != 0:
            raise ValueError("Indexed triangle-list index_count must be a multiple of 3.")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "pipeline": self.pipeline,
            "type": self.type,
            "topology": self.topology,
        }
        if self.type == "indexed":
            result["index_count"] = self.index_count
            result["first_index"] = self.first_index
            if self.vertex_offset != 0:
                result["vertex_offset"] = self.vertex_offset
            result["index_type"] = self.index_type
        else:
            result["vertex_count"] = self.vertex_count
            result["first_vertex"] = self.first_vertex

        if self.instance_count != 1:
            result["instance_count"] = self.instance_count
        if self.first_instance != 0:
            result["first_instance"] = self.first_instance
        if self.constant_offset != 0 or self.constant_size != 0:
            result["constant_offset"] = self.constant_offset
            result["constant_size"] = self.constant_size
        if self.parameters is not None:
            result["parameters"] = self.parameters
        return result


@dataclass(frozen=True)
class MultiDrawManifest:
    """Top-level pipeline manifest v5 expressing multiple pipelines and draws."""

    pipelines: Mapping[str, Any]
    draws: Sequence[DrawCall | Mapping[str, Any]]
    target: str = "vulkan"
    target_profile: TargetProfile | Mapping[str, Any] | None = None
    raster_convention: Mapping[str, Any] = field(default_factory=lambda: dict(RASTER_CONVENTION))
    parameters: list[dict[str, Any]] | dict[str, Any] | None = None
    schema_version: int = MULTI_DRAW_SCHEMA_VERSION
    runtime_verified: bool = False
    backend_status: str = "validated-binary"

    def __post_init__(self) -> None:
        if self.schema_version != MULTI_DRAW_SCHEMA_VERSION:
            raise ValueError(f"MultiDrawManifest requires schema_version == {MULTI_DRAW_SCHEMA_VERSION}")
        if not isinstance(self.pipelines, Mapping):
            raise TypeError("pipelines must be a mapping of pipeline IDs to pipeline configurations.")

        for draw in self.draws:
            draw_dict = draw.to_dict() if hasattr(draw, "to_dict") else draw
            if not isinstance(draw_dict, Mapping):
                raise TypeError("Each draw item must be a mapping or DrawCall instance.")
            pipe_ref = draw_dict.get("pipeline")
            if pipe_ref not in self.pipelines:
                raise ValueError(f"Draw references nonexistent pipeline: {pipe_ref!r}")
            for count_key in ("vertex_count", "index_count", "first_vertex", "first_index", "constant_offset", "constant_size"):
                val = draw_dict.get(count_key)
                if val is not None and (type(val) is not int or val < 0):
                    raise ValueError(f"Draw parameter {count_key} must be a nonnegative integer, got {val}")
            inst_cnt = draw_dict.get("instance_count")
            if inst_cnt is not None and (type(inst_cnt) is not int or inst_cnt < 1):
                raise ValueError(f"Draw parameter instance_count must be >= 1, got {inst_cnt}")

    def to_dict(self) -> dict[str, Any]:
        profile_dict = (
            self.target_profile.to_dict()
            if hasattr(self.target_profile, "to_dict")
            else (dict(self.target_profile) if isinstance(self.target_profile, Mapping)
                  else TargetProfile.default_for_target(self.target).to_dict())
        )
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "target": self.target,
            "target_profile": profile_dict,
            "raster_convention": dict(self.raster_convention),
            "runtime_verified": self.runtime_verified,
            "backend_status": self.backend_status,
            "pipelines": {k: (v.to_dict() if hasattr(v, "to_dict") else dict(v) if isinstance(v, Mapping) else v) for k, v in self.pipelines.items()},
            "draws": [(dict(d) if isinstance(d, Mapping) else d.to_dict()) for d in self.draws],
        }
        if self.parameters is not None:
            result["parameters"] = self.parameters
        return result


def verify_multi_draw_manifest(manifest: Mapping[str, Any] | MultiDrawManifest) -> None:
    """Strictly validate a Schema v5 multi-draw manifest dictionary."""
    data = manifest.to_dict() if hasattr(manifest, "to_dict") else manifest
    if not isinstance(data, Mapping):
        raise TypeError(f"Manifest must be a mapping, got {type(data).__name__}")

    version = data.get("schema_version")
    if version != MULTI_DRAW_SCHEMA_VERSION:
        raise ValueError(f"verify_multi_draw_manifest requires schema_version == {MULTI_DRAW_SCHEMA_VERSION}, got {version}")

    pipelines = data.get("pipelines")
    if not isinstance(pipelines, Mapping):
        raise ValueError("Manifest 'pipelines' must be a mapping.")

    draws = data.get("draws")
    if not isinstance(draws, (list, tuple)):
        raise ValueError("Manifest 'draws' must be a list or tuple.")

    for idx, draw in enumerate(draws):
        if hasattr(draw, "to_dict"):
            draw = draw.to_dict()
        if not isinstance(draw, Mapping):
            raise ValueError(f"Draw {idx} must be a dictionary.")
        pipe_ref = draw.get("pipeline")
        if not pipe_ref or pipe_ref not in pipelines:
            raise ValueError(f"Draw {idx} references nonexistent pipeline {pipe_ref!r}.")
        for k in ("vertex_count", "index_count", "first_vertex", "first_index", "constant_offset", "constant_size"):
            v = draw.get(k)
            if v is not None and (type(v) is not int or v < 0):
                raise ValueError(f"Draw {idx} field {k} must be nonnegative integer, got {v}")
        inst = draw.get("instance_count")
        if inst is not None and (type(inst) is not int or inst < 1):
            raise ValueError(f"Draw {idx} instance_count must be >= 1, got {inst}")

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
        spirv_env = (
            "vulkan1.2"
            if target_profile is not None
            and (
                target_profile.supports_capability("buffer_device_address")
                or target_profile.supports_capability("PhysicalStorageBuffer64")
            )
            else "vulkan1.0"
        )
        _run([tools["spirv-val"], "--target-env", spirv_env, str(output)])
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
    pipeline: Pipeline | ComputePipeline | MLIRPipeline | Any,
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
    is_compute_pipeline = isinstance(pipeline, ComputePipeline)
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
        if is_compute_pipeline:
            stage = stages[0]
            compute_refl = interface["compute"]
            if compute_refl.get("name") != stage.name or compute_refl.get("stage") != "compute":
                raise CompileError(
                    f"Reflected entry does not match the declared compute stage {stage.name!r}."
                )
            if "parameters" in compute_refl:
                verify_parameters(compute_refl["parameters"])
            if reflection_path is not None:
                reflection_path = Path(reflection_path).absolute()
                reflection_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(directory / "compute.reflection.json", reflection_path)
            manifest = {
                "schema_version": COMPUTE_SCHEMA_VERSION,
                "target": target.value,
                "target_profile": profile.to_dict(),
                "stage": "compute",
                "entries": entries,
                "interface": interface,
                "workgroup_size": compute_refl.get(
                    "workgroup_size", list(pipeline.compute.workgroup_size)
                ),
                "buffers": compute_refl.get("buffers", []),
                "dispatch": pipeline.dispatch.to_dict(),
                "runtime_verified": False,
                "backend_status": "converted-binary" if target == Target.METAL else "validated-binary",
            }
            if "parameters" in compute_refl:
                manifest["parameters"] = compute_refl["parameters"]
            if compute_refl.get("shared_memory_bytes", 0) > 0:
                manifest["shared_memory_bytes"] = compute_refl["shared_memory_bytes"]
            (directory / "pipeline.json").write_text(json.dumps(manifest, indent=2) + "\n")
            directory.rename(output)
            return output / "pipeline.json"

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
