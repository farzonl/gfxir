from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from .frontend import Pipeline, Stage
from .interfaces import verify_graphics_interface


class CompileError(RuntimeError):
    pass


class Target(Enum):
    VULKAN = "vulkan"
    DIRECTX = "directx"
    METAL = "metal"


PIPELINE_SCHEMA_VERSION = 3
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


def _compile_stage(stage: Stage, target: Target, directory: Path, tools: dict[str, str]) -> str:
    stem = stage.kind
    source = directory / f"{stem}.json"
    source.write_text(json.dumps(stage.to_dict(), allow_nan=False, indent=2) + "\n")
    reflection = directory / f"{stem}.reflection.json"
    output = directory / f"{stem}.{'spv' if target == Target.VULKAN else 'll'}"
    _run([
        tools["gfx-compile"], str(source), "--target", "vulkan" if target == Target.VULKAN else "directx",
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
    pipeline: Pipeline,
    target: Target,
    output: Path,
    toolchain: Toolchain | None = None,
) -> Path:
    if not isinstance(target, Target):
        raise TypeError("target must be a Target.")
    output = Path(output).absolute()
    if output.exists():
        raise CompileError(f"Output already exists; choose a new directory: {output}")
    stages = pipeline.trace()
    tools = (toolchain or Toolchain()).preflight(target)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".mlir-gpu-", dir=output.parent) as scratch:
        directory = Path(scratch) / "pipeline"
        directory.mkdir()
        entries = []
        for stage in stages:
            binary = _compile_stage(stage, target, directory, tools)
            entries.append({"stage": stage.kind, "entry_point": stage.name, "binary": binary})
        interface = {
            stage.kind: json.loads((directory / f"{stage.kind}.reflection.json").read_text())
            for stage in stages
        }
        verify_graphics_interface(interface)
        manifest = {
            "schema_version": PIPELINE_SCHEMA_VERSION,
            "target": target.value,
            "entries": entries,
            "interface": interface,
            "draw": pipeline.draw.to_dict(),
            "raster_convention": RASTER_CONVENTION,
            "runtime_verified": False,
            "backend_status": "converted-binary" if target == Target.METAL else "validated-binary",
        }
        (directory / "pipeline.json").write_text(json.dumps(manifest, indent=2) + "\n")
        directory.rename(output)
    return output / "pipeline.json"
