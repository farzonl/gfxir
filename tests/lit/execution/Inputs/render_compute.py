"""Compile and execute compute shader examples, writing rendered RGBA8 PNGs per target."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import tempfile

from mlir_gpu import Target, Toolchain, compile_pipeline, render_compute_rgba, write_png
from execution_targets import target_output_path, targets_from_environment

ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and execute a compute shader example to PNG.")
    parser.add_argument("example", choices=("mandelbrot", "particle_life", "tiled_particle_life"))
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--target", required=True, metavar="ENVIRONMENT_VARIABLE")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()

    try:
        targets = [Target(target) for target in targets_from_environment(args.target)]
    except ValueError as error:
        parser.error(str(error))

    llvm_bin = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
    tools_bin = Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None
    toolchain = Toolchain(llvm_bin=llvm_bin, tools_bin=tools_bin)
    validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    if args.example == "mandelbrot":
        ns = runpy.run_path(str(ROOT / "examples" / "mandelbrot_compute.py"))
        rgba_pipeline = ns["rgba_pipeline"]
        params = {
            **ns["DEFAULT_MANDELBROT_PARAMS"],
            "width": args.width,
            "height": args.height,
            "pitch": args.width,
        }
        for target in targets:
            output = target_output_path(args.output_prefix, target.value)
            with tempfile.TemporaryDirectory(prefix="gfxir-mandelbrot-") as directory:
                manifest = compile_pipeline(rgba_pipeline, target, Path(directory) / "artifacts", toolchain)
                image = render_compute_rgba(
                    manifest,
                    args.width,
                    args.height,
                    parameters=params,
                    validation=validation,
                    toolchain=toolchain,
                )
                output.unlink(missing_ok=True)
                write_png(output, image)
            print(f"Rendered mandelbrot for {target.value}: {image.width}x{image.height} on {image.device}")
    elif args.example == "particle_life":
        ns = runpy.run_path(str(ROOT / "examples" / "particle_life_compute.py"))
        simulate_and_render = ns["simulate_and_render_particle_life"]
        for target in targets:
            output = target_output_path(args.output_prefix, target.value)
            _, _, image = simulate_and_render(
                target,
                steps=args.steps,
                width=args.width,
                height=args.height,
                validation=validation,
                toolchain=toolchain,
            )
            output.unlink(missing_ok=True)
            write_png(output, image)
            print(f"Rendered particle_life for {target.value}: {image.width}x{image.height} on {image.device}")
    else:
        ns = runpy.run_path(str(ROOT / "examples" / "tiled_particle_life.py"))
        simulate_and_render = ns["simulate_and_render_tiled_particle_life"]
        for target in targets:
            output = target_output_path(args.output_prefix, target.value)
            res = simulate_and_render(
                target,
                steps=args.steps,
                width=args.width,
                height=args.height,
                validation=validation,
                toolchain=toolchain,
            )
            output.unlink(missing_ok=True)
            write_png(output, res.image)
            print(
                f"Rendered tiled_particle_life for {target.value}: "
                f"{res.image.width}x{res.image.height} on {res.image.device}"
            )


if __name__ == "__main__":
    main()
