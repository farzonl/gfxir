import argparse
import os
import runpy
import tempfile
from pathlib import Path

from mlir_gpu import Target, Toolchain, compile_pipeline
from mlir_gpu.viewer import load_program, render_metal_offscreen, render_offscreen, write_png

from execution_targets import target_output_path, targets_from_environment


ROOT = Path(__file__).resolve().parents[4]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile and render an example shader.")
    parser.add_argument("example", choices=("triangle", "quad", "coordinates", "frame_bands"))
    parser.add_argument("output_prefix", type=Path)
    parser.add_argument("--target", required=True, metavar="ENVIRONMENT_VARIABLE")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--time", type=float, default=0.0)
    args = parser.parse_args()

    try:
        targets = [Target(target) for target in targets_from_environment(args.target)]
    except ValueError as error:
        parser.error(str(error))

    pipeline = runpy.run_path(str(ROOT / "examples" / f"{args.example}.py"))["pipeline"]
    llvm_bin = Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None
    toolchain = Toolchain(llvm_bin=llvm_bin)
    for target in targets:
        output = target_output_path(args.output_prefix, target.value)
        with tempfile.TemporaryDirectory(prefix=f"gfxir-{args.example}-") as directory:
            manifest = compile_pipeline(pipeline, target, Path(directory) / "artifacts", toolchain)
            validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
            program = load_program(manifest)
            if target == Target.METAL:
                image = render_metal_offscreen(
                    program, args.width, args.height, validation, toolchain, time=args.time,
                )
            else:
                image = render_offscreen(
                    program, args.width, args.height,
                    validation=validation, time=args.time,
                )
            output.unlink(missing_ok=True)
            write_png(output, image)
        print(f"Rendered {args.example} for {target.value}: {image.width}x{image.height} on {image.device}")


if __name__ == "__main__":
    main()
