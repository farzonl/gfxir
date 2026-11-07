import argparse
import os
import runpy
import sys
from pathlib import Path

from .compiler import CompileError, Target, Toolchain, compile_pipeline
from .frontend import Pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile Python graphics entry points through MLIR.")
    parser.add_argument("program", type=Path, help="Python file defining a 'pipeline' object")
    parser.add_argument("--target", choices=[target.value for target in Target], required=True)
    parser.add_argument("--output", type=Path, required=True, help="New artifact directory")
    parser.add_argument("--llvm-bin", type=Path, default=os.environ.get("LLVM_BIN"))
    parser.add_argument("--tools-bin", type=Path, default=os.environ.get("GPU_TOOLS_BIN"))
    args = parser.parse_args()
    try:
        namespace = runpy.run_path(str(args.program))
        pipeline = namespace.get("pipeline")
        if not isinstance(pipeline, Pipeline):
            raise CompileError("The program must define 'pipeline = Pipeline(vertex_entry, fragment_entry)'.")
        manifest = compile_pipeline(
            pipeline, Target(args.target), args.output, Toolchain(args.llvm_bin, args.tools_bin)
        )
    except (CompileError, OSError, TypeError, ValueError) as error:
        print(f"mlir-gpu: {error}", file=sys.stderr)
        return 1
    print(f"Compiled artifacts: {manifest}")
    print("Runtime execution has not been verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
