import argparse
import runpy
from pathlib import Path

from mlir_gpu import Target, compile_pipeline
from test_compiler import (
    arithmetic_pipeline, expanded_arithmetic_pipeline, shader_math_pipeline,
    triangle, unused_varying_pipeline,
)
from test_support import ROOT, TOOLS


parser = argparse.ArgumentParser()
parser.add_argument("program")
parser.add_argument("target", choices=[target.value for target in Target])
parser.add_argument("output", type=Path)
parser.add_argument("--cap", type=int)
args = parser.parse_args()
programs = {
    "triangle": triangle,
    "arithmetic": arithmetic_pipeline,
    "expanded": expanded_arithmetic_pipeline,
    "math": shader_math_pipeline,
    "unused": unused_varying_pipeline,
}
if args.program in programs:
    pipeline = programs[args.program]()
else:
    pipeline = runpy.run_path(str(ROOT / "examples" / f"{args.program}.py"))["pipeline"]
if args.cap is not None:
    pipeline.fragment.function.__globals__["MAX_STEPS"] = args.cap
manifest = compile_pipeline(pipeline, Target(args.target), args.output, TOOLS)
print(manifest.read_text())
