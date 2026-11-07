from pathlib import Path

from mlir_gpu import MLIRPipeline


ir = Path(__file__).with_name("ir")
pipeline = MLIRPipeline(
    vertex=ir / "fullscreen.mlir",
    fragment=ir / "julia.json",
    vertex_entry="fullscreen",
    fragment_entry="julia",
)
