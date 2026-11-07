from pathlib import Path

from mlir_gpu import MLIRPipeline


ir = Path(__file__).with_name("ir")
pipeline = MLIRPipeline(
    vertex=ir / "fullscreen.mlir",
    fragment=ir / "sphere_march.mlir",
    vertex_entry="fullscreen",
    fragment_entry="sphere_march",
)
