import os
from pathlib import Path

from mlir_gpu import CompileError, Toolchain


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Toolchain(
    Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None,
    Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None,
)


def available(*names):
    try:
        for name in names or ("gfx-opt", "gfx-compile"):
            TOOLS.find(name)
    except CompileError:
        return False
    return True
