import os
from pathlib import Path

from mlir_gpu import CompileError, Toolchain


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LLVM_BIN = ROOT / ".toolchains" / "llvm-build" / "bin"
DEFAULT_GPU_TOOLS_BIN = ROOT / ".tools" / "bin"
DEFAULT_SDK_BIN = ROOT / ".sdk" / "bin"
for _default_bin in (DEFAULT_LLVM_BIN, DEFAULT_GPU_TOOLS_BIN, DEFAULT_SDK_BIN):
    if _default_bin.is_dir():
        _path_entries = [p for p in os.environ.get("PATH", "").split(os.pathsep) if p]
        if str(_default_bin) not in _path_entries:
            os.environ["PATH"] = os.pathsep.join([*_path_entries, str(_default_bin)])

TOOLS = Toolchain(
    Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else (
        DEFAULT_LLVM_BIN if DEFAULT_LLVM_BIN.is_dir() else None
    ),
    Path(os.environ["GPU_TOOLS_BIN"]) if "GPU_TOOLS_BIN" in os.environ else None,
)


def available(*names):
    try:
        for name in names or ("gfx-opt", "gfx-compile"):
            TOOLS.find(name)
    except CompileError:
        return False
    return True
