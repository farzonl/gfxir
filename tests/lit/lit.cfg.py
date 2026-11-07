import os
import shlex
import shutil
import sys
from pathlib import Path

import lit.formats

config.name = "GfxIR"
config.test_format = lit.formats.ShTest()
config.suffixes = [".mlir", ".test", ".py"]
config.excludes = ["Inputs", "lit.cfg.py", "lit.site.cfg.py"]
root = Path(__file__).resolve().parents[2]
config.test_source_root = str(root / "tests/lit")
config.test_exec_root = str(root / "build/lit")

site = os.environ.get("GFX_LIT_SITE_CONFIG")
if site:
    lit_config.load_config(config, site)

llvm_bin = Path(os.environ.get("LLVM_BIN", getattr(config, "gfx_tools_dir", root / ".toolchains/llvm-build/bin")))
tools_bin = Path(os.environ.get("GPU_TOOLS_BIN", root / ".tools/bin"))
path = os.pathsep.join((
    str(llvm_bin), str(getattr(config, "llvm_tools_dir", llvm_bin)),
    str(tools_bin), os.environ.get("PATH", ""),
))
config.environment["PATH"] = path
if "LLVM_BIN" in os.environ:
    config.environment["LLVM_BIN"] = str(llvm_bin)
if "GPU_TOOLS_BIN" in os.environ:
    config.environment["GPU_TOOLS_BIN"] = str(tools_bin)
config.environment["PYTHONPATH"] = os.pathsep.join(
    (str(root / "src"), str(root / "tests"), os.environ.get("PYTHONPATH", ""))
)
config.substitutions.extend([
    ("%python", shlex.quote(sys.executable)),
    ("%repo", shlex.quote(str(root))),
])
for name in ("gfx-opt", "gfx-compile", "FileCheck", "split-file", "llc",
             "spirv-val", "dxv", "dxil-assemble", "metal-shaderconverter"):
    tool = shutil.which(name, path=path)
    if tool:
        config.available_features.add(name)
        config.substitutions.append(("%" + name, shlex.quote(tool)))
    elif name in ("gfx-opt", "gfx-compile", "FileCheck", "split-file"):
        lit_config.fatal(
            f"Required tool {name} not found; build tools/build_toolchain.py "
            "and set LLVM_BIN to its bin directory."
        )
