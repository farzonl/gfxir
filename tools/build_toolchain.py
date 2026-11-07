#!/usr/bin/env python3
import argparse
import os
import platform
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    subprocess.run([str(argument) for argument in arguments], check=True)


def main():
    parser = argparse.ArgumentParser(description="Build the pinned LLVM/MLIR DirectX toolchain and gfx compiler.")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    revision = (ROOT / "tools/llvm-revision.txt").read_text().strip()
    source, build = ROOT / ".toolchains/llvm-project", ROOT / ".toolchains/llvm-build"
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "init", source)
        run("git", "-C", source, "remote", "add", "origin", "https://github.com/llvm/llvm-project.git")
        run("git", "-C", source, "sparse-checkout", "set", "llvm", "mlir", "cmake", "third-party", "libc")
        run("git", "-C", source, "fetch", "--depth", "1", "--filter=blob:none", "origin", revision)
        run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != revision:
        raise SystemExit(f"LLVM checkout is {actual}, expected {revision}; use a separate matching checkout.")
    changed = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True
    )
    if changed:
        raise SystemExit("The LLVM checkout has local modifications; preserve them and use a clean pinned checkout.")
    configuration = [
        "cmake", "-S", source / "llvm", "-B", build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
        "-DLLVM_ENABLE_PROJECTS=mlir", "-DLLVM_TARGETS_TO_BUILD=Native;DirectX",
        "-DLLVM_ENABLE_ASSERTIONS=ON", "-DLLVM_INCLUDE_TESTS=OFF", "-DMLIR_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF", "-DLLVM_ENABLE_ZSTD=OFF", "-DLLVM_ENABLE_ZLIB=OFF",
        "-DLLVM_PARALLEL_LINK_JOBS=1", "-DLLVM_EXTERNAL_PROJECTS=gfx",
        f"-DLLVM_EXTERNAL_GFX_SOURCE_DIR={ROOT / 'native/compiler'}",
    ]
    if platform.system() == "Darwin":
        sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
        configuration.extend([f"-DCMAKE_OSX_SYSROOT={sdk}", "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0"])
    run(*configuration)
    run("cmake", "--build", build, "--parallel", args.jobs, "--target",
        "gfx-opt", "gfx-compile", "llc", "llvm-config", "FileCheck", "split-file")
    print(f"Compiler tools: {build / 'bin'}")


if __name__ == "__main__":
    main()
