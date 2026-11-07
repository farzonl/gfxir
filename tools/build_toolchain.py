#!/usr/bin/env python3
import argparse
import os
import platform
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*arguments):
    subprocess.run([str(argument) for argument in arguments], check=True)


def _darwin_cmake_flags():
    if platform.system() != "Darwin":
        return []
    sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    return [f"-DCMAKE_OSX_SYSROOT={sdk}", "-DCMAKE_OSX_DEPLOYMENT_TARGET=14.0"]


def _patch_cmake_export_checks(build: Path) -> None:
    cmake_dir = build / "lib/cmake"
    if not cmake_dir.is_dir():
        return
    marker = "set(_cmake_import_check_targets \"\")\nset(_IMPORT_CHECK_TARGETS \"\")\n"
    for targets_file in cmake_dir.rglob("*Targets.cmake"):
        text = targets_file.read_text()
        if marker in text:
            continue
        for token in (
            "foreach(_cmake_target IN LISTS _cmake_import_check_targets)",
            "foreach(target ${_IMPORT_CHECK_TARGETS})",
        ):
            if token in text:
                targets_file.write_text(text.replace(token, marker + token, 1))
                break


def _prune_object_files(build: Path) -> None:
    for obj in build.rglob("*.o"):
        obj.unlink(missing_ok=True)


def _ensure_toolchain_dependencies(build: Path, jobs: int) -> None:
    cmake_text = (ROOT / "native/compiler/CMakeLists.txt").read_text()
    required_libs = sorted(set(re.findall(r"\bMLIR[A-Za-z0-9_]+\b", cmake_text)))
    missing_targets = [
        lib for lib in required_libs
        if not (build / "lib" / f"lib{lib}.a").is_file() and not (build / "lib" / f"{lib}.lib").is_file()
    ]
    exe_suffix = ".exe" if os.name == "nt" else ""
    for tool in ("llvm-tblgen", "mlir-tblgen", "llc", "llvm-config", "FileCheck", "split-file"):
        if not (build / "bin" / f"{tool}{exe_suffix}").is_file():
            missing_targets.append(tool)
    if missing_targets:
        run("cmake", "--build", build, "--parallel", jobs, "--target", *missing_targets)


def _build_compiler_only(build: Path, jobs: int) -> None:
    mlir_cmake = build / "lib/cmake/mlir"
    llvm_cmake = build / "lib/cmake/llvm"
    if not (mlir_cmake / "MLIRConfig.cmake").is_file() or not (llvm_cmake / "LLVMConfig.cmake").is_file():
        raise SystemExit(
            f"Prebuilt LLVM/MLIR CMake config not found in {build}. "
            "Run tools/build_toolchain.py without --compiler-only first."
        )
    _ensure_toolchain_dependencies(build, jobs)
    _patch_cmake_export_checks(build)
    compiler_build = ROOT / "build/compiler"
    configuration = [
        "cmake", "-S", ROOT / "native/compiler", "-B", compiler_build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
        f"-DLLVM_DIR={llvm_cmake}",
        f"-DMLIR_DIR={mlir_cmake}",
        f"-DCMAKE_INSTALL_PREFIX={build}",
        "-DGFX_INCLUDE_TESTS=OFF",
        *_darwin_cmake_flags(),
    ]
    run(*configuration)
    run("cmake", "--build", compiler_build, "--parallel", jobs, "--target", "gfx-opt", "gfx-compile")
    run("cmake", "--install", compiler_build)
    print(f"Compiler tools: {build / 'bin'}")


def main():
    parser = argparse.ArgumentParser(description="Build the pinned LLVM/MLIR DirectX toolchain and gfx compiler.")
    parser.add_argument("--jobs", type=int, default=min(os.cpu_count() or 1, 8))
    parser.add_argument(
        "--compiler-only",
        action="store_true",
        help="Rebuild only native/compiler (gfx-opt, gfx-compile) against an existing .toolchains/llvm-build.",
    )
    parser.add_argument(
        "--prune-objects",
        action="store_true",
        help="Remove intermediate .o files from .toolchains/llvm-build after building to reduce cache size.",
    )
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    revision = (ROOT / "tools/llvm-revision.txt").read_text().strip()
    source, build = ROOT / ".toolchains/llvm-project", ROOT / ".toolchains/llvm-build"
    if args.compiler_only:
        _build_compiler_only(build, args.jobs)
        return
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
        f"-DLLVM_EXTERNAL_LIT={source / 'llvm/utils/lit/lit.py'}",
        *_darwin_cmake_flags(),
    ]
    run(*configuration)
    run(
        "cmake", "--build", build, "--parallel", args.jobs,
        "--target", "llvm-tblgen", "mlir-tblgen", "gfx-opt", "gfx-compile", "llc", "llvm-config", "FileCheck", "split-file",
    )
    _patch_cmake_export_checks(build)
    if args.prune_objects:
        _prune_object_files(build)
    print(f"Compiler tools: {build / 'bin'}")


if __name__ == "__main__":
    main()
