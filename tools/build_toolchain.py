#!/usr/bin/env python3
import argparse
import os
import platform
import re
import shutil
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


def _rewrite_stale_sdk_paths(build: Path) -> None:
    if platform.system() != "Darwin" or not build.is_dir():
        return
    current_sdk = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
    candidates = [build / "CMakeCache.txt", build / "build.ninja"]
    cmake_dir = build / "lib/cmake"
    if cmake_dir.is_dir():
        candidates.extend(cmake_dir.rglob("*.cmake"))
    pattern = re.compile(r"/Applications/Xcode[^\s:;\"']+\.sdk")
    for path in candidates:
        if not path.is_file():
            continue
        text = path.read_text(errors="replace")
        stale = {sdk for sdk in pattern.findall(text) if sdk != current_sdk and not Path(sdk).exists()}
        if stale:
            for old_sdk in stale:
                text = text.replace(old_sdk, current_sdk)
            path.write_text(text)


def _freeze_existing_build_targets(build: Path) -> None:
    ninja_file = build / "build.ninja"
    if not ninja_file.is_file():
        return
    lines = ninja_file.read_text(errors="replace").splitlines(keepends=True)
    out = []
    changed = False
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("build "):
            block = [line]
            while block[-1].rstrip("\r\n").endswith("$") and i + 1 < len(lines):
                i += 1
                block.append(lines[i])
            while i + 1 < len(lines) and (lines[i + 1].startswith("  ") or lines[i + 1].startswith("\t")):
                i += 1
                block.append(lines[i])
            header = "".join(block)
            match = re.match(r"^build\s+([^:]+):\s+(\S+)", header)
            if match:
                targets = match.group(1).replace("$\n", " ").replace("$\r\n", " ").split()
                rule = match.group(2)
                if rule == "RERUN_CMAKE" or "build.ninja" in targets:
                    out.append(f"build {' '.join(targets)}: phony\n")
                    changed = True
                    i += 1
                    continue
                if (
                    targets
                    and rule != "phony"
                    and all(t.startswith("lib/") or t.startswith("bin/") for t in targets)
                    and all((build / t).exists() for t in targets)
                ):
                    out.append(f"build {' '.join(targets)}: phony\n")
                    changed = True
                    i += 1
                    continue
            out.extend(block)
        else:
            out.append(line)
        i += 1
    if changed:
        ninja_file.write_text("".join(out))


def _llvm_cmake_configuration(source: Path, build: Path):
    return [
        "cmake", "-S", source / "llvm", "-B", build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
        "-DLLVM_ENABLE_PROJECTS=mlir", "-DLLVM_TARGETS_TO_BUILD=Native;DirectX",
        "-DLLVM_ENABLE_ASSERTIONS=ON", "-DLLVM_INCLUDE_TESTS=OFF", "-DMLIR_INCLUDE_TESTS=OFF",
        "-DLLVM_INCLUDE_BENCHMARKS=OFF", "-DLLVM_ENABLE_ZSTD=OFF", "-DLLVM_ENABLE_ZLIB=OFF",
        "-DLLVM_ENABLE_LIBEDIT=OFF", "-DLLVM_ENABLE_TERMINFO=OFF", "-DLLVM_ENABLE_LIBXML2=OFF",
        "-DLLVM_PARALLEL_LINK_JOBS=1", "-DLLVM_EXTERNAL_PROJECTS=gfx",
        f"-DLLVM_EXTERNAL_GFX_SOURCE_DIR={ROOT / 'native/compiler'}",
        f"-DLLVM_EXTERNAL_LIT={source / 'llvm/utils/lit/lit.py'}",
        *_darwin_cmake_flags(),
    ]


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


def _ensure_toolchain_dependencies(source: Path, build: Path, jobs: int) -> None:
    _rewrite_stale_sdk_paths(build)
    cmake_text = (ROOT / "native/compiler/CMakeLists.txt").read_text()
    required_libs = sorted(set(re.findall(r"\bMLIR[A-Za-z0-9]+\b", cmake_text)))
    missing_targets = [
        lib for lib in required_libs
        if not (build / "lib" / f"lib{lib}.a").is_file() and not (build / "lib" / f"{lib}.lib").is_file()
    ]
    exe_suffix = ".exe" if os.name == "nt" else ""
    for tool in ("llvm-tblgen", "mlir-tblgen", "llc", "llvm-config", "FileCheck", "split-file"):
        if not (build / "bin" / f"{tool}{exe_suffix}").is_file():
            missing_targets.append(tool)
    if missing_targets:
        _freeze_existing_build_targets(build)
        run("cmake", "--build", build, "--parallel", jobs, "--target", *missing_targets)


def _build_compiler_only(source: Path, build: Path, jobs: int) -> None:
    mlir_cmake = build / "lib/cmake/mlir"
    llvm_cmake = build / "lib/cmake/llvm"
    if not (mlir_cmake / "MLIRConfig.cmake").is_file() or not (llvm_cmake / "LLVMConfig.cmake").is_file():
        raise SystemExit(
            f"Prebuilt LLVM/MLIR CMake config not found in {build}. "
            "Run tools/build_toolchain.py without --compiler-only first."
        )
    _ensure_toolchain_dependencies(source, build, jobs)
    _patch_cmake_export_checks(build)
    compiler_build = ROOT / "build/compiler"
    _rewrite_stale_sdk_paths(compiler_build)
    configuration = [
        "cmake", "-S", ROOT / "native/compiler", "-B", compiler_build, "-G", "Ninja",
        "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG",
        "-DLLVM_ENABLE_LIBEDIT=OFF", "-DLLVM_ENABLE_TERMINFO=OFF", "-DLLVM_ENABLE_LIBXML2=OFF",
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
        _build_compiler_only(source, build, args.jobs)
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
    _rewrite_stale_sdk_paths(build)
    run(*_llvm_cmake_configuration(source, build))
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
