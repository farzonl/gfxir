#!/usr/bin/env python3
"""Cross-platform setup and cache orchestration for pinned SwiftShader Vulkan ICD."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*arguments: str | Path, **kwargs) -> None:
    cmd = [str(arg) for arg in arguments]
    subprocess.run(cmd, check=True, **kwargs)


def get_library_name() -> str:
    if sys.platform == "win32" or os.name == "nt":
        return "vk_swiftshader.dll"
    if sys.platform == "darwin":
        return "libvk_swiftshader.dylib"
    return "libvk_swiftshader.so"


def write_icd_json(icd_path: Path, lib_path: Path) -> Path:
    icd_content = {
        "file_format_version": "1.0.0",
        "ICD": {
            "library_path": lib_path.resolve().as_posix(),
            "api_version": "1.3.0",
        },
    }
    icd_path.parent.mkdir(parents=True, exist_ok=True)
    icd_path.write_text(json.dumps(icd_content, indent=2) + "\n", encoding="utf-8")
    return icd_path


def get_spirv_val_name() -> str:
    if sys.platform == "win32" or os.name == "nt":
        return "spirv-val.exe"
    return "spirv-val"


def _sync_auxiliary_binaries(prefix: Path, lib_path: Path) -> None:
    if sys.platform == "win32" or os.name == "nt":
        loader_fallback = prefix / "vulkan-1.dll"
        if not loader_fallback.is_file():
            shutil.copy2(lib_path, loader_fallback)
        if (
            (os.environ.get("GITHUB_ACTIONS") == "true" or os.environ.get("CI") == "true")
            and lib_path.stat().st_size > 1024 * 1024
        ):
            try:
                py_dir = Path(sys.executable).resolve().parent
                shutil.copy2(lib_path, py_dir / "vulkan-1.dll")
            except OSError:
                pass
            try:
                import winreg

                icd_win_path = str((prefix / "vk_swiftshader_icd.json").resolve())
                with winreg.CreateKeyEx(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"SOFTWARE\Khronos\Vulkan\Drivers",
                    0,
                    winreg.KEY_SET_VALUE | winreg.KEY_WOW64_64KEY,
                ) as key:
                    winreg.SetValueEx(key, icd_win_path, 0, winreg.REG_DWORD, 0)
            except OSError:
                pass
    spirv_val_name = get_spirv_val_name()
    spirv_val_src = prefix / spirv_val_name
    tools_bin = ROOT / ".tools" / "bin"
    if spirv_val_src.is_file() and tools_bin.is_dir():
        target_spirv_val = tools_bin / spirv_val_name
        if not target_spirv_val.is_file():
            shutil.copy2(spirv_val_src, target_spirv_val)


def check_fast_path(prefix: Path) -> Path | None:
    lib_name = get_library_name()
    lib_path = prefix / lib_name
    if lib_path.is_file() and lib_path.stat().st_size > 0:
        icd_path = prefix / "vk_swiftshader_icd.json"
        write_icd_json(icd_path, lib_path)
        _sync_auxiliary_binaries(prefix, lib_path)
        return icd_path
    return None


def setup_swiftshader(prefix: Path, build_dir: Path | None = None, jobs: int | None = None) -> Path:
    prefix = prefix.resolve()
    fast_icd = check_fast_path(prefix)
    if fast_icd is not None:
        return fast_icd

    rev_file = ROOT / "tools" / "swiftshader-revision.txt"
    if not rev_file.is_file():
        raise SystemExit(f"Missing pinned revision file: {rev_file}")
    revision = rev_file.read_text(encoding="utf-8").strip()
    if not revision:
        raise SystemExit(f"Empty revision hash in {rev_file}")

    num_jobs = jobs if jobs is not None and jobs > 0 else min(os.cpu_count() or 1, 8)
    source = ROOT / ".toolchains" / "swiftshader"
    build = build_dir.resolve() if build_dir else (ROOT / ".toolchains" / "swiftshader-build")

    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        try:
            run("git", "init", source)
            run("git", "-C", source, "remote", "add", "origin", "https://github.com/google/swiftshader.git")
            run("git", "-C", source, "fetch", "--depth", "1", "--filter=blob:none", "origin", revision)
            run("git", "-C", source, "checkout", "--detach", "FETCH_HEAD")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"Failed to fetch pinned SwiftShader sources at {revision}: {exc}") from exc

    cmake_lists = source / "CMakeLists.txt"
    if cmake_lists.is_file():
        cmake_text = cmake_lists.read_text(encoding="utf-8", errors="replace")
        patched = cmake_text.replace(
            "set(SPIRV_SKIP_EXECUTABLES TRUE CACHE BOOL \"\" FORCE)",
            "set(SPIRV_SKIP_EXECUTABLES FALSE CACHE BOOL \"\" FORCE)",
        )
        if patched != cmake_text:
            cmake_lists.write_text(patched, encoding="utf-8")

    cmake_args = [
        "cmake",
        "-S",
        source,
        "-B",
        build,
        *(["-G", "Ninja"] if shutil.which("ninja") else []),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DSWIFTSHADER_BUILD_VULKAN=ON",
        "-DSWIFTSHADER_BUILD_TESTS=OFF",
        "-DSWIFTSHADER_BUILD_BENCHMARKS=OFF",
        "-DSWIFTSHADER_BUILD_SAMPLES=OFF",
        "-DSWIFTSHADER_BUILD_PVR=OFF",
        "-DSWIFTSHADER_WARNINGS_AS_ERRORS=OFF",
        "-DSPIRV_SKIP_EXECUTABLES=OFF",
    ]
    try:
        run(*cmake_args)
        try:
            run("cmake", "--build", build, "--config", "Release", "--target", "vk_swiftshader", "spirv-val", "--parallel", str(num_jobs))
        except subprocess.CalledProcessError:
            run("cmake", "--build", build, "--config", "Release", "--target", "vk_swiftshader", "--parallel", str(num_jobs))
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"SwiftShader build failed: {exc}") from exc

    lib_name = get_library_name()
    candidates = list(build.rglob(lib_name))
    if not candidates:
        raise SystemExit(f"Build succeeded but target library '{lib_name}' not found under {build}")

    built_lib = candidates[0]
    prefix.mkdir(parents=True, exist_ok=True)
    target_lib = prefix / lib_name
    shutil.copy2(built_lib, target_lib)

    spirv_val_name = get_spirv_val_name()
    spirv_val_candidates = list(build.rglob(spirv_val_name))
    if spirv_val_candidates:
        shutil.copy2(spirv_val_candidates[0], prefix / spirv_val_name)

    icd_path = prefix / "vk_swiftshader_icd.json"
    write_icd_json(icd_path, target_lib)
    _sync_auxiliary_binaries(prefix, target_lib)
    return icd_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Set up pinned SwiftShader Vulkan ICD with fast cache validation."
    )
    parser.add_argument(
        "--prefix",
        type=Path,
        default=ROOT / ".sdk" / "swiftshader",
        help="Target installation directory for SwiftShader library and ICD JSON (default: .sdk/swiftshader)",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Optional CMake build directory on cache miss",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(os.cpu_count() or 1, 8),
        help="Parallel build concurrency on cache miss",
    )
    args = parser.parse_args()

    icd_path = setup_swiftshader(args.prefix, args.build_dir, args.jobs)
    print(icd_path.resolve().as_posix())


if __name__ == "__main__":
    main()
