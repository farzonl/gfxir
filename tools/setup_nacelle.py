#!/usr/bin/env python3
"""Install the pinned Nacelle binary SDK for Linux x64 or macOS arm64."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import platform
import shutil
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.0.1"
RELEASES = {
    ("Linux", "x86_64"): ("linux-x64", "44b4a22dd4eabd40d49c50690a21463a0a5d650887f333df4f56030357b452b1"),
    ("Darwin", "arm64"): ("macos-arm64", "03c1176523eb14a6c0c56d9c358edf4abec1ac1b25864f9603feff2d664201a9"),
}


def install(destination: Path) -> None:
    host = (platform.system(), platform.machine())
    if host not in RELEASES:
        raise RuntimeError(f"Nacelle v{VERSION} has no supported release for {host[0]} {host[1]}")
    package_platform, expected = RELEASES[host]
    package = f"nacelle-{VERSION}-{package_platform}"
    url = f"https://github.com/farzonl/Nacelle/releases/download/v{VERSION}/{package}.tar.gz"
    with tempfile.TemporaryDirectory(prefix="gfx-nacelle-") as tmp:
        archive = Path(tmp) / f"{package}.tar.gz"
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            digest = hashlib.file_digest(source, "sha256").hexdigest()
        if digest != expected:
            raise RuntimeError(f"Nacelle archive SHA256 mismatch: expected {expected}, got {digest}")
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(Path(tmp) / "sdk", filter="data")
        sdk = Path(tmp) / "sdk" / package
        suffix = ".dylib" if host[0] == "Darwin" else ".so"
        for required in ("include/nacelle/nacelle.h", f"lib/libnacelle_d3d12{suffix}"):
            if not (sdk / required).is_file():
                raise RuntimeError(f"Nacelle archive is missing {required}")
        shutil.copytree(sdk, destination, dirs_exist_ok=True)
    print(f"Installed Nacelle v{VERSION} to {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", type=Path, default=ROOT / ".sdk" / "nacelle")
    args = parser.parse_args()
    install(args.prefix.resolve())


if __name__ == "__main__":
    main()
