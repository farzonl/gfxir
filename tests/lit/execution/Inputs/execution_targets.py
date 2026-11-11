import os
from pathlib import Path


def targets_from_environment(name: str) -> list[str]:
    try:
        value = os.environ[name]
    except KeyError as error:
        raise ValueError(f"environment variable {name} is not set") from error
    return parse_targets(value)


def parse_targets(value: str) -> list[str]:
    targets = value.split(",")
    if not targets or any(target not in ("vulkan", "directx", "metal") for target in targets):
        raise ValueError(f"targets must be a comma-separated list of vulkan, directx, and/or metal: {value!r}")
    if len(set(targets)) != len(targets):
        raise ValueError(f"targets must not contain duplicates: {value!r}")
    return targets


def target_output_path(prefix: Path, target: str) -> Path:
    return prefix.with_name(f"{prefix.stem}-{target}{prefix.suffix}")
