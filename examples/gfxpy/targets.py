"""Platform target selection and error types for gfxpy."""

from __future__ import annotations

import sys


class GfxPyError(Exception):
    """Base error for gfxpy."""


class UnsupportedTargetError(GfxPyError, ValueError):
    """Raised when a platform or explicit backend target is unsupported."""


class UnsupportedDTypeError(GfxPyError, TypeError):
    """Raised when an unsupported dtype is requested."""


class UnsupportedFusionError(GfxPyError, TypeError):
    """Raised when an unsupported operation or value is used inside gp.fuse()."""


class ShapeMismatchError(GfxPyError, ValueError):
    """Raised when array shapes or lengths are incompatible."""


class CrossDeviceError(GfxPyError, ValueError):
    """Raised when an operation combines arrays from different devices."""


class DeviceClosedError(GfxPyError, RuntimeError):
    """Raised when using a device after it has been closed."""


class ArrayClosedError(GfxPyError, RuntimeError):
    """Raised when using an array after it or its device has been closed."""


class BackendInitializationError(GfxPyError, RuntimeError):
    """Raised when a selected GPU backend fails to initialize."""

    def __init__(self, target: str, cause: BaseException | str) -> None:
        self.target = target
        self.cause = cause
        super().__init__(
            f"Failed to initialize '{target}' backend: {cause}. "
            "No fallback to CPU or another backend will be performed."
        )


class BackendNotImplementedError(GfxPyError, NotImplementedError):
    """Raised when a target backend does not yet have a persistent GPU runtime attached."""

    def __init__(self, target: str, operation: str) -> None:
        self.target = target
        self.operation = operation
        super().__init__(
            f"Persistent GPU runtime for target '{target}' is not attached "
            f"(cannot execute '{operation}'). No CPU fallback is performed."
        )


SUPPORTED_TARGETS = ("metal", "vulkan", "directx")


def default_target(platform_name: str | None = None) -> str:
    """Select the default GPU backend for the given or current OS platform."""
    plat = (sys.platform if platform_name is None else platform_name).lower()
    if plat == "darwin":
        return "metal"
    if plat == "win32":
        return "directx"
    if plat.startswith("linux"):
        return "vulkan"
    raise UnsupportedTargetError(
        f"No default GPU target is defined for platform '{plat}'. "
        f"Supported targets: {', '.join(SUPPORTED_TARGETS)}."
    )


def normalize_target(
    target: str | None = None,
    *,
    platform_name: str | None = None,
) -> str:
    """Resolve None/'auto' to the platform default or validate an explicit target."""
    if target is None or (isinstance(target, str) and target.strip().lower() == "auto"):
        return default_target(platform_name)
    if not isinstance(target, str):
        raise UnsupportedTargetError(
            f"Target must be a string or None, got {type(target).__name__}."
        )
    normalized = target.strip().lower()
    if normalized not in SUPPORTED_TARGETS:
        raise UnsupportedTargetError(
            f"Unsupported target '{target}'. Expected one of: {', '.join(SUPPORTED_TARGETS)}."
        )
    return normalized
