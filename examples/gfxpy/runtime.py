"""Device ownership and persistent compute backend contract for gfxpy."""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar, Token
from typing import Protocol, runtime_checkable

from .targets import (
    BackendInitializationError,
    BackendNotImplementedError,
    DeviceClosedError,
    normalize_target,
)


@runtime_checkable
class ComputeBuffer(Protocol):
    """Opaque handle to a device-resident byte buffer."""

    @property
    def size_bytes(self) -> int: ...

    @property
    def closed(self) -> bool: ...

    def close(self) -> None: ...


@runtime_checkable
class ComputeBackend(Protocol):
    """Persistent compute backend contract implemented by Vulkan/Metal/DirectX runtimes."""

    @property
    def target(self) -> str: ...

    @property
    def closed(self) -> bool: ...

    def allocate(self, size_bytes: int) -> ComputeBuffer: ...

    def upload(self, buf: ComputeBuffer, data: bytes) -> None: ...

    def download(self, buf: ComputeBuffer) -> bytes: ...

    def binary_op(
        self,
        op: str,
        lhs: ComputeBuffer,
        rhs: ComputeBuffer,
        out: ComputeBuffer,
        length: int,
    ) -> None: ...

    def scalar_op(
        self,
        op: str,
        lhs: ComputeBuffer,
        scalar: float,
        out: ComputeBuffer,
        length: int,
    ) -> None: ...

    def close(self) -> None: ...


class UninitializedBackend:
    """Placeholder backend until a persistent GPU runtime is attached for the target."""

    def __init__(self, target: str) -> None:
        self._target = target
        self._closed = False

    @property
    def target(self) -> str:
        return self._target

    @property
    def closed(self) -> bool:
        return self._closed

    def allocate(self, size_bytes: int) -> ComputeBuffer:
        raise BackendNotImplementedError(self._target, "allocate")

    def upload(self, buf: ComputeBuffer, data: bytes) -> None:
        raise BackendNotImplementedError(self._target, "upload")

    def download(self, buf: ComputeBuffer) -> bytes:
        raise BackendNotImplementedError(self._target, "download")

    def binary_op(
        self,
        op: str,
        lhs: ComputeBuffer,
        rhs: ComputeBuffer,
        out: ComputeBuffer,
        length: int,
    ) -> None:
        raise BackendNotImplementedError(self._target, f"binary_op:{op}")

    def scalar_op(
        self,
        op: str,
        lhs: ComputeBuffer,
        scalar: float,
        out: ComputeBuffer,
        length: int,
    ) -> None:
        raise BackendNotImplementedError(self._target, f"scalar_op:{op}")

    def close(self) -> None:
        self._closed = True


_ACTIVE_DEVICE: ContextVar[Device | None] = ContextVar("gfxpy_active_device", default=None)


class Device:
    """Represents an explicit GPU target context and owns its compute backend."""

    def __init__(
        self,
        target: str | None = None,
        *,
        backend: ComputeBackend | None = None,
        backend_factory: Callable[[str], ComputeBackend] | None = None,
        platform_name: str | None = None,
        validation: bool = False,
        provider: str | None = None,
        toolchain: object | None = None,
    ) -> None:
        resolved = normalize_target(target, platform_name=platform_name)
        self._target = resolved
        self._closed = False
        self._tokens: list[Token[Device | None]] = []

        if backend is not None and backend_factory is not None:
            raise ValueError("Specify at most one of 'backend' or 'backend_factory'.")

        if backend is not None:
            if backend.target != resolved:
                raise BackendInitializationError(
                    resolved,
                    f"Provided backend targets '{backend.target}', expected '{resolved}'.",
                )
            self._backend: ComputeBackend = backend
        elif backend_factory is not None:
            try:
                created = backend_factory(resolved)
            except BackendInitializationError:
                raise
            except Exception as exc:
                raise BackendInitializationError(resolved, exc) from exc
            self._backend = created
        else:
            from .gpu_backend import GfxIRComputeBackend

            try:
                self._backend = GfxIRComputeBackend(
                    resolved,
                    validation=validation,
                    provider=provider,
                    toolchain=toolchain,  # type: ignore[arg-type]
                )
            except BackendInitializationError:
                raise
            except Exception as exc:
                raise BackendInitializationError(resolved, exc) from exc

    @property
    def target(self) -> str:
        return self._target

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def backend(self) -> ComputeBackend:
        self.ensure_open()
        return self._backend

    def ensure_open(self) -> None:
        if self._closed or self._backend.closed:
            raise DeviceClosedError(f"Device '{self._target}' has been closed.")

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._backend.close()

    def __enter__(self) -> Device:
        self.ensure_open()
        self._tokens.append(_ACTIVE_DEVICE.set(self))
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._tokens:
            _ACTIVE_DEVICE.reset(self._tokens.pop())

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"Device(target={self._target!r}, state={state!r})"


def device(
    target: str | None = None,
    *,
    backend: ComputeBackend | None = None,
    backend_factory: Callable[[str], ComputeBackend] | None = None,
    platform_name: str | None = None,
    validation: bool = False,
    provider: str | None = None,
    toolchain: object | None = None,
) -> Device:
    """Create a Device for the selected or platform-default target."""
    return Device(
        target,
        backend=backend,
        backend_factory=backend_factory,
        platform_name=platform_name,
        validation=validation,
        provider=provider,
        toolchain=toolchain,
    )


def current_device() -> Device:
    """Return the active Device in the current context, or create a default one."""
    active = _ACTIVE_DEVICE.get()
    if active is not None:
        active.ensure_open()
        return active
    return Device()
