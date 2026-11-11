"""MiniGfx sample applications and demonstrations."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model_viewer import ModelViewerApp


def __getattr__(name: str):
    if name == "ModelViewerApp":
        from .model_viewer import ModelViewerApp

        return ModelViewerApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelViewerApp"]
