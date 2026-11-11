"""MiniGfx: Python-authored graphics and game engine proof-of-concept."""

from typing import Any

__version__ = "0.1.0"
__all__ = ["App", "__version__"]


def __getattr__(name: str) -> Any:
    if name == "App":
        from .app import App

        return App
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
