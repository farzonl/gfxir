"""Tests for MiniGfx engine."""

from __future__ import annotations

import sys
from pathlib import Path

# Automatically inject repository root and src into sys.path.
# Do NOT inject `examples/` directly because `examples/math.py` would shadow the Python stdlib `math` module.
_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parents[2]
_SRC_DIR = _REPO_ROOT / "src"

for _p in (_REPO_ROOT, _SRC_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)
