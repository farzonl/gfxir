"""Pytest configuration, fixtures, and execution probes for MiniGfx E2E tests.

Provides hermetic temporary directories, procedural synthetic OBJ assets,
camera/scene rigs, hardware/backend capability probes, and dual-runner compatibility.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Generator, Tuple

# Automatically ensure repo root and src are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

# Try importing pytest for fixture decoration; provide shims if pytest is absent
try:
    import pytest
    _HAVE_PYTEST = True
except ImportError:
    _HAVE_PYTEST = False
    pytest = None  # type: ignore

def fixture(*args, **kwargs):
    """Decorator compatible with pytest if installed; no-op otherwise."""
    if _HAVE_PYTEST:
        return pytest.fixture(*args, **kwargs)
    def decorator(fn):
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Procedural Assets & Synthetic Corpi
# ---------------------------------------------------------------------------

SAMPLE_CUBE_OBJ = """# Unit cube centered at origin
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v -0.5  0.5  0.5
v  0.5  0.5  0.5
v -0.5  0.5 -0.5
v  0.5  0.5 -0.5
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
vn  0.0  0.0  1.0
vn  0.0  1.0  0.0
vn  0.0  0.0 -1.0
vn  0.0 -1.0  0.0
vn  1.0  0.0  0.0
vn -1.0  0.0  0.0
vt 0.0 0.0
vt 1.0 0.0
vt 1.0 1.0
vt 0.0 1.0
# 6 quad faces
f 1/1/1 2/2/1 4/3/1 3/4/1
f 3/1/2 4/2/2 6/3/2 5/4/2
f 5/1/3 6/2/3 8/3/3 7/4/3
f 7/1/4 8/2/4 2/3/4 1/4/4
f 2/1/5 8/2/5 6/3/5 4/4/5
f 7/1/6 1/2/6 3/3/6 5/4/6
"""

SAMPLE_TRI_OBJ = """# Minimal single triangle
v  0.0  0.5  0.0
v -0.5 -0.5  0.0
v  0.5 -0.5  0.0
vn 0.0  0.0  1.0
vt 0.5  1.0
vt 0.0  0.0
vt 1.0  0.0
f 1/1/1 2/2/1 3/3/1
"""

MALFORMED_OBJ_CORPUS = {
    "empty": "",
    "comments_only": "# Only comments\n# Second comment\n",
    "missing_face_indices": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2\n",
    "out_of_bounds_index": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 999\n",
    "zero_index": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n",
    "non_numeric_vertex": "v 0.0 foo 1.0\nf 1 2 3\n",
    "negative_relative_indices": "v 0 0 0\nv 1 0 0\nv 0 1 0\nf -1 -2 -3\n",
    "quad_face": "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n",
    "polygon_6gon": "v 0 0 0\nv 1 0 0\nv 2 1 0\nv 1 2 0\nv 0 2 0\nv -1 1 0\nf 1 2 3 4 5 6\n",
}

GOLDEN_PARITY_RULES_YAML = """# Golden parity comparison rules matching tests/test_imgdiff.py
Mode: CIE76
MaxDeltaE: 2.3
MaxPixelPercent: 0.2
MaxAlphaDelta: 0
"""


# ---------------------------------------------------------------------------
# Pytest Fixtures
# ---------------------------------------------------------------------------

@fixture
def temp_dir() -> Generator[Path, None, None]:
    """Isolated temporary directory fixture automatically cleaned up after test."""
    td = tempfile.mkdtemp(prefix="minigfx_test_")
    p = Path(td)
    try:
        yield p
    finally:
        shutil.rmtree(td, ignore_errors=True)


@fixture(scope="session")
def sample_cube_obj() -> str:
    """Session-scoped standard 8-vertex, 6-quad Wavefront OBJ string."""
    return SAMPLE_CUBE_OBJ


@fixture(scope="session")
def sample_tri_obj() -> str:
    """Session-scoped minimal single-triangle Wavefront OBJ string."""
    return SAMPLE_TRI_OBJ


@fixture(scope="session")
def malformed_obj_corpus() -> Dict[str, str]:
    """Session-scoped corpus of malformed and corner-case Wavefront OBJ strings."""
    return MALFORMED_OBJ_CORPUS


@fixture(scope="session")
def parity_rules(tmp_path_factory) -> Path:
    """Temporary YAML rule file for CIE76 visual parity testing."""
    fn = tmp_path_factory.mktemp("rules") / "golden-rules.yaml"
    fn.write_text(GOLDEN_PARITY_RULES_YAML, encoding="utf-8")
    return fn


@fixture(scope="session")
def backend_probes() -> Dict[str, bool]:
    """Probes host platform graphics backends and environment variable activations."""
    has_vulkan = bool(os.environ.get("MLIR_GPU_TEST_VULKAN") == "1")
    has_metal = bool(sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL", "1") == "1")
    has_directx = bool(sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1")
    has_warp = bool(sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_WARP") == "1")
    return {
        "vulkan": has_vulkan,
        "metal": has_metal,
        "directx": has_directx,
        "warp": has_warp,
    }


# ---------------------------------------------------------------------------
# Test Assertion Helpers
# ---------------------------------------------------------------------------

def assert_std430_aligned(buffer: bytes | bytearray, alignment: int = 16) -> None:
    """Assert that raw buffer length is a positive multiple of required alignment."""
    assert len(buffer) > 0, "Buffer cannot be empty"
    assert len(buffer) % alignment == 0, (
        f"Buffer length {len(buffer)} is not a multiple of std430 alignment {alignment}"
    )


def assert_channel_delta_within_threshold(
    rgba_a: bytes, rgba_b: bytes, max_delta: int = 2
) -> Tuple[bool, int, float]:
    """Compute maximum per-channel delta and percentage of differing pixels."""
    if len(rgba_a) != len(rgba_b):
        raise ValueError(f"Byte buffer length mismatch: {len(rgba_a)} vs {len(rgba_b)}")
    
    max_observed = 0
    differing_pixels = 0
    total_pixels = len(rgba_a) // 4
    
    for i in range(0, len(rgba_a), 4):
        p_diff = False
        for c in range(4):
            diff = abs(rgba_a[i + c] - rgba_b[i + c])
            if diff > max_observed:
                max_observed = diff
            if diff > max_delta:
                p_diff = True
        if p_diff:
            differing_pixels += 1
            
    pct_diff = (differing_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0
    passed = max_observed <= max_delta
    return passed, max_observed, pct_diff
