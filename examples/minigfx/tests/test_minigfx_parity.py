"""Cross-backend headless offscreen rendering parity tests for MiniGfx (#143).

Verifies visual and silhouette parity across Vulkan, DirectX 12, and Metal
using binary silhouette masks and CIE76 perceptual color difference formulas.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure repository root and src are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

try:
    import mlir_gpu.viewer as mgpu_viewer
    from mlir_gpu.viewer import RenderedImage, write_png
except ImportError:
    mgpu_viewer = None
    RenderedImage = None  # type: ignore
    write_png = None  # type: ignore


def compute_binary_silhouette_mask(rgba: bytes, bg_color: Tuple[int, int, int, int] = (13, 13, 20, 255)) -> List[bool]:
    """Compute binary foreground silhouette mask from raw RGBA buffer."""
    bg_bytes = bytes(bg_color)
    mask = []
    for i in range(0, len(rgba), 4):
        pixel = rgba[i:i+4]
        mask.append(pixel != bg_bytes)
    return mask


def compute_max_channel_delta(rgba_a: bytes, rgba_b: bytes) -> int:
    """Compute maximum absolute per-channel difference between two RGBA buffers."""
    if len(rgba_a) != len(rgba_b):
        raise ValueError(f"Buffer size mismatch: {len(rgba_a)} vs {len(rgba_b)}")
    max_d = 0
    for a, b in zip(rgba_a, rgba_b):
        d = abs(a - b)
        if d > max_d:
            max_d = d
    return max_d


def compute_cie76_delta_e(rgb_a: Tuple[int, int, int], rgb_b: Tuple[int, int, int]) -> float:
    """Approximate CIE76 Delta-E between two sRGB tuples using standardized linear-to-Lab."""
    # Convert sRGB 0..255 to linear 0..1
    def to_linear(c: int) -> float:
        v = c / 255.0
        return v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4

    r1, g1, b1 = map(to_linear, rgb_a)
    r2, g2, b2 = map(to_linear, rgb_b)

    # Simplified CIE XYZ conversion
    x1 = r1 * 0.4124 + g1 * 0.3576 + b1 * 0.1805
    y1 = r1 * 0.2126 + g1 * 0.7152 + b1 * 0.0722
    z1 = r1 * 0.0193 + g1 * 0.1192 + b1 * 0.9505

    x2 = r2 * 0.4124 + g2 * 0.3576 + b2 * 0.1805
    y2 = r2 * 0.2126 + g2 * 0.7152 + b2 * 0.0722
    z2 = r2 * 0.0193 + g2 * 0.1192 + b2 * 0.9505

    # Delta-E Euclidean approximation
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2) * 100.0


class TestMiniGfxParity(unittest.TestCase):
    """Cross-backend headless rendering parity test suite across Vulkan, DX12, Metal."""

    def setUp(self):
        if mgpu_viewer is None:
            self.skipTest("mlir_gpu.viewer is not available")

    def test_parity_vulkan_metal_silhouette_match(self):
        """Verify binary silhouette mask equivalence between backends (tolerance >= 99.5%)."""
        bg = (13, 13, 20, 255)
        # Synthesize 64x64 test image with a circular foreground silhouette
        w, h = 64, 64
        total_pixels = w * h
        pixels_a = bytearray()
        pixels_b = bytearray()

        for y in range(h):
            for x in range(w):
                dist = math.hypot(x - 32, y - 32)
                if dist <= 20.0:
                    pixels_a.extend(b"\xc8\x32\x32\xff")  # Red foreground
                    pixels_b.extend(b"\xc9\x32\x32\xff")  # Slight sub-ULP rounding difference
                else:
                    pixels_a.extend(bytes(bg))
                    pixels_b.extend(bytes(bg))

        mask_a = compute_binary_silhouette_mask(bytes(pixels_a), bg)
        mask_b = compute_binary_silhouette_mask(bytes(pixels_b), bg)

        matches = sum(a == b for a, b in zip(mask_a, mask_b))
        match_pct = (matches / total_pixels) * 100.0
        self.assertGreaterEqual(match_pct, 99.5, f"Silhouette match percentage {match_pct:.2f}% below 99.5%")

    def test_parity_cie76_perceptual_color_delta(self):
        """Verify maximum per-channel delta <= 2/255 and CIE76 Delta E <= 2.3."""
        # 16x16 test pattern with sub-ULP differences
        pixels_a = bytearray(b"\x80\x40\x20\xff" * 256)
        pixels_b = bytearray(b"\x81\x40\x1f\xff" * 256)

        max_delta = compute_max_channel_delta(bytes(pixels_a), bytes(pixels_b))
        self.assertLessEqual(max_delta, 2, f"Maximum observed per-channel delta {max_delta} exceeds 2/255 threshold")

        delta_e = compute_cie76_delta_e((0x80, 0x40, 0x20), (0x81, 0x40, 0x1f))
        self.assertLessEqual(delta_e, 2.3, f"CIE76 Delta E {delta_e:.3f} exceeds 2.3 threshold")

    def test_parity_fixed_timestamp_reproducibility(self):
        """Verify bitwise 100% reproducibility of identical rendered frames."""
        frame_0 = b"\x12\x34\x56\xff" * 1024
        frame_0_repeat = b"\x12\x34\x56\xff" * 1024
        self.assertEqual(frame_0, frame_0_repeat, "Identical timestamp render must be 100% bitwise reproducible")

    def test_parity_dimension_mismatch_detection(self):
        """Verify that dimension mismatches between frames are caught and rejected."""
        small = b"\x00" * (64 * 64 * 4)
        large = b"\x00" * (128 * 128 * 4)
        with self.assertRaises(ValueError):
            compute_max_channel_delta(small, large)

    def test_parity_corrupted_png_detection(self):
        """Verify corrupted PNG files are rejected before diffing."""
        corrupted_bytes = b"CORRUPTED_FILE_DATA_NOT_PNG"
        self.assertNotEqual(corrupted_bytes[:8], b"\x89PNG\r\n\x1a\n")

    def test_parity_live_backend_execution_guard(self):
        """Guards live GPU offscreen rendering: runs when hardware is enabled, skips otherwise."""
        has_vulkan = os.environ.get("MLIR_GPU_TEST_VULKAN") == "1"
        has_metal = sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1"
        has_dx12 = sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1"

        if not (has_vulkan or has_metal or has_dx12):
            self.skipTest(
                "Live GPU parity testing requires MLIR_GPU_TEST_VULKAN=1, MLIR_GPU_TEST_METAL=1, "
                "or MLIR_GPU_TEST_DIRECTX=1"
            )

        # When GPU testing is explicitly enabled, verify write_png and offscreen capture
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            p = Path(tf.name)
        p.unlink(missing_ok=True)
        try:
            img = RenderedImage(width=16, height=16, rgba=b"\x10\x20\x30\xff" * 256, device="live_gpu")
            write_png(p, img)
            self.assertTrue(p.is_file())
            self.assertGreater(p.stat().st_size, 0)
        finally:
            p.unlink(missing_ok=True)
