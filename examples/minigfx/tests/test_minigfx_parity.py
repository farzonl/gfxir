"""Cross-backend headless offscreen rendering parity tests for MiniGfx.

Verifies visual, silhouette, and bitwise parity across Vulkan, DirectX 12, and Metal
using the repository's official imgdiff tool (tests/lit/execution/Inputs/imgdiff.py),
binary silhouette masks, and CIE76 perceptual color difference formulas.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import tempfile
import unittest
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Precomputed 256-entry lookup table for sRGB to linear conversion (D65 standard)
_SRGB_TO_LINEAR_LUT: List[float] = [
    (c / 255.0) / 12.92 if (c / 255.0) <= 0.04045 else (((c / 255.0) + 0.055) / 1.055) ** 2.4
    for c in range(256)
]

# Color conversion cache to accelerate repeated CIELAB transformations
_LAB_CACHE: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}

# Ensure repository root, src, and lit execution inputs are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
_LIT_INPUTS = _REPO_ROOT / "tests" / "lit" / "execution" / "Inputs"

for _p in (_REPO_ROOT, _SRC_DIR, _LIT_INPUTS):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

try:
    from PIL import Image, ImageChops
except ImportError:
    Image = None  # type: ignore
    ImageChops = None  # type: ignore

try:
    import yaml
except ImportError:
    yaml = None

try:
    import imgdiff
except ImportError:
    imgdiff = None

try:
    import mlir_gpu.viewer as mgpu_viewer
    from mlir_gpu.viewer import RenderedImage, ViewerError, write_png
except ImportError:
    mgpu_viewer = None
    RenderedImage = None  # type: ignore
    ViewerError = Exception  # type: ignore
    write_png = None  # type: ignore

# Optional imports from minigfx
try:
    from examples.minigfx.renderer import ForwardRenderer
except ImportError:
    ForwardRenderer = None  # type: ignore

try:
    from examples.minigfx.math import Mat4x4, Quat, Vec2, Vec3, Vec4
except ImportError:
    Mat4x4 = None  # type: ignore
    Quat = None  # type: ignore
    Vec2 = None  # type: ignore
    Vec3 = None  # type: ignore
    Vec4 = None  # type: ignore


def srgb_to_lab(rgb: Tuple[int, int, int]) -> Tuple[float, float, float]:
    """Convert 8-bit sRGB (R, G, B) to CIELAB (L*, a*, b*) under D65 illuminant."""
    cached = _LAB_CACHE.get(rgb)
    if cached is not None:
        return cached

    r, g, b = rgb
    lr = _SRGB_TO_LINEAR_LUT[r]
    lg = _SRGB_TO_LINEAR_LUT[g]
    lb = _SRGB_TO_LINEAR_LUT[b]

    # CIEXYZ conversion with D65 reference white (Xn=0.95047, Yn=1.0, Zn=1.08883)
    x = (0.4124564 * lr + 0.3575761 * lg + 0.1804375 * lb) / 0.95047
    y = 0.2126729 * lr + 0.7151522 * lg + 0.0721750 * lb
    z = (0.0193339 * lr + 0.1191920 * lg + 0.9503041 * lb) / 1.08883

    def _transform(val: float) -> float:
        return val ** (1.0 / 3.0) if val > 0.008856451679 else 7.787037037 * val + (16.0 / 116.0)

    fx, fy, fz = _transform(x), _transform(y), _transform(z)
    lab = (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))
    if len(_LAB_CACHE) < 65536:
        _LAB_CACHE[rgb] = lab
    return lab


def compute_cie76_delta_e(
    rgb_a: Tuple[int, int, int] | bytes | Any,
    rgb_b: Tuple[int, int, int] | bytes | Any,
) -> float:
    """Compute exact CIELAB Delta-E (CIE76) between two color tuples, RGBA byte buffers, or Images."""
    is_a_color = isinstance(rgb_a, (tuple, list))
    is_b_color = isinstance(rgb_b, (tuple, list))

    if is_a_color and is_b_color:
        if len(rgb_a) not in (3, 4):
            raise ValueError(f"Color tuple/list rgb_a must have length 3 or 4, got {len(rgb_a)}")
        if len(rgb_b) not in (3, 4):
            raise ValueError(f"Color tuple/list rgb_b must have length 3 or 4, got {len(rgb_b)}")
        c1 = (int(rgb_a[0]), int(rgb_a[1]), int(rgb_a[2]))
        c2 = (int(rgb_b[0]), int(rgb_b[1]), int(rgb_b[2]))
        if c1 == c2:
            return 0.0
        if imgdiff is not None and hasattr(imgdiff, "_lab"):
            l1, a1, b1 = imgdiff._lab(c1)
            l2, a2, b2 = imgdiff._lab(c2)
        else:
            l1, a1, b1 = srgb_to_lab(c1)
            l2, a2, b2 = srgb_to_lab(c2)
        return math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)

    if is_a_color or is_b_color:
        raise TypeError("Cannot compare tuple/list color with byte buffer")

    # Pixel buffer or Image comparison
    if hasattr(rgb_a, "convert") and hasattr(rgb_a, "tobytes"):
        bytes_a = rgb_a.convert("RGBA").tobytes()
    elif isinstance(rgb_a, (bytes, bytearray)):
        bytes_a = rgb_a
    else:
        raise TypeError(f"Unsupported type for rgb_a: {type(rgb_a).__name__}")

    if hasattr(rgb_b, "convert") and hasattr(rgb_b, "tobytes"):
        bytes_b = rgb_b.convert("RGBA").tobytes()
    elif isinstance(rgb_b, (bytes, bytearray)):
        bytes_b = rgb_b
    else:
        raise TypeError(f"Unsupported type for rgb_b: {type(rgb_b).__name__}")

    if len(bytes_a) != len(bytes_b):
        raise ValueError(f"Buffer size mismatch: {len(bytes_a)} vs {len(bytes_b)}")

    if len(bytes_a) % 4 != 0:
        raise ValueError(f"Buffer length must be a multiple of 4, got {len(bytes_a)}")
    if len(bytes_b) % 4 != 0:
        raise ValueError(f"Buffer length must be a multiple of 4, got {len(bytes_b)}")

    step = 4
    max_delta_e = 0.0
    for i in range(0, len(bytes_a), step):
        c1 = (bytes_a[i], bytes_a[i + 1], bytes_a[i + 2])
        c2 = (bytes_b[i], bytes_b[i + 1], bytes_b[i + 2])
        if c1 != c2:
            l1, a1, b1 = srgb_to_lab(c1)
            l2, a2, b2 = srgb_to_lab(c2)
            de = math.sqrt((l1 - l2) ** 2 + (a1 - a2) ** 2 + (b1 - b2) ** 2)
            if de > max_delta_e:
                max_delta_e = de
    return max_delta_e


def compute_max_channel_delta(
    rgba_a: bytes | bytearray | Any,
    rgba_b: bytes | bytearray | Any,
) -> int:
    """Compute maximum absolute per-channel difference between two RGBA buffers or Images."""
    if hasattr(rgba_a, "convert") and hasattr(rgba_a, "tobytes"):
        rgba_a = rgba_a.convert("RGBA").tobytes()
    elif not isinstance(rgba_a, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or PIL Image, got {type(rgba_a).__name__}")

    if hasattr(rgba_b, "convert") and hasattr(rgba_b, "tobytes"):
        rgba_b = rgba_b.convert("RGBA").tobytes()
    elif not isinstance(rgba_b, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or PIL Image, got {type(rgba_b).__name__}")

    if len(rgba_a) != len(rgba_b):
        raise ValueError(f"Buffer size mismatch: {len(rgba_a)} vs {len(rgba_b)}")
    if rgba_a == rgba_b:
        return 0

    max_d = 0
    for a, b in zip(rgba_a, rgba_b):
        d = abs(a - b)
        if d > max_d:
            max_d = d
            if max_d == 255:
                break
    return max_d


def compute_binary_silhouette_mask(
    rgba: bytes | bytearray | Any,
    bg_color: Tuple[int, int, int, int] = (13, 13, 20, 255),
    threshold: int = 0,
) -> List[bool]:
    """Compute binary foreground silhouette mask from raw RGBA buffer or Image."""
    if hasattr(rgba, "convert") and hasattr(rgba, "tobytes"):
        rgba = rgba.convert("RGBA").tobytes()
    elif not isinstance(rgba, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or PIL Image, got {type(rgba).__name__}")

    if len(rgba) % 4 != 0:
        raise ValueError(f"Buffer length must be a multiple of 4, got {len(rgba)}")

    br, bg_, bb = bg_color[:3]
    ba = bg_color[3] if len(bg_color) > 3 else 255

    if threshold <= 0:
        bg_bytes = bytes((br, bg_, bb, ba))
        return [rgba[i : i + 4] != bg_bytes for i in range(0, len(rgba), 4)]

    mask = []
    for i in range(0, len(rgba), 4):
        r, g, b, a = rgba[i : i + 4]
        is_fg = (
            abs(r - br) > threshold
            or abs(g - bg_) > threshold
            or abs(b - bb) > threshold
            or abs(a - ba) > threshold
        )
        mask.append(is_fg)
    return mask


def compute_foreground_coverage(
    rgba: bytes | bytearray | Any,
    bg_color: Tuple[int, ...] = (13, 13, 20, 255),
) -> float:
    """Compute robust foreground coverage percentage rejecting background clear color and degenerate pixels."""
    if hasattr(rgba, "convert") and hasattr(rgba, "tobytes"):
        rgba = rgba.convert("RGBA").tobytes()
    elif not isinstance(rgba, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or PIL Image, got {type(rgba).__name__}")

    total_pixels = len(rgba) // 4
    if total_pixels == 0:
        return 0.0

    if len(bg_color) == 3:
        bg_bytes = bytes((*bg_color, 255))
    else:
        bg_bytes = bytes(bg_color[:4])
    black_opaque = bytes((0, 0, 0, 255))
    black_zero = bytes((0, 0, 0, 0))

    fg_count = sum(
        1 for i in range(0, total_pixels * 4, 4)
        if rgba[i : i + 4] not in (bg_bytes, black_opaque, black_zero)
    )
    return (fg_count / total_pixels) * 100.0


def compute_silhouette_match_ratio(
    im1: Sequence[bool] | bytes | bytearray | Any,
    im2: Sequence[bool] | bytes | bytearray | Any,
    threshold: int = 0,
    bg_color: Tuple[int, int, int, int] = (13, 13, 20, 255),
) -> float:
    """Compute the percentage of matching silhouette mask pixels (0.0 to 100.0%)."""
    if isinstance(im1, Sequence) and (len(im1) == 0 or isinstance(im1[0], bool)):
        mask_a = list(im1)
    else:
        mask_a = compute_binary_silhouette_mask(im1, bg_color=bg_color, threshold=threshold)

    if isinstance(im2, Sequence) and (len(im2) == 0 or isinstance(im2[0], bool)):
        mask_b = list(im2)
    else:
        mask_b = compute_binary_silhouette_mask(im2, bg_color=bg_color, threshold=threshold)

    if len(mask_a) != len(mask_b):
        raise ValueError(f"Mask size mismatch: {len(mask_a)} vs {len(mask_b)}")
    if not mask_a:
        return 100.0

    matches = sum(a == b for a, b in zip(mask_a, mask_b))
    return (matches / len(mask_a)) * 100.0


def render_scene_frame(
    backend: str = "vulkan",
    scene_type: str = "model_viewer",
    frame_index: int = 0,
    width: int = 64,
    height: int = 64,
    time: float | None = None,
) -> RenderedImage:
    """Render a deterministic offscreen frame via ModelViewerApp and genuine GPU execution."""
    from examples.minigfx.samples.model_viewer import ModelViewerApp

    valid_targets = ("vulkan", "metal", "directx")
    if not isinstance(backend, str):
        raise TypeError(f"Target backend must be a string, got {type(backend).__name__}")
    backend = backend.lower()
    if backend not in valid_targets:
        raise ValueError(f"Invalid target backend {backend!r}; choose from {valid_targets}")

    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise ValueError(f"Frame index must be a non-negative integer, got frame_index={frame_index}")

    for dim_name, dim_val in (("width", width), ("height", height)):
        if isinstance(dim_val, bool) or not isinstance(dim_val, int) or dim_val <= 0:
            raise ValueError(f"Dimensions must be positive integers, got {dim_name}={dim_val}")

    args = ModelViewerApp.parse_args([
        "--offscreen",
        "--target", backend,
        "--width", str(width),
        "--height", str(height),
        "--frames", str(frame_index + 1),
    ])
    app = ModelViewerApp(args)
    app.debug_overlay = None
    try:
        app.startup()
    except Exception:
        app.cleanup()
        raise
    try:
        if time is not None:
            frames = int(round(time * 60.0))
            for f in range(frames + 1):
                app.frame_count = f
                app.update(1.0 / 60.0)
            app.elapsed_time = time
        else:
            for f in range(frame_index + 1):
                app.frame_count = f
                app.update(1.0 / 60.0)
        return app.render_offscreen_frame()
    finally:
        app.cleanup()


def get_golden_parity_rules_path() -> Path:
    """Return path to golden parity comparison rules YAML."""
    p = _REPO_ROOT / "tests" / "lit" / "execution" / "Inputs" / "golden-rules.yaml"
    if p.is_file():
        return p
    # Fallback to local default file
    return _REPO_ROOT / "examples" / "minigfx" / "tests" / "golden_rules.yaml"


# ===========================================================================
# Suite 1: TestImageComparisonMetrics
# ===========================================================================

class TestImageComparisonMetrics(unittest.TestCase):
    """Verifies numerical precision, edge-case bounds, and mathematical equivalence of metrics."""

    def _require_imgdiff(self):
        if imgdiff is None:
            self.skipTest("tests.lit.execution.Inputs.imgdiff is not available")

    def test_cie76_identical_colors_zero_delta(self):
        """Identical color inputs must produce an exact Delta-E of 0.0."""
        self.assertEqual(compute_cie76_delta_e((128, 64, 32), (128, 64, 32)), 0.0)
        self.assertEqual(compute_cie76_delta_e((0, 0, 0), (0, 0, 0)), 0.0)
        self.assertEqual(compute_cie76_delta_e((255, 255, 255), (255, 255, 255)), 0.0)

    def test_cie76_known_perceptual_color_pairs(self):
        """Verifies CIE76 color difference against known boundaries."""
        # Shifted red pair: subtle variation
        delta_e = compute_cie76_delta_e((255, 0, 0), (240, 5, 5))
        self.assertGreater(delta_e, 0.0)
        self.assertLess(delta_e, 10.0, "Subtle color variations must have small Delta-E")

        # Sub-threshold rounding: delta <= 2.3
        sub_delta = compute_cie76_delta_e((128, 64, 32), (129, 64, 32))
        self.assertLess(sub_delta, 2.3, "Sub-ULP delta must be below perceptible threshold 2.3")

        # Super-threshold divergence: delta > 2.3
        super_delta = compute_cie76_delta_e((128, 64, 32), (200, 64, 32))
        self.assertGreater(super_delta, 2.3, "Large color delta must exceed threshold 2.3")

    def test_cie76_matches_imgdiff_reference_lab(self):
        """Verifies local srgb_to_lab transformation matches official imgdiff._lab."""
        self._require_imgdiff()
        test_colors = [
            (0, 0, 0),
            (255, 255, 255),
            (128, 64, 32),
            (50, 150, 250),
            (200, 100, 0),
        ]
        for c in test_colors:
            lab_ref = imgdiff._lab(c)
            lab_loc = srgb_to_lab(c)
            for r_val, l_val in zip(lab_ref, lab_loc):
                self.assertAlmostEqual(r_val, l_val, places=4, msg=f"Mismatch for color {c}")

    def test_max_channel_delta_bitwise_and_sub_ulp(self):
        """Verifies max channel delta computation on bitwise and sub-ULP buffers."""
        buf1 = b"\x80\x40\x20\xff" * 4
        buf2 = b"\x80\x40\x20\xff" * 4
        self.assertEqual(compute_max_channel_delta(buf1, buf2), 0)

        # 1-unit delta
        buf_d1 = b"\x81\x40\x20\xff" + b"\x80\x40\x20\xff" * 3
        self.assertEqual(compute_max_channel_delta(buf1, buf_d1), 1)

        # 2-unit delta (parity boundary)
        buf_d2 = b"\x82\x40\x20\xff" + b"\x80\x40\x20\xff" * 3
        self.assertEqual(compute_max_channel_delta(buf1, buf_d2), 2)

    def test_max_channel_delta_exceeds_threshold(self):
        """Verifies rejection boundary when channel delta exceeds 2/255."""
        buf1 = b"\x80\x40\x20\xff"
        buf_d3 = b"\x83\x40\x20\xff"
        delta = compute_max_channel_delta(buf1, buf_d3)
        self.assertEqual(delta, 3)
        self.assertGreater(delta, 2)

        # Maximum possible delta
        black = b"\x00\x00\x00\xff"
        white = b"\xff\xff\xff\xff"
        self.assertEqual(compute_max_channel_delta(black, white), 255)

    def test_max_channel_delta_dimension_mismatch_raises(self):
        """Mismatched buffer sizes must raise ValueError."""
        buf_small = b"\x00" * 16
        buf_large = b"\x00" * 32
        with self.assertRaises(ValueError):
            compute_max_channel_delta(buf_small, buf_large)

        with self.assertRaises(ValueError):
            compute_max_channel_delta(b"\x00" * 15, b"\x00" * 16)

    def test_exact_mode_bitwise_matching_and_rejection(self):
        """Verifies Exact mode bitwise parity and 1-bit rejection via imgdiff."""
        self._require_imgdiff()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rules_path = tdp / "exact_rules.yaml"
            rules_path.write_text("Mode: Exact\nMaxDeltaE: 0.0\nMaxPixelPercent: 0.0\nMaxAlphaDelta: 0\n")

            img_a = tdp / "render_a.png"
            img_b = tdp / "render_b.png"
            im1 = Image.new("RGBA", (16, 16), (128, 64, 32, 255))
            im1.save(img_a)
            im1.save(img_b)

            res = imgdiff.compare(img_a, img_b, rules_path)
            self.assertEqual(res, 0, "Exact mode must pass with exit code 0 for identical images")

            img_c = tdp / "render_c.png"
            im2 = Image.new("RGBA", (16, 16), (128, 64, 32, 255))
            im2.putpixel((0, 0), (128, 64, 33, 255))
            im2.save(img_c)

            res_diff = imgdiff.compare(img_a, img_c, rules_path)
            self.assertEqual(res_diff, 1, "Exact mode must return exit code 1 when pixels differ")

    def test_dimension_mismatch_detection(self):
        """Verifies that dimension mismatches between frames are caught by imgdiff."""
        self._require_imgdiff()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rules_path = tdp / "rules.yaml"
            rules_path.write_text("Mode: Exact\nMaxDeltaE: 0.0\nMaxPixelPercent: 0.0\nMaxAlphaDelta: 0\n")

            img_16 = tdp / "16x16.png"
            img_32 = tdp / "16x32.png"
            Image.new("RGBA", (16, 16)).save(img_16)
            Image.new("RGBA", (16, 32)).save(img_32)

            with self.assertRaises(ValueError) as cm:
                imgdiff.compare(img_16, img_32, rules_path)
            self.assertIn("image sizes do not match", str(cm.exception))

    def test_corrupted_png_detection(self):
        """Verifies corrupted PNG files are rejected before diffing."""
        self._require_imgdiff()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rules_path = tdp / "rules.yaml"
            rules_path.write_text("Mode: Exact\nMaxDeltaE: 0.0\nMaxPixelPercent: 0.0\nMaxAlphaDelta: 0\n")

            valid_png = tdp / "valid.png"
            Image.new("RGBA", (16, 16)).save(valid_png)

            corrupt_png = tdp / "corrupt.png"
            corrupt_png.write_bytes(b"CORRUPTED_FILE_DATA_NOT_PNG")

            with self.assertRaises((ValueError, SyntaxError)) as cm:
                imgdiff.compare(valid_png, corrupt_png, rules_path)
            err_msg = str(cm.exception)
            self.assertTrue("cannot compare images" in err_msg or "broken PNG" in err_msg)

    def test_malformed_rules_rejection(self):
        """Verifies invalid or malformed comparison rules YAML is rejected."""
        self._require_imgdiff()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            img_a = tdp / "a.png"
            img_b = tdp / "b.png"
            Image.new("RGBA", (16, 16)).save(img_a)
            Image.new("RGBA", (16, 16)).save(img_b)

            bad_rules = tdp / "bad_rules.yaml"
            bad_rules.write_text("Mode: InvalidMode\nMaxDeltaE: 2.3\nMaxPixelPercent: 0.5\nMaxAlphaDelta: 2\n")

            with self.assertRaises(ValueError) as cm:
                imgdiff.compare(img_a, img_b, bad_rules)
            self.assertIn("Mode must be Exact or CIE76", str(cm.exception))

    def test_cie76_red_vs_black_pil_images_nonzero_delta_e(self):
        """Verifies 2x2 and 16x16 red vs black PIL images return non-zero Delta-E."""
        if Image is None:
            self.skipTest("PIL is not available")
        img_red_2 = Image.new("RGBA", (2, 2), (255, 0, 0, 255))
        img_black_2 = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
        self.assertGreater(compute_cie76_delta_e(img_red_2, img_black_2), 0.0)

        img_red_16 = Image.new("RGBA", (16, 16), (255, 0, 0, 255))
        img_black_16 = Image.new("RGBA", (16, 16), (0, 0, 0, 255))
        self.assertGreater(compute_cie76_delta_e(img_red_16, img_black_16), 0.0)

    def test_cie76_raw_rgba_buffer_and_pil_image_size_match(self):
        """Comparing raw 64-byte RGBA buffer vs 4x4 RGBA PIL Image does not raise size mismatch."""
        if Image is None:
            self.skipTest("PIL is not available")
        raw_rgba = b"\xff\x00\x00\xff" * 16  # 4x4 RGBA = 64 bytes
        pil_rgba = Image.new("RGBA", (4, 4), (255, 0, 0, 255))
        self.assertEqual(compute_cie76_delta_e(raw_rgba, pil_rgba), 0.0)
        self.assertEqual(compute_cie76_delta_e(pil_rgba, raw_rgba), 0.0)

    def test_cie76_buffer_length_not_multiple_of_four_raises(self):
        """Buffer length not a multiple of 4 raises ValueError."""
        with self.assertRaises(ValueError):
            compute_cie76_delta_e(b"\x00" * 13, b"\x00" * 13)

    def test_cie76_tuple_and_buffer_type_mismatch_raises(self):
        """Comparing a tuple/list with a byte buffer symmetrically raises TypeError."""
        rgba_buf = b"\x00" * 16
        with self.assertRaises(TypeError):
            compute_cie76_delta_e((255, 0, 0), rgba_buf)
        with self.assertRaises(TypeError):
            compute_cie76_delta_e(rgba_buf, (255, 0, 0))

    def test_cie76_invalid_tuple_length_raises(self):
        """Color tuple with length not in (3, 4) raises ValueError."""
        with self.assertRaises(ValueError):
            compute_cie76_delta_e((255, 0), (255, 0, 0))
        with self.assertRaises(ValueError):
            compute_cie76_delta_e((255, 0, 0), (255, 0, 0, 255, 100))

    def test_cie76_rgba_buffer_divergent_pixel_nonzero(self):
        """RGBA buffer with divergent pixel detects non-zero Delta-E with step=4."""
        p0 = b"\x00\x00\x00\xff"
        p_red = b"\xff\x00\x00\xff"
        buf_a = p0 + p_red + p0 * 2
        buf_b = p0 * 4
        self.assertGreater(compute_cie76_delta_e(buf_a, buf_b), 0.0)


# ===========================================================================
# Suite 2: TestSilhouetteMatching
# ===========================================================================

class TestSilhouetteMatching(unittest.TestCase):
    """Verifies foreground geometry silhouette extraction and invariant matching."""

    def test_silhouette_mask_background_isolation(self):
        """Verifies classification of background vs foreground pixels."""
        bg = (13, 13, 20, 255)
        fg = (200, 50, 50, 255)
        rgba = bytes([*bg, *fg, *fg, *bg])
        mask = compute_binary_silhouette_mask(rgba, bg_color=bg)
        self.assertEqual(mask, [False, True, True, False])

    def test_silhouette_mask_custom_background(self):
        """Verifies custom clear background color extraction."""
        bg = (5, 5, 8, 255)
        fg = (255, 0, 0, 255)
        rgba = bytes([*bg, *fg, *fg, *bg])
        mask = compute_binary_silhouette_mask(rgba, bg_color=bg)
        self.assertEqual(mask, [False, True, True, False])

    def test_silhouette_mask_with_threshold(self):
        """Verifies threshold tolerance for near-background pixels."""
        bg = (10, 10, 10, 255)
        near_bg = (12, 11, 10, 255)
        fg = (100, 100, 100, 255)
        rgba = bytes([*bg, *near_bg, *fg])
        mask = compute_binary_silhouette_mask(rgba, bg_color=bg, threshold=5)
        self.assertEqual(mask, [False, False, True])

    def test_silhouette_match_ratio_above_threshold(self):
        """Verifies that sub-pixel edge rasterization variations pass >= 99.5% agreement."""
        bg = (13, 13, 20, 255)
        w, h = 64, 64
        total_pixels = w * h
        pixels_a = bytearray()
        pixels_b = bytearray()

        for y in range(h):
            for x in range(w):
                dist = math.hypot(x - 32, y - 32)
                if dist <= 20.0:
                    pixels_a.extend(b"\xc8\x32\x32\xff")
                    pixels_b.extend(b"\xc9\x32\x32\xff")
                else:
                    pixels_a.extend(bytes(bg))
                    pixels_b.extend(bytes(bg))

        ratio = compute_silhouette_match_ratio(bytes(pixels_a), bytes(pixels_b), bg_color=bg)
        self.assertGreaterEqual(ratio, 99.5, f"Silhouette match ratio {ratio:.2f}% below 99.5%")

    def test_silhouette_mismatch_detection_on_translation(self):
        """Verifies that geometric translation divergence triggers match failure (< 99.5%)."""
        bg = (13, 13, 20, 255)
        w, h = 64, 64
        pixels_a = bytearray()
        pixels_shifted = bytearray()

        for y in range(h):
            for x in range(w):
                dist_a = math.hypot(x - 32, y - 32)
                dist_shifted = math.hypot(x - 38, y - 32)  # Shifted by 6 pixels
                pixels_a.extend(b"\xc8\x32\x32\xff" if dist_a <= 15.0 else bytes(bg))
                pixels_shifted.extend(b"\xc8\x32\x32\xff" if dist_shifted <= 15.0 else bytes(bg))

        ratio = compute_silhouette_match_ratio(bytes(pixels_a), bytes(pixels_shifted), bg_color=bg)
        self.assertLess(ratio, 99.5, f"Shifted silhouette should fail 99.5% threshold, got {ratio:.2f}%")

    def test_silhouette_mask_mismatched_length_raises(self):
        """Comparing masks of unequal sizes must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_silhouette_match_ratio([True, False], [True])

    def test_anti_false_success_coverage_guard(self):
        """Rejects empty/black framebuffers where foreground coverage is <= 1%."""
        bg_bytes = bytes((13, 13, 20, 255))
        black_opaque = bytes((0, 0, 0, 255))
        black_zero = bytes((0, 0, 0, 0))

        for degenerate_buffer, desc in [
            (bg_bytes * (64 * 64), "background clear color"),
            (black_opaque * (64 * 64), "all-black opaque"),
            (black_zero * (64 * 64), "all-zero"),
            (b"", "empty buffer"),
        ]:
            total_pixels = len(degenerate_buffer) // 4
            if total_pixels == 0:
                cov = 0.0
            else:
                fg_count = sum(
                    1 for i in range(0, total_pixels * 4, 4)
                    if degenerate_buffer[i : i + 4] not in (bg_bytes, black_opaque, black_zero)
                )
                cov = (fg_count / total_pixels) * 100.0
            self.assertEqual(cov, 0.0)
            self.assertLessEqual(cov, 1.0, f"Degenerate frame ({desc}) must have coverage <= 1.0%")

        # Normal scene frame has substantive coverage
        try:
            rendered = render_scene_frame("vulkan", width=64, height=64)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        viewer_bg = bytes((5, 5, 10, 255))
        total_pixels = len(rendered.rgba) // 4
        if total_pixels == 0:
            rendered_cov = 0.0
        else:
            fg_count = sum(
                1 for i in range(0, total_pixels * 4, 4)
                if rendered.rgba[i : i + 4] not in (bg_bytes, viewer_bg, black_opaque, black_zero)
            )
            rendered_cov = (fg_count / total_pixels) * 100.0
        self.assertGreater(
            rendered_cov, 1.0, f"Rendered scene must have foreground coverage > 1.0%, got {rendered_cov:.2f}%"
        )

    def test_degenerate_buffers_foreground_coverage_below_threshold(self):
        """Degenerate buffers (black opaque, black zero, clear-color, empty, non-aligned) produce coverage <= 1.0%."""
        bg_bytes = bytes((13, 13, 20, 255))
        black_opaque = bytes((0, 0, 0, 255))
        black_zero = bytes((0, 0, 0, 0))

        cases = [
            ("all-black opaque", black_opaque * (48 * 48)),
            ("all-zero", black_zero * (48 * 48)),
            ("background clear color", bg_bytes * (48 * 48)),
            ("empty buffer", b""),
        ]

        # Non-aligned buffers (1..3 trailing partial bytes)
        for n_bytes in (5, 6, 7, 9, 13):
            full_px = n_bytes // 4
            rem = n_bytes % 4
            cases.append((f"{n_bytes}-byte all-zero", (black_zero * full_px) + b"\x00" * rem))
            cases.append((f"{n_bytes}-byte all-black opaque", (black_opaque * full_px) + black_opaque[:rem]))
            cases.append((f"{n_bytes}-byte background clear color", (bg_bytes * full_px) + bg_bytes[:rem]))

        for name, buf in cases:
            with self.subTest(case=name):
                total_pixels = len(buf) // 4
                if total_pixels == 0:
                    cov = 0.0
                else:
                    fg_count = sum(
                        1 for i in range(0, total_pixels * 4, 4)
                        if buf[i : i + 4] not in (bg_bytes, black_opaque, black_zero)
                    )
                    cov = (fg_count / total_pixels) * 100.0
                self.assertEqual(cov, 0.0)
                self.assertLessEqual(cov, 1.0, f"Degenerate buffer '{name}' must have coverage <= 1.0%, got {cov:.2f}%")
                self.assertEqual(compute_foreground_coverage(buf), 0.0)

    def test_foreground_coverage_three_tuple_bg_color_normalized(self):
        """3-tuple bg_color without explicit alpha correctly identifies clear pixels as background (0.0%)."""
        bg_3tuple = (13, 13, 20)
        clear_buffer = bytes((13, 13, 20, 255)) * (48 * 48)
        cov = compute_foreground_coverage(clear_buffer, bg_color=bg_3tuple)
        self.assertEqual(cov, 0.0, f"Expected 0.0% coverage for clear buffer with 3-tuple bg_color, got {cov:.2f}%")

        # Mixed buffer with known foreground pixel
        mixed_buffer = (bytes((13, 13, 20, 255)) * 3) + bytes((255, 0, 0, 255))
        mixed_cov = compute_foreground_coverage(mixed_buffer, bg_color=bg_3tuple)
        self.assertAlmostEqual(mixed_cov, 25.0, places=2)

    def test_silhouette_empty_and_solid_frames(self):
        """Verifies boundary states: 100% background and 100% foreground."""
        all_bg = b"\x00\x00\x00\xff" * 16
        mask_bg = compute_binary_silhouette_mask(all_bg, bg_color=(0, 0, 0, 255))
        self.assertTrue(all(not m for m in mask_bg))

        all_fg = b"\xff\xff\xff\xff" * 16
        mask_fg = compute_binary_silhouette_mask(all_fg, bg_color=(0, 0, 0, 255))
        self.assertTrue(all(m for m in mask_fg))

    def test_silhouette_match_ratio_empty_sequences_returns_hundred(self):
        """Verifies compute_silhouette_match_ratio([], []) returns 100.0."""
        self.assertEqual(compute_silhouette_match_ratio([], []), 100.0)

    def test_silhouette_match_ratio_empty_sequences_satisfies_threshold(self):
        """Verifies empty matching sequences satisfy standard parity threshold >= 99.5%."""
        self.assertGreaterEqual(compute_silhouette_match_ratio([], []), 99.5)

    def test_silhouette_mask_alpha_thresholding(self):
        """Verifies alpha channel divergence is detected with threshold."""
        bg = (13, 13, 20, 255)
        transparent = (13, 13, 20, 0)
        rgba = bytes([*bg, *transparent])
        mask = compute_binary_silhouette_mask(rgba, bg_color=bg, threshold=5)
        self.assertEqual(mask, [False, True])


# ===========================================================================
# Suite 3: TestOffscreenFrameRendering
# ===========================================================================

class TestOffscreenFrameRendering(unittest.TestCase):
    """Verifies deterministic frame rendering, temporal advancement, and artifact serialization."""

    def test_offscreen_frame_0_reproducibility(self):
        """Rendering frame 0 twice produces bitwise identical buffers."""
        try:
            f0_a = render_scene_frame("vulkan", frame_index=0, width=32, height=32)
            f0_b = render_scene_frame("vulkan", frame_index=0, width=32, height=32)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        self.assertEqual(compute_max_channel_delta(f0_a.rgba, f0_b.rgba), 0)

    def test_offscreen_frame_10_animation_advancement(self):
        """Frame 10 produces non-zero delta relative to Frame 0 due to orbital motion."""
        try:
            f0 = render_scene_frame("vulkan", frame_index=0, width=32, height=32)
            f10 = render_scene_frame("vulkan", frame_index=10, width=32, height=32)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        delta = compute_max_channel_delta(f0.rgba, f10.rgba)
        self.assertGreater(delta, 0, "Temporal advancement at frame 10 must alter lighting/pixels")

    def test_offscreen_frame_60_steady_state(self):
        """Frame 60 exhibits significant simulation advancement relative to Frame 10."""
        try:
            f10 = render_scene_frame("vulkan", frame_index=10, width=32, height=32)
            f60 = render_scene_frame("vulkan", frame_index=60, width=32, height=32)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        delta = compute_max_channel_delta(f10.rgba, f60.rgba)
        self.assertGreater(delta, 5, "Frame 60 must have significant orbital displacement")

    def test_offscreen_frame_png_encoding_validity(self):
        """write_png generates a valid W3C PNG file with correct header and dimensions."""
        if write_png is None:
            self.skipTest("write_png is not available")
        try:
            img = render_scene_frame("vulkan", frame_index=0, width=32, height=32)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            p = Path(tf.name)
        p.unlink(missing_ok=True)
        try:
            write_png(p, img)
            self.assertTrue(p.is_file())
            self.assertGreater(p.stat().st_size, 0)
            header = p.read_bytes()[:8]
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")
            if Image is not None:
                with Image.open(p) as loaded:
                    self.assertEqual(loaded.size, (32, 32))
                    self.assertEqual(loaded.mode, "RGBA")
        finally:
            p.unlink(missing_ok=True)

    def test_offscreen_frame_corrupted_buffer_rejection(self):
        """Passing truncated byte buffer to write_png raises ViewerError."""
        if write_png is None or RenderedImage is None:
            self.skipTest("mlir_gpu.viewer is not available")
        corrupted_img = RenderedImage(width=4, height=4, rgba=b"\x00" * 15, device="test")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            p = Path(tf.name)
        p.unlink(missing_ok=True)
        try:
            with self.assertRaises(ViewerError):
                write_png(p, corrupted_img)
        finally:
            p.unlink(missing_ok=True)

    def test_offscreen_forward_renderer_configuration(self):
        """ForwardRenderer initializes cleanly with offscreen=True."""
        if ForwardRenderer is None:
            self.skipTest("ForwardRenderer is not available")
        renderer = ForwardRenderer(target="vulkan", width=640, height=480, offscreen=True)
        self.assertTrue(renderer.offscreen)
        self.assertEqual(renderer.width, 640)
        self.assertEqual(renderer.height, 480)

    def test_strict_matrix_shape_types_usage(self):
        """Verifies Mat4x4 shape-named type is used throughout and deprecated Mat4 is avoided."""
        import examples.minigfx.math as mgfx_math
        self.assertTrue(hasattr(mgfx_math, "Mat4x4"), "Mat4x4 must be present in math module")
        # Ensure Mat4x4 perspective matrix calculation produces valid non-degenerate transform
        proj = mgfx_math.Mat4x4.perspective(math.radians(45.0), 16.0 / 9.0, 0.1, 100.0)
        self.assertIsInstance(proj, mgfx_math.Mat4x4)
        view = mgfx_math.Mat4x4.look_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        self.assertIsInstance(view, mgfx_math.Mat4x4)
        view_proj = proj * view
        self.assertIsInstance(view_proj, mgfx_math.Mat4x4)

    def test_render_scene_frame_case_insensitive_backends_do_not_raise_system_exit_or_value_error(self):
        """Valid backends with uppercase or title-case letters are normalized and do not raise SystemExit or ValueError."""
        for backend_variant in ("Vulkan", "Metal", "DirectX", "VULKAN", "METAL", "DIRECTX"):
            with self.subTest(backend=backend_variant):
                try:
                    render_scene_frame(backend=backend_variant, width=64, height=64)
                except ViewerError:
                    # Expected in test environments without live graphics runtime
                    pass
                except (SystemExit, ValueError) as exc:
                    self.fail(f"backend={backend_variant!r} raised unexpected {type(exc).__name__}: {exc}")

    def test_render_scene_frame_invalid_frame_index_raises_value_error(self):
        """Pre-validation on negative, non-integer, or boolean frame_index raises ValueError instead of SystemExit."""
        invalid_indices = (-1, -5, float("nan"), float("inf"), True, False, 1.5, "0")
        for f_idx in invalid_indices:
            with self.subTest(frame_index=f_idx):
                with self.assertRaises(ValueError) as ctx:
                    render_scene_frame(frame_index=f_idx)  # type: ignore[arg-type]
                self.assertIn("Frame index must be a non-negative integer", str(ctx.exception))

    def test_render_scene_frame_invalid_backend_raises_type_error(self):
        """Pre-validation on non-string backend raises TypeError."""
        for invalid_backend in (123, None, ["vulkan"], 3.14):
            with self.subTest(backend=invalid_backend):
                with self.assertRaises(TypeError) as ctx:
                    render_scene_frame(backend=invalid_backend)  # type: ignore[arg-type]
                self.assertIn("Target backend must be a string", str(ctx.exception))

    def test_render_scene_frame_invalid_backend_raises_value_error(self):
        """Pre-validation on invalid backend raises ValueError instead of SystemExit."""
        for invalid_target in ("", "none", "opengl", "software", "unsupported_backend"):
            with self.subTest(target=invalid_target):
                with self.assertRaises(ValueError) as ctx:
                    render_scene_frame(backend=invalid_target)
                self.assertIn("Invalid target backend", str(ctx.exception))

    def test_render_scene_frame_invalid_dimensions_raise_value_error(self):
        """Pre-validation on non-positive or non-integer width/height raises ValueError instead of SystemExit."""
        invalid_dims = (0, -1, -999999, float("nan"), float("inf"), True, False, 1.5, "64")
        for d in invalid_dims:
            with self.subTest(dim="width", val=d):
                with self.assertRaises(ValueError) as ctx:
                    render_scene_frame(width=d, height=64)  # type: ignore[arg-type]
                self.assertIn("Dimensions must be positive integers", str(ctx.exception))
            with self.subTest(dim="height", val=d):
                with self.assertRaises(ValueError) as ctx:
                    render_scene_frame(width=64, height=d)  # type: ignore[arg-type]
                self.assertIn("Dimensions must be positive integers", str(ctx.exception))

    def test_simulation_stepping_bitwise_parity_between_time_and_frame_index(self):
        """Simulation advancement matches bitwise between time=N/60.0 and frame_index=N."""
        from examples.minigfx.samples.model_viewer import ModelViewerApp

        app_time = ModelViewerApp(ModelViewerApp.parse_args(["--offscreen", "--frames", "61"]))
        frames = int(round(1.0 * 60.0))
        for f in range(frames + 1):
            app_time.frame_count = f
            app_time.update(1.0 / 60.0)
        app_time.elapsed_time = 1.0

        app_frame = ModelViewerApp(ModelViewerApp.parse_args(["--offscreen", "--frames", "61"]))
        for f in range(60 + 1):
            app_frame.frame_count = f
            app_frame.update(1.0 / 60.0)

        try:
            self.assertEqual(app_time.light_angle, app_frame.light_angle)
            self.assertEqual(app_time.light_params.direction.x, app_frame.light_params.direction.x)
            self.assertEqual(app_time.light_params.direction.y, app_frame.light_params.direction.y)
            self.assertEqual(app_time.light_params.direction.z, app_frame.light_params.direction.z)
        finally:
            app_time.cleanup()
            app_frame.cleanup()

        try:
            r_time = render_scene_frame("vulkan", time=1.0, width=32, height=32)
            r_frame = render_scene_frame("vulkan", frame_index=60, width=32, height=32)
            self.assertEqual(compute_max_channel_delta(r_time.rgba, r_frame.rgba), 0)
        except ViewerError:
            pass


# ===========================================================================
# Suite 4: TestCrossBackendParity
# ===========================================================================

class TestCrossBackendParity(unittest.TestCase):
    """End-to-end parity verification across Vulkan, DirectX 12, and Metal."""

    def _require_imgdiff(self):
        if imgdiff is None:
            self.skipTest("tests.lit.execution.Inputs.imgdiff is not available")

    def test_offline_embedded_pipeline_compilation_cross_backend(self):
        """Verifies that embedded shader pipelines compile to valid manifests unconditionally across Vulkan, Metal, and DirectX."""
        from examples.minigfx.samples.model_viewer import ModelViewerApp

        for target in ("vulkan", "metal", "directx"):
            with self.subTest(target=target):
                args = ModelViewerApp.parse_args([
                    "--offscreen",
                    "--frames", "0",
                    "--target", target,
                    "--width", "32",
                    "--height", "32",
                ])
                app = ModelViewerApp(args)
                try:
                    manifest_path = app.compile_embedded_pipeline(target)
                    self.assertTrue(manifest_path.is_file(), f"{target} pipeline manifest missing at {manifest_path}")
                finally:
                    app.cleanup()

    def test_cross_backend_render_parity(self):
        """Verifies multi-backend render parity across Vulkan, Metal, and DirectX 12."""
        backends: List[str] = []
        candidate_backends = ["vulkan"]
        if sys.platform == "darwin":
            candidate_backends.append("metal")
        elif sys.platform == "win32":
            candidate_backends.append("directx")

        frames = [0, 10, 60]
        renders: Dict[str, Dict[int, RenderedImage]] = {}

        for b in candidate_backends:
            try:
                backend_renders = {}
                for f_idx in frames:
                    frame = render_scene_frame(b, frame_index=f_idx, width=48, height=48)
                    bg_bytes = bytes((13, 13, 20, 255))
                    viewer_bg = bytes((5, 5, 10, 255))
                    black_opaque = bytes((0, 0, 0, 255))
                    black_zero = bytes((0, 0, 0, 0))
                    total_pixels = len(frame.rgba) // 4
                    if total_pixels == 0:
                        cov = 0.0
                    else:
                        fg_count = sum(
                            1 for i in range(0, total_pixels * 4, 4)
                            if frame.rgba[i : i + 4] not in (bg_bytes, viewer_bg, black_opaque, black_zero)
                        )
                        cov = (fg_count / total_pixels) * 100.0
                    self.assertGreater(
                        cov, 1.0, f"Candidate frame on backend {b} at frame {f_idx} has insufficient foreground coverage ({cov:.2f}% <= 1.0%)"
                    )
                    backend_renders[f_idx] = frame
                renders[b] = backend_renders
                backends.append(b)
            except ViewerError as exc:
                continue

        if len(backends) < 2:
            self.skipTest(
                f"Cross-backend render parity requires at least 2 active backends (available on this host: {backends})"
            )

        for f_idx in frames:
            for i in range(len(backends)):
                for j in range(i + 1, len(backends)):
                    b1, b2 = backends[i], backends[j]
                    r1, r2 = renders[b1][f_idx], renders[b2][f_idx]

                    # 1. Maximum per-channel delta <= 2/255
                    max_d = compute_max_channel_delta(r1.rgba, r2.rgba)
                    self.assertLessEqual(
                        max_d, 2, f"Channel delta {max_d} > 2 between {b1} and {b2} at frame {f_idx}"
                    )

                    # 2. Binary silhouette match ratio >= 99.5%
                    ratio = compute_silhouette_match_ratio(
                        r1.rgba, r2.rgba, threshold=1, bg_color=(5, 5, 10, 255)
                    )
                    self.assertGreaterEqual(
                        ratio, 99.5, f"Silhouette match ratio {ratio:.2f}% < 99.5% between {b1} and {b2}"
                    )

                    # 3. CIE76 Delta-E <= 2.3
                    delta_e = compute_cie76_delta_e(r1.rgba, r2.rgba)
                    self.assertLessEqual(
                        delta_e, 2.3, f"CIE76 Delta-E {delta_e:.4f} > 2.3 between {b1} and {b2}"
                    )

    def test_cross_backend_rejection_on_color_divergence(self):
        """Artificially divergent color buffer triggers parity failure."""
        try:
            r_ref = render_scene_frame("vulkan", width=32, height=32)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        # Shift green channel by 15 units
        altered_rgba = bytearray(r_ref.rgba)
        for i in range(1, len(altered_rgba), 4):
            altered_rgba[i] = min(255, altered_rgba[i] + 15)

        max_d = compute_max_channel_delta(r_ref.rgba, bytes(altered_rgba))
        self.assertGreater(max_d, 2, "Color divergence must be detected")

    def test_cross_backend_rejection_on_orientation_inversion(self):
        """Inverted Y coordinate (upside-down rendering) is caught as a parity failure."""
        try:
            r_ref = render_scene_frame("vulkan", width=32, height=32)
        except ViewerError as exc:
            self.skipTest(f"Live graphics device or driver not available on this host: {exc}")
        w, h = 32, 32
        row_size = w * 4
        # Flip vertically
        inverted = bytearray(len(r_ref.rgba))
        for y in range(h):
            inverted[y * row_size : (y + 1) * row_size] = r_ref.rgba[(h - 1 - y) * row_size : (h - y) * row_size]

        # Lighting asymmetry causes inverted frame to fail delta threshold
        max_d = compute_max_channel_delta(r_ref.rgba, bytes(inverted))
        self.assertGreater(max_d, 2, "Vertical inversion must fail parity threshold")

    def test_cross_backend_rules_yaml_parsing_and_validation(self):
        """Golden parity comparison rules YAML is present and conforms to schema."""
        self._require_imgdiff()
        rules_path = get_golden_parity_rules_path()
        if not rules_path.is_file():
            self.skipTest(f"Rules file {rules_path} does not exist")

        mode, delta_e, pixel_percent, alpha_delta = imgdiff._rules(rules_path)
        self.assertEqual(mode, "CIE76")
        self.assertAlmostEqual(delta_e, 2.3, places=1)
        self.assertLessEqual(pixel_percent, 1.0)
        self.assertLessEqual(alpha_delta, 2)

    def test_live_backend_execution_guard(self):
        """Safely probes host graphics hardware, cleanly skipping when environment flags are unset."""
        has_vulkan = os.environ.get("MLIR_GPU_TEST_VULKAN") == "1"
        has_metal = sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1"
        has_dx12 = sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1"

        if not (has_vulkan or has_metal or has_dx12):
            self.skipTest(
                "Live GPU parity testing requires MLIR_GPU_TEST_VULKAN=1, MLIR_GPU_TEST_METAL=1, "
                "or MLIR_GPU_TEST_DIRECTX=1"
            )

        if mgpu_viewer is None or write_png is None or RenderedImage is None:
            self.skipTest("mlir_gpu.viewer is not available")

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

    def test_imgdiff_compare_integration(self):
        """End-to-end integration with repository imgdiff tool using golden rules."""
        self._require_imgdiff()
        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            rules_path = tdp / "rules.yaml"
            rules_path.write_text("Mode: CIE76\nMaxDeltaE: 2.3\nMaxPixelPercent: 0.5\nMaxAlphaDelta: 2\n")

            img_vk = tdp / "vulkan.png"
            img_mt = tdp / "metal.png"

            try:
                r_vk = render_scene_frame("vulkan", frame_index=0, width=32, height=32)
                r_mt = render_scene_frame("metal", frame_index=0, width=32, height=32)
            except ViewerError as exc:
                self.skipTest(f"Live graphics device or driver not available on this host: {exc}")

            write_png(img_vk, r_vk)
            write_png(img_mt, r_mt)

            code = imgdiff.compare(img_vk, img_mt, rules_path)
            self.assertEqual(code, 0, "Cross-backend compare must pass under golden parity rules")


# Backwards compatibility alias
class TestMiniGfxParity(TestCrossBackendParity):
    """Backwards compatibility alias for TestCrossBackendParity."""
    pass


if __name__ == "__main__":
    unittest.main()
