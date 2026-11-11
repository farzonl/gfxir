"""Unit tests for pure-Python PNG decoding and imgdiff comparison utilities."""

from __future__ import annotations

import json
from pathlib import Path
import struct
import tempfile
import unittest
import zlib

from mlir_gpu import (
    ImageDiffError,
    RenderedImage,
    imgdiff,
    read_png,
    write_png,
)
from mlir_gpu.imgdiff import _paeth_predictor, main as imgdiff_main


def _encode_custom_png(
    width: int,
    height: int,
    color_type: int,
    raw_pixels: bytes,
    filter_mode: int,
) -> bytes:
    """Encode an 8-bit PNG using a specific scanline filter mode (0..4) for decoder testing."""
    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    stride = width * channels
    prev_row = bytearray(stride)
    filtered_rows = bytearray()

    for y in range(height):
        row = raw_pixels[y * stride:(y + 1) * stride]
        filtered = bytearray(stride)
        if filter_mode == 0:
            filtered[:] = row
        elif filter_mode == 1:
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                filtered[i] = (row[i] - left) & 0xFF
        elif filter_mode == 2:
            for i in range(stride):
                filtered[i] = (row[i] - prev_row[i]) & 0xFF
        elif filter_mode == 3:
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                filtered[i] = (row[i] - ((left + prev_row[i]) >> 1)) & 0xFF
        elif filter_mode == 4:
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                up = prev_row[i]
                up_left = prev_row[i - channels] if i >= channels else 0
                filtered[i] = (row[i] - _paeth_predictor(left, up, up_left)) & 0xFF
        filtered_rows.append(filter_mode)
        filtered_rows.extend(filtered)
        prev_row = bytearray(row)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    compressed = zlib.compress(bytes(filtered_rows))
    mid = max(1, len(compressed) // 2)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed[:mid])
        + chunk(b"IDAT", compressed[mid:])
        + chunk(b"IEND", b"")
    )


class TestPngDecoderAndImgDiff(unittest.TestCase):
    """Tests covering PNG scanline filter reconstruction, color types, and imgdiff metrics."""

    def test_write_and_read_png_roundtrip(self):
        width, height = 16, 12
        flat = bytearray()
        for y in range(height):
            for x in range(width):
                flat.extend(((x * 15) & 0xFF, (y * 20) & 0xFF, ((x + y) * 9) & 0xFF, 255))
        original = RenderedImage(width, height, bytes(flat), "synthetic")

        with tempfile.TemporaryDirectory() as td:
            png_path = Path(td) / "roundtrip.png"
            write_png(png_path, original)
            decoded = read_png(png_path)
            self.assertEqual(decoded.width, width)
            self.assertEqual(decoded.height, height)
            self.assertEqual(decoded.rgba, original.rgba)

    def test_read_png_all_scanline_filters_and_color_types(self):
        width, height = 8, 6
        rgba = bytearray()
        rgb = bytearray()
        gray = bytearray()
        gray_alpha = bytearray()
        for y in range(height):
            for x in range(width):
                r = (x * 31 + y * 17) & 0xFF
                g = (x * 13 + y * 43) & 0xFF
                b = (x * 7 + y * 29) & 0xFF
                a = 200 + ((x + y) % 55)
                rgba.extend((r, g, b, a))
                rgb.extend((r, g, b))
                gray.append(r)
                gray_alpha.extend((r, a))

        with tempfile.TemporaryDirectory() as td:
            for filter_mode in range(5):
                with self.subTest(filter_mode=filter_mode):
                    p = Path(td) / f"rgba_f{filter_mode}.png"
                    p.write_bytes(_encode_custom_png(width, height, 6, bytes(rgba), filter_mode))
                    dec = read_png(p)
                    self.assertEqual(dec.rgba, bytes(rgba))

            rgb_path = Path(td) / "rgb.png"
            rgb_path.write_bytes(_encode_custom_png(width, height, 2, bytes(rgb), 4))
            dec_rgb = read_png(rgb_path)
            self.assertEqual(len(dec_rgb.rgba), width * height * 4)
            self.assertEqual(dec_rgb.rgba[3::4], bytes([255] * (width * height)))

            gray_path = Path(td) / "gray.png"
            gray_path.write_bytes(_encode_custom_png(width, height, 0, bytes(gray), 1))
            dec_gray = read_png(gray_path)
            self.assertEqual(dec_gray.rgba[0::4], bytes(gray))
            self.assertEqual(dec_gray.rgba[1::4], bytes(gray))
            self.assertEqual(dec_gray.rgba[2::4], bytes(gray))

            ga_path = Path(td) / "ga.png"
            ga_path.write_bytes(_encode_custom_png(width, height, 4, bytes(gray_alpha), 3))
            dec_ga = read_png(ga_path)
            self.assertEqual(dec_ga.rgba[0::4], bytes(gray))
            self.assertEqual(dec_ga.rgba[3::4], bytes(gray_alpha[1::2]))

    def test_read_png_rejects_corrupted_files(self):
        with tempfile.TemporaryDirectory() as td:
            bad_sig = Path(td) / "bad_sig.png"
            bad_sig.write_bytes(b"NOT_A_PNG_AT_ALL" * 4)
            with self.assertRaises(ImageDiffError):
                read_png(bad_sig)

            valid = Path(td) / "valid.png"
            write_png(valid, RenderedImage(2, 2, bytes([10, 20, 30, 255] * 4), "test"))
            corrupted = bytearray(valid.read_bytes())
            corrupted[20] ^= 0xFF
            bad_crc = Path(td) / "bad_crc.png"
            bad_crc.write_bytes(bytes(corrupted))
            with self.assertRaisesRegex(ImageDiffError, "CRC mismatch"):
                read_png(bad_crc)

    def test_imgdiff_exact_match_and_tolerances(self):
        base_pixels = bytearray([40, 120, 200, 255] * 16)
        img_ref = RenderedImage(4, 4, bytes(base_pixels), "ref")
        exact = imgdiff(img_ref, img_ref)
        self.assertTrue(exact.passed)
        self.assertEqual(exact.mismatched_pixels, 0)
        self.assertEqual(exact.max_channel_diff, 0)
        self.assertEqual(exact.rms_error, 0.0)
        self.assertEqual(exact.max_cie76, 0.0)

        near_pixels = bytearray(base_pixels)
        near_pixels[0] = 42
        img_near = RenderedImage(4, 4, bytes(near_pixels), "near")

        strict = imgdiff(img_near, img_ref, channel_tolerance=0, max_outlier_fraction=0.0)
        self.assertFalse(strict.passed)
        self.assertEqual(strict.mismatched_pixels, 1)
        self.assertEqual(strict.max_channel_diff, 2)

        relaxed_chan = imgdiff(img_near, img_ref, channel_tolerance=2, max_outlier_fraction=0.0)
        self.assertTrue(relaxed_chan.passed)
        self.assertEqual(relaxed_chan.mismatched_pixels, 0)

        relaxed_cie = imgdiff(img_near, img_ref, cie76_threshold=2.0, max_outlier_fraction=0.0)
        self.assertTrue(relaxed_cie.passed)
        self.assertEqual(relaxed_cie.mismatched_pixels, 0)
        self.assertGreater(relaxed_cie.max_cie76, 0.0)
        self.assertLess(relaxed_cie.max_cie76, 2.0)

        outlier_allowed = imgdiff(img_near, img_ref, channel_tolerance=0, max_outlier_fraction=1.0 / 16.0)
        self.assertTrue(outlier_allowed.passed)

    def test_imgdiff_dimension_mismatch_and_heatmap_cli(self):
        img_a = RenderedImage(4, 4, bytes([10, 20, 30, 255] * 16), "a")
        img_b = RenderedImage(8, 4, bytes([10, 20, 30, 255] * 32), "b")
        with self.assertRaisesRegex(ImageDiffError, "dimension mismatch"):
            imgdiff(img_a, img_b)

        with tempfile.TemporaryDirectory() as td:
            p_actual = Path(td) / "actual.png"
            p_expected = Path(td) / "expected.png"
            p_diff = Path(td) / "diff.png"
            mod_pixels = bytearray([10, 20, 30, 255] * 16)
            mod_pixels[0:4] = bytes([250, 20, 30, 255])
            write_png(p_actual, RenderedImage(4, 4, bytes(mod_pixels), "actual"))
            write_png(p_expected, img_a)

            rc_fail = imgdiff_main([str(p_actual), str(p_expected), "--diff-output", str(p_diff)])
            self.assertEqual(rc_fail, 1)
            self.assertTrue(p_diff.is_file())
            diff_img = read_png(p_diff)
            self.assertEqual((diff_img.width, diff_img.height), (4, 4))
            self.assertEqual(diff_img.rgba[0], 255)

            rc_pass = imgdiff_main([
                str(p_actual),
                str(p_expected),
                "--max-outlier-fraction",
                "0.1",
                "--json",
            ])
            self.assertEqual(rc_pass, 0)

            res = imgdiff(p_actual, p_expected, max_outlier_fraction=0.1)
            payload = json.loads(json.dumps(res.to_dict()))
            self.assertTrue(payload["passed"])
            self.assertEqual(payload["mismatched_pixels"], 1)


if __name__ == "__main__":
    unittest.main()
