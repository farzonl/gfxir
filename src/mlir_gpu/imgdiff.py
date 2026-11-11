"""Pure-Python PNG decoding and perceptual/numerical image comparison utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct
import sys
import zlib

from .viewer import RenderedImage, ViewerError, write_png

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_SRGB_TO_LINEAR: tuple[float, ...] = tuple(
    (i / 255.0) / 12.92
    if (i / 255.0) <= 0.04045
    else (((i / 255.0) + 0.055) / 1.055) ** 2.4
    for i in range(256)
)

_LAB_DELTA = 6.0 / 29.0
_LAB_THRESHOLD = _LAB_DELTA ** 3
_LAB_SLOPE = 1.0 / (3.0 * _LAB_DELTA * _LAB_DELTA)
_LAB_OFFSET = 4.0 / 29.0


class ImageDiffError(RuntimeError):
    """Raised when PNG decoding or image comparison inputs are invalid."""


@dataclass(frozen=True)
class ImageDiffResult:
    """ Quantitative comparison metrics between an actual and reference RGBA8 image."""

    width: int
    height: int
    pixel_count: int
    mismatched_pixels: int
    outlier_fraction: float
    max_channel_diff: int
    rms_error: float
    mean_cie76: float
    max_cie76: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "pixel_count": self.pixel_count,
            "mismatched_pixels": self.mismatched_pixels,
            "outlier_fraction": self.outlier_fraction,
            "max_channel_diff": self.max_channel_diff,
            "rms_error": self.rms_error,
            "mean_cie76": self.mean_cie76,
            "max_cie76": self.max_cie76,
            "passed": self.passed,
        }

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{status}: {self.width}x{self.height} "
            f"outliers={self.mismatched_pixels}/{self.pixel_count} ({self.outlier_fraction * 100.0:.3f}%), "
            f"max_channel_diff={self.max_channel_diff}, "
            f"rms={self.rms_error:.6f}, "
            f"mean_cie76={self.mean_cie76:.4f}, "
            f"max_cie76={self.max_cie76:.4f}"
        )


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def read_png(path: Path | str) -> RenderedImage:
    """Decode an 8-bit non-interlaced PNG file into an RGBA8 RenderedImage."""
    file_path = Path(path)
    data = file_path.read_bytes()
    if len(data) < 33 or data[:8] != PNG_SIGNATURE:
        raise ImageDiffError(f"Not a valid PNG file: {file_path}")

    offset = 8
    width = 0
    height = 0
    bit_depth = 0
    color_type = 0
    seen_ihdr = False
    seen_iend = False
    idat_parts: list[bytes] = []

    while offset < len(data):
        if offset + 8 > len(data):
            raise ImageDiffError(f"Truncated PNG chunk header in {file_path}")
        length, chunk_type = struct.unpack(">I4s", data[offset:offset + 8])
        offset += 8
        if offset + length + 4 > len(data):
            raise ImageDiffError(f"Truncated PNG chunk {chunk_type!r} in {file_path}")
        payload = data[offset:offset + length]
        (expected_crc,) = struct.unpack(">I", data[offset + length:offset + length + 4])
        offset += length + 4
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise ImageDiffError(f"CRC mismatch in PNG chunk {chunk_type!r} of {file_path}")

        if chunk_type == b"IHDR":
            if seen_ihdr or length != 13:
                raise ImageDiffError(f"Invalid IHDR chunk in {file_path}")
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB", payload
            )
            if width <= 0 or height <= 0:
                raise ImageDiffError(f"PNG dimensions must be positive in {file_path}")
            if bit_depth != 8:
                raise ImageDiffError(f"Only 8-bit PNG images are supported (got {bit_depth}-bit) in {file_path}")
            if color_type not in (0, 2, 4, 6):
                raise ImageDiffError(f"Unsupported PNG color type {color_type} in {file_path}")
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ImageDiffError(f"Unsupported PNG compression/filter/interlace mode in {file_path}")
            seen_ihdr = True
        elif chunk_type == b"IDAT":
            if not seen_ihdr:
                raise ImageDiffError(f"IDAT appeared before IHDR in {file_path}")
            idat_parts.append(payload)
        elif chunk_type == b"IEND":
            seen_iend = True
            break

    if not seen_ihdr or not seen_iend or not idat_parts:
        raise ImageDiffError(f"Incomplete PNG stream in {file_path}")

    channels = {0: 1, 2: 3, 4: 2, 6: 4}[color_type]
    stride = width * channels
    try:
        raw = zlib.decompress(b"".join(idat_parts))
    except zlib.error as error:
        raise ImageDiffError(f"Failed to decompress PNG IDAT data in {file_path}: {error}") from error

    expected_raw_len = height * (1 + stride)
    if len(raw) != expected_raw_len:
        raise ImageDiffError(
            f"Decompressed PNG scanline size mismatch in {file_path}: "
            f"expected {expected_raw_len} bytes, got {len(raw)}"
        )

    reconstructed = bytearray(height * stride)
    prev_row = bytearray(stride)
    src_pos = 0
    dst_pos = 0

    for _ in range(height):
        filter_type = raw[src_pos]
        src_pos += 1
        row = bytearray(raw[src_pos:src_pos + stride])
        src_pos += stride

        if filter_type == 0:
            pass
        elif filter_type == 1:
            for i in range(channels, stride):
                row[i] = (row[i] + row[i - channels]) & 0xFF
        elif filter_type == 2:
            for i in range(stride):
                row[i] = (row[i] + prev_row[i]) & 0xFF
        elif filter_type == 3:
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                row[i] = (row[i] + ((left + prev_row[i]) >> 1)) & 0xFF
        elif filter_type == 4:
            for i in range(stride):
                left = row[i - channels] if i >= channels else 0
                up = prev_row[i]
                up_left = prev_row[i - channels] if i >= channels else 0
                row[i] = (row[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise ImageDiffError(f"Unsupported PNG scanline filter {filter_type} in {file_path}")

        reconstructed[dst_pos:dst_pos + stride] = row
        prev_row = row
        dst_pos += stride

    if color_type == 6:
        rgba = bytes(reconstructed)
    elif color_type == 2:
        out = bytearray(width * height * 4)
        p_in = 0
        p_out = 0
        for _ in range(width * height):
            out[p_out] = reconstructed[p_in]
            out[p_out + 1] = reconstructed[p_in + 1]
            out[p_out + 2] = reconstructed[p_in + 2]
            out[p_out + 3] = 255
            p_in += 3
            p_out += 4
        rgba = bytes(out)
    elif color_type == 0:
        out = bytearray(width * height * 4)
        p_out = 0
        for v in reconstructed:
            out[p_out] = v
            out[p_out + 1] = v
            out[p_out + 2] = v
            out[p_out + 3] = 255
            p_out += 4
        rgba = bytes(out)
    else:
        out = bytearray(width * height * 4)
        p_in = 0
        p_out = 0
        for _ in range(width * height):
            v = reconstructed[p_in]
            out[p_out] = v
            out[p_out + 1] = v
            out[p_out + 2] = v
            out[p_out + 3] = reconstructed[p_in + 1]
            p_in += 2
            p_out += 4
        rgba = bytes(out)

    return RenderedImage(width=width, height=height, rgba=rgba, device="png")


def _lab_f(t: float) -> float:
    return t ** (1.0 / 3.0) if t > _LAB_THRESHOLD else (_LAB_SLOPE * t + _LAB_OFFSET)


def _srgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    rl = _SRGB_TO_LINEAR[r]
    gl = _SRGB_TO_LINEAR[g]
    bl = _SRGB_TO_LINEAR[b]

    x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047
    y = 0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl
    z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883

    fx = _lab_f(x)
    fy = _lab_f(y)
    fz = _lab_f(z)
    return (116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz))


def _coerce_image(source: RenderedImage | Path | str) -> RenderedImage:
    if isinstance(source, RenderedImage):
        expected_len = source.width * source.height * 4
        if source.width <= 0 or source.height <= 0 or len(source.rgba) != expected_len:
            raise ImageDiffError("RenderedImage RGBA buffer length does not match its dimensions.")
        return source
    if isinstance(source, (Path, str)):
        return read_png(source)
    raise TypeError("imgdiff inputs must be RenderedImage instances or PNG file paths.")


def imgdiff(
    actual: RenderedImage | Path | str,
    expected: RenderedImage | Path | str,
    *,
    channel_tolerance: int = 0,
    cie76_threshold: float | None = None,
    max_outlier_fraction: float = 0.0,
    max_rms: float | None = None,
    diff_output: Path | str | None = None,
) -> ImageDiffResult:
    """Compare an actual image against a reference image using channel, RMS, and CIE76 metrics."""
    if type(channel_tolerance) is not int or not 0 <= channel_tolerance <= 255:
        raise ValueError("channel_tolerance must be an integer between 0 and 255.")
    if cie76_threshold is not None and (
        type(cie76_threshold) not in (int, float) or cie76_threshold < 0.0 or not math.isfinite(cie76_threshold)
    ):
        raise ValueError("cie76_threshold must be a finite nonnegative float.")
    if type(max_outlier_fraction) not in (int, float) or not 0.0 <= max_outlier_fraction <= 1.0:
        raise ValueError("max_outlier_fraction must be between 0.0 and 1.0.")
    if max_rms is not None and (
        type(max_rms) not in (int, float) or max_rms < 0.0 or not math.isfinite(max_rms)
    ):
        raise ValueError("max_rms must be a finite nonnegative float.")

    img_a = _coerce_image(actual)
    img_e = _coerce_image(expected)
    if (img_a.width, img_a.height) != (img_e.width, img_e.height):
        raise ImageDiffError(
            f"Image dimension mismatch: actual is {img_a.width}x{img_a.height}, "
            f"expected is {img_e.width}x{img_e.height}."
        )

    width = img_a.width
    height = img_a.height
    pixel_count = width * height
    buf_a = img_a.rgba
    buf_e = img_e.rgba

    mismatched_pixels = 0
    max_channel_diff = 0
    sum_sq = 0
    sum_cie76 = 0.0
    max_cie76 = 0.0
    lab_cache: dict[int, tuple[float, float, float]] = {}
    heatmap = bytearray(pixel_count * 4) if diff_output is not None else None

    for idx in range(0, pixel_count * 4, 4):
        r1, g1, b1, a1 = buf_a[idx], buf_a[idx + 1], buf_a[idx + 2], buf_a[idx + 3]
        r2, g2, b2, a2 = buf_e[idx], buf_e[idx + 1], buf_e[idx + 2], buf_e[idx + 3]

        if r1 == r2 and g1 == g2 and b1 == b2 and a1 == a2:
            if heatmap is not None:
                lum = (77 * r2 + 150 * g2 + 29 * b2) >> 10
                heatmap[idx] = lum
                heatmap[idx + 1] = lum
                heatmap[idx + 2] = lum
                heatmap[idx + 3] = 255
            continue

        dr = abs(r1 - r2)
        dg = abs(g1 - g2)
        db = abs(b1 - b2)
        da = abs(a1 - a2)
        px_max_diff = max(dr, dg, db, da)
        if px_max_diff > max_channel_diff:
            max_channel_diff = px_max_diff
        sum_sq += dr * dr + dg * dg + db * db + da * da

        if r1 == r2 and g1 == g2 and b1 == b2:
            delta_e = 0.0
        else:
            key1 = (r1 << 16) | (g1 << 8) | b1
            lab1 = lab_cache.get(key1)
            if lab1 is None:
                lab1 = _srgb_to_lab(r1, g1, b1)
                lab_cache[key1] = lab1
            key2 = (r2 << 16) | (g2 << 8) | b2
            lab2 = lab_cache.get(key2)
            if lab2 is None:
                lab2 = _srgb_to_lab(r2, g2, b2)
                lab_cache[key2] = lab2
            dl = lab1[0] - lab2[0]
            da_lab = lab1[1] - lab2[1]
            db_lab = lab1[2] - lab2[2]
            delta_e = math.sqrt(dl * dl + da_lab * da_lab + db_lab * db_lab)

        sum_cie76 += delta_e
        if delta_e > max_cie76:
            max_cie76 = delta_e

        if cie76_threshold is not None:
            is_outlier = delta_e > cie76_threshold or da > channel_tolerance
        else:
            is_outlier = px_max_diff > channel_tolerance

        if is_outlier:
            mismatched_pixels += 1

        if heatmap is not None:
            if is_outlier:
                intensity = min(255, max(96, px_max_diff * 4))
                heatmap[idx] = 255
                heatmap[idx + 1] = max(0, 255 - intensity)
                heatmap[idx + 2] = intensity
                heatmap[idx + 3] = 255
            else:
                lum = (77 * r2 + 150 * g2 + 29 * b2) >> 10
                heatmap[idx] = lum
                heatmap[idx + 1] = min(255, lum + 32)
                heatmap[idx + 2] = lum
                heatmap[idx + 3] = 255

    rms_error = math.sqrt(sum_sq / (pixel_count * 4 * 255.0 * 255.0))
    mean_cie76 = sum_cie76 / pixel_count
    outlier_fraction = mismatched_pixels / pixel_count
    passed = outlier_fraction <= max_outlier_fraction and (max_rms is None or rms_error <= max_rms)

    if heatmap is not None and diff_output is not None:
        out_path = Path(diff_output)
        out_path.unlink(missing_ok=True)
        try:
            write_png(out_path, RenderedImage(width, height, bytes(heatmap), "imgdiff"))
        except ViewerError as error:
            raise ImageDiffError(str(error)) from error

    return ImageDiffResult(
        width=width,
        height=height,
        pixel_count=pixel_count,
        mismatched_pixels=mismatched_pixels,
        outlier_fraction=outlier_fraction,
        max_channel_diff=max_channel_diff,
        rms_error=rms_error,
        mean_cie76=mean_cie76,
        max_cie76=max_cie76,
        passed=passed,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare an actual PNG image against a reference PNG image."
    )
    parser.add_argument("actual", type=Path, help="Rendered PNG image path")
    parser.add_argument("expected", type=Path, help="Golden reference PNG image path")
    parser.add_argument(
        "--channel-tolerance",
        type=int,
        default=0,
        help="Maximum allowed per-channel [0..255] absolute difference per pixel (default: 0)",
    )
    parser.add_argument(
        "--cie76-threshold",
        type=float,
        default=None,
        help="Perceptual CIE76 Delta-E threshold per pixel (optional)",
    )
    parser.add_argument(
        "--max-outlier-fraction",
        type=float,
        default=0.0,
        help="Maximum fraction [0.0..1.0] of outlier pixels allowed (default: 0.0)",
    )
    parser.add_argument(
        "--max-rms",
        type=float,
        default=None,
        help="Maximum normalized [0.0..1.0] RMS channel error allowed (optional)",
    )
    parser.add_argument(
        "--diff-output",
        type=Path,
        default=None,
        help="Optional output path for a visual diff PNG heatmap",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit comparison metrics as JSON",
    )
    args = parser.parse_args(argv)
    try:
        result = imgdiff(
            args.actual,
            args.expected,
            channel_tolerance=args.channel_tolerance,
            cie76_threshold=args.cie76_threshold,
            max_outlier_fraction=args.max_outlier_fraction,
            max_rms=args.max_rms,
            diff_output=args.diff_output,
        )
    except (ImageDiffError, OSError, ValueError) as error:
        print(f"mlir-gpu-imgdiff: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(result.summary())
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
