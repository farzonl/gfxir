import argparse
import math
import struct
import sys
from pathlib import Path

import yaml
from PIL import Image

from execution_targets import target_output_path, targets_from_environment


def _linearize(value: int) -> float:
    channel = value / 255
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def _lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    red, green, blue = (_linearize(channel) for channel in rgb)
    x = (0.4124564 * red + 0.3575761 * green + 0.1804375 * blue) / 0.95047
    y = 0.2126729 * red + 0.7151522 * green + 0.0721750 * blue
    z = (0.0193339 * red + 0.1191920 * green + 0.9503041 * blue) / 1.08883

    def transform(value: float) -> float:
        return value ** (1 / 3) if value > 0.008856451679 else 7.787037037 * value + 16 / 116

    x, y, z = transform(x), transform(y), transform(z)
    return 116 * y - 16, 500 * (x - y), 200 * (y - z)


def _rules(path: Path) -> tuple[str, float, float, int]:
    try:
        rules = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot read image-diff rules {path}: {error}") from error
    if not isinstance(rules, dict) or set(rules) != {
        "Mode", "MaxDeltaE", "MaxPixelPercent", "MaxAlphaDelta",
    }:
        raise ValueError(
            "rules must contain exactly Mode, MaxDeltaE, MaxPixelPercent, and MaxAlphaDelta"
        )
    mode, delta_e, pixel_percent, alpha_delta = (
        rules["Mode"], rules["MaxDeltaE"], rules["MaxPixelPercent"], rules["MaxAlphaDelta"],
    )
    if mode not in ("Exact", "CIE76"):
        raise ValueError("Mode must be Exact or CIE76")
    if (
        isinstance(delta_e, bool) or not isinstance(delta_e, (int, float))
        or not math.isfinite(delta_e) or delta_e < 0
    ):
        raise ValueError("MaxDeltaE must be a finite nonnegative number")
    if (
        isinstance(pixel_percent, bool) or not isinstance(pixel_percent, (int, float))
        or not math.isfinite(pixel_percent) or not 0 <= pixel_percent <= 100
    ):
        raise ValueError("MaxPixelPercent must be between 0 and 100")
    if type(alpha_delta) is not int or not 0 <= alpha_delta <= 255:
        raise ValueError("MaxAlphaDelta must be an integer between 0 and 255")
    return mode, float(delta_e), float(pixel_percent), alpha_delta


def compare(
    expected_path: Path,
    actual_path: Path,
    rules_path: Path,
    output_path: Path | None = None,
) -> int:
    mode, max_delta_e, max_pixel_percent, max_alpha_delta = _rules(rules_path)
    try:
        with Image.open(expected_path) as expected_file, Image.open(actual_path) as actual_file:
            expected = expected_file.convert("RGBA")
            actual = actual_file.convert("RGBA")
            if expected.size != actual.size:
                raise ValueError(f"image sizes do not match: {expected.size} != {actual.size}")
            expected_pixels = struct.iter_unpack("4B", expected.tobytes())
            actual_pixels = struct.iter_unpack("4B", actual.tobytes())
            pixel_count = expected.width * expected.height
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot compare images: {error}") from error

    differing = 0
    largest_delta_e = 0.0
    diff_pixels = []
    for left, right in zip(expected_pixels, actual_pixels):
        if mode == "Exact":
            mismatch = left != right
            delta_e = 0.0
        else:
            left_lab = _lab(left[:3])
            right_lab = _lab(right[:3])
            delta_e = math.sqrt(sum((a - b) ** 2 for a, b in zip(left_lab, right_lab)))
            mismatch = delta_e > max_delta_e or abs(left[3] - right[3]) > max_alpha_delta
        largest_delta_e = max(largest_delta_e, delta_e)
        differing += mismatch
        diff_pixels.append((255, 0, 0, 255) if mismatch else (0, 0, 0, 0))

    percent = 100 * differing / pixel_count
    if output_path is not None:
        try:
            diff = Image.new("RGBA", expected.size)
            diff.putdata(diff_pixels)
            diff.save(output_path)
        except OSError as error:
            raise ValueError(f"cannot write diff image {output_path}: {error}") from error
    print(
        f"imgdiff: {differing}/{pixel_count} pixels differ "
        f"({percent:.4f}%); max CIE76={largest_delta_e:.4f}; "
        f"allowed={max_pixel_percent:.4f}%"
    )
    if mode == "Exact" and differing:
        print("imgdiff: images are not an exact match", file=sys.stderr)
        return 1
    if mode == "CIE76" and percent > max_pixel_percent:
        print("imgdiff: pixel difference exceeds the configured limit", file=sys.stderr)
        return 1
    return 0


def compare_targets(
    expected: Path,
    actual_prefix: Path,
    targets: list[str],
    rules: Path,
) -> int:
    result = 0
    for target in targets:
        actual = target_output_path(actual_prefix, target)
        output = actual.with_name(f"{actual.stem}-diff{actual.suffix}")
        result |= compare(expected, actual, rules, output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare target renders with a shared PNG golden.")
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual_prefix", type=Path)
    parser.add_argument("--target", required=True, metavar="ENVIRONMENT_VARIABLE")
    parser.add_argument("--rules", type=Path, required=True)
    args = parser.parse_args()
    try:
        targets = targets_from_environment(args.target)
        return compare_targets(args.expected, args.actual_prefix, targets, args.rules)
    except ValueError as error:
        print(f"imgdiff: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
