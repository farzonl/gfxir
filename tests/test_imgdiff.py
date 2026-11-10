import sys
import tempfile
import unittest
from pathlib import Path

try:
    from PIL import Image
    from imgdiff import compare, compare_targets
except ImportError:
    Image = None
    compare = None
    compare_targets = None


@unittest.skipIf(Image is None, "Pillow is not installed")
class ImageDiffTests(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)
        self.expected = self.directory / "expected.png"
        self.actual = self.directory / "actual.png"
        self.rules = self.directory / "rules.yaml"
        self.rules.write_text(
            "Mode: CIE76\nMaxDeltaE: 2.3\nMaxPixelPercent: 0.2\nMaxAlphaDelta: 0\n"
        )

    def test_cie76_tolerates_small_channel_deltas_and_writes_diff(self):
        Image.new("RGBA", (16, 16), (20, 40, 60, 255)).save(self.expected)
        actual = Image.new("RGBA", (16, 16), (20, 40, 60, 255))
        actual.putpixel((0, 0), (21, 40, 60, 255))
        actual.save(self.actual)

        diff = self.directory / "diff.png"
        self.assertEqual(compare(self.expected, self.actual, self.rules, diff), 0)
        self.assertIsNone(Image.open(diff).getbbox())

    def test_cie76_rejects_excess_different_pixels(self):
        Image.new("RGBA", (16, 16), (20, 40, 60, 255)).save(self.expected)
        actual = Image.new("RGBA", (16, 16), (20, 40, 60, 255))
        actual.putpixel((0, 0), (255, 0, 0, 255))
        actual.save(self.actual)

        diff = self.directory / "diff.png"
        self.assertEqual(compare(self.expected, self.actual, self.rules, diff), 1)
        self.assertIsNotNone(Image.open(diff).getbbox())

    def test_exact_mode_rejects_any_pixel_change(self):
        Image.new("RGBA", (1, 1), (20, 40, 60, 255)).save(self.expected)
        Image.new("RGBA", (1, 1), (21, 40, 60, 255)).save(self.actual)
        self.rules.write_text(
            "Mode: Exact\nMaxDeltaE: 2.3\nMaxPixelPercent: 100\nMaxAlphaDelta: 255\n"
        )

        self.assertEqual(compare(self.expected, self.actual, self.rules), 1)

    def test_different_image_sizes_are_an_error(self):
        Image.new("RGBA", (1, 1), (0, 0, 0, 255)).save(self.expected)
        Image.new("RGBA", (2, 1), (0, 0, 0, 255)).save(self.actual)

        with self.assertRaisesRegex(ValueError, "image sizes do not match"):
            compare(self.expected, self.actual, self.rules)

    def test_compare_targets_checks_each_render_against_the_golden(self):
        Image.new("RGBA", (1, 1), (20, 40, 60, 255)).save(self.expected)
        for target in ("vulkan", "metal"):
            Image.new("RGBA", (1, 1), (20, 40, 60, 255)).save(
                target_output_path(self.actual, target)
            )

        self.assertEqual(
            compare_targets(self.expected, self.actual, ["vulkan", "metal"], self.rules),
            0,
        )
        for target in ("vulkan", "metal"):
            diff = target_output_path(self.actual, target)
            diff = diff.with_name(f"{diff.stem}-diff{diff.suffix}")
            self.assertTrue(diff.exists())


if __name__ == "__main__":
    unittest.main()
