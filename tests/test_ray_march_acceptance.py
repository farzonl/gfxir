import collections
import json
import math
import os
import re
import runpy
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from mlir_gpu import Pipeline, Target, compile_pipeline, fullscreen_vertex
from mlir_gpu.viewer import _metal_libraries, _metal_run, load_program, render_metal_offscreen, render_offscreen
from test_beacon_gallery import gallery_pipeline
from test_native_ir import ROOT, TOOLS, available
from test_vulkan_parameters import both_stages, frame_vertex


def budget_reference(x, y, width, height, seconds):
    uv = ((x + 0.5 - width * 0.5) / height, -(y + 0.5 - height * 0.5) / height)
    length = math.sqrt(uv[0] ** 2 + uv[1] ** 2 + 1)
    ray = uv[0] / length, uv[1] / length, 1 / length
    center = math.sin(seconds) * 0.15
    radius = 0.65 + math.sin(seconds) * 0.08
    cap = 4 if x + 0.5 < width * 0.5 else 64
    distance, used, hit, escaped, margin = 0.0, 0, False, False, float("inf")
    for step in range(cap):
        used += 1
        point = ray[0] * distance - center, ray[1] * distance, -3 + ray[2] * distance
        surface = math.sqrt(sum(component * component for component in point)) - radius
        margin = min(margin, abs(surface - 0.002))
        if surface < 0.002:
            hit = True
            break
        distance += max(surface * 0.75, 0.002)
        if distance > 8:
            escaped = True
            break
    return (round(used / 64 * 255), 255 if hit else 0, 255 if escaped else 38, 255), margin


@unittest.skipUnless(available(), "Build pinned native compiler for loop-size acceptance")
class LoopSizeAcceptanceTests(unittest.TestCase):
    def test_fourfold_cap_changes_the_bound_not_the_body(self):
        with tempfile.TemporaryDirectory() as work:
            for target in (Target.VULKAN, Target.DIRECTX):
                counts, sizes = [], []
                for cap in (72, 288):
                    pipeline = gallery_pipeline()
                    pipeline.fragment.function.__globals__["MAX_STEPS"] = cap
                    directory = Path(work) / f"{target.value}-{cap}"
                    compile_pipeline(pipeline, target, directory, TOOLS)
                    original = (directory / "fragment.mlir").read_text()
                    self.assertEqual(original.count("scf.while"), 1)
                    self.assertRegex(original, rf"arith.constant {cap} : i32")
                    counts.append(collections.Counter(re.findall(
                        r"^\s*(?:%.*?=\s*)?([a-z]+\.[\w]+)", original, re.MULTILINE)))
                    if target == Target.VULKAN:
                        self.assertIn("spirv.mlir.loop", (directory / "fragment.lowered.mlir").read_text())
                    else:
                        self.assertIn("phi <3 x float>", (directory / "fragment.ll").read_text())
                    sizes.append((directory / ("fragment.spv" if target == Target.VULKAN else "fragment.dxil")).stat().st_size)
                self.assertEqual(counts[0], counts[1])
                self.assertLessEqual(abs(sizes[0] - sizes[1]), 128)
                print(f"Loop-size acceptance {target.value}: caps 72/288, bytes {sizes}")


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Enable both Vulkan and Metal for real ray-march acceptance",
)
class RayMarchAcceptanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        work = tempfile.TemporaryDirectory()
        cls.addClassCleanup(work.cleanup)
        cls.directory = Path(work.name)
        cls.programs = {}
        for target in Target:
            for name, pipeline in (
                ("gallery", gallery_pipeline()),
                ("budget", runpy.run_path(str(ROOT / "examples/march_budget.py"))["pipeline"]),
                ("frame", Pipeline(frame_vertex, both_stages)),
            ):
                start = time.perf_counter()
                manifest = compile_pipeline(pipeline, target, cls.directory / f"{name}-{target.value}", TOOLS)
                cls.programs[name, target] = load_program(manifest)
                if name == "gallery":
                    extension = {"vulkan": "spv", "directx": "dxil", "metal": "metallib"}[target.value]
                    size = (manifest.parent / f"fragment.{extension}").stat().st_size
                    print(f"Gallery compile {target.value}: {time.perf_counter() - start:.3f}s, fragment {size} bytes")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"

    def render(self, name, target, width, height, seconds):
        render = render_offscreen if target == Target.VULKAN else render_metal_offscreen
        return render(self.programs[name, target], width, height, self.validation, time=seconds).rgba

    def test_gallery_parity_at_square_rectangular_and_odd_sizes_and_times(self):
        for width, height in ((64, 64), (96, 54), (97, 73)):
            for seconds in (0, 0.25, 1.75):
                images = {target: self.render("gallery", target, width, height, seconds) for target in Target}
                self.assertEqual(images[Target.DIRECTX], images[Target.METAL])
                self.assertEqual(set(images[Target.VULKAN][3::4]), {255})
                differences = [abs(a - b) for i, (a, b) in enumerate(zip(
                    images[Target.VULKAN], images[Target.METAL])) if i % 4 != 3]
                mean = sum(differences) / len(differences)
                peaks = [max(differences[i:i + 3]) for i in range(0, len(differences), 3)]
                p99 = sorted(peaks)[math.ceil(len(peaks) * 0.99) - 1]
                outliers = sum(value > 16 for value in peaks) / len(peaks)
                print(f"Gallery parity {width}x{height} t={seconds}: mean={mean:.4f}, p99={p99}, max={max(peaks)}, >16={outliers:.5f}")
                self.assertLessEqual(mean, 1)
                self.assertLessEqual(p99, 4)
                self.assertLessEqual(outliers, 0.002)

    def test_documented_320x240_workload_renders_on_every_preview_path(self):
        for target in Target:
            started = time.perf_counter()
            image = self.render("gallery", target, 320, 240, 0.25)
            print(f"Gallery 320x240 {target.value}: {(time.perf_counter() - started) * 1000:.1f}ms including setup/readback")
            self.assertEqual(len(image), 320 * 240 * 4)
            self.assertEqual(set(image[3::4]), {255})

    def test_cpu_reference_checks_hit_miss_cap_and_early_exit_order(self):
        regimes = set()
        for width, height in ((32, 32), (48, 30), (49, 37)):
            for seconds in (0, 1.5):
                references = [budget_reference(x, y, width, height, seconds)
                              for y in range(height) for x in range(width)]
                marginal = sum(margin < 0.00001 for _, margin in references)
                print(f"Budget reference {width}x{height} t={seconds}: {marginal} near-threshold pixels")
                for target in Target:
                    image = self.render("budget", target, width, height, seconds)
                    for index, (expected, margin) in enumerate(references):
                        regimes.add("hit" if expected[1] else "distance" if expected[2] == 255 else "cap")
                        actual = image[index * 4:index * 4 + 4]
                        self.assertEqual(actual[1:], bytes(expected[1:]),
                                         f"{target.value}, {width}x{height}, t={seconds}, pixel {index}")
                        self.assertAlmostEqual(actual[0], expected[0], delta=5 if margin < 0.00001 else 1)
        self.assertEqual(regimes, {"hit", "distance", "cap"})

    @unittest.skipUnless(os.environ.get("MLIR_GPU_TEST_WINDOW") == "1", "Enable desktop window acceptance")
    def test_metal_live_clock_retina_pixels_and_resize_using_actual_drawable_readback(self):
        executable = self.directory / "metal-window-probe"
        sdk = os.environ.get("METAL_CONVERTER_RUNTIME_INCLUDE_DIR", "/usr/local/include")
        compiled = subprocess.run([
            "clang++", "-std=c++17", "-fobjc-arc", f"-I{sdk}", str(ROOT / "tests/metal_window_probe.mm"),
            "-framework", "Cocoa", "-framework", "Metal", "-framework", "MetalKit", "-o", str(executable),
        ], check=False, capture_output=True, text=True)
        self.assertEqual(compiled.returncode, 0, compiled.stderr)
        for target in (Target.DIRECTX, Target.METAL):
            with _metal_libraries(self.programs["frame", target], TOOLS) as (_, arguments):
                flags = ("--vertex", "--vertex-entry", "--fragment", "--fragment-entry", "--frame-layout")
                text = _metal_run([str(executable), *(arguments[arguments.index(flag) + 1] for flag in flags)],
                                  self.validation, True)
                first, advanced, resized, frozen = json.loads(text)
                for report in (first, advanced, resized, frozen):
                    self.assertEqual(report["rgba"][:2], [report["width"], report["height"]])
                    self.assertEqual(report["rgba"][3], 255)
                    self.assertAlmostEqual(report["width"], report["points_width"] * report["scale"], delta=1)
                self.assertGreater(advanced["rgba"][2], first["rgba"][2] + 4)
                self.assertNotEqual((first["width"], first["height"]), (resized["width"], resized["height"]))
                self.assertEqual(resized["rgba"], frozen["rgba"])
                self.assertEqual(frozen["rgba"][2], round(1.5 / 4 * 255))
