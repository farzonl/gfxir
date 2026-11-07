import json
import os
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path

from mlir_gpu import CompileError, Toolchain


ROOT = Path(__file__).resolve().parents[1]
TOOLS = Toolchain(Path(os.environ["LLVM_BIN"]) if "LLVM_BIN" in os.environ else None)


def available():
    try:
        TOOLS.find("gfx-opt")
        TOOLS.find("gfx-compile")
    except CompileError:
        return False
    return True


@unittest.skipUnless(available(), "Build tools/build_toolchain.py and select its LLVM_BIN for native gfx IR tests")
class NativeIRTests(unittest.TestCase):
    def optimize(self, source, *passes):
        return subprocess.run(
            [TOOLS.find("gfx-opt"), str(source), *passes],
            text=True, capture_output=True, check=False,
        )

    def test_graphics_ods_round_trip_and_verification(self):
        result = self.optimize(ROOT / "tests/ir/graphics.mlir", "--gfx-verify")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("#gfx.stage<vertex>", result.stdout)
        self.assertIn("vector<2xf32>", result.stdout)
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "roundtrip.mlir"
            path.write_text(result.stdout)
            again = self.optimize(path, "--gfx-verify")
            self.assertEqual(again.returncode, 0, again.stderr)
            self.assertEqual(again.stdout, result.stdout)

    def test_pointer_and_bindless_types_round_trip_without_assuming_target_abi(self):
        result = self.optimize(ROOT / "tests/ir/types.mlir")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("!gfx.ptr<f32, global>", result.stdout)
        self.assertIn("!gfx.handle<texture_2d>", result.stdout)

    def test_compute_entry_contract_is_first_class_but_not_silently_lowered_as_graphics(self):
        source = ROOT / "tests/ir/compute.mlir"
        verified = self.optimize(source, "--gfx-verify")
        self.assertEqual(verified.returncode, 0, verified.stderr)
        for conversion in ("--convert-gfx-to-spirv", "--gfx-to-directx"):
            result = self.optimize(source, conversion)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("graphics-first", result.stderr)

    def test_compute_dimensions_are_verified(self):
        source = (ROOT / "tests/ir/compute.mlir").read_text()
        with tempfile.TemporaryDirectory() as work:
            path = Path(work) / "invalid.mlir"
            path.write_text(source.replace("8, 8, 1", "0, 8, 1"))
            result = self.optimize(path, "--gfx-verify")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("three positive dimensions", result.stderr)

    def test_upstream_spirv_conversion(self):
        result = self.optimize(ROOT / "tests/ir/graphics.mlir", "--convert-gfx-to-spirv")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("spirv.module", result.stdout)
        self.assertIn('built_in("Position")', result.stdout)
        self.assertIn("spirv.FNegate", result.stdout)
        self.assertNotIn("gfx.output", result.stdout)
        self.assertNotIn("arith.", result.stdout)

    def test_upstream_llvm_conversion_keeps_intrinsics_for_directx_backend(self):
        result = self.optimize(ROOT / "tests/ir/graphics.mlir", "--gfx-to-directx")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("llvm.dx.load.input", result.stdout)
        self.assertIn("llvm.dx.store.output", result.stdout)
        self.assertNotIn("dx.op.", result.stdout)
        self.assertNotIn("gfx.output", result.stdout)

    def test_invalid_graph_reference_is_a_diagnostic_not_a_crash_or_binary(self):
        pipeline = runpy.run_path(str(ROOT / "examples/triangle.py"))["pipeline"]
        shader = pipeline.trace()[0].to_dict()
        shader["nodes"][0]["args"] = [999]
        with tempfile.TemporaryDirectory() as work:
            directory = Path(work)
            source, output, reflection = directory / "bad.json", directory / "bad.spv", directory / "bad.reflection.json"
            source.write_text(json.dumps(shader))
            result = subprocess.run(
                [TOOLS.find("gfx-compile"), str(source), "--target", "vulkan",
                 "--output", str(output), "--reflection", str(reflection)],
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("earlier nodes", result.stderr)
            self.assertFalse(output.exists())
            self.assertFalse(reflection.exists())


if __name__ == "__main__":
    unittest.main()
