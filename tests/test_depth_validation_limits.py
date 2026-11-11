import json
import math
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path
from unittest.mock import MagicMock, patch

from mlir_gpu import (
    DepthStencil,
    Draw,
    MLIRPipeline,
    Pipeline,
    compile_pipeline,
    fragment,
    vertex,
)
from mlir_gpu.compiler import PIPELINE_SCHEMA_VERSION, RASTER_CONVENTION
from mlir_gpu.interfaces import (
    COMPARE_OPS,
    SUPPORTED_DEPTH_FORMATS,
    verify_depth_stencil,
)
from mlir_gpu.viewer import (
    GraphicsProgram,
    Shader,
    ViewerError,
    _directx_shaders,
    _metal_libraries,
    _vulkan_depth_format,
    load_program,
)


class DepthFormatValidationLimitsTests(unittest.TestCase):
    def test_supported_format_d32_sfloat_accepted(self):
        config = {
            "format": "d32_sfloat",
            "depth_test_enable": True,
            "depth_write_enable": True,
            "depth_compare_op": "less_equal",
        }
        verify_depth_stencil(config)

    def test_unsupported_string_formats_rejected(self):
        unsupported = [
            "d16_unorm",
            "d24_unorm_s8_uint",
            "d32_sfloat_s8_uint",
            "d24_unorm",
            "d16_sfloat",
            "rgba8_unorm",
            "bgra8_unorm",
            "",
            "   ",
            "D32_SFLOAT",
            "d32_sfloat ",
            " d32_sfloat",
            "d32_sfloat\0",
        ]
        for fmt in unsupported:
            with self.subTest(fmt=fmt):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": fmt,
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": "less_equal",
                    })

    def test_non_string_formats_rejected(self):
        non_strings = [
            None,
            126,
            32.0,
            True,
            False,
            ["d32_sfloat"],
            {"format": "d32_sfloat"},
            b"d32_sfloat",
        ]
        for val in non_strings:
            with self.subTest(val=val):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": val,
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": "less_equal",
                    })

    def test_missing_format_key_rejected(self):
        with self.assertRaises(ValueError):
            verify_depth_stencil({
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_compare_op": "less_equal",
            })


class DepthCompareOperatorLimitsTests(unittest.TestCase):
    def test_all_eight_valid_compare_ops_accepted(self):
        expected_ops = {
            "never", "less", "equal", "less_equal",
            "greater", "not_equal", "greater_equal", "always",
        }
        self.assertEqual(COMPARE_OPS, expected_ops)
        for op in expected_ops:
            with self.subTest(op=op):
                config = {
                    "format": "d32_sfloat",
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_compare_op": op,
                }
                verify_depth_stencil(config)

    def test_invalid_compare_ops_rejected(self):
        invalid_ops = [
            "foo",
            "LESS_EQUAL",
            "less_than",
            "greater_or_equal",
            "le",
            "ge",
            "==",
            "<=",
            "",
            " ",
            "less_equal ",
            " less_equal",
            None,
            1,
            0,
            True,
            False,
            ["less_equal"],
            {"op": "less_equal"},
        ]
        for bad_op in invalid_ops:
            with self.subTest(bad_op=bad_op):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": bad_op,
                    })

    def test_missing_compare_op_key_rejected(self):
        with self.assertRaises(ValueError):
            verify_depth_stencil({
                "format": "d32_sfloat",
                "depth_test_enable": True,
                "depth_write_enable": True,
            })


class DepthClearValueBoundaryLimitsTests(unittest.TestCase):
    def test_valid_clear_depth_boundary_values(self):
        valid_values = [0.0, 1.0, 0.5, 0, 1, 1e-6, 1.0 - 1e-6]
        for val in valid_values:
            with self.subTest(val=val):
                verify_depth_stencil({
                    "format": "d32_sfloat",
                    "depth_test_enable": True,
                    "depth_write_enable": True,
                    "depth_compare_op": "less_equal",
                    "clear_depth": val,
                })

    def test_out_of_bounds_clear_depth_rejected(self):
        out_of_bounds = [-0.1, -0.00001, -1.0, -100.0, 1.00001, 1.1, 2.0, 100.0]
        for bad_val in out_of_bounds:
            with self.subTest(bad_val=bad_val):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": "less_equal",
                        "clear_depth": bad_val,
                    })

    def test_non_finite_clear_depth_rejected(self):
        non_finite = [float("nan"), float("inf"), float("-inf"), math.nan, math.inf, -math.inf]
        for nf in non_finite:
            with self.subTest(nf=nf):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": "less_equal",
                        "clear_depth": nf,
                    })

    def test_non_numeric_clear_depth_types_rejected(self):
        invalid_types = [
            True,
            False,
            "1.0",
            "0.0",
            "0",
            "",
            None,
            [1.0],
            {"depth": 1.0},
        ]
        for val in invalid_types:
            with self.subTest(val=val):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": "less_equal",
                        "clear_depth": val,
                    })


class DepthBooleanFlagTypeStrictnessTests(unittest.TestCase):
    def test_strict_boolean_depth_test_enable(self):
        for valid in [True, False]:
            verify_depth_stencil({
                "format": "d32_sfloat",
                "depth_test_enable": valid,
                "depth_write_enable": True,
                "depth_compare_op": "less_equal",
            })

        invalid_values = [1, 0, "true", "false", "True", "False", "", None, [True], 1.0, 0.0]
        for bad in invalid_values:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": "d32_sfloat",
                        "depth_test_enable": bad,
                        "depth_write_enable": True,
                        "depth_compare_op": "less_equal",
                    })

    def test_strict_boolean_depth_write_enable(self):
        for valid in [True, False]:
            verify_depth_stencil({
                "format": "d32_sfloat",
                "depth_test_enable": True,
                "depth_write_enable": valid,
                "depth_compare_op": "less_equal",
            })

        invalid_values = [1, 0, "true", "false", "True", "False", "", None, [True], 1.0, 0.0]
        for bad in invalid_values:
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    verify_depth_stencil({
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": bad,
                        "depth_compare_op": "less_equal",
                    })


class DepthStateImmutabilityTests(unittest.TestCase):
    def test_depth_stencil_frozen_attributes(self):
        ds = DepthStencil()
        with self.assertRaises(FrozenInstanceError):
            ds.format = "d16_unorm"
        with self.assertRaises(FrozenInstanceError):
            ds.depth_test_enable = False
        with self.assertRaises(FrozenInstanceError):
            ds.depth_write_enable = False
        with self.assertRaises(FrozenInstanceError):
            ds.depth_compare_op = "always"
        with self.assertRaises(FrozenInstanceError):
            ds.clear_depth = 0.5
        with self.assertRaises(FrozenInstanceError):
            del ds.format

    def test_pipeline_frozen_attributes(self):
        @vertex
        def vs():
            return None

        @fragment
        def fs():
            return None

        p = Pipeline(vs, fs, depth_stencil=DepthStencil())
        with self.assertRaises(FrozenInstanceError):
            p.depth_stencil = None
        with self.assertRaises(FrozenInstanceError):
            p.depth_stencil = DepthStencil(depth_compare_op="always")
        with self.assertRaises(FrozenInstanceError):
            p.vertex = vs
        with self.assertRaises(FrozenInstanceError):
            p.fragment = fs
        with self.assertRaises(FrozenInstanceError):
            p.draw = Draw()

    def test_mlir_pipeline_frozen_attributes(self):
        mlp = MLIRPipeline(Path("v.mlir"), Path("f.mlir"), "vs", "fs", depth_stencil=DepthStencil())
        with self.assertRaises(FrozenInstanceError):
            mlp.depth_stencil = None
        with self.assertRaises(FrozenInstanceError):
            mlp.depth_stencil = DepthStencil(depth_compare_op="always")


class DepthManifestRoundTripTests(unittest.TestCase):
    def test_depth_stencil_dict_round_trip(self):
        ds = DepthStencil(
            format="d32_sfloat",
            depth_test_enable=True,
            depth_write_enable=False,
            depth_compare_op="greater_equal",
            clear_depth=0.75,
        )
        as_dict = ds.to_dict()
        serialized = json.dumps(as_dict)
        deserialized = json.loads(serialized)
        self.assertEqual(deserialized, as_dict)

        reconstructed = DepthStencil(**deserialized)
        self.assertEqual(reconstructed, ds)

    def test_pipeline_manifest_preserves_raster_convention(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_file = Path(temp) / "pipeline.json"
            pipe = MLIRPipeline(
                Path("v.mlir"), Path("f.mlir"), "vs", "fs",
                depth_stencil=DepthStencil(depth_compare_op="less_equal"),
            )
            data = {
                "schema_version": PIPELINE_SCHEMA_VERSION,
                "target": "vulkan",
                "entries": [],
                "interface": {"vertex": {}, "fragment": {}},
                "draw": pipe.draw.to_dict(),
                "raster_convention": RASTER_CONVENTION,
                "runtime_verified": False,
                "backend_status": "validated-binary",
                "depth_stencil": pipe.depth_stencil.to_dict(),
            }
            manifest_file.write_text(json.dumps(data, indent=2))
            loaded = json.loads(manifest_file.read_text())

            self.assertEqual(loaded["raster_convention"]["clip_z"], "zero-to-one")
            self.assertEqual(loaded["raster_convention"], RASTER_CONVENTION)
            self.assertIn("depth_stencil", loaded)
            self.assertEqual(loaded["depth_stencil"]["format"], "d32_sfloat")
            self.assertTrue(loaded["depth_stencil"]["depth_test_enable"])
            self.assertTrue(loaded["depth_stencil"]["depth_write_enable"])
            self.assertEqual(loaded["depth_stencil"]["depth_compare_op"], "less_equal")
            self.assertEqual(loaded["depth_stencil"]["clear_depth"], 1.0)

    def test_pipeline_dict_coercion_and_invalid_type_rejection(self):
        @vertex
        def vs():
            return None

        @fragment
        def fs():
            return None

        valid_dict = {
            "format": "d32_sfloat",
            "depth_test_enable": False,
            "depth_write_enable": False,
            "depth_compare_op": "never",
            "clear_depth": 0.0,
        }
        p = Pipeline(vs, fs, depth_stencil=valid_dict)
        self.assertIsInstance(p.depth_stencil, DepthStencil)
        self.assertFalse(p.depth_stencil.depth_test_enable)
        self.assertEqual(p.depth_stencil.depth_compare_op, "never")

        with self.assertRaises(TypeError):
            Pipeline(vs, fs, depth_stencil="d32_sfloat")
        with self.assertRaises(TypeError):
            Pipeline(vs, fs, depth_stencil=[valid_dict])
        with self.assertRaises(TypeError):
            Pipeline(vs, fs, depth_stencil=123)

        invalid_dict = {"format": "d16_unorm", "depth_test_enable": True, "depth_write_enable": True, "depth_compare_op": "less"}
        with self.assertRaises(ValueError):
            Pipeline(vs, fs, depth_stencil=invalid_dict)


class DepthBackendCliGenerationTests(unittest.TestCase):
    def test_directx_shaders_cli_all_ops(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-directx-view"

        for op in COMPARE_OPS:
            with self.subTest(op=op):
                prog = GraphicsProgram(
                    Shader("vs", b"DXBC" + b"\x00" * 30),
                    Shader("fs", b"DXBC" + b"\x00" * 30),
                    target="directx",
                    depth_stencil={
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": True,
                        "depth_compare_op": op,
                    },
                )
                with patch("sys.platform", "win32"):
                    with _directx_shaders(prog, toolchain) as (dir_path, args):
                        self.assertIn("--depth", args)
                        self.assertIn("--depth-compare", args)
                        idx = args.index("--depth-compare")
                        self.assertEqual(args[idx + 1], op)
                        self.assertIn("--depth-write", args)
                        w_idx = args.index("--depth-write")
                        self.assertEqual(args[w_idx + 1], "1")
                        self.assertIn("--depth-format", args)
                        f_idx = args.index("--depth-format")
                        self.assertEqual(args[f_idx + 1], "d32_sfloat")

    @unittest.skipUnless(sys.platform == "darwin", "Metal requires macOS")
    def test_metal_libraries_cli_all_ops(self):
        toolchain = MagicMock()
        toolchain.find.return_value = "mlir-gpu-metal-view"

        for op in COMPARE_OPS:
            with self.subTest(op=op):
                prog = GraphicsProgram(
                    Shader("vs", b"MTLB" + b"\x00" * 20),
                    Shader("fs", b"MTLB" + b"\x00" * 20),
                    target="metal",
                    depth_stencil={
                        "format": "d32_sfloat",
                        "depth_test_enable": True,
                        "depth_write_enable": False,
                        "depth_compare_op": op,
                    },
                )
                with _metal_libraries(prog, toolchain) as (dir_path, args):
                    self.assertIn("--depth", args)
                    self.assertIn("--depth-compare", args)
                    idx = args.index("--depth-compare")
                    self.assertEqual(args[idx + 1], op)
                    self.assertIn("--depth-write", args)
                    w_idx = args.index("--depth-write")
                    self.assertEqual(args[w_idx + 1], "0")
                    self.assertIn("--depth-format", args)
                    f_idx = args.index("--depth-format")
                    self.assertEqual(args[f_idx + 1], "d32_sfloat")


if __name__ == "__main__":
    unittest.main()
