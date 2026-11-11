"""Unit tests for Schema v5 multi-draw pipeline manifests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mlir_gpu.compiler import (
    MULTI_DRAW_SCHEMA_VERSION,
    PIPELINE_SCHEMA_VERSION,
    DrawCall,
    MultiDrawManifest,
    verify_multi_draw_manifest,
)
from mlir_gpu.viewer import ViewerError, load_program


class MultiDrawManifestValidationTests(unittest.TestCase):
    """Validation tests for Schema v5 multi-draw manifests and DrawCall descriptors."""

    def test_multi_draw_schema_version_is_distinct_from_compute(self):
        self.assertEqual(MULTI_DRAW_SCHEMA_VERSION, 6)

    def test_draw_call_standard_defaults_and_serialization(self):
        dc = DrawCall(pipeline="p1", vertex_count=36)
        self.assertEqual(dc.pipeline, "p1")
        self.assertEqual(dc.type, "standard")
        self.assertEqual(dc.vertex_count, 36)
        self.assertEqual(dc.first_vertex, 0)
        self.assertEqual(dc.instance_count, 1)
        d = dc.to_dict()
        self.assertEqual(d["pipeline"], "p1")
        self.assertEqual(d["type"], "standard")
        self.assertEqual(d["vertex_count"], 36)

    def test_draw_call_indexed_parameters_and_serialization(self):
        dc = DrawCall(
            pipeline="p2",
            type="indexed",
            index_count=72,
            first_index=12,
            vertex_offset=4,
            index_type="uint32",
            constant_offset=80,
            constant_size=80,
        )
        self.assertEqual(dc.type, "indexed")
        self.assertEqual(dc.index_count, 72)
        d = dc.to_dict()
        self.assertEqual(d["type"], "indexed")
        self.assertEqual(d["index_count"], 72)
        self.assertEqual(d["first_index"], 12)
        self.assertEqual(d["vertex_offset"], 4)
        self.assertEqual(d["index_type"], "uint32")
        self.assertEqual(d["constant_offset"], 80)
        self.assertEqual(d["constant_size"], 80)

    def test_draw_call_rejects_invalid_type(self):
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", type="point_cloud")

    def test_draw_call_rejects_unsupported_topology(self):
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", topology="line-strip")

    def test_draw_call_rejects_non_multiple_of_three_vertices(self):
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", type="standard", vertex_count=7)

    def test_draw_call_rejects_non_multiple_of_three_indices(self):
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", type="indexed", index_count=10)

    def test_draw_call_rejects_negative_counts_or_offsets(self):
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", vertex_count=-3)
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", first_vertex=-1)
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", constant_offset=-10)
        with self.assertRaises(ValueError):
            DrawCall(pipeline="p1", instance_count=0)

    def test_multi_draw_manifest_valid_construction_and_dict(self):
        manifest = MultiDrawManifest(
            pipelines={"pbr": {"shader": "pbr"}, "unlit": {"shader": "unlit"}},
            draws=[
                DrawCall(pipeline="pbr", vertex_count=36),
                {"pipeline": "unlit", "vertex_count": 6, "constant_offset": 80, "constant_size": 80},
            ],
        )
        d = manifest.to_dict()
        self.assertEqual(d["schema_version"], 6)
        self.assertIn("pbr", d["pipelines"])
        self.assertIn("unlit", d["pipelines"])
        self.assertEqual(len(d["draws"]), 2)
        self.assertEqual(d["draws"][0]["vertex_count"], 36)
        self.assertEqual(d["draws"][1]["constant_offset"], 80)

    def test_multi_draw_manifest_empty_draws_allowed(self):
        manifest = MultiDrawManifest(pipelines={"p1": {}}, draws=[])
        d = manifest.to_dict()
        self.assertEqual(len(d["draws"]), 0)

    def test_multi_draw_manifest_rejects_schema_version_mismatch(self):
        with self.assertRaises(ValueError):
            MultiDrawManifest(pipelines={"p1": {}}, draws=[], schema_version=4)

    def test_multi_draw_manifest_rejects_nonexistent_pipeline_reference(self):
        with self.assertRaises(ValueError):
            MultiDrawManifest(pipelines={"p1": {}}, draws=[{"pipeline": "missing_p2"}])

    def test_multi_draw_manifest_rejects_negative_draw_counts(self):
        with self.assertRaises(ValueError):
            MultiDrawManifest(pipelines={"p1": {}}, draws=[{"pipeline": "p1", "vertex_count": -3}])

    def test_multi_draw_manifest_high_draw_count_scaling(self):
        draw_count = 1000
        draws = [DrawCall(pipeline="p1", vertex_count=3) for _ in range(draw_count)]
        manifest = MultiDrawManifest(pipelines={"p1": {}}, draws=draws)
        d = manifest.to_dict()
        self.assertEqual(len(d["draws"]), draw_count)

    def test_verify_multi_draw_manifest_success_and_failures(self):
        valid = {
            "schema_version": 6,
            "pipelines": {"p1": {}},
            "draws": [{"pipeline": "p1", "vertex_count": 3}],
        }
        verify_multi_draw_manifest(valid)

        bad_version = dict(valid, schema_version=4)
        with self.assertRaises(ValueError):
            verify_multi_draw_manifest(bad_version)

        bad_pipeline = dict(valid, draws=[{"pipeline": "unknown"}])
        with self.assertRaises(ValueError):
            verify_multi_draw_manifest(bad_pipeline)

        negative_count = dict(valid, draws=[{"pipeline": "p1", "vertex_count": -1}])
        with self.assertRaises(ValueError):
            verify_multi_draw_manifest(negative_count)


class MultiDrawProgramLoaderTests(unittest.TestCase):
    """End-to-end program loading tests with Schema v3, v4, and v5 manifests."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp_dir.name)
        # Create minimal valid SPIR-V files
        self.vs_path = self.dir / "vs.spv"
        self.fs_path = self.dir / "fs.spv"
        # 0x07230203 little-endian magic, 20 bytes total, word-aligned
        spirv_bytes = b"#" + bytes(16)
        self.vs_path.write_bytes(spirv_bytes)
        self.fs_path.write_bytes(spirv_bytes)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_schema_five_program_loads_multiple_pipelines_and_draws(self):
        manifest_data = {
            "schema_version": 6,
            "target": "vulkan",
            "raster_convention": {
                "clip_x": "right",
                "clip_y": "up",
                "clip_z": "zero-to-one",
                "framebuffer_origin": "top-left",
                "pixel_centers": "half-integer",
                "color": "linear-rgba8-unorm",
                "presentation_color_space": "srgb",
                "samples": 1,
                "culling": "none",
            },
            "pipelines": {
                "pipe_alpha": {
                    "entries": [
                        {"stage": "vertex", "entry_point": "vs_main", "binary": "vs.spv"},
                        {"stage": "fragment", "entry_point": "fs_main", "binary": "fs.spv"},
                    ],
                },
                "pipe_beta": {
                    "entries": [
                        {"stage": "vertex", "entry_point": "vs_main", "binary": "vs.spv"},
                        {"stage": "fragment", "entry_point": "fs_main", "binary": "fs.spv"},
                    ],
                },
            },
            "draws": [
                {"pipeline": "pipe_alpha", "type": "standard", "vertex_count": 36},
                {"pipeline": "pipe_beta", "type": "indexed", "index_count": 72, "constant_offset": 80, "constant_size": 80},
            ],
        }
        manifest_file = self.dir / "pipeline.json"
        manifest_file.write_text(json.dumps(manifest_data))

        program = load_program(manifest_file)
        self.assertIn("pipe_alpha", program.pipelines)
        self.assertIn("pipe_beta", program.pipelines)
        self.assertEqual(len(program.draws), 2)
        self.assertEqual(program.draws[0].vertex_count, 36)
        self.assertEqual(program.draws[1].index_count, 72)
        self.assertEqual(program.draws[1].constant_offset, 80)
        self.assertEqual(program.target, "vulkan")

    def test_schema_five_program_rejects_missing_pipelines(self):
        manifest_data = {
            "schema_version": 6,
            "target": "vulkan",
            "raster_convention": {
                "clip_x": "right",
                "clip_y": "up",
                "clip_z": "zero-to-one",
                "framebuffer_origin": "top-left",
                "pixel_centers": "half-integer",
                "color": "linear-rgba8-unorm",
                "presentation_color_space": "srgb",
                "samples": 1,
                "culling": "none",
            },
            "pipelines": {},
            "draws": [],
        }
        manifest_file = self.dir / "pipeline.json"
        manifest_file.write_text(json.dumps(manifest_data))
        with self.assertRaises(ViewerError):
            load_program(manifest_file)

    def test_backward_compatibility_schema_three_normalizes_to_draw_sequence(self):
        manifest_data = {
            "schema_version": 3,
            "target": "vulkan",
            "raster_convention": {
                "clip_x": "right",
                "clip_y": "up",
                "clip_z": "zero-to-one",
                "framebuffer_origin": "top-left",
                "pixel_centers": "half-integer",
                "color": "linear-rgba8-unorm",
                "presentation_color_space": "srgb",
                "samples": 1,
                "culling": "none",
            },
            "entries": [
                {"stage": "vertex", "entry_point": "vs_main", "binary": "vs.spv"},
                {"stage": "fragment", "entry_point": "fs_main", "binary": "fs.spv"},
            ],
            "interface": {
                "vertex": {
                    "name": "vs_main", "stage": "vertex",
                    "inputs": [{"name": "vertex_index", "builtin": "vertex_index", "type": "u32"}],
                    "outputs": [{"name": "position", "builtin": "position", "type": "f32x4"}, {"name": "color", "type": "f32x4", "location": 0, "interpolation": "perspective"}],
                },
                "fragment": {
                    "name": "fs_main", "stage": "fragment",
                    "inputs": [{"name": "color", "type": "f32x4", "location": 0, "interpolation": "perspective"}],
                    "outputs": [{"name": "target0", "type": "f32x4", "location": 0, "interpolation": "perspective"}],
                },
            },
            "draw": {"topology": "triangle-list", "vertex_count": 3},
        }
        manifest_file = self.dir / "pipeline.json"
        manifest_file.write_text(json.dumps(manifest_data))

        program = load_program(manifest_file)
        self.assertEqual(len(program.draws), 1)
        self.assertEqual(program.draws[0].pipeline, "default")
        self.assertEqual(program.draws[0].vertex_count, 3)
        self.assertIn("default", program.pipelines)


if __name__ == "__main__":
    unittest.main()
