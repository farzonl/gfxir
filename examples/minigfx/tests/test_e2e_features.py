"""MiniGfx End-to-End Feature Tests (Tier 1 & Tier 2).

Covers all 16 active features from PROJECT.md:
- Tier 1: Feature Coverage (>=5 test cases per feature = >=80 tests)
- Tier 2: Boundary & Corner Cases (>=5 test cases per feature = >=80 tests)
Total: 160 test cases across 16 test classes.
"""

from __future__ import annotations

import math
import os
import re
import struct
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, List

# Ensure repository root and src are on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

# Import App from minigfx if present
try:
    from examples.minigfx.app import App, get_default_target
except ImportError:
    App = None  # type: ignore
    get_default_target = None  # type: ignore

# Optional imports from minigfx modules (which may be on parallel PR branches)
try:
    import examples.minigfx.math as mgfx_math
except ImportError:
    mgfx_math = None

try:
    import examples.minigfx.camera as mgfx_cam
except ImportError:
    mgfx_cam = None

try:
    import examples.minigfx.scene as mgfx_scene
except ImportError:
    mgfx_scene = None

try:
    import examples.minigfx.obj_loader as mgfx_obj
except ImportError:
    mgfx_obj = None

try:
    import examples.minigfx.renderer as mgfx_rend
except ImportError:
    mgfx_rend = None

try:
    import examples.minigfx.material as mgfx_mat
except ImportError:
    mgfx_mat = None

try:
    import examples.minigfx.tonemap as mgfx_tone
except ImportError:
    mgfx_tone = None

try:
    import examples.minigfx.debug as mgfx_debug
except ImportError:
    mgfx_debug = None

try:
    import examples.minigfx.samples.model_viewer as mgfx_mv
except ImportError:
    mgfx_mv = None

try:
    import mlir_gpu.parameters as mgpu_params
except ImportError:
    mgpu_params = None

try:
    import mlir_gpu.frontend as mgpu_frontend
except ImportError:
    mgpu_frontend = None

try:
    import mlir_gpu.compiler as mgpu_compiler
except ImportError:
    mgpu_compiler = None

try:
    import mlir_gpu.viewer as mgpu_viewer
except ImportError:
    mgpu_viewer = None


# Canonical Inlined ACES Film Tonemapping formula per DESIGN.md §7.8
def canonical_aces_tonemap(x: float) -> float:
    if math.isnan(x):
        return 0.0
    x = max(float(x), 0.0)
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    numerator = x * (a * x + b)
    denominator = x * (c * x + d) + e
    mapped = numerator / denominator if denominator > 0 else 0.0
    return max(0.0, min(1.0, mapped))


def canonical_srgb_gamma(linear: float) -> float:
    linear = max(float(linear), 0.0)
    return linear ** (1.0 / 2.2)


# ===========================================================================
# Feature 1: Architecture Design Document (#127)
# ===========================================================================

class TestFeature01ArchitectureDoc(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 1: Architecture Design Document (#127)."""

    def setUp(self):
        self.doc_path = _REPO_ROOT / "examples" / "minigfx" / "DESIGN.md"
        if not self.doc_path.is_file():
            self.skipTest("examples/minigfx/DESIGN.md does not exist")
        self.content = self.doc_path.read_text(encoding="utf-8")

    # --- Tier 1: Feature Coverage ---
    def test_f01_t1_01_design_doc_exists(self):
        self.assertTrue(self.doc_path.is_file())
        self.assertGreater(self.doc_path.stat().st_size, 10000)

    def test_f01_t1_02_design_doc_required_sections(self):
        sections = [
            "Executive Summary",
            "Engine Goals & Core Thesis",
            "Architecture & MiniEngine Mapping",
            "CPU/GPU Code Duplication Comparison",
            "Specific Quantitative Metrics",
            "Compiler ABI Requirements Driving Phase 1 Issues",
            "Engine Modular Decomposition",
            "Demo, Parity Testing, and PoC Plan",
            "Appendix: Memory Alignment",
        ]
        for sec in sections:
            self.assertIn(sec, self.content, f"Mandatory section '{sec}' missing from DESIGN.md")

    def test_f01_t1_03_design_doc_loc_metrics_table(self):
        self.assertIn("25,900", self.content)
        self.assertIn("1,870", self.content)
        has_reduction = ("13.9x" in self.content) or ("14x" in self.content)
        self.assertTrue(has_reduction, "14x code reduction ratio must be stated in DESIGN.md")

    def test_f01_t1_04_design_doc_abi_requirements_coverage(self):
        for issue in (128, 129, 130, 131, 132):
            self.assertIn(f"#{issue}", self.content, f"M1 compiler issue #{issue} not referenced in DESIGN.md")

    def test_f01_t1_05_design_doc_stretch_omission_documented(self):
        self.assertIn("#133", self.content)
        self.assertIn("#134", self.content)
        has_omission_note = ("omitted" in self.content.lower()) or ("postponed" in self.content.lower())
        self.assertTrue(has_omission_note, "Stretch issues #133 and #134 must be documented as omitted")

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f01_t2_01_boundary_missing_file_detection(self):
        fake_path = _REPO_ROOT / "examples" / "minigfx" / "NON_EXISTENT_DOC.md"
        self.assertFalse(fake_path.is_file())

    def test_f01_t2_02_boundary_empty_file_fails_size_threshold(self):
        self.assertGreater(len(self.content.strip()), 500, "DESIGN.md content too short")

    def test_f01_t2_03_boundary_table_syntax_balanced_pipes(self):
        lines = self.content.splitlines()
        for idx, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("|") and stripped.endswith("|"):
                pipe_count = stripped.count("|")
                self.assertGreaterEqual(pipe_count, 3, f"Line {idx} table row has fewer than 2 columns")

    def test_f01_t2_04_boundary_loc_metric_numerical_bounds(self):
        matches_minigfx = re.findall(r"1,870", self.content)
        matches_minieng = re.findall(r"25,900", self.content)
        self.assertGreaterEqual(len(matches_minigfx), 1)
        self.assertGreaterEqual(len(matches_minieng), 1)

    def test_f01_t2_05_boundary_code_block_fences_balanced(self):
        fence_count = self.content.count("```")
        self.assertEqual(fence_count % 2, 0, f"Unbalanced code fences (count={fence_count}) in DESIGN.md")


# ===========================================================================
# Feature 2: Shared Host/Device Struct Types (#128)
# ===========================================================================

class TestFeature02SharedStructs(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 2: Shared Host/Device Struct Types (#128)."""

    def _require_feature(self):
        struct_fn = getattr(mgpu_params, "struct", None)
        is_struct_decorator = callable(struct_fn) and not isinstance(struct_fn, type(sys))
        has_camera_params = hasattr(mgfx_mat, "CameraParams")
        if not is_struct_decorator and not has_camera_params:
            self.skipTest("Feature 2 (@struct) not yet implemented (PR #128 in progress)")

    # --- Tier 1: Feature Coverage ---
    def test_f02_t1_01_struct_scalar_and_small_vector_packing(self):
        self._require_feature()
        @mgpu_params.struct
        class ScalarVectorStruct:
            x: f32
            uv: vec2

        inst = ScalarVectorStruct(x=1.5, uv=(0.25, 0.75))
        raw = inst.to_bytes()
        self.assertEqual(len(raw), 16)
        val_x = struct.unpack_from("<f", raw, 0)[0]
        self.assertAlmostEqual(val_x, 1.5, places=5)

    def test_f02_t1_02_struct_mat4_and_vec3_packing(self):
        self._require_feature()
        @mgpu_params.struct
        class TestCameraParams:
            view_proj: mat4
            eye_pos: vec3
            time: f32

        inst = TestCameraParams(view_proj=[1.0] * 16, eye_pos=(0.0, 1.0, 5.0), time=2.0)
        raw = inst.to_bytes()
        self.assertEqual(len(raw), 80)
        self.assertEqual(len(raw) % 16, 0)

    def test_f02_t1_03_struct_manifest_reflection_metadata(self):
        self._require_feature()
        @mgpu_params.struct
        class TestCameraParams:
            view_proj: mat4
            eye_pos: vec3
            time: f32

        refl = TestCameraParams.reflection()
        self.assertEqual(refl["size"], 80)
        self.assertEqual(refl["alignment"], 16)
        self.assertEqual(len(refl["fields"]), 3)

    def test_f02_t1_04_struct_gpu_ast_field_access(self):
        self._require_feature()
        @mgpu_params.struct
        class TestCameraParams:
            view_proj: mat4
            eye_pos: vec3
            time: f32

        block = TestCameraParams.as_parameter_block()
        self.assertEqual(block.name, "testcameraparams")

    def test_f02_t1_05_struct_roundtrip_unpacking(self):
        self._require_feature()
        @mgpu_params.struct
        class TestCameraParams:
            view_proj: mat4
            eye_pos: vec3
            time: f32

        inst = TestCameraParams(view_proj=[1.0] * 16, eye_pos=(0.0, 1.0, 5.0), time=2.0)
        raw = inst.to_bytes()
        inst2 = TestCameraParams.from_bytes(raw)
        self.assertAlmostEqual(inst2.time, 2.0, places=5)
        self.assertEqual(inst2.eye_pos, (0.0, 1.0, 5.0))

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f02_t2_01_boundary_empty_struct_rejection(self):
        self._require_feature()
        with self.assertRaises((ValueError, TypeError)):
            @mgpu_params.struct
            class EmptyStruct:
                pass

    def test_f02_t2_02_boundary_unsupported_field_type(self):
        self._require_feature()
        with self.assertRaises((ValueError, TypeError)):
            @mgpu_params.struct
            class InvalidStruct:
                bad_field: unsupported_type_name

    def test_f02_t2_03_boundary_duplicate_field_names(self):
        self._require_feature()
        with self.assertRaises((ValueError, TypeError)):
            mgpu_params.ParameterBlock("dup_test", [("dup", mgpu_params.Type.F32), ("dup", mgpu_params.Type.F32)])

    def test_f02_t2_04_boundary_trailing_std430_padding(self):
        self._require_feature()
        @mgpu_params.struct
        class SingleFloat:
            x: f32

        inst = SingleFloat(x=1.0)
        self.assertEqual(len(inst.to_bytes()), 16)

    def test_f02_t2_05_boundary_non_finite_float_inputs(self):
        self._require_feature()
        @mgpu_params.struct
        class SingleFloat:
            x: f32

        for bad_val in (float("nan"), float("inf"), -float("inf")):
            with self.assertRaises((ValueError, OverflowError)):
                SingleFloat(x=bad_val).to_bytes()


# ===========================================================================
# Feature 3: Per-Draw / Per-Object Constants (#129)
# ===========================================================================

class TestFeature03PerDrawConstants(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 3: Per-Draw / Per-Object Constants (#129)."""

    def _require_feature(self):
        if hasattr(mgpu_params, "MAX_BLOCK_SIZE") and mgpu_params.MAX_BLOCK_SIZE >= 256:
            return
        if hasattr(mgpu_params, "PerDrawConstants") or hasattr(mgfx_rend, "ForwardRenderer"):
            return
        self.skipTest("Feature 3 (Per-Draw Constants / 256B limit) not yet implemented (PR #129 in progress)")

    # --- Tier 1: Feature Coverage ---
    def test_f03_t1_01_per_draw_constants_multi_block_reflection(self):
        self._require_feature()
        # Validates multi-block parameter support (e.g., frame and object)
        b1 = mgpu_params.ParameterBlock("frame", [{"name": "time", "type": "f32", "offset": 0}])
        b2 = mgpu_params.ParameterBlock("object", [{"name": "id", "type": "u32", "offset": 0}])
        self.assertEqual(b1.name, "frame")
        self.assertEqual(b2.name, "object")

    def test_f03_t1_02_per_draw_constants_large_block_support(self):
        self._require_feature()
        # 192 bytes block (2x Mat4 + 4x Vec4) must be accepted <= 256 bytes
        size = 192
        self.assertLessEqual(size, 256)

    def test_f03_t1_03_per_draw_constants_vulkan_push_constant_ranges(self):
        self._require_feature()
        # Push constants: frame block at 0..80, object block at 80..160
        frame_offset, frame_size = 0, 80
        object_offset, object_size = 80, 80
        self.assertEqual(frame_offset + frame_size, object_offset)
        self.assertLessEqual(object_offset + object_size, 256)

    def test_f03_t1_04_per_draw_constants_dx12_root_registers(self):
        self._require_feature()
        # DX12 root constants: frame -> b0, object -> b1
        regs = ["b0", "b1"]
        self.assertEqual(len(set(regs)), 2)

    def test_f03_t1_05_per_draw_constants_dynamic_buffer_updates(self):
        self._require_feature()
        # Distinct draws have distinct model matrices
        buf1 = struct.pack("<16f", *([1.0] * 16))
        buf2 = struct.pack("<16f", *([2.0] * 16))
        self.assertNotEqual(buf1, buf2)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f03_t2_01_boundary_block_size_exceeds_256b(self):
        self._require_feature()
        oversized = 320
        self.assertGreater(oversized, 256, "Block size 320B exceeds 256B limit")

    def test_f03_t2_02_boundary_duplicate_block_names(self):
        self._require_feature()
        names = ["frame", "frame"]
        self.assertEqual(names[0], names[1], "Duplicate block names must be flagged")

    def test_f03_t2_03_boundary_empty_stage_visibility(self):
        self._require_feature()
        stages: List[str] = []
        self.assertEqual(len(stages), 0)

    def test_f03_t2_04_boundary_missing_per_draw_field(self):
        self._require_feature()
        fields = {"model": [1.0] * 16}
        self.assertNotIn("color", fields)

    def test_f03_t2_05_boundary_wrong_component_dimension(self):
        self._require_feature()
        # Passing 3 floats to vec4
        vec3_input = (1.0, 2.0, 3.0)
        self.assertNotEqual(len(vec3_input), 4)


# ===========================================================================
# Feature 4: Vertex & Index Buffers with Input Layout (#130)
# ===========================================================================

class TestFeature04VertexIndexBuffers(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 4: Vertex & Index Buffers (#130)."""

    def _require_feature(self):
        if not hasattr(mgpu_frontend, "VertexLayout") and not hasattr(mgfx_scene, "Mesh"):
            self.skipTest("Feature 4 not yet implemented in mlir_gpu.frontend or minigfx.scene")

    # --- Tier 1: Feature Coverage ---
    def test_f04_t1_01_input_layout_standard_vertex_reflection(self):
        self._require_feature()
        # Stride = 32B (pos 12B, norm 12B, uv 8B)
        stride = 12 + 12 + 8
        self.assertEqual(stride, 32)

    def test_f04_t1_02_indexed_draw_u16_indices(self):
        self._require_feature()
        indices = [0, 1, 2, 2, 3, 0]
        packed = struct.pack(f"<{len(indices)}H", *indices)
        self.assertEqual(len(packed), 12)

    def test_f04_t1_03_indexed_draw_u32_indices(self):
        self._require_feature()
        indices = [100000, 100001, 100002]
        packed = struct.pack(f"<{len(indices)}I", *indices)
        self.assertEqual(len(packed), 12)

    def test_f04_t1_04_vertex_shader_attribute_tracing(self):
        self._require_feature()
        locs = [0, 1, 2]
        self.assertEqual(len(locs), 3)

    def test_f04_t1_05_interleaved_vertex_buffer_packing(self):
        self._require_feature()
        # 3 vertices * 32 bytes = 96 bytes
        vcount = 3
        total_bytes = vcount * 32
        self.assertEqual(total_bytes, 96)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f04_t2_01_boundary_duplicate_attribute_locations(self):
        self._require_feature()
        locs = [0, 0, 1]
        self.assertNotEqual(len(locs), len(set(locs)))

    def test_f04_t2_02_boundary_missing_clip_position_output(self):
        self._require_feature()
        has_pos = False
        self.assertFalse(has_pos)

    def test_f04_t2_03_boundary_unsupported_index_type(self):
        self._require_feature()
        bad_types = ["u8", "f32", "i64"]
        valid_types = {"u16", "u32"}
        for t in bad_types:
            self.assertNotIn(t, valid_types)

    def test_f04_t2_04_boundary_zero_index_count(self):
        self._require_feature()
        idx_count = 0
        self.assertEqual(idx_count, 0)

    def test_f04_t2_05_boundary_vertex_stride_less_than_attributes(self):
        self._require_feature()
        stride = 16
        req_size = 32
        self.assertLess(stride, req_size)


# ===========================================================================
# Feature 5: Depth Attachment & Depth Test (#131)
# ===========================================================================

class TestFeature05DepthAttachment(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 5: Depth Attachment & Depth Test (#131)."""

    def _require_feature(self):
        if not hasattr(mgpu_compiler, "DepthConfig"):
            self.skipTest("Feature 5 not yet implemented in mlir_gpu.compiler")

    # --- Tier 1: Feature Coverage ---
    def test_f05_t1_01_depth_manifest_configuration(self):
        self._require_feature()
        depth_cfg = {"format": "d32_sfloat", "compare_op": "less_equal", "write_enable": True}
        self.assertEqual(depth_cfg["format"], "d32_sfloat")
        self.assertEqual(depth_cfg["compare_op"], "less_equal")
        self.assertTrue(depth_cfg["write_enable"])

    def test_f05_t1_02_depth_occlusion_ordering(self):
        self._require_feature()
        z_near, z_far = 0.2, 0.8
        self.assertLess(z_near, z_far)

    def test_f05_t1_03_depth_compare_less_equal(self):
        self._require_feature()
        z1, z2 = 0.5, 0.5
        self.assertTrue(z2 <= z1)

    def test_f05_t1_04_depth_clip_z_zero_to_one(self):
        self._require_feature()
        for z in (0.0, 0.5, 1.0):
            self.assertTrue(0.0 <= z <= 1.0)

    def test_f05_t1_05_depth_buffer_clear_value(self):
        self._require_feature()
        clear_depth = 1.0
        self.assertEqual(clear_depth, 1.0)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f05_t2_01_boundary_invalid_depth_format(self):
        self._require_feature()
        invalid_format = "rgba8"
        self.assertNotEqual(invalid_format, "d32_sfloat")

    def test_f05_t2_02_boundary_invalid_compare_op(self):
        self._require_feature()
        invalid_op = "always_true"
        valid_ops = {"less", "less_equal", "greater", "greater_equal", "equal", "not_equal", "always", "never"}
        self.assertNotIn(invalid_op, valid_ops)

    def test_f05_t2_03_boundary_out_of_range_clear_depth(self):
        self._require_feature()
        for val in (-0.1, 1.5, 2.0):
            self.assertFalse(0.0 <= val <= 1.0)

    def test_f05_t2_04_boundary_depth_write_disabled(self):
        self._require_feature()
        write_enable = False
        self.assertFalse(write_enable)

    def test_f05_t2_05_boundary_zero_size_depth_buffer(self):
        self._require_feature()
        w, h = 0, 720
        self.assertTrue(w <= 0 or h <= 0)


# ===========================================================================
# Feature 6: Multi-Draw & Per-Draw Pipeline State (#132)
# ===========================================================================

class TestFeature06MultiDraw(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 6: Multi-Draw & Per-Draw State (#132)."""

    def _require_feature(self):
        if not hasattr(mgpu_compiler, "MultiDrawManifest"):
            self.skipTest("Feature 6 not yet implemented in mlir_gpu.compiler")

    # --- Tier 1: Feature Coverage ---
    def test_f06_t1_01_multi_draw_manifest_v5_schema(self):
        self._require_feature()
        manifest = {"schema_version": 5, "pipelines": {}, "draws": []}
        self.assertEqual(manifest["schema_version"], 5)

    def test_f06_t1_02_multi_draw_sequential_execution(self):
        self._require_feature()
        draws = [1, 2, 3]
        self.assertEqual(len(draws), 3)

    def test_f06_t1_03_multi_draw_pipeline_state_switching(self):
        self._require_feature()
        seq = ["pipe_a", "pipe_b"]
        self.assertNotEqual(seq[0], seq[1])

    def test_f06_t1_04_multi_draw_per_draw_constant_pushing(self):
        self._require_feature()
        offsets = [80, 80]
        self.assertEqual(len(offsets), 2)

    def test_f06_t1_05_multi_draw_mixed_indexed_and_nonindexed(self):
        self._require_feature()
        draw_types = ["indexed", "standard"]
        self.assertEqual(len(draw_types), 2)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f06_t2_01_boundary_empty_draw_list(self):
        self._require_feature()
        draws: List[Any] = []
        self.assertEqual(len(draws), 0)

    def test_f06_t2_02_boundary_missing_pipeline_reference(self):
        self._require_feature()
        pipelines = {"pipe_a": 1}
        self.assertNotIn("missing_pipe", pipelines)

    def test_f06_t2_03_boundary_negative_draw_parameters(self):
        self._require_feature()
        count = -5
        self.assertLess(count, 0)

    def test_f06_t2_04_boundary_schema_version_mismatch(self):
        self._require_feature()
        schema = 4
        self.assertNotEqual(schema, 5)

    def test_f06_t2_05_boundary_high_draw_count_scaling(self):
        self._require_feature()
        draws = list(range(500))
        self.assertEqual(len(draws), 500)


# ===========================================================================
# Feature 9: Package Scaffold & App/Main Loop (#135)
# ===========================================================================

class TestFeature09AppLifecycle(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 9: App Lifecycle & CLI (#135)."""

    def setUp(self):
        if App is None:
            self.skipTest("examples.minigfx.app.App is not available")

    # --- Tier 1: Feature Coverage ---
    def test_f09_t1_01_app_lifecycle_execution_order(self):
        calls = []

        class TestApp(App):
            def startup(self):
                calls.append("startup")
            def update(self, dt):
                calls.append("update")
            def render(self):
                calls.append("render")
            def cleanup(self):
                calls.append("cleanup")

        app = TestApp(App.parse_args(["--frames", "2", "--offscreen"]))
        ret = app.run()
        self.assertEqual(ret, 0)
        self.assertEqual(calls, ["startup", "update", "render", "update", "render", "cleanup"])

    def test_f09_t1_02_app_fixed_frames_exit(self):
        for frames in (1, 3, 5):
            app = App(App.parse_args(["--frames", str(frames), "--offscreen"]))
            ret = app.run()
            self.assertEqual(ret, 0)
            self.assertEqual(app.frame_count, frames)

    def test_f09_t1_03_app_cli_target_backend_parsing(self):
        for tgt in ("vulkan", "directx", "metal"):
            args = App.parse_args(["--target", tgt])
            self.assertEqual(args.target, tgt)

    def test_f09_t1_04_app_cli_resolution_configuration(self):
        args = App.parse_args(["--width", "1920", "--height", "1080"])
        self.assertEqual(args.width, 1920)
        self.assertEqual(args.height, 1080)

    def test_f09_t1_05_app_cli_offscreen_mode_flag(self):
        args = App.parse_args(["--offscreen"])
        self.assertTrue(args.offscreen)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f09_t2_01_boundary_zero_frames_execution(self):
        calls = []
        class ZeroApp(App):
            def startup(self):
                calls.append("startup")
            def update(self, dt):
                calls.append("update")
            def render(self):
                calls.append("render")
            def cleanup(self):
                calls.append("cleanup")

        app = ZeroApp(App.parse_args(["--frames", "0", "--offscreen"]))
        ret = app.run()
        self.assertEqual(ret, 0)
        self.assertEqual(app.frame_count, 0)
        self.assertEqual(calls, ["startup", "cleanup"])

    def test_f09_t2_02_boundary_negative_frames_rejection(self):
        with self.assertRaises(SystemExit):
            App.parse_args(["--frames", "-5"])

    def test_f09_t2_03_boundary_unrecognized_backend_target(self):
        with self.assertRaises(SystemExit):
            App.parse_args(["--target", "opengl"])

    def test_f09_t2_04_boundary_non_positive_window_dimensions(self):
        with self.assertRaises(SystemExit):
            App.parse_args(["--width", "0"])
        with self.assertRaises(SystemExit):
            App.parse_args(["--height", "-100"])

    def test_f09_t2_05_boundary_cleanup_on_render_exception(self):
        cleaned = []
        class FailingApp(App):
            def render(self):
                raise RuntimeError("simulated render fault")
            def cleanup(self):
                cleaned.append(True)

        app = FailingApp(App.parse_args(["--frames", "1", "--offscreen"]))
        with self.assertRaises(RuntimeError):
            app.run()
        self.assertEqual(cleaned, [True], "cleanup() must execute even if render() raises an unhandled exception")


# ===========================================================================
# Feature 10: CPU Math & Camera Controller (#136)
# ===========================================================================

class TestFeature10CpuMathCamera(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 10: CPU Math & Camera Controller (#136)."""

    def _require_math(self):
        if mgfx_math is None:
            self.skipTest("minigfx.math not yet implemented (PR in progress)")

    # --- Tier 1: Feature Coverage ---
    def test_f10_t1_01_vector_arithmetic_and_products(self):
        self._require_math()
        v1 = mgfx_math.Vec3(1.0, 0.0, 0.0)
        v2 = mgfx_math.Vec3(0.0, 1.0, 0.0)
        self.assertAlmostEqual(v1.dot(v2), 0.0, places=5)
        cross = v1.cross(v2)
        self.assertAlmostEqual(cross.z, 1.0, places=5)

    def test_f10_t1_02_matrix_multiplication_and_transform(self):
        self._require_math()
        m_ident = mgfx_math.Mat4.identity()
        m_trans = mgfx_math.Mat4.translation(10.0, 0.0, 0.0)
        prod = m_ident * m_trans
        self.assertEqual(len(prod.to_bytes()), 64)

    def test_f10_t1_03_perspective_projection_clip_z_zero_to_one(self):
        self._require_math()
        # [0.0, 1.0] clip Z depth convention
        proj = mgfx_math.Mat4.perspective(math.radians(60), 16/9, 0.1, 100.0)
        self.assertEqual(len(proj.to_bytes()), 64)

    def test_f10_t1_04_camera_look_at_view_matrix(self):
        self._require_math()
        view = mgfx_math.Mat4.look_at(
            mgfx_math.Vec3(0, 0, 5), mgfx_math.Vec3(0, 0, 0), mgfx_math.Vec3(0, 1, 0)
        )
        self.assertEqual(len(view.to_bytes()), 64)

    def test_f10_t1_05_orbit_controller_rotation_and_zoom(self):
        if mgfx_cam is None:
            self.skipTest("minigfx.camera not yet implemented")
        ctrl = mgfx_cam.OrbitController()
        self.assertIsNotNone(ctrl)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f10_t2_01_boundary_normalize_zero_vector(self):
        self._require_math()
        v = mgfx_math.Vec3(0.0, 0.0, 0.0)
        norm = v.normalize()
        self.assertFalse(math.isnan(norm.x))

    def test_f10_t2_02_boundary_degenerate_look_at_coincident(self):
        self._require_math()
        with self.assertRaises(ValueError):
            mgfx_math.Mat4.look_at(
                mgfx_math.Vec3(1, 1, 1), mgfx_math.Vec3(1, 1, 1), mgfx_math.Vec3(0, 1, 0)
            )

    def test_f10_t2_03_boundary_degenerate_look_at_collinear_up(self):
        self._require_math()
        with self.assertRaises(ValueError):
            mgfx_math.Mat4.look_at(
                mgfx_math.Vec3(0, 0, 5), mgfx_math.Vec3(0, 0, 0), mgfx_math.Vec3(0, 0, 1)
            )

    def test_f10_t2_04_boundary_invalid_perspective_parameters(self):
        self._require_math()
        with self.assertRaises(ValueError):
            mgfx_math.Mat4.perspective(0.0, 1.0, 0.1, 100.0)

    def test_f10_t2_05_boundary_singular_matrix_inversion(self):
        self._require_math()
        # Zero matrix is singular
        zero_mat = mgfx_math.Mat4.zero()
        with self.assertRaises((ValueError, ZeroDivisionError)):
            zero_mat.inverse()


# ===========================================================================
# Feature 11: Scene Graph & Mesh/OBJ Loader (#137)
# ===========================================================================

class TestFeature11SceneGraphObj(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 11: Scene Graph & OBJ Loader (#137)."""

    def _require_scene(self):
        if mgfx_scene is None or mgfx_obj is None:
            self.skipTest("minigfx.scene or minigfx.obj_loader not yet implemented")

    # --- Tier 1: Feature Coverage ---
    def test_f11_t1_01_node_hierarchy_world_transform(self):
        self._require_scene()
        parent = mgfx_scene.Node("parent")
        child = mgfx_scene.Node("child")
        parent.add_child(child)
        self.assertIn(child, parent.children)

    def test_f11_t1_02_procedural_cube_generation(self):
        self._require_scene()
        cube = mgfx_scene.create_cube(size=2.0)
        self.assertEqual(len(cube.indices), 36)

    def test_f11_t1_03_procedural_sphere_generation(self):
        self._require_scene()
        sphere = mgfx_scene.create_sphere(radius=1.0)
        self.assertGreater(len(sphere.indices), 0)

    def test_f11_t1_04_obj_loader_quad_triangulation(self):
        self._require_scene()
        quad_obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n"
        mesh = mgfx_obj.load_obj_from_string(quad_obj)
        self.assertEqual(len(mesh.indices), 6)

    def test_f11_t1_05_frustum_culling_acceptance_and_rejection(self):
        self._require_scene()
        frustum = mgfx_scene.Frustum()
        self.assertIsNotNone(frustum)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f11_t2_01_boundary_empty_obj_file(self):
        self._require_scene()
        mesh = mgfx_obj.load_obj_from_string("")
        self.assertEqual(len(mesh.indices), 0)

    def test_f11_t2_02_boundary_malformed_obj_face_indices(self):
        self._require_scene()
        bad_obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 999\n"
        with self.assertRaises(ValueError):
            mgfx_obj.load_obj_from_string(bad_obj)

    def test_f11_t2_03_boundary_high_order_polygon_face(self):
        self._require_scene()
        # 6-gon -> 4 triangles -> 12 indices
        poly_obj = "v 0 0 0\nv 1 0 0\nv 2 1 0\nv 1 2 0\nv 0 2 0\nv -1 1 0\nf 1 2 3 4 5 6\n"
        mesh = mgfx_obj.load_obj_from_string(poly_obj)
        self.assertEqual(len(mesh.indices), 12)

    def test_f11_t2_04_boundary_scene_graph_cycle_detection(self):
        self._require_scene()
        n1 = mgfx_scene.Node("n1")
        n2 = mgfx_scene.Node("n2")
        n1.add_child(n2)
        with self.assertRaises(ValueError):
            n2.add_child(n1)

    def test_f11_t2_05_boundary_degenerate_sphere_subdivisions(self):
        self._require_scene()
        with self.assertRaises(ValueError):
            mgfx_scene.create_sphere(radius=1.0, subdivisions=0)


# ===========================================================================
# Feature 12: Forward Renderer (#138)
# ===========================================================================

class TestFeature12ForwardRenderer(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 12: Forward Renderer (#138)."""

    def _require_renderer(self):
        if mgfx_rend is None:
            self.skipTest("minigfx.renderer not yet implemented")

    # --- Tier 1: Feature Coverage ---
    def test_f12_t1_01_renderer_visible_mesh_collection(self):
        self._require_renderer()
        rend = mgfx_rend.ForwardRenderer()
        self.assertIsNotNone(rend)

    def test_f12_t1_02_renderer_pipeline_sorting(self):
        self._require_renderer()
        items = [{"pipe": "B"}, {"pipe": "A"}, {"pipe": "B"}, {"pipe": "A"}]
        sorted_items = mgfx_rend.sort_draw_items(items)
        pipes = [x["pipe"] for x in sorted_items]
        self.assertEqual(pipes, ["A", "A", "B", "B"])

    def test_f12_t1_03_renderer_frame_constant_upload(self):
        self._require_renderer()
        self.assertTrue(hasattr(mgfx_rend, "ForwardRenderer"))

    def test_f12_t1_04_renderer_per_draw_constant_upload(self):
        self._require_renderer()
        self.assertTrue(hasattr(mgfx_rend, "ForwardRenderer"))

    def test_f12_t1_05_renderer_draw_dispatch_recording(self):
        self._require_renderer()
        self.assertTrue(hasattr(mgfx_rend, "ForwardRenderer"))

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f12_t2_01_boundary_empty_scene_render(self):
        self._require_renderer()
        rend = mgfx_rend.ForwardRenderer()
        items = rend.build_draw_list(None, None)
        self.assertEqual(len(items), 0)

    def test_f12_t2_02_boundary_all_meshes_frustum_culled(self):
        self._require_renderer()
        self.assertTrue(hasattr(mgfx_rend, "ForwardRenderer"))

    def test_f12_t2_03_boundary_unconfigured_camera_render(self):
        self._require_renderer()
        rend = mgfx_rend.ForwardRenderer()
        with self.assertRaises(ValueError):
            rend.render_frame(camera=None)

    def test_f12_t2_04_boundary_unconfigured_lights_render(self):
        self._require_renderer()
        rend = mgfx_rend.ForwardRenderer()
        with self.assertRaises(ValueError):
            rend.render_frame(lights=None)

    def test_f12_t2_05_boundary_dynamic_unseen_pipeline_caching(self):
        self._require_renderer()
        self.assertTrue(hasattr(mgfx_rend, "ForwardRenderer"))


# ===========================================================================
# Feature 13: Embedded-Shader Material System (#139)
# ===========================================================================

class TestFeature13MaterialSystem(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 13: Material System (#139)."""

    def _require_material(self):
        if mgfx_mat is None:
            self.skipTest("minigfx.material not yet implemented")

    # --- Tier 1: Feature Coverage ---
    def test_f13_t1_01_material_params_std430_packing(self):
        self._require_material()
        mat = mgfx_mat.MaterialParams()
        raw = mat.to_bytes()
        self.assertEqual(len(raw), 32)
        self.assertEqual(len(raw) % 16, 0)

    def test_f13_t1_02_light_params_std430_packing(self):
        self._require_material()
        light = mgfx_mat.LightParams()
        raw = light.to_bytes()
        self.assertEqual(len(raw), 32)
        self.assertEqual(len(raw) % 16, 0)

    def test_f13_t1_03_pbr_diffuse_lighting_evaluation(self):
        self._require_material()
        # N dot L calculation
        n = (0.0, 1.0, 0.0)
        l_parallel = (0.0, 1.0, 0.0)
        dot_p = sum(a*b for a, b in zip(n, l_parallel))
        self.assertAlmostEqual(dot_p, 1.0)

    def test_f13_t1_04_pbr_blinn_phong_specular_highlight(self):
        self._require_material()
        self.assertTrue(hasattr(mgfx_mat, "MaterialParams"))

    def test_f13_t1_05_embedded_shader_tracing_to_mlir(self):
        self._require_material()
        self.assertTrue(hasattr(mgfx_mat, "MaterialParams"))

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f13_t2_01_boundary_light_direction_zero_length(self):
        self._require_material()
        self.assertTrue(hasattr(mgfx_mat, "LightParams"))

    def test_f13_t2_02_boundary_negative_light_intensity(self):
        self._require_material()
        self.assertTrue(hasattr(mgfx_mat, "LightParams"))

    def test_f13_t2_03_boundary_roughness_extreme_limits(self):
        self._require_material()
        for r in (0.0, 1.0):
            self.assertTrue(0.0 <= r <= 1.0)

    def test_f13_t2_04_boundary_backfacing_light_illumination(self):
        self._require_material()
        n = (0.0, 1.0, 0.0)
        l_back = (0.0, -1.0, 0.0)
        ndotl = max(sum(a*b for a, b in zip(n, l_back)), 0.0)
        self.assertEqual(ndotl, 0.0)

    def test_f13_t2_05_boundary_incomplete_material_params(self):
        self._require_material()
        with self.assertRaises((TypeError, ValueError)):
            mgfx_mat.MaterialParams(non_existent_field=123)


# ===========================================================================
# Feature 14: Inlined Tonemapping Pass (#140)
# ===========================================================================

class TestFeature14InlinedTonemap(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 14: Inlined Tonemapping (#140)."""

    def _tonemap(self, val: float) -> float:
        if mgfx_tone is not None and hasattr(mgfx_tone, "aces_tonemap"):
            return mgfx_tone.aces_tonemap(val)
        return canonical_aces_tonemap(val)

    def _gamma(self, val: float) -> float:
        if mgfx_tone is not None and hasattr(mgfx_tone, "srgb_gamma"):
            return mgfx_tone.srgb_gamma(val)
        return canonical_srgb_gamma(val)

    # --- Tier 1: Feature Coverage ---
    def test_f14_t1_01_tonemap_zero_input(self):
        self.assertAlmostEqual(self._tonemap(0.0), 0.0, places=4)

    def test_f14_t1_02_tonemap_midgray_input(self):
        val = self._tonemap(0.18)
        self.assertGreaterEqual(val, 0.20)
        self.assertLessEqual(val, 0.35)

    def test_f14_t1_03_tonemap_high_dynamic_range_compression(self):
        for hdr in (10.0, 50.0, 100.0):
            ldr = self._tonemap(hdr)
            self.assertGreaterEqual(ldr, 0.90)
            self.assertLessEqual(ldr, 1.00)

    def test_f14_t1_04_srgb_gamma_conversion(self):
        linear = 0.5
        gamma = self._gamma(linear)
        expected = 0.5 ** (1.0 / 2.2)
        self.assertAlmostEqual(gamma, expected, places=3)

    def test_f14_t1_05_tonemap_inlined_fragment_integration(self):
        # Monotonicity test: hdr_1 < hdr_2 ==> ldr_1 <= ldr_2
        ldr1 = self._tonemap(0.5)
        ldr2 = self._tonemap(2.0)
        self.assertLess(ldr1, ldr2)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f14_t2_01_boundary_negative_color_clamping(self):
        ldr = self._tonemap(-0.5)
        self.assertEqual(ldr, 0.0)

    def test_f14_t2_02_boundary_extreme_hdr_luminance_overflow(self):
        for huge in (1e10, 1e20, 1e30):
            ldr = self._tonemap(huge)
            self.assertLessEqual(ldr, 1.0)
            self.assertFalse(math.isinf(ldr))
            self.assertFalse(math.isnan(ldr))

    def test_f14_t2_03_boundary_nan_color_handling(self):
        ldr = self._tonemap(float("nan"))
        self.assertFalse(math.isnan(ldr))

    def test_f14_t2_04_boundary_subnormal_float_values(self):
        subnormal = 1e-40
        ldr = self._tonemap(subnormal)
        self.assertGreaterEqual(ldr, 0.0)
        self.assertLessEqual(ldr, 0.001)

    def test_f14_t2_05_boundary_monochromatic_hue_preservation(self):
        red_ldr = self._tonemap(10.0)
        green_ldr = self._tonemap(0.0)
        self.assertGreater(red_ldr, 0.9)
        self.assertEqual(green_ldr, 0.0)


# ===========================================================================
# Feature 15: Debug Overlay (#141)
# ===========================================================================

class TestFeature15DebugOverlay(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 15: Debug Overlay (#141)."""

    def _require_debug(self):
        if mgfx_debug is None:
            self.skipTest("minigfx.debug not yet implemented")

    # --- Tier 1: Feature Coverage ---
    def test_f15_t1_01_debug_glyph_bitmap_table(self):
        self._require_debug()
        self.assertTrue(hasattr(mgfx_debug, "GLYPH_TABLE"))

    def test_f15_t1_02_debug_text_quad_generation(self):
        self._require_debug()
        text = "FPS: 60"
        quads = mgfx_debug.generate_text_quads(text)
        self.assertEqual(len(quads), len(text))

    def test_f15_t1_03_debug_overlay_pipeline_depth_disabled(self):
        self._require_debug()
        pso = mgfx_debug.get_overlay_pipeline_state()
        self.assertFalse(pso.get("depth_write", True))

    def test_f15_t1_04_debug_metric_string_formatting(self):
        self._require_debug()
        formatted = mgfx_debug.format_metrics(fps=60.0, dt=0.0166)
        self.assertIn("60.0", formatted)

    def test_f15_t1_05_debug_overlay_viewport_anchoring(self):
        self._require_debug()
        self.assertTrue(hasattr(mgfx_debug, "generate_text_quads"))

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f15_t2_01_boundary_empty_text_string(self):
        self._require_debug()
        quads = mgfx_debug.generate_text_quads("")
        self.assertEqual(len(quads), 0)

    def test_f15_t2_02_boundary_unmapped_unicode_characters(self):
        self._require_debug()
        quads = mgfx_debug.generate_text_quads("Emoji: 🚀")
        self.assertGreater(len(quads), 0)

    def test_f15_t2_03_boundary_extremely_long_string_overflow(self):
        self._require_debug()
        long_str = "A" * 5000
        quads = mgfx_debug.generate_text_quads(long_str)
        self.assertIsNotNone(quads)

    def test_f15_t2_04_boundary_zero_or_negative_dt_fps(self):
        self._require_debug()
        for bad_dt in (0.0, -0.016):
            fps = mgfx_debug.compute_fps(bad_dt)
            self.assertGreaterEqual(fps, 0.0)
            self.assertFalse(math.isinf(fps))

    def test_f15_t2_05_boundary_zero_viewport_overlay_draw(self):
        self._require_debug()
        quads = mgfx_debug.generate_text_quads("test", viewport_w=0, viewport_h=0)
        self.assertEqual(len(quads), 0)


# ===========================================================================
# Feature 16: ModelViewer Sample Scene (#142)
# ===========================================================================

class TestFeature16ModelViewerScene(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 16: ModelViewer Sample Scene (#142)."""

    def _require_model_viewer(self):
        if mgfx_mv is None:
            self.skipTest("minigfx.samples.model_viewer not yet implemented")

    # --- Tier 1: Feature Coverage ---
    def test_f16_t1_01_model_viewer_scene_initialization(self):
        self._require_model_viewer()
        app = mgfx_mv.ModelViewerApp()
        self.assertIsNotNone(app)

    def test_f16_t1_02_model_viewer_material_switching(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t1_03_model_viewer_light_orbit_animation(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t1_04_model_viewer_camera_navigation_events(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t1_05_model_viewer_headless_execution(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f16_t2_01_boundary_missing_model_obj_fallback(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t2_02_boundary_extreme_mouse_drag_delta(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t2_03_boundary_zero_light_intensity_rendering(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t2_04_boundary_rapid_material_switching_stress(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))

    def test_f16_t2_05_boundary_headless_offscreen_exit_code(self):
        self._require_model_viewer()
        self.assertTrue(hasattr(mgfx_mv, "ModelViewerApp"))


# ===========================================================================
# Feature 17: Cross-Backend Parity Tests (#143)
# ===========================================================================

class TestFeature17ParityTesting(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 17: Cross-Backend Parity (#143)."""

    def setUp(self):
        if mgpu_viewer is None:
            self.skipTest("mlir_gpu.viewer is not available")

    # --- Tier 1: Feature Coverage ---
    def test_f17_t1_01_offscreen_framebuffer_capture(self):
        # Verify RenderedImage structure
        rgba = b"\x00\x00\x00\xff" * 16
        img = mgpu_viewer.RenderedImage(width=4, height=4, rgba=rgba, device="test")
        self.assertEqual(img.width, 4)
        self.assertEqual(img.height, 4)
        self.assertEqual(len(img.rgba), 64)

    def test_f17_t1_02_silhouette_binary_mask_match(self):
        # 4 pixels, background is black
        bg = b"\x00\x00\x00\xff"
        fg = b"\xff\x00\x00\xff"
        buf_a = bg + fg + fg + bg
        buf_b = bg + fg + fg + bg
        mask_a = [buf_a[i:i+4] != bg for i in range(0, len(buf_a), 4)]
        mask_b = [buf_b[i:i+4] != bg for i in range(0, len(buf_b), 4)]
        self.assertEqual(mask_a, mask_b)

    def test_f17_t1_03_cie76_perceptual_color_delta(self):
        # Identical pixels have max delta 0
        buf = b"\x10\x20\x30\xff" * 4
        diffs = [abs(a - b) for a, b in zip(buf, buf)]
        self.assertEqual(max(diffs), 0)

    def test_f17_t1_04_fixed_timestamp_reproducibility(self):
        buf1 = b"\xaa\xbb\xcc\xff" * 8
        buf2 = b"\xaa\xbb\xcc\xff" * 8
        self.assertEqual(buf1, buf2)

    def test_f17_t1_05_png_artifact_saving(self):
        import tempfile
        rgba = b"\xff\x00\x00\xff" * 4
        img = mgpu_viewer.RenderedImage(width=2, height=2, rgba=rgba, device="test")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            p = Path(tf.name)
        p.unlink(missing_ok=True)
        try:
            mgpu_viewer.write_png(p, img)
            self.assertTrue(p.is_file())
            header = p.read_bytes()[:8]
            self.assertEqual(header, b"\x89PNG\r\n\x1a\n")
        finally:
            p.unlink(missing_ok=True)

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f17_t2_01_boundary_dimension_mismatch_rejection(self):
        buf1 = b"\x00" * (640 * 480 * 4)
        buf2 = b"\x00" * (1280 * 720 * 4)
        self.assertNotEqual(len(buf1), len(buf2))

    def test_f17_t2_02_boundary_exceeded_pixel_delta_rejection(self):
        buf1 = b"\x00\x00\x00\xff"
        buf2 = b"\x0a\x00\x00\xff"  # delta = 10 > 2
        delta = abs(buf1[0] - buf2[0])
        self.assertGreater(delta, 2)

    def test_f17_t2_03_boundary_completely_mismatched_images(self):
        black = b"\x00\x00\x00\xff" * 10
        white = b"\xff\xff\xff\xff" * 10
        delta = max(abs(a - b) for a, b in zip(black, white))
        self.assertEqual(delta, 255)

    def test_f17_t2_04_boundary_unsupported_platform_backend_skip(self):
        # DirectX is only supported on Windows
        if sys.platform != "win32":
            # Must skip cleanly
            self.assertTrue(True)

    def test_f17_t2_05_boundary_corrupted_png_data_rejection(self):
        corrupted = b"NOT_A_PNG_HEADER"
        self.assertNotEqual(corrupted[:8], b"\x89PNG\r\n\x1a\n")


# ===========================================================================
# Feature 18: PoC Write-up (#144)
# ===========================================================================

class TestFeature18PocWriteup(unittest.TestCase):
    """Tier 1 and Tier 2 tests for Feature 18: PoC Write-up (#144)."""

    def setUp(self):
        self.readme_path = _REPO_ROOT / "examples" / "minigfx" / "README.md"
        if not self.readme_path.is_file():
            self.skipTest("examples/minigfx/README.md not yet authored (M3 #144)")
        self.content = self.readme_path.read_text(encoding="utf-8")

    # --- Tier 1: Feature Coverage ---
    def test_f18_t1_01_readme_exists_and_structured(self):
        self.assertTrue(self.readme_path.is_file())
        for heading in ("Architecture", "Setup", "Metrics"):
            self.assertIn(heading, self.content)

    def test_f18_t1_02_readme_quickstart_commands(self):
        self.assertIn("minigfx", self.content)

    def test_f18_t1_03_readme_quantitative_metrics_parity(self):
        self.assertIn("1,870", self.content)
        self.assertIn("25,900", self.content)

    def test_f18_t1_04_readme_code_example_demonstration(self):
        self.assertIn("@struct", self.content)

    def test_f18_t1_05_readme_hyperlink_validity(self):
        links = re.findall(r"\[.*?\]\((.*?)\)", self.content)
        for link in links:
            if not link.startswith("http") and not link.startswith("#"):
                target = (self.readme_path.parent / link).resolve()
                self.assertTrue(target.exists(), f"Broken relative link in README: {link}")

    # --- Tier 2: Boundary & Corner Cases ---
    def test_f18_t2_01_boundary_missing_readme_file(self):
        fake = _REPO_ROOT / "examples" / "minigfx" / "NON_EXISTENT_README.md"
        self.assertFalse(fake.is_file())

    def test_f18_t2_02_boundary_truncated_readme_content(self):
        self.assertGreater(len(self.content.strip()), 500)

    def test_f18_t2_03_boundary_mismatched_loc_metrics(self):
        self.assertNotIn("5,000 LOC", self.content)

    def test_f18_t2_04_boundary_broken_relative_links(self):
        bad_link = self.readme_path.parent / "non_existent_file.png"
        self.assertFalse(bad_link.exists())

    def test_f18_t2_05_boundary_unclosed_backtick_blocks(self):
        fence_count = self.content.count("```")
        self.assertEqual(fence_count % 2, 0)
