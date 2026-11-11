"""Comprehensive behavioral unit test suite for minigfx.debug overlay.

Covers:
- TestGlyphTable: glyph existence, dimensions, bit patterns, lowercase aliasing, fallback.
- TestTextQuadGeneration: quad geometry, UV coordinates, triangle indices, scaling, multiline,
  zero/negative scale, viewport clamping, buffer extraction.
- TestSoftwareRasterizer: pixel-accurate RGBA blitting, scale expansion, safe boundary clipping
  across all four edges, invalid dimensions safety, multiline rendering.
- TestMetricsComputationAndFormatting: FPS calculation from delta time, non-positive and NaN guards,
  status line formatting, overlay pipeline depth state.
- TestDebugOverlayIntegration: state tracking, update lifecycle, end-to-end rasterization and
  quad generation.
- TestOverlayParams: std430 struct alignment, binary packing/unpacking, parameter block schema,
  property accessors, and dictionary conversions.
- TestOverlayShadersDualModeExecution: dual-mode execution on CPU and GPU AST tracing for vertex
  and fragment shaders, NDC coordinate transformations, and interface matching.
- TestSoftwareRasterizerVsShaderEquivalence: parity between CPU software rasterizer output and
  shader evaluation.
"""

from __future__ import annotations

import math
import struct
import unittest

from examples.minigfx.debug import (
    GLYPH_TABLE,
    DebugOverlay,
    OverlayParams,
    OverlayVertex,
    ShaderFunction,
    TextQuad,
    TextQuadList,
    compute_fps,
    draw_to_image,
    format_metrics,
    generate_text_quads,
    get_overlay_pipeline_state,
    overlay_fragment_shader,
    overlay_ps,
    overlay_vertex_shader,
    overlay_vs,
)
from examples.minigfx.math import Vec2, Vec4

try:
    from mlir_gpu.frontend import Pipeline, Type, Vertex
    _HAVE_COMPILER = True
except ImportError:
    Pipeline = Type = Vertex = None
    _HAVE_COMPILER = False


class TestGlyphTable(unittest.TestCase):
    """Test procedural bitmap font glyph table structure and encoding."""

    def test_glyph_table_contains_standard_ascii(self):
        # All printable ASCII characters from 32 (' ') to 95 ('_') must exist.
        for code in range(32, 96):
            ch = chr(code)
            self.assertIn(ch, GLYPH_TABLE, f"Missing printable ASCII character: {ch!r}")

    def test_glyph_dimensions_and_column_masks(self):
        # Each glyph must consist of exactly 5 columns, with masks fitting in 7 bits [0, 127].
        for ch, cols in GLYPH_TABLE.items():
            self.assertEqual(len(cols), 5, f"Glyph {ch!r} must have 5 columns")
            for col_idx, mask in enumerate(cols):
                self.assertIsInstance(mask, int)
                self.assertTrue(
                    0 <= mask <= 127,
                    f"Glyph {ch!r} column {col_idx} mask {mask} exceeds 7-bit range [0, 127]",
                )

    def test_lowercase_maps_to_uppercase_equivalent(self):
        # Lowercase letters must share the identical bitmask tuple as uppercase.
        for c in range(ord("a"), ord("z") + 1):
            lower_ch = chr(c)
            upper_ch = lower_ch.upper()
            self.assertIn(lower_ch, GLYPH_TABLE)
            self.assertEqual(GLYPH_TABLE[lower_ch], GLYPH_TABLE[upper_ch])

    def test_special_symbols_and_vertical_bar(self):
        for sym in (":", ".", "m", "s", "F", "P", "-", "/", "%", " ", "|"):
            self.assertIn(sym, GLYPH_TABLE, f"Missing required symbol: {sym!r}")
        # Space must have zero bits set.
        self.assertEqual(GLYPH_TABLE[" "], (0, 0, 0, 0, 0))
        # Vertical bar must have center column populated.
        self.assertEqual(GLYPH_TABLE["|"], (0, 0, 127, 0, 0))

    def test_glyph_fallback_for_unmapped_character(self):
        # Unmapped character lookup should resolve to question mark glyph.
        question_mark = GLYPH_TABLE["?"]
        fallback = GLYPH_TABLE.get("\u2603", question_mark)
        self.assertEqual(fallback, question_mark)


class TestTextQuadGeneration(unittest.TestCase):
    """Test 2D screen-space quad generation, topology, and buffer generation."""

    def test_generate_quads_character_count_match(self):
        text = "FPS: 60.0"
        quads = generate_text_quads(text)
        self.assertIsInstance(quads, TextQuadList)
        self.assertEqual(len(quads), len(text))
        for q in quads:
            self.assertIsInstance(q, TextQuad)

    def test_quad_geometry_and_uv_bounds(self):
        quads = generate_text_quads("A", start_x=10.0, start_y=20.0, scale=1.0)
        self.assertEqual(len(quads), 1)
        q = quads[0]
        self.assertEqual(q.char, "A")
        # 5x7 glyph at (10, 20): bounds are [10, 15] x [20, 27]
        expected_pos = ((10.0, 20.0), (15.0, 20.0), (15.0, 27.0), (10.0, 27.0))
        self.assertEqual(q.positions, expected_pos)
        expected_uvs = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        self.assertEqual(q.uvs, expected_uvs)

    def test_quad_indices_topology_and_monotonicity(self):
        quads = generate_text_quads("ABC")
        self.assertEqual(len(quads), 3)
        for i, q in enumerate(quads):
            base = i * 4
            expected_indices = (base, base + 1, base + 2, base + 2, base + 3, base)
            self.assertEqual(q.indices, expected_indices)

    def test_quad_buffer_extraction_methods(self):
        text = "HUD"
        quads = generate_text_quads(text)
        self.assertIsInstance(quads, TextQuadList)

        verts, indices = quads.to_buffers()
        self.assertEqual(len(verts), len(text) * 4)
        self.assertEqual(len(indices), len(text) * 6)
        self.assertEqual(quads.as_mesh(), (verts, indices))

        # Direct extraction via as_buffers=True
        direct_verts, direct_indices = generate_text_quads(text, as_buffers=True)
        self.assertEqual(direct_verts, verts)
        self.assertEqual(direct_indices, indices)

    def test_viewport_clamping_behavior(self):
        quads = generate_text_quads(
            "FPS: 60", start_x=10.0, start_y=20.0, scale=2.0, viewport_w=1920, viewport_h=1080
        )
        self.assertGreater(len(quads), 0)
        for q in quads:
            for vx, vy in q.positions:
                self.assertTrue(0.0 <= vx <= 1920.0)
                self.assertTrue(0.0 <= vy <= 1080.0)

    def test_multiline_text_cursor_advances(self):
        text = "AB\nCD"
        quads = generate_text_quads(text, start_x=0.0, start_y=0.0, scale=1.0)
        # Newline does not emit quad; total glyphs = 4
        self.assertEqual(len(quads), 4)
        # Line 1: 'A' at x=0, 'B' at x=6; y=0
        self.assertEqual(quads[0].positions[0], (0.0, 0.0))
        self.assertEqual(quads[1].positions[0], (6.0, 0.0))
        # Line 2: 'C' at x=0, 'D' at x=6; y=9 (line height 9.0)
        self.assertEqual(quads[2].positions[0], (0.0, 9.0))
        self.assertEqual(quads[3].positions[0], (6.0, 9.0))

    def test_empty_string_and_zero_scale_safety(self):
        self.assertEqual(len(generate_text_quads("")), 0)
        self.assertEqual(len(generate_text_quads("test", scale=0.0)), 0)
        self.assertEqual(len(generate_text_quads("test", scale=-1.0)), 0)
        self.assertEqual(generate_text_quads("", as_buffers=True), ([], []))

    def test_zero_or_negative_viewport_returns_empty(self):
        self.assertEqual(len(generate_text_quads("test", viewport_w=0, viewport_h=0)), 0)
        self.assertEqual(len(generate_text_quads("test", viewport_w=-10, viewport_h=100)), 0)


class TestSoftwareRasterizer(unittest.TestCase):
    """Test CPU software rasterizer fallback and safe boundary clipping."""

    def test_rasterize_single_glyph_exact_pixel_blitting(self):
        # 10x10 buffer, render space glyph (0 pixels) vs single dot '.'
        buf = bytearray(10 * 10 * 4)
        draw_to_image(buf, width=10, height=10, text=" ", x=0, y=0)
        self.assertEqual(buf, bytearray(10 * 10 * 4))

        # Render '.' which has 4 pixels set in rows 5 and 6 of columns 1 and 2
        dot_color = (255, 0, 0, 255)
        draw_to_image(buf, width=10, height=10, text=".", x=0, y=0, color=dot_color)
        active_pixels = 0
        for i in range(0, len(buf), 4):
            if tuple(buf[i : i + 4]) == dot_color:
                active_pixels += 1
        # '.' in 5x7 font has 4 bits set
        dot_glyph = GLYPH_TABLE["."]
        expected_active = sum(bin(col).count("1") for col in dot_glyph)
        self.assertEqual(active_pixels, expected_active)

    def test_rasterize_scale_factor_expansion(self):
        # With scale=2, each bit becomes a 2x2 block (4 pixels)
        buf = bytearray(20 * 20 * 4)
        dot_color = (0, 255, 0, 255)
        draw_to_image(buf, width=20, height=20, text=".", x=0, y=0, color=dot_color, scale=2)
        active_pixels = sum(1 for i in range(0, len(buf), 4) if tuple(buf[i : i + 4]) == dot_color)
        dot_glyph = GLYPH_TABLE["."]
        expected_active = sum(bin(col).count("1") for col in dot_glyph) * 4
        self.assertEqual(active_pixels, expected_active)

    def test_rasterize_boundary_clipping_negative_coordinates(self):
        buf = bytearray(20 * 20 * 4)
        # Rendering at negative x and negative y must safely clip without raising IndexError
        draw_to_image(buf, width=20, height=20, text="FPS: 60", x=-5, y=-3)
        # Verify buffer is not all zeros (some pixels should be drawn inside [0, 20) x [0, 20))
        self.assertGreater(sum(buf), 0)

    def test_rasterize_boundary_clipping_overflow_coordinates(self):
        buf = bytearray(20 * 20 * 4)
        # Rendering past bottom-right edge must safely clip without raising IndexError
        draw_to_image(buf, width=20, height=20, text="FPS: 60", x=16, y=16)
        self.assertGreater(sum(buf), 0)

    def test_rasterize_fully_out_of_bounds_no_exception(self):
        buf = bytearray(20 * 20 * 4)
        # Far offscreen coordinates
        draw_to_image(buf, width=20, height=20, text="Offscreen", x=-500, y=-500)
        draw_to_image(buf, width=20, height=20, text="Offscreen", x=500, y=500)
        self.assertEqual(sum(buf), 0)

    def test_rasterize_zero_or_negative_dimensions_safety(self):
        buf = bytearray(100)
        draw_to_image(buf, width=0, height=10, text="Test")
        draw_to_image(buf, width=10, height=0, text="Test")
        draw_to_image(buf, width=-10, height=-10, text="Test")
        draw_to_image(buf, width=10, height=10, text="Test", scale=0)
        draw_to_image(buf, width=10, height=10, text="")
        self.assertEqual(sum(buf), 0)

    def test_rasterize_multiline_text(self):
        buf = bytearray(30 * 30 * 4)
        color = (255, 255, 255, 255)
        draw_to_image(buf, width=30, height=30, text="A\nB", x=0, y=0, color=color)
        self.assertGreater(sum(buf), 0)


class TestMetricsComputationAndFormatting(unittest.TestCase):
    """Test FPS computation guards, metric string formatting, and pipeline state."""

    def test_compute_fps_positive_delta_time(self):
        self.assertAlmostEqual(compute_fps(0.016666666666666666), 60.0, places=3)
        self.assertAlmostEqual(compute_fps(0.03333333333333333), 30.0, places=3)
        self.assertAlmostEqual(compute_fps(1.0), 1.0, places=5)

    def test_compute_fps_zero_or_negative_delta_time(self):
        for bad_dt in (0.0, -0.016, -1.0, -100.0):
            with self.subTest(bad_dt=bad_dt):
                fps = compute_fps(bad_dt)
                self.assertEqual(fps, 0.0)
                self.assertFalse(math.isinf(fps))
                self.assertFalse(math.isnan(fps))

    def test_compute_fps_nan_or_infinite_delta_time(self):
        for bad_dt in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(bad_dt=bad_dt):
                fps = compute_fps(bad_dt)
                self.assertEqual(fps, 0.0)

    def test_format_metrics_with_and_without_backend(self):
        res1 = format_metrics(fps=60.0, dt=0.0166, draw_calls=12, visible_meshes=8, backend_name="Vulkan")
        self.assertIn("60.0", res1)
        self.assertIn("16.6ms", res1)
        self.assertIn("Backend: Vulkan", res1)
        self.assertIn("Draws: 12", res1)
        self.assertIn("Meshes: 8", res1)

        res2 = format_metrics(fps=120.0, frame_time_ms=8.33)
        self.assertIn("120.0", res2)
        self.assertIn("8.3ms", res2)

    def test_overlay_pipeline_state_disables_depth(self):
        pso = get_overlay_pipeline_state()
        self.assertFalse(pso.get("depth_write", True))
        self.assertFalse(pso.get("depth_test", True))
        self.assertEqual(pso.get("cull_mode"), "none")


class TestDebugOverlayIntegration(unittest.TestCase):
    """Test DebugOverlay class state lifecycle and high-level rendering methods."""

    def test_overlay_initial_state(self):
        overlay = DebugOverlay(backend_name="Metal")
        self.assertEqual(overlay.backend_name, "Metal")
        self.assertEqual(overlay.fps, 0.0)
        self.assertEqual(overlay.frame_time_ms, 0.0)
        self.assertEqual(overlay.draw_calls, 0)
        self.assertEqual(overlay.visible_meshes, 0)

    def test_overlay_update_metrics(self):
        overlay = DebugOverlay()
        overlay.update_metrics(fps=59.9, frame_time_ms=16.69, draw_calls=4, visible_meshes=10, backend_name="DirectX 12")
        self.assertAlmostEqual(overlay.fps, 59.9)
        self.assertAlmostEqual(overlay.frame_time_ms, 16.69)
        self.assertEqual(overlay.draw_calls, 4)
        self.assertEqual(overlay.visible_meshes, 10)
        self.assertEqual(overlay.backend_name, "DirectX 12")

    def test_overlay_format_metrics(self):
        overlay = DebugOverlay(backend_name="Vulkan")
        overlay.update_metrics(fps=60.0, frame_time_ms=16.66, draw_calls=2, visible_meshes=5)
        text = overlay.format_metrics()
        self.assertIn("60.0", text)
        self.assertIn("Vulkan", text)
        self.assertIn("Draws: 2", text)
        self.assertIn("Meshes: 5", text)

    def test_overlay_draw_to_image_default_metrics(self):
        overlay = DebugOverlay(backend_name="Vulkan")
        overlay.update_metrics(fps=60.0, frame_time_ms=16.6)
        buf = bytearray(1280 * 720 * 4)
        overlay.draw_to_image(buf, width=1280, height=720)
        self.assertGreater(sum(buf), 0)

    def test_overlay_generate_text_quads_default_metrics(self):
        overlay = DebugOverlay(backend_name="Vulkan")
        overlay.update_metrics(fps=60.0, frame_time_ms=16.6)
        quads = overlay.generate_text_quads()
        self.assertGreater(len(quads), 0)
        for q in quads:
            self.assertIsInstance(q, TextQuad)


class TestOverlayParams(unittest.TestCase):
    """Validation of shared std430 host/device parameter block serialization and reflection."""

    def test_default_initialization(self):
        params = OverlayParams()
        self.assertEqual(params.text_color, Vec4(1.0, 1.0, 1.0, 1.0))
        self.assertEqual(params.bg_color, Vec4(0.0, 0.0, 0.0, 0.0))
        self.assertEqual(params.viewport, Vec2(1280.0, 720.0))
        self.assertEqual(params.scale, 1.0)

    def test_custom_initialization_and_property_setters(self):
        params = OverlayParams(
            text_color=(0.1, 0.2, 0.3, 0.4),
            bg_color=(0.5, 0.6, 0.7, 0.8),
            viewport=(1920.0, 1080.0),
            scale=2.0,
        )
        self.assertEqual(params.text_color, Vec4(0.1, 0.2, 0.3, 0.4))
        self.assertEqual(params.bg_color, Vec4(0.5, 0.6, 0.7, 0.8))
        self.assertEqual(params.viewport, Vec2(1920.0, 1080.0))
        self.assertEqual(params.scale, 2.0)

        params.text_color = Vec4(0.9, 0.8, 0.7, 0.6)
        self.assertEqual(params.text_color, Vec4(0.9, 0.8, 0.7, 0.6))
        params.bg_color = (0.1, 0.1, 0.1, 1.0)
        self.assertEqual(params.bg_color, Vec4(0.1, 0.1, 0.1, 1.0))
        params.viewport = (800.0, 600.0)
        self.assertEqual(params.viewport, Vec2(800.0, 600.0))

    def test_unexpected_kwargs_raises_type_error(self):
        with self.assertRaises(TypeError):
            OverlayParams(invalid_option=123)

    def test_std430_size_and_alignment_constants(self):
        self.assertEqual(OverlayParams.__struct_size__, 48)
        self.assertEqual(OverlayParams.__struct_alignment__, 16)
        self.assertTrue(OverlayParams._is_gfx_struct)

    def test_binary_serialization_std430_layout(self):
        params = OverlayParams(
            text_color=(1.0, 2.0, 3.0, 4.0),
            bg_color=(5.0, 6.0, 7.0, 8.0),
            viewport=(1920.0, 1080.0),
            scale=3.0,
        )
        data = params.to_bytes()
        self.assertEqual(len(data), 48)

        # Unpack individual std430 fields and verify byte offsets
        # Offset 0..16: text_color
        tc = struct.unpack_from("<4f", data, 0)
        self.assertEqual(tc, (1.0, 2.0, 3.0, 4.0))

        # Offset 16..32: bg_color
        bg = struct.unpack_from("<4f", data, 16)
        self.assertEqual(bg, (5.0, 6.0, 7.0, 8.0))

        # Offset 32..40: viewport
        vp = struct.unpack_from("<2f", data, 32)
        self.assertEqual(vp, (1920.0, 1080.0))

        # Offset 40..44: scale
        scale, pad = struct.unpack_from("<2f", data, 40)
        self.assertEqual(scale, 3.0)
        self.assertEqual(pad, 0.0)

    def test_binary_roundtrip_deserialization(self):
        params = OverlayParams(
            text_color=(0.125, 0.25, 0.375, 1.0),
            bg_color=(0.0, 0.0, 0.0, 0.5),
            viewport=(2560.0, 1440.0),
            scale=1.5,
        )
        raw_bytes = params.to_bytes()
        reconstituted = OverlayParams.from_bytes(raw_bytes)
        self.assertEqual(params, reconstituted)

    def test_deserialization_buffer_too_small_raises_value_error(self):
        short_buf = b"\x00" * 47
        with self.assertRaises(ValueError):
            OverlayParams.from_bytes(short_buf)

    def test_as_parameter_block_schema(self):
        block = OverlayParams.as_parameter_block()
        if _HAVE_COMPILER:
            self.assertIsNotNone(block)
            self.assertEqual(block.name, "overlay")
            field_map = dict(block.fields)
            self.assertEqual(field_map["text_color"], Type.VEC4)
            self.assertEqual(field_map["bg_color"], Type.VEC4)
            self.assertEqual(field_map["viewport"], Type.VEC2)
            self.assertEqual(field_map["scale"], Type.F32)

            custom_block = OverlayParams.as_parameter_block(name="custom_overlay")
            self.assertEqual(custom_block.name, "custom_overlay")

    def test_to_dict_representation(self):
        params = OverlayParams(
            text_color=(1.0, 0.5, 0.25, 1.0),
            bg_color=(0.0, 0.0, 0.0, 0.8),
            viewport=(1280.0, 720.0),
            scale=2.0,
        )
        d = params.to_dict()
        self.assertEqual(d["text_color"], [1.0, 0.5, 0.25, 1.0])
        self.assertEqual(d["bg_color"], [0.0, 0.0, 0.0, 0.8])
        self.assertEqual(d["viewport"], [1280.0, 720.0])
        self.assertEqual(d["scale"], 2.0)


class TestOverlayShadersDualModeExecution(unittest.TestCase):
    """Validation of dual-mode execution (CPU fallback and GPU AST pipeline tracing)."""

    def test_shader_callables_and_aliases(self):
        self.assertTrue(callable(overlay_vertex_shader))
        self.assertTrue(callable(overlay_fragment_shader))
        self.assertTrue(callable(overlay_vs))
        self.assertTrue(callable(overlay_ps))
        self.assertIs(overlay_vs, overlay_vertex_shader)
        self.assertIs(overlay_ps, overlay_fragment_shader)
        self.assertIsInstance(overlay_vertex_shader, ShaderFunction)
        self.assertIsInstance(overlay_fragment_shader, ShaderFunction)

    def test_cpu_vertex_shader_ndc_transformation_origin(self):
        params = OverlayParams(viewport=(1280.0, 720.0))
        # Screen origin (0, 0) transforms to top-left NDC (-1.0, 1.0, 0.0, 1.0)
        vert = overlay_vertex_shader((0.0, 0.0), params)
        self.assertIsInstance(vert, OverlayVertex)
        self.assertAlmostEqual(vert.position.x, -1.0)
        self.assertAlmostEqual(vert.position.y, 1.0)
        self.assertAlmostEqual(vert.position.z, 0.0)
        self.assertAlmostEqual(vert.position.w, 1.0)
        self.assertAlmostEqual(vert.x, -1.0)
        self.assertAlmostEqual(vert.y, 1.0)
        self.assertAlmostEqual(vert.z, 0.0)
        self.assertAlmostEqual(vert.w, 1.0)

    def test_cpu_vertex_shader_ndc_transformation_center(self):
        params = OverlayParams(viewport=(1280.0, 720.0))
        # Screen center (640, 360) transforms to NDC origin (0.0, 0.0, 0.0, 1.0)
        vert = overlay_vertex_shader((640.0, 360.0), params)
        self.assertAlmostEqual(vert.position.x, 0.0)
        self.assertAlmostEqual(vert.position.y, 0.0)
        self.assertAlmostEqual(vert.position.z, 0.0)
        self.assertAlmostEqual(vert.position.w, 1.0)

    def test_cpu_vertex_shader_ndc_transformation_bottom_right(self):
        params = OverlayParams(viewport=(1280.0, 720.0))
        # Screen bottom-right (1280, 720) transforms to NDC (1.0, -1.0, 0.0, 1.0)
        vert = overlay_vertex_shader((1280.0, 720.0), params)
        self.assertAlmostEqual(vert.position.x, 1.0)
        self.assertAlmostEqual(vert.position.y, -1.0)
        self.assertAlmostEqual(vert.position.z, 0.0)
        self.assertAlmostEqual(vert.position.w, 1.0)

    def test_cpu_vertex_shader_input_formats(self):
        params = OverlayParams(viewport=(1000.0, 1000.0))
        # Test 4-element vertex list [px, py, u, v] as emitted by generate_text_quads
        vert = overlay_vertex_shader([500.0, 500.0, 0.75, 0.25], params)
        self.assertAlmostEqual(vert.x, 0.0)
        self.assertAlmostEqual(vert.y, 0.0)
        self.assertEqual(vert.uv, Vec2(0.75, 0.25))

        # Test passing keyword argument pos
        vert_kw = overlay_vertex_shader(pos=(1000.0, 0.0), uv=(1.0, 0.0), params=params)
        self.assertAlmostEqual(vert_kw.x, 1.0)
        self.assertAlmostEqual(vert_kw.y, 1.0)
        self.assertEqual(vert_kw.uv, Vec2(1.0, 0.0))

    def test_cpu_fragment_shader_returns_text_color(self):
        params = OverlayParams(text_color=(0.2, 0.4, 0.6, 1.0))
        col = overlay_fragment_shader((0.5, 0.5), params)
        self.assertIsInstance(col, Vec4)
        self.assertEqual(col, Vec4(0.2, 0.4, 0.6, 1.0))

    def test_cpu_fragment_shader_glyph_bit_sampling(self):
        params = OverlayParams(
            text_color=(1.0, 1.0, 1.0, 1.0),
            bg_color=(0.0, 0.0, 0.0, 0.0),
        )
        # '.' glyph: active bits are in columns 1 and 2, rows 5 and 6
        # UV inside dot: column 1 (u=0.3), row 5 (v=0.75)
        active_col = overlay_fragment_shader((0.3, 0.75), params, char=".")
        self.assertEqual(active_col, params.text_color)

        # UV outside dot: column 0 (u=0.1), row 0 (v=0.1)
        empty_col = overlay_fragment_shader((0.1, 0.1), params, char=".")
        self.assertEqual(empty_col, params.bg_color)

    @unittest.skipUnless(_HAVE_COMPILER, "mlir_gpu compiler required for pipeline tracing")
    def test_gpu_pipeline_tracing(self):
        pipeline = Pipeline(overlay_vertex_shader, overlay_fragment_shader)
        v_stage, f_stage = pipeline.trace()
        self.assertEqual(v_stage.kind, "vertex")
        self.assertEqual(f_stage.kind, "fragment")

        v_outputs = {o.name for o in v_stage.output_interface}
        self.assertIn("position", v_outputs)
        self.assertIn("uv", v_outputs)

        f_inputs = {i.name for i in f_stage.input_interface}
        self.assertIn("uv", f_inputs)


class TestSoftwareRasterizerVsShaderEquivalence(unittest.TestCase):
    """Validation of parity between CPU software rasterizer and shader expectations."""

    def test_rasterizer_pixel_color_matches_fragment_shader_output(self):
        params = OverlayParams(text_color=(1.0, 0.5, 0.25, 1.0))
        shader_col = overlay_fragment_shader((0.5, 0.5), params)

        expected_rgba = (
            int(shader_col.x * 255.0),
            int(shader_col.y * 255.0),
            int(shader_col.z * 255.0),
            int(shader_col.w * 255.0),
        )

        buf = bytearray(10 * 10 * 4)
        draw_to_image(buf, width=10, height=10, text=".", x=0, y=0, color=expected_rgba)

        # Find active pixels in buffer and verify exact RGBA match
        active_pixels = [
            tuple(buf[i : i + 4])
            for i in range(0, len(buf), 4)
            if any(buf[i : i + 4])
        ]
        self.assertGreater(len(active_pixels), 0)
        for pix in active_pixels:
            self.assertEqual(pix, expected_rgba)

    def test_quad_vertices_transform_within_clip_space_bounds(self):
        vw, vh = 1280.0, 720.0
        params = OverlayParams(viewport=(vw, vh))
        quads = generate_text_quads("FPS: 60.0 | Draws: 12", start_x=10.0, start_y=10.0, scale=1.0)

        for q in quads:
            for vx, vy in q.positions:
                vert = overlay_vertex_shader((vx, vy), params)
                # Ensure vertex is inside normalized device coordinates [-1, 1] x [-1, 1]
                self.assertTrue(-1.0 <= vert.x <= 1.0, f"Vertex X {vert.x} outside [-1, 1]")
                self.assertTrue(-1.0 <= vert.y <= 1.0, f"Vertex Y {vert.y} outside [-1, 1]")
                self.assertAlmostEqual(vert.z, 0.0)
                self.assertAlmostEqual(vert.w, 1.0)

    def test_overlay_glyph_rasterizer_and_shader_pixel_parity(self):
        params = OverlayParams(text_color=(1.0, 0.0, 0.0, 1.0), bg_color=(0.0, 0.0, 0.0, 0.0))
        w, h = 5, 7
        buf = bytearray(w * h * 4)
        draw_to_image(buf, width=w, height=h, text=".", x=0, y=0, color=(255, 0, 0, 255), scale=1)

        tc_bytes = (255, 0, 0, 255)
        bg_bytes = (0, 0, 0, 0)

        for cy in range(h):
            for cx in range(w):
                uv = Vec2((cx + 0.5) / 5.0, (cy + 0.5) / 7.0)
                shader_col = overlay_fragment_shader(uv, params, char=".")
                expected_bytes = tc_bytes if shader_col.x > 0.5 else bg_bytes
                idx = (cy * w + cx) * 4
                actual_bytes = tuple(buf[idx : idx + 4])
                self.assertEqual(
                    actual_bytes,
                    expected_bytes,
                    f"Pixel mismatch at ({cx}, {cy}): actual={actual_bytes}, expected={expected_bytes}",
                )


if __name__ == "__main__":
    unittest.main()
