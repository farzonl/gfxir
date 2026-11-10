"""Unit test suite for minigfx.material: PBR-lite lighting, shared structs, and shaders.

Covers:
- TestMaterialParametersLayout: std430 32B packing, offsets, base_color alias, validation
- TestLightParametersLayout: std430 32B packing, offsets, zero direction & negative intensity checks
- TestObjectAndCameraParametersLayout: std430 80B packing and roundtrip
- TestDiffuseLightingEvaluation: Lambertian N·L, backfacing clamping, color tinting, metallic factor
- TestSpecularHighlightEvaluation: Blinn-Phong (N·H)^alpha, roughness falloff, visibility guards
- TestPbrLightingComposite: Ambient, metallic diffuse cancellation, energy conservation bounds
- TestTonemappingIntegration: Inlined ACES tonemapper + sRGB gamma, extreme HDR stability
- TestShaderTracingAndDualModeExecution: Callability, CPU execution, and AST pipeline tracing
"""

from __future__ import annotations

import math
import struct
import sys
import unittest
from pathlib import Path

# Hermetic path configuration
_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_DIR = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_DIR):
    _str_p = str(_p)
    if _str_p not in sys.path:
        sys.path.insert(0, _str_p)

from examples.minigfx.math import Mat4, Vec3, Vec4
from examples.minigfx.material import (
    CameraParams,
    LightParams,
    MaterialParams,
    ObjectParams,
    ShaderFunction,
    default_ps,
    default_vs,
    evaluate_diffuse_lighting,
    evaluate_lighting,
    evaluate_specular,
    material_fragment_shader,
    material_vertex_shader,
    pbr_fragment,
    pbr_vertex,
)
from examples.minigfx.tonemap import aces_tonemap, tonemap_and_gamma

try:
    from mlir_gpu.frontend import Pipeline
    _HAVE_COMPILER = True
except ImportError:
    _HAVE_COMPILER = False


class TestMaterialParametersLayout(unittest.TestCase):
    """Validation of MaterialParams std430 binary packing, layout, and constraints."""

    def test_default_construction(self):
        mat = MaterialParams()
        raw = mat.to_bytes()
        self.assertEqual(len(raw), 32)
        self.assertEqual(len(raw) % 16, 0)
        self.assertAlmostEqual(mat.roughness, 0.5)
        self.assertAlmostEqual(mat.metallic, 0.0)
        self.assertAlmostEqual(mat.ambient, 0.0)
        self.assertEqual(mat.albedo, Vec4(1.0, 1.0, 1.0, 1.0))
        self.assertEqual(mat.base_color, Vec4(1.0, 1.0, 1.0, 1.0))

    def test_std430_field_offsets(self):
        mat = MaterialParams(albedo=(1.0, 0.5, 0.25, 0.8), roughness=0.3, metallic=0.7, ambient=0.15)
        raw = mat.to_bytes()
        self.assertEqual(len(raw), 32)
        r, g, b, a = struct.unpack_from("<4f", raw, 0)
        self.assertAlmostEqual(r, 1.0, places=5)
        self.assertAlmostEqual(g, 0.5, places=5)
        self.assertAlmostEqual(b, 0.25, places=5)
        self.assertAlmostEqual(a, 0.8, places=5)
        ro, me, am, pad = struct.unpack_from("<4f", raw, 16)
        self.assertAlmostEqual(ro, 0.3, places=5)
        self.assertAlmostEqual(me, 0.7, places=5)
        self.assertAlmostEqual(am, 0.15, places=5)
        self.assertAlmostEqual(pad, 0.0, places=5)

    def test_base_color_alias(self):
        mat = MaterialParams(base_color=(0.2, 0.4, 0.6, 1.0), roughness=0.4, metallic=0.2)
        self.assertEqual(mat.base_color, Vec4(0.2, 0.4, 0.6, 1.0))
        self.assertEqual(mat.albedo, Vec4(0.2, 0.4, 0.6, 1.0))
        mat.base_color = (0.9, 0.8, 0.7, 1.0)
        self.assertEqual(mat.albedo, Vec4(0.9, 0.8, 0.7, 1.0))
        mat.albedo = (0.1, 0.2, 0.3, 1.0)
        self.assertEqual(mat.base_color, Vec4(0.1, 0.2, 0.3, 1.0))

    def test_from_bytes_roundtrip(self):
        orig = MaterialParams(albedo=(0.6, 0.3, 0.1, 1.0), roughness=0.25, metallic=0.85, ambient=0.05)
        restored = MaterialParams.from_bytes(orig.to_bytes())
        self.assertEqual(orig, restored)
        self.assertAlmostEqual(restored.roughness, 0.25, places=5)
        self.assertAlmostEqual(restored.metallic, 0.85, places=5)

    def test_buffer_underflow_rejected(self):
        with self.assertRaises(ValueError):
            MaterialParams.from_bytes(b"\x00" * 31)

    def test_roughness_boundary_validation(self):
        MaterialParams(roughness=0.0)
        MaterialParams(roughness=1.0)
        for r in (-0.1, -1.0, 1.01, 2.5):
            with self.assertRaises(ValueError):
                MaterialParams(roughness=r)

    def test_metallic_boundary_validation(self):
        MaterialParams(metallic=0.0)
        MaterialParams(metallic=1.0)
        for m in (-0.05, 1.1):
            with self.assertRaises(ValueError):
                MaterialParams(metallic=m)

    def test_negative_ambient_rejected(self):
        with self.assertRaises(ValueError):
            MaterialParams(ambient=-0.1)

    def test_material_params_non_finite_ambient_rejection(self):
        """Verify that NaN and infinite ambient values raise ValueError."""
        for bad_ambient in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"Accepted non-finite ambient: {bad_ambient}"):
                MaterialParams(ambient=bad_ambient)

    def test_material_params_non_finite_roughness_and_metallic_rejection(self):
        """Verify that NaN and infinite roughness and metallic values raise ValueError."""
        for bad_val in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"Accepted non-finite roughness: {bad_val}"):
                MaterialParams(roughness=bad_val)
            with self.assertRaises(ValueError, msg=f"Accepted non-finite metallic: {bad_val}"):
                MaterialParams(metallic=bad_val)

    def test_unexpected_keyword_arguments_rejected(self):
        with self.assertRaises(TypeError):
            MaterialParams(unknown_field=42)

    def test_parameter_block_reflection(self):
        block = MaterialParams.as_parameter_block("mat")
        if block is not None:
            self.assertEqual(block.name, "mat")
            field_names = [f[0] for f in block.fields]
            self.assertEqual(field_names, ["albedo", "roughness", "metallic", "ambient"])


class TestLightParametersLayout(unittest.TestCase):
    """Validation of LightParams std430 binary packing, layout, and constraints."""

    def test_default_construction(self):
        light = LightParams()
        raw = light.to_bytes()
        self.assertEqual(len(raw), 32)
        self.assertEqual(len(raw) % 16, 0)
        self.assertAlmostEqual(light.intensity, 1.0)
        self.assertAlmostEqual(light.ambient, 0.1)
        self.assertAlmostEqual(light.direction.length(), 1.0)

    def test_std430_field_offsets_and_packing_holes(self):
        light = LightParams(direction=(0.0, -1.0, 0.0), intensity=2.5, color=(1.0, 0.8, 0.6), ambient=0.2)
        raw = light.to_bytes()
        self.assertEqual(len(raw), 32)
        dx, dy, dz = struct.unpack_from("<3f", raw, 0)
        self.assertAlmostEqual(dy, -1.0, places=5)
        intensity = struct.unpack_from("<f", raw, 12)[0]
        self.assertAlmostEqual(intensity, 2.5, places=5)
        cx, cy, cz = struct.unpack_from("<3f", raw, 16)
        self.assertAlmostEqual(cx, 1.0, places=5)
        self.assertAlmostEqual(cy, 0.8, places=5)
        self.assertAlmostEqual(cz, 0.6, places=5)
        ambient = struct.unpack_from("<f", raw, 28)[0]
        self.assertAlmostEqual(ambient, 0.2, places=5)

    def test_direction_normalization(self):
        light = LightParams(direction=(0.0, -3.0, 4.0))
        self.assertAlmostEqual(light.direction.length(), 1.0)
        self.assertAlmostEqual(light.direction.y, -0.6, places=5)
        self.assertAlmostEqual(light.direction.z, 0.8, places=5)

    def test_zero_length_direction_rejected(self):
        with self.assertRaises(ValueError):
            LightParams(direction=(0.0, 0.0, 0.0))

    def test_negative_intensity_rejected(self):
        with self.assertRaises(ValueError):
            LightParams(intensity=-0.5)

    def test_negative_ambient_rejected(self):
        with self.assertRaises(ValueError):
            LightParams(ambient=-0.01)

    def test_from_bytes_roundtrip(self):
        orig = LightParams(direction=(1.0, 2.0, 3.0), intensity=4.0, color=(0.9, 0.8, 0.7), ambient=0.3)
        restored = LightParams.from_bytes(orig.to_bytes())
        self.assertEqual(orig, restored)

    def test_buffer_underflow_rejected(self):
        with self.assertRaises(ValueError):
            LightParams.from_bytes(b"\x00" * 31)

    def test_unexpected_keyword_arguments_rejected(self):
        with self.assertRaises(TypeError):
            LightParams(invalid_arg=1)


class TestObjectAndCameraParametersLayout(unittest.TestCase):
    """Validation of ObjectParams and CameraParams std430 80B layouts."""

    def test_object_params_packing_and_roundtrip(self):
        m = Mat4.translation(1.0, 2.0, 3.0)
        c = Vec4(0.1, 0.2, 0.3, 1.0)
        obj = ObjectParams(model=m, color=c)
        raw = obj.to_bytes()
        self.assertEqual(len(raw), 80)
        self.assertEqual(len(raw) % 16, 0)
        restored = ObjectParams.from_bytes(raw)
        self.assertEqual(obj, restored)

    def test_camera_params_packing_and_roundtrip(self):
        vp = Mat4.look_at(Vec3(0, 0, 5), Vec3(0, 0, 0), Vec3(0, 1, 0))
        eye = Vec3(0, 0, 5)
        cam = CameraParams(view_proj=vp, eye_pos=eye, time=3.5)
        raw = cam.to_bytes()
        self.assertEqual(len(raw), 80)
        self.assertEqual(len(raw) % 16, 0)
        restored = CameraParams.from_bytes(raw)
        self.assertEqual(cam, restored)

    def test_camera_params_float32_time_roundtrip_precision(self):
        """Verify CameraParams roundtrip equality for non-dyadic float timestamps affected by float32 truncation."""
        vp = Mat4.look_at(Vec3(1.0, 2.0, 3.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 1.0, 0.0))
        eye = Vec3(1.0, 2.0, 3.0)
        for t in (0.1, 0.3, 0.7, 1.1):
            cam = CameraParams(view_proj=vp, eye_pos=eye, time=t)
            raw = cam.to_bytes()
            self.assertEqual(len(raw), 80)
            self.assertEqual(len(raw) % 16, 0)
            restored = CameraParams.from_bytes(raw)
            self.assertEqual(
                cam,
                restored,
                f"CameraParams equality failed for non-dyadic time={t}: {cam.time} != {restored.time}",
            )
            self.assertAlmostEqual(restored.time, t, places=5)

    def test_camera_params_non_finite_time_rejection(self):
        """Verify that NaN and infinite camera time values raise ValueError."""
        for bad_time in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError, msg=f"Accepted non-finite camera time: {bad_time}"):
                CameraParams(time=bad_time)


class TestDiffuseLightingEvaluation(unittest.TestCase):
    """Validation of pure Python Lambertian diffuse lighting."""

    def test_parallel_light_full_illumination(self):
        n = (0.0, 1.0, 0.0)
        l = (0.0, 1.0, 0.0)
        diffuse = evaluate_diffuse_lighting(normal=n, light_dir=l, albedo=(0.8, 0.5, 0.2))
        self.assertAlmostEqual(diffuse[0], 0.8, places=5)
        self.assertAlmostEqual(diffuse[1], 0.5, places=5)
        self.assertAlmostEqual(diffuse[2], 0.2, places=5)

    def test_angled_light_falloff(self):
        n = (0.0, 1.0, 0.0)
        l = (1.0, 1.0, 0.0)  # 45 degrees
        diffuse = evaluate_diffuse_lighting(normal=n, light_dir=l, albedo=(1.0, 1.0, 1.0))
        expected = math.cos(math.pi / 4.0)
        self.assertAlmostEqual(diffuse[0], expected, places=4)

    def test_grazing_angle_illumination(self):
        n = (0.0, 1.0, 0.0)
        l = (1.0, 0.0, 0.0)  # 90 degrees
        diffuse = evaluate_diffuse_lighting(normal=n, light_dir=l, albedo=(1.0, 1.0, 1.0))
        self.assertAlmostEqual(diffuse[0], 0.0, places=5)

    def test_backfacing_light_zero_contribution(self):
        n = (0.0, 1.0, 0.0)
        l_back = (0.0, -1.0, 0.0)
        diffuse = evaluate_diffuse_lighting(normal=n, light_dir=l_back, albedo=(1.0, 1.0, 1.0))
        self.assertEqual(diffuse, (0.0, 0.0, 0.0))

    def test_unnormalized_vectors_safe(self):
        n = (0.0, 5.0, 0.0)
        l = (0.0, 10.0, 0.0)
        diffuse = evaluate_diffuse_lighting(normal=n, light_dir=l, albedo=(1.0, 1.0, 1.0))
        self.assertAlmostEqual(diffuse[0], 1.0, places=5)

    def test_metallic_diffuse_cancellation(self):
        n = (0.0, 1.0, 0.0)
        l = (0.0, 1.0, 0.0)
        alb = (0.8, 0.4, 0.2)
        diff_dielectric = evaluate_diffuse_lighting(n, l, alb, metallic=0.0)
        diff_conductor = evaluate_diffuse_lighting(n, l, alb, metallic=1.0)
        self.assertAlmostEqual(diff_dielectric[0], 0.8, places=5)
        self.assertAlmostEqual(diff_conductor[0], 0.0, places=5)
        self.assertEqual(diff_conductor, (0.0, 0.0, 0.0))


class TestSpecularHighlightEvaluation(unittest.TestCase):
    """Validation of pure Python Blinn-Phong specular highlight evaluation."""

    def test_aligned_specular_strong_highlight(self):
        n = (0.0, 1.0, 0.0)
        v = (0.0, 1.0, 0.0)
        l = (0.0, 1.0, 0.0)
        spec = evaluate_specular(normal=n, view_dir=v, light_dir=l, roughness=0.2)
        self.assertGreater(spec, 0.5)
        self.assertAlmostEqual(spec, 1.0, places=5)

    def test_roughness_exponent_falloff(self):
        n = (0.0, 1.0, 0.0)
        l = (0.1, 0.99, 0.0)
        v = (0.0, 1.0, 0.0)
        spec_smooth = evaluate_specular(normal=n, view_dir=v, light_dir=l, roughness=0.1)
        spec_rough = evaluate_specular(normal=n, view_dir=v, light_dir=l, roughness=0.8)
        self.assertTrue(0.0 <= spec_smooth <= 1.0)
        self.assertTrue(0.0 <= spec_rough <= 1.0)

    def test_backfacing_light_zero_specular(self):
        n = (0.0, 1.0, 0.0)
        v = (0.0, 1.0, 0.0)
        l_back = (0.0, -1.0, 0.0)
        spec = evaluate_specular(normal=n, view_dir=v, light_dir=l_back, roughness=0.2)
        self.assertEqual(spec, 0.0)

    def test_backfacing_view_zero_specular(self):
        n = (0.0, 1.0, 0.0)
        v_back = (0.0, -1.0, 0.0)
        l = (0.0, 1.0, 0.0)
        spec = evaluate_specular(normal=n, view_dir=v_back, light_dir=l, roughness=0.2)
        self.assertEqual(spec, 0.0)

    def test_collinear_opposing_view_and_light(self):
        n = (0.0, 1.0, 0.0)
        v = (0.0, 0.0, 1.0)
        l = (0.0, 0.0, -1.0)
        spec = evaluate_specular(normal=n, view_dir=v, light_dir=l, roughness=0.5)
        self.assertEqual(spec, 0.0)

    def test_extreme_roughness_limits(self):
        n = (0.0, 1.0, 0.0)
        v = (0.0, 1.0, 0.0)
        l = (0.0, 1.0, 0.0)
        spec_mirror = evaluate_specular(normal=n, view_dir=v, light_dir=l, roughness=0.0)
        spec_matte = evaluate_specular(normal=n, view_dir=v, light_dir=l, roughness=1.0)
        self.assertAlmostEqual(spec_mirror, 1.0, places=5)
        self.assertAlmostEqual(spec_matte, 1.0, places=5)


class TestPbrLightingComposite(unittest.TestCase):
    """Validation of composite lighting (ambient + diffuse + specular)."""

    def test_ambient_only_lighting(self):
        light = LightParams(direction=(0.0, -1.0, 0.0), intensity=0.0, ambient=0.2)
        mat = MaterialParams(albedo=(0.5, 0.5, 0.5, 1.0), ambient=0.3)
        res = evaluate_lighting(normal=(0, 1, 0), view_dir=(0, 1, 0), light=light, material=mat)
        expected_amb = 0.5 * (0.2 + 0.3)
        self.assertAlmostEqual(res[0], expected_amb, places=5)

    def test_dielectric_vs_conductor_energy_conservation(self):
        light = LightParams(direction=(0.0, 1.0, 0.0), intensity=1.0, ambient=0.0)
        dielectric = MaterialParams(albedo=(0.8, 0.2, 0.2, 1.0), roughness=0.5, metallic=0.0, ambient=0.0)
        conductor = MaterialParams(albedo=(0.8, 0.2, 0.2, 1.0), roughness=0.5, metallic=1.0, ambient=0.0)

        res_d = evaluate_lighting(normal=(0, 1, 0), view_dir=(0, 1, 0), light=light, material=dielectric)
        res_c = evaluate_lighting(normal=(0, 1, 0), view_dir=(0, 1, 0), light=light, material=conductor)

        # Dielectric has diffuse + specular; conductor has zero diffuse
        diff_d = evaluate_diffuse_lighting((0, 1, 0), (0, 1, 0), dielectric.albedo, dielectric.metallic)
        diff_c = evaluate_diffuse_lighting((0, 1, 0), (0, 1, 0), conductor.albedo, conductor.metallic)
        self.assertAlmostEqual(diff_d[0], 0.8, places=5)
        self.assertAlmostEqual(diff_c[0], 0.0, places=5)

        for val in res_d + res_c:
            self.assertTrue(math.isfinite(val))


class TestTonemappingIntegration(unittest.TestCase):
    """Validation of tonemapper integration with lighting evaluation."""

    def test_extreme_hdr_luminance_bounded(self):
        res = tonemap_and_gamma(Vec3(500.0, 1000.0, 10000.0))
        self.assertTrue(0.0 <= res.x <= 1.0)
        self.assertTrue(0.0 <= res.y <= 1.0)
        self.assertTrue(0.0 <= res.z <= 1.0)

    def test_zero_hdr_maps_to_black(self):
        res = tonemap_and_gamma(Vec3(0.0, 0.0, 0.0))
        self.assertAlmostEqual(res.x, 0.0, places=5)
        self.assertAlmostEqual(res.y, 0.0, places=5)
        self.assertAlmostEqual(res.z, 0.0, places=5)

    def test_monotonicity(self):
        v1 = aces_tonemap(0.2)
        v2 = aces_tonemap(0.8)
        v3 = aces_tonemap(2.5)
        self.assertLess(v1, v2)
        self.assertLess(v2, v3)


class TestShaderTracingAndDualModeExecution(unittest.TestCase):
    """Validation of dual-mode execution (CPU evaluation and GPU pipeline tracing)."""

    def test_shaders_are_callable(self):
        self.assertTrue(callable(material_vertex_shader))
        self.assertTrue(callable(material_fragment_shader))
        self.assertTrue(callable(pbr_vertex))
        self.assertTrue(callable(pbr_fragment))
        self.assertTrue(callable(default_vs))
        self.assertTrue(callable(default_ps))

    def test_cpu_vertex_execution(self):
        cam = CameraParams(view_proj=Mat4.identity())
        obj = ObjectParams(model=Mat4.identity())
        clip_pos = material_vertex_shader(pos=(1.0, 2.0, 3.0, 1.0), cam=cam, obj=obj)
        self.assertIsInstance(clip_pos, Vec4)
        self.assertAlmostEqual(clip_pos.x, 1.0, places=5)
        self.assertAlmostEqual(clip_pos.y, 2.0, places=5)
        self.assertAlmostEqual(clip_pos.z, 3.0, places=5)

    def test_cpu_fragment_execution(self):
        cam = CameraParams(eye_pos=(0, 0, 5))
        mat = MaterialParams(albedo=(0.8, 0.2, 0.2, 1.0), roughness=0.5, metallic=0.0)
        light = LightParams(direction=(0, 1, 0), intensity=1.0)
        col = material_fragment_shader(normal=(0, 1, 0), world_pos=(0, 0, 0), cam=cam, mat=mat, light=light)
        self.assertIsInstance(col, Vec4)
        self.assertTrue(0.0 <= col.x <= 1.0)
        self.assertTrue(0.0 <= col.y <= 1.0)
        self.assertTrue(0.0 <= col.z <= 1.0)
        self.assertAlmostEqual(col.w, 1.0)

    @unittest.skipUnless(_HAVE_COMPILER, "mlir_gpu compiler required for pipeline tracing")
    def test_gpu_pipeline_tracing(self):
        pipe = Pipeline(material_vertex_shader, material_fragment_shader)
        v_st, f_st = pipe.trace()
        self.assertEqual(v_st.kind, "vertex")
        self.assertEqual(f_st.kind, "fragment")
        v_outputs = {o.name for o in v_st.output_interface}
        self.assertIn("position", v_outputs)
        self.assertIn("normal", v_outputs)
        self.assertIn("world_pos", v_outputs)
        self.assertIn("uv", v_outputs)
        f_inputs = {i.name for i in f_st.input_interface}
        self.assertIn("normal", f_inputs)
        self.assertIn("world_pos", f_inputs)
        self.assertIn("uv", f_inputs)


if __name__ == "__main__":
    unittest.main()
