"""Unit tests for TargetProfile capability profiles and feature requirements (Issue #43)."""

from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import MagicMock, patch

from mlir_gpu import (
    DIRECTX_SM60,
    DIRECTX_SM66,
    HIGH_END_COMPUTE,
    PRESETS,
    VULKAN_PORTABLE_1_0,
    CompileError,
    Target,
    TargetProfile,
)


class TestTargetProfileDefaultsAndPresets(unittest.TestCase):
    """Verify default values, immutability, equality, and standard presets."""

    def test_default_profile_attributes(self):
        profile = TargetProfile()
        self.assertEqual(profile.spirv_version, "1.0")
        self.assertEqual(profile.shader_model, "6.0")
        self.assertEqual(profile.required_capabilities, ())
        self.assertEqual(profile.max_workgroup_size, (1024, 1024, 64))
        self.assertEqual(profile.max_workgroup_invocations, 1024)
        self.assertEqual(profile.max_shared_memory_bytes, 32768)

    def test_immutability(self):
        profile = TargetProfile()
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            profile.spirv_version = "1.5"  # type: ignore
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            profile.max_workgroup_invocations = 512  # type: ignore

    def test_equality_and_hashing(self):
        p1 = TargetProfile(spirv_version="1.3", max_workgroup_invocations=512)
        p2 = TargetProfile(spirv_version="1.3", max_workgroup_invocations=512)
        p3 = TargetProfile(spirv_version="1.4", max_workgroup_invocations=512)
        self.assertEqual(p1, p2)
        self.assertNotEqual(p1, p3)
        self.assertEqual(hash(p1), hash(p2))

    def test_presets_exist_and_conform(self):
        self.assertIn("VULKAN_PORTABLE_1_0", PRESETS)
        self.assertIn("DIRECTX_SM60", PRESETS)
        self.assertIn("DIRECTX_SM66", PRESETS)
        self.assertIn("HIGH_END_COMPUTE", PRESETS)

        vk = TargetProfile.preset("VULKAN_PORTABLE_1_0")
        self.assertEqual(vk.spirv_version, "1.0")
        self.assertEqual(vk.max_workgroup_size, (1024, 1024, 64))
        self.assertEqual(vk.max_shared_memory_bytes, 32768)

        dx60 = TargetProfile.preset("DIRECTX_SM60")
        self.assertEqual(dx60.shader_model, "6.0")
        self.assertEqual(dx60.max_shared_memory_bytes, 32768)

        dx66 = TargetProfile.preset("DIRECTX_SM66")
        self.assertEqual(dx66.shader_model, "6.6")
        self.assertEqual(dx66.spirv_version, "1.4")

        he = TargetProfile.preset("HIGH_END_COMPUTE")
        self.assertEqual(he.max_shared_memory_bytes, 65536)
        self.assertTrue(he.supports_capability("subgroup_operations"))

    def test_default_for_target(self):
        self.assertEqual(TargetProfile.default_for_target(Target.VULKAN), VULKAN_PORTABLE_1_0)
        self.assertEqual(TargetProfile.default_for_target(Target.DIRECTX), DIRECTX_SM60)
        self.assertEqual(TargetProfile.default_for_target(Target.METAL), DIRECTX_SM60)
        self.assertEqual(TargetProfile.default_for_target("vulkan"), VULKAN_PORTABLE_1_0)

    def test_unknown_preset_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            TargetProfile.preset("NON_EXISTENT_PROFILE")
        self.assertIn("Unknown preset profile", str(ctx.exception))


class TestTargetProfileWorkgroupValidation(unittest.TestCase):
    """Verify workgroup dimensions validation and rejection boundaries."""

    def setUp(self):
        self.profile = TargetProfile()

    def test_valid_workgroup_sizes(self):
        valid_cases = [
            (1, 1, 1),
            (8, 8, 1),
            [8, 8, 1],
            (64, 1, 1),
            (1024, 1, 1),
            (1, 1024, 1),
            (1, 1, 64),
            (32, 32, 1),
            (16, 16, 4),
            (4, 4, 64),
        ]
        for case in valid_cases:
            with self.subTest(case=case):
                self.profile.verify_workgroup_size(case)

    def test_non_positive_dimensions_rejected(self):
        invalid_cases = [
            (0, 8, 1),
            (8, 0, 1),
            (8, 8, 0),
            (-1, 1, 1),
            (1, -4, 1),
            (1, 1, -1),
        ]
        for case in invalid_cases:
            with self.subTest(case=case):
                with self.assertRaises(ValueError) as ctx:
                    self.profile.verify_workgroup_size(case)
                self.assertIn("positive", str(ctx.exception).lower())

    def test_exceeding_individual_dimension_limits_rejected(self):
        exceed_cases = [
            ((1025, 1, 1), "X", 1025, 1024),
            ((1, 1025, 1), "Y", 1025, 1024),
            ((1, 1, 65), "Z", 65, 64),
            ((2048, 1, 1), "X", 2048, 1024),
        ]
        for dims, axis, val, limit in exceed_cases:
            with self.subTest(dims=dims):
                with self.assertRaises(ValueError) as ctx:
                    self.profile.verify_workgroup_size(dims)
                msg = str(ctx.exception)
                self.assertIn(axis, msg)
                self.assertIn(str(limit), msg)

    def test_exceeding_total_invocations_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.profile.verify_workgroup_size((32, 33, 1))
        self.assertIn("1056", str(ctx.exception))
        self.assertIn("1024", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            self.profile.verify_workgroup_size((16, 16, 8))
        self.assertIn("2048", str(ctx.exception))

    def test_malformed_dimensions_rejected(self):
        with self.assertRaises(TypeError):
            self.profile.verify_workgroup_size("8, 8, 1")  # type: ignore

        with self.assertRaises(ValueError):
            self.profile.verify_workgroup_size((8, 8))
        with self.assertRaises(ValueError):
            self.profile.verify_workgroup_size((8, 8, 1, 1))

        with self.assertRaises(TypeError):
            self.profile.verify_workgroup_size((8.0, 8, 1))  # type: ignore
        with self.assertRaises(TypeError):
            self.profile.verify_workgroup_size((True, 8, 1))  # type: ignore


class TestTargetProfileSharedMemoryValidation(unittest.TestCase):
    """Verify shared memory allocation gating against profile limits."""

    def setUp(self):
        self.profile = TargetProfile(max_shared_memory_bytes=32768)

    def test_valid_shared_memory_sizes(self):
        self.profile.verify_shared_memory_bytes(0)
        self.profile.verify_shared_memory_bytes(1024)
        self.profile.verify_shared_memory_bytes(32768)

    def test_exceeding_shared_memory_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.profile.verify_shared_memory_bytes(32769)
        self.assertIn("32769", str(ctx.exception))
        self.assertIn("32768", str(ctx.exception))

    def test_negative_shared_memory_rejected(self):
        with self.assertRaises(ValueError) as ctx:
            self.profile.verify_shared_memory_bytes(-1)
        self.assertIn("negative", str(ctx.exception).lower())

    def test_non_integer_shared_memory_rejected(self):
        with self.assertRaises(TypeError):
            self.profile.verify_shared_memory_bytes(1024.5)  # type: ignore
        with self.assertRaises(TypeError):
            self.profile.verify_shared_memory_bytes("1024")  # type: ignore
        with self.assertRaises(TypeError):
            self.profile.verify_shared_memory_bytes(True)  # type: ignore


class TestTargetProfileCapabilityChecking(unittest.TestCase):
    """Verify capability query and requirement satisfaction."""

    def test_supports_capability(self):
        profile = TargetProfile(required_capabilities=("spirv_1_5", "physical_storage_buffer"))
        self.assertTrue(profile.supports_capability("spirv_1_5"))
        self.assertTrue(profile.supports_capability("physical_storage_buffer"))
        self.assertFalse(profile.supports_capability("shader_model_6_6"))

    def test_check_capabilities_empty_when_satisfied(self):
        profile = TargetProfile(required_capabilities=("shader_model_6_6", "compute"))
        available = {"shader_model_6_6", "compute", "mesh_shaders"}
        missing = profile.check_capabilities(available)
        self.assertEqual(missing, [])

    def test_check_capabilities_reports_missing(self):
        profile = TargetProfile(required_capabilities=("shader_model_6_6", "int64_atomics"))
        available = {"shader_model_6_6"}
        missing = profile.check_capabilities(available)
        self.assertEqual(missing, ["int64_atomics"])

    def test_check_capabilities_against_device_mapping(self):
        profile = TargetProfile(required_capabilities=("6.6",))
        device_caps = {
            "shader_model": "6.6",
            "extensions": ["VK_KHR_storage_buffer_storage_class"],
        }
        self.assertEqual(profile.check_capabilities(device_caps), [])


class TestTargetProfileSerialization(unittest.TestCase):
    """Verify dictionary and JSON serialization roundtrips."""

    def test_to_dict_and_from_dict(self):
        original = TargetProfile(
            spirv_version="1.5",
            shader_model="6.6",
            required_capabilities=("bda", "int64"),
            max_workgroup_size=(512, 512, 32),
            max_workgroup_invocations=512,
            max_shared_memory_bytes=49152,
        )
        d = original.to_dict()
        self.assertEqual(d["spirv_version"], "1.5")
        self.assertEqual(d["shader_model"], "6.6")
        self.assertEqual(d["required_capabilities"], ["bda", "int64"])
        self.assertEqual(d["max_workgroup_size"], [512, 512, 32])
        self.assertEqual(d["max_workgroup_invocations"], 512)
        self.assertEqual(d["max_shared_memory_bytes"], 49152)

        reconstructed = TargetProfile.from_dict(d)
        self.assertEqual(original, reconstructed)

    def test_json_roundtrip(self):
        original = TargetProfile.preset("HIGH_END_COMPUTE")
        json_str = original.to_json()
        self.assertIsInstance(json_str, str)
        reconstructed = TargetProfile.from_json(json_str)
        self.assertEqual(original, reconstructed)

    def test_from_dict_with_invalid_data_fails(self):
        with self.assertRaises(TypeError):
            TargetProfile.from_dict("not-a-dict")  # type: ignore


class TestTargetProfileViewerIntegration(unittest.TestCase):
    """Verify viewer and capabilities introspection with TargetProfile."""

    def test_match_device_profile_success(self):
        from mlir_gpu.viewer import match_device_profile
        profile = TargetProfile(
            max_workgroup_size=(1024, 1024, 64),
            max_workgroup_invocations=1024,
            max_shared_memory_bytes=32768,
        )
        device_caps = {
            "max_workgroup_size": [1024, 1024, 64],
            "max_workgroup_invocations": 1024,
            "max_shared_memory_bytes": 32768,
            "extensions": [],
        }
        matched, missing = match_device_profile(device_caps, profile)
        self.assertTrue(matched)
        self.assertEqual(missing, [])

    def test_match_device_profile_limit_insufficient(self):
        from mlir_gpu.viewer import match_device_profile
        profile = TargetProfile(max_shared_memory_bytes=65536)
        device_caps = {
            "max_workgroup_size": [1024, 1024, 64],
            "max_workgroup_invocations": 1024,
            "max_shared_memory_bytes": 32768,  # Device only has 32KB
            "extensions": [],
        }
        matched, missing = match_device_profile(device_caps, profile)
        self.assertFalse(matched)
        self.assertTrue(any("shared_memory" in m for m in missing))


class TestTargetProfileCompilerIntegration(unittest.TestCase):
    """Verify target profile handling in compiler pipeline."""

    @patch("mlir_gpu.compiler._run")
    def test_compile_stage_forwards_target_profile(self, mock_run):
        import tempfile
        from pathlib import Path
        from mlir_gpu.compiler import _compile_stage
        mock_run.return_value = "ok"

        stage_mock = MagicMock()
        stage_mock.kind = "compute"
        stage_mock.name = "compute_main"
        stage_mock.to_dict.return_value = {"functions": []}

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            tools = {"gfx-compile": "gfx-compile", "spirv-val": "spirv-val"}
            profile = TargetProfile(spirv_version="1.3")

            with patch("mlir_gpu.compiler._check_binary"):
                _compile_stage(stage_mock, Target.VULKAN, tmp_path, tools, target_profile=profile)

            calls = mock_run.call_args_list
            compile_call_args = calls[0][0][0]
            self.assertIn("--target-profile", compile_call_args)


if __name__ == "__main__":
    unittest.main()
