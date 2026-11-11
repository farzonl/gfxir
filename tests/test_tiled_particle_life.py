"""Shared-memory tiled Particle Life trajectory parity, write_png, and golden imgdiff tests."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest

from mlir_gpu import Target, compile_pipeline, imgdiff, read_png, write_png
from test_support import ROOT, TOOLS, available

GOLDEN_PARTICLE_LIFE_PNG = (
    ROOT / "tests" / "lit" / "execution" / "Inputs" / "goldens" / "common" / "particle-life.png"
)


@unittest.skipUnless(available("gfx-compile"), "Requires native gfx-compile toolchain")
class TestTiledParticleLifeCompilation(unittest.TestCase):
    """Hermetic compilation and shared-memory footprint checks for tiled_particle_life.py."""

    @classmethod
    def setUpClass(cls):
        cls.tiled = runpy.run_path(str(ROOT / "examples" / "tiled_particle_life.py"))

    def test_step_and_render_kernels_compile_and_reflect_shared_memory(self):
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            for name, pipe in (
                ("step", self.tiled["pipeline"]),
                ("render", self.tiled["render_pipeline"]),
            ):
                vk_manifest_path = compile_pipeline(pipe, Target.VULKAN, td_path / f"vk_{name}", TOOLS)
                vk_manifest = json.loads(vk_manifest_path.read_text(encoding="utf-8"))
                # 64 x vec2 (512 B) + 64 x i32 (256 B) = 768 bytes.
                self.assertEqual(vk_manifest["shared_memory_bytes"], 768)

                if available("llc") and available("dxv"):
                    dx_manifest_path = compile_pipeline(pipe, Target.DIRECTX, td_path / f"dx_{name}", TOOLS)
                    dx_manifest = json.loads(dx_manifest_path.read_text(encoding="utf-8"))
                    self.assertEqual(dx_manifest["shared_memory_bytes"], 768)


class _TiledParticleLifeExecutionMixin:
    target: Target
    provider: str | None = None
    validation: bool = False
    tiled: dict
    baseline: dict

    def test_tiled_simulation_matches_baseline_and_golden_png_imgdiff(self):
        pos_cpu, vel_cpu, _ = self.baseline["cpu_simulate_particle_life"](steps=8)
        pos_base, vel_base, _ = self.baseline["simulate_and_render_particle_life"](
            self.target,
            steps=8,
            width=128,
            height=128,
            validation=self.validation,
            provider=self.provider,
            toolchain=TOOLS,
        )

        res = self.tiled["simulate_and_render_tiled_particle_life"](
            self.target,
            steps=8,
            width=128,
            height=128,
            validation=self.validation,
            provider=self.provider,
            toolchain=TOOLS,
        )
        self.assertEqual(len(res.positions), 64)
        self.assertEqual(len(res.velocities), 64)
        self.assertGreater(res.sim_elapsed_ms, 0.0)
        self.assertGreater(res.render_elapsed_ms, 0.0)

        max_base_pos_diff = 0.0
        max_base_vel_diff = 0.0
        max_cpu_pos_diff = 0.0
        for i in range(64):
            tx, ty = res.positions[i]
            tvx, tvy = res.velocities[i]
            bx, by = pos_base[i]
            bvx, bvy = vel_base[i]
            cx, cy = pos_cpu[i]
            self.assertTrue(math.isfinite(tx) and math.isfinite(ty))
            self.assertTrue(math.isfinite(tvx) and math.isfinite(tvy))
            max_base_pos_diff = max(max_base_pos_diff, abs(tx - bx), abs(ty - by))
            max_base_vel_diff = max(max_base_vel_diff, abs(tvx - bvx), abs(tvy - bvy))
            max_cpu_pos_diff = max(max_cpu_pos_diff, abs(tx - cx), abs(ty - cy))

        self.assertLess(max_base_pos_diff, 1e-4, f"Tiled vs baseline GPU position drift: {max_base_pos_diff}")
        self.assertLess(max_base_vel_diff, 1e-4, f"Tiled vs baseline GPU velocity drift: {max_base_vel_diff}")
        self.assertLess(max_cpu_pos_diff, 1e-3, f"Tiled GPU vs CPU position drift: {max_cpu_pos_diff}")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_png = td_path / f"tiled_particle_life_{self.target.value}.png"
            diff_png = td_path / f"tiled_particle_life_{self.target.value}_diff.png"
            write_png(out_png, res.image)
            self.assertTrue(out_png.is_file())
            decoded = read_png(out_png)
            self.assertEqual((decoded.width, decoded.height), (128, 128))

            diff = imgdiff(
                out_png,
                GOLDEN_PARTICLE_LIFE_PNG,
                cie76_threshold=2.3,
                max_outlier_fraction=0.002,
                diff_output=diff_png,
            )
            self.assertTrue(diff.passed, diff.summary())
            self.assertTrue(diff_png.is_file())

    def test_multi_tile_non_aligned_particle_count_parity(self):
        params = dict(self.tiled["DEFAULT_SIMULATION_PARAMS"])
        params["count"] = 160  # Spans 3 shared-memory tiles (64 + 64 + 32).
        pos_cpu, vel_cpu, _ = self.baseline["cpu_simulate_particle_life"](steps=4, params=params)
        res = self.tiled["simulate_and_render_tiled_particle_life"](
            self.target,
            steps=4,
            width=64,
            height=64,
            params=params,
            validation=self.validation,
            provider=self.provider,
            toolchain=TOOLS,
        )
        self.assertEqual(len(res.positions), 160)
        max_pos_err = max(
            max(abs(res.positions[i][0] - pos_cpu[i][0]), abs(res.positions[i][1] - pos_cpu[i][1]))
            for i in range(160)
        )
        max_vel_err = max(
            max(abs(res.velocities[i][0] - vel_cpu[i][0]), abs(res.velocities[i][1] - vel_cpu[i][1]))
            for i in range(160)
        )
        self.assertLess(max_pos_err, 1e-3)
        self.assertLess(max_vel_err, 1e-3)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan tiled Particle Life tests",
)
class TestTiledParticleLifeVulkanExecution(_TiledParticleLifeExecutionMixin, unittest.TestCase):
    target = Target.VULKAN

    @classmethod
    def setUpClass(cls):
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.tiled = runpy.run_path(str(ROOT / "examples" / "tiled_particle_life.py"))
        cls.baseline = runpy.run_path(str(ROOT / "examples" / "particle_life_compute.py"))


@unittest.skipUnless(
    sys.platform == "win32" and os.environ.get("MLIR_GPU_TEST_DIRECTX") == "1",
    "Set MLIR_GPU_TEST_DIRECTX=1 on Windows to run live DirectX 12 WARP tiled Particle Life tests",
)
class TestTiledParticleLifeDirectXExecution(_TiledParticleLifeExecutionMixin, unittest.TestCase):
    target = Target.DIRECTX
    provider = "warp"

    @classmethod
    def setUpClass(cls):
        if not (available("llc") and available("dxv") and available("mlir-gpu-directx-view")):
            raise unittest.SkipTest("DirectX 12 compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.tiled = runpy.run_path(str(ROOT / "examples" / "tiled_particle_life.py"))
        cls.baseline = runpy.run_path(str(ROOT / "examples" / "particle_life_compute.py"))


@unittest.skipUnless(
    sys.platform == "darwin" and os.environ.get("MLIR_GPU_TEST_METAL") == "1",
    "Set MLIR_GPU_TEST_METAL=1 on macOS to run live Metal tiled Particle Life tests",
)
class TestTiledParticleLifeMetalExecution(_TiledParticleLifeExecutionMixin, unittest.TestCase):
    target = Target.METAL

    @classmethod
    def setUpClass(cls):
        required = ("llc", "dxv", "metal-shaderconverter", "mlir-gpu-metal-view")
        if not all(available(tool) for tool in required):
            raise unittest.SkipTest("Metal compute execution tools are unavailable")
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.tiled = runpy.run_path(str(ROOT / "examples" / "tiled_particle_life.py"))
        cls.baseline = runpy.run_path(str(ROOT / "examples" / "particle_life_compute.py"))


if __name__ == "__main__":
    unittest.main()
