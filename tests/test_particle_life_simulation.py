"""Headless multi-buffer simulation, CPU parity, and golden PNG imgdiff tests for Particle Life."""

from __future__ import annotations

import math
import os
from pathlib import Path
import runpy
import struct
import tempfile
import unittest

from mlir_gpu import (
    Buffer,
    ComputePipeline,
    Dispatch,
    GlobalInvocationId,
    Target,
    Type,
    compile_pipeline,
    compute,
    dispatch_compute,
    imgdiff,
    read_png,
    write_png,
)
from test_support import ROOT, TOOLS, available

GOLDEN_PARTICLE_LIFE_PNG = (
    ROOT / "tests" / "lit" / "execution" / "Inputs" / "goldens" / "common" / "particle-life.png"
)


@unittest.skipUnless(available("gfx-compile"), "Requires native gfx-compile toolchain")
class TestParticleLifeGoldenReference(unittest.TestCase):
    """Hermetic checks verifying the CPU Particle Life simulation and golden reference PNG."""

    @classmethod
    def setUpClass(cls):
        cls.example = runpy.run_path(str(ROOT / "examples" / "particle_life_compute.py"))

    def test_cpu_simulation_matches_golden_reference_png(self):
        self.assertTrue(GOLDEN_PARTICLE_LIFE_PNG.is_file(), f"Missing golden PNG: {GOLDEN_PARTICLE_LIFE_PNG}")
        pos, vel, sp = self.example["cpu_simulate_particle_life"](steps=8)
        self.assertEqual(len(pos), 64)
        self.assertEqual(len(vel), 64)
        for px, py in pos:
            self.assertTrue(math.isfinite(px) and math.isfinite(py))
            self.assertGreaterEqual(px, 0.0)
            self.assertLess(px, 128.0)
            self.assertGreaterEqual(py, 0.0)
            self.assertLess(py, 128.0)

        cpu_img = self.example["cpu_render_particle_life"](pos, sp)
        diff = imgdiff(cpu_img, GOLDEN_PARTICLE_LIFE_PNG)
        self.assertTrue(diff.passed, diff.summary())
        self.assertEqual(diff.mismatched_pixels, 0)


@unittest.skipUnless(
    os.environ.get("MLIR_GPU_TEST_VULKAN") == "1",
    "Set MLIR_GPU_TEST_VULKAN=1 to run live Vulkan Particle Life compute simulation",
)
class TestParticleLifeVulkanExecution(unittest.TestCase):
    """Live Vulkan multi-step ping-pong simulation, read_write buffer, write_png, and imgdiff tests."""

    @classmethod
    def setUpClass(cls):
        cls.validation = os.environ.get("MLIR_GPU_TEST_VALIDATION") == "1"
        cls.example = runpy.run_path(str(ROOT / "examples" / "particle_life_compute.py"))

    def test_vulkan_read_write_storage_buffer_in_place_update(self):
        @compute(workgroup_size=(32, 1, 1))
        def axpy_in_place(
            gid: GlobalInvocationId,
            x: Buffer[Type.F32, "read_only"],
            y: Buffer[Type.F32, "read_write"],
        ):
            idx = gid.x
            if idx < 32:
                y[idx] = y[idx] + x[idx] * 2.5

        pipe = ComputePipeline(axpy_in_place, dispatch=Dispatch(grid=(1, 1, 1)))
        x_vals = [float(i) for i in range(32)]
        y_vals = [10.0 + float(i) * 0.5 for i in range(32)]
        x_bytes = struct.pack("<32f", *x_vals)
        y_bytes = struct.pack("<32f", *y_vals)

        with tempfile.TemporaryDirectory() as td:
            manifest = compile_pipeline(pipe, Target.VULKAN, Path(td) / "vk_rw", TOOLS)
            res = dispatch_compute(
                manifest,
                buffers={"x": x_bytes, "y": y_bytes},
                validation=self.validation,
                toolchain=TOOLS,
            )
            self.assertNotIn("x", res.buffers)
            self.assertIn("y", res.buffers)
            out_y = struct.unpack("<32f", res.buffers["y"])
            for idx in range(32):
                self.assertAlmostEqual(out_y[idx], y_vals[idx] + x_vals[idx] * 2.5, places=5)

    def test_vulkan_particle_life_ping_pong_simulation_and_png_imgdiff(self):
        pos_cpu, vel_cpu, _ = self.example["cpu_simulate_particle_life"](steps=8)
        pos_init, _, _ = self.example["create_initial_particle_state"]()

        pos_gpu, vel_gpu, rendered = self.example["simulate_and_render_particle_life"](
            Target.VULKAN,
            steps=8,
            width=128,
            height=128,
            validation=self.validation,
            toolchain=TOOLS,
        )

        self.assertEqual(len(pos_gpu), 64)
        self.assertEqual(len(vel_gpu), 64)

        max_pos_err = 0.0
        max_vel_err = 0.0
        total_displacement = 0.0
        for i in range(64):
            gx, gy = pos_gpu[i]
            vx, vy = vel_gpu[i]
            self.assertTrue(math.isfinite(gx) and math.isfinite(gy), f"Non-finite position at {i}: {(gx, gy)}")
            self.assertTrue(math.isfinite(vx) and math.isfinite(vy), f"Non-finite velocity at {i}: {(vx, vy)}")
            self.assertGreaterEqual(gx, 0.0)
            self.assertLess(gx, 128.0)
            self.assertGreaterEqual(gy, 0.0)
            self.assertLess(gy, 128.0)

            cx, cy = pos_cpu[i]
            cvx, cvy = vel_cpu[i]
            max_pos_err = max(max_pos_err, abs(gx - cx), abs(gy - cy))
            max_vel_err = max(max_vel_err, abs(vx - cvx), abs(vy - cvy))
            total_displacement += math.hypot(gx - pos_init[i][0], gy - pos_init[i][1])

        self.assertGreater(total_displacement, 1.0, "Particles must move across 8 simulation steps")
        self.assertLess(max_pos_err, 1e-3, f"Max GPU vs CPU position drift {max_pos_err} exceeded 1e-3")
        self.assertLess(max_vel_err, 1e-3, f"Max GPU vs CPU velocity drift {max_vel_err} exceeded 1e-3")

        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            out_png = td_path / "particle_life_vulkan.png"
            diff_png = td_path / "particle_life_vulkan_diff.png"
            write_png(out_png, rendered)
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


if __name__ == "__main__":
    unittest.main()
