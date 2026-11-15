"""Host-only unit tests for the cross-platform benchmark runner and website generator."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_benchmark_site  # noqa: E402
import run_numpy_benchmarks  # noqa: E402


class TestBenchmarkRunner(unittest.TestCase):
    def test_resolve_target_validates_and_defaults(self) -> None:
        self.assertIn(run_numpy_benchmarks.resolve_target("auto"), {"metal", "vulkan", "directx"})
        self.assertEqual(run_numpy_benchmarks.resolve_target("Metal"), "metal")
        self.assertEqual(run_numpy_benchmarks.resolve_target("vulkan"), "vulkan")
        self.assertEqual(run_numpy_benchmarks.resolve_target("directx"), "directx")
        with self.assertRaises(ValueError):
            run_numpy_benchmarks.resolve_target("cuda")

    def test_build_suites_presets(self) -> None:
        quick = run_numpy_benchmarks.build_suites("quick")
        full = run_numpy_benchmarks.build_suites("full")
        self.assertGreaterEqual(len(quick), 2)
        self.assertGreater(len(full), len(quick))
        self.assertTrue(any(s.kind == "saxpy_chain" for s in quick))
        self.assertTrue(any(s.kind == "matmul" for s in quick))

    def test_run_benchmarks_skip_gpu_schema(self) -> None:
        report = run_numpy_benchmarks.run_benchmarks(
            target_arg="auto",
            preset="quick",
            warmup=0,
            iters=1,
            skip_gpu=True,
        )
        self.assertEqual(report["version"], 1)
        self.assertIn("results", report)
        self.assertGreater(len(report["results"]), 0)
        for rec in report["results"]:
            self.assertIn("benchId", rec)
            self.assertIn("backend", rec)
            self.assertIn("ran", rec)
            self.assertIn("status", rec)
            if rec["backend"].startswith("gfxir-"):
                self.assertFalse(rec["ran"])
                self.assertEqual(rec["status"], "skipped: --skip-gpu")

    def test_cli_list_suites(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS_DIR / "run_numpy_benchmarks.py"), "--list-suites", "--preset", "quick"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("saxpy_16k_k1", proc.stdout)
        self.assertIn("matmul_64", proc.stdout)


class TestBenchmarkSiteBuilder(unittest.TestCase):
    def test_generate_site_deduplicates_and_sorts_history(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            bench_json = tmp / "bench.json"
            cache_file = tmp / "cache" / "bench-history.json"
            out_dir = tmp / "_site"

            sample_report = {
                "version": 1,
                "os": "Darwin",
                "arch": "arm64",
                "target": "metal",
                "provider": "default",
                "numpyVersion": "2.0.0",
                "overallWinner": "gfxir-metal",
                "overallGeomeanMs": {"numpy-cpu-f32": 2.5, "gfxir-metal": 1.2},
                "results": [
                    {
                        "benchId": "matmul_256",
                        "benchName": "MatMul 256x256",
                        "category": "Matrix Multiply",
                        "backend": "numpy-cpu-f32",
                        "ran": True,
                        "medianMs": 2.5,
                        "status": "ok",
                    },
                    {
                        "benchId": "matmul_256",
                        "benchName": "MatMul 256x256",
                        "category": "Matrix Multiply",
                        "backend": "gfxir-metal",
                        "ran": True,
                        "medianMs": 1.2,
                        "status": "ok",
                    },
                ],
            }
            bench_json.write_text(json.dumps(sample_report), encoding="utf-8")

            build_benchmark_site.generate_site(
                bench_json=bench_json,
                site_src=ROOT / "pages",
                out_dir=out_dir,
                sha="aaaaaaa1111111",
                message="first commit",
                author="dev",
                timestamp="2026-11-14T10:00:00Z",
                run_url="",
                pages_url="",
                history_cache=cache_file,
            )
            # Re-run same SHA with updated message -> should deduplicate, not append a second entry.
            history = build_benchmark_site.generate_site(
                bench_json=bench_json,
                site_src=ROOT / "pages",
                out_dir=out_dir,
                sha="aaaaaaa1111111",
                message="updated commit",
                author="dev",
                timestamp="2026-11-14T11:00:00Z",
                run_url="",
                pages_url="",
                history_cache=cache_file,
            )
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["message"], "updated commit")

            # Add an earlier commit -> should sort chronologically.
            history = build_benchmark_site.generate_site(
                bench_json=bench_json,
                site_src=ROOT / "pages",
                out_dir=out_dir,
                sha="bbbbbbb2222222",
                message="earlier commit",
                author="dev",
                timestamp="2026-11-14T09:00:00Z",
                run_url="",
                pages_url="",
                history_cache=cache_file,
            )
            self.assertEqual([h["shortSha"] for h in history], ["bbbbbbb", "aaaaaaa"])
            self.assertTrue((out_dir / "index.html").is_file())
            self.assertTrue((out_dir / "data" / "bench-history.json").is_file())

    def test_merge_reports_deduplicates_numpy_and_keeps_both_vulkan_runtimes(self) -> None:
        r1 = {
            "version": 1,
            "os": "Darwin",
            "arch": "arm64",
            "target": "vulkan",
            "overallWinner": "numpy-cpu-f32",
            "overallGeomeanMs": {"numpy-cpu-f32": 0.01, "gfxir-vulkan-moltenvk": 1.2},
            "results": [
                {"benchId": "matmul_64", "benchName": "MatMul 64x64", "backend": "numpy-cpu-f32", "ran": True, "medianMs": 0.01},
                {"benchId": "matmul_64", "benchName": "MatMul 64x64", "backend": "gfxir-vulkan-moltenvk", "ran": True, "medianMs": 1.2},
            ],
        }
        r2 = {
            "version": 1,
            "os": "Darwin",
            "arch": "arm64",
            "target": "vulkan",
            "overallWinner": "numpy-cpu-f32",
            "overallGeomeanMs": {"numpy-cpu-f32": 0.01, "gfxir-vulkan-kosmickrisp": 2.1},
            "results": [
                {"benchId": "matmul_64", "benchName": "MatMul 64x64", "backend": "numpy-cpu-f32", "ran": True, "medianMs": 0.01},
                {"benchId": "matmul_64", "benchName": "MatMul 64x64", "backend": "gfxir-vulkan-kosmickrisp", "ran": True, "medianMs": 2.1},
            ],
        }
        merged = build_benchmark_site.merge_reports([r1, r2])
        backends = [r["backend"] for r in merged["results"]]
        self.assertEqual(backends, ["numpy-cpu-f32", "gfxir-vulkan-moltenvk", "gfxir-vulkan-kosmickrisp"])
        self.assertIn("gfxir-vulkan-moltenvk", merged["overallGeomeanMs"])
        self.assertIn("gfxir-vulkan-kosmickrisp", merged["overallGeomeanMs"])


if __name__ == "__main__":
    unittest.main()
