"""Host-only unit tests for the cross-platform benchmark runner and website generator."""

from __future__ import annotations

import json
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import build_benchmark_site  # noqa: E402
import run_numpy_benchmarks  # noqa: E402


class TestBenchmarkRunner(unittest.TestCase):
    def test_nacelle_report_label_is_distinct_and_can_be_overridden(self) -> None:
        for label in (None, "custom-nacelle"):
            with self.subTest(label=label), mock.patch.dict(sys.modules, {"numpy": None}):
                report = run_numpy_benchmarks.run_benchmarks(
                    target_arg="directx", provider="nacelle", backend_label=label,
                    preset="quick", warmup=0, iters=1, skip_gpu=True,
                )
                expected = label or "gfxir-directx-nacelle"
                self.assertEqual(report["provider"], "nacelle")
                self.assertIn(expected, report["overallGeomeanMs"])
                self.assertEqual(
                    {r["backend"] for r in report["results"]}, {"numpy-cpu-f32", expected},
                )

    def test_numerically_incorrect_result_is_not_a_successful_benchmark(self) -> None:
        np = mock.MagicMock()
        np.max.return_value = 1.0
        np.allclose.return_value = False
        with mock.patch("mlir_gpu.compile_pipeline"), mock.patch("mlir_gpu.open_compute_context"):
            row = run_numpy_benchmarks._benchmark_gfx_suite(
                run_numpy_benchmarks.SuiteSpec("bad", "Bad result", "test", "saxpy_chain", "1", n=1),
                "directx", "nacelle", 0, 1, np, mock.Mock(shape=(1,)),
            )
        self.assertFalse(row["ran"])
        self.assertIn("Numerical mismatch", row["status"])

    def test_require_all_rejects_partial_results_and_keeps_diagnostic_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "bench.json"
            for success in (True, False):
                report = {"results": [
                    {"backend": "numpy-cpu-f32", "benchId": "a", "ran": True, "status": "ok"},
                    {"backend": "gfxir-directx-nacelle", "benchId": "a", "ran": True, "status": "ok"},
                    {"backend": "gfxir-directx-nacelle", "benchId": "b", "ran": success, "status": "unavailable"},
                ]}
                with (
                    self.subTest(success=success),
                    mock.patch.object(run_numpy_benchmarks, "run_benchmarks", return_value=report),
                    mock.patch.object(sys, "argv", ["bench", "--require-all", "--out", str(path)]),
                    mock.patch("sys.stdout", new_callable=io.StringIO),
                    mock.patch("sys.stderr", new_callable=io.StringIO) as stderr,
                ):
                    self.assertEqual(run_numpy_benchmarks.main(), 0 if success else 1)
                    self.assertEqual(json.loads(path.read_text()), report)
                    if not success:
                        self.assertIn("gfxir-directx-nacelle / b: unavailable", stderr.getvalue())

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

    def test_merge_reports_deduplicates_numpy_and_keeps_vulkan_and_nacelle(self) -> None:
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
        r3 = {
            "version": 1,
            "os": "Darwin",
            "arch": "arm64",
            "target": "directx",
            "provider": "nacelle",
            "overallGeomeanMs": {"numpy-cpu-f32": 0.01, "gfxir-directx-nacelle": 3.2},
            "results": [
                {"benchId": "matmul_64", "backend": "numpy-cpu-f32", "ran": True, "medianMs": 0.01},
                {"benchId": "matmul_64", "backend": "gfxir-directx-nacelle", "ran": True, "medianMs": 3.2,
                 "runtimeMode": "persistent:Nacelle"},
            ],
        }
        merged = build_benchmark_site.merge_reports([r1, r2, r3])
        backends = [r["backend"] for r in merged["results"]]
        self.assertEqual(backends, [
            "numpy-cpu-f32", "gfxir-vulkan-moltenvk", "gfxir-vulkan-kosmickrisp", "gfxir-directx-nacelle",
        ])
        self.assertIn("gfxir-vulkan-moltenvk", merged["overallGeomeanMs"])
        self.assertIn("gfxir-vulkan-kosmickrisp", merged["overallGeomeanMs"])
        self.assertEqual(merged["overallGeomeanMs"]["gfxir-directx-nacelle"], 3.2)
        self.assertEqual(merged["target"], "vulkan, directx")

    def test_site_legend_renders_accessible_backend_toggles(self) -> None:
        html = (ROOT / "pages" / "index.html").read_text(encoding="utf-8")
        self.assertIn('class="legend-toggle"', html)
        self.assertIn("aria-pressed=", html)
        self.assertIn("hiddenBackends", html)


if __name__ == "__main__":
    unittest.main()
