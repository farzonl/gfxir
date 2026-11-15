#!/usr/bin/env python3
"""Merge benchmark JSON results into history and build the static benchmark website."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request
from typing import Any


def _git_output(args: list[str], cwd: Path) -> str:
    """Return stripped stdout from a git command, or empty string on failure."""
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
        )
        return res.stdout.strip() if res.returncode == 0 else ""
    except Exception:
        return ""


def fetch_json_url(url: str) -> Any:
    """Fetch JSON from an HTTP(S) URL, returning None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "GfxIR-Bench-CI"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def load_existing_history(pages_url: str | None, cache_file: Path | None) -> list[dict[str, Any]]:
    """Load existing benchmark history from local cache or live GitHub Pages."""
    if cache_file and cache_file.exists():
        try:
            loaded = json.loads(cache_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                return loaded
        except Exception as exc:
            print(f"Warning: failed to read cached bench history {cache_file}: {exc}")

    if pages_url:
        loaded = fetch_json_url(pages_url.rstrip("/") + "/data/bench-history.json")
        if isinstance(loaded, list):
            return loaded

    return []


def build_history_entry(
    raw: dict[str, Any],
    *,
    sha: str,
    message: str,
    author: str,
    timestamp: str,
    run_url: str,
) -> dict[str, Any]:
    """Normalize a benchmark report into a history entry."""
    results = raw.get("results", [])
    return {
        "version": int(raw.get("version", 1)),
        "sha": sha,
        "shortSha": sha[:7] if sha else "local",
        "message": message.splitlines()[0][:120] if message else "",
        "author": author,
        "timestamp": timestamp,
        "runUrl": run_url,
        "os": raw.get("os", ""),
        "arch": raw.get("arch", ""),
        "target": raw.get("target", ""),
        "provider": raw.get("provider", ""),
        "numpyVersion": raw.get("numpyVersion"),
        "overallWinner": raw.get("overallWinner", ""),
        "overallGeomeanMs": raw.get("overallGeomeanMs", {}),
        "results": results,
    }


def merge_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple benchmark JSON reports (e.g. Metal + MoltenVK + KosmicKrisp) into one."""
    if not reports:
        raise ValueError("At least one benchmark report is required.")
    if len(reports) == 1:
        return reports[0]

    base = dict(reports[0])
    seen_numpy_suites: set[str] = set()
    merged_results: list[dict[str, Any]] = []
    merged_geomeans: dict[str, float] = {}
    targets: list[str] = []

    for report in reports:
        t = str(report.get("target", "")).strip()
        if t and t not in targets:
            targets.append(t)
        for k, v in (report.get("overallGeomeanMs") or {}).items():
            if isinstance(v, (int, float)) and (k not in merged_geomeans or (v > 0 and merged_geomeans[k] <= 0)):
                merged_geomeans[k] = float(v)
        for row in report.get("results") or []:
            backend = row.get("backend")
            bench_id = row.get("benchId")
            if backend == "numpy-cpu-f32":
                if bench_id in seen_numpy_suites:
                    continue
                seen_numpy_suites.add(str(bench_id))
            merged_results.append(row)

    positive = [(k, v) for k, v in merged_geomeans.items() if v > 0]
    overall_winner = min(positive, key=lambda kv: kv[1])[0] if positive else base.get("overallWinner", "")

    base["target"] = ", ".join(targets) if targets else base.get("target", "")
    base["overallGeomeanMs"] = merged_geomeans
    base["overallWinner"] = overall_winner
    base["results"] = merged_results
    return base


def generate_site(
    *,
    bench_json: Path | list[Path],
    site_src: Path,
    out_dir: Path,
    sha: str,
    message: str,
    author: str,
    timestamp: str,
    run_url: str,
    pages_url: str,
    history_cache: Path | None,
) -> list[dict[str, Any]]:
    """Update benchmark history and assemble the static website in out_dir."""
    paths = [bench_json] if isinstance(bench_json, Path) else list(bench_json)
    reports = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    raw = merge_reports(reports)
    entry = build_history_entry(
        raw,
        sha=sha,
        message=message,
        author=author,
        timestamp=timestamp,
        run_url=run_url,
    )

    history = load_existing_history(pages_url or None, history_cache)
    history = [h for h in history if h.get("sha") != sha and h.get("results")]
    history.append(entry)
    history.sort(key=lambda h: h.get("timestamp", ""))

    if history_cache:
        history_cache.parent.mkdir(parents=True, exist_ok=True)
        history_cache.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")

    out_dir.mkdir(parents=True, exist_ok=True)
    for item in site_src.iterdir():
        dst = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    data_dir = out_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "bench-history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    return history


def main() -> int:
    repo_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Build the GfxIR vs NumPy benchmark website.")
    parser.add_argument("--bench-json", required=True, nargs="+", type=Path)
    parser.add_argument("--site-src", type=Path, default=repo_root / "pages")
    parser.add_argument("--out-dir", type=Path, default=repo_root / "_site")
    parser.add_argument("--sha", default="")
    parser.add_argument("--message", default="")
    parser.add_argument("--author", default="")
    parser.add_argument("--timestamp", default="")
    parser.add_argument("--run-url", default="")
    parser.add_argument("--pages-url", default="")
    parser.add_argument("--history-cache", type=Path, default=None)
    args = parser.parse_args()

    sha = args.sha or _git_output(["rev-parse", "HEAD"], repo_root) or "local"
    message = args.message or _git_output(["log", "-1", "--pretty=%s"], repo_root)
    author = args.author or _git_output(["log", "-1", "--pretty=%an"], repo_root)
    ts = (
        args.timestamp
        or _git_output(["log", "-1", "--pretty=%cI"], repo_root)
        or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    history = generate_site(
        bench_json=args.bench_json,
        site_src=args.site_src,
        out_dir=args.out_dir,
        sha=sha,
        message=message,
        author=author,
        timestamp=ts,
        run_url=args.run_url,
        pages_url=args.pages_url,
        history_cache=args.history_cache,
    )
    print(f"Built benchmark site in {args.out_dir} ({len(history)} history entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
