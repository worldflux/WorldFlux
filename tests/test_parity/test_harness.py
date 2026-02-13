"""Tests for parity run/aggregate/report pipeline."""

from __future__ import annotations

import json
from pathlib import Path

from worldflux.parity.harness import aggregate_runs, render_markdown_report, run_suite
from worldflux.parity.loaders import load_score_points


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_suite(path: Path, upstream: Path, worldflux: Path, *, margin: float = 0.05) -> None:
    suite = {
        "suite_id": "test_suite",
        "family": "dreamerv3",
        "metric": "episode_return",
        "higher_is_better": True,
        "margin_ratio": margin,
        "confidence": 0.95,
        "tasks": ["task_a", "task_b"],
        "upstream": {
            "repo": "https://example.com/upstream",
            "commit": "deadbeef",
            "format": "canonical_json",
            "path": str(upstream),
        },
        "worldflux": {
            "repo": "https://example.com/worldflux",
            "format": "canonical_json",
            "path": str(worldflux),
        },
    }
    _write_json(path, suite)


def test_run_suite_generates_pass_result(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    worldflux = tmp_path / "worldflux.json"
    suite = tmp_path / "suite.yaml"
    run_output = tmp_path / "run.json"

    _write_json(
        upstream,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 100.0},
                {"task": "task_a", "seed": 1, "step": 100, "score": 102.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 80.0},
                {"task": "task_b", "seed": 1, "step": 100, "score": 78.0},
            ]
        },
    )
    _write_json(
        worldflux,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 97.0},
                {"task": "task_a", "seed": 1, "step": 100, "score": 99.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 78.0},
                {"task": "task_b", "seed": 1, "step": 100, "score": 77.0},
            ]
        },
    )
    _write_suite(suite, upstream, worldflux)

    payload = run_suite(suite, output_path=run_output)

    assert run_output.exists()
    assert payload["suite"]["suite_id"] == "test_suite"
    assert payload["counts"]["paired"] == 4
    assert payload["stats"]["pass_non_inferiority"] is True
    assert payload["evaluation_manifest"]["runner"] == "worldflux.parity.run_suite"
    assert payload["artifact_integrity"]["suite_sha256"]
    assert payload["suite_lock_ref"]["suite_id"] == "test_suite"
    assert "PASS:" in payload["stats"]["verdict_reason"]


def test_run_suite_generates_fail_result(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    worldflux = tmp_path / "worldflux.json"
    suite = tmp_path / "suite.yaml"
    run_output = tmp_path / "run_fail.json"

    _write_json(
        upstream,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 100.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 90.0},
            ]
        },
    )
    _write_json(
        worldflux,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 70.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 60.0},
            ]
        },
    )
    _write_suite(suite, upstream, worldflux, margin=0.05)

    payload = run_suite(suite, output_path=run_output)
    assert payload["stats"]["pass_non_inferiority"] is False


def test_aggregate_and_report(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    worldflux = tmp_path / "worldflux.json"
    suite = tmp_path / "suite.yaml"
    run_path = tmp_path / "run.json"
    aggregate_path = tmp_path / "aggregate.json"

    _write_json(
        upstream,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 10.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 20.0},
            ]
        },
    )
    _write_json(
        worldflux,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 10.1},
                {"task": "task_b", "seed": 0, "step": 100, "score": 19.8},
            ]
        },
    )
    _write_suite(suite, upstream, worldflux)

    run_suite(suite, output_path=run_path)
    aggregate = aggregate_runs([run_path], output_path=aggregate_path)

    assert aggregate_path.exists()
    assert aggregate["run_count"] == 1
    assert "verdict_reason" in aggregate["suites"][0]
    markdown = render_markdown_report(aggregate)
    assert "WorldFlux Parity Report" in markdown
    assert "test_suite" in markdown


def test_load_tdmpc_csv_directory(tmp_path: Path) -> None:
    csv_dir = tmp_path / "tdmpc"
    csv_dir.mkdir(parents=True)
    (csv_dir / "task_a.csv").write_text(
        "step,reward,seed\n0,1.0,0\n100,2.0,0\n0,1.5,1\n100,2.5,1\n",
        encoding="utf-8",
    )

    points = load_score_points(csv_dir, "tdmpc2_results_csv_dir")
    assert len(points) == 4
    assert {point.task for point in points} == {"task_a"}
