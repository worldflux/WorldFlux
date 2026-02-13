"""CLI tests for parity subcommands."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

if importlib.util.find_spec("typer") is None or importlib.util.find_spec("rich") is None:
    pytest.skip("CLI dependencies are not installed", allow_module_level=True)

from typer.testing import CliRunner

import worldflux.cli as cli

runner = CliRunner()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_parity_cli_run_aggregate_report(tmp_path: Path) -> None:
    upstream = tmp_path / "upstream.json"
    worldflux = tmp_path / "worldflux.json"
    suite = tmp_path / "suite.yaml"
    run_output = tmp_path / "run.json"
    aggregate_output = tmp_path / "aggregate.json"
    report_output = tmp_path / "report.md"

    _write_json(
        upstream,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 100.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 50.0},
            ]
        },
    )
    _write_json(
        worldflux,
        {
            "scores": [
                {"task": "task_a", "seed": 0, "step": 100, "score": 98.0},
                {"task": "task_b", "seed": 0, "step": 100, "score": 49.5},
            ]
        },
    )
    _write_json(
        suite,
        {
            "suite_id": "cli_suite",
            "family": "dreamerv3",
            "metric": "episode_return",
            "higher_is_better": True,
            "margin_ratio": 0.05,
            "confidence": 0.95,
            "tasks": ["task_a", "task_b"],
            "upstream": {"format": "canonical_json", "path": str(upstream)},
            "worldflux": {"format": "canonical_json", "path": str(worldflux)},
        },
    )

    run_result = runner.invoke(cli.app, ["parity", "run", str(suite), "--output", str(run_output)])
    assert run_result.exit_code == 0
    assert run_output.exists()

    aggregate_result = runner.invoke(
        cli.app,
        [
            "parity",
            "aggregate",
            "--run",
            str(run_output),
            "--output",
            str(aggregate_output),
        ],
    )
    assert aggregate_result.exit_code == 0
    assert aggregate_output.exists()

    report_result = runner.invoke(
        cli.app,
        [
            "parity",
            "report",
            "--aggregate",
            str(aggregate_output),
            "--output",
            str(report_output),
        ],
    )
    assert report_result.exit_code == 0
    assert report_output.exists()
    assert "WorldFlux Parity Report" in report_output.read_text(encoding="utf-8")


def test_parity_campaign_export_cli(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    campaign = tmp_path / "campaign.yaml"
    output = tmp_path / "export.json"

    _write_json(
        source,
        {
            "scores": [
                {"task": "task_a", "seed": 1, "step": 10, "score": 3.2},
            ]
        },
    )
    _write_json(
        campaign,
        {
            "schema_version": "worldflux.parity.campaign.v1",
            "suite_id": "cli_campaign",
            "family": "tdmpc2",
            "default_seeds": [1],
            "tasks": ["task_a"],
            "sources": {
                "worldflux": {
                    "input_path": str(source),
                    "input_format": "canonical_json",
                    "output_path": str(output),
                }
            },
        },
    )

    result = runner.invoke(
        cli.app,
        [
            "parity",
            "campaign",
            "export",
            str(campaign),
            "--source",
            "worldflux",
            "--output",
            str(output),
            "--seeds",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert output.exists()
