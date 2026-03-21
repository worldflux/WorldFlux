# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for the eval CLI surface."""

from __future__ import annotations

from dataclasses import replace

import pytest
from typer.testing import CliRunner

import worldflux.cli as cli
from worldflux.evals.result import EvalReport, EvalResult

runner = CliRunner()


def _fake_report(*, mode: str) -> EvalReport:
    result = EvalResult(
        suite="quick",
        metric="reconstruction_fidelity",
        value=0.1,
        threshold=0.5,
        passed=True,
        timestamp=0.0,
        model_id="demo",
    )
    report = EvalReport(
        suite="quick",
        model_id="demo",
        results=(result,),
        timestamp=0.0,
        wall_time_sec=0.1,
        all_passed=True,
    )
    return replace(
        report,
        mode=mode,
        provenance={
            "kind": "synthetic" if mode == "synthetic" else "dataset_manifest",
            "env_id": "mujoco/HalfCheetah-v5",
        },
    )


def test_eval_json_includes_synthetic_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "worldflux.evals.run_eval_suite",
        lambda *args, **kwargs: _fake_report(mode="synthetic"),
    )

    result = runner.invoke(cli.app, ["eval", "dreamer:ci", "--format", "json"])
    assert result.exit_code == 0
    assert '"mode": "synthetic"' in result.output
    assert '"synthetic_provenance"' in result.output


def test_eval_real_mode_requires_dataset_manifest_or_env_id() -> None:
    result = runner.invoke(cli.app, ["eval", "dreamer:ci", "--mode", "real", "--format", "json"])
    assert result.exit_code == 1
    assert "dataset manifest" in result.output.lower() or "env id" in result.output.lower()


def test_eval_real_mode_emits_real_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    manifest = tmp_path / "dataset_manifest.json"
    manifest.write_text(
        '{"schema_version":"worldflux.dataset.manifest.v1","env_id":"mujoco/HalfCheetah-v5"}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "worldflux.cli._eval._load_real_eval_data",
        lambda *args, **kwargs: (
            {
                "obs": None,
                "actions": None,
                "rewards": None,
            },
            {"kind": "dataset_manifest", "env_id": "mujoco/HalfCheetah-v5"},
        ),
    )
    monkeypatch.setattr(
        "worldflux.evals.run_eval_suite",
        lambda *args, **kwargs: _fake_report(mode="real"),
    )

    result = runner.invoke(
        cli.app,
        [
            "eval",
            "dreamer:ci",
            "--mode",
            "real",
            "--dataset-manifest",
            str(manifest),
            "--format",
            "json",
        ],
    )
    assert result.exit_code == 0
    assert '"mode": "real"' in result.output
    assert '"real_provenance"' in result.output
