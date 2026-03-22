# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for run_sensitivity.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from worldflux.parity.sensitivity import ParameterSensitivity, SensitivityReport


def _load_module():
    repo_root = Path(__file__).resolve().parents[2]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    script_path = repo_root / "scripts" / "run_sensitivity.py"
    spec = importlib.util.spec_from_file_location("run_sensitivity", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load run_sensitivity.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sensitivity"] = module
    spec.loader.exec_module(module)
    return module


def test_dry_run_prints_configurations(capsys) -> None:
    mod = _load_module()
    exit_code = mod.main(["--dry-run"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[dry-run] Configurations:" in output
    assert "kl_dynamics" in output


def test_execution_mode_writes_aggregated_json(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output = tmp_path / "dreamerv3_sensitivity.json"

    fake_report = SensitivityReport(
        family="dreamerv3",
        environment="atari100k_pong",
        seeds=[0],
        total_steps=12,
        task_id="atari100k_pong",
        env_backend="stub",
        model_profile="wf12m",
        parameters=[
            ParameterSensitivity(
                name="learning_rate",
                default_value=1e-4,
                values=[3e-5, 1e-4],
                mean_rewards=[1.0, 2.0],
                std_rewards=[0.0, 0.1],
                sensitivity_score=0.33,
                default_rank_percentile=100.0,
            )
        ],
        generated_at_utc="2026-03-21T00:00:00Z",
    )

    def _fake_run_campaign(
        *,
        analysis,
        lane,
        task_id,
        env_backend,
        device,
        model_profile,
        run_root,
        progress_callback=None,
    ):
        assert analysis.total_steps == 12
        assert lane == "smoke"
        assert task_id == "atari100k_pong"
        assert env_backend == "stub"
        assert model_profile == "wf12m"
        assert run_root == tmp_path / "runs"
        assert callable(progress_callback)
        return fake_report

    monkeypatch.setattr(mod, "run_sensitivity_campaign", _fake_run_campaign)

    exit_code = mod.main(
        [
            "--task-id",
            "atari100k_pong",
            "--env-backend",
            "stub",
            "--seeds",
            "0",
            "--steps",
            "12",
            "--run-root",
            str(tmp_path / "runs"),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "worldflux.sensitivity.v1"
    assert payload["task_id"] == "atari100k_pong"
    assert payload["env_backend"] == "stub"
    assert payload["model_profile"] == "wf12m"


def test_publish_mode_rejects_stub_backend(tmp_path: Path) -> None:
    mod = _load_module()
    exit_code = mod.main(
        [
            "--lane",
            "publish",
            "--env-backend",
            "stub",
            "--seeds",
            "0,1,2",
            "--steps",
            "200",
            "--model-profile",
            "wf12m",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 2


def test_publish_mode_rejects_ci_profile(tmp_path: Path) -> None:
    mod = _load_module()
    exit_code = mod.main(
        [
            "--lane",
            "publish",
            "--env-backend",
            "gymnasium",
            "--seeds",
            "0,1,2",
            "--steps",
            "200",
            "--model-profile",
            "ci",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 2


def test_publish_mode_rejects_degenerate_report(tmp_path: Path, monkeypatch) -> None:
    mod = _load_module()
    output = tmp_path / "dreamerv3_sensitivity.json"

    degenerate_report = SensitivityReport(
        family="dreamerv3",
        environment="atari100k_pong",
        seeds=[0, 1, 2],
        total_steps=200,
        task_id="atari100k_pong",
        env_backend="gymnasium",
        model_profile="wf12m",
        parameters=[
            ParameterSensitivity(
                name="learning_rate",
                default_value=1e-4,
                values=[3e-5, 1e-4],
                mean_rewards=[0.0, 0.0],
                std_rewards=[0.0, 0.0],
                sensitivity_score=0.0,
                default_rank_percentile=100.0,
            )
        ],
        generated_at_utc="2026-03-21T00:00:00Z",
    )

    monkeypatch.setattr(mod, "run_sensitivity_campaign", lambda **kwargs: degenerate_report)

    exit_code = mod.main(
        [
            "--lane",
            "publish",
            "--task-id",
            "atari100k_pong",
            "--env-backend",
            "gymnasium",
            "--seeds",
            "0,1,2",
            "--steps",
            "200",
            "--model-profile",
            "wf12m",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    assert not output.exists()


def test_publish_mode_rejects_noncanonical_seed_set(tmp_path: Path) -> None:
    mod = _load_module()
    exit_code = mod.main(
        [
            "--lane",
            "publish",
            "--env-backend",
            "gymnasium",
            "--seeds",
            "0",
            "--steps",
            "200",
            "--model-profile",
            "wf12m",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 2


def test_publish_mode_rejects_noncanonical_step_count(tmp_path: Path) -> None:
    mod = _load_module()
    exit_code = mod.main(
        [
            "--lane",
            "publish",
            "--env-backend",
            "gymnasium",
            "--seeds",
            "0,1,2",
            "--steps",
            "48",
            "--model-profile",
            "wf12m",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert exit_code == 2


def test_smoke_mode_rejects_tracked_publish_paths(tmp_path: Path) -> None:
    mod = _load_module()
    exit_code = mod.main(
        [
            "--lane",
            "smoke",
            "--env-backend",
            "stub",
            "--seeds",
            "0",
            "--steps",
            "12",
            "--model-profile",
            "wf12m",
            "--output",
            "reports/parity/sensitivity/dreamerv3_sensitivity.json",
            "--output-md",
            str(tmp_path / "ignored.md"),
        ]
    )
    assert exit_code == 2


def test_report_from_renders_markdown_with_optional_metadata(tmp_path: Path) -> None:
    mod = _load_module()
    report_json = tmp_path / "input.json"
    report_md = tmp_path / "output.md"
    report_json.write_text(
        json.dumps(
            {
                "schema_version": "worldflux.sensitivity.v1",
                "generated_at_utc": "2026-03-21T00:00:00Z",
                "family": "dreamerv3",
                "environment": "atari100k_pong",
                "task_id": "atari100k_pong",
                "env_backend": "stub",
                "model_profile": "wf12m",
                "seeds": [0],
                "total_steps": 12,
                "parameters": [
                    {
                        "name": "learning_rate",
                        "default_value": 1e-4,
                        "values": [3e-5, 1e-4],
                        "mean_rewards": [1.0, 2.0],
                        "std_rewards": [0.0, 0.1],
                        "sensitivity_score": 0.33,
                        "default_rank_percentile": 100.0,
                        "default_in_safe_range": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    exit_code = mod.main(["--report-from", str(report_json), "--output-md", str(report_md)])
    assert exit_code == 0
    text = report_md.read_text(encoding="utf-8")
    assert "Task ID: atari100k_pong" in text
    assert "Env backend: stub" in text
