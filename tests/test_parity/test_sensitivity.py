# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for sensitivity analysis helpers."""

from __future__ import annotations

import json
from pathlib import Path

from worldflux.parity.sensitivity import (
    ParameterSensitivity,
    SensitivityReport,
    render_sensitivity_markdown,
)
from worldflux.parity.sensitivity_runner import build_dreamer_runner_override_payload


def test_build_dreamer_runner_override_payload_maps_all_supported_sweeps() -> None:
    payload = build_dreamer_runner_override_payload(
        {
            "kl_dynamics": 0.3,
            "kl_representation": 0.05,
            "free_nats": 2.0,
            "learning_rate": 3e-4,
            "imagination_horizon": 20.0,
        }
    )

    assert payload["learning_rate_override"] == 3e-4
    assert payload["model_config_overrides"]["kl_free"] == 2.0
    assert payload["model_config_overrides"]["imagination_horizon"] == 20
    assert payload["model_config_overrides"]["loss_scales"]["kl_dynamics"] == 0.3
    assert payload["model_config_overrides"]["loss_scales"]["kl_representation"] == 0.05
    assert payload["model_config_overrides"]["loss_scales"]["reconstruction"] == 1.0


def test_sensitivity_report_json_roundtrip_preserves_optional_metadata(tmp_path: Path) -> None:
    report = SensitivityReport(
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

    payload = report.to_json_payload()
    path = tmp_path / "sensitivity.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    restored = SensitivityReport.from_json_payload(json.loads(path.read_text(encoding="utf-8")))
    assert restored.task_id == "atari100k_pong"
    assert restored.env_backend == "stub"
    assert restored.model_profile == "wf12m"


def test_render_sensitivity_markdown_includes_runner_metadata() -> None:
    report = SensitivityReport(
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

    text = render_sensitivity_markdown(report)
    assert "Task ID: atari100k_pong" in text
    assert "Env backend: stub" in text
    assert "Model profile: wf12m" in text
