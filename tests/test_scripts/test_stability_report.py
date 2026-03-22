# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for proof stability report aggregation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "stability_report.py"
    spec = importlib.util.spec_from_file_location("stability_report", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load stability_report")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _write_runs(path: Path) -> None:
    rows = [
        {
            "task_id": "walker-run",
            "seed": 0,
            "system": "official",
            "status": "success",
            "metrics": {"final_return_mean": 1.0, "auc_return": 0.8},
        },
        {
            "task_id": "walker-run",
            "seed": 0,
            "system": "worldflux",
            "status": "success",
            "metrics": {"final_return_mean": 1.0, "auc_return": 0.8},
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def _report(*, parity_pass_final: bool, ratio_mean: float) -> dict:
    return {
        "global": {
            "parity_pass_final": parity_pass_final,
            "validity_pass": True,
            "missing_pairs": 0,
            "parity_pass_bayesian": parity_pass_final,
            "parity_pass_frequentist": parity_pass_final,
        },
        "tasks": [
            {
                "task_id": "walker-run",
                "metrics": {
                    "final_return_mean": {
                        "status": "ok",
                        "ratio_mean": ratio_mean,
                    }
                },
            }
        ],
    }


def test_build_stability_report_marks_single_run_stable(tmp_path: Path) -> None:
    mod = _load_module()
    runs = tmp_path / "parity_runs.jsonl"
    current = tmp_path / "equivalence_report.json"
    _write_runs(runs)
    current.write_text(
        json.dumps(_report(parity_pass_final=True, ratio_mean=0.1)), encoding="utf-8"
    )

    report = mod.build_stability_report(
        runs_path=runs,
        equivalence_report_path=current,
    )

    assert report["status"] == "stable"
    assert report["rerun_consistency"]["mode"] == "single_run"
    assert report["rerun_consistency"]["verdict_flip_detected"] is False


def test_build_stability_report_detects_verdict_flip_with_history(tmp_path: Path) -> None:
    mod = _load_module()
    runs = tmp_path / "parity_runs.jsonl"
    current = tmp_path / "equivalence_report.json"
    history = tmp_path / "history_equivalence_report.json"
    _write_runs(runs)
    current.write_text(
        json.dumps(_report(parity_pass_final=True, ratio_mean=0.1)), encoding="utf-8"
    )
    history.write_text(
        json.dumps(_report(parity_pass_final=False, ratio_mean=0.1)), encoding="utf-8"
    )

    report = mod.build_stability_report(
        runs_path=runs,
        equivalence_report_path=current,
        history_equivalence_report_paths=[history],
    )

    assert report["status"] == "unstable"
    assert report["rerun_consistency"]["mode"] == "multi_run"
    assert report["rerun_consistency"]["verdict_flip_detected"] is True


def test_build_stability_report_detects_metric_sign_flip_with_history(tmp_path: Path) -> None:
    mod = _load_module()
    runs = tmp_path / "parity_runs.jsonl"
    current = tmp_path / "equivalence_report.json"
    history = tmp_path / "history_equivalence_report.json"
    _write_runs(runs)
    current.write_text(
        json.dumps(_report(parity_pass_final=True, ratio_mean=0.1)), encoding="utf-8"
    )
    history.write_text(
        json.dumps(_report(parity_pass_final=True, ratio_mean=-0.2)), encoding="utf-8"
    )

    report = mod.build_stability_report(
        runs_path=runs,
        equivalence_report_path=current,
        history_equivalence_report_paths=[history],
    )

    assert report["status"] == "unstable"
    assert report["rerun_consistency"]["pairwise_metric_sign_flip_detected"] is True
