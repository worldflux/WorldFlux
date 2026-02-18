"""Tests for parity manifest schema parsing."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "parity" / "run_parity_matrix.py"
    )
    spec = importlib.util.spec_from_file_location("run_parity_matrix", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_parity_matrix module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _valid_manifest() -> dict:
    return {
        "schema_version": "parity.manifest.v1",
        "defaults": {"alpha": 0.05, "equivalence_margin": 0.05},
        "seed_policy": {
            "mode": "fixed",
            "values": [0, 1],
            "pilot_seeds": 10,
            "min_seeds": 20,
            "max_seeds": 50,
            "power_target": 0.8,
        },
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "family": "dreamerv3",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                },
            }
        ],
    }


def test_parse_manifest_accepts_valid_manifest() -> None:
    mod = _load_module()
    parsed = mod._parse_manifest(_valid_manifest())

    assert parsed.schema_version == "parity.manifest.v1"
    assert len(parsed.tasks) == 1
    assert parsed.tasks[0].task_id == "atari100k_pong"


def test_parse_manifest_requires_task_id() -> None:
    mod = _load_module()
    manifest = _valid_manifest()
    del manifest["tasks"][0]["task_id"]

    with pytest.raises(RuntimeError, match="task_id"):
        mod._parse_manifest(manifest)


def test_parse_manifest_rejects_unsupported_adapter() -> None:
    mod = _load_module()
    manifest = _valid_manifest()
    manifest["tasks"][0]["official"]["adapter"] = "unknown_adapter"

    with pytest.raises(RuntimeError, match="Unsupported adapter"):
        mod._parse_manifest(manifest)


def test_full_manifest_includes_65_tasks() -> None:
    mod = _load_module()
    full_manifest_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "manifests"
        / "official_vs_worldflux_full_v1.yaml"
    )
    parsed = mod._parse_manifest(mod._load_manifest(full_manifest_path))

    assert len(parsed.tasks) == 65
    families = {task.family for task in parsed.tasks}
    assert families == {"dreamerv3", "tdmpc2"}
    assert len({task.task_id for task in parsed.tasks}) == 65
