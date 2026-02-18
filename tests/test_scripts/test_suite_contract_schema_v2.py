"""Tests for parity suite contract v2 schema."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "contract_schema.py"
    spec = importlib.util.spec_from_file_location("contract_schema", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load contract_schema")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _valid_v2() -> dict:
    return {
        "schema_version": "parity.suite.v2",
        "suite_id": "dreamerv3.atari100k",
        "family": "dreamerv3",
        "primary_metric": "final_return_mean",
        "secondary_metrics": ["auc_return"],
        "higher_is_better": True,
        "effect_transform": "paired_log_ratio",
        "equivalence_margin": 0.05,
        "noninferiority_margin": 0.05,
        "alpha": 0.05,
        "holm_scope": "all_metrics",
        "seed_policy": {
            "mode": "fixed",
            "values": [0, 1],
            "pilot_seeds": 10,
            "min_seeds": 20,
            "max_seeds": 50,
            "power_target": 0.8,
        },
        "train_budget": {"steps": 100000},
        "eval_protocol": {
            "eval_interval": 5000,
            "eval_episodes": 4,
            "eval_window": 10,
        },
        "validity_requirements": {
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
            "forbidden_shortcuts": ["mode=mock", "policy=random"],
        },
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                    "source": {
                        "commit": "b65cf81a6fb13625af8722127459283f899a35d9",
                        "artifact_path": "dreamerv3-official/logs",
                    },
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                    "source": {
                        "commit": "deadbeef",
                        "artifact_path": "reports/parity",
                    },
                },
            }
        ],
    }


def test_load_suite_contract_v2_accepts_valid_payload() -> None:
    mod = _load_module()
    contract = mod.load_suite_contract(_valid_v2())

    assert contract.schema_version == "parity.suite.v2"
    assert contract.suite_id == "dreamerv3.atari100k"
    assert len(contract.tasks) == 1
    assert contract.tasks[0].official.source is not None
    assert contract.tasks[0].worldflux.source is not None


def test_load_suite_contract_v2_requires_validity_requirements() -> None:
    mod = _load_module()
    payload = _valid_v2()
    del payload["validity_requirements"]["policy_mode"]

    with pytest.raises(RuntimeError, match="validity_requirements.policy_mode"):
        mod.load_suite_contract(payload)
