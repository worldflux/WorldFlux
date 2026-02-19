"""Tests for parity validity gate behavior in proof mode."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "validity_gate.py"
    spec = importlib.util.spec_from_file_location("validity_gate", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load validity_gate")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _entry(system: str, *, random_policy: bool) -> dict:
    metadata = {
        "mode": "native_real_env",
        "env_backend": "gymnasium",
        "eval_protocol_hash": "abc123",
    }
    if system == "worldflux":
        metadata.update(
            {
                "policy_mode": "parity_candidate",
                "policy_impl": "random_env_sampler" if random_policy else "candidate_actor",
                "policy": "random" if random_policy else "learned",
            }
        )
    else:
        metadata.update(
            {
                "policy_mode": "official_reference",
                "policy_impl": "official_ref",
            }
        )

    return {
        "schema_version": "parity.v1",
        "task_id": "atari100k_pong",
        "family": "dreamerv3",
        "seed": 0,
        "system": system,
        "status": "success",
        "metrics": {
            "final_return_mean": 1.0,
            "auc_return": 1.0,
            "metadata": metadata,
        },
    }


def test_validity_gate_rejects_random_policy_in_proof_mode() -> None:
    mod = _load_module()
    entries = [
        _entry("official", random_policy=False),
        _entry("worldflux", random_policy=True),
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=True,
        required_policy_mode="parity_candidate",
        requirements={
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
            "forbidden_shortcuts": ["policy=random"],
        },
    )

    assert report["pass"] is False
    codes = {issue["code"] for issue in report["issues"]}
    assert "random_policy_forbidden" in codes


def test_validity_gate_allows_random_policy_outside_proof_mode() -> None:
    mod = _load_module()
    entries = [
        _entry("official", random_policy=False),
        _entry("worldflux", random_policy=True),
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=False,
        required_policy_mode="parity_candidate",
        requirements={
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
            "forbidden_shortcuts": [],
        },
    )

    assert report["pass"] is True


def test_validity_gate_uses_per_entry_validity_requirements_override() -> None:
    mod = _load_module()
    entries = [
        _entry("official", random_policy=False),
        {
            **_entry("worldflux", random_policy=False),
            "validity_requirements": {
                "policy_mode": "parity_candidate",
                "environment_backend": "gymnasium",
                "forbidden_shortcuts": ["policy=random"],
            },
        },
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=True,
        required_policy_mode="diagnostic_random",
        requirements={
            "policy_mode": "diagnostic_random",
            "environment_backend": "auto",
            "forbidden_shortcuts": [],
        },
    )

    assert report["pass"] is True
