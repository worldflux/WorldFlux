# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
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


def _entry(system: str, *, random_policy: bool, env_backend: str | None = "gymnasium") -> dict:
    backend_kind = "jax_subprocess"
    adapter_id = (
        "official_dreamerv3_jax_subprocess"
        if system == "official"
        else "worldflux_dreamerv3_jax_subprocess"
    )
    metadata = {
        "mode": "worldflux_jax" if system == "worldflux" else "official",
        "eval_protocol_hash": "abc123",
        "backend_kind": backend_kind,
        "adapter_id": adapter_id,
        "recipe_hash": "recipe123",
        "artifact_manifest": {
            "backend_kind": backend_kind,
            "adapter_id": adapter_id,
            "recipe_hash": "recipe123",
            "checkpoint_paths": [],
            "score_paths": [],
            "metrics_paths": ["metrics.json"],
        },
        "model_id": "dreamerv3:official_xl",
        "model_profile": "official_xl",
        "train_budget": {
            "steps": 100,
            "warmup_steps": 1,
            "buffer_capacity": 64,
            "sequence_length": 2,
            "batch_size": 2,
            "replay_ratio": 1,
            "train_chunk_size": 1,
            "max_episode_steps": 4,
        }
        if system == "worldflux"
        else {"steps": 100},
        "eval_protocol": {
            "eval_interval": 6,
            "eval_episodes": 1,
            "eval_window": 2,
        },
    }
    if env_backend is not None:
        metadata["env_backend"] = env_backend
    if system == "worldflux":
        metadata.update(
            {
                "policy_mode": "parity_candidate",
                "policy_impl": "random_env_sampler"
                if random_policy
                else "candidate_actor_stateful",
                "policy": "random" if random_policy else "learned",
                "strict_official_semantics": not random_policy,
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
        "source_commit": "sha",
        "source_artifact_path": "artifact",
        "train_budget": {
            "steps": 100,
            "warmup_steps": 1,
            "buffer_capacity": 64,
            "sequence_length": 2,
            "batch_size": 2,
            "replay_ratio": 1,
            "train_chunk_size": 1,
            "max_episode_steps": 4,
        },
        "eval_protocol": {
            "eval_interval": 6,
            "eval_episodes": 1,
            "eval_window": 2,
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
        },
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
            "metrics": {
                **_entry("worldflux", random_policy=False)["metrics"],
                "metadata": {
                    **_entry("worldflux", random_policy=False)["metrics"]["metadata"],
                    "strict_official_semantics": True,
                },
            },
            "source_commit": "wf-sha",
            "source_artifact_path": "wf-artifact",
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


def test_validity_gate_requires_explicit_official_backend_in_proof_mode() -> None:
    mod = _load_module()
    entries = [
        _entry("official", random_policy=False, env_backend=None),
        _entry("worldflux", random_policy=False, env_backend="gymnasium"),
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=True,
        required_policy_mode="parity_candidate",
        requirements={
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
            "forbidden_shortcuts": [],
        },
    )

    assert report["pass"] is False
    codes = {issue["code"] for issue in report["issues"]}
    assert "missing_env_backend" in codes


def test_validity_gate_rejects_explicit_backend_mismatch_for_official() -> None:
    mod = _load_module()
    entries = [
        _entry("official", random_policy=False, env_backend="stub"),
        _entry("worldflux", random_policy=False, env_backend="gymnasium"),
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=True,
        required_policy_mode="parity_candidate",
        requirements={
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
            "forbidden_shortcuts": [],
        },
    )

    assert report["pass"] is False
    assert report["issue_count"] >= 1
    codes = {issue["code"] for issue in report["issues"]}
    assert "environment_backend_mismatch" in codes


def test_validity_gate_rejects_dreamer_without_strict_semantics() -> None:
    mod = _load_module()
    entries = [
        _entry("official", random_policy=False, env_backend="gymnasium"),
        {
            **_entry("worldflux", random_policy=False, env_backend="gymnasium"),
            "metrics": {
                **_entry("worldflux", random_policy=False, env_backend="gymnasium")["metrics"],
                "metadata": {
                    **_entry("worldflux", random_policy=False, env_backend="gymnasium")["metrics"][
                        "metadata"
                    ],
                    "strict_official_semantics": False,
                },
            },
        },
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=True,
        required_policy_mode="parity_candidate",
        requirements={
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
            "forbidden_shortcuts": [],
        },
    )

    assert report["pass"] is False
    codes = {issue["code"] for issue in report["issues"]}
    assert "strict_semantics_required" in codes


def test_validity_gate_requires_tdmpc2_alignment_metadata_in_proof_mode() -> None:
    mod = _load_module()
    entries = [
        {
            "schema_version": "parity.v1",
            "task_id": "dog-run",
            "family": "tdmpc2",
            "seed": 0,
            "system": "official",
            "status": "success",
            "source_commit": "sha",
            "source_artifact_path": "artifact",
            "train_budget": {"steps": 100},
            "eval_protocol": {
                "eval_interval": 6,
                "eval_episodes": 1,
                "eval_window": 2,
                "environment_backend": "dmcontrol",
            },
            "metrics": {
                "final_return_mean": 1.0,
                "auc_return": 1.0,
                "metadata": {
                    "mode": "official",
                    "env_backend": "dmcontrol",
                    "backend_kind": "torch_subprocess",
                    "adapter_id": "official_tdmpc2_torch_subprocess",
                    "recipe_hash": "recipe123",
                    "artifact_manifest": {"metrics_paths": ["metrics.json"]},
                    "model_id": "tdmpc2:5m",
                    "model_profile": "5m",
                    "canonical_compare_profile": "proof_5m",
                    "official_model_size": 5,
                    "train_budget": {"steps": 100},
                    "eval_protocol": {"eval_interval": 6, "eval_episodes": 1, "eval_window": 2},
                    "eval_protocol_hash": "abc123",
                    "policy_mode": "official_reference",
                    "policy_impl": "official_tdmpc2_reference",
                },
            },
        },
        {
            "schema_version": "parity.v1",
            "task_id": "dog-run",
            "family": "tdmpc2",
            "seed": 0,
            "system": "worldflux",
            "status": "success",
            "source_commit": "sha",
            "source_artifact_path": "artifact",
            "train_budget": {
                "steps": 100,
                "warmup_steps": 1,
                "buffer_capacity": 64,
                "sequence_length": 2,
                "batch_size": 2,
                "replay_ratio": 1,
                "train_chunk_size": 1,
                "max_episode_steps": 4,
            },
            "eval_protocol": {
                "eval_interval": 6,
                "eval_episodes": 1,
                "eval_window": 2,
                "environment_backend": "dmcontrol",
                "policy_mode": "parity_candidate",
            },
            "metrics": {
                "final_return_mean": 1.0,
                "auc_return": 1.0,
                "metadata": {
                    "mode": "native_real_env",
                    "env_backend": "dmcontrol",
                    "backend_kind": "native_torch",
                    "adapter_id": "worldflux_tdmpc2_native_torch",
                    "recipe_hash": "recipe123",
                    "artifact_manifest": {"metrics_paths": ["metrics.json"]},
                    "model_id": "tdmpc2:proof_5m",
                    "model_profile": "proof_5m",
                    "canonical_compare_profile": "proof_5m",
                    "train_budget": {
                        "steps": 100,
                        "warmup_steps": 1,
                        "buffer_capacity": 64,
                        "sequence_length": 2,
                        "batch_size": 2,
                        "replay_ratio": 1,
                        "train_chunk_size": 1,
                        "max_episode_steps": 4,
                    },
                    "eval_protocol": {
                        "eval_interval": 6,
                        "eval_episodes": 1,
                        "eval_window": 2,
                    },
                    "eval_protocol_hash": "abc123",
                    "policy_mode": "parity_candidate",
                    "policy_impl": "cem_planner",
                },
            },
        },
    ]

    report = mod.evaluate_validity(
        entries,
        proof_mode=True,
        required_policy_mode="parity_candidate",
        requirements={
            "policy_mode": "parity_candidate",
            "environment_backend": "dmcontrol",
            "forbidden_shortcuts": [],
        },
    )

    assert report["pass"] is False
    codes = {issue["code"] for issue in report["issues"]}
    assert "alignment_report_missing" in codes
    assert "alignment_status_mismatch" in codes
