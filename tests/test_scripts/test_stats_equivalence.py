# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for parity statistical equivalence script."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _record(
    task_id: str,
    seed: int,
    system: str,
    final_return: float,
    auc_return: float,
    *,
    metadata: dict | None = None,
    adapter: str = "dummy",
) -> dict:
    metric_payload = {
        "final_return_mean": final_return,
        "auc_return": auc_return,
    }
    if metadata is not None:
        metric_payload["metadata"] = metadata

    return {
        "schema_version": "parity.v1",
        "run_id": "test",
        "task_id": task_id,
        "family": "dreamerv3",
        "seed": seed,
        "system": system,
        "adapter": adapter,
        "status": "success",
        "metrics": metric_payload,
    }


def _dreamer_backend_metadata(*, system: str, policy_impl: str, env_backend: str) -> dict:
    backend_kind = "jax_subprocess"
    adapter_id = (
        "official_dreamerv3_jax_subprocess"
        if system == "official"
        else "worldflux_dreamerv3_jax_subprocess"
    )
    return {
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
        "env_backend": env_backend,
        "model_id": "dreamerv3:official_xl",
        "model_profile": "official_xl",
        "strict_official_semantics": system != "official",
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
        if system != "official"
        else {"steps": 100},
        "eval_protocol": {"eval_interval": 6, "eval_episodes": 1, "eval_window": 2},
        "eval_protocol_hash": "abc123",
        "policy_mode": "parity_candidate" if system != "official" else "official_reference",
        "policy_impl": policy_impl,
        "policy": "learned" if system != "official" else "official",
    }


def _tdmpc2_backend_metadata(*, system: str, policy_impl: str, env_backend: str) -> dict:
    backend_kind = "torch_subprocess" if system == "official" else "native_torch"
    adapter_id = (
        "official_tdmpc2_torch_subprocess"
        if system == "official"
        else "worldflux_tdmpc2_native_torch"
    )
    model_id = "tdmpc2:5m" if system == "official" else "tdmpc2:proof_5m"
    model_profile = "5m" if system == "official" else "proof_5m"
    return {
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
        "env_backend": env_backend,
        "model_id": model_id,
        "model_profile": model_profile,
        "canonical_compare_profile": "proof_5m",
        "official_model_size": 5 if system == "official" else None,
        "alignment_report_path": "/tmp/tdmpc2_alignment_report.json"
        if system != "official"
        else "",
        "alignment_status": "aligned" if system != "official" else "",
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
        if system != "official"
        else {"steps": 100},
        "eval_protocol": {"eval_interval": 6, "eval_episodes": 1, "eval_window": 2},
        "eval_protocol_hash": "abc123",
        "policy_mode": "parity_candidate" if system != "official" else "official_reference",
        "policy_impl": policy_impl,
        "policy": "learned" if system != "official" else "official",
    }


def _suite_v2_manifest(
    path: Path, *, family: str = "dreamerv3", task_id: str = "atari100k_pong"
) -> Path:
    manifest = {
        "schema_version": "parity.suite.v2",
        "suite_id": "proof-suite",
        "family": family,
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
            "environment_backend": "gymnasium" if family == "dreamerv3" else "dmcontrol",
        },
        "validity_requirements": {
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium" if family == "dreamerv3" else "dmcontrol",
            "forbidden_shortcuts": ["mode=mock", "policy=random"],
        },
        "defaults": {
            "component_report_required": True,
        },
        "tasks": [
            {
                "task_id": task_id,
                "family": family,
                "official": {
                    "adapter": (
                        "official_dreamerv3_jax_subprocess"
                        if family == "dreamerv3"
                        else "official_tdmpc2_torch_subprocess"
                    ),
                    "backend_kind": "jax_subprocess"
                    if family == "dreamerv3"
                    else "torch_subprocess",
                    "artifact_requirements": {"metrics_paths": ["metrics.json"]},
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                    "source": {"commit": "official-sha", "artifact_path": "official-artifact"},
                },
                "worldflux": {
                    "adapter": (
                        "worldflux_dreamerv3_jax_subprocess"
                        if family == "dreamerv3"
                        else "worldflux_tdmpc2_native"
                    ),
                    "backend_kind": "jax_subprocess" if family == "dreamerv3" else "native_torch",
                    "artifact_requirements": {"metrics_paths": ["metrics.json"]},
                    "cwd": ".",
                    "command": ["python3", "-c", "print('ok')"],
                    "env": {},
                    "source": {"commit": "wf-sha", "artifact_path": "wf-artifact"},
                },
            }
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def _run_stats(
    input_path: Path,
    output_path: Path,
    *,
    extra_args: list[str] | None = None,
    expected_returncode: int = 0,
) -> dict:
    root = _repo_root()
    cmd = [
        "python3",
        "scripts/parity/stats_equivalence.py",
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--alpha",
        "0.05",
        "--equivalence-margin",
        "0.05",
        "--noninferiority-margin",
        "0.05",
        "--min-pairs",
        "2",
    ]
    if extra_args:
        cmd.extend(extra_args)
    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == expected_returncode, completed.stderr
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_stats_equivalence_passes_when_ratios_within_margin(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        official = 100.0 + seed * 0.1
        worldflux = official * 1.01
        rows.append(_record("atari100k_pong", seed, "official", official, official * 0.9))
        rows.append(_record("atari100k_pong", seed, "worldflux", worldflux, worldflux * 0.9))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(runs_path, tmp_path / "report.json")

    assert report["schema_version"] == "parity.v1"
    assert report["global"]["parity_pass_primary"] is True
    assert report["tasks"][0]["task_pass_primary"] is True


def test_stats_equivalence_fails_when_large_underperformance(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        official = 100.0 + seed * 0.1
        worldflux = official * 0.7
        rows.append(_record("dog-run", seed, "official", official, official * 0.9))
        rows.append(_record("dog-run", seed, "worldflux", worldflux, worldflux * 0.9))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(runs_path, tmp_path / "report.json")

    assert report["global"]["parity_pass_primary"] is False
    assert report["tasks"][0]["task_pass_primary"] is False
    metric = report["tasks"][0]["metrics"]["final_return_mean"]
    assert metric["tost"]["pass_raw"] is False


def test_stats_equivalence_applies_holm_correction_across_primary_hypotheses(
    tmp_path: Path,
) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []

    for seed in range(20):
        off_a = 100.0 + seed * 0.1
        wf_a = off_a * 1.01
        off_b = 100.0 + seed * 0.1
        wf_b = off_b * 1.06

        rows.append(_record("task_a", seed, "official", off_a, off_a))
        rows.append(_record("task_a", seed, "worldflux", wf_a, wf_a))
        rows.append(_record("task_b", seed, "official", off_b, off_b))
        rows.append(_record("task_b", seed, "worldflux", wf_b, wf_b))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(runs_path, tmp_path / "report.json")

    task_map = {task["task_id"]: task for task in report["tasks"]}

    assert task_map["task_a"]["metrics"]["final_return_mean"]["pass_with_holm_primary"] is True
    assert task_map["task_b"]["metrics"]["final_return_mean"]["pass_with_holm_primary"] is False
    assert report["global"]["parity_pass_primary"] is False


def test_stats_equivalence_strict_completeness_fails_on_missing_pair(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = [
        _record("task_x", 0, "official", 100.0, 90.0),
        _record("task_x", 1, "official", 101.0, 91.0),
        _record("task_x", 1, "worldflux", 101.0, 91.0),
    ]
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=["--strict-completeness"],
        expected_returncode=1,
    )

    assert report["global"]["strict_mode_failed"] is True
    assert report["completeness"]["missing_pairs"] == 1
    assert report["global"]["parity_pass_final"] is False


def test_stats_equivalence_strict_validity_fails_on_random_policy(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(2):
        metadata_official = {
            "mode": "official",
            "eval_protocol_hash": "abc123",
            "policy_mode": "official_reference",
            "policy_impl": "official_ref",
        }
        metadata_worldflux = {
            "mode": "native_real_env",
            "eval_protocol_hash": "abc123",
            "policy_mode": "parity_candidate",
            "policy_impl": "random_env_sampler",
            "policy": "random",
            "env_backend": "gymnasium",
        }
        rows.append(
            _record(
                "task_x",
                seed,
                "official",
                100.0 + seed,
                90.0 + seed,
                metadata=metadata_official,
            )
        )
        rows.append(
            _record(
                "task_x",
                seed,
                "worldflux",
                100.0 + seed,
                90.0 + seed,
                metadata=metadata_worldflux,
            )
        )
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=["--strict-validity", "--proof-mode"],
        expected_returncode=1,
    )

    assert report["global"]["strict_mode_failed"] is True
    assert report["global"]["validity_pass"] is False
    assert report["validity"]["pass"] is False


def test_stats_equivalence_strict_validity_passes_on_parity_candidate_policy(
    tmp_path: Path,
) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(2):
        metadata_official = {
            "mode": "official",
            **_tdmpc2_backend_metadata(
                system="official",
                policy_impl="official_ref",
                env_backend="dmcontrol",
            ),
        }
        metadata_worldflux = {
            "mode": "native_real_env",
            **_tdmpc2_backend_metadata(
                system="worldflux",
                policy_impl="cem_planner",
                env_backend="dmcontrol",
            ),
        }
        official_row = _record(
            "dog-run",
            seed,
            "official",
            100.0 + seed,
            90.0 + seed,
            metadata=metadata_official,
            adapter="official_tdmpc2_torch_subprocess",
        )
        official_row["family"] = "tdmpc2"
        official_row["source_commit"] = "official-sha"
        official_row["source_artifact_path"] = "official-artifact"
        official_row["train_budget"] = {
            "steps": 100,
            "warmup_steps": 1,
            "buffer_capacity": 64,
            "sequence_length": 2,
            "batch_size": 2,
            "replay_ratio": 1,
            "train_chunk_size": 1,
            "max_episode_steps": 4,
        }
        official_row["eval_protocol"] = {
            "eval_interval": 6,
            "eval_episodes": 1,
            "eval_window": 2,
            "policy_mode": "parity_candidate",
            "environment_backend": "dmcontrol",
        }
        rows.append(official_row)

        worldflux_row = _record(
            "dog-run",
            seed,
            "worldflux",
            100.0 + seed,
            90.0 + seed,
            metadata=metadata_worldflux,
            adapter="worldflux_tdmpc2_native_torch",
        )
        worldflux_row["family"] = "tdmpc2"
        worldflux_row["source_commit"] = "wf-sha"
        worldflux_row["source_artifact_path"] = "wf-artifact"
        worldflux_row["train_budget"] = dict(official_row["train_budget"])
        worldflux_row["eval_protocol"] = dict(official_row["eval_protocol"])
        rows.append(worldflux_row)
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest_path = _suite_v2_manifest(tmp_path / "suite.json", family="tdmpc2", task_id="dog-run")
    component_path = tmp_path / "component_match_report.json"
    component_path.write_text(
        json.dumps({"family": "tdmpc2", "all_pass": True, "results": []}), encoding="utf-8"
    )

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=[
            "--strict-validity",
            "--proof-mode",
            "--manifest",
            str(manifest_path),
            "--component-report",
            str(component_path),
        ],
        expected_returncode=0,
    )

    assert report["global"]["strict_mode_failed"] is False
    assert report["global"]["validity_pass"] is True
    assert report["validity"]["pass"] is True


def test_stats_equivalence_proof_mode_requires_explicit_official_backend(
    tmp_path: Path,
) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    validity_requirements = {
        "policy_mode": "parity_candidate",
        "environment_backend": "gymnasium",
        "forbidden_shortcuts": [],
    }
    for seed in range(2):
        metadata_official = {
            "mode": "official",
            **_dreamer_backend_metadata(
                system="official",
                policy_impl="official_ref",
                env_backend="",
            ),
        }
        metadata_worldflux = {
            "mode": "native_real_env",
            **_dreamer_backend_metadata(
                system="worldflux",
                policy_impl="candidate_actor_stateful",
                env_backend="gymnasium",
            ),
        }
        official_row = _record(
            "atari100k_pong",
            seed,
            "official",
            100.0 + seed,
            90.0 + seed,
            metadata=metadata_official,
        )
        official_row["validity_requirements"] = dict(validity_requirements)
        official_row["source_commit"] = "official-sha"
        official_row["source_artifact_path"] = "official-artifact"
        official_row["train_budget"] = {
            "steps": 100,
            "warmup_steps": 1,
            "buffer_capacity": 64,
            "sequence_length": 2,
            "batch_size": 2,
            "replay_ratio": 1,
            "train_chunk_size": 1,
            "max_episode_steps": 4,
        }
        official_row["eval_protocol"] = {
            "eval_interval": 6,
            "eval_episodes": 1,
            "eval_window": 2,
            "policy_mode": "parity_candidate",
            "environment_backend": "gymnasium",
        }
        rows.append(official_row)

        worldflux_row = _record(
            "atari100k_pong",
            seed,
            "worldflux",
            100.0 + seed,
            90.0 + seed,
            metadata=metadata_worldflux,
        )
        worldflux_row["validity_requirements"] = dict(validity_requirements)
        worldflux_row["source_commit"] = "wf-sha"
        worldflux_row["source_artifact_path"] = "wf-artifact"
        worldflux_row["train_budget"] = dict(official_row["train_budget"])
        worldflux_row["eval_protocol"] = dict(official_row["eval_protocol"])
        rows.append(worldflux_row)
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest_path = _suite_v2_manifest(tmp_path / "suite.json")
    component_path = tmp_path / "component_match_report.json"
    component_path.write_text(
        json.dumps({"family": "dreamerv3", "all_pass": True, "results": []}), encoding="utf-8"
    )

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=[
            "--strict-validity",
            "--proof-mode",
            "--manifest",
            str(manifest_path),
            "--component-report",
            str(component_path),
        ],
        expected_returncode=1,
    )

    assert report["global"]["strict_mode_failed"] is True
    assert report["global"]["validity_pass"] is False
    issue_codes = {issue["code"] for issue in report["validity"]["issues"]}
    assert "missing_env_backend" in issue_codes


def test_stats_equivalence_proof_mode_uses_suite_min_pairs(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(2):
        off = _record(
            "atari100k_pong",
            seed,
            "official",
            100.0 + seed,
            90.0 + seed,
            adapter="official_dreamerv3_jax_subprocess",
        )
        wf = _record(
            "atari100k_pong",
            seed,
            "worldflux",
            100.0 + seed,
            90.0 + seed,
            adapter="worldflux_dreamerv3_jax_subprocess",
        )
        for row, commit in ((off, "official-sha"), (wf, "wf-sha")):
            row["source_commit"] = commit
            row["source_artifact_path"] = f"{commit}-artifact"
            row["train_budget"] = {
                "steps": 100,
                "warmup_steps": 1,
                "buffer_capacity": 64,
                "sequence_length": 2,
                "batch_size": 2,
                "replay_ratio": 1,
                "train_chunk_size": 1,
                "max_episode_steps": 4,
            }
            row["eval_protocol"] = {
                "eval_interval": 6,
                "eval_episodes": 1,
                "eval_window": 2,
                "policy_mode": "parity_candidate",
                "environment_backend": "gymnasium",
            }
        off["metrics"]["metadata"] = {
            "mode": "official",
            **_dreamer_backend_metadata(
                system="official",
                policy_impl="official_ref",
                env_backend="gymnasium",
            ),
        }
        wf["metrics"]["metadata"] = {
            "mode": "worldflux_jax",
            **_dreamer_backend_metadata(
                system="worldflux",
                policy_impl="candidate_actor_stateful",
                env_backend="gymnasium",
            ),
        }
        rows.extend([off, wf])
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest_path = _suite_v2_manifest(tmp_path / "suite.json")
    component_path = tmp_path / "component_match_report.json"
    component_path.write_text(
        json.dumps({"family": "dreamerv3", "all_pass": True, "results": []}), encoding="utf-8"
    )

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=[
            "--strict-validity",
            "--proof-mode",
            "--manifest",
            str(manifest_path),
            "--component-report",
            str(component_path),
        ],
        expected_returncode=0,
    )

    assert report["config"]["min_pairs"] == 20
    metric = report["tasks"][0]["metrics"]["final_return_mean"]
    assert metric["status"] == "insufficient_pairs"


def test_stats_equivalence_proof_mode_requires_component_report(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        off = _record(
            "atari100k_pong",
            seed,
            "official",
            100.0 + seed,
            90.0 + seed,
            adapter="official_dreamerv3_jax_subprocess",
        )
        wf = _record(
            "atari100k_pong",
            seed,
            "worldflux",
            100.0 + seed,
            90.0 + seed,
            adapter="worldflux_dreamerv3_jax_subprocess",
        )
        for row, commit, system in (
            (off, "official-sha", "official"),
            (wf, "wf-sha", "worldflux"),
        ):
            row["source_commit"] = commit
            row["source_artifact_path"] = f"{commit}-artifact"
            row["train_budget"] = {
                "steps": 100,
                "warmup_steps": 1,
                "buffer_capacity": 64,
                "sequence_length": 2,
                "batch_size": 2,
                "replay_ratio": 1,
                "train_chunk_size": 1,
                "max_episode_steps": 4,
            }
            row["eval_protocol"] = {
                "eval_interval": 6,
                "eval_episodes": 1,
                "eval_window": 2,
                "policy_mode": "parity_candidate",
                "environment_backend": "gymnasium",
            }
            row["metrics"]["metadata"] = {
                "mode": "official" if system == "official" else "worldflux_jax",
                **_dreamer_backend_metadata(
                    system=system,
                    policy_impl=(
                        "official_ref" if system == "official" else "candidate_actor_stateful"
                    ),
                    env_backend="gymnasium",
                ),
            }
        rows.extend([off, wf])
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    manifest_path = _suite_v2_manifest(tmp_path / "suite.json")

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=[
            "--strict-validity",
            "--proof-mode",
            "--manifest",
            str(manifest_path),
        ],
        expected_returncode=1,
    )

    assert report["component_match"]["required"] is True
    assert report["component_match"]["present"] is False
    assert report["global"]["component_match_pass"] is False


def test_stats_equivalence_dual_pass_fails_when_bayesian_not_enabled(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        official = 100.0 + seed * 0.1
        worldflux = official * 1.01
        rows.append(_record("atari100k_pong", seed, "official", official, official * 0.9))
        rows.append(_record("atari100k_pong", seed, "worldflux", worldflux, worldflux * 0.9))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=["--dual-pass-required"],
    )

    assert report["global"]["parity_pass_all_metrics"] is True
    assert report["global"]["parity_pass_bayesian"] is None
    assert report["global"]["parity_pass_final"] is False


def test_stats_equivalence_bayesian_dual_pass_succeeds_for_good_data(tmp_path: Path) -> None:
    runs_path = tmp_path / "runs.jsonl"
    rows: list[dict] = []
    for seed in range(20):
        official = 100.0 + seed * 0.1
        ratio = 1.0 + ((seed % 4) - 1.5) * 0.005
        worldflux = official * ratio
        rows.append(_record("atari100k_pong", seed, "official", official, official * 0.9))
        rows.append(_record("atari100k_pong", seed, "worldflux", worldflux, worldflux * 0.9))

    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=["--bayes-enable", "--dual-pass-required", "--bayes-seed", "123"],
    )

    assert report["global"]["parity_pass_all_metrics"] is True
    assert report["global"]["parity_pass_bayesian"] is True
    assert report["global"]["parity_pass_final"] is True
    metric = report["tasks"][0]["metrics"]["final_return_mean"]
    assert metric["bayesian"]["status"] == "ok"
    assert metric["pass_with_bayesian"] is True
