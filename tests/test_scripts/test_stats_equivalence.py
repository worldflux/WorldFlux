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
        "adapter": "dummy",
        "status": "success",
        "metrics": metric_payload,
    }


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
            "eval_protocol_hash": "abc123",
            "policy_mode": "official_reference",
            "policy_impl": "official_ref",
        }
        metadata_worldflux = {
            "mode": "native_real_env",
            "eval_protocol_hash": "abc123",
            "policy_mode": "parity_candidate",
            "policy_impl": "cem_planner",
            "policy": "learned",
            "env_backend": "dmcontrol",
        }
        rows.append(
            _record(
                "dog-run",
                seed,
                "official",
                100.0 + seed,
                90.0 + seed,
                metadata=metadata_official,
            )
        )
        rows.append(
            _record(
                "dog-run",
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
        expected_returncode=0,
    )

    assert report["global"]["strict_mode_failed"] is False
    assert report["global"]["validity_pass"] is True
    assert report["validity"]["pass"] is True


def test_stats_equivalence_strict_validity_infers_legacy_official_backend_with_manifest(
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
            "eval_protocol_hash": "abc123",
            "policy_mode": "official_reference",
            "policy_impl": "official_ref",
        }
        metadata_worldflux = {
            "mode": "native_real_env",
            "eval_protocol_hash": "abc123",
            "policy_mode": "parity_candidate",
            "policy_impl": "candidate_actor",
            "policy": "learned",
            "env_backend": "gymnasium",
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
        rows.append(worldflux_row)
    runs_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    report = _run_stats(
        runs_path,
        tmp_path / "report.json",
        extra_args=[
            "--strict-validity",
            "--proof-mode",
            "--manifest",
            "scripts/parity/manifests/preseed_demo_v1.yaml",
        ],
        expected_returncode=0,
    )

    assert report["global"]["strict_mode_failed"] is False
    assert report["global"]["validity_pass"] is True
    assert report["validity"]["pass"] is True
    assert report["validity"]["issue_count"] == 0
    assert report["validity"]["info_count"] >= 1
    info_codes = {
        issue["code"]
        for issue in report["validity"]["issues"]
        if str(issue.get("severity", "error")) == "info"
    }
    assert "official_env_backend_inferred" in info_codes


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
