"""Tests for backend-native execution routing."""

from __future__ import annotations

import json
from pathlib import Path

from worldflux.execution import (
    BackendExecutionRequest,
    ParityBackedExecutor,
    normalize_distributed_proof_summary,
    normalize_dreamer_batch_status,
    normalize_dreamer_official_batch_summary,
    normalize_parity_run_row,
    resolve_execution_manifest,
)


def test_resolve_execution_manifest_dreamer_bootstrap(tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts" / "parity"
    (scripts_root / "manifests").mkdir(parents=True, exist_ok=True)
    request = BackendExecutionRequest(
        backend="official_dreamerv3_jax_subprocess",
        family="dreamer",
        mode="proof_bootstrap",
        target="m.pt",
        baseline="official/dreamerv3",
        task_filter="atari100k_pong",
        env="atari/pong",
        seed_list=list(range(10)),
        device="cpu",
        proof_requirements={"allow_official_only": True},
    )
    resolution = resolve_execution_manifest(
        request,
        scripts_root=scripts_root,
        allow_official_only=True,
    )
    assert resolution.early_result is None
    assert resolution.manifest_path is not None
    assert resolution.manifest_path.name == "dreamerv3_official_checkpoint_bootstrap_v1.json"


def test_resolve_execution_manifest_dreamer_compare_incomplete_before_20_seeds(
    tmp_path: Path,
) -> None:
    scripts_root = tmp_path / "scripts" / "parity"
    (scripts_root / "manifests").mkdir(parents=True, exist_ok=True)
    request = BackendExecutionRequest(
        backend="official_dreamerv3_jax_subprocess",
        family="dreamer",
        mode="proof_compare",
        target="m.pt",
        baseline="official/dreamerv3",
        task_filter="atari100k_pong",
        env="atari/pong",
        seed_list=[0, 1, 2],
        device="cpu",
    )
    resolution = resolve_execution_manifest(request, scripts_root=scripts_root)
    assert resolution.manifest_path is None
    assert resolution.early_result is not None
    assert resolution.early_result.status == "incomplete"
    assert resolution.early_result.reason_code == "minimum_proof_not_reached"


def test_resolve_execution_manifest_tdmpc2_compare_blocked(tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts" / "parity"
    (scripts_root / "manifests").mkdir(parents=True, exist_ok=True)
    request = BackendExecutionRequest(
        backend="official_tdmpc2_torch_subprocess",
        family="tdmpc2",
        mode="proof_compare",
        target="m.pt",
        baseline="official/tdmpc2",
        task_filter="walker-run",
        env="dmcontrol/walker-run",
        seed_list=list(range(20)),
        device="cpu",
    )
    resolution = resolve_execution_manifest(request, scripts_root=scripts_root)
    assert resolution.manifest_path is None
    assert resolution.early_result is not None
    assert resolution.early_result.status == "blocked"
    assert resolution.early_result.reason_code == "tdmpc2_architecture_mismatch_open"


def test_resolve_execution_manifest_tdmpc2_compare_allowed_when_alignment_report_passes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_root = tmp_path / "scripts" / "parity"
    manifests_root = scripts_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_root / "official_vs_worldflux_full_v2.yaml"
    manifest_path.write_text("schema_version: parity.suite.v2\n", encoding="utf-8")

    report_path = tmp_path / "alignment.json"
    report_path.write_text(json.dumps({"status": "aligned"}), encoding="utf-8")
    monkeypatch.setenv("WORLDFLUX_TDMPC2_ALIGNMENT_REPORT", str(report_path))

    request = BackendExecutionRequest(
        backend="official_tdmpc2_torch_subprocess",
        family="tdmpc2",
        mode="proof_compare",
        target="m.pt",
        baseline="official/tdmpc2",
        task_filter="walker-run",
        env="dmcontrol/walker-run",
        seed_list=list(range(20)),
        device="cpu",
        profile="proof_5m",
    )
    resolution = resolve_execution_manifest(request, scripts_root=scripts_root)
    assert resolution.early_result is None
    assert resolution.manifest_path == manifest_path.resolve()


def test_normalize_dreamer_official_batch_summary_marks_incomplete_below_locked_minimum(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.json"
    payload = {
        "total_seeds": 5,
        "success_count": 5,
        "completed_seeds": [0, 1, 2, 3, 4],
        "failed_seeds": [],
        "stalled_seeds": [],
    }
    result = normalize_dreamer_official_batch_summary(payload, summary_path=summary_path)
    assert result.status == "incomplete"
    assert result.reason_code == "dreamer_bootstrap_incomplete"
    assert result.proof_phase == "official_only"


def test_normalize_dreamer_official_batch_summary_requires_artifact_gates(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    payload = {
        "total_seeds": 10,
        "success_count": 10,
        "completed_seeds": list(range(10)),
        "failed_seeds": [],
        "stalled_seeds": [],
        "required_artifact_complete_count": 9,
        "component_match_present_count": 10,
        "artifact_manifest_present_count": 10,
        "baseline_drift_zero_count": 10,
    }
    result = normalize_dreamer_official_batch_summary(payload, summary_path=summary_path)
    assert result.status == "incomplete"
    assert result.reason_code == "dreamer_bootstrap_incomplete"


def test_normalize_dreamer_batch_status_success(tmp_path: Path) -> None:
    status_path = tmp_path / "status.json"
    payload = {
        "seed": 3,
        "state": "success",
        "attempt": 1,
        "exit_code": 0,
        "required_artifacts_complete": True,
        "baseline_drift_zero": True,
        "component_match_present": True,
        "artifact_manifest_present": True,
    }
    result = normalize_dreamer_batch_status(payload, status_path=status_path)
    assert result.status == "succeeded"
    assert result.reason_code == "none"
    assert result.proof_phase == "official_only"


def test_normalize_parity_run_row_success() -> None:
    row = {
        "status": "success",
        "system": "official",
        "backend_kind": "jax_subprocess",
        "family": "dreamer",
        "task_id": "atari100k_pong",
        "seed": 0,
        "metrics": {"final_return_mean": 1.0},
    }
    result = normalize_parity_run_row(row)
    assert result.status == "succeeded"
    assert result.metrics["task_id"] == "atari100k_pong"
    assert result.proof_phase == "compare"


def test_normalize_distributed_proof_summary_marks_success(tmp_path: Path) -> None:
    eq_json = tmp_path / "equivalence_report.json"
    eq_json.write_text(
        json.dumps({"global": {"parity_pass_final": True, "validity_pass": True}}),
        encoding="utf-8",
    )
    summary = {
        "run_id": "proof_run",
        "manifest": "/tmp/manifest.yaml",
        "failed_shards": 0,
        "coverage": {"pass": True, "missing_pairs": 0, "rerun_command": ""},
        "artifacts": {
            "equivalence_report": str(eq_json),
            "equivalence_markdown": str(tmp_path / "equivalence_report.md"),
        },
        "errors": {"stats_or_report": ""},
    }
    summary_path = tmp_path / "orchestrator_summary.json"
    result = normalize_distributed_proof_summary(summary, summary_path=summary_path)
    assert result.status == "succeeded"
    assert result.reason_code == "none"
    assert result.proof_phase == "compare"


def test_executor_blocks_dreamer_compare_without_bootstrap_summary(tmp_path: Path) -> None:
    scripts_root = tmp_path / "scripts" / "parity"
    (scripts_root / "manifests").mkdir(parents=True, exist_ok=True)
    executor = ParityBackedExecutor(repo_root=tmp_path, scripts_root=scripts_root)
    request = BackendExecutionRequest(
        backend="official_dreamerv3_jax_subprocess",
        family="dreamer",
        mode="proof_compare",
        target="m.pt",
        baseline="official/dreamerv3",
        task_filter="atari100k_pong",
        env="atari/pong",
        seed_list=list(range(20)),
        device="cpu",
    )
    result = executor.execute(request)
    assert result.status == "incomplete"
    assert result.reason_code == "dreamer_bootstrap_incomplete"
    assert result.proof_phase == "compare"


def test_executor_allows_dreamer_compare_when_bootstrap_summary_succeeded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scripts_root = tmp_path / "scripts" / "parity"
    manifests_root = scripts_root / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_root / "official_vs_worldflux_full_v2.yaml"
    manifest_path.write_text("schema_version: parity.suite.v2\n", encoding="utf-8")

    reports_root = tmp_path / "reports" / "parity" / "dreamer_official_bootstrap_20260311T000000Z"
    reports_root.mkdir(parents=True, exist_ok=True)
    summary_path = reports_root / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "usable_seed_count": 10,
                "execution_result": {
                    "status": "succeeded",
                    "metrics": {"success_count": 10},
                },
            }
        ),
        encoding="utf-8",
    )

    eq_json = tmp_path / "reports" / "parity" / "verify" / "verify_run" / "equivalence_report.json"
    eq_json.parent.mkdir(parents=True, exist_ok=True)
    eq_json.write_text(
        json.dumps({"global": {"parity_pass_final": True, "validity_pass": True}}),
        encoding="utf-8",
    )

    def _fake_run_subprocess(self, command: list[str]):
        del self
        if "run_parity_matrix.py" in " ".join(command):
            runs_jsonl = eq_json.parent / "parity_runs.jsonl"
            runs_jsonl.write_text("{}\n", encoding="utf-8")
        elif "stats_equivalence.py" in " ".join(command):
            pass
        elif "report_markdown.py" in " ".join(command):
            (eq_json.parent / "equivalence_report.md").write_text("# ok\n", encoding="utf-8")
        return type(
            "_Completed",
            (),
            {"returncode": 0, "stdout": "", "stderr": ""},
        )()

    monkeypatch.setattr(ParityBackedExecutor, "_run_subprocess", _fake_run_subprocess)

    executor = ParityBackedExecutor(repo_root=tmp_path, scripts_root=scripts_root)
    request = BackendExecutionRequest(
        backend="official_dreamerv3_jax_subprocess",
        family="dreamer",
        mode="proof_compare",
        target="m.pt",
        baseline="official/dreamerv3",
        task_filter="atari100k_pong",
        env="atari/pong",
        seed_list=list(range(20)),
        device="cpu",
        run_id="verify_run",
        output_root=str(tmp_path / "reports" / "parity" / "verify"),
    )
    result = executor.execute(request)
    assert result.status == "succeeded"
    assert result.proof_phase == "compare"
