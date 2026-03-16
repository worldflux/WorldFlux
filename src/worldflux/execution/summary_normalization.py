"""Normalize runner/orchestrator summaries into BackendExecutionResult payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .backend_policy import DREAMER_MIN_LOCKED_SEEDS, DREAMER_MIN_PROOF_SEEDS
from .contracts import BackendExecutionResult, ExecutionMode


def normalize_parity_run_row(row: dict[str, Any]) -> BackendExecutionResult:
    status = str(row.get("status", "")).strip().lower() or "failed"
    system = str(row.get("system", "")).strip() or "unknown"
    backend = str(row.get("backend_kind", "")).strip() or system
    family = str(row.get("family", "")).strip() or "unknown"
    mode = cast(ExecutionMode, "proof_compare")
    task_id = str(row.get("task_id", "")).strip() or None
    seed = row.get("seed")
    metrics = dict(row.get("metrics", {})) if isinstance(row.get("metrics"), dict) else {}
    metrics.update(
        {
            "task_id": task_id,
            "seed": int(seed) if isinstance(seed, int) else seed,
            "system": system,
        }
    )
    if status == "success":
        return BackendExecutionResult(
            status="succeeded",
            reason_code="none",
            message="Parity run row completed successfully.",
            backend=backend,
            family=family,
            mode=mode,
            proof_phase="compare",
            metrics=metrics,
        )
    return BackendExecutionResult(
        status="failed",
        reason_code="subprocess_failed",
        message=f"Parity run row did not succeed (status={status}).",
        backend=backend,
        family=family,
        mode=mode,
        proof_phase="compare",
        metrics=metrics,
        next_action="Inspect parity row logs/artifacts for this task and seed.",
    )


def normalize_dreamer_batch_status(
    status_payload: dict[str, Any],
    *,
    status_path: Path | None = None,
) -> BackendExecutionResult:
    raw_state = str(status_payload.get("state", "")).strip().lower() or "planned"
    seed = status_payload.get("seed")
    metrics: dict[str, Any] = {
        "seed": int(seed) if isinstance(seed, int) else seed,
        "attempt": int(status_payload.get("attempt", 0) or 0),
        "exit_code": status_payload.get("exit_code"),
        "required_artifacts_complete": bool(
            status_payload.get("required_artifacts_complete", False)
        ),
        "baseline_drift_zero": bool(status_payload.get("baseline_drift_zero", False)),
        "component_match_present": bool(status_payload.get("component_match_present", False)),
        "artifact_manifest_present": bool(status_payload.get("artifact_manifest_present", False)),
    }
    summary_path = str(status_path.resolve()) if status_path is not None else None

    if raw_state == "success":
        return BackendExecutionResult(
            status="succeeded",
            reason_code="none",
            message="Dreamer official batch seed succeeded.",
            backend="official_dreamerv3_jax_subprocess",
            family="dreamer",
            mode="proof_bootstrap",
            proof_phase="official_only",
            summary_path=summary_path,
            metrics=metrics,
        )
    if raw_state == "running":
        return BackendExecutionResult(
            status="running",
            reason_code="none",
            message="Dreamer official batch seed is still running.",
            backend="official_dreamerv3_jax_subprocess",
            family="dreamer",
            mode="proof_bootstrap",
            proof_phase="official_only",
            summary_path=summary_path,
            metrics=metrics,
            next_action="Wait for completion or inspect monitor output.",
        )
    if raw_state in {"stalled", "planned"}:
        return BackendExecutionResult(
            status="incomplete",
            reason_code="dreamer_bootstrap_incomplete",
            message=f"Dreamer official batch seed is {raw_state}.",
            backend="official_dreamerv3_jax_subprocess",
            family="dreamer",
            mode="proof_bootstrap",
            proof_phase="official_only",
            summary_path=summary_path,
            metrics=metrics,
            next_action="Resume or rerun the incomplete seed.",
        )
    return BackendExecutionResult(
        status="failed",
        reason_code="subprocess_failed",
        message=str(status_payload.get("message", "")).strip()
        or "Dreamer official batch seed failed.",
        backend="official_dreamerv3_jax_subprocess",
        family="dreamer",
        mode="proof_bootstrap",
        proof_phase="official_only",
        summary_path=summary_path,
        metrics=metrics,
        next_action="Inspect runner logs and status payload for this seed.",
    )


def normalize_dreamer_official_batch_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path | None = None,
) -> BackendExecutionResult:
    total_seeds = int(summary.get("total_seeds", 0) or 0)
    success_count = int(summary.get("success_count", 0) or 0)
    failed_seeds = tuple(
        int(seed) for seed in summary.get("failed_seeds", []) if isinstance(seed, int)
    )
    stalled_seeds = tuple(
        int(seed) for seed in summary.get("stalled_seeds", []) if isinstance(seed, int)
    )
    completed_seeds = tuple(
        int(seed) for seed in summary.get("completed_seeds", []) if isinstance(seed, int)
    )

    metrics: dict[str, Any] = {
        "total_seeds": total_seeds,
        "success_count": success_count,
        "completed_seeds": list(completed_seeds),
        "failed_seeds": list(failed_seeds),
        "stalled_seeds": list(stalled_seeds),
        "locked_minimum_met": success_count >= DREAMER_MIN_LOCKED_SEEDS,
        "proof_minimum_met": success_count >= DREAMER_MIN_PROOF_SEEDS,
        "required_artifact_complete_count": int(
            summary.get("required_artifact_complete_count", 0) or 0
        ),
        "component_match_present_count": int(summary.get("component_match_present_count", 0) or 0),
        "artifact_manifest_present_count": int(
            summary.get("artifact_manifest_present_count", 0) or 0
        ),
        "baseline_drift_zero_count": int(summary.get("baseline_drift_zero_count", 0) or 0),
    }
    artifact_gate_pass = (
        metrics["required_artifact_complete_count"] == success_count
        and metrics["component_match_present_count"] == success_count
        and metrics["artifact_manifest_present_count"] == success_count
        and metrics["baseline_drift_zero_count"] == success_count
    )

    if failed_seeds:
        return BackendExecutionResult(
            status="failed",
            reason_code="subprocess_failed",
            message="Dreamer official bootstrap batch finished with failed seeds.",
            backend="official_dreamerv3_jax_subprocess",
            family="dreamer",
            mode="proof_bootstrap",
            proof_phase="official_only",
            summary_path=str(summary_path.resolve()) if summary_path is not None else None,
            metrics=metrics,
            next_action="Inspect failed seeds and rerun the batch.",
        )

    if stalled_seeds or success_count < DREAMER_MIN_LOCKED_SEEDS or not artifact_gate_pass:
        return BackendExecutionResult(
            status="incomplete",
            reason_code="dreamer_bootstrap_incomplete",
            message=(
                "Dreamer official bootstrap has not yet locked the required 10 seeds "
                "with complete artifacts and zero baseline drift."
            ),
            backend="official_dreamerv3_jax_subprocess",
            family="dreamer",
            mode="proof_bootstrap",
            proof_phase="official_only",
            summary_path=str(summary_path.resolve()) if summary_path is not None else None,
            metrics=metrics,
            next_action=f"Reach at least {DREAMER_MIN_LOCKED_SEEDS} successful seeds.",
        )

    return BackendExecutionResult(
        status="succeeded",
        reason_code="none",
        message=(
            "Dreamer official bootstrap batch completed, locked the required seeds, "
            "and preserved artifacts/component match/baseline consistency."
        ),
        backend="official_dreamerv3_jax_subprocess",
        family="dreamer",
        mode="proof_bootstrap",
        proof_phase="official_only",
        summary_path=str(summary_path.resolve()) if summary_path is not None else None,
        metrics=metrics,
        next_action=f"Continue toward {DREAMER_MIN_PROOF_SEEDS}-seed proof compare.",
    )


def normalize_distributed_proof_summary(
    summary: dict[str, Any],
    *,
    summary_path: Path | None = None,
) -> BackendExecutionResult:
    artifacts = summary.get("artifacts", {}) if isinstance(summary.get("artifacts"), dict) else {}
    coverage = summary.get("coverage", {}) if isinstance(summary.get("coverage"), dict) else {}
    errors = summary.get("errors", {}) if isinstance(summary.get("errors"), dict) else {}
    failed_shards = int(summary.get("failed_shards", 0) or 0)
    equivalence_report_raw = str(artifacts.get("equivalence_report", "")).strip()
    coverage_pass = bool(coverage.get("pass", False))
    missing_pairs = int(coverage.get("missing_pairs", 0) or 0)
    stats_error = str(errors.get("stats_or_report", "")).strip()
    rerun_command = str(coverage.get("rerun_command", "")).strip()

    metrics: dict[str, Any] = {
        "failed_shards": failed_shards,
        "missing_pairs": missing_pairs,
        "coverage_pass": coverage_pass,
    }

    summary_path_value = str(summary_path.resolve()) if summary_path is not None else None

    if failed_shards > 0:
        return BackendExecutionResult(
            status="failed",
            reason_code="subprocess_failed",
            message="Distributed proof finished with failed shards.",
            backend="official_and_worldflux",
            family="mixed",
            mode="proof_compare",
            proof_phase="compare",
            run_id=str(summary.get("run_id", "")).strip() or None,
            manifest_path=str(summary.get("manifest", "")).strip() or None,
            summary_path=summary_path_value,
            metrics=metrics,
            next_action="Inspect failed shards before rerunning proof.",
        )

    if not coverage_pass or missing_pairs > 0:
        return BackendExecutionResult(
            status="incomplete",
            reason_code="artifact_missing",
            message="Distributed proof is incomplete because coverage/completeness did not pass.",
            backend="official_and_worldflux",
            family="mixed",
            mode="proof_compare",
            proof_phase="compare",
            run_id=str(summary.get("run_id", "")).strip() or None,
            manifest_path=str(summary.get("manifest", "")).strip() or None,
            summary_path=summary_path_value,
            metrics=metrics,
            next_action=rerun_command or "Rerun missing pairs and regenerate reports.",
        )

    if stats_error:
        return BackendExecutionResult(
            status="failed",
            reason_code="stats_failed",
            message=stats_error,
            backend="official_and_worldflux",
            family="mixed",
            mode="proof_compare",
            proof_phase="compare",
            run_id=str(summary.get("run_id", "")).strip() or None,
            manifest_path=str(summary.get("manifest", "")).strip() or None,
            summary_path=summary_path_value,
            metrics=metrics,
            next_action="Fix stats/report generation and rerun.",
        )

    if not equivalence_report_raw:
        return BackendExecutionResult(
            status="running",
            reason_code="none",
            message="Distributed proof has no equivalence report yet.",
            backend="official_and_worldflux",
            family="mixed",
            mode="proof_compare",
            proof_phase="compare",
            run_id=str(summary.get("run_id", "")).strip() or None,
            manifest_path=str(summary.get("manifest", "")).strip() or None,
            summary_path=summary_path_value,
            metrics=metrics,
            next_action="Wait for equivalence report generation.",
        )

    report_path = Path(equivalence_report_raw)
    if not report_path.exists():
        return BackendExecutionResult(
            status="failed",
            reason_code="artifact_missing",
            message=f"Equivalence report is missing: {report_path}",
            backend="official_and_worldflux",
            family="mixed",
            mode="proof_compare",
            proof_phase="compare",
            run_id=str(summary.get("run_id", "")).strip() or None,
            manifest_path=str(summary.get("manifest", "")).strip() or None,
            summary_path=summary_path_value,
            metrics=metrics,
            next_action="Rebuild proof reports.",
        )

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    global_block = report_payload.get("global", {}) if isinstance(report_payload, dict) else {}
    final_pass = (
        bool(global_block.get("parity_pass_final", False))
        if isinstance(global_block, dict)
        else False
    )
    validity_pass = (
        bool(global_block.get("validity_pass", False)) if isinstance(global_block, dict) else False
    )
    metrics.update(
        {
            "parity_pass_final": final_pass,
            "validity_pass": validity_pass,
        }
    )

    if final_pass:
        return BackendExecutionResult(
            status="succeeded",
            reason_code="none",
            message="Distributed proof completed successfully.",
            backend="official_and_worldflux",
            family="mixed",
            mode="proof_compare",
            proof_phase="compare",
            run_id=str(summary.get("run_id", "")).strip() or None,
            manifest_path=str(summary.get("manifest", "")).strip() or None,
            summary_path=summary_path_value,
            equivalence_report_json=str(report_path.resolve()),
            equivalence_report_md=str(artifacts.get("equivalence_markdown", "")).strip() or None,
            evidence_bundle=str((summary_path.parent / "evidence_bundle.zip").resolve())
            if summary_path is not None
            else None,
            metrics=metrics,
            next_action=None,
        )

    return BackendExecutionResult(
        status="failed",
        reason_code="validity_failed",
        message="Distributed proof completed, but proof-grade validity/parity gates did not pass.",
        backend="official_and_worldflux",
        family="mixed",
        mode="proof_compare",
        proof_phase="compare",
        run_id=str(summary.get("run_id", "")).strip() or None,
        manifest_path=str(summary.get("manifest", "")).strip() or None,
        summary_path=summary_path_value,
        equivalence_report_json=str(report_path.resolve()),
        equivalence_report_md=str(artifacts.get("equivalence_markdown", "")).strip() or None,
        metrics=metrics,
        next_action="Inspect equivalence and validity reports.",
    )
