# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Source-of-truth policy flags and status helpers for execution routing."""

from __future__ import annotations

import json
import os
from pathlib import Path

from .contracts import BackendExecutionRequest, BackendExecutionResult

TDMPC2_COMPARE_ENABLED = False
DREAMER_MIN_LOCKED_SEEDS = 10
DREAMER_MIN_PROOF_SEEDS = 20


def blocked_result(
    request: BackendExecutionRequest,
    *,
    reason_code: str,
    message: str,
    next_action: str | None,
    manifest_path: str | None = None,
) -> BackendExecutionResult:
    return BackendExecutionResult(
        status="blocked",
        reason_code=reason_code,  # type: ignore[arg-type]
        message=message,
        backend=request.backend,
        family=request.family,
        mode=request.mode,
        proof_phase="compare" if request.mode in {"verify", "proof_compare"} else "official_only",
        profile=request.profile,
        run_id=request.run_id,
        manifest_path=manifest_path,
        next_action=next_action,
    )


def incomplete_result(
    request: BackendExecutionRequest,
    *,
    reason_code: str,
    message: str,
    next_action: str | None,
    manifest_path: str | None = None,
    metrics: dict[str, object] | None = None,
) -> BackendExecutionResult:
    return BackendExecutionResult(
        status="incomplete",
        reason_code=reason_code,  # type: ignore[arg-type]
        message=message,
        backend=request.backend,
        family=request.family,
        mode=request.mode,
        proof_phase="compare" if request.mode in {"verify", "proof_compare"} else "official_only",
        profile=request.profile,
        run_id=request.run_id,
        manifest_path=manifest_path,
        metrics=metrics or {},
        next_action=next_action,
    )


def evaluate_seed_gates(request: BackendExecutionRequest) -> BackendExecutionResult | None:
    seed_count = len(request.seed_list)
    if request.family != "dreamer":
        return None
    if request.mode == "proof_bootstrap" and seed_count < DREAMER_MIN_LOCKED_SEEDS:
        return incomplete_result(
            request,
            reason_code="dreamer_bootstrap_incomplete",
            message=(
                f"Dreamer official bootstrap requires at least {DREAMER_MIN_LOCKED_SEEDS} seeds; "
                f"got {seed_count}."
            ),
            next_action=f"Re-run with at least {DREAMER_MIN_LOCKED_SEEDS} seeds.",
            metrics={
                "seed_count": seed_count,
                "minimum_locked_seeds": DREAMER_MIN_LOCKED_SEEDS,
            },
        )
    if request.mode in {"verify", "proof_compare"} and seed_count < DREAMER_MIN_PROOF_SEEDS:
        return incomplete_result(
            request,
            reason_code="minimum_proof_not_reached",
            message=(
                f"Dreamer proof compare requires at least {DREAMER_MIN_PROOF_SEEDS} seeds; "
                f"got {seed_count}."
            ),
            next_action=f"Re-run with at least {DREAMER_MIN_PROOF_SEEDS} seeds.",
            metrics={
                "seed_count": seed_count,
                "minimum_proof_seeds": DREAMER_MIN_PROOF_SEEDS,
            },
        )
    return None


def evaluate_family_policy(request: BackendExecutionRequest) -> BackendExecutionResult | None:
    if request.family == "tdmpc2" and request.mode in {"verify", "proof_compare"}:
        report_path = _resolve_tdmpc2_alignment_report(request)
        report_status = _tdmpc2_alignment_status(report_path)
        if request.profile and str(request.profile).strip().lower() != "proof_5m":
            return blocked_result(
                request,
                reason_code="tdmpc2_architecture_mismatch_open",
                message="TD-MPC2 compare requires backend_profile='proof_5m'.",
                next_action="Use tdmpc2:proof_5m / backend_profile=proof_5m for compare.",
                manifest_path=str(report_path) if report_path is not None else None,
            )
        if report_status == "mismatched":
            return blocked_result(
                request,
                reason_code="tdmpc2_architecture_mismatch_open",
                message=(
                    "TD-MPC2 compare is blocked because the latest alignment report is mismatched."
                ),
                next_action=(
                    "Regenerate a passing TD-MPC2 alignment report or use the canonical proof_5m recipe."
                ),
                manifest_path=str(report_path) if report_path is not None else None,
            )
        if report_status != "aligned":
            return blocked_result(
                request,
                reason_code="tdmpc2_architecture_mismatch_open",
                message=("TD-MPC2 compare requires an aligned proof_5m alignment report."),
                next_action=(
                    "Generate or refresh a passing TD-MPC2 alignment report before proof-grade comparison."
                ),
                manifest_path=str(report_path) if report_path is not None else None,
            )
    return None


def _resolve_tdmpc2_alignment_report(request: BackendExecutionRequest) -> Path | None:
    configured = str(request.proof_requirements.get("tdmpc2_alignment_report_path", "")).strip()
    if configured:
        return Path(configured).expanduser().resolve()
    override = os.getenv("WORLDFLUX_TDMPC2_ALIGNMENT_REPORT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    candidates = sorted(
        Path("reports/parity").glob("tdmpc2_alignment/**/*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0].resolve() if candidates else None


def _tdmpc2_alignment_status(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("status", "")).strip().lower()
