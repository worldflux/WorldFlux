# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Contracts for backend-native execution flows."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

ExecutionMode = Literal["train", "verify", "proof_bootstrap", "proof_compare"]
ExecutionStatus = Literal["queued", "running", "blocked", "incomplete", "succeeded", "failed"]
ReasonCode = Literal[
    "none",
    "backend_unsupported",
    "manifest_missing",
    "task_family_unsupported",
    "tdmpc2_architecture_mismatch_open",
    "dreamer_bootstrap_incomplete",
    "minimum_proof_not_reached",
    "validity_failed",
    "stats_failed",
    "subprocess_failed",
    "artifact_missing",
    "user_configuration_error",
]


@dataclass(frozen=True)
class BackendExecutionRequest:
    backend: str
    family: str
    mode: ExecutionMode
    target: str | None
    baseline: str | None
    task_filter: str | None
    env: str | None
    seed_list: list[int]
    device: str
    profile: str | None = None
    manifest_path: str | None = None
    run_id: str | None = None
    output_root: str | None = None
    artifact_policy: str = "proof"
    require_component_match: bool = True
    proof_requirements: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not str(self.backend).strip():
            raise ValueError("backend must not be empty")
        if not str(self.family).strip():
            raise ValueError("family must not be empty")
        if not str(self.device).strip():
            raise ValueError("device must not be empty")
        if not self.seed_list:
            raise ValueError("seed_list must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BackendExecutionResult:
    status: ExecutionStatus
    reason_code: ReasonCode
    message: str
    backend: str
    family: str
    mode: ExecutionMode
    proof_phase: str | None = None
    profile: str | None = None
    run_id: str | None = None
    manifest_path: str | None = None
    summary_path: str | None = None
    equivalence_report_json: str | None = None
    equivalence_report_md: str | None = None
    stability_report_json: str | None = None
    evidence_bundle: str | None = None
    artifact_manifest: dict[str, object] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    next_action: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
