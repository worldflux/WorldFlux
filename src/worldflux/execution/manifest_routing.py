# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Single-source manifest routing for verify/parity execution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .backend_policy import evaluate_family_policy, evaluate_seed_gates
from .contracts import BackendExecutionRequest, BackendExecutionResult


@dataclass(frozen=True)
class ManifestResolution:
    manifest_path: Path | None
    early_result: BackendExecutionResult | None = None


def resolve_execution_manifest(
    request: BackendExecutionRequest,
    *,
    scripts_root: Path,
    allow_official_only: bool = False,
) -> ManifestResolution:
    family_policy = evaluate_family_policy(request)
    if family_policy is not None:
        return ManifestResolution(manifest_path=None, early_result=family_policy)

    seed_gate = evaluate_seed_gates(request)
    if seed_gate is not None:
        return ManifestResolution(manifest_path=None, early_result=seed_gate)

    override = os.getenv("WORLDFLUX_VERIFY_MANIFEST", "").strip()
    if override:
        override_path = Path(override).expanduser()
        return ManifestResolution(manifest_path=override_path.resolve())

    if request.manifest_path:
        return ManifestResolution(manifest_path=Path(request.manifest_path).expanduser().resolve())

    manifests_root = scripts_root / "manifests"
    if request.family == "dreamer":
        if request.mode == "proof_bootstrap" or allow_official_only:
            return ManifestResolution(
                manifest_path=(
                    manifests_root / "dreamerv3_official_checkpoint_bootstrap_v1.json"
                ).resolve()
            )
        return ManifestResolution(
            manifest_path=(manifests_root / "official_vs_worldflux_full_v2.yaml").resolve()
        )

    if request.family == "tdmpc2":
        return ManifestResolution(
            manifest_path=(manifests_root / "official_vs_worldflux_full_v2.yaml").resolve(),
        )

    return ManifestResolution(
        manifest_path=None,
        early_result=BackendExecutionResult(
            status="blocked",
            reason_code="task_family_unsupported",
            message=f"Unsupported family for execution routing: {request.family}",
            backend=request.backend,
            family=request.family,
            mode=request.mode,
            proof_phase="compare" if request.mode == "proof_compare" else "official_only",
            profile=request.profile,
            run_id=request.run_id,
            next_action="Use dreamer or tdmpc2 family.",
        ),
    )
