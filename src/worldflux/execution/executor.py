"""Executor implementations for backend-native verification flows."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from .backend_policy import (
    DREAMER_MIN_LOCKED_SEEDS,
    _resolve_tdmpc2_alignment_report,
    _tdmpc2_alignment_status,
)
from .contracts import BackendExecutionRequest, BackendExecutionResult
from .manifest_routing import resolve_execution_manifest


class BackendExecutor(Protocol):
    def execute(self, request: BackendExecutionRequest) -> BackendExecutionResult: ...


class ParityBackedExecutor:
    """Executor backed by existing scripts/parity tooling."""

    def __init__(self, *, repo_root: Path, scripts_root: Path):
        self.repo_root = repo_root
        self.scripts_root = scripts_root

    @staticmethod
    def _load_json(path: Path) -> dict[str, object] | None:
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def _resolve_dreamer_bootstrap_summary_path(
        self, request: BackendExecutionRequest
    ) -> Path | None:
        configured = str(
            request.proof_requirements.get("dreamer_bootstrap_summary_path", "")
        ).strip()
        if configured:
            return Path(configured).expanduser().resolve()
        override = os.getenv("WORLDFLUX_DREAMER_BOOTSTRAP_SUMMARY", "").strip()
        if override:
            return Path(override).expanduser().resolve()
        reports_root = self.repo_root / "reports" / "parity"
        candidates = sorted(
            reports_root.glob("dreamer_official_bootstrap_*/summary.json"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        return candidates[0] if candidates else None

    def _validate_dreamer_bootstrap_ready(
        self,
        request: BackendExecutionRequest,
    ) -> BackendExecutionResult | None:
        if request.family != "dreamer" or request.mode != "proof_compare":
            return None
        summary_path = self._resolve_dreamer_bootstrap_summary_path(request)
        if summary_path is None or not summary_path.exists():
            return BackendExecutionResult(
                status="incomplete",
                reason_code="dreamer_bootstrap_incomplete",
                message="Dreamer compare is blocked until an official bootstrap summary is available.",
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                profile=request.profile,
                run_id=request.run_id,
                summary_path=str(summary_path) if summary_path is not None else None,
                metrics={"minimum_locked_seeds": DREAMER_MIN_LOCKED_SEEDS},
                next_action="Run Dreamer official bootstrap and produce a usable summary.json first.",
            )

        summary_payload = self._load_json(summary_path)
        if summary_payload is None:
            return BackendExecutionResult(
                status="failed",
                reason_code="user_configuration_error",
                message=f"Dreamer bootstrap summary is unreadable: {summary_path}",
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                profile=request.profile,
                run_id=request.run_id,
                summary_path=str(summary_path),
                next_action="Regenerate the Dreamer bootstrap summary.json.",
            )

        execution_result = summary_payload.get("execution_result")
        usable_seed_raw = summary_payload.get("usable_seed_count", 0)
        usable_seed_count = int(usable_seed_raw) if isinstance(usable_seed_raw, int | float) else 0
        if isinstance(execution_result, dict):
            metrics = execution_result.get("metrics")
            if isinstance(metrics, dict):
                usable_seed_count = int(
                    metrics.get("success_count", usable_seed_count) or usable_seed_count
                )
        if (
            not isinstance(execution_result, dict)
            or str(execution_result.get("status")) != "succeeded"
        ):
            return BackendExecutionResult(
                status="incomplete",
                reason_code="dreamer_bootstrap_incomplete",
                message=(
                    "Dreamer compare is blocked because the latest official bootstrap summary "
                    "is not yet proof-ready."
                ),
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                profile=request.profile,
                run_id=request.run_id,
                summary_path=str(summary_path),
                metrics={
                    "usable_seed_count": usable_seed_count,
                    "minimum_locked_seeds": DREAMER_MIN_LOCKED_SEEDS,
                },
                next_action="Finish Dreamer official bootstrap until the summary execution_result succeeds.",
            )
        return None

    def execute(self, request: BackendExecutionRequest) -> BackendExecutionResult:
        if request.mode == "train":
            return self._execute_train(request)
        dreamer_bootstrap_gate = self._validate_dreamer_bootstrap_ready(request)
        if dreamer_bootstrap_gate is not None:
            return dreamer_bootstrap_gate
        allow_official_only = bool(request.proof_requirements.get("allow_official_only", False))
        resolution = resolve_execution_manifest(
            request,
            scripts_root=self.scripts_root,
            allow_official_only=allow_official_only,
        )
        if resolution.early_result is not None:
            return resolution.early_result
        assert resolution.manifest_path is not None
        manifest_path = resolution.manifest_path
        if not manifest_path.exists():
            return BackendExecutionResult(
                status="failed",
                reason_code="manifest_missing",
                message=f"Manifest not found: {manifest_path}",
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="official_only" if request.mode == "proof_bootstrap" else "compare",
                profile=request.profile,
                run_id=request.run_id,
                manifest_path=str(manifest_path),
                next_action="Provide a valid manifest path or fix routing policy.",
            )

        if request.mode == "proof_bootstrap":
            return self._execute_dreamer_bootstrap(request, manifest_path=manifest_path)
        return self._execute_proof_compare(request, manifest_path=manifest_path)

    def _run_subprocess(self, command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(self.repo_root),
            check=False,
            text=True,
            capture_output=True,
        )

    @staticmethod
    def _format_subprocess_failure(
        command: list[str], result: subprocess.CompletedProcess[str]
    ) -> str:
        stderr_tail = (result.stderr or "").strip()[-2000:]
        stdout_tail = (result.stdout or "").strip()[-1000:]
        lines = [
            f"Command failed with exit code {result.returncode}",
            f"Command: {' '.join(command)}",
        ]
        if stderr_tail:
            lines.append(f"stderr:\n{stderr_tail}")
        if stdout_tail:
            lines.append(f"stdout:\n{stdout_tail}")
        return "\n".join(lines)

    @staticmethod
    def _utc_run_id(prefix: str) -> str:
        return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"

    def _execute_dreamer_bootstrap(
        self,
        request: BackendExecutionRequest,
        *,
        manifest_path: Path,
    ) -> BackendExecutionResult:
        run_id = request.run_id or self._utc_run_id("dreamer_official_bootstrap")
        output_root = Path(request.output_root or (self.repo_root / "reports" / "parity" / run_id))
        official_repo = (self.repo_root / "third_party" / "dreamerv3_official").resolve()
        command = [
            sys.executable,
            str(self.scripts_root / "run_dreamerv3_official_batch.py"),
            "--repo-root",
            str(official_repo),
            "--output-root",
            str(output_root),
            "--task-id",
            str(request.task_filter or "atari100k_pong"),
            "--seed-list",
            ",".join(str(seed) for seed in request.seed_list),
            "--device",
            request.device,
        ]
        result = self._run_subprocess(command)
        summary_path = output_root / "summary.json"
        if result.returncode != 0:
            return BackendExecutionResult(
                status="failed",
                reason_code="subprocess_failed",
                message=self._format_subprocess_failure(command, result),
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="official_only",
                profile=request.profile,
                run_id=run_id,
                manifest_path=str(manifest_path),
                summary_path=str(summary_path.resolve())
                if summary_path.exists()
                else str((output_root / "batch.log").resolve()),
                next_action="Inspect bootstrap batch logs and official repo availability.",
            )
        return BackendExecutionResult(
            status="succeeded",
            reason_code="none",
            message="Dreamer official bootstrap batch completed.",
            backend=request.backend,
            family=request.family,
            mode=request.mode,
            proof_phase="official_only",
            profile=request.profile,
            run_id=run_id,
            manifest_path=str(manifest_path),
            summary_path=str(summary_path.resolve())
            if summary_path.exists()
            else str((output_root / "batch.log").resolve()),
            metrics={"seed_count": len(request.seed_list)},
            next_action="Review artifacts and continue toward minimum proof threshold.",
        )

    def _execute_train(self, request: BackendExecutionRequest) -> BackendExecutionResult:
        if request.family == "dreamer" and request.backend == "official_dreamerv3_jax_subprocess":
            manifest_path = (
                self.scripts_root / "manifests" / "dreamerv3_official_checkpoint_bootstrap_v1.json"
            ).resolve()
            return self._execute_dreamer_bootstrap(request, manifest_path=manifest_path)
        if request.family == "tdmpc2":
            report_path = _resolve_tdmpc2_alignment_report(request)
            report_status = _tdmpc2_alignment_status(report_path)
            if (
                str(request.profile or "").strip().lower() == "proof_5m"
                and report_status == "aligned"
            ):
                return BackendExecutionResult(
                    status="blocked",
                    reason_code="backend_unsupported",
                    message=(
                        "TD-MPC2 proof_5m alignment passed, but delegated training execution "
                        "is not implemented yet."
                    ),
                    backend=request.backend,
                    family=request.family,
                    mode=request.mode,
                    proof_phase="official_only",
                    profile=request.profile,
                    run_id=request.run_id,
                    manifest_path=str(report_path) if report_path is not None else None,
                    metrics={"alignment_status": report_status},
                    next_action="Use native_torch local training until TD-MPC2 delegated execution is implemented.",
                )
            return BackendExecutionResult(
                status="blocked",
                reason_code="tdmpc2_architecture_mismatch_open",
                message=(
                    "TD-MPC2 delegated training remains blocked until proof_5m alignment "
                    "is validated."
                ),
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="official_only",
                profile=request.profile,
                run_id=request.run_id,
                next_action="Generate a passing TD-MPC2 alignment report before delegated training.",
            )
        return BackendExecutionResult(
            status="blocked",
            reason_code="backend_unsupported",
            message=f"Unsupported delegated training backend: family={request.family} backend={request.backend}",
            backend=request.backend,
            family=request.family,
            mode=request.mode,
            proof_phase="official_only",
            profile=request.profile,
            run_id=request.run_id,
            next_action="Use native_torch local training or a supported delegated backend.",
        )

    def _execute_proof_compare(
        self,
        request: BackendExecutionRequest,
        *,
        manifest_path: Path,
    ) -> BackendExecutionResult:
        run_id = request.run_id or self._utc_run_id("verify")
        output_root = Path(
            request.output_root or (self.repo_root / "reports" / "parity" / "verify")
        )
        run_root = output_root / run_id
        runs_jsonl = run_root / "parity_runs.jsonl"
        equivalence_json = run_root / "equivalence_report.json"
        equivalence_md = run_root / "equivalence_report.md"

        run_cmd = [
            sys.executable,
            str(self.scripts_root / "run_parity_matrix.py"),
            "--manifest",
            str(manifest_path),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_root),
            "--device",
            request.device,
            "--max-retries",
            "1",
            "--resume",
            "--systems",
            "official,worldflux",
            "--seed-list",
            ",".join(str(seed) for seed in request.seed_list),
        ]
        if request.task_filter:
            run_cmd.extend(["--task-filter", request.task_filter])
        run_result = self._run_subprocess(run_cmd)
        if run_result.returncode != 0:
            return BackendExecutionResult(
                status="failed",
                reason_code="subprocess_failed",
                message=self._format_subprocess_failure(run_cmd, run_result),
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                profile=request.profile,
                run_id=run_id,
                manifest_path=str(manifest_path),
                next_action="Inspect parity run logs and manifest setup.",
            )

        stats_cmd = [
            sys.executable,
            str(self.scripts_root / "stats_equivalence.py"),
            "--input",
            str(runs_jsonl),
            "--output",
            str(equivalence_json),
            "--manifest",
            str(manifest_path),
            "--proof-mode",
            "--strict-completeness",
            "--strict-validity",
            "--policy-mode-required",
            str(request.proof_requirements.get("policy_mode_required", "parity_candidate")),
        ]
        stats_result = self._run_subprocess(stats_cmd)
        if stats_result.returncode != 0:
            return BackendExecutionResult(
                status="failed",
                reason_code="stats_failed",
                message=self._format_subprocess_failure(stats_cmd, stats_result),
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                profile=request.profile,
                run_id=run_id,
                manifest_path=str(manifest_path),
                next_action="Inspect equivalence/stats configuration and run artifacts.",
            )

        report_cmd = [
            sys.executable,
            str(self.scripts_root / "report_markdown.py"),
            "--input",
            str(equivalence_json),
            "--output",
            str(equivalence_md),
        ]
        report_result = self._run_subprocess(report_cmd)
        if report_result.returncode != 0:
            return BackendExecutionResult(
                status="failed",
                reason_code="subprocess_failed",
                message=self._format_subprocess_failure(report_cmd, report_result),
                backend=request.backend,
                family=request.family,
                mode=request.mode,
                proof_phase="compare",
                profile=request.profile,
                run_id=run_id,
                manifest_path=str(manifest_path),
                next_action="Inspect report rendering inputs.",
            )

        report_payload = json.loads(equivalence_json.read_text(encoding="utf-8"))
        global_block = report_payload.get("global", {})
        passed = (
            bool(global_block.get("parity_pass_final", False))
            if isinstance(global_block, dict)
            else False
        )
        status = "succeeded" if passed else "failed"
        reason_code = "none" if passed else "validity_failed"
        message = (
            "Proof compare completed successfully."
            if passed
            else "Proof compare finished but parity/validity gates did not pass."
        )
        metrics = {
            "missing_pairs": int(global_block.get("missing_pairs", 0) or 0)
            if isinstance(global_block, dict)
            else 0,
            "validity_pass": bool(global_block.get("validity_pass", False))
            if isinstance(global_block, dict)
            else False,
            "parity_pass_final": passed,
        }
        return BackendExecutionResult(
            status=status,  # type: ignore[arg-type]
            reason_code=reason_code,  # type: ignore[arg-type]
            message=message,
            backend=request.backend,
            family=request.family,
            mode=request.mode,
            proof_phase="compare",
            profile=request.profile,
            run_id=run_id,
            manifest_path=str(manifest_path),
            summary_path=str(equivalence_json.resolve()),
            equivalence_report_json=str(equivalence_json.resolve()),
            equivalence_report_md=str(equivalence_md.resolve()),
            metrics=metrics,
            next_action=None if passed else "Inspect validity and equivalence reports.",
        )
