"""Parity verification runner for end-user model validation."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from worldflux.execution import BackendExecutionRequest, ExecutionMode, ParityBackedExecutor


@dataclass(frozen=True)
class VerifyResult:
    """Result of a parity verification run."""

    passed: bool
    target: str
    baseline: str
    env: str
    demo: bool
    elapsed_seconds: float
    stats: dict[str, Any] = field(default_factory=dict)
    verdict_reason: str = ""


class ParityVerifier:
    """Run parity verification between a target model and an official baseline."""

    _DEFAULT_MANIFEST = "official_vs_worldflux_v1.yaml"
    _DEFAULT_OUTPUT_DIR = Path("reports/parity/verify")
    _DEFAULT_SEED_LIST = "0,1,2"

    @classmethod
    def run(
        cls,
        *,
        target: str,
        baseline: str = "official/dreamerv3",
        env: str = "atari/pong",
        demo: bool = False,
        device: str = "cpu",
        backend: str = "native_torch",
        backend_profile: str = "",
        allow_official_only: bool = False,
        proof_claim: str = "compare",
    ) -> VerifyResult:
        """Execute verification and return a :class:`VerifyResult`.

        Parameters
        ----------
        target:
            Path or registry ID of the custom model.
        baseline:
            Baseline model identifier.
        env:
            Target simulation environment.
        demo:
            When *True*, return synthetic results suitable for demonstrations only.
        device:
            Execution device string.
        """
        if demo:
            return cls._run_demo(target=target, baseline=baseline, env=env, device=device)
        return cls._run_real(
            target=target,
            baseline=baseline,
            env=env,
            device=device,
            backend=backend,
            backend_profile=backend_profile,
            allow_official_only=allow_official_only,
            proof_claim=proof_claim,
        )

    @classmethod
    def _repo_root(cls) -> Path:
        # src/worldflux/verify/runner.py -> repo root is three parents up.
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _parity_scripts_root(cls) -> Path:
        return cls._repo_root() / "scripts" / "parity"

    @staticmethod
    def _infer_family(*, baseline: str, target: str) -> str:
        baseline_value = str(baseline).strip().lower()
        target_value = str(target).strip().lower()
        if "dreamer" in baseline_value or target_value.startswith("dreamer"):
            return "dreamer"
        if "tdmpc2" in baseline_value or target_value.startswith("tdmpc2"):
            return "tdmpc2"
        return "unknown"

    @staticmethod
    def _env_to_task_filter(env: str) -> str:
        value = str(env).strip().lower()
        if not value:
            return ""

        if value.startswith("atari/"):
            game = value.split("/", 1)[1].strip().replace("-", "_")
            return f"atari100k_{game}" if game else ""

        if value.startswith("dmcontrol/"):
            suffix = value.split("/", 1)[1].strip().replace("/", "-")
            return suffix

        if value.startswith("mujoco/"):
            suffix = value.split("/", 1)[1].strip().replace("_", "-")
            return suffix

        # Already task-id like "atari100k_pong" or "walker-run".
        return value

    @staticmethod
    def _load_report(path: Path) -> dict[str, Any]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"Invalid parity report payload: {path}")
        return payload

    @staticmethod
    def _extract_stats_from_report(report: dict[str, Any]) -> dict[str, Any]:
        global_block = report.get("global", {})
        if not isinstance(global_block, dict):
            global_block = {}
        config_block = report.get("config", {})
        if not isinstance(config_block, dict):
            config_block = {}
        primary_metric = str(config_block.get("primary_metric", "final_return_mean"))

        metric_reports: list[dict[str, Any]] = []
        for task in report.get("tasks", []):
            if not isinstance(task, dict):
                continue
            metrics = task.get("metrics", {})
            if not isinstance(metrics, dict):
                continue
            metric_payload = metrics.get(primary_metric)
            if isinstance(metric_payload, dict) and metric_payload.get("status") == "ok":
                metric_reports.append(metric_payload)

        samples = sum(int(m.get("n_pairs", 0) or 0) for m in metric_reports)
        margin_ratio = float(config_block.get("equivalence_margin", 0.05))

        if metric_reports:
            tost_p_value = max(
                float(m.get("tost", {}).get("p_value", 1.0) or 1.0) for m in metric_reports
            )
            ci_upper_ratio = max(
                float((m.get("ci90_ratio", [None, 1.0]) or [None, 1.0])[1] or 1.0)
                for m in metric_reports
            )
            bayesian_equivalence_hdi = min(
                float(m.get("bayesian", {}).get("p_equivalence", 0.0) or 0.0)
                for m in metric_reports
            )
        else:
            tost_p_value = 1.0
            ci_upper_ratio = 1.0
            bayesian_equivalence_hdi = 0.0

        return {
            "samples": samples,
            "mean_drop_ratio": max(0.0, ci_upper_ratio - 1.0),
            "ci_upper_ratio": ci_upper_ratio,
            "margin_ratio": margin_ratio,
            "bayesian_equivalence_hdi": bayesian_equivalence_hdi,
            "tost_p_value": tost_p_value,
            "missing_pairs": int(global_block.get("missing_pairs", 0) or 0),
            "validity_pass": bool(global_block.get("validity_pass", False)),
            "tasks_total": int(global_block.get("tasks_total", 0) or 0),
        }

    @classmethod
    def _preflight_check(cls, *, manifest: Path, scripts_root: Path) -> None:
        """Compatibility preflight validation for proof-mode prerequisites."""
        try:
            raw = manifest.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Cannot read manifest: {manifest} ({exc})") from exc

        manifest_data: dict[str, Any] | None = None
        try:
            manifest_data = json.loads(raw)
        except json.JSONDecodeError:
            pass

        if manifest_data is None:
            try:
                import yaml  # type: ignore[import-untyped]

                manifest_data = yaml.safe_load(raw)
            except ImportError:
                raise RuntimeError(
                    f"Manifest {manifest.name} is not valid JSON and pyyaml is not installed."
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Manifest {manifest.name} is neither valid JSON nor valid YAML: {exc}"
                ) from exc

        if not isinstance(manifest_data, dict):
            raise RuntimeError(f"Manifest must be a JSON/YAML object, got {type(manifest_data)}")

        for script_name in ("run_parity_matrix.py", "stats_equivalence.py"):
            script_path = scripts_root / script_name
            if not script_path.exists():
                raise RuntimeError(f"Required proof script not found: {script_path}")

    @staticmethod
    def _format_subprocess_error(
        label: str,
        command: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> str:
        """Compatibility formatter for subprocess failure diagnostics."""
        stderr_tail = (result.stderr or "").strip()[-2000:]
        stdout_tail = (result.stdout or "").strip()[-1000:]

        lines = [f"{label} (exit code {result.returncode})", f"Command: {' '.join(command)}"]
        if stderr_tail:
            lines.append(f"stderr (last 2000 chars):\n{stderr_tail}")
        if stdout_tail:
            lines.append(f"stdout (last 1000 chars):\n{stdout_tail}")

        combined = stderr_tail + stdout_tail
        if "ModuleNotFoundError" in combined:
            lines.append("Hint: A Python dependency is missing. Run `uv sync --extra dev`.")
        if "CUDA" in combined or "cuda" in combined:
            lines.append("Hint: CUDA error detected. Try --device cpu or check GPU drivers.")
        if "No such file or directory" in combined:
            lines.append("Hint: An expected directory is missing.")

        return "\n".join(lines)

    @classmethod
    def _run_demo(
        cls,
        *,
        target: str,
        baseline: str,
        env: str,
        device: str,
    ) -> VerifyResult:
        start = time.monotonic()
        rng = random.Random(42)
        samples = 500
        mean_drop = rng.uniform(0.001, 0.015)
        ci_upper = mean_drop + rng.uniform(0.005, 0.02)
        margin = 0.05
        bayesian_hdi = round(rng.uniform(0.975, 0.995), 3)
        tost_p = round(rng.uniform(0.005, 0.025), 3)
        time.sleep(rng.uniform(2.5, 3.5))
        elapsed = time.monotonic() - start
        return VerifyResult(
            passed=True,
            target=target,
            baseline=baseline,
            env=env,
            demo=True,
            elapsed_seconds=round(elapsed, 3),
            stats={
                "samples": samples,
                "mean_drop_ratio": round(mean_drop, 6),
                "ci_upper_ratio": round(ci_upper, 6),
                "margin_ratio": margin,
                "bayesian_equivalence_hdi": bayesian_hdi,
                "tost_p_value": tost_p,
                "device": device,
            },
            verdict_reason="Synthetic demo mode: example pass (not proof)",
        )

    @classmethod
    def _run_real(
        cls,
        *,
        target: str,
        baseline: str,
        env: str,
        device: str,
        backend: str,
        backend_profile: str,
        allow_official_only: bool,
        proof_claim: str,
    ) -> VerifyResult:
        start = time.monotonic()
        repo_root = cls._repo_root()
        scripts_root = cls._parity_scripts_root()
        if not scripts_root.exists():
            raise RuntimeError(
                "Unable to locate proof scripts at scripts/parity. "
                "Run verify from the WorldFlux repository checkout."
            )
        task_filter = cls._env_to_task_filter(env)
        raw_seed_list = os.getenv("WORLDFLUX_VERIFY_SEED_LIST", cls._DEFAULT_SEED_LIST).strip()
        seed_list = [int(part.strip()) for part in raw_seed_list.split(",") if part.strip()]
        mode = cast(ExecutionMode, "proof_bootstrap" if allow_official_only else "proof_compare")
        request = BackendExecutionRequest(
            backend=backend,
            family=cls._infer_family(baseline=baseline, target=target),
            mode=mode,
            target=target,
            baseline=baseline,
            task_filter=task_filter or None,
            env=env,
            seed_list=seed_list,
            device=device,
            profile=backend_profile or None,
            run_id=f"verify_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
            output_root=str(cls._DEFAULT_OUTPUT_DIR.resolve()),
            proof_requirements={
                "allow_official_only": allow_official_only,
                "proof_claim": proof_claim,
                "policy_mode_required": "parity_candidate",
            },
        )
        executor = ParityBackedExecutor(repo_root=repo_root, scripts_root=scripts_root)
        execution_result = executor.execute(request)

        elapsed = time.monotonic() - start
        stats = {
            "device": device,
            "task_filter": task_filter,
            "execution_backend": backend,
            "execution_profile": backend_profile or None,
            "execution_phase": "official_only" if allow_official_only else "compare",
            "execution_result": execution_result.to_dict(),
            **dict(execution_result.metrics),
        }
        if execution_result.equivalence_report_json:
            report_payload = cls._load_report(Path(execution_result.equivalence_report_json))
            stats.update(cls._extract_stats_from_report(report_payload))
            stats["equivalence_report_json"] = execution_result.equivalence_report_json
            stats["equivalence_report_md"] = execution_result.equivalence_report_md

        passed = execution_result.status == "succeeded"
        return VerifyResult(
            passed=passed,
            target=target,
            baseline=baseline,
            env=env,
            demo=False,
            elapsed_seconds=round(elapsed, 3),
            stats=stats,
            verdict_reason=(
                f"status={execution_result.status} reason={execution_result.reason_code} "
                f"run_id={execution_result.run_id or '-'}"
            ),
        )
