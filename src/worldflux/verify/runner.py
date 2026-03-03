"""Parity verification runner for end-user model validation."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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
            When *True*, return synthetic results suitable for demonstrations.
        device:
            Execution device string.
        """
        if demo:
            return cls._run_demo(target=target, baseline=baseline, env=env, device=device)
        return cls._run_real(target=target, baseline=baseline, env=env, device=device)

    @classmethod
    def _repo_root(cls) -> Path:
        # src/worldflux/verify/runner.py -> repo root is three parents up.
        return Path(__file__).resolve().parents[3]

    @classmethod
    def _parity_scripts_root(cls) -> Path:
        return cls._repo_root() / "scripts" / "parity"

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
    def _select_manifest(*, target: str, baseline: str, scripts_root: Path) -> Path:
        target_path = Path(target).expanduser()
        if target_path.exists() and target_path.suffix.lower() in {".json", ".yaml", ".yml"}:
            return target_path.resolve()

        override = os.getenv("WORLDFLUX_VERIFY_MANIFEST", "").strip()
        if override:
            override_path = Path(override).expanduser()
            if override_path.exists():
                return override_path.resolve()

        # Single manifest supports both dreamerv3 and tdmpc2 quick proof tracks.
        _ = baseline
        return (scripts_root / "manifests" / ParityVerifier._DEFAULT_MANIFEST).resolve()

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
    def _run_subprocess(cls, command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(command, cwd=str(cwd), check=False, text=True, capture_output=True)

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
            verdict_reason="Demo mode: synthetic pass",
        )

    @classmethod
    def _preflight_check(cls, *, manifest: Path, scripts_root: Path) -> None:
        """Validate proof-mode prerequisites before launching subprocesses.

        Raises :class:`RuntimeError` with a descriptive message on failure.
        """
        # 1. Manifest must be parseable (JSON or YAML)
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
                    f"Manifest {manifest.name} is not valid JSON and pyyaml is not "
                    "installed.  Install pyyaml or fix the JSON syntax."
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Manifest {manifest.name} is neither valid JSON nor valid YAML: {exc}"
                ) from exc

        if not isinstance(manifest_data, dict):
            raise RuntimeError(f"Manifest must be a JSON/YAML object, got {type(manifest_data)}")

        # 2. Required scripts exist
        for script_name in ("run_parity_matrix.py", "stats_equivalence.py"):
            script_path = scripts_root / script_name
            if not script_path.exists():
                raise RuntimeError(
                    f"Required proof script not found: {script_path}\n"
                    "Ensure you are running from a complete WorldFlux repository checkout."
                )

        # 3. Check official repo paths referenced in commands
        for task in manifest_data.get("tasks", []):
            if not isinstance(task, dict):
                continue
            official = task.get("official", {})
            if not isinstance(official, dict):
                continue
            cmd = official.get("command", [])
            for i, arg in enumerate(cmd):
                if arg == "--repo-root" and i + 1 < len(cmd):
                    repo_path_str = cmd[i + 1]
                    # Skip template variables
                    if "{" in repo_path_str:
                        continue
                    # Resolve relative to the cwd specified in manifest
                    cwd = official.get("cwd", ".")
                    base = (scripts_root / "manifests" / cwd).resolve()
                    repo_path = (base / repo_path_str).resolve()
                    if not repo_path.exists():
                        task_id = task.get("task_id", "?")
                        raise RuntimeError(
                            f"Official repository not found for task '{task_id}': "
                            f"{repo_path}\n"
                            f"Clone the official repo or use --mode quick for "
                            "lightweight verification."
                        )

    @staticmethod
    def _format_subprocess_error(
        label: str,
        command: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> str:
        """Build a descriptive error message from a failed subprocess."""
        stderr_tail = (result.stderr or "").strip()[-2000:]
        stdout_tail = (result.stdout or "").strip()[-1000:]

        lines = [
            f"{label} (exit code {result.returncode})",
            f"Command: {' '.join(command)}",
        ]
        if stderr_tail:
            lines.append(f"stderr (last 2000 chars):\n{stderr_tail}")
        if stdout_tail:
            lines.append(f"stdout (last 1000 chars):\n{stdout_tail}")

        # Actionable hints based on common error patterns
        combined = stderr_tail + stdout_tail
        if "repo root not found" in combined or "No such file or directory" in combined:
            lines.append(
                "Hint: An expected directory is missing. Check that official "
                "repositories are cloned alongside the WorldFlux repo."
            )
        if "ModuleNotFoundError" in combined:
            lines.append("Hint: A Python dependency is missing. Run `uv sync --extra dev`.")
        if "CUDA" in combined or "cuda" in combined:
            lines.append("Hint: CUDA error detected. Try --device cpu or check GPU drivers.")

        return "\n".join(lines)

    @classmethod
    def _run_real(
        cls,
        *,
        target: str,
        baseline: str,
        env: str,
        device: str,
    ) -> VerifyResult:
        start = time.monotonic()
        repo_root = cls._repo_root()
        scripts_root = cls._parity_scripts_root()
        if not scripts_root.exists():
            raise RuntimeError(
                "Unable to locate proof scripts at scripts/parity. "
                "Run verify from the WorldFlux repository checkout."
            )

        manifest = cls._select_manifest(target=target, baseline=baseline, scripts_root=scripts_root)
        if not manifest.exists():
            raise RuntimeError(f"Parity manifest not found: {manifest}")

        # Preflight validation
        cls._preflight_check(manifest=manifest, scripts_root=scripts_root)

        output_root = cls._DEFAULT_OUTPUT_DIR
        output_root.mkdir(parents=True, exist_ok=True)

        task_filter = cls._env_to_task_filter(env)
        seed_list = os.getenv("WORLDFLUX_VERIFY_SEED_LIST", cls._DEFAULT_SEED_LIST).strip()
        run_id = f"verify_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        run_root = output_root / run_id
        runs_jsonl = run_root / "parity_runs.jsonl"
        equivalence_json = run_root / "equivalence_report.json"
        equivalence_md = run_root / "equivalence_report.md"

        run_cmd = [
            sys.executable,
            str(scripts_root / "run_parity_matrix.py"),
            "--manifest",
            str(manifest),
            "--run-id",
            run_id,
            "--output-dir",
            str(output_root),
            "--device",
            str(device),
            "--max-retries",
            "1",
            "--resume",
            "--systems",
            "official,worldflux",
        ]
        if task_filter:
            run_cmd.extend(["--task-filter", task_filter])
        if seed_list:
            run_cmd.extend(["--seed-list", seed_list])

        run_result = cls._run_subprocess(run_cmd, cwd=repo_root)
        if run_result.returncode != 0:
            raise RuntimeError(
                cls._format_subprocess_error("Proof run failed", run_cmd, run_result)
            )

        if not runs_jsonl.exists():
            raise RuntimeError(f"Missing proof artifact: {runs_jsonl}")

        stats_cmd = [
            sys.executable,
            str(scripts_root / "stats_equivalence.py"),
            "--input",
            str(runs_jsonl),
            "--output",
            str(equivalence_json),
            "--manifest",
            str(manifest),
            "--proof-mode",
            "--strict-completeness",
            "--strict-validity",
            "--policy-mode-required",
            "parity_candidate",
        ]
        stats_result = cls._run_subprocess(stats_cmd, cwd=repo_root)
        if stats_result.returncode != 0:
            raise RuntimeError(
                cls._format_subprocess_error(
                    "Equivalence stats computation failed", stats_cmd, stats_result
                )
            )

        report_cmd = [
            sys.executable,
            str(scripts_root / "report_markdown.py"),
            "--input",
            str(equivalence_json),
            "--output",
            str(equivalence_md),
        ]
        report_result = cls._run_subprocess(report_cmd, cwd=repo_root)
        if report_result.returncode != 0:
            raise RuntimeError(
                cls._format_subprocess_error(
                    "Markdown report rendering failed", report_cmd, report_result
                )
            )

        report_payload = cls._load_report(equivalence_json)
        global_block = report_payload.get("global", {})
        if not isinstance(global_block, dict):
            global_block = {}

        elapsed = time.monotonic() - start
        stats = cls._extract_stats_from_report(report_payload)
        stats.update(
            {
                "run_id": run_id,
                "runs_jsonl": str(runs_jsonl.resolve()),
                "equivalence_report_json": str(equivalence_json.resolve()),
                "equivalence_report_md": str(equivalence_md.resolve()),
                "device": device,
                "task_filter": task_filter,
            }
        )

        return VerifyResult(
            passed=bool(global_block.get("parity_pass_final", False)),
            target=target,
            baseline=baseline,
            env=env,
            demo=False,
            elapsed_seconds=round(elapsed, 3),
            stats=stats,
            verdict_reason=(
                f"proof_run_id={run_id} "
                f"final={bool(global_block.get('parity_pass_final', False))} "
                f"missing_pairs={int(global_block.get('missing_pairs', 0) or 0)}"
            ),
        )
