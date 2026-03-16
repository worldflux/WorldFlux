#!/usr/bin/env python3
"""Run DreamerV3 official checkpoints across multiple seeds with artifact normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from worldflux.execution import (  # noqa: E402
    DREAMER_MIN_LOCKED_SEEDS,
    DREAMER_MIN_PROOF_SEEDS,
    normalize_dreamer_official_batch_summary,
)
from worldflux.parity import stable_recipe_hash  # noqa: E402
from worldflux.parity.backend_contract import resolve_latest_checkpoint_dir  # noqa: E402


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task-id", type=str, default="atari100k_pong")
    parser.add_argument("--seed-list", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--steps", type=int, default=110_000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--parallelism", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--python-executable", type=str, default=sys.executable)
    parser.add_argument("--gpu-devices", type=str, default="")
    parser.add_argument("--xla-mem-fraction", type=float, default=0.85)
    parser.add_argument("--s3-prefix", type=str, default="")
    parser.add_argument("--source-commit", type=str, default="")
    parser.add_argument("--obs-shape", type=str, default="3,64,64")
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-interval-sec", type=int, default=30)
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock", action="store_true")
    return parser.parse_args()


def _parse_seed_list(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit("--seed-list must not be empty")
    return out


def _parse_obs_shape(raw: str) -> tuple[int, ...]:
    dims = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if not dims:
        raise SystemExit("--obs-shape must not be empty")
    return dims


def _parse_gpu_devices(raw: str, *, parallelism: int) -> list[str]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if values:
        return values
    return [str(index) for index in range(max(1, int(parallelism)))]


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _seed_root(output_root: Path, seed: int) -> Path:
    return output_root / f"seed_{seed}"


def _status_path(output_root: Path, seed: int) -> Path:
    return _seed_root(output_root, seed) / "status.json"


def _artifact_manifest_path(output_root: Path, seed: int) -> Path:
    return _seed_root(output_root, seed) / "artifact_manifest.json"


def _component_report_path(output_root: Path, seed: int) -> Path:
    return _seed_root(output_root, seed) / "component_match_report.json"


def _batch_log_path(output_root: Path) -> Path:
    return output_root / "batch.log"


def _append_batch_log(output_root: Path, message: str) -> None:
    path = _batch_log_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} {message}\n")


def _source_commit(repo_root: Path, provided: str) -> str:
    if provided.strip():
        return provided.strip()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root),
        check=False,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() or "unknown"


def _build_command(args: argparse.Namespace, *, seed: int) -> list[str]:
    seed_dir = _seed_root(args.output_root, seed)
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "wrappers" / "official_dreamerv3.py"),
        "--repo-root",
        str(args.repo_root.resolve()),
        "--task-id",
        args.task_id,
        "--seed",
        str(seed),
        "--steps",
        str(args.steps),
        "--device",
        args.device,
        "--run-dir",
        str((seed_dir / "official").resolve()),
        "--metrics-out",
        str((seed_dir / "official" / "metrics.json").resolve()),
        "--eval-episodes",
        str(args.eval_episodes),
        "--python-executable",
        args.python_executable,
    ]
    if args.mock:
        command.append("--mock")
    return command


def _launch_env(args: argparse.Namespace, *, gpu_id: str) -> dict[str, str]:
    return {
        "CUDA_VISIBLE_DEVICES": gpu_id,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": str(float(args.xla_mem_fraction)),
    }


def _select_gpu_id(gpu_devices: list[str], running_gpu: dict[int, str]) -> str:
    in_use = set(running_gpu.values())
    for gpu_id in gpu_devices:
        if gpu_id not in in_use:
            return gpu_id
    return gpu_devices[0]


def _command_signature(command: list[str]) -> str:
    normalized: list[str] = []
    skip_next = False
    dynamic_flags = {"--seed", "--run-dir", "--metrics-out"}
    for token in command:
        if skip_next:
            skip_next = False
            continue
        if token in dynamic_flags:
            normalized.append(token)
            normalized.append("<dynamic>")
            skip_next = True
            continue
        normalized.append(token)
    return hashlib.sha256(json.dumps(normalized, separators=(",", ":")).encode("utf-8")).hexdigest()


def _normalized_config_text(text: str) -> str:
    dynamic_prefixes = ("logdir:", "seed:", "replica:")
    lines: list[str] = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if any(stripped.startswith(prefix) for prefix in dynamic_prefixes):
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def _config_signature(config_path: Path) -> str:
    return hashlib.sha256(
        _normalized_config_text(config_path.read_text(encoding="utf-8")).encode("utf-8")
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_runtime_paths(output_root: Path, seed: int) -> dict[str, Path]:
    seed_dir = _seed_root(output_root, seed)
    official_dir = seed_dir / "official"
    logdir = official_dir / "dreamerv3_logdir"
    ckpt_dir = logdir / "ckpt"
    latest_dir = resolve_latest_checkpoint_dir(ckpt_dir)
    return {
        "seed_dir": seed_dir,
        "official_dir": official_dir,
        "logdir": logdir,
        "ckpt_dir": ckpt_dir,
        "latest_dir": latest_dir if latest_dir is not None else ckpt_dir,
        "metrics_json": official_dir / "metrics.json",
        "config_yaml": logdir / "config.yaml",
        "scores_jsonl": logdir / "scores.jsonl",
        "metrics_jsonl": logdir / "metrics.jsonl",
        "runner_stdout": seed_dir / "runner.stdout.log",
        "runner_stderr": seed_dir / "runner.stderr.log",
    }


def _required_files(paths: dict[str, Path]) -> dict[str, Path]:
    latest_dir = paths["latest_dir"]
    return {
        "config_snapshot": paths["config_yaml"],
        "scores": paths["scores_jsonl"],
        "metrics_jsonl": paths["metrics_jsonl"],
        "latest": paths["ckpt_dir"] / "latest",
        "agent": latest_dir / "agent.pkl",
        "replay": latest_dir / "replay.pkl",
        "step": latest_dir / "step.pkl",
        "done": latest_dir / "done",
        "runner_stdout": paths["runner_stdout"],
        "runner_stderr": paths["runner_stderr"],
        "metrics_json": paths["metrics_json"],
    }


def _missing_required_files(paths: dict[str, Path]) -> list[str]:
    missing = []
    for name, path in _required_files(paths).items():
        if not path.exists():
            missing.append(name)
    return missing


def _latest_mtime(paths: dict[str, Path]) -> float:
    mtimes = [path.stat().st_mtime for path in _required_files(paths).values() if path.exists()]
    if not mtimes:
        return 0.0
    return max(mtimes)


def _initial_status(*, seed: int, attempt: int, command: list[str]) -> dict[str, Any]:
    return {
        "seed": seed,
        "attempt": attempt,
        "state": "planned",
        "command": command,
        "command_signature": _command_signature(command),
        "started_at": None,
        "last_heartbeat": None,
        "completed_at": None,
        "exit_code": None,
        "message": "",
    }


def _artifact_manifest_from_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    recipe_hash: str,
    component_report: Path,
) -> dict[str, Any]:
    paths = _seed_runtime_paths(args.output_root, seed)
    latest_dir = resolve_latest_checkpoint_dir(paths["ckpt_dir"])
    if latest_dir is None:
        raise RuntimeError(f"latest checkpoint missing for seed {seed}")
    return {
        "backend_kind": "jax_subprocess",
        "adapter_id": "official_dreamerv3_jax_subprocess",
        "recipe_hash": recipe_hash,
        "source_commit": _source_commit(args.repo_root, args.source_commit),
        "config_snapshot": str(paths["config_yaml"].resolve()),
        "checkpoint_paths": sorted(str(path.resolve()) for path in latest_dir.glob("*.pkl")),
        "score_paths": [str(paths["scores_jsonl"].resolve())],
        "metrics_paths": [
            str(paths["metrics_json"].resolve()),
            str(paths["metrics_jsonl"].resolve()),
        ],
        "component_match_path": str(component_report.resolve()),
        "seed": seed,
    }


def _load_baseline(output_root: Path) -> dict[str, Any] | None:
    path = output_root / "baseline.json"
    if not path.exists():
        return None
    return _load_json(path)


def _write_baseline(output_root: Path, payload: dict[str, Any]) -> None:
    _atomic_write_json(output_root / "baseline.json", payload)


def _validate_against_baseline(*, baseline: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    for key in (
        "backend_kind",
        "adapter_id",
        "recipe_hash",
        "config_signature",
        "command_signature",
    ):
        if str(candidate.get(key)) != str(baseline.get(key)):
            mismatches.append(key)
    return mismatches


def _run_component_match(args: argparse.Namespace, *, seed: int) -> Path:
    paths = _seed_runtime_paths(args.output_root, seed)
    latest_dir = resolve_latest_checkpoint_dir(paths["ckpt_dir"])
    if latest_dir is None:
        raise RuntimeError(f"no checkpoint directory for seed {seed}")
    output = _component_report_path(args.output_root, seed)
    command = [
        args.python_executable,
        str(SCRIPT_DIR / "generate_component_match_report.py"),
        "--family",
        "dreamerv3",
        "--official-checkpoint",
        str((latest_dir / "agent.pkl").resolve()),
        "--output",
        str(output.resolve()),
        "--obs-shape",
        args.obs_shape,
        "--action-dim",
        str(args.action_dim),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(
        command, check=False, text=True, capture_output=True, cwd=str(SCRIPT_DIR.parents[1])
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"component match failed for seed {seed}: {completed.stderr or completed.stdout}"
        )
    return output


def _upload_seed_artifacts(
    args: argparse.Namespace, *, seed: int, manifest: dict[str, Any]
) -> None:
    if not args.s3_prefix.strip():
        return
    base = args.s3_prefix.rstrip("/") + f"/seed_{seed}/"
    uploads = [
        (manifest["config_snapshot"], base + "config.yaml"),
    ]
    for source in manifest["score_paths"]:
        uploads.append((source, base + Path(source).name))
    for source in manifest["metrics_paths"]:
        uploads.append((source, base + Path(source).name))
    for source in manifest["checkpoint_paths"]:
        uploads.append((source, base + Path(source).name))
    uploads.append((manifest["component_match_path"], base + "component_match_report.json"))
    seed_dir = _seed_root(args.output_root, seed)
    uploads.append((str((seed_dir / "runner.stdout.log").resolve()), base + "runner.stdout.log"))
    uploads.append((str((seed_dir / "runner.stderr.log").resolve()), base + "runner.stderr.log"))
    for source, target in uploads:
        subprocess.run(
            ["aws", "s3", "cp", source, target], check=False, text=True, capture_output=True
        )


def _summarize_statuses(output_root: Path, seeds: list[int]) -> dict[str, list[int]]:
    summary = {
        "running_seeds": [],
        "completed_seeds": [],
        "failed_seeds": [],
        "stalled_seeds": [],
        "required_artifact_complete_count": 0,
        "component_match_present_count": 0,
        "artifact_manifest_present_count": 0,
        "baseline_drift_zero_count": 0,
    }
    for seed in seeds:
        path = _status_path(output_root, seed)
        if not path.exists():
            continue
        payload = _load_json(path)
        state = str(payload.get("state", ""))
        if state == "running":
            summary["running_seeds"].append(seed)
        elif state == "success":
            summary["completed_seeds"].append(seed)
            if bool(payload.get("required_artifacts_complete", False)):
                summary["required_artifact_complete_count"] += 1
            if bool(payload.get("component_match_present", False)):
                summary["component_match_present_count"] += 1
            if bool(payload.get("artifact_manifest_present", False)):
                summary["artifact_manifest_present_count"] += 1
            if bool(payload.get("baseline_drift_zero", False)):
                summary["baseline_drift_zero_count"] += 1
        elif state == "failed":
            summary["failed_seeds"].append(seed)
        elif state == "stalled":
            summary["stalled_seeds"].append(seed)
    return summary


def _phase_progress_payload(summary: dict[str, Any]) -> dict[str, Any]:
    success_count = int(summary.get("success_count", 0) or 0)
    failed_count = len(summary.get("failed_seeds", []))
    running_count = len(summary.get("running_seeds", []))
    stalled_count = len(summary.get("stalled_seeds", []))
    expected = int(summary.get("total_seeds", 0) or 0)
    started = success_count + failed_count + running_count + stalled_count
    required_artifact_complete_count = int(summary.get("required_artifact_complete_count", 0) or 0)
    component_match_present_count = int(summary.get("component_match_present_count", 0) or 0)
    artifact_manifest_present_count = int(summary.get("artifact_manifest_present_count", 0) or 0)
    baseline_drift_zero_count = int(summary.get("baseline_drift_zero_count", 0) or 0)
    usable_seed_count = min(
        success_count,
        required_artifact_complete_count,
        component_match_present_count,
        artifact_manifest_present_count,
        baseline_drift_zero_count,
    )
    return {
        "expected": expected,
        "started": started,
        "success": success_count,
        "failed": failed_count,
        "running": running_count,
        "stalled": stalled_count,
        "usable_seed_count": usable_seed_count,
        "locked_minimum": DREAMER_MIN_LOCKED_SEEDS,
        "proof_minimum": DREAMER_MIN_PROOF_SEEDS,
        "locked_minimum_met": usable_seed_count >= DREAMER_MIN_LOCKED_SEEDS,
        "proof_minimum_met": usable_seed_count >= DREAMER_MIN_PROOF_SEEDS,
        "proof_phase": "official_only",
    }


def main() -> int:
    args = _parse_args()
    seeds = _parse_seed_list(args.seed_list)
    gpu_devices = _parse_gpu_devices(args.gpu_devices, parallelism=int(args.parallelism))
    args.output_root.mkdir(parents=True, exist_ok=True)
    recipe_hash = stable_recipe_hash(
        {
            **{
                "steps": int(args.steps),
                "task_id": args.task_id,
                "eval_episodes": int(args.eval_episodes),
            }
        }
    )

    pending = deque(seeds)
    running: dict[int, subprocess.Popen[str]] = {}
    running_gpu: dict[int, str] = {}
    attempts: dict[int, int] = {seed: 0 for seed in seeds}
    baseline = _load_baseline(args.output_root)

    if args.resume:
        status_summary = _summarize_statuses(args.output_root, seeds)
        pending = deque(
            seed
            for seed in seeds
            if seed not in status_summary["completed_seeds"]
            and seed not in status_summary["running_seeds"]
        )

    while pending or running:
        while pending and len(running) < int(args.parallelism):
            seed = pending.popleft()
            attempts[seed] += 1
            gpu_id = _select_gpu_id(gpu_devices, running_gpu)
            seed_dir = _seed_root(args.output_root, seed)
            seed_dir.mkdir(parents=True, exist_ok=True)
            command = _build_command(args, seed=seed)
            status = _initial_status(seed=seed, attempt=attempts[seed], command=command)
            status["state"] = "running"
            status["started_at"] = _now()
            status["last_heartbeat"] = _now()
            status["gpu_id"] = gpu_id
            _atomic_write_json(_status_path(args.output_root, seed), status)
            _append_batch_log(
                args.output_root, f"launched seed={seed} attempt={attempts[seed]} gpu={gpu_id}"
            )
            stdout = (seed_dir / "runner.stdout.log").open("w", encoding="utf-8")
            stderr = (seed_dir / "runner.stderr.log").open("w", encoding="utf-8")
            proc = subprocess.Popen(
                command,
                cwd=str(SCRIPT_DIR.parents[1]),
                text=True,
                stdout=stdout,
                stderr=stderr,
                env={**os.environ, **_launch_env(args, gpu_id=gpu_id)},
            )
            running[seed] = proc
            running_gpu[seed] = gpu_id

        time.sleep(max(1, int(args.poll_interval_sec)))

        for seed, proc in list(running.items()):
            status_path = _status_path(args.output_root, seed)
            status = _load_json(status_path)
            status["last_heartbeat"] = _now()
            seed_paths = _seed_runtime_paths(args.output_root, seed)
            latest_mtime = _latest_mtime(seed_paths)
            if latest_mtime and (time.time() - latest_mtime) > int(args.stale_seconds):
                proc.kill()
                status["state"] = "stalled"
                status["message"] = "No artifact updates within stale_seconds window."
                status["completed_at"] = _now()
                _atomic_write_json(status_path, status)
                _append_batch_log(
                    args.output_root,
                    f"stalled seed={seed} attempt={attempts[seed]} gpu={running_gpu.get(seed, '?')}",
                )
                del running[seed]
                running_gpu.pop(seed, None)
                if attempts[seed] < int(args.max_attempts):
                    pending.append(seed)
                continue

            return_code = proc.poll()
            if return_code is None:
                _atomic_write_json(status_path, status)
                continue

            status["exit_code"] = int(return_code)
            status["completed_at"] = _now()
            del running[seed]
            gpu_id = running_gpu.pop(seed, "?")

            if return_code != 0:
                status["state"] = "failed"
                status["message"] = f"wrapper exited with code {return_code}"
                status["required_artifacts_complete"] = False
                status["baseline_drift_zero"] = False
                status["component_match_present"] = False
                status["artifact_manifest_present"] = False
                _atomic_write_json(status_path, status)
                _append_batch_log(
                    args.output_root,
                    f"failed seed={seed} attempt={attempts[seed]} gpu={gpu_id} rc={return_code}",
                )
                if attempts[seed] < int(args.max_attempts):
                    pending.append(seed)
                continue

            missing = _missing_required_files(seed_paths)
            if missing:
                status["state"] = "failed"
                status["message"] = f"missing required files: {missing}"
                status["required_artifacts_complete"] = False
                status["baseline_drift_zero"] = False
                status["component_match_present"] = False
                status["artifact_manifest_present"] = False
                _atomic_write_json(status_path, status)
                _append_batch_log(
                    args.output_root, f"failed seed={seed} gpu={gpu_id} missing={','.join(missing)}"
                )
                if attempts[seed] < int(args.max_attempts):
                    pending.append(seed)
                continue

            component_report = _run_component_match(args, seed=seed)
            config_signature = _config_signature(seed_paths["config_yaml"])
            candidate = {
                "backend_kind": "jax_subprocess",
                "adapter_id": "official_dreamerv3_jax_subprocess",
                "recipe_hash": recipe_hash,
                "config_signature": config_signature,
                "command_signature": _command_signature(_build_command(args, seed=seed)),
            }
            if baseline is None:
                baseline = dict(candidate)
                baseline["seed"] = seed
                _write_baseline(args.output_root, baseline)
            else:
                mismatches = _validate_against_baseline(baseline=baseline, candidate=candidate)
                if mismatches:
                    status["state"] = "failed"
                    status["message"] = f"baseline mismatch: {mismatches}"
                    status["required_artifacts_complete"] = True
                    status["baseline_drift_zero"] = False
                    status["component_match_present"] = True
                    status["artifact_manifest_present"] = False
                    _atomic_write_json(status_path, status)
                    _append_batch_log(
                        args.output_root,
                        f"rejected seed={seed} gpu={gpu_id} mismatch={','.join(mismatches)}",
                    )
                    if attempts[seed] < int(args.max_attempts):
                        pending.append(seed)
                    continue

            manifest = _artifact_manifest_from_seed(
                args,
                seed=seed,
                recipe_hash=recipe_hash,
                component_report=component_report,
            )
            _atomic_write_json(_artifact_manifest_path(args.output_root, seed), manifest)
            _upload_seed_artifacts(args, seed=seed, manifest=manifest)

            status["state"] = "success"
            status["message"] = "completed"
            status["artifact_manifest"] = str(_artifact_manifest_path(args.output_root, seed))
            status["component_match_report"] = str(component_report)
            status["required_artifacts_complete"] = True
            status["baseline_drift_zero"] = True
            status["component_match_present"] = True
            status["artifact_manifest_present"] = True
            _atomic_write_json(status_path, status)
            _append_batch_log(
                args.output_root, f"finished seed={seed} attempt={attempts[seed]} gpu={gpu_id}"
            )

    summary = _summarize_statuses(args.output_root, seeds)
    summary["total_seeds"] = len(seeds)
    summary["success_count"] = len(summary["completed_seeds"])
    summary["recipe_hash"] = recipe_hash
    summary_path = args.output_root / "summary.json"
    summary["execution_result"] = normalize_dreamer_official_batch_summary(
        summary,
        summary_path=summary_path,
    ).to_dict()
    _atomic_write_json(summary_path, summary)
    _atomic_write_json(args.output_root / "phase_progress.json", _phase_progress_payload(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["success_count"] == len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
