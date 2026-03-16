#!/usr/bin/env python3
"""Monitor a DreamerV3 official batch run from local artifacts and optional S3."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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
    normalize_dreamer_batch_status,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--s3-prefix", type=str, default="")
    parser.add_argument("--stale-seconds", type=int, default=900)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _last_mtime(seed_root: Path) -> float:
    files = [path.stat().st_mtime for path in seed_root.rglob("*") if path.is_file()]
    return max(files) if files else 0.0


def _collect_local(output_root: Path, *, stale_seconds: int) -> dict[str, Any]:
    seed_dirs = sorted(path for path in output_root.glob("seed_*") if path.is_dir())
    now = datetime.now(timezone.utc).timestamp()
    states: dict[str, list[int]] = {
        "planned_seeds": [],
        "running_seeds": [],
        "completed_seeds": [],
        "failed_seeds": [],
        "stalled_seeds": [],
        "uploaded_seeds": [],
    }
    normalized_states: dict[str, list[int]] = {
        "queued_seeds": [],
        "running_seeds": [],
        "succeeded_seeds": [],
        "failed_seeds": [],
        "incomplete_seeds": [],
    }
    details: list[dict[str, Any]] = []
    usable_seed_count = 0
    for seed_dir in seed_dirs:
        try:
            seed = int(seed_dir.name.split("_", 1)[1])
        except Exception:
            continue
        status_path = seed_dir / "status.json"
        if not status_path.exists():
            states["planned_seeds"].append(seed)
            normalized_states["queued_seeds"].append(seed)
            continue
        payload = _load_json(status_path)
        raw_state = str(payload.get("state", "planned"))
        state = raw_state
        if state == "success":
            states["completed_seeds"].append(seed)
        elif state == "running":
            states["running_seeds"].append(seed)
        elif state == "failed":
            states["failed_seeds"].append(seed)
        elif state == "stalled":
            states["stalled_seeds"].append(seed)
        else:
            states["planned_seeds"].append(seed)
        if (seed_dir / "artifact_manifest.json").exists():
            states["uploaded_seeds"].append(seed)
        age = None
        last_mtime = _last_mtime(seed_dir)
        stale = False
        if last_mtime:
            age = max(0.0, now - last_mtime)
            if state == "running" and age > stale_seconds:
                stale = True
            if stale and seed not in states["stalled_seeds"]:
                states["stalled_seeds"].append(seed)
        normalized_input = dict(payload)
        if stale and raw_state == "running":
            normalized_input["state"] = "stalled"
        execution_result = normalize_dreamer_batch_status(normalized_input, status_path=status_path)
        status = execution_result.status
        reason_code = (
            execution_result.reason_code if execution_result.reason_code != "none" else None
        )
        message = execution_result.message
        normalized_states[f"{status}_seeds"].append(seed)
        details.append(
            {
                "seed": seed,
                "state": status,
                "raw_state": raw_state,
                "reason_code": reason_code,
                "message": message,
                "execution_result": execution_result.to_dict(),
                "attempt": payload.get("attempt"),
                "exit_code": payload.get("exit_code"),
                "last_heartbeat": payload.get("last_heartbeat"),
                "artifact_manifest": (seed_dir / "artifact_manifest.json").exists(),
                "age_sec": age,
            }
        )
        if (
            status == "succeeded"
            and payload.get("required_artifacts_complete")
            and payload.get("baseline_drift_zero")
            and payload.get("component_match_present")
            and payload.get("artifact_manifest_present")
        ):
            usable_seed_count += 1
    progress = {
        "expected": len(seed_dirs),
        "started": len(seed_dirs) - len(normalized_states["queued_seeds"]),
        "success": len(normalized_states["succeeded_seeds"]),
        "failed": len(normalized_states["failed_seeds"]),
        "running": len(normalized_states["running_seeds"]),
        "incomplete": len(normalized_states["incomplete_seeds"]),
        "usable_seed_count": usable_seed_count,
        "locked_minimum": DREAMER_MIN_LOCKED_SEEDS,
        "proof_minimum": DREAMER_MIN_PROOF_SEEDS,
        "locked_minimum_met": usable_seed_count >= DREAMER_MIN_LOCKED_SEEDS,
        "proof_minimum_met": usable_seed_count >= DREAMER_MIN_PROOF_SEEDS,
        "proof_phase": "official_only",
    }
    return {
        "states": states,
        "normalized_states": normalized_states,
        "progress": progress,
        "details": details,
    }


def _collect_s3(prefix: str) -> dict[str, Any]:
    if not prefix.strip():
        return {"present": False}
    proc = subprocess.run(
        ["aws", "s3", "ls", prefix.rstrip("/") + "/", "--recursive"],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        return {"present": False, "error": proc.stderr.strip()}
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    uploaded_seeds = sorted(
        {
            int(part.split("/", 1)[0].split("_", 1)[1])
            for line in lines
            for part in [line.split()[-1].removeprefix(prefix.rstrip("/") + "/")]
            if part.startswith("seed_") and "_" in part
        }
    )
    return {"present": bool(lines), "object_count": len(lines), "uploaded_seeds": uploaded_seeds}


def main() -> int:
    args = _parse_args()
    local = _collect_local(args.output_root.resolve(), stale_seconds=int(args.stale_seconds))
    s3 = _collect_s3(args.s3_prefix)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "output_root": str(args.output_root.resolve()),
        "local": local,
        "s3": s3,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
