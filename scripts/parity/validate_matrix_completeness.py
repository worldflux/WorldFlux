#!/usr/bin/env python3
"""Validate parity run matrix completeness (task x seed x system)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-plan", type=Path, default=None)
    parser.add_argument("--run-context", type=Path, default=None)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--max-missing-pairs", type=int, default=0)
    parser.add_argument("--systems", type=str, default="official,worldflux")
    parser.add_argument(
        "--rerun-manifest-output",
        type=Path,
        default=None,
        help="Optional path to emit a rerun-focused manifest for missing task/seed pairs.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"JSON root must be object: {path}")
    return payload


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _parse_manifest(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Manifest must be JSON, or install pyyaml for YAML support."
            ) from exc
        payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise RuntimeError("Manifest root must be object.")
    return payload


def _manifest_task_ids(payload: dict[str, Any]) -> list[str]:
    tasks_raw = payload.get("tasks", [])
    if not isinstance(tasks_raw, list):
        raise RuntimeError("manifest.tasks must be list")

    task_ids: list[str] = []
    for task in tasks_raw:
        if not isinstance(task, dict):
            continue
        task_id = task.get("task_id")
        if isinstance(task_id, str) and task_id.strip():
            task_ids.append(task_id.strip())
    return sorted(set(task_ids))


def _write_rerun_manifest(
    *,
    manifest_payload: dict[str, Any],
    missing_entries: list[dict[str, Any]],
    output_path: Path,
) -> None:
    missing_tasks = {
        str(entry.get("task_id", "")).strip()
        for entry in missing_entries
        if str(entry.get("task_id", "")).strip()
    }
    missing_seeds = sorted(
        {
            int(entry.get("seed", -1))
            for entry in missing_entries
            if isinstance(entry.get("seed"), int)
        }
    )
    tasks_raw = manifest_payload.get("tasks", [])
    selected_tasks = [
        task
        for task in tasks_raw
        if isinstance(task, dict) and str(task.get("task_id", "")).strip() in missing_tasks
    ]

    seed_policy_raw = manifest_payload.get("seed_policy", {})
    seed_policy: dict[str, Any]
    if isinstance(seed_policy_raw, dict):
        seed_policy = dict(seed_policy_raw)
    else:
        seed_policy = {}
    seed_policy["mode"] = "fixed"
    seed_policy["values"] = missing_seeds

    rerun_payload: dict[str, Any] = dict(manifest_payload)
    rerun_payload["tasks"] = selected_tasks
    rerun_payload["seed_policy"] = seed_policy
    rerun_payload["rerun_metadata"] = {
        "generated_from": str(manifest_payload.get("schema_version", "unknown")),
        "missing_pairs": len(missing_entries),
        "missing_tasks": sorted(missing_tasks),
        "missing_seeds": missing_seeds,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rerun_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_seed_csv(raw: str) -> list[int]:
    if not raw.strip():
        return []
    return sorted({int(token.strip()) for token in raw.split(",") if token.strip()})


def _resolve_seeds(
    *,
    rows: list[dict[str, Any]],
    seed_plan: dict[str, Any] | None,
    run_context: dict[str, Any] | None,
    override: list[int],
) -> list[int]:
    if override:
        return sorted(set(override))

    if isinstance(seed_plan, dict):
        values = seed_plan.get("seed_values")
        if isinstance(values, list) and all(isinstance(v, int) for v in values):
            return sorted(set(values))

    if isinstance(run_context, dict):
        values = run_context.get("seeds")
        if isinstance(values, list) and all(isinstance(v, int) for v in values):
            return sorted(set(values))

    discovered = sorted(
        {
            int(row.get("seed", -1))
            for row in rows
            if "seed" in row and int(row.get("seed", -1)) >= 0
        }
    )
    return discovered


def _resolve_tasks(manifest_tasks: list[str], run_context: dict[str, Any] | None) -> list[str]:
    if isinstance(run_context, dict):
        selected = run_context.get("selected_tasks")
        if isinstance(selected, list):
            values = [str(v).strip() for v in selected if str(v).strip()]
            if values:
                return sorted(set(values))
    return sorted(set(manifest_tasks))


def main() -> int:
    args = _parse_args()
    if args.max_missing_pairs < 0:
        raise SystemExit("--max-missing-pairs must be >= 0")

    manifest_payload = _parse_manifest(args.manifest)
    manifest_tasks = _manifest_task_ids(manifest_payload)
    rows = _load_jsonl(args.runs)

    seed_plan = _load_json(args.seed_plan) if args.seed_plan and args.seed_plan.exists() else None
    run_context = (
        _load_json(args.run_context) if args.run_context and args.run_context.exists() else None
    )

    tasks = _resolve_tasks(manifest_tasks, run_context)
    seeds = _resolve_seeds(
        rows=rows,
        seed_plan=seed_plan,
        run_context=run_context,
        override=_parse_seed_csv(args.seeds),
    )
    systems = [item.strip() for item in args.systems.split(",") if item.strip()]

    expected = {
        (task_id, int(seed), system) for task_id in tasks for seed in seeds for system in systems
    }

    success = {
        (
            str(row.get("task_id", "")),
            int(row.get("seed", -1)),
            str(row.get("system", "")),
        )
        for row in rows
        if row.get("status") == "success"
    }

    missing = sorted(expected - success)

    failed_records = [row for row in rows if row.get("status") == "failed"]

    report = {
        "schema_version": "parity.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "runs": str(args.runs),
        "task_count": len(tasks),
        "seed_count": len(seeds),
        "systems": systems,
        "expected_pairs": len(expected),
        "success_pairs": len(success & expected),
        "missing_pairs": len(missing),
        "missing_entries": [
            {"task_id": task, "seed": seed, "system": system} for task, seed, system in missing
        ],
        "failed_records": len(failed_records),
        "pass": len(missing) <= int(args.max_missing_pairs),
        "max_missing_pairs": int(args.max_missing_pairs),
        "tasks": tasks,
        "seeds": seeds,
    }

    if args.rerun_manifest_output is not None and missing:
        _write_rerun_manifest(
            manifest_payload=manifest_payload,
            missing_entries=report["missing_entries"],
            output_path=args.rerun_manifest_output,
        )
        report["rerun_manifest"] = str(args.rerun_manifest_output)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
