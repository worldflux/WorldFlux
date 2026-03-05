#!/usr/bin/env python3
"""Generate reproducible parity fixtures used by release-gate dry-runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from worldflux.parity.harness import aggregate_runs, run_suite

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_GENERATED_AT = "2026-03-05T00:00:00Z"
FIXTURE_COMMIT = "release-gate-fixture.v1"
FIXTURE_NOTE = (
    "Deterministic release-gate fixture for local reproducibility only. "
    "Not proof-grade evidence."
)

FIXTURE_SPECS: dict[str, dict[str, Any]] = {
    "dreamer_atari100k": {
        "family": "dreamerv3",
        "tasks": ("atari_freeway", "atari_pong"),
        "upstream_scores": (
            {"task": "atari_freeway", "seed": 0, "step": 100000, "score": 30.0},
            {"task": "atari_freeway", "seed": 1, "step": 100000, "score": 31.0},
            {"task": "atari_pong", "seed": 0, "step": 100000, "score": 20.0},
            {"task": "atari_pong", "seed": 1, "step": 100000, "score": 22.0},
        ),
        "worldflux_scores": (
            {"task": "atari_freeway", "seed": 0, "step": 100000, "score": 30.0},
            {"task": "atari_freeway", "seed": 1, "step": 100000, "score": 31.0},
            {"task": "atari_pong", "seed": 0, "step": 100000, "score": 20.0},
            {"task": "atari_pong", "seed": 1, "step": 100000, "score": 22.0},
        ),
    },
    "tdmpc2_dmcontrol39": {
        "family": "tdmpc2",
        "tasks": ("cheetah-run", "dog-run"),
        "upstream_scores": (
            {"task": "cheetah-run", "seed": 1, "step": 4000000, "score": 710.0},
            {"task": "cheetah-run", "seed": 2, "step": 4000000, "score": 720.0},
            {"task": "dog-run", "seed": 1, "step": 4000000, "score": 850.0},
            {"task": "dog-run", "seed": 2, "step": 4000000, "score": 870.0},
        ),
        "worldflux_scores": (
            {"task": "cheetah-run", "seed": 1, "step": 4000000, "score": 710.0},
            {"task": "cheetah-run", "seed": 2, "step": 4000000, "score": 720.0},
            {"task": "dog-run", "seed": 1, "step": 4000000, "score": 850.0},
            {"task": "dog-run", "seed": 2, "step": 4000000, "score": 870.0},
        ),
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_relative(path: Path, *, repo_root: Path) -> str:
    return str(path.resolve().relative_to(repo_root.resolve()))


def _fixture_metadata() -> dict[str, Any]:
    return {
        "classification": "release_gate_fixture",
        "generated_by": "scripts/generate_release_parity_fixtures.py",
        "generated_at_utc": FIXTURE_GENERATED_AT,
        "proof_claim_allowed": False,
        "public_claims_allowed": False,
        "note": FIXTURE_NOTE,
    }


def _parity_paths(repo_root: Path) -> dict[str, Path]:
    parity_root = repo_root / "reports" / "parity"
    return {
        "fixtures_root": parity_root / "fixtures",
        "runs_root": parity_root / "runs",
        "aggregate_path": parity_root / "aggregate.json",
        "lock_path": parity_root / "upstream_lock.json",
    }


def _canonicalize_run_artifact(*, run_path: Path, repo_root: Path, suite_id: str) -> None:
    payload = _read_json(run_path)
    fixture_dir = repo_root / "reports/parity/fixtures"

    payload["generated_at_utc"] = FIXTURE_GENERATED_AT
    payload["evaluation_manifest"] = {
        "runner": "worldflux.parity.run_suite",
        "generation_mode": "release_gate_fixture",
        "generated_at_utc": FIXTURE_GENERATED_AT,
        "source": "scripts/generate_release_parity_fixtures.py",
        "reproducible": True,
        "note": FIXTURE_NOTE,
    }
    payload["release_fixture"] = _fixture_metadata()

    sources = payload.get("sources")
    if isinstance(sources, dict):
        upstream = sources.get("upstream")
        if isinstance(upstream, dict):
            upstream["path"] = _repo_relative(
                fixture_dir / f"{suite_id}_upstream.json",
                repo_root=repo_root,
            )
        worldflux = sources.get("worldflux")
        if isinstance(worldflux, dict):
            worldflux["path"] = _repo_relative(
                fixture_dir / f"{suite_id}_worldflux.json",
                repo_root=repo_root,
            )
            worldflux["commit"] = FIXTURE_COMMIT

    suite_lock_ref = payload.get("suite_lock_ref")
    if isinstance(suite_lock_ref, dict):
        suite_lock_ref["lock_path"] = "reports/parity/upstream_lock.json"

    _write_json(run_path, payload)


def _canonicalize_aggregate_artifact(*, aggregate_path: Path) -> None:
    payload = _read_json(aggregate_path)
    payload["generated_at_utc"] = FIXTURE_GENERATED_AT
    payload["release_fixture"] = _fixture_metadata()
    payload["run_paths"] = [f"reports/parity/runs/{suite_id}.json" for suite_id in FIXTURE_SPECS]

    suites = payload.get("suites")
    if isinstance(suites, list):
        suites.sort(key=lambda row: str(row.get("suite_id", "")) if isinstance(row, dict) else "")
        for row in suites:
            if not isinstance(row, dict):
                continue
            suite_id = str(row.get("suite_id", "")).strip()
            if suite_id:
                row["run_path"] = f"reports/parity/runs/{suite_id}.json"

    families = payload.get("families")
    if isinstance(families, list):
        families.sort(key=lambda row: str(row.get("family", "")) if isinstance(row, dict) else "")

    _write_json(aggregate_path, payload)


def generate_release_parity_fixtures(*, repo_root: Path) -> tuple[list[Path], Path]:
    paths = _parity_paths(repo_root)
    fixtures_root = paths["fixtures_root"]
    runs_root = paths["runs_root"]
    aggregate_path = paths["aggregate_path"]
    lock_path = paths["lock_path"]

    lock_payload = _read_json(lock_path)
    lock_suites = lock_payload.get("suites", {})
    if not isinstance(lock_suites, dict):
        raise ValueError("reports/parity/upstream_lock.json must define object at suites")

    fixtures_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    run_paths: list[Path] = []
    for suite_id, spec in FIXTURE_SPECS.items():
        lock_entry = lock_suites.get(suite_id)
        if not isinstance(lock_entry, dict):
            raise ValueError(f"upstream lock missing suite {suite_id}")

        suite_path = fixtures_root / f"{suite_id}_suite.json"
        upstream_path = fixtures_root / f"{suite_id}_upstream.json"
        worldflux_path = fixtures_root / f"{suite_id}_worldflux.json"
        run_path = runs_root / f"{suite_id}.json"

        _write_json(upstream_path, {"scores": list(spec["upstream_scores"])})
        _write_json(worldflux_path, {"scores": list(spec["worldflux_scores"])})
        _write_json(
            suite_path,
            {
                "suite_id": suite_id,
                "family": spec["family"],
                "metric": "episode_return",
                "higher_is_better": True,
                "margin_ratio": 0.05,
                "confidence": 0.95,
                "tasks": list(spec["tasks"]),
                "release_fixture": _fixture_metadata(),
                "upstream": {
                    "repo": str(lock_entry.get("repo", "")),
                    "commit": str(lock_entry.get("commit", "")),
                    "format": "canonical_json",
                    "path": upstream_path.name,
                },
                "worldflux": {
                    "repo": "https://github.com/worldflux/WorldFlux",
                    "commit": FIXTURE_COMMIT,
                    "format": "canonical_json",
                    "path": worldflux_path.name,
                },
            },
        )

        run_suite(
            suite_path,
            output_path=run_path,
            upstream_lock_path=lock_path,
        )
        _canonicalize_run_artifact(
            run_path=run_path,
            repo_root=repo_root,
            suite_id=suite_id,
        )
        run_paths.append(run_path)

    aggregate_runs(run_paths, output_path=aggregate_path)
    _canonicalize_aggregate_artifact(aggregate_path=aggregate_path)
    return run_paths, aggregate_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root containing reports/parity/upstream_lock.json.",
    )
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()

    try:
        run_paths, aggregate_path = generate_release_parity_fixtures(repo_root=repo_root)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[release-parity-fixtures] failed: {exc}")
        return 1

    print("[release-parity-fixtures] wrote deterministic release-gate fixtures")
    for run_path in run_paths:
        print(f"  - {run_path.relative_to(repo_root)}")
    print(f"  - {aggregate_path.relative_to(repo_root)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
