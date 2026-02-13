"""Tests for parity artifact release validation script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def _load_module(script_name: str):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module: {script_name}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_validate_parity_artifacts_passes_for_matching_lock(tmp_path: Path) -> None:
    mod = _load_module("validate_parity_artifacts.py")

    run_a = tmp_path / "runs" / "dreamer_atari100k.json"
    run_b = tmp_path / "runs" / "tdmpc2_dmcontrol39.json"
    aggregate = tmp_path / "aggregate.json"
    lock = tmp_path / "upstream_lock.json"

    _write_json(
        lock,
        {
            "schema_version": "worldflux.parity.lock.v1",
            "suites": {
                "dreamer_atari100k": {"commit": "aaa"},
                "tdmpc2_dmcontrol39": {"commit": "bbb"},
            },
        },
    )

    def _run_payload(suite_id: str, commit: str) -> dict:
        return {
            "suite": {"suite_id": suite_id},
            "sources": {"upstream": {"commit": commit}},
            "counts": {"upstream_only": 0, "worldflux_only": 0},
            "stats": {"pass_non_inferiority": True},
            "evaluation_manifest": {"runner": "worldflux.parity.run_suite"},
            "artifact_integrity": {"suite_sha256": "x"},
            "suite_lock_ref": {"matches_lock": True},
        }

    _write_json(run_a, _run_payload("dreamer_atari100k", "aaa"))
    _write_json(run_b, _run_payload("tdmpc2_dmcontrol39", "bbb"))
    _write_json(
        aggregate,
        {
            "all_suites_pass": True,
            "suites": [
                {"suite_id": "dreamer_atari100k", "pass_non_inferiority": True},
                {"suite_id": "tdmpc2_dmcontrol39", "pass_non_inferiority": True},
            ],
        },
    )

    failures, report = mod.validate_parity_artifacts(
        run_paths=[run_a, run_b],
        aggregate_path=aggregate,
        lock_path=lock,
        required_suites=["dreamer_atari100k", "tdmpc2_dmcontrol39"],
        max_missing_pairs=0,
    )
    assert failures == []
    assert report["success"] is True


def test_validate_parity_artifacts_fails_on_commit_mismatch(tmp_path: Path) -> None:
    mod = _load_module("validate_parity_artifacts.py")

    run_a = tmp_path / "runs" / "dreamer_atari100k.json"
    run_b = tmp_path / "runs" / "tdmpc2_dmcontrol39.json"
    aggregate = tmp_path / "aggregate.json"
    lock = tmp_path / "upstream_lock.json"

    _write_json(
        lock,
        {
            "schema_version": "worldflux.parity.lock.v1",
            "suites": {
                "dreamer_atari100k": {"commit": "aaa"},
                "tdmpc2_dmcontrol39": {"commit": "bbb"},
            },
        },
    )
    _write_json(
        run_a,
        {
            "suite": {"suite_id": "dreamer_atari100k"},
            "sources": {"upstream": {"commit": "mismatch"}},
            "counts": {"upstream_only": 0, "worldflux_only": 0},
            "stats": {"pass_non_inferiority": True},
            "evaluation_manifest": {"runner": "worldflux.parity.run_suite"},
            "artifact_integrity": {"suite_sha256": "x"},
            "suite_lock_ref": {"matches_lock": False},
        },
    )
    _write_json(
        run_b,
        {
            "suite": {"suite_id": "tdmpc2_dmcontrol39"},
            "sources": {"upstream": {"commit": "bbb"}},
            "counts": {"upstream_only": 0, "worldflux_only": 0},
            "stats": {"pass_non_inferiority": True},
            "evaluation_manifest": {"runner": "worldflux.parity.run_suite"},
            "artifact_integrity": {"suite_sha256": "x"},
            "suite_lock_ref": {"matches_lock": True},
        },
    )
    _write_json(
        aggregate,
        {
            "all_suites_pass": True,
            "suites": [
                {"suite_id": "dreamer_atari100k", "pass_non_inferiority": True},
                {"suite_id": "tdmpc2_dmcontrol39", "pass_non_inferiority": True},
            ],
        },
    )

    failures, report = mod.validate_parity_artifacts(
        run_paths=[run_a, run_b],
        aggregate_path=aggregate,
        lock_path=lock,
        required_suites=["dreamer_atari100k", "tdmpc2_dmcontrol39"],
        max_missing_pairs=0,
    )
    assert report["success"] is False
    assert any("commit mismatch" in failure for failure in failures)
