"""Tests for parity suite policy coverage validation script."""

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


def test_validate_parity_suite_coverage_passes(tmp_path: Path) -> None:
    mod = _load_module("check_parity_suite_coverage.py")
    policy = tmp_path / "suite_policy.json"
    lock = tmp_path / "upstream_lock.json"
    aggregate = tmp_path / "aggregate.json"

    _write_json(
        policy,
        {
            "schema_version": "worldflux.parity.policy.v1",
            "families": {
                "dreamerv3": {"stage": "required", "suites": ["dreamer_atari100k"]},
                "tdmpc2": {"stage": "required", "suites": ["tdmpc2_dmcontrol39"]},
            },
        },
    )
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
        aggregate,
        {
            "all_suites_pass": True,
            "suites": [
                {"suite_id": "dreamer_atari100k", "pass_non_inferiority": True},
                {"suite_id": "tdmpc2_dmcontrol39", "pass_non_inferiority": True},
            ],
        },
    )

    failures, report = mod.validate_parity_suite_coverage(
        policy_path=policy,
        lock_path=lock,
        aggregate_path=aggregate,
        enforce_pass=True,
    )
    assert failures == []
    assert report["success"] is True


def test_validate_parity_suite_coverage_fails_when_required_suite_missing_in_lock(
    tmp_path: Path,
) -> None:
    mod = _load_module("check_parity_suite_coverage.py")
    policy = tmp_path / "suite_policy.json"
    lock = tmp_path / "upstream_lock.json"

    _write_json(
        policy,
        {
            "schema_version": "worldflux.parity.policy.v1",
            "families": {
                "dreamerv3": {"stage": "required", "suites": ["dreamer_atari100k"]},
            },
        },
    )
    _write_json(lock, {"schema_version": "worldflux.parity.lock.v1", "suites": {}})

    failures, report = mod.validate_parity_suite_coverage(
        policy_path=policy,
        lock_path=lock,
        aggregate_path=None,
        enforce_pass=False,
    )
    assert report["success"] is False
    assert any("required suite missing in upstream lock" in failure for failure in failures)
