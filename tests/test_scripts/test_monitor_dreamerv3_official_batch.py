"""Tests for DreamerV3 official batch monitor."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "monitor_dreamerv3_official_batch.py"
    )
    spec = importlib.util.spec_from_file_location("monitor_dreamerv3_official_batch", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load monitor script")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_collect_local_summarizes_seed_states(tmp_path: Path) -> None:
    mod = _load_module()
    root = tmp_path / "batch"
    for seed, state in ((0, "success"), (1, "running"), (2, "failed")):
        seed_dir = root / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        (seed_dir / "status.json").write_text(
            json.dumps({"seed": seed, "state": state, "attempt": 1, "exit_code": 0}),
            encoding="utf-8",
        )
        (seed_dir / "runner.stdout.log").write_text("x", encoding="utf-8")
        if state == "success":
            (seed_dir / "artifact_manifest.json").write_text("{}", encoding="utf-8")

    summary = mod._collect_local(root, stale_seconds=900)
    assert summary["states"]["completed_seeds"] == [0]
    assert summary["states"]["running_seeds"] == [1]
    assert summary["states"]["failed_seeds"] == [2]
    assert summary["states"]["uploaded_seeds"] == [0]
    assert summary["normalized_states"]["succeeded_seeds"] == [0]
    assert summary["normalized_states"]["running_seeds"] == [1]
    assert summary["normalized_states"]["failed_seeds"] == [2]
    assert summary["details"][0]["raw_state"] == "success"
    assert summary["details"][0]["state"] == "succeeded"
    assert summary["details"][0]["execution_result"]["status"] == "succeeded"
    assert summary["progress"]["expected"] == 3
    assert summary["progress"]["started"] == 3
    assert summary["progress"]["success"] == 1
    assert summary["progress"]["usable_seed_count"] == 0
    assert summary["progress"]["proof_phase"] == "official_only"
