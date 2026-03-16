"""Tests for DreamerV3 official batch runner helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module(name: str, relative: str):
    script_path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_normalized_config_text_strips_dynamic_keys() -> None:
    mod = _load_module(
        "run_dreamerv3_official_batch", "scripts/parity/run_dreamerv3_official_batch.py"
    )
    raw = "logdir: /tmp/a\nseed: 7\nreplica: 0\nbatch_size: 16\n"
    normalized = mod._normalized_config_text(raw)
    assert "logdir:" not in normalized
    assert "seed:" not in normalized
    assert "replica:" not in normalized
    assert "batch_size: 16" in normalized


def test_validate_against_baseline_reports_mismatched_fields() -> None:
    mod = _load_module(
        "run_dreamerv3_official_batch", "scripts/parity/run_dreamerv3_official_batch.py"
    )
    baseline = {
        "backend_kind": "jax_subprocess",
        "adapter_id": "official_dreamerv3_jax_subprocess",
        "recipe_hash": "abc",
        "config_signature": "cfg1",
        "command_signature": "cmd1",
    }
    candidate = {
        "backend_kind": "jax_subprocess",
        "adapter_id": "official_dreamerv3_jax_subprocess",
        "recipe_hash": "abc",
        "config_signature": "cfg2",
        "command_signature": "cmd1",
    }
    assert mod._validate_against_baseline(baseline=baseline, candidate=candidate) == [
        "config_signature"
    ]


def test_build_command_uses_official_wrapper_and_required_flags(tmp_path: Path) -> None:
    mod = _load_module(
        "run_dreamerv3_official_batch", "scripts/parity/run_dreamerv3_official_batch.py"
    )
    args = mod._parse_args.__wrapped__ if hasattr(mod._parse_args, "__wrapped__") else None
    _ = args

    class _Args:
        repo_root = tmp_path / "dreamerv3-official"
        output_root = tmp_path / "reports"
        task_id = "atari100k_pong"
        steps = 110000
        device = "cuda"
        eval_episodes = 1
        python_executable = "python3"
        mock = False

    command = mod._build_command(_Args(), seed=3)
    assert "official_dreamerv3.py" in " ".join(command)
    assert "--task-id" in command
    assert "atari100k_pong" in command
    assert "--steps" in command
    assert "110000" in command


def test_batch_summary_can_be_normalized_to_execution_result(tmp_path: Path) -> None:
    mod = _load_module(
        "run_dreamerv3_official_batch", "scripts/parity/run_dreamerv3_official_batch.py"
    )
    summary = {
        "total_seeds": 10,
        "success_count": 10,
        "completed_seeds": list(range(10)),
        "failed_seeds": [],
        "stalled_seeds": [],
        "required_artifact_complete_count": 10,
        "component_match_present_count": 10,
        "artifact_manifest_present_count": 10,
        "baseline_drift_zero_count": 10,
    }
    result = mod.normalize_dreamer_official_batch_summary(
        summary, summary_path=tmp_path / "summary.json"
    )
    assert result.status == "succeeded"
    assert result.reason_code == "none"
    phase_progress = mod._phase_progress_payload(summary)
    assert phase_progress["expected"] == 10
    assert phase_progress["success"] == 10
    assert phase_progress["usable_seed_count"] == 10
    assert phase_progress["locked_minimum_met"] is True
    assert phase_progress["proof_minimum_met"] is False
    assert phase_progress["proof_phase"] == "official_only"
