"""Tests for two-stage proof gate behavior in AWS parity orchestrator."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "parity"
        / "aws_distributed_orchestrator.py"
    )
    spec = importlib.util.spec_from_file_location("aws_distributed_orchestrator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aws_distributed_orchestrator")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        region="us-west-2",
        instance_ids="",
        official_instance_ids="",
        worldflux_instance_ids="",
        manifest=tmp_path / "manifest.json",
        full_manifest=tmp_path / "full_manifest.json",
        run_id="run_demo",
        s3_prefix="s3://bucket/run_demo",
        device="cuda",
        systems="official,worldflux",
        seed_list="",
        max_retries=2,
        timeout_seconds=172800,
        poll_interval_sec=30,
        sync_interval_sec=60,
        resume_from_s3=True,
        worldflux_sha="",
        dreamer_sha="dsha",
        tdmpc2_sha="tsha",
        workspace_root="/opt/parity",
        bootstrap_root="/opt/parity/bootstrap",
        output_dir=tmp_path / "reports",
        skip_bootstrap=False,
        wait=True,
        sharding_mode="seed_system",
        seed_shard_unit="pair",
        phase_plan="two_stage_proof",
        phase_gate="strict_pass",
        pilot_seed_list="0,1,2",
        seed_min=20,
        seed_max=50,
        power_target=0.8,
        alpha=0.05,
        equivalence_margin=0.05,
        thread_limit_profile="strict1",
        cpu_affinity_policy="p4d_8gpu_12vcpu",
        gpu_slots_per_instance=8,
        auto_provision=False,
        auto_terminate=False,
        fleet_size=4,
        fleet_split="2,2",
        instance_type="p4d.24xlarge",
        image_id="",
        subnet_id="",
        security_group_ids="",
        iam_instance_profile="",
        key_name="",
        volume_size_gb=200,
        provision_timeout_sec=1800,
    )


def test_two_stage_gate_blocks_suite65_when_full_phase_fails_gate(
    monkeypatch, tmp_path: Path
) -> None:
    mod = _load_module()

    (tmp_path / "manifest.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "full_manifest.json").write_text("{}\n", encoding="utf-8")

    pilot_runs = tmp_path / "pilot_runs.jsonl"
    pilot_runs.write_text("\n", encoding="utf-8")

    full_equivalence = tmp_path / "full_equivalence.json"
    full_equivalence.write_text(
        json.dumps({"global": {"parity_pass_final": False}}), encoding="utf-8"
    )

    called_run_ids: list[str] = []

    def fake_resolve_fleets(_args):
        return (["i-off"], ["i-wf"], ["i-off", "i-wf"], [])

    def fake_estimate_required_seed_count_from_runs(**_kwargs):
        return 20

    def fake_execute_phase(*, run_id: str, **_kwargs):
        called_run_ids.append(run_id)
        summary_path = tmp_path / f"{run_id}_summary.json"
        if run_id.endswith("_pilot"):
            summary = {"artifacts": {"merged_runs": str(pilot_runs)}}
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return mod.PhaseResult(
                run_id=run_id, return_code=0, summary_path=summary_path, summary=summary
            )
        if run_id.endswith("_full"):
            summary = {"artifacts": {"equivalence_report": str(full_equivalence)}}
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            return mod.PhaseResult(
                run_id=run_id, return_code=0, summary_path=summary_path, summary=summary
            )
        summary = {"artifacts": {}}
        summary_path.write_text(json.dumps(summary), encoding="utf-8")
        return mod.PhaseResult(
            run_id=run_id, return_code=0, summary_path=summary_path, summary=summary
        )

    monkeypatch.setattr(mod, "_resolve_fleets", fake_resolve_fleets)
    monkeypatch.setattr(
        mod,
        "_estimate_required_seed_count_from_runs",
        fake_estimate_required_seed_count_from_runs,
    )
    monkeypatch.setattr(mod, "_execute_phase", fake_execute_phase)

    args = _args(tmp_path)
    rc = mod._run_two_stage_proof(args)
    assert rc == 1
    assert called_run_ids == ["run_demo_pilot", "run_demo_full"]

    summary_path = args.output_dir / args.run_id / "two_stage_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["gate"]["strict_pass"] is False
    assert summary["gate"]["run_full_suite"] is False
