"""Tests for bootstrap-root reuse behavior in AWS parity orchestrator."""

from __future__ import annotations

import importlib.util
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


def test_build_remote_commands_reuses_bootstrap_root_across_run_ids() -> None:
    mod = _load_module()

    shard = mod.ShardPlan(
        shard_id=0,
        instance_id="i-abcd",
        task_ids=("dog-run",),
        shard_run_id="run_a_shard00",
        estimated_cost=1.0,
        estimated_duration_sec=1.0,
        seed_values=(3,),
        systems=("official",),
        gpu_slot=0,
    )

    commands_a = mod._build_remote_commands(
        shard=shard,
        manifest_rel="scripts/parity/manifests/official_vs_worldflux_v1.yaml",
        run_id="run_a",
        s3_prefix="s3://bucket/run_a",
        device="cuda",
        max_retries=2,
        workspace_root="/opt/parity",
        bootstrap_root="/opt/parity/bootstrap",
        worldflux_sha="",
        dreamer_sha="dreamer_sha",
        tdmpc2_sha="tdmpc2_sha",
        sync_interval_sec=60,
        resume_from_s3=True,
        thread_limit_profile="strict1",
        cpu_affinity_policy="p4d_8gpu_12vcpu",
        skip_bootstrap=False,
    )
    commands_b = mod._build_remote_commands(
        shard=shard,
        manifest_rel="scripts/parity/manifests/official_vs_worldflux_v1.yaml",
        run_id="run_b",
        s3_prefix="s3://bucket/run_b",
        device="cuda",
        max_retries=2,
        workspace_root="/opt/parity",
        bootstrap_root="/opt/parity/bootstrap",
        worldflux_sha="",
        dreamer_sha="dreamer_sha",
        tdmpc2_sha="tdmpc2_sha",
        sync_interval_sec=60,
        resume_from_s3=True,
        thread_limit_profile="strict1",
        cpu_affinity_policy="p4d_8gpu_12vcpu",
        skip_bootstrap=False,
    )

    joined_a = "\n".join(commands_a)
    joined_b = "\n".join(commands_b)
    assert "cd /opt/parity/bootstrap/i-abcd" in joined_a
    assert "cd /opt/parity/bootstrap/i-abcd" in joined_b
    assert "mkdir -p /opt/parity/run_a/i-abcd" in joined_a
    assert "mkdir -p /opt/parity/run_b/i-abcd" in joined_b


def test_build_remote_commands_skip_bootstrap_avoids_clone_and_install() -> None:
    mod = _load_module()
    shard = mod.ShardPlan(
        shard_id=7,
        instance_id="i-abcd",
        task_ids=("atari100k_pong",),
        shard_run_id="run_a_shard07",
        estimated_cost=1.0,
        estimated_duration_sec=1.0,
        seed_values=(1,),
        systems=("worldflux",),
        gpu_slot=2,
    )
    commands = mod._build_remote_commands(
        shard=shard,
        manifest_rel="scripts/parity/manifests/official_vs_worldflux_v1.yaml",
        run_id="run_a",
        s3_prefix="s3://bucket/run_a",
        device="cuda",
        max_retries=2,
        workspace_root="/opt/parity",
        bootstrap_root="/opt/parity/bootstrap",
        worldflux_sha="",
        dreamer_sha="dreamer_sha",
        tdmpc2_sha="tdmpc2_sha",
        sync_interval_sec=60,
        resume_from_s3=True,
        thread_limit_profile="strict1",
        cpu_affinity_policy="p4d_8gpu_12vcpu",
        skip_bootstrap=True,
    )
    joined = "\n".join(commands)
    assert "Missing bootstrap repo: worldflux" in joined
    assert "git clone https://github.com/worldflux/WorldFlux.git worldflux" not in joined
    assert "python -m pip install -e ." not in joined
