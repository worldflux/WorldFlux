"""Tests for resource pinning command generation in AWS parity orchestrator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


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


def test_build_remote_commands_includes_thread_and_cpu_gpu_pinning() -> None:
    mod = _load_module()

    shard = mod.ShardPlan(
        shard_id=7,
        instance_id="i-abcd",
        task_ids=("dog-run",),
        shard_run_id="run_a_shard07",
        estimated_cost=1.0,
        estimated_duration_sec=1.0,
        seed_values=(3,),
        systems=("worldflux",),
        gpu_slot=5,
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
        skip_bootstrap=False,
    )

    joined = "\n".join(commands)
    assert "export OMP_NUM_THREADS=1" in joined
    assert "export MKL_NUM_THREADS=1" in joined
    assert "export OPENBLAS_NUM_THREADS=1" in joined
    assert "export NUMEXPR_NUM_THREADS=1" in joined
    assert "export CUDA_VISIBLE_DEVICES=5" in joined
    assert "taskset -c 60-71" in joined  # gpu slot 5 in p4d_8gpu_12vcpu map
    assert "--systems worldflux" in joined
    assert "--seed-list 3" in joined
    assert "--artifact-retention minimal" in joined
    assert "required_root_kb=41943040" in joined
    assert "required_data_kb=838860800" in joined
    assert "set -eu" in joined


def test_cpu_affinity_policy_rejects_non_p4d_gpu_workers() -> None:
    mod = _load_module()

    with pytest.raises(RuntimeError, match="requires exactly 8 GPUs"):
        mod._validate_cpu_affinity_policy(
            policy="p4d_8gpu_12vcpu",
            instance_gpu_counts={"i-g6": 1},
            instance_types={"i-g6": "g6.2xlarge"},
        )
