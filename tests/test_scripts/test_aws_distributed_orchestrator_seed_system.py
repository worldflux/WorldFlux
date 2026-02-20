"""Tests for seed_system sharding in AWS parity orchestrator."""

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
        / "aws_distributed_orchestrator.py"
    )
    spec = importlib.util.spec_from_file_location("aws_distributed_orchestrator", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load aws_distributed_orchestrator")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_seed_system_sharding_routes_systems_to_split_fleets(tmp_path: Path) -> None:
    mod = _load_module()

    manifest = {
        "schema_version": "parity.manifest.v1",
        "seed_policy": {
            "mode": "fixed",
            "values": [0, 1],
            "pilot_seeds": 10,
            "min_seeds": 20,
            "max_seeds": 50,
            "power_target": 0.8,
        },
        "tasks": [
            {
                "task_id": "atari100k_pong",
                "family": "dreamerv3",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_dreamerv3",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')", "--steps", "110000"],
                },
                "worldflux": {
                    "adapter": "worldflux_dreamerv3_native",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')", "--steps", "110000"],
                },
            },
            {
                "task_id": "dog-run",
                "family": "tdmpc2",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_tdmpc2",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')", "--steps", "7000000"],
                },
                "worldflux": {
                    "adapter": "worldflux_tdmpc2_native",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')", "--steps", "7000000"],
                },
            },
        ],
    }

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    task_costs = mod._manifest_task_costs(manifest_path)
    shards = mod._build_seed_system_shards(
        "run_x",
        task_costs,
        seed_values=[0, 1],
        systems=("official", "worldflux"),
        official_instances=["i-off0", "i-off1"],
        worldflux_instances=["i-wf0", "i-wf1"],
        gpu_slots_per_instance=2,
        seed_shard_unit="packed",
    )

    assert len(shards) == 2 * 2  # tasks * systems

    seen = set()
    for shard in shards:
        assert len(shard.task_ids) == 1
        assert len(shard.seed_values) == 2
        assert len(shard.systems) == 1
        assert shard.gpu_slot is not None

        key = (shard.task_ids[0], shard.systems[0])
        assert key not in seen
        seen.add(key)

        if shard.systems[0] == "official":
            assert shard.instance_id in {"i-off0", "i-off1"}
        else:
            assert shard.instance_id in {"i-wf0", "i-wf1"}

    assert len(seen) == len(shards)


def test_seed_system_pair_unit_splits_each_seed_into_own_shard(tmp_path: Path) -> None:
    mod = _load_module()

    manifest = {
        "schema_version": "parity.manifest.v1",
        "seed_policy": {
            "mode": "fixed",
            "values": [0, 1, 2],
            "pilot_seeds": 10,
            "min_seeds": 20,
            "max_seeds": 50,
            "power_target": 0.8,
        },
        "tasks": [
            {
                "task_id": "dog-run",
                "family": "tdmpc2",
                "required_metrics": ["final_return_mean", "auc_return"],
                "official": {
                    "adapter": "official_tdmpc2",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')", "--steps", "7000000"],
                },
                "worldflux": {
                    "adapter": "worldflux_tdmpc2_native",
                    "cwd": ".",
                    "env": {},
                    "command": ["python3", "-c", "print('ok')", "--steps", "7000000"],
                },
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    task_costs = mod._manifest_task_costs(manifest_path)
    shards = mod._build_seed_system_shards(
        "run_pair",
        task_costs,
        seed_values=[0, 1, 2],
        systems=("official", "worldflux"),
        official_instances=["i-off0"],
        worldflux_instances=["i-wf0"],
        gpu_slots_per_instance=2,
        seed_shard_unit="pair",
    )

    # 1 task * 2 systems * 3 seeds
    assert len(shards) == 6
    assert all(len(shard.seed_values) == 1 for shard in shards)
    assert {shard.systems[0] for shard in shards} == {"official", "worldflux"}
