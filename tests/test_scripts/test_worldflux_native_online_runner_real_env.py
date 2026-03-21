# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for real-environment path in worldflux native online runner."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_wrapper_module():
    root = _repo_root()
    wrappers_dir = root / "scripts" / "parity" / "wrappers"
    runtime_root = root / "scripts" / "parity"
    if str(wrappers_dir) not in sys.path:
        sys.path.insert(0, str(wrappers_dir))
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))
    spec = importlib.util.spec_from_file_location(
        "worldflux_native_online_runner",
        wrappers_dir / "worldflux_native_online_runner.py",
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load worldflux_native_online_runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_worldflux_native_online_runner_uses_real_env_backend_when_not_mock(tmp_path: Path) -> None:
    root = _repo_root()
    metrics_out = tmp_path / "metrics.json"

    cmd = [
        sys.executable,
        "scripts/parity/wrappers/worldflux_native_online_runner.py",
        "--family",
        "dreamerv3",
        "--task-id",
        "atari100k_pong",
        "--seed",
        "0",
        "--steps",
        "12",
        "--eval-interval",
        "6",
        "--eval-episodes",
        "1",
        "--eval-window",
        "2",
        "--env-backend",
        "stub",
        "--device",
        "cpu",
        "--buffer-capacity",
        "64",
        "--warmup-steps",
        "1",
        "--train-steps-per-eval",
        "1",
        "--sequence-length",
        "2",
        "--batch-size",
        "2",
        "--max-episode-steps",
        "4",
        "--dreamer-model-profile",
        "ci",
        "--dreamer-replay-ratio",
        "1",
        "--dreamer-train-chunk-size",
        "1",
        "--run-dir",
        str(tmp_path / "run"),
        "--metrics-out",
        str(metrics_out),
    ]

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "parity.v1"
    assert payload["metadata"]["mode"] == "native_real_env"
    assert payload["metadata"]["env_backend"] == "stub"
    assert payload["metadata"]["policy_mode"] == "parity_candidate"
    assert payload["metadata"]["policy"] == "model_based"
    assert payload["metadata"]["policy_impl"] == "candidate_actor_stateful"
    assert payload["metadata"]["eval_policy"] == "model_based"
    assert payload["metadata"]["eval_policy_impl"] == "candidate_actor_stateful_eval"
    assert isinstance(payload["metadata"]["eval_protocol_hash"], str)
    assert payload["metadata"]["eval_protocol_hash"]
    assert payload["metadata"]["model_profile"] == "ci"
    assert payload["metadata"]["model_id"] == "dreamerv3:ci"
    assert payload["num_curve_points"] >= 1


def test_worldflux_native_online_runner_keeps_diagnostic_random_mode(tmp_path: Path) -> None:
    root = _repo_root()
    metrics_out = tmp_path / "metrics_random.json"

    cmd = [
        sys.executable,
        "scripts/parity/wrappers/worldflux_native_online_runner.py",
        "--family",
        "tdmpc2",
        "--task-id",
        "dog-run",
        "--seed",
        "1",
        "--steps",
        "12",
        "--eval-interval",
        "6",
        "--eval-episodes",
        "1",
        "--eval-window",
        "2",
        "--env-backend",
        "stub",
        "--device",
        "cpu",
        "--buffer-capacity",
        "64",
        "--warmup-steps",
        "1",
        "--train-steps-per-eval",
        "1",
        "--sequence-length",
        "2",
        "--batch-size",
        "2",
        "--max-episode-steps",
        "4",
        "--policy-mode",
        "diagnostic_random",
        "--run-dir",
        str(tmp_path / "run_random"),
        "--metrics-out",
        str(metrics_out),
    ]

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload["metadata"]["policy_mode"] == "diagnostic_random"
    assert payload["metadata"]["policy"] == "random"
    assert payload["metadata"]["policy_impl"] == "random_env_sampler"
    assert payload["metadata"]["eval_policy"] == "random"
    assert payload["metadata"]["eval_policy_impl"] == "random_env_sampler_eval"


def test_worldflux_native_online_runner_uses_tdmpc2_learned_eval_policy(
    tmp_path: Path,
) -> None:
    root = _repo_root()
    metrics_out = tmp_path / "metrics_tdmpc2.json"

    cmd = [
        sys.executable,
        "scripts/parity/wrappers/worldflux_native_online_runner.py",
        "--family",
        "tdmpc2",
        "--task-id",
        "dog-run",
        "--seed",
        "2",
        "--steps",
        "12",
        "--eval-interval",
        "6",
        "--eval-episodes",
        "1",
        "--eval-window",
        "2",
        "--env-backend",
        "stub",
        "--device",
        "cpu",
        "--buffer-capacity",
        "64",
        "--warmup-steps",
        "1",
        "--train-steps-per-eval",
        "1",
        "--sequence-length",
        "2",
        "--batch-size",
        "2",
        "--max-episode-steps",
        "4",
        "--run-dir",
        str(tmp_path / "run_tdmpc2"),
        "--metrics-out",
        str(metrics_out),
    ]

    completed = subprocess.run(cmd, cwd=root, check=False, text=True, capture_output=True)
    assert completed.returncode == 0, completed.stderr

    payload = json.loads(metrics_out.read_text(encoding="utf-8"))
    assert payload["metadata"]["policy_mode"] == "parity_candidate"
    assert payload["metadata"]["policy"] == "learned"
    assert payload["metadata"]["policy_impl"] == "cem_planner"
    assert payload["metadata"]["eval_policy"] == "learned"
    assert payload["metadata"]["eval_policy_impl"] == "cem_planner_eval"
    assert payload["metadata"]["model_profile"] == "5m"
    assert payload["metadata"]["model_id"] == "tdmpc2:5m"


def test_worldflux_native_online_runner_passes_dreamer_override_json(
    tmp_path: Path, monkeypatch
) -> None:
    mod = _load_wrapper_module()
    captured: dict[str, object] = {}

    fake_dreamer_module = types.ModuleType("runtime.dreamer_native_agent")

    def _cfg(**kwargs):
        return types.SimpleNamespace(**kwargs)

    def _run(config):
        captured["config"] = config
        return [(0.0, 1.0), (12.0, 2.0)], {"policy_impl": "candidate_actor_stateful"}

    fake_dreamer_module.DreamerNativeRunConfig = _cfg
    fake_dreamer_module.run_dreamer_native = _run

    fake_atari_module = types.ModuleType("runtime.atari_env")
    fake_atari_module.AtariEnvError = RuntimeError
    fake_dmcontrol_module = types.ModuleType("runtime.dmcontrol_env")
    fake_dmcontrol_module.DMControlEnvError = RuntimeError
    fake_recipe_module = types.ModuleType("runtime.dreamer_official_recipe")
    fake_recipe_module.OFFICIAL_DREAMER_ATARI100K_RECIPE = types.SimpleNamespace(replay_size=64)
    fake_tdmpc2_module = types.ModuleType("runtime.tdmpc2_native_agent")
    fake_tdmpc2_module.TDMPC2NativeRunConfig = _cfg
    fake_tdmpc2_module.run_tdmpc2_native = lambda config: ([(0.0, 0.0)], {})

    monkeypatch.setitem(sys.modules, "runtime.atari_env", fake_atari_module)
    monkeypatch.setitem(sys.modules, "runtime.dmcontrol_env", fake_dmcontrol_module)
    monkeypatch.setitem(sys.modules, "runtime.dreamer_native_agent", fake_dreamer_module)
    monkeypatch.setitem(sys.modules, "runtime.dreamer_official_recipe", fake_recipe_module)
    monkeypatch.setitem(sys.modules, "runtime.tdmpc2_native_agent", fake_tdmpc2_module)
    monkeypatch.setattr(
        mod,
        "_parse_args",
        lambda: types.SimpleNamespace(
            family="dreamerv3",
            task_id="atari100k_pong",
            seed=0,
            steps=12,
            eval_interval=6,
            eval_episodes=1,
            eval_window=2,
            device="cpu",
            env_backend="stub",
            policy_mode="parity_candidate",
            run_dir=tmp_path / "run",
            metrics_out=tmp_path / "metrics.json",
            mock=False,
            buffer_capacity=64,
            warmup_steps=1,
            train_steps_per_eval=1,
            sequence_length=2,
            batch_size=2,
            max_episode_steps=4,
            dreamer_policy_impl="actor",
            dreamer_train_ratio=256.0,
            dreamer_replay_ratio=0.0,
            dreamer_train_chunk_size=64,
            dreamer_model_profile="wf12m",
            tdmpc2_model_profile="5m",
            dreamer_diagnostic="false",
            dreamer_lr=4e-5,
            dreamer_config_overrides_json=json.dumps(
                {
                    "learning_rate_override": 3e-4,
                    "model_config_overrides": {
                        "kl_free": 2.0,
                        "loss_scales": {"kl_dynamics": 0.3, "kl_representation": 0.05},
                        "imagination_horizon": 20,
                    },
                }
            ),
        ),
    )

    exit_code = mod.main()
    assert exit_code == 0
    config = captured["config"]
    assert config.learning_rate_override == 3e-4
    assert config.model_config_overrides["kl_free"] == 2.0
    assert config.model_config_overrides["loss_scales"]["kl_dynamics"] == 0.3
    assert config.model_config_overrides["imagination_horizon"] == 20
