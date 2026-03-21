# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""MuJoCo dataset collection helpers with dataset-manifest support."""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import ReplayBuffer
from .dataset_manifest import build_dataset_manifest, load_dataset_manifest, write_dataset_manifest


def collect_mujoco_dataset(
    *,
    env_name: str,
    output_dir: str | Path,
    num_episodes: int,
    max_steps_per_episode: int,
    collector_policy: str,
    policy_checkpoint: str | None,
    action_noise: float,
    seed: int,
    device: str = "cpu",
) -> tuple[ReplayBuffer, Path, dict[str, Any]]:
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("gymnasium is required for MuJoCo collection.") from exc

    rng = np.random.default_rng(seed)
    env = gym.make(env_name)
    try:
        obs_dim = int(env.observation_space.shape[0])
        action_dim = int(env.action_space.shape[0])
        act_low = np.asarray(env.action_space.low, dtype=np.float32)
        act_high = np.asarray(env.action_space.high, dtype=np.float32)

        action_fn = _build_action_fn(
            collector_policy=collector_policy,
            policy_checkpoint=policy_checkpoint,
            action_dim=action_dim,
            device=device,
        )

        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_rewards: list[float] = []
        all_dones: list[float] = []

        for episode in range(num_episodes):
            obs, _ = env.reset(seed=seed + episode)
            done = False
            steps = 0
            while not done and steps < max_steps_per_episode:
                action = action_fn(np.asarray(obs, dtype=np.float32), rng=rng)
                if collector_policy == "noisy":
                    action = action + rng.normal(0.0, action_noise, size=action_dim).astype(
                        np.float32
                    )
                action = np.clip(action, act_low, act_high)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                done = bool(terminated or truncated)
                all_obs.append(np.asarray(obs, dtype=np.float32))
                all_actions.append(np.asarray(action, dtype=np.float32))
                all_rewards.append(float(reward))
                all_dones.append(float(done))
                obs = next_obs
                steps += 1
    finally:
        env.close()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    raw_dataset_path = output_root / "mujoco_raw.npz"
    replay_buffer_path = output_root / "replay_buffer.npz"
    manifest_path = output_root / "dataset_manifest.json"

    obs_array = np.asarray(all_obs, dtype=np.float32)
    actions_array = np.asarray(all_actions, dtype=np.float32)
    rewards_array = np.asarray(all_rewards, dtype=np.float32)
    dones_array = np.asarray(all_dones, dtype=np.float32)

    np.savez_compressed(
        raw_dataset_path,
        obs=obs_array,
        actions=actions_array,
        rewards=rewards_array,
        dones=dones_array,
        obs_dim=np.array(obs_dim),
        action_dim=np.array(action_dim),
        env_name=np.array(env_name),
    )

    buffer = ReplayBuffer(
        capacity=max(len(obs_array), 1),
        obs_shape=(obs_dim,),
        action_dim=action_dim,
    )
    if len(obs_array) > 0:
        starts: list[int] = [0] + [int(x) for x in np.where(dones_array[:-1] == 1.0)[0] + 1]
        ends: list[int] = [int(x) for x in np.where(dones_array == 1.0)[0] + 1]
        if len(ends) < len(starts):
            ends.append(len(obs_array))
        for start, end in zip(starts, ends, strict=False):
            if start >= end:
                continue
            buffer.add_episode(
                obs=obs_array[start:end],
                actions=actions_array[start:end],
                rewards=rewards_array[start:end],
                dones=dones_array[start:end],
            )
    buffer.save(replay_buffer_path)

    manifest = build_dataset_manifest(
        env_id=f"mujoco/{env_name}",
        collector_kind="mujoco_collector",
        collector_policy=collector_policy,
        seed=seed,
        episodes=num_episodes,
        transitions=len(obs_array),
        reward_stats={
            "mean": float(rewards_array.mean()) if len(rewards_array) else 0.0,
            "std": float(rewards_array.std()) if len(rewards_array) else 0.0,
            "sum": float(rewards_array.sum()) if len(rewards_array) else 0.0,
        },
        source_commit=_resolve_source_commit(),
        created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        preprocessing={"obs_norm": False},
        artifact_paths={
            "raw_dataset": raw_dataset_path.name,
            "replay_buffer": replay_buffer_path.name,
        },
    )
    write_dataset_manifest(manifest_path, manifest)
    buffer.data_provenance = {
        "kind": "dataset_manifest",
        "env_id": manifest["env_id"],
        "dataset_manifest": str(manifest_path.resolve()),
    }
    return buffer, manifest_path, manifest


def load_buffer_from_dataset_manifest(
    manifest_path: str | Path,
) -> tuple[ReplayBuffer, dict[str, Any]]:
    from .dataset_manifest import resolve_dataset_artifact_path

    manifest = load_dataset_manifest(manifest_path)
    replay_path = resolve_dataset_artifact_path(
        manifest,
        artifact_key="replay_buffer",
        manifest_path=manifest_path,
    )
    buffer = ReplayBuffer.load(replay_path)
    buffer.data_provenance = {
        "kind": "dataset_manifest",
        "env_id": manifest["env_id"],
        "dataset_manifest": str(Path(manifest_path).resolve()),
    }
    return buffer, manifest


def _build_action_fn(
    *,
    collector_policy: str,
    policy_checkpoint: str | None,
    action_dim: int,
    device: str,
):
    policy = str(collector_policy).strip().lower()
    if policy not in {"policy_checkpoint", "noisy", "random"}:
        raise ValueError(f"Unsupported collector_policy: {collector_policy!r}")

    if policy == "random":
        return lambda _obs, *, rng: rng.uniform(-1.0, 1.0, size=action_dim).astype(np.float32)

    if not str(policy_checkpoint or "").strip():
        if policy == "noisy":
            return lambda _obs, *, rng: np.zeros(action_dim, dtype=np.float32)
        raise ValueError("policy_checkpoint collector requires --policy-checkpoint.")

    from worldflux.verify.quick import _load_model_from_target

    assert policy_checkpoint is not None
    model = _load_model_from_target(Path(policy_checkpoint), device=device)
    if str(getattr(model.config, "model_type", "")).strip().lower() != "tdmpc2":
        raise ValueError("MuJoCo policy checkpoint collector currently supports TD-MPC2 only.")

    def _predict(obs: np.ndarray, *, rng) -> np.ndarray:
        del rng
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            state = model.encode(obs_tensor)  # type: ignore[operator]
            action = model._policy(state.tensors["latent"])  # type: ignore[operator,attr-defined]
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    return _predict


def _resolve_source_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
