# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Env-backed learned-policy rollout helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class EnvPolicyRollout:
    """Collected env-policy rollouts plus provenance."""

    episode_returns: list[float]
    provenance: dict[str, Any] = field(default_factory=dict)
    obs: torch.Tensor | None = None
    actions: torch.Tensor | None = None
    rewards: torch.Tensor | None = None


def collect_env_policy_rollout(
    model,
    *,
    env_id: str,
    family: str | None = None,
    episodes: int,
    seed: int,
    horizon: int | None = None,
    device: str = "cpu",
    env_factory: Callable[[], Any] | None = None,
    policy_impl: str = "auto",
    allow_fallback: bool = True,
    shooting_horizon: int = 5,
    shooting_num_candidates: int = 128,
) -> EnvPolicyRollout:
    """Roll out a learned policy in an environment and capture provenance."""
    del shooting_horizon, shooting_num_candidates

    resolved_family = (
        str(family or getattr(getattr(model, "config", None), "model_type", "")).strip().lower()
    )
    if resolved_family not in {"dreamer", "dreamerv3", "tdmpc2"}:
        raise ValueError(f"Unsupported env_policy family: {resolved_family!r}")

    config = getattr(model, "config", None)
    obs_shape = tuple(getattr(config, "obs_shape", (4,)))
    action_dim = int(getattr(config, "action_dim", 1))
    torch_device = torch.device(device)
    max_steps = max(1, int(horizon)) if horizon is not None else None

    if max_steps is not None:
        obs_batch = np.zeros((episodes, *obs_shape), dtype=np.float32)
        actions = np.zeros((max_steps, episodes, action_dim), dtype=np.float32)
        rewards = np.zeros((max_steps, episodes), dtype=np.float32)
    else:
        obs_batch = None
        actions = None
        rewards = None

    episode_returns: list[float] = []
    seed_schedule: list[int] = []
    effective_policy_impl: str | None = None

    was_training = bool(getattr(model, "training", False))
    if was_training:
        model.eval()

    try:
        for episode_idx in range(max(1, int(episodes))):
            episode_seed = int(seed) + episode_idx
            seed_schedule.append(episode_seed)
            env = env_factory() if env_factory is not None else _build_env(env_id)
            try:
                obs, _ = _reset_env(env, episode_seed)
                if obs_batch is not None:
                    obs_batch[episode_idx] = _fit_observation(obs, obs_shape)

                total_reward = 0.0
                done = False
                step_idx = 0

                policy_state = None
                prev_action = None
                if resolved_family in {"dreamer", "dreamerv3"}:
                    policy_state = model.initial_state(1, torch_device)
                    prev_action = torch.zeros(
                        (1, action_dim), dtype=torch.float32, device=torch_device
                    )

                while not done and (max_steps is None or step_idx < max_steps):
                    action_env: Any
                    if resolved_family in {"dreamer", "dreamerv3"}:
                        assert policy_state is not None
                        assert prev_action is not None
                        action_env, action_model, policy_state, used_impl = _dreamer_action(
                            model=model,
                            obs=obs,
                            state=policy_state,
                            prev_action=prev_action,
                            action_dim=action_dim,
                            device=torch_device,
                            policy_impl=policy_impl,
                            allow_fallback=allow_fallback,
                        )
                        prev_action = torch.as_tensor(
                            action_model, dtype=torch.float32, device=torch_device
                        ).unsqueeze(0)
                    else:
                        action_env, action_model, used_impl = _tdmpc2_action(
                            model=model,
                            obs=obs,
                            action_dim=action_dim,
                            device=torch_device,
                            env=env,
                        )

                    next_obs, reward, terminated, truncated, _ = env.step(action_env)
                    total_reward += float(reward)
                    done = bool(terminated or truncated)

                    if actions is not None and rewards is not None:
                        actions[step_idx, episode_idx] = action_model
                        rewards[step_idx, episode_idx] = float(reward)

                    obs = next_obs
                    step_idx += 1
                    effective_policy_impl = used_impl

                episode_returns.append(total_reward)
            finally:
                env.close()
    finally:
        if was_training:
            model.train()

    provenance = {
        "kind": "env_policy",
        "env_id": env_id,
        "policy_impl": effective_policy_impl or "unknown",
        "eval_mode": "env_policy",
        "seed_schedule": seed_schedule,
        "episodes": int(episodes),
    }

    return EnvPolicyRollout(
        episode_returns=episode_returns,
        provenance=provenance,
        obs=None if obs_batch is None else torch.as_tensor(obs_batch, device=device),
        actions=None if actions is None else torch.as_tensor(actions, device=device),
        rewards=None if rewards is None else torch.as_tensor(rewards, device=device),
    )


def _build_env(env_id: str):
    try:
        import gymnasium as gym
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise RuntimeError("gymnasium is required for env-backed policy evaluation.") from exc

    try:
        import ale_py
    except ImportError:
        ale_py = None

    if ale_py is not None:
        gym.register_envs(ale_py)
    return gym.make(env_id, render_mode=None)


def _reset_env(env: Any, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    reset = env.reset(seed=seed)
    if isinstance(reset, tuple) and len(reset) == 2:
        obs, info = reset
    else:  # pragma: no cover - compatibility guard
        obs, info = reset, {}
    return np.asarray(obs, dtype=np.float32), dict(info)


def _fit_observation(obs: object, obs_shape: tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(obs, dtype=np.float32)
    if len(obs_shape) == 1:
        flat = arr.reshape(-1)
        out = np.zeros(obs_shape[0], dtype=np.float32)
        size = min(obs_shape[0], flat.size)
        out[:size] = flat[:size]
        return out

    if arr.ndim == 2:
        arr = np.expand_dims(arr, axis=-1)
    if arr.ndim == 3 and arr.shape[0] not in (1, 3, 4):
        arr = arr.transpose(2, 0, 1)
    if arr.shape != obs_shape:
        arr = np.resize(arr, obs_shape)
    if arr.size and float(np.max(arr)) > 1.0:
        arr = arr / 255.0
    return arr.astype(np.float32, copy=False)


def _state_to_features(state: Any) -> torch.Tensor:
    deter = state.tensors.get("deter")
    stoch = state.tensors.get("stoch")
    if deter is None or stoch is None:
        raise RuntimeError("Dreamer state must contain 'deter' and 'stoch' tensors.")
    if stoch.dim() == 3:
        stoch = stoch.flatten(start_dim=1)
    return torch.cat([deter, stoch], dim=-1)


def _dreamer_action(
    *,
    model: Any,
    obs: np.ndarray,
    state: Any,
    prev_action: torch.Tensor,
    action_dim: int,
    device: torch.device,
    policy_impl: str,
    allow_fallback: bool,
) -> tuple[int, np.ndarray, Any, str]:
    normalized_impl = str(policy_impl).strip().lower() or "auto"
    if normalized_impl not in {"auto", "actor"}:
        raise RuntimeError(f"Unsupported Dreamer env_policy implementation: {policy_impl!r}")

    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0).to(device=device)
    posterior = model.update(state, prev_action, obs_tensor)
    actor_head = getattr(model, "actor_head", None)
    if actor_head is None:
        raise RuntimeError("Dreamer env_policy requires actor_head.")

    features = _state_to_features(posterior)
    logits = actor_head(features)
    if not torch.isfinite(logits).all():
        if allow_fallback:
            raise RuntimeError("Dreamer env_policy fallback is disabled in evidence mode.")
        raise RuntimeError("Dreamer actor logits contain non-finite values.")

    action_onehot, _ = actor_head.sample(features)
    if not torch.isfinite(action_onehot).all():
        raise RuntimeError("Dreamer actor action contains non-finite values.")

    action_index = int(torch.argmax(action_onehot, dim=-1).item())
    action_model = (
        F.one_hot(torch.tensor([action_index], device=device), num_classes=action_dim)
        .to(torch.float32)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )
    return action_index, action_model, posterior, "candidate_actor_stateful_eval"


def _tdmpc2_action(
    *,
    model: Any,
    obs: np.ndarray,
    action_dim: int,
    device: torch.device,
    env: Any,
) -> tuple[np.ndarray, np.ndarray, str]:
    obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
    state = model.encode(obs_tensor)
    action = model._policy(state.tensors["latent"])  # type: ignore[attr-defined]
    action_np = action.squeeze(0).detach().cpu().numpy().astype(np.float32, copy=False)

    low = getattr(getattr(env, "action_space", None), "low", None)
    high = getattr(getattr(env, "action_space", None), "high", None)
    if low is not None and high is not None:
        action_np = np.clip(
            action_np,
            np.asarray(low, dtype=np.float32),
            np.asarray(high, dtype=np.float32),
        )

    model_action = np.zeros((action_dim,), dtype=np.float32)
    size = min(action_dim, action_np.size)
    model_action[:size] = action_np[:size]
    return action_np, model_action, "cem_planner_eval"
