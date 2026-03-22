# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for env-backed learned-policy rollout helpers."""

from __future__ import annotations

import numpy as np
import torch


class _DiscreteActionSpace:
    def __init__(self, n: int) -> None:
        self.n = int(n)

    def sample(self) -> int:
        raise AssertionError("random env sampling must not be used in env_policy rollouts")


class _BoxActionSpace:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape
        self.low = -np.ones(shape, dtype=np.float32)
        self.high = np.ones(shape, dtype=np.float32)

    def sample(self) -> np.ndarray:
        raise AssertionError("random env sampling must not be used in env_policy rollouts")


class _DreamerStubEnv:
    def __init__(self) -> None:
        self.action_space = _DiscreteActionSpace(3)
        self.actions: list[int] = []
        self.steps = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self.steps = 0
        return np.zeros((3, 64, 64), dtype=np.float32), {}

    def step(self, action: int):
        self.actions.append(int(action))
        self.steps += 1
        done = self.steps >= 2
        return np.zeros((3, 64, 64), dtype=np.float32), float(action), done, False, {}

    def close(self) -> None:
        return


class _TDMPC2StubEnv:
    def __init__(self) -> None:
        self.action_space = _BoxActionSpace((2,))
        self.actions: list[np.ndarray] = []
        self.steps = 0

    def reset(self, *, seed: int | None = None):
        _ = seed
        self.steps = 0
        return np.zeros((4,), dtype=np.float32), {}

    def step(self, action: np.ndarray):
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.steps += 1
        done = self.steps >= 2
        return np.zeros((4,), dtype=np.float32), float(np.sum(action)), done, False, {}

    def close(self) -> None:
        return


class _DreamerActorHead:
    def __call__(self, features: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros((features.shape[0], 3), dtype=torch.float32, device=features.device)
        logits[:, 2] = 1.0
        return logits

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        action = torch.zeros((features.shape[0], 3), dtype=torch.float32, device=features.device)
        action[:, 2] = 1.0
        log_prob = torch.zeros((features.shape[0],), dtype=torch.float32, device=features.device)
        return action, log_prob


class _DreamerModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "dreamer", "action_dim": 3})()
        self.actor_head = _DreamerActorHead()

    def initial_state(self, batch_size: int, device: torch.device) -> object:
        deter = torch.zeros((batch_size, 2), device=device)
        stoch = torch.zeros((batch_size, 1, 2), device=device)
        return type("State", (), {"tensors": {"deter": deter, "stoch": stoch}, "meta": {}})()

    def update(self, state, action, obs):
        _ = action
        _ = obs
        return state


class _TDMPC2Policy(torch.nn.Module):
    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.ones((latent.shape[0], 2), dtype=torch.float32, device=latent.device) * 0.5


class _TDMPC2Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"model_type": "tdmpc2", "action_dim": 2})()
        self._policy = _TDMPC2Policy()

    def encode(self, obs: torch.Tensor):
        latent = torch.ones((obs.shape[0], 4), dtype=torch.float32, device=obs.device)
        return type("State", (), {"tensors": {"latent": latent}})()


def test_collect_env_policy_rollout_uses_dreamer_actor() -> None:
    from worldflux.evals.env_policy import collect_env_policy_rollout

    env = _DreamerStubEnv()
    rollout = collect_env_policy_rollout(
        _DreamerModel(),
        env_id="ALE/Breakout-v5",
        family="dreamer",
        episodes=1,
        horizon=2,
        seed=7,
        env_factory=lambda: env,
        policy_impl="actor",
        allow_fallback=False,
    )

    assert rollout.episode_returns == [4.0]
    assert env.actions == [2, 2]
    assert rollout.provenance["policy_impl"] == "candidate_actor_stateful_eval"
    assert rollout.provenance["eval_mode"] == "env_policy"


def test_collect_env_policy_rollout_uses_tdmpc2_policy_head() -> None:
    from worldflux.evals.env_policy import collect_env_policy_rollout

    env = _TDMPC2StubEnv()
    rollout = collect_env_policy_rollout(
        _TDMPC2Model(),
        env_id="HalfCheetah-v5",
        family="tdmpc2",
        episodes=1,
        horizon=2,
        seed=7,
        env_factory=lambda: env,
    )

    assert rollout.episode_returns == [2.0]
    assert len(env.actions) == 2
    assert np.allclose(env.actions[0], np.array([0.5, 0.5], dtype=np.float32))
    assert rollout.provenance["policy_impl"] == "cem_planner_eval"
