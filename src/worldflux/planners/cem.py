"""Minimal Cross-Entropy Method (CEM) planner."""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.exceptions import CapabilityError
from ..core.model import WorldModel
from ..core.state import State


class CEMPlanner:
    """Simple planner that ranks sampled action sequences by predicted reward."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 256,
        num_elites: int = 32,
        iterations: int = 1,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations

    def plan(self, model: WorldModel, state, device: torch.device | None = None) -> Tensor:
        if not model.supports_reward:
            raise CapabilityError("Planner requires reward prediction capability")

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                if hasattr(state, "device"):
                    device = state.device
                else:
                    device = torch.device("cpu")
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.ones_like(mean)

        for _ in range(self.iterations):
            noise = torch.randn(self.horizon, self.num_samples, self.action_dim, device=device)
            actions = mean[:, None, :] + std[:, None, :] * noise
            rollout_state = state
            if isinstance(state, State):
                batch_size = state.batch_size
                if batch_size == 1:
                    rollout_state = State(
                        tensors={
                            k: v.expand(self.num_samples, *v.shape[1:])
                            for k, v in state.tensors.items()
                        },
                        meta=state.meta,
                    )
                elif batch_size != self.num_samples:
                    raise CapabilityError(
                        f"Planner requires batch size 1 or {self.num_samples}, got {batch_size}"
                    )
            trajectory = model.rollout(rollout_state, actions)
            if trajectory.rewards is None:
                raise CapabilityError("Model rollout did not produce rewards")
            scores = trajectory.rewards.sum(dim=0)
            elite_idx = scores.topk(self.num_elites).indices
            elite = actions[:, elite_idx]
            mean = elite.mean(dim=1)
            std = elite.std(dim=1).clamp_min(1e-3)

        return mean
