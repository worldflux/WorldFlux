"""Minimal Cross-Entropy Method (CEM) planner."""

from __future__ import annotations

import torch
from torch import Tensor

from ..core.exceptions import CapabilityError
from ..core.model import WorldModel
from ..core.payloads import PLANNER_HORIZON_KEY, ActionPayload, ConditionPayload
from ..core.state import State
from .interfaces import Planner


class CEMPlanner(Planner):
    """Simple planner that ranks sampled action sequences by predicted reward."""

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        num_samples: int = 256,
        num_elites: int = 32,
        iterations: int = 1,
        action_low: float | Tensor | None = None,
        action_high: float | Tensor | None = None,
    ):
        self.horizon = horizon
        self.action_dim = action_dim
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.iterations = iterations
        self.action_low = action_low
        self.action_high = action_high

    def _apply_action_bounds(self, actions: Tensor) -> Tensor:
        if self.action_low is not None:
            actions = torch.maximum(
                actions, torch.as_tensor(self.action_low, device=actions.device)
            )
        if self.action_high is not None:
            actions = torch.minimum(
                actions, torch.as_tensor(self.action_high, device=actions.device)
            )
        return actions

    def plan(
        self,
        model: WorldModel,
        state: State,
        conditions: ConditionPayload | None = None,
        device: torch.device | None = None,
    ) -> ActionPayload:
        if not model.supports_reward:
            raise CapabilityError("Planner requires reward prediction capability")
        contract_fn = getattr(model, "io_contract", None)
        contract = contract_fn() if callable(contract_fn) else None

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
            actions = self._apply_action_bounds(actions)
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
                if contract is not None:
                    missing_state = [
                        k for k in contract.required_state_keys if k not in rollout_state.tensors
                    ]
                    if missing_state:
                        raise CapabilityError(
                            f"State is missing required keys for planner contract: {missing_state}"
                        )

            rewards = []
            step_state = rollout_state
            for t in range(self.horizon):
                step_state = model.plan_step(step_state, actions[t], conditions=conditions)
                step_output = model.sample_step(step_state, conditions=conditions)
                reward_t = step_output.preds.get("reward")
                if reward_t is None:
                    raise CapabilityError("Planner requires reward prediction from sample_step")
                rewards.append(reward_t.squeeze(-1))

            scores = torch.stack(rewards, dim=0).sum(dim=0)
            elite_idx = scores.topk(self.num_elites).indices
            elite = actions[:, elite_idx]
            mean = elite.mean(dim=1)
            std = elite.std(dim=1).clamp_min(1e-3)

        return ActionPayload(
            kind="continuous",
            tensor=mean,
            extras={PLANNER_HORIZON_KEY: self.horizon},
        )
