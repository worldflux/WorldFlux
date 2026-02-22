"""Planner interface contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from ..core.output import ModelOutput
from ..core.payloads import ActionPayload, ConditionPayload
from ..core.state import State


@runtime_checkable
class Planner(Protocol):
    """Planning strategy interface."""

    def plan(
        self,
        model: Any,
        state: State,
        conditions: ConditionPayload | None = None,
    ) -> ActionPayload: ...


@runtime_checkable
class PlannerObjective(Protocol):
    """Objective interface used by planners to score rollout steps."""

    @property
    def requires_reward(self) -> bool: ...

    def score_step(
        self,
        model: Any,
        step_output: ModelOutput,
        step_state: State,
        step: int,
    ) -> Tensor: ...


class RewardObjective:
    """Default objective: maximize predicted reward."""

    @property
    def requires_reward(self) -> bool:
        return True

    def score_step(
        self,
        model: Any,
        step_output: ModelOutput,
        step_state: State,
        step: int,
    ) -> Tensor:
        del model, step_state, step
        reward = step_output.predictions.get("reward")
        if reward is None:
            raise ValueError("RewardObjective requires 'reward' in model step output")
        return reward.squeeze(-1)
