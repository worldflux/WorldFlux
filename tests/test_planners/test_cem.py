"""Tests for CEM planner."""

import torch

from worldflux.core.batch import Batch
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.spec import Capability
from worldflux.core.state import State
from worldflux.planners import CEMPlanner


class DummyRewardModel(WorldModel):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.capabilities = {Capability.REWARD_PRED}

    def encode(self, obs, deterministic: bool = False) -> State:
        return State(tensors={"reward": torch.zeros(obs.shape[0], 1)})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        reward = action.sum(dim=-1, keepdim=True)
        return State(tensors={"reward": reward})

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        return state

    def decode(self, state: State) -> ModelOutput:
        return ModelOutput(preds={"reward": state.tensors["reward"]})

    def loss(self, batch: Batch) -> LossOutput:
        loss = torch.tensor(0.0)
        return LossOutput(loss=loss)


def test_cem_planner_returns_actions():
    model = DummyRewardModel(action_dim=3)
    planner = CEMPlanner(horizon=5, action_dim=3, num_samples=32, num_elites=8, iterations=1)
    init_state = model.encode(torch.zeros(1, 4))
    actions = planner.plan(model, init_state)
    assert actions.shape == (5, 3)
