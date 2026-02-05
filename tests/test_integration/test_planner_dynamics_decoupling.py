"""Integration tests for planner <-> dynamics decoupling invariants."""

from __future__ import annotations

import torch

from worldflux.core.batch import Batch
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.payloads import PLANNER_HORIZON_KEY, ActionPayload, normalize_planned_action
from worldflux.core.spec import Capability
from worldflux.core.state import State
from worldflux.planners import CEMPlanner


class _RewardModelAdd(WorldModel):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.capabilities = {Capability.REWARD_PRED}

    def encode(self, obs, deterministic: bool = False) -> State:
        del deterministic
        return State(tensors={"latent": torch.zeros(obs.shape[0], self.action_dim)})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        del deterministic
        latent = state.tensors["latent"] + action
        return State(tensors={"latent": latent})

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        del action, obs
        return state

    def decode(self, state: State) -> ModelOutput:
        reward = state.tensors["latent"].sum(dim=-1, keepdim=True)
        return ModelOutput(preds={"reward": reward})

    def loss(self, batch: Batch) -> LossOutput:
        del batch
        return LossOutput(loss=torch.tensor(0.0))


class _RewardModelTanh(WorldModel):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.capabilities = {Capability.REWARD_PRED}

    def encode(self, obs, deterministic: bool = False) -> State:
        del deterministic
        return State(tensors={"latent": torch.zeros(obs.shape[0], self.action_dim)})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        del deterministic
        latent = torch.tanh(state.tensors["latent"] + 0.5 * action)
        return State(tensors={"latent": latent})

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        del action, obs
        return state

    def decode(self, state: State) -> ModelOutput:
        reward = (state.tensors["latent"] ** 2).sum(dim=-1, keepdim=True)
        return ModelOutput(preds={"reward": reward})

    def loss(self, batch: Batch) -> LossOutput:
        del batch
        return LossOutput(loss=torch.tensor(0.0))


def test_same_planner_works_with_different_dynamics_models():
    planner = CEMPlanner(horizon=4, action_dim=3, num_samples=24, num_elites=6, iterations=1)

    model_a = _RewardModelAdd(action_dim=3)
    model_b = _RewardModelTanh(action_dim=3)

    for model in (model_a, model_b):
        state = model.encode(torch.zeros(1, 3))
        planned = planner.plan(model, state)
        assert isinstance(planned, ActionPayload)
        assert planned.extras[PLANNER_HORIZON_KEY] == 4
        seq = normalize_planned_action(planned, api_version="v0.2")
        assert seq.tensor is not None
        assert seq.tensor.shape == (4, 1, 3)
