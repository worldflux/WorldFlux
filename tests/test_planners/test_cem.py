"""Tests for CEM planner."""

import torch

from worldflux.core.batch import Batch
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.payloads import (
    PLANNER_HORIZON_KEY,
    ActionPayload,
    normalize_planned_action,
)
from worldflux.core.spec import Capability
from worldflux.core.state import State
from worldflux.planners import CEMPlanner, PlannerObjective


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
    planned = planner.plan(model, init_state)
    assert isinstance(planned, ActionPayload)
    assert planned.extras[PLANNER_HORIZON_KEY] == 5
    seq = normalize_planned_action(planned, api_version="v0.2")
    assert seq.tensor is not None
    assert seq.tensor.shape == (5, 1, 3)


def test_cem_planner_respects_action_bounds():
    model = DummyRewardModel(action_dim=2)
    planner = CEMPlanner(
        horizon=3,
        action_dim=2,
        num_samples=16,
        num_elites=4,
        iterations=1,
        action_low=-0.1,
        action_high=0.1,
    )
    init_state = model.encode(torch.zeros(1, 4))
    planned = planner.plan(model, init_state)
    seq = normalize_planned_action(planned, api_version="v0.2")
    assert seq.tensor is not None
    assert torch.all(seq.tensor <= 0.10001)
    assert torch.all(seq.tensor >= -0.10001)


class DummyRepresentationModel(WorldModel):
    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.capabilities = {Capability.REPRESENTATION}

    def encode(self, obs, deterministic: bool = False) -> State:
        del deterministic
        return State(tensors={"latent": torch.zeros(obs.shape[0], self.action_dim)})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        del deterministic
        return State(tensors={"latent": state.tensors["latent"] + action})

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        del action, obs
        return state

    def decode(self, state: State) -> ModelOutput:
        return ModelOutput(preds={"representation": state.tensors["latent"]})

    def loss(self, batch: Batch) -> LossOutput:
        del batch
        return LossOutput(loss=torch.tensor(0.0))


class RepresentationObjective(PlannerObjective):
    @property
    def requires_reward(self) -> bool:
        return False

    def score_step(
        self, model, step_output: ModelOutput, step_state: State, step: int
    ) -> torch.Tensor:
        del model, step_output, step
        latent = step_state.tensors["latent"]
        return latent.square().sum(dim=-1)


def test_cem_planner_supports_non_reward_objective():
    model = DummyRepresentationModel(action_dim=3)
    planner = CEMPlanner(
        horizon=4,
        action_dim=3,
        num_samples=16,
        num_elites=4,
        iterations=1,
        objective=RepresentationObjective(),
    )
    init_state = model.encode(torch.zeros(1, 4))
    planned = planner.plan(model, init_state)
    seq = normalize_planned_action(planned, api_version="v0.2")
    assert seq.tensor is not None
    assert seq.tensor.shape == (4, 1, 3)
