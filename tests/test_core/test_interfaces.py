"""Tests for universal component interfaces."""

from __future__ import annotations

from typing import cast

import torch

from worldflux.core.interfaces import (
    ActionConditioner,
    ComponentSpec,
    Decoder,
    DynamicsModel,
    ObservationEncoder,
    RolloutEngine,
    RolloutExecutor,
)
from worldflux.core.payloads import PLANNER_HORIZON_KEY, ActionPayload, ConditionPayload
from worldflux.core.state import State
from worldflux.planners.interfaces import Planner


class DummyEncoder:
    def encode(self, observations):
        obs = observations["obs"]
        return State(tensors={"latent": obs})


class DummyConditioner:
    def condition(self, state, action, conditions=None):
        del state, conditions
        if action is None or action.tensor is None:
            return {}
        return {"action": action.tensor}


class DummyDynamics:
    def transition(self, state, conditioned, deterministic=False):
        del deterministic
        latent = state.tensors["latent"]
        action = conditioned.get("action", torch.zeros_like(latent))
        return State(tensors={"latent": latent + action})


class DummyDecoder:
    def decode(self, state, conditions=None):
        del conditions
        return {"obs": state.tensors["latent"]}


class DummyRollout:
    def rollout_open_loop(
        self,
        model,
        initial_state,
        action_sequence,
        conditions=None,
        deterministic=False,
    ):
        del model, action_sequence, conditions, deterministic
        return initial_state

    # Legacy compatibility shape.
    def rollout(
        self,
        model,
        initial_state,
        action_sequence,
        conditions=None,
        deterministic=False,
        mode="autoregressive",
    ):
        del model, action_sequence, conditions, deterministic, mode
        return initial_state


class DummyPlanner:
    def plan(self, model, state, conditions=None):
        del model, state, conditions
        return ActionPayload(
            kind="continuous",
            tensor=torch.randn(3, 2),
            extras={PLANNER_HORIZON_KEY: 3},
        )


def test_component_spec_fields():
    spec = ComponentSpec(name="enc", component_type="observation_encoder", version="v0.2")
    assert spec.name == "enc"
    assert spec.component_type == "observation_encoder"


def test_protocol_runtime_compatibility():
    assert isinstance(DummyEncoder(), ObservationEncoder)
    assert isinstance(DummyConditioner(), ActionConditioner)
    assert isinstance(DummyDynamics(), DynamicsModel)
    assert isinstance(DummyDecoder(), Decoder)
    assert isinstance(DummyRollout(), RolloutExecutor)
    assert isinstance(DummyRollout(), RolloutEngine)
    assert isinstance(DummyPlanner(), Planner)


def test_dummy_components_work_together():
    encoder = cast(ObservationEncoder, DummyEncoder())
    conditioner = cast(ActionConditioner, DummyConditioner())
    dynamics = cast(DynamicsModel, DummyDynamics())
    decoder = cast(Decoder, DummyDecoder())

    state = encoder.encode({"obs": torch.randn(2, 3)})
    action = ActionPayload(kind="continuous", tensor=torch.randn(2, 3))
    conditioned = conditioner.condition(state, action, ConditionPayload())
    next_state = dynamics.transition(state, conditioned)
    preds = decoder.decode(next_state)

    assert "obs" in preds
    assert preds["obs"].shape == (2, 3)
