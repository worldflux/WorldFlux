"""Tests for universal component interfaces."""

from __future__ import annotations

import asyncio
from typing import cast

import torch

from worldflux.core.interfaces import (
    ActionConditioner,
    AsyncDecoder,
    AsyncDecoderAdapter,
    AsyncDynamicsModel,
    AsyncDynamicsModelAdapter,
    AsyncObservationEncoder,
    AsyncObservationEncoderAdapter,
    AsyncRolloutExecutor,
    AsyncRolloutExecutorAdapter,
    ComponentSpec,
    Decoder,
    DynamicsModel,
    ObservationEncoder,
    RolloutEngine,
    RolloutExecutor,
    ensure_async_decoder,
    ensure_async_dynamics_model,
    ensure_async_observation_encoder,
    ensure_async_rollout_executor,
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


class DummyAsyncEncoder:
    async def encode_async(self, observations):
        obs = observations["obs"]
        return State(tensors={"latent": obs})


class DummyAsyncDynamics:
    async def transition_async(self, state, conditioned, deterministic=False):
        del deterministic
        latent = state.tensors["latent"]
        action = conditioned.get("action", torch.zeros_like(latent))
        return State(tensors={"latent": latent + action})


class DummyAsyncDecoder:
    async def decode_async(self, state, conditions=None):
        del conditions
        return {"obs": state.tensors["latent"]}


class DummyAsyncRollout:
    async def rollout_open_loop_async(
        self,
        model,
        initial_state,
        action_sequence,
        conditions=None,
        deterministic=False,
    ):
        del model, action_sequence, conditions, deterministic
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
    assert isinstance(DummyAsyncEncoder(), AsyncObservationEncoder)
    assert isinstance(DummyConditioner(), ActionConditioner)
    assert isinstance(DummyDynamics(), DynamicsModel)
    assert isinstance(DummyAsyncDynamics(), AsyncDynamicsModel)
    assert isinstance(DummyDecoder(), Decoder)
    assert isinstance(DummyAsyncDecoder(), AsyncDecoder)
    assert isinstance(DummyRollout(), RolloutExecutor)
    assert isinstance(DummyAsyncRollout(), AsyncRolloutExecutor)
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


def test_async_adapters_wrap_sync_components() -> None:
    obs = torch.randn(2, 3)
    encoder = AsyncObservationEncoderAdapter(DummyEncoder())
    state = asyncio.run(encoder.encode_async({"obs": obs}))
    assert torch.allclose(state.tensors["latent"], obs)

    dynamics = AsyncDynamicsModelAdapter(DummyDynamics())
    next_state = asyncio.run(
        dynamics.transition_async(
            state,
            {"action": torch.ones_like(obs)},
            deterministic=False,
        )
    )
    assert torch.allclose(next_state.tensors["latent"], obs + 1.0)

    decoder = AsyncDecoderAdapter(DummyDecoder())
    decoded = asyncio.run(decoder.decode_async(next_state))
    assert torch.allclose(decoded["obs"], obs + 1.0)

    rollout = AsyncRolloutExecutorAdapter(DummyRollout())
    rolled = asyncio.run(
        rollout.rollout_open_loop_async(
            model=object(),
            initial_state=state,
            action_sequence=None,
            conditions=None,
            deterministic=False,
        )
    )
    assert rolled is state


def test_ensure_async_helpers_return_protocol_compatible_components() -> None:
    sync_encoder = ensure_async_observation_encoder(DummyEncoder())
    sync_dynamics = ensure_async_dynamics_model(DummyDynamics())
    sync_decoder = ensure_async_decoder(DummyDecoder())
    sync_rollout = ensure_async_rollout_executor(DummyRollout())

    assert isinstance(sync_encoder, AsyncObservationEncoder)
    assert isinstance(sync_dynamics, AsyncDynamicsModel)
    assert isinstance(sync_decoder, AsyncDecoder)
    assert isinstance(sync_rollout, AsyncRolloutExecutor)

    async_encoder = DummyAsyncEncoder()
    async_dynamics = DummyAsyncDynamics()
    async_decoder = DummyAsyncDecoder()
    async_rollout = DummyAsyncRollout()

    assert ensure_async_observation_encoder(async_encoder) is async_encoder
    assert ensure_async_dynamics_model(async_dynamics) is async_dynamics
    assert ensure_async_decoder(async_decoder) is async_decoder
    assert ensure_async_rollout_executor(async_rollout) is async_rollout
