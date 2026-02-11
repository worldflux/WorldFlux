"""Tests for additive async protocol wrappers on WorldModel."""

from __future__ import annotations

import asyncio

import pytest
import torch

from worldflux.core.exceptions import ValidationError
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.spec import (
    ActionSpec,
    ConditionSpec,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from worldflux.core.state import State
from worldflux.core.trajectory import Trajectory


class _SyncEncoder:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = 0

    def encode(self, observations: dict[str, torch.Tensor]) -> State:
        self.sync_calls += 1
        return State(tensors={"latent": observations["obs"]})


class _AsyncEncoder(_SyncEncoder):
    async def encode_async(self, observations: dict[str, torch.Tensor]) -> State:
        self.async_calls += 1
        return State(tensors={"latent": observations["obs"] + 1.0})


class _SyncDynamics:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = 0

    def transition(
        self,
        state: State,
        conditioned: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> State:
        del deterministic
        self.sync_calls += 1
        action = conditioned.get("action")
        latent = state.tensors["latent"]
        if action is None:
            action = torch.zeros(latent.shape[0], 1, device=latent.device)
        return State(tensors={"latent": latent + action.expand_as(latent)})


class _AsyncDynamics(_SyncDynamics):
    async def transition_async(
        self,
        state: State,
        conditioned: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> State:
        del deterministic
        self.async_calls += 1
        action = conditioned.get("action")
        latent = state.tensors["latent"]
        if action is None:
            action = torch.zeros(latent.shape[0], 1, device=latent.device)
        return State(tensors={"latent": latent - action.expand_as(latent)})


class _SyncDecoder:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = 0

    def decode(
        self,
        state: State,
        conditions=None,
    ) -> dict[str, torch.Tensor]:
        del conditions
        self.sync_calls += 1
        return {"obs": state.tensors["latent"]}


class _AsyncDecoder(_SyncDecoder):
    async def decode_async(
        self,
        state: State,
        conditions=None,
    ) -> dict[str, torch.Tensor]:
        del conditions
        self.async_calls += 1
        return {"obs": state.tensors["latent"] * 2.0}


class _SyncRolloutExecutor:
    def __init__(self) -> None:
        self.sync_calls = 0
        self.async_calls = 0

    def rollout_open_loop(
        self,
        model: WorldModel,
        initial_state: State,
        action_sequence,
        conditions=None,
        deterministic: bool = False,
    ) -> Trajectory:
        del model, action_sequence, conditions, deterministic
        self.sync_calls += 1
        actions = torch.zeros(0, initial_state.batch_size, 0, device=initial_state.device)
        return Trajectory(states=[initial_state], actions=actions, rewards=None, continues=None)


class _AsyncRolloutExecutor(_SyncRolloutExecutor):
    async def rollout_open_loop_async(
        self,
        model: WorldModel,
        initial_state: State,
        action_sequence,
        conditions=None,
        deterministic: bool = False,
    ) -> Trajectory:
        del model, action_sequence, conditions, deterministic
        self.async_calls += 1
        actions = torch.zeros(0, initial_state.batch_size, 0, device=initial_state.device)
        return Trajectory(states=[initial_state], actions=actions, rewards=None, continues=None)


class _AsyncHarnessModel(WorldModel):
    def __init__(
        self,
        *,
        use_async_components: bool,
        use_async_rollout_executor: bool = False,
        required_modalities: tuple[str, ...] = ("obs",),
    ) -> None:
        super().__init__()
        self._required_modalities = required_modalities

        self.encoder_impl = _AsyncEncoder() if use_async_components else _SyncEncoder()
        self.dynamics_impl = _AsyncDynamics() if use_async_components else _SyncDynamics()
        self.decoder_impl = _AsyncDecoder() if use_async_components else _SyncDecoder()
        rollout = _AsyncRolloutExecutor() if use_async_rollout_executor else _SyncRolloutExecutor()

        self.observation_encoder = self.encoder_impl
        self.dynamics_model = self.dynamics_impl
        self.decoder_module = self.decoder_impl
        self.rollout_executor = rollout

    def io_contract(self) -> ModelIOContract:
        modalities = {
            name: ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))
            for name in self._required_modalities
        }
        return ModelIOContract(
            observation_spec=ObservationSpec(modalities=modalities),
            input_spec=ObservationSpec(modalities=modalities),
            target_spec=ObservationSpec(
                modalities={"next_obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))}
            ),
            action_spec=ActionSpec(kind="continuous", dim=1),
            state_spec=StateSpec(
                tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))}
            ),
            prediction_spec=PredictionSpec(
                tensors={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))}
            ),
            condition_spec=ConditionSpec(allowed_extra_keys=()),
            sequence_layout=SequenceLayout(axes_by_field={"obs": "BT...", "actions": "BT..."}),
            required_batch_keys=("obs",),
            required_state_keys=("latent",),
        )

    def loss(self, batch) -> LossOutput:
        del batch
        return LossOutput(loss=torch.tensor(0.0))


def test_async_wrappers_fallback_to_sync_components() -> None:
    model = _AsyncHarnessModel(use_async_components=False, use_async_rollout_executor=False)

    obs = torch.ones(2, 2)
    state = asyncio.run(model.async_encode(obs))
    assert torch.allclose(state.tensors["latent"], obs)

    action = torch.ones(2, 1)
    next_state = asyncio.run(model.async_transition(state, action))
    decoded = asyncio.run(model.async_decode(next_state))
    assert torch.allclose(decoded.predictions["obs"], torch.full((2, 2), 2.0))

    trajectory = asyncio.run(model.async_rollout(state, torch.ones(3, 2, 1)))
    assert len(trajectory.states) == 1

    assert model.encoder_impl.sync_calls > 0
    assert model.encoder_impl.async_calls == 0
    assert model.dynamics_impl.sync_calls > 0
    assert model.dynamics_impl.async_calls == 0
    assert model.decoder_impl.sync_calls > 0
    assert model.decoder_impl.async_calls == 0


def test_async_wrappers_prefer_native_async_components() -> None:
    model = _AsyncHarnessModel(use_async_components=True, use_async_rollout_executor=True)

    obs = {"obs": torch.ones(2, 2)}
    state = asyncio.run(model.async_encode(obs))
    assert torch.allclose(state.tensors["latent"], torch.full((2, 2), 2.0))

    action = torch.ones(2, 1)
    next_state = asyncio.run(model.async_transition(state, action))
    decoded = asyncio.run(model.async_decode(next_state))
    assert torch.allclose(decoded.predictions["obs"], torch.full((2, 2), 2.0))

    trajectory = asyncio.run(model.async_rollout(state, torch.ones(3, 2, 1)))
    assert len(trajectory.states) == 1

    assert model.encoder_impl.async_calls > 0
    assert model.encoder_impl.sync_calls == 0
    assert model.dynamics_impl.async_calls > 0
    assert model.dynamics_impl.sync_calls == 0
    assert model.decoder_impl.async_calls > 0
    assert model.decoder_impl.sync_calls == 0

    executor = model.rollout_executor
    assert executor is not None
    assert executor.async_calls > 0
    assert executor.sync_calls == 0


def test_async_encode_validates_required_modalities() -> None:
    model = _AsyncHarnessModel(
        use_async_components=False,
        required_modalities=("obs", "goal"),
    )
    with pytest.raises(ValidationError, match="missing required modalities"):
        asyncio.run(model.async_encode({"obs": torch.ones(2, 2)}))
