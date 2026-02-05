"""Tests for runtime I/O contract validation."""

from __future__ import annotations

import pytest
import torch

from worldflux.core.batch import Batch
from worldflux.core.exceptions import ValidationError
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput
from worldflux.core.spec import (
    ActionSpec,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from worldflux.core.state import State


class _ContractModel(WorldModel):
    """Small helper model exposing a strict contract for tests."""

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs["obs"]
        return State(tensors={"latent": obs})

    def transition(self, state: State, action: torch.Tensor, deterministic: bool = False) -> State:
        return state

    def update(self, state: State, action: torch.Tensor, obs) -> State:
        return self.encode(obs)

    def decode(self, state: State):
        return None

    def loss(self, batch: Batch) -> LossOutput:
        loss = torch.tensor(0.0)
        return LossOutput(loss=loss, components={"zero": loss})

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
            ),
            action_spec=ActionSpec(kind="continuous", dim=2),
            state_spec=StateSpec(
                tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(8,))}
            ),
            prediction_spec=PredictionSpec(
                tensors={"reward": ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,))}
            ),
            sequence_layout=SequenceLayout(axes_by_field={"obs": "BT...", "actions": "BT..."}),
            required_batch_keys=("obs", "actions"),
            required_state_keys=("latent",),
        )


class _ExtendedContractModel(_ContractModel):
    def io_contract(self) -> ModelIOContract:
        base = super().io_contract()
        return ModelIOContract(
            observation_spec=base.observation_spec,
            action_spec=base.action_spec,
            state_spec=base.state_spec,
            prediction_spec=base.prediction_spec,
            sequence_layout=SequenceLayout(
                axes_by_field={
                    **base.sequence_layout.axes_by_field,
                    "extras.goal": "BT...",
                }
            ),
            required_batch_keys=("obs", "actions", "extras.goal"),
            required_state_keys=base.required_state_keys,
            additional_batch_fields={
                "extras.goal": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))
            },
        )


def test_model_io_contract_validate_passes_on_valid_contract():
    contract = _ContractModel().io_contract()
    contract.validate()


def test_model_io_contract_validate_rejects_unknown_layout_field():
    contract = ModelIOContract(
        observation_spec=ObservationSpec(
            modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
        ),
        action_spec=ActionSpec(kind="continuous", dim=2),
        state_spec=StateSpec(
            tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(8,))}
        ),
        sequence_layout=SequenceLayout(axes_by_field={"imaginary": "BT..."}),
        required_batch_keys=("obs",),
        required_state_keys=("latent",),
    )
    with pytest.raises(ValidationError, match="Unknown sequence layout field"):
        contract.validate()


def test_validate_batch_contract_raises_for_missing_required_key():
    model = _ContractModel()
    batch = Batch(obs=torch.randn(2, 3, 4))
    with pytest.raises(ValidationError, match="missing required batch keys"):
        model.validate_batch_contract(batch)


def test_validate_state_contract_raises_for_missing_required_tensor():
    model = _ContractModel()
    state = State(tensors={"other": torch.randn(2, 8)})
    with pytest.raises(ValidationError, match="missing required state tensors"):
        model.validate_state_contract(state)


def test_validate_batch_contract_supports_additional_batch_fields():
    model = _ExtendedContractModel()
    batch = Batch(
        obs=torch.randn(2, 3, 4),
        actions=torch.randn(2, 3, 2),
        extras={"goal": torch.randn(2, 3, 2)},
    )
    model.validate_batch_contract(batch)


def test_validate_batch_contract_rejects_missing_additional_batch_field():
    model = _ExtendedContractModel()
    batch = Batch(
        obs=torch.randn(2, 3, 4),
        actions=torch.randn(2, 3, 2),
    )
    with pytest.raises(ValidationError, match="missing required batch keys"):
        model.validate_batch_contract(batch)
