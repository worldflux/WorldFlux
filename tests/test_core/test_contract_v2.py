"""Tests for v0.2+ universal contract behavior."""

from __future__ import annotations

import pytest
import torch

from worldflux.core.exceptions import CapabilityError, ContractValidationError, ValidationError
from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.payloads import ConditionPayload
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


class _NoDecoderModel(WorldModel):
    def loss(self, batch):
        return LossOutput(loss=torch.tensor(0.0))


class _ConditionAwareModel(WorldModel):
    def encode(self, obs, deterministic: bool = False):
        del obs, deterministic
        return State(tensors={"latent": torch.zeros(1, 2)})

    def transition(self, state, action, conditions=None, deterministic: bool = False):
        del action, deterministic
        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        return state

    def update(self, state, action, obs, conditions=None):
        del action, obs, conditions
        return state

    def decode(self, state, conditions=None):
        del conditions
        return ModelOutput(predictions={"latent": state.tensors["latent"]})

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))}
            ),
            action_spec=ActionSpec(kind="continuous", dim=2),
            state_spec=StateSpec(
                tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))}
            ),
            prediction_spec=PredictionSpec(
                tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(2,))}
            ),
            condition_spec=ConditionSpec(allowed_extra_keys=("wf.planner.horizon",)),
            required_state_keys=("latent",),
        )

    def loss(self, batch):
        del batch
        return LossOutput(loss=torch.tensor(0.0))


def test_action_spec_accepts_extended_kinds():
    for kind in ("none", "continuous", "discrete", "token", "latent", "text"):
        spec = ActionSpec(
            kind=kind,
            dim=0 if kind in {"none", "text"} else 2,
            discrete=kind == "discrete",
            num_actions=2 if kind == "discrete" else None,
        )
        assert spec.kind == kind

    with pytest.warns(DeprecationWarning, match="deprecated"):
        spec = ActionSpec(kind="hybrid", dim=2)
    assert spec.kind == "hybrid"


def test_action_spec_rejects_unknown_kind():
    with pytest.raises(ContractValidationError, match="Unknown action kind"):
        ActionSpec(kind="invalid", dim=1)


def test_model_io_contract_v2_fields_validate():
    contract = ModelIOContract(
        observation_spec=ObservationSpec(
            modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
        ),
        input_spec=ObservationSpec(
            modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
        ),
        target_spec=ObservationSpec(
            modalities={"next_obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
        ),
        condition_spec=ConditionSpec(
            modalities={"goal": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))},
            allowed_extra_keys=("wf.planner.horizon",),
        ),
        action_spec=ActionSpec(kind="latent", dim=8),
        state_spec=StateSpec(
            tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(16,))}
        ),
        prediction_spec=PredictionSpec(
            tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(16,))}
        ),
        sequence_layout=SequenceLayout(
            axes_by_field={"obs": "BT...", "goal": "BT...", "actions": "BT..."}
        ),
        required_batch_keys=("obs", "goal", "actions"),
        required_state_keys=("latent",),
    )
    contract.validate()


def test_model_io_contract_rejects_invalid_condition_extra_key():
    contract = ModelIOContract(
        observation_spec=ObservationSpec(
            modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
        ),
        action_spec=ActionSpec(kind="continuous", dim=2),
        state_spec=StateSpec(
            tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(8,))}
        ),
        prediction_spec=PredictionSpec(tensors={}),
        condition_spec=ConditionSpec(allowed_extra_keys=("planner.horizon",)),
        required_state_keys=("latent",),
    )
    with pytest.raises(ContractValidationError, match="invalid namespaced extra key"):
        contract.validate()


def test_model_io_contract_v2_rejects_undeclared_required_field():
    contract = ModelIOContract(
        observation_spec=ObservationSpec(
            modalities={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(4,))}
        ),
        action_spec=ActionSpec(kind="continuous", dim=2),
        state_spec=StateSpec(
            tensors={"latent": ModalitySpec(kind=ModalityKind.VECTOR, shape=(8,))}
        ),
        prediction_spec=PredictionSpec(tensors={}),
        sequence_layout=SequenceLayout(axes_by_field={"obs": "BT..."}),
        required_batch_keys=("imaginary_field",),
        required_state_keys=("latent",),
    )
    with pytest.raises(ContractValidationError, match="Unknown required batch key"):
        contract.validate()


def test_world_model_decode_without_decoder_raises_capability_error():
    model = _NoDecoderModel()
    with pytest.raises(CapabilityError):
        model.decode(State(tensors={"latent": torch.randn(2, 4)}))


def test_condition_payload_undeclared_extra_warns_in_v02():
    model = _ConditionAwareModel()
    state = model.encode(torch.randn(1, 2))
    with pytest.warns(DeprecationWarning, match="undeclared keys"):
        model.transition(
            state,
            torch.randn(1, 2),
            conditions=ConditionPayload(extras={"wf.custom.goal": torch.randn(1, 1)}),
        )


def test_condition_payload_undeclared_extra_fails_in_v3():
    model = _ConditionAwareModel()
    model._wf_api_version = "v3"
    state = model.encode(torch.randn(1, 2))
    with pytest.raises(ValidationError, match="undeclared keys"):
        model.transition(
            state,
            torch.randn(1, 2),
            conditions=ConditionPayload(extras={"wf.custom.goal": torch.randn(1, 1)}),
        )
