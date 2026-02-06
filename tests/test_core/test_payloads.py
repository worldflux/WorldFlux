"""Tests for universal payload types."""

from __future__ import annotations

import pytest
import torch

from worldflux.core.payloads import (
    ACTION_COMPONENTS_KEY,
    PLANNER_HORIZON_KEY,
    PLANNER_SEQUENCE_KEY,
    ActionPayload,
    ActionSequence,
    ConditionPayload,
    WorldModelInput,
    first_action,
    is_namespaced_extra_key,
    normalize_planned_action,
    validate_action_payload_against_spec,
    validate_action_payload_against_union,
)
from worldflux.core.spec import ActionSpec


def test_action_payload_primary_prefers_tensor_then_tokens_then_latent():
    p = ActionPayload(kind="continuous", tensor=torch.randn(2, 3))
    assert p.primary() is p.tensor

    p = ActionPayload(kind="token", tokens=torch.randint(0, 10, (2, 4)))
    assert p.primary() is p.tokens

    p = ActionPayload(kind="latent", latent=torch.randn(2, 8))
    assert p.primary() is p.latent


def test_action_payload_validate_rejects_multiple_primaries():
    payload = ActionPayload(
        kind="continuous",
        tensor=torch.randn(2, 3),
        latent=torch.randn(2, 3),
    )
    with pytest.raises(ValueError, match="only one primary representation"):
        payload.validate()


def test_condition_payload_defaults_are_optional():
    c = ConditionPayload()
    assert c.text_condition is None
    assert c.goal is None
    assert c.spatial is None
    assert c.camera_pose is None
    assert c.extras == {}


def test_condition_payload_requires_namespaced_extras_in_strict_mode():
    payload = ConditionPayload(extras={"goal": torch.randn(2, 4)})
    with pytest.raises(ValueError, match="namespaced format"):
        payload.validate(strict=True, api_version="v3")


def test_action_sequence_length_from_tensor_or_payloads():
    seq = ActionSequence(tensor=torch.randn(5, 2, 3))
    assert len(seq) == 5

    seq = ActionSequence(payloads=[ActionPayload(kind="none") for _ in range(3)])
    assert len(seq) == 3


def test_world_model_input_holds_observation_context_action_conditions():
    inp = WorldModelInput(
        observations={"obs": torch.randn(2, 4)},
        context={"obs": torch.randn(2, 4)},
        action=ActionPayload(kind="continuous", tensor=torch.randn(2, 2)),
        conditions=ConditionPayload(goal=torch.randn(2, 4)),
    )
    assert "obs" in inp.observations
    assert "obs" in inp.context
    assert inp.action is not None
    assert inp.conditions.goal is not None


def test_normalize_planned_action_with_explicit_horizon():
    payload = ActionPayload(
        kind="continuous",
        tensor=torch.randn(4, 3),
        extras={PLANNER_HORIZON_KEY: 4},
    )
    seq = normalize_planned_action(payload, api_version="v0.2")
    assert seq.tensor is not None
    assert seq.tensor.shape == (4, 1, 3)


def test_normalize_planned_action_infers_horizon_with_warning_in_v02():
    payload = ActionPayload(kind="continuous", tensor=torch.randn(3, 2))
    with pytest.warns(DeprecationWarning, match="missing extras"):
        seq = normalize_planned_action(payload, api_version="v0.2")
    assert seq.tensor is not None
    assert seq.tensor.shape == (3, 1, 2)


def test_normalize_planned_action_missing_horizon_errors_in_v3():
    payload = ActionPayload(kind="continuous", tensor=torch.randn(3, 2))
    with pytest.raises(ValueError, match="Missing required planner metadata"):
        normalize_planned_action(payload, api_version="v3")


def test_normalize_planned_action_legacy_sequence_key_warns_in_v02():
    payload = ActionPayload(
        kind="continuous",
        tensor=torch.randn(2, 3),
        extras={PLANNER_SEQUENCE_KEY: True, PLANNER_HORIZON_KEY: 2},
    )
    with pytest.warns(DeprecationWarning, match="deprecated"):
        seq = normalize_planned_action(payload, api_version="v0.2")
    assert seq.tensor is not None


def test_normalize_planned_action_legacy_sequence_key_errors_in_v3():
    payload = ActionPayload(
        kind="continuous",
        tensor=torch.randn(2, 3),
        extras={PLANNER_SEQUENCE_KEY: True, PLANNER_HORIZON_KEY: 2},
    )
    with pytest.raises(ValueError, match="removed in v0.3"):
        normalize_planned_action(payload, api_version="v3")


def test_first_action_extracts_single_step_payload():
    payload = ActionPayload(
        kind="continuous",
        tensor=torch.randn(5, 3),
        extras={PLANNER_HORIZON_KEY: 5},
    )
    first = first_action(payload, api_version="v0.2")
    assert first.tensor is not None
    assert first.tensor.shape == (1, 3)
    assert first.extras[PLANNER_HORIZON_KEY] == 1


def test_namespaced_extra_key_helper_and_reserved_action_components_key():
    assert is_namespaced_extra_key("wf.planner.horizon")
    assert not is_namespaced_extra_key("planner.horizon")
    assert ACTION_COMPONENTS_KEY == "wf.action.components"


def test_validate_action_payload_against_spec_accepts_matching_continuous_payload():
    payload = ActionPayload(kind="continuous", tensor=torch.randn(2, 3))
    spec = ActionSpec(kind="continuous", dim=3)
    validate_action_payload_against_spec(payload, spec, api_version="v3")


def test_validate_action_payload_against_spec_rejects_kind_mismatch():
    payload = ActionPayload(kind="token", tokens=torch.randint(0, 10, (2, 4)))
    spec = ActionSpec(kind="continuous", dim=4)
    with pytest.raises(ValueError, match="incompatible"):
        validate_action_payload_against_spec(payload, spec, api_version="v3")


def test_validate_action_payload_against_spec_rejects_dim_mismatch():
    payload = ActionPayload(kind="continuous", tensor=torch.randn(2, 2))
    spec = ActionSpec(kind="continuous", dim=3)
    with pytest.raises(ValueError, match="feature dim mismatch"):
        validate_action_payload_against_spec(payload, spec, api_version="v3")


def test_validate_action_payload_against_spec_accepts_discrete_indices():
    payload = ActionPayload(kind="discrete", tensor=torch.tensor([0, 1, 2], dtype=torch.int64))
    spec = ActionSpec(kind="discrete", dim=4, discrete=True, num_actions=4)
    validate_action_payload_against_spec(payload, spec, api_version="v3")


def test_validate_action_payload_against_union_accepts_any_matching_variant():
    payload = ActionPayload(kind="token", tokens=torch.randint(0, 10, (2, 4)))
    specs = (ActionSpec(kind="continuous", dim=4), ActionSpec(kind="token", dim=4))
    validate_action_payload_against_union(payload, specs, api_version="v3")


def test_validate_action_payload_against_union_rejects_when_none_match():
    payload = ActionPayload(kind="text", text=["left"])
    specs = (ActionSpec(kind="continuous", dim=2), ActionSpec(kind="token", dim=2))
    with pytest.raises(ValueError, match="did not match any action union"):
        validate_action_payload_against_union(payload, specs, api_version="v3")


def test_condition_payload_schema_validation_in_strict_mode():
    payload = ConditionPayload(extras={"wf.custom.goal": torch.ones(2, 3, dtype=torch.float32)})
    payload.validate(
        strict=True,
        allowed_extra_keys={"wf.custom.goal"},
        extra_schema={"wf.custom.goal": {"dtype": "float32", "shape": (3,)}},
        api_version="v3",
    )

    bad = ConditionPayload(extras={"wf.custom.goal": torch.ones(2, 2, dtype=torch.int64)})
    with pytest.raises(ValueError, match="dtype mismatch|shape mismatch"):
        bad.validate(
            strict=True,
            allowed_extra_keys={"wf.custom.goal"},
            extra_schema={"wf.custom.goal": {"dtype": "float32", "shape": (3,)}},
            api_version="v3",
        )
