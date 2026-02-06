"""Typed payload objects for unified world-model interfaces."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import torch
from torch import Tensor

ActionKind = Literal["none", "continuous", "discrete", "token", "latent", "text", "hybrid"]
PLANNER_HORIZON_KEY = "wf.planner.horizon"
PLANNER_SEQUENCE_KEY = "wf.planner.sequence"
ACTION_COMPONENTS_KEY = "wf.action.components"

_NAMESPACED_KEY_PATTERN = re.compile(r"^wf\.[a-zA-Z0-9_-]+\.[a-zA-Z0-9_.-]+$")


class ActionSpecLike(Protocol):
    """Structural protocol for action contract checks."""

    @property
    def kind(self) -> str: ...

    @property
    def dim(self) -> int: ...

    @property
    def discrete(self) -> bool: ...

    @property
    def num_actions(self) -> int | None: ...


def is_namespaced_extra_key(key: str) -> bool:
    """Return True for valid WorldFlux namespaced extras key."""
    return bool(_NAMESPACED_KEY_PATTERN.match(key))


@dataclass
class ActionPayload:
    """Polymorphic action container that supports multiple control modalities."""

    kind: ActionKind = "none"
    tensor: Tensor | None = None
    tokens: Tensor | None = None
    latent: Tensor | None = None
    text: list[str] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def primary(self) -> Tensor | None:
        if self.tensor is not None:
            return self.tensor
        if self.tokens is not None:
            return self.tokens
        if self.latent is not None:
            return self.latent
        return None

    def validate(self, *, api_version: str = "v0.2") -> None:
        """Validate payload consistency."""
        if self.kind == "hybrid":
            msg = "ActionPayload.kind='hybrid' is deprecated in v0.2 and will be removed in v0.3."
            if api_version == "v3":
                raise ValueError(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        primary_count = sum(
            v is not None for v in (self.tensor, self.tokens, self.latent, self.text)
        )
        if primary_count > 1:
            raise ValueError(
                "ActionPayload must define only one primary representation among "
                "tensor/tokens/latent/text."
            )
        if self.kind == "none" and primary_count > 0:
            raise ValueError(
                "ActionPayload.kind='none' must not include tensor/tokens/latent/text."
            )


@dataclass
class ConditionPayload:
    """Optional side-conditions for conditional world modeling."""

    text_condition: Tensor | list[str] | None = None
    goal: Tensor | None = None
    spatial: Tensor | None = None
    camera_pose: Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def validate(
        self,
        *,
        strict: bool = False,
        allowed_extra_keys: set[str] | None = None,
        api_version: str = "v0.2",
    ) -> None:
        """Validate condition extras naming and optional allow-list contract."""
        invalid = [k for k in self.extras if not is_namespaced_extra_key(k)]
        if invalid:
            msg = (
                "ConditionPayload.extras keys must follow namespaced format "
                f"'wf.<domain>.<name>', got: {sorted(invalid)}"
            )
            if strict or api_version == "v3":
                raise ValueError(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        if allowed_extra_keys is not None:
            unknown = [k for k in self.extras if k not in allowed_extra_keys]
            if unknown:
                msg = f"ConditionPayload.extras contains undeclared keys: {sorted(unknown)}"
                if strict or api_version == "v3":
                    raise ValueError(msg)
                warnings.warn(msg, DeprecationWarning, stacklevel=2)


@dataclass
class ActionSequence:
    """Action sequence used by rollout APIs."""

    tensor: Tensor | None = None
    payloads: list[ActionPayload] | None = None

    def __len__(self) -> int:
        if self.tensor is not None:
            return int(self.tensor.shape[0])
        if self.payloads is not None:
            return len(self.payloads)
        return 0


@dataclass
class WorldModelInput:
    """Unified model input object."""

    observations: dict[str, Tensor] = field(default_factory=dict)
    context: dict[str, Tensor] = field(default_factory=dict)
    action: ActionPayload | None = None
    conditions: ConditionPayload = field(default_factory=ConditionPayload)


@dataclass
class WorldModelOutput:
    """Typed top-level output shape for future public API migration."""

    predictions: dict[str, Tensor] = field(default_factory=dict)
    latent_state: Tensor | None = None
    reward: Tensor | None = None
    termination: Tensor | None = None
    value: Tensor | None = None
    action_logits: Tensor | None = None
    uncertainty: Tensor | None = None
    auxiliary: dict[str, Tensor] = field(default_factory=dict)


def _infer_feature_dim(tensor: Tensor) -> int:
    if tensor.dim() == 0:
        raise ValueError("Action tensors must have rank >= 1")
    if tensor.dim() == 1:
        return int(tensor.shape[0])
    return int(tensor.shape[-1])


def validate_action_payload_against_spec(
    payload: ActionPayload,
    action_spec: ActionSpecLike,
    *,
    api_version: str = "v0.2",
) -> None:
    """Validate payload against model action contract."""
    payload.validate(api_version=api_version)

    spec_kind = str(action_spec.kind)
    if spec_kind == "hybrid":
        msg = "ActionSpec.kind='hybrid' is deprecated in v0.2 and removed in v0.3."
        if api_version == "v3":
            raise ValueError(msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    if payload.kind == "none":
        if spec_kind not in {"none", "hybrid"}:
            raise ValueError(
                f"Action payload kind 'none' is incompatible with action_spec.kind={spec_kind!r}"
            )
        return

    if spec_kind not in {"hybrid", payload.kind}:
        raise ValueError(
            f"Action payload kind {payload.kind!r} is incompatible with action_spec.kind={spec_kind!r}"
        )

    if payload.kind in {"continuous", "discrete"} and payload.tensor is None:
        raise ValueError(f"Action payload kind {payload.kind!r} requires tensor field")
    if payload.kind == "token" and payload.tokens is None and payload.tensor is None:
        raise ValueError("Action payload kind 'token' requires tokens or tensor field")
    if payload.kind == "latent" and payload.latent is None and payload.tensor is None:
        raise ValueError("Action payload kind 'latent' requires latent or tensor field")
    if payload.kind == "text" and not payload.text:
        raise ValueError("Action payload kind 'text' requires non-empty text field")

    expected_dim = int(action_spec.dim)
    if expected_dim <= 0:
        return

    if payload.kind in {"continuous", "token", "latent"}:
        primary = payload.primary()
        if primary is None:
            return
        got_dim = _infer_feature_dim(primary)
        if got_dim != expected_dim:
            raise ValueError(
                f"Action feature dim mismatch for kind={payload.kind!r}: "
                f"expected {expected_dim}, got {got_dim}"
            )
        return

    if payload.kind == "discrete" and payload.tensor is not None:
        tensor = payload.tensor
        num_actions = int(action_spec.num_actions or expected_dim)
        if not tensor.dtype.is_floating_point and tensor.dtype != torch.bool:
            # Integer-like tensors are treated as class indices.
            return
        got_dim = _infer_feature_dim(tensor)
        if got_dim != num_actions:
            raise ValueError(
                f"Discrete action dim mismatch: expected trailing dim {num_actions}, got {got_dim}"
            )


def _infer_horizon_from_tensor(tensor: Tensor) -> int:
    if tensor.dim() == 0:
        raise ValueError("Cannot infer planning horizon from scalar tensor")
    if tensor.dim() == 1:
        return 1
    if tensor.dim() == 2:
        first = int(tensor.shape[0])
        return first if first > 1 else 1
    return int(tensor.shape[0])


def _normalize_tensor_for_horizon(tensor: Tensor, horizon: int) -> Tensor:
    if horizon < 1:
        raise ValueError(f"Planning horizon must be >= 1, got {horizon}")

    if horizon == 1:
        if tensor.dim() == 1:
            return tensor.unsqueeze(0).unsqueeze(0)
        if tensor.dim() == 2:
            return tensor.unsqueeze(0)
        if tensor.dim() >= 3:
            if int(tensor.shape[0]) != 1:
                raise ValueError(
                    f"Horizon mismatch: expected first dimension 1 for single-step planning, got {tuple(tensor.shape)}"
                )
            return tensor
        raise ValueError("Planner tensor must have rank >= 1")

    if tensor.dim() == 1:
        raise ValueError("Sequence planning requires tensor rank >= 2")
    if int(tensor.shape[0]) != horizon:
        raise ValueError(
            f"Horizon mismatch: extras[{PLANNER_HORIZON_KEY!r}]={horizon}, "
            f"but tensor first dimension is {int(tensor.shape[0])}"
        )
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    return tensor


def normalize_planned_action(
    payload: ActionPayload,
    *,
    api_version: str = "v0.2",
) -> ActionSequence:
    """
    Normalize planner output payload into an ActionSequence.

    v0.2 behavior:
    - missing ``wf.planner.horizon`` is inferred with DeprecationWarning.
    - legacy ``wf.planner.sequence`` is accepted with DeprecationWarning.

    v3 behavior:
    - ``wf.planner.horizon`` is mandatory.
    - ``wf.planner.sequence`` is rejected.
    """

    payload.validate(api_version=api_version)
    extras = payload.extras

    if PLANNER_SEQUENCE_KEY in extras:
        msg = (
            f"{PLANNER_SEQUENCE_KEY!r} is deprecated in v0.2 and removed in v0.3. "
            f"Use {PLANNER_HORIZON_KEY!r}."
        )
        if api_version == "v3":
            raise ValueError(msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)

    horizon_raw = extras.get(PLANNER_HORIZON_KEY)
    horizon: int
    if horizon_raw is None:
        if api_version == "v3":
            raise ValueError(f"Missing required planner metadata: extras[{PLANNER_HORIZON_KEY!r}]")
        primary = payload.primary()
        if primary is None:
            raise ValueError("Cannot infer planner horizon without primary tensor payload")
        horizon = _infer_horizon_from_tensor(primary)
        msg = (
            f"Planner payload missing extras[{PLANNER_HORIZON_KEY!r}]. "
            "Inferred horizon from tensor shape for v0.2 compatibility; this will be an error in v0.3."
        )
        if api_version == "v3":
            raise ValueError(msg)
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
    else:
        horizon = int(horizon_raw)
        if horizon < 1:
            raise ValueError(
                f"Planner horizon in extras[{PLANNER_HORIZON_KEY!r}] must be >= 1, got {horizon}"
            )

    primary = payload.primary()
    if primary is not None:
        tensor = _normalize_tensor_for_horizon(primary, horizon)
        return ActionSequence(tensor=tensor)

    if horizon == 1:
        return ActionSequence(payloads=[payload])
    raise ValueError("Planner payload requires tensor/tokens/latent for multi-step horizon")


def first_action(payload: ActionPayload, *, api_version: str = "v0.2") -> ActionPayload:
    """Return first-step action payload from planner output."""
    sequence = normalize_planned_action(payload, api_version=api_version)
    if sequence.tensor is not None:
        first = sequence.tensor[0]
        return ActionPayload(
            kind=payload.kind,
            tensor=first,
            extras={PLANNER_HORIZON_KEY: 1},
        )
    if sequence.payloads:
        first_payload = sequence.payloads[0]
        return ActionPayload(
            kind=first_payload.kind,
            tensor=first_payload.tensor,
            tokens=first_payload.tokens,
            latent=first_payload.latent,
            text=first_payload.text,
            extras={**first_payload.extras, PLANNER_HORIZON_KEY: 1},
        )
    raise ValueError("Planner payload sequence is empty")
