"""Specification types for observations, actions, tokens, and capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .exceptions import ContractValidationError
from .payloads import is_namespaced_extra_key


class ModalityKind(str, Enum):
    """Common modality kinds."""

    IMAGE = "image"
    VIDEO = "video"
    TEXT = "text"
    TOKENS = "tokens"
    VECTOR = "vector"
    AUDIO = "audio"
    OTHER = "other"


class Capability(str, Enum):
    """Model capability flags."""

    LATENT_DYNAMICS = "latent_dynamics"
    OBS_DECODER = "obs_decoder"
    REPRESENTATION = "representation"
    REWARD_PRED = "reward_pred"
    CONTINUE_PRED = "continue_pred"
    VIDEO_PRED = "video_pred"
    TOKEN_MODEL = "token_model"
    DIFFUSION = "diffusion"
    PLANNING = "planning"
    VALUE = "value"
    POLICY = "policy"


class ModelMaturity(str, Enum):
    """Public maturity tier for model families."""

    REFERENCE = "reference"
    EXPERIMENTAL = "experimental"
    SKELETON = "skeleton"


@dataclass(frozen=True)
class ModalitySpec:
    """Specification of a single modality tensor."""

    kind: ModalityKind
    shape: tuple[int, ...]
    dtype: str = "float32"
    layout: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class ObservationSpec:
    """Specification of observation or generic input tensors."""

    modalities: dict[str, ModalitySpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ConditionSpec:
    """Specification for optional conditioning signals."""

    modalities: dict[str, ModalitySpec] = field(default_factory=dict)
    allowed_extra_keys: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionSpec:
    """Specification of actions."""

    kind: str = "continuous"
    dim: int = 0
    discrete: bool = False
    num_actions: int | None = None

    def __post_init__(self) -> None:
        valid_kinds = {
            "none",
            "continuous",
            "discrete",
            "token",
            "latent",
            "text",
            "hybrid",
        }
        if self.kind not in valid_kinds:
            raise ContractValidationError(
                f"Unknown action kind '{self.kind}'. Expected one of: {sorted(valid_kinds)}"
            )
        if self.kind == "hybrid":
            warnings.warn(
                "Action kind 'hybrid' is deprecated in v0.2 and will be removed in v0.3.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.dim < 0:
            raise ContractValidationError(f"Action dim must be non-negative, got {self.dim}")
        if self.discrete and self.num_actions is None and self.dim > 0:
            raise ContractValidationError("Discrete actions require num_actions or a positive dim")


@dataclass(frozen=True)
class StateSpec:
    """Specification of latent state tensors."""

    tensors: dict[str, ModalitySpec] = field(default_factory=dict)


@dataclass(frozen=True)
class TokenSpec:
    """Specification for tokenized representations."""

    vocab_size: int
    seq_len: int
    dtype: str = "int64"


@dataclass(frozen=True)
class PredictionSpec:
    """Specification for model prediction tensors."""

    tensors: dict[str, ModalitySpec] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceLayout:
    """
    Explicit axis layout by field key.

    Layout strings use axis markers like ``B`` (batch) and ``T`` (time),
    for example ``BT...`` or ``B...``.
    """

    axes_by_field: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceFieldSpec:
    """
    Per-field sequence validation contract.

    This augments ``SequenceLayout`` with variable-length and axis constraints.
    """

    layout: str | None = None
    variable_length: bool = False
    required_axes: tuple[str, ...] = ("B",)
    allow_multiple_time_axes: bool = False


@dataclass(frozen=True)
class ModelIOContract:
    """Runtime I/O contract for unified validation across model families."""

    # Legacy name preserved for v0.2 compatibility.
    observation_spec: ObservationSpec = field(default_factory=ObservationSpec)
    action_spec: ActionSpec = field(default_factory=ActionSpec)
    state_spec: StateSpec = field(default_factory=StateSpec)
    prediction_spec: PredictionSpec = field(default_factory=PredictionSpec)
    sequence_layout: SequenceLayout = field(default_factory=SequenceLayout)
    required_batch_keys: tuple[str, ...] = ()
    required_state_keys: tuple[str, ...] = ()
    additional_batch_fields: dict[str, ModalitySpec] = field(default_factory=dict)

    # New contract fields for universal API.
    input_spec: ObservationSpec | None = None
    target_spec: ObservationSpec = field(default_factory=ObservationSpec)
    condition_spec: ConditionSpec = field(default_factory=ConditionSpec)
    # v3.1 additive fields.
    action_union_spec: tuple[ActionSpec, ...] = ()
    condition_extras_schema: dict[str, ModalitySpec] = field(default_factory=dict)
    sequence_field_spec: dict[str, SequenceFieldSpec] = field(default_factory=dict)

    @staticmethod
    def _root_name(key: str) -> str:
        return key.split(".", 1)[0].split(":", 1)[0]

    @property
    def effective_input_spec(self) -> ObservationSpec:
        return self.input_spec or self.observation_spec

    @property
    def effective_action_specs(self) -> tuple[ActionSpec, ...]:
        # v3 compatibility: if action_union_spec is omitted, fall back to singleton action_spec.
        return self.action_union_spec or (self.action_spec,)

    @property
    def effective_condition_extra_keys(self) -> tuple[str, ...]:
        # v3 compatibility: schema-only keys are treated as allowed extras automatically.
        merged = dict.fromkeys(
            tuple(self.condition_spec.allowed_extra_keys)
            + tuple(self.condition_extras_schema.keys())
        )
        return tuple(merged.keys())

    @property
    def effective_sequence_field_spec(self) -> dict[str, SequenceFieldSpec]:
        # v3 compatibility: sequence_layout entries are auto-upconverted into field specs.
        merged: dict[str, SequenceFieldSpec] = {
            field_name: SequenceFieldSpec(layout=layout)
            for field_name, layout in self.sequence_layout.axes_by_field.items()
        }
        for field_name, spec in self.sequence_field_spec.items():
            if field_name in merged and spec.layout is None:
                merged[field_name] = SequenceFieldSpec(
                    layout=merged[field_name].layout,
                    variable_length=spec.variable_length,
                    required_axes=spec.required_axes,
                    allow_multiple_time_axes=spec.allow_multiple_time_axes,
                )
            else:
                merged[field_name] = spec
        return merged

    def condition_extras_schema_dict(self) -> dict[str, dict[str, Any]]:
        return {
            key: {"dtype": spec.dtype, "shape": spec.shape}
            for key, spec in self.condition_extras_schema.items()
        }

    def validate(self) -> None:
        """Validate contract consistency."""
        self._validate_action_union_spec()
        self._validate_batch_key_requirements()
        self._validate_state_key_requirements()
        self._validate_sequence_layout()
        self._validate_sequence_field_spec()
        self._validate_condition_spec()

    def _declared_fields(self) -> set[str]:
        fields: set[str] = set()
        fields.update(self.effective_input_spec.modalities.keys())
        fields.update(self.target_spec.modalities.keys())
        fields.update(self.condition_spec.modalities.keys())
        fields.update(self._root_name(k) for k in self.effective_condition_extra_keys)
        fields.update(self._root_name(k) for k in self.condition_extras_schema.keys())
        fields.update(self._root_name(k) for k in self.additional_batch_fields.keys())
        return fields

    def _allowed_batch_roots(self) -> set[str]:
        reserved = {
            "obs",
            "inputs",
            "targets",
            "actions",
            "action",
            "next_obs",
            "rewards",
            "terminations",
            "mask",
            "context",
            "target",
            "conditions",
            "extras",
        }
        return reserved | self._declared_fields()

    def _validate_batch_key_requirements(self) -> None:
        valid = self._allowed_batch_roots()
        for key in self.required_batch_keys:
            root = self._root_name(key)
            if root not in valid:
                raise ContractValidationError(
                    f"Unknown required batch key '{key}'. Expected root in: {sorted(valid)}"
                )

    def _validate_state_key_requirements(self) -> None:
        available = set(self.state_spec.tensors.keys())
        for key in self.required_state_keys:
            if key not in available:
                raise ContractValidationError(
                    f"required_state_keys contains '{key}' but state_spec does not define it"
                )

    def _validate_sequence_layout(self) -> None:
        valid_fields = self._allowed_batch_roots()
        merged_layouts = dict(self.sequence_layout.axes_by_field)
        for field_name, spec in self.sequence_field_spec.items():
            if spec.layout is not None:
                merged_layouts[field_name] = spec.layout

        for field_name, layout in merged_layouts.items():
            root = self._root_name(field_name)
            if root not in valid_fields:
                raise ContractValidationError(
                    f"Unknown sequence layout field '{field_name}'. "
                    f"Expected root field in {sorted(valid_fields)}"
                )
            if not layout:
                raise ContractValidationError(
                    f"Sequence layout for '{field_name}' must not be empty"
                )
            if "B" not in layout:
                raise ContractValidationError(
                    f"Sequence layout for '{field_name}' must include batch axis 'B', got '{layout}'"
                )

    def _validate_sequence_field_spec(self) -> None:
        valid_fields = self._allowed_batch_roots()
        for field_name, spec in self.effective_sequence_field_spec.items():
            root = self._root_name(field_name)
            if root not in valid_fields:
                raise ContractValidationError(
                    f"Unknown sequence field spec '{field_name}'. "
                    f"Expected root field in {sorted(valid_fields)}"
                )
            layout = spec.layout
            if layout is None:
                if spec.variable_length:
                    raise ContractValidationError(
                        f"sequence_field_spec['{field_name}'] has variable_length=True but no layout"
                    )
                continue

            for axis in spec.required_axes:
                if axis not in layout:
                    raise ContractValidationError(
                        f"sequence_field_spec['{field_name}'] requires axis '{axis}' in layout '{layout}'"
                    )
            if spec.variable_length and "T" not in layout:
                raise ContractValidationError(
                    f"sequence_field_spec['{field_name}'] has variable_length=True but layout '{layout}' has no 'T' axis"
                )
            if not spec.allow_multiple_time_axes and layout.count("T") > 1:
                raise ContractValidationError(
                    f"sequence_field_spec['{field_name}'] layout '{layout}' contains multiple 'T' axes"
                )

    def _validate_condition_spec(self) -> None:
        self._validate_extra_keys(self.effective_condition_extra_keys, source="condition_spec")
        self._validate_extra_keys(
            self.condition_extras_schema.keys(), source="condition_extras_schema"
        )

    def _validate_action_union_spec(self) -> None:
        union = self.action_union_spec
        if not union:
            return
        if len({spec.kind for spec in union}) != len(union):
            raise ContractValidationError(
                "action_union_spec must not contain duplicate action kinds"
            )
        if self.action_spec.kind not in {spec.kind for spec in union}:
            raise ContractValidationError(
                "action_spec.kind must be represented in action_union_spec for compatibility"
            )

    @staticmethod
    def _validate_extra_keys(keys: Iterable[str], *, source: str) -> None:
        for key in keys:
            if not is_namespaced_extra_key(key):
                raise ContractValidationError(
                    f"{source} contains invalid namespaced extra key {key!r}. "
                    "Expected format 'wf.<domain>.<name>'."
                )
