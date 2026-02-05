"""Specification types for observations, actions, tokens, and capabilities."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

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

    @staticmethod
    def _root_name(key: str) -> str:
        return key.split(".", 1)[0].split(":", 1)[0]

    @property
    def effective_input_spec(self) -> ObservationSpec:
        return self.input_spec or self.observation_spec

    def validate(self) -> None:
        """Validate contract consistency."""
        self._validate_batch_key_requirements()
        self._validate_state_key_requirements()
        self._validate_sequence_layout()
        self._validate_condition_spec()

    def _declared_fields(self) -> set[str]:
        fields: set[str] = set()
        fields.update(self.effective_input_spec.modalities.keys())
        fields.update(self.target_spec.modalities.keys())
        fields.update(self.condition_spec.modalities.keys())
        fields.update(self._root_name(k) for k in self.condition_spec.allowed_extra_keys)
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
        for field_name, layout in self.sequence_layout.axes_by_field.items():
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

    def _validate_condition_spec(self) -> None:
        self._validate_extra_keys(self.condition_spec.allowed_extra_keys, source="condition_spec")

    @staticmethod
    def _validate_extra_keys(keys: Iterable[str], *, source: str) -> None:
        for key in keys:
            if not is_namespaced_extra_key(key):
                raise ContractValidationError(
                    f"{source} contains invalid namespaced extra key {key!r}. "
                    "Expected format 'wf.<domain>.<name>'."
                )
