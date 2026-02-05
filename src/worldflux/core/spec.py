"""Specification types for observations, actions, tokens, and capabilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    """Specification of observation inputs."""

    modalities: dict[str, ModalitySpec] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionSpec:
    """Specification of actions."""

    kind: str = "continuous"
    dim: int = 0
    discrete: bool = False
    num_actions: int | None = None


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

    observation_spec: ObservationSpec = field(default_factory=ObservationSpec)
    action_spec: ActionSpec = field(default_factory=ActionSpec)
    state_spec: StateSpec = field(default_factory=StateSpec)
    prediction_spec: PredictionSpec = field(default_factory=PredictionSpec)
    sequence_layout: SequenceLayout = field(default_factory=SequenceLayout)
    required_batch_keys: tuple[str, ...] = ()
    required_state_keys: tuple[str, ...] = ()
