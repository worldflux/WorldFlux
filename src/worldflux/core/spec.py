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
