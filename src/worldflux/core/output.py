"""Model output and loss containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from .state import State


@dataclass
class ModelOutput:
    """Standardized model output container."""

    preds: dict[str, Tensor] = field(default_factory=dict)
    state: State | None = None
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class LossOutput:
    """Standardized loss container."""

    loss: Tensor
    components: dict[str, Tensor] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
