"""Model output and loss containers."""

from __future__ import annotations

from collections.abc import ItemsView
from dataclasses import dataclass, field
from typing import Any

from torch import Tensor

from .exceptions import ShapeMismatchError
from .spec import PredictionSpec, SequenceLayout
from .state import State


@dataclass
class ModelOutput:
    """Standardized model output container."""

    predictions: dict[str, Tensor] = field(default_factory=dict)
    state: State | None = None
    uncertainty: Tensor | None = None
    aux: dict[str, Any] = field(default_factory=dict)
    prediction_spec: PredictionSpec | None = None
    sequence_layout: SequenceLayout | None = None

    # v0.2 compatibility for legacy keyword argument.
    def __init__(
        self,
        predictions: dict[str, Tensor] | None = None,
        state: State | None = None,
        uncertainty: Tensor | None = None,
        aux: dict[str, Any] | None = None,
        prediction_spec: PredictionSpec | None = None,
        sequence_layout: SequenceLayout | None = None,
        preds: dict[str, Tensor] | None = None,
    ) -> None:
        self.predictions = dict(predictions or preds or {})
        self.state = state
        self.uncertainty = uncertainty
        self.aux = dict(aux or {})
        self.prediction_spec = prediction_spec
        self.sequence_layout = sequence_layout

    @property
    def preds(self) -> dict[str, Tensor]:
        """Backward-compatible alias for predictions."""
        return self.predictions

    @preds.setter
    def preds(self, value: dict[str, Tensor]) -> None:
        self.predictions = value

    def validate(self) -> None:
        """Validate prediction tensor shapes and batch consistency."""
        if not self.predictions:
            return
        batch_size = None
        for name, tensor in self.predictions.items():
            if batch_size is None:
                batch_size = tensor.shape[0]
            elif tensor.shape[0] != batch_size:
                raise ShapeMismatchError(
                    f"Prediction '{name}' batch size mismatch",
                    expected=(batch_size,),
                    got=(tensor.shape[0],),
                )
        if self.state is not None and self.state.tensors:
            state_batch = self.state.batch_size
            if batch_size is not None and state_batch != batch_size:
                raise ShapeMismatchError(
                    "ModelOutput state/predictions batch size mismatch",
                    expected=(batch_size,),
                    got=(state_batch,),
                )
        if self.prediction_spec is not None:
            for key in self.prediction_spec.tensors:
                if key not in self.predictions:
                    raise ShapeMismatchError(f"Prediction spec requires missing key: {key}")

    def items(self) -> ItemsView[str, Tensor]:
        """Compatibility helper for iterating over predictions."""
        return self.predictions.items()


class WorldModelOutput(ModelOutput):
    """Future-facing alias used by the universal API naming."""


@dataclass
class LossOutput:
    """Standardized loss container."""

    loss: Tensor
    components: dict[str, Tensor] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def items(self) -> ItemsView[str, Tensor]:
        """Compatibility helper for iterating over losses."""
        return {"loss": self.loss, **self.components}.items()
