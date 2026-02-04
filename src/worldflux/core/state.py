"""Unified latent state representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .exceptions import ShapeMismatchError, StateError


@dataclass
class State:
    """Generic state container (tensor dictionary + metadata)."""

    tensors: dict[str, Tensor] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Tensor | None = None) -> Tensor | None:
        return self.tensors.get(key, default)

    @property
    def batch_size(self) -> int:
        for tensor in self.tensors.values():
            return tensor.shape[0]
        raise ValueError("State has no tensors to infer batch size")

    @property
    def device(self) -> torch.device:
        for tensor in self.tensors.values():
            return tensor.device
        raise ValueError("State has no tensors to infer device")

    def to(self, device: torch.device | str) -> State:
        device_obj = torch.device(device) if isinstance(device, str) else device
        return State(
            tensors={k: v.to(device_obj) for k, v in self.tensors.items()},
            meta=self.meta,
        )

    def detach(self) -> State:
        return State(
            tensors={k: v.detach() for k, v in self.tensors.items()},
            meta=self.meta,
        )

    def clone(self) -> State:
        return State(
            tensors={k: v.clone() for k, v in self.tensors.items()},
            meta=dict(self.meta),
        )

    def validate(self) -> None:
        """Validate state tensor shapes and batch consistency."""
        if not self.tensors:
            raise StateError("State has no tensors")
        batch_size = None
        for name, tensor in self.tensors.items():
            if batch_size is None:
                batch_size = tensor.shape[0]
                continue
            if tensor.shape[0] != batch_size:
                raise ShapeMismatchError(
                    f"State tensor '{name}' batch size mismatch",
                    expected=(batch_size,),
                    got=(tensor.shape[0],),
                )
