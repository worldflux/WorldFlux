"""Batch representation for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor


def _map_tensors(value: Any, fn) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


@dataclass
class Batch:
    """Unified batch container for world models."""

    obs: Tensor | dict[str, Tensor]
    actions: Tensor | None = None
    next_obs: Tensor | dict[str, Tensor] | None = None
    rewards: Tensor | None = None
    terminations: Tensor | None = None
    mask: Tensor | None = None
    context: Tensor | dict[str, Tensor] | None = None
    target: Tensor | dict[str, Tensor] | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to(self, device: torch.device | str) -> Batch:
        device_obj = torch.device(device) if isinstance(device, str) else device
        return Batch(
            obs=_map_tensors(self.obs, lambda t: t.to(device_obj)),
            actions=_map_tensors(self.actions, lambda t: t.to(device_obj))
            if self.actions is not None
            else None,
            next_obs=_map_tensors(self.next_obs, lambda t: t.to(device_obj))
            if self.next_obs is not None
            else None,
            rewards=_map_tensors(self.rewards, lambda t: t.to(device_obj))
            if self.rewards is not None
            else None,
            terminations=_map_tensors(self.terminations, lambda t: t.to(device_obj))
            if self.terminations is not None
            else None,
            mask=_map_tensors(self.mask, lambda t: t.to(device_obj))
            if self.mask is not None
            else None,
            context=_map_tensors(self.context, lambda t: t.to(device_obj))
            if self.context is not None
            else None,
            target=_map_tensors(self.target, lambda t: t.to(device_obj))
            if self.target is not None
            else None,
            extras=self.extras,
        )

    def detach(self) -> Batch:
        return Batch(
            obs=_map_tensors(self.obs, lambda t: t.detach()),
            actions=_map_tensors(self.actions, lambda t: t.detach())
            if self.actions is not None
            else None,
            next_obs=_map_tensors(self.next_obs, lambda t: t.detach())
            if self.next_obs is not None
            else None,
            rewards=_map_tensors(self.rewards, lambda t: t.detach())
            if self.rewards is not None
            else None,
            terminations=_map_tensors(self.terminations, lambda t: t.detach())
            if self.terminations is not None
            else None,
            mask=_map_tensors(self.mask, lambda t: t.detach()) if self.mask is not None else None,
            context=_map_tensors(self.context, lambda t: t.detach())
            if self.context is not None
            else None,
            target=_map_tensors(self.target, lambda t: t.detach())
            if self.target is not None
            else None,
            extras=self.extras,
        )

    def clone(self) -> Batch:
        return Batch(
            obs=_map_tensors(self.obs, lambda t: t.clone()),
            actions=_map_tensors(self.actions, lambda t: t.clone())
            if self.actions is not None
            else None,
            next_obs=_map_tensors(self.next_obs, lambda t: t.clone())
            if self.next_obs is not None
            else None,
            rewards=_map_tensors(self.rewards, lambda t: t.clone())
            if self.rewards is not None
            else None,
            terminations=_map_tensors(self.terminations, lambda t: t.clone())
            if self.terminations is not None
            else None,
            mask=_map_tensors(self.mask, lambda t: t.clone()) if self.mask is not None else None,
            context=_map_tensors(self.context, lambda t: t.clone())
            if self.context is not None
            else None,
            target=_map_tensors(self.target, lambda t: t.clone())
            if self.target is not None
            else None,
            extras=dict(self.extras),
        )

    @property
    def batch_size(self) -> int:
        if isinstance(self.obs, Tensor):
            return self.obs.shape[0]
        for tensor in self.obs.values():
            return tensor.shape[0]
        raise ValueError("Cannot determine batch size from empty observation dict")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Batch:
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return {
            "obs": self.obs,
            "actions": self.actions,
            "next_obs": self.next_obs,
            "rewards": self.rewards,
            "terminations": self.terminations,
            "mask": self.mask,
            "context": self.context,
            "target": self.target,
            "extras": self.extras,
        }


class BatchProvider(Protocol):
    """Protocol for batch providers."""

    def sample(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: torch.device | str = "cpu",
    ) -> Batch: ...
