"""Batch representation for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch
from torch import Tensor

from .exceptions import ShapeMismatchError


def _map_tensors(value: Any, fn) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    return value


def _iter_tensors(value: Any):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_tensors(v)


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
    layouts: dict[str, str] = field(default_factory=dict)
    strict_layout: bool = False

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
            layouts=dict(self.layouts),
            strict_layout=self.strict_layout,
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
            layouts=dict(self.layouts),
            strict_layout=self.strict_layout,
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
            layouts=dict(self.layouts),
            strict_layout=self.strict_layout,
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
            "layouts": self.layouts,
            "strict_layout": self.strict_layout,
        }

    def with_layouts(self, layouts: dict[str, str], *, strict: bool | None = None) -> Batch:
        """Return a shallow copy with merged explicit axis layouts."""
        merged = dict(self.layouts)
        merged.update(layouts)
        return Batch(
            obs=self.obs,
            actions=self.actions,
            next_obs=self.next_obs,
            rewards=self.rewards,
            terminations=self.terminations,
            mask=self.mask,
            context=self.context,
            target=self.target,
            extras=self.extras,
            layouts=merged,
            strict_layout=self.strict_layout if strict is None else strict,
        )

    def _layout_for(self, field: str, subkey: str | None = None) -> str | None:
        candidates = []
        if subkey:
            candidates.extend([f"{field}.{subkey}", f"{field}:{subkey}"])
        candidates.append(field)
        for key in candidates:
            layout = self.layouts.get(key)
            if layout is not None:
                return layout
        return None

    def _validate_layout_keys(self) -> None:
        valid_root_fields = {
            "obs",
            "actions",
            "next_obs",
            "rewards",
            "terminations",
            "mask",
            "context",
            "target",
            "extras",
        }
        for key in self.layouts:
            root = key.split(".", 1)[0].split(":", 1)[0]
            if root not in valid_root_fields:
                raise ShapeMismatchError(
                    f"Unknown layout field '{key}'. "
                    f"Expected root field in {sorted(valid_root_fields)}",
                )

    @staticmethod
    def _axis_from_layout(layout: str, axis: str) -> int | None:
        pos = layout.find(axis)
        return None if pos < 0 else pos

    def _time_axis_for(self, field: str, tensor: Tensor, subkey: str | None = None) -> int | None:
        layout = self._layout_for(field, subkey=subkey)
        if layout is not None:
            axis = self._axis_from_layout(layout, "T")
            if axis is not None and axis >= tensor.dim():
                raise ShapeMismatchError(
                    f"{field} layout '{layout}' has invalid time axis for tensor rank {tensor.dim()}",
                    expected=(axis + 1,),
                    got=(tensor.dim(),),
                )
            return axis

        if self.strict_layout:
            return None

        # Backward-compatible fallback heuristic.
        if field == "obs":
            return 1 if tensor.dim() >= 3 else None
        return 1 if tensor.dim() >= 2 else None

    @staticmethod
    def _iter_named(value: Any, prefix: str = ""):
        if isinstance(value, Tensor):
            yield prefix, value
            return
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from Batch._iter_named(child, child_prefix)

    def validate(self, *, strict_time: bool = True) -> None:
        """Validate batch shapes and consistency.

        Args:
            strict_time: Enforce time dimension consistency when present.
        """
        # Infer batch/time from obs
        if self.strict_layout:
            self._validate_layout_keys()

        obs_entries = list(self._iter_named(self.obs))
        if not obs_entries:
            raise ShapeMismatchError("Cannot validate batch with empty obs dict")
        batch_size = obs_entries[0][1].shape[0]
        obs_time_dim: int | None = None

        for subkey, tensor in obs_entries:
            if tensor.shape[0] != batch_size:
                raise ShapeMismatchError(
                    "Observation batch size mismatch",
                    expected=(batch_size,),
                    got=(tensor.shape[0],),
                )
            time_axis = self._time_axis_for("obs", tensor, subkey=subkey or None)
            if not strict_time or time_axis is None:
                continue
            t_size = tensor.shape[time_axis]
            if obs_time_dim is None:
                obs_time_dim = t_size
            elif t_size != obs_time_dim:
                raise ShapeMismatchError(
                    "Observation time dimension mismatch",
                    expected=(obs_time_dim,),
                    got=(t_size,),
                )

        def _check(name: str, value: Any) -> None:
            for subkey, tensor in self._iter_named(value):
                if tensor.shape[0] != batch_size:
                    raise ShapeMismatchError(
                        f"{name} batch size mismatch",
                        expected=(batch_size,),
                        got=(tensor.shape[0],),
                    )
                if not strict_time or obs_time_dim is None:
                    continue
                time_axis = self._time_axis_for(name, tensor, subkey=subkey or None)
                if time_axis is None:
                    continue
                if tensor.shape[time_axis] != obs_time_dim:
                    raise ShapeMismatchError(
                        f"{name} time dimension mismatch",
                        expected=(obs_time_dim,),
                        got=(tensor.shape[time_axis],),
                    )

        if self.actions is not None:
            _check("actions", self.actions)
        if self.next_obs is not None:
            _check("next_obs", self.next_obs)
        if self.rewards is not None:
            _check("rewards", self.rewards)
        if self.terminations is not None:
            _check("terminations", self.terminations)
        if self.mask is not None:
            _check("mask", self.mask)
        if self.context is not None:
            _check("context", self.context)
        if self.target is not None:
            _check("target", self.target)


class BatchProvider(Protocol):
    """Protocol for batch providers."""

    def sample(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: torch.device | str = "cpu",
    ) -> Batch: ...


class TransitionProvider(BatchProvider, Protocol):
    """Provider for transition-style batches."""


class SequenceProvider(BatchProvider, Protocol):
    """Provider for sequence batches. May expose explicit layouts."""

    def batch_layout(self) -> dict[str, str]: ...


class TokenProvider(SequenceProvider, Protocol):
    """Provider for token sequence batches."""


class VideoProvider(SequenceProvider, Protocol):
    """Provider for video sequence batches."""
