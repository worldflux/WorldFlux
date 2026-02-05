"""Batch representation for training and inference."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Protocol

import torch
from torch import Tensor

from .exceptions import ShapeMismatchError


def _map_tensors(value: Any, fn) -> Any:
    if isinstance(value, Tensor):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_tensors(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_tensors(v, fn) for v in value]
    if isinstance(value, tuple):
        return tuple(_map_tensors(v, fn) for v in value)
    if is_dataclass(value) and not isinstance(value, type):
        updates = {
            name: _map_tensors(getattr(value, name), fn)
            for name in value.__dataclass_fields__
            if hasattr(value, name)
        }
        return replace(value, **updates)
    return value


def _iter_tensors(value: Any):
    if isinstance(value, Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _iter_tensors(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from _iter_tensors(v)
    elif is_dataclass(value) and not isinstance(value, type):
        for name in value.__dataclass_fields__:
            if hasattr(value, name):
                yield from _iter_tensors(getattr(value, name))


@dataclass
class Batch:
    """Unified batch container for world models.

    v0.2 keeps legacy fields (obs/actions/...) while introducing generic
    structures (inputs/targets/conditions).
    """

    # Legacy fields.
    obs: Tensor | dict[str, Tensor] | None = None
    actions: Tensor | None = None
    next_obs: Tensor | dict[str, Tensor] | None = None
    rewards: Tensor | None = None
    terminations: Tensor | None = None
    mask: Tensor | None = None
    context: Tensor | dict[str, Tensor] | None = None
    target: Tensor | dict[str, Tensor] | None = None

    # New generic fields.
    inputs: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    conditions: dict[str, Any] = field(default_factory=dict)

    extras: dict[str, Any] = field(default_factory=dict)
    layouts: dict[str, str] = field(default_factory=dict)
    strict_layout: bool = False

    def __post_init__(self) -> None:
        if not self.inputs:
            inferred_inputs: dict[str, Any] = {}
            if self.obs is not None:
                inferred_inputs["obs"] = self.obs
            if self.context is not None:
                inferred_inputs["context"] = self.context
            if self.actions is not None:
                inferred_inputs["actions"] = self.actions
            self.inputs = inferred_inputs

        if not self.targets:
            inferred_targets: dict[str, Any] = {}
            if self.target is not None:
                inferred_targets["target"] = self.target
            if self.next_obs is not None:
                inferred_targets["next_obs"] = self.next_obs
            if self.rewards is not None:
                inferred_targets["rewards"] = self.rewards
            if self.terminations is not None:
                inferred_targets["terminations"] = self.terminations
            self.targets = inferred_targets

        if self.obs is None and "obs" in self.inputs:
            self.obs = self.inputs["obs"]
        if self.context is None and "context" in self.inputs:
            self.context = self.inputs["context"]
        if (
            self.actions is None
            and "actions" in self.inputs
            and isinstance(self.inputs["actions"], Tensor)
        ):
            self.actions = self.inputs["actions"]

        if self.target is None and "target" in self.targets:
            self.target = self.targets["target"]
        if self.next_obs is None and "next_obs" in self.targets:
            self.next_obs = self.targets["next_obs"]
        if (
            self.rewards is None
            and "rewards" in self.targets
            and isinstance(self.targets["rewards"], Tensor)
        ):
            self.rewards = self.targets["rewards"]
        if (
            self.terminations is None
            and "terminations" in self.targets
            and isinstance(self.targets["terminations"], Tensor)
        ):
            self.terminations = self.targets["terminations"]

    def to(self, device: torch.device | str) -> Batch:
        device_obj = torch.device(device) if isinstance(device, str) else device
        return Batch(
            obs=_map_tensors(self.obs, lambda t: t.to(device_obj))
            if self.obs is not None
            else None,
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
            inputs=_map_tensors(self.inputs, lambda t: t.to(device_obj)),
            targets=_map_tensors(self.targets, lambda t: t.to(device_obj)),
            conditions=_map_tensors(self.conditions, lambda t: t.to(device_obj)),
            extras=_map_tensors(self.extras, lambda t: t.to(device_obj)),
            layouts=dict(self.layouts),
            strict_layout=self.strict_layout,
        )

    def detach(self) -> Batch:
        return Batch(
            obs=_map_tensors(self.obs, lambda t: t.detach()) if self.obs is not None else None,
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
            inputs=_map_tensors(self.inputs, lambda t: t.detach()),
            targets=_map_tensors(self.targets, lambda t: t.detach()),
            conditions=_map_tensors(self.conditions, lambda t: t.detach()),
            extras=_map_tensors(self.extras, lambda t: t.detach()),
            layouts=dict(self.layouts),
            strict_layout=self.strict_layout,
        )

    def clone(self) -> Batch:
        return Batch(
            obs=_map_tensors(self.obs, lambda t: t.clone()) if self.obs is not None else None,
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
            inputs=_map_tensors(self.inputs, lambda t: t.clone()),
            targets=_map_tensors(self.targets, lambda t: t.clone()),
            conditions=_map_tensors(self.conditions, lambda t: t.clone()),
            extras=_map_tensors(self.extras, lambda t: t.clone()),
            layouts=dict(self.layouts),
            strict_layout=self.strict_layout,
        )

    @staticmethod
    def _first_tensor(value: Any) -> Tensor | None:
        for tensor in _iter_tensors(value):
            return tensor
        return None

    @property
    def batch_size(self) -> int:
        obs_source: Any = self.obs if self.obs is not None else self.inputs.get("obs")
        if obs_source is None:
            obs_source = self.inputs
        tensor = self._first_tensor(obs_source)
        if tensor is None:
            raise ValueError("Cannot determine batch size from empty observation dict")
        return int(tensor.shape[0])

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
            "inputs": self.inputs,
            "targets": self.targets,
            "conditions": self.conditions,
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
            inputs=dict(self.inputs),
            targets=dict(self.targets),
            conditions=dict(self.conditions),
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
        dynamic_roots = (
            set(self.inputs.keys()) | set(self.targets.keys()) | set(self.conditions.keys())
        )
        valid_root_fields = {
            "obs",
            "actions",
            "next_obs",
            "rewards",
            "terminations",
            "mask",
            "context",
            "target",
            "inputs",
            "targets",
            "conditions",
            "extras",
        } | dynamic_roots

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
        """Validate batch shapes and consistency."""
        if self.strict_layout:
            self._validate_layout_keys()

        obs_source: Any = self.obs if self.obs is not None else self.inputs.get("obs")
        if obs_source is None:
            obs_source = self.inputs

        obs_entries = list(self._iter_named(obs_source))
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

        # Validate new generic groups as well.
        if self.inputs:
            _check("inputs", self.inputs)
        if self.targets:
            _check("targets", self.targets)


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
