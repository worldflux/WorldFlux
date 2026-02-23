"""Abstract base class for world models."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import tempfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from .batch import Batch
from .exceptions import CapabilityError, ValidationError
from .interfaces import (
    ActionConditioner,
    AsyncDecoder,
    AsyncDynamicsModel,
    AsyncObservationEncoder,
    AsyncRolloutExecutor,
    Decoder,
    DynamicsModel,
    ObservationEncoder,
    RolloutEngine,
    RolloutExecutor,
)
from .output import LossOutput, ModelOutput
from .payloads import (
    ActionKind,
    ActionPayload,
    ActionSequence,
    ConditionPayload,
    WorldModelInput,
    normalize_planned_action,
    validate_action_payload_against_union,
)
from .spec import (
    ActionSpec,
    Capability,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from .state import State
from .trajectory import Trajectory


class WorldModel(nn.Module, ABC):
    """Abstract base class for all world models in the WorldFlux framework.

    ``WorldModel`` defines the canonical interface that every world model must
    implement.  It inherits from :class:`torch.nn.Module` and exposes a
    *composable component architecture* where each stage of the
    observe-predict-decode pipeline can be overridden independently:

    1. **observation_encoder** -- encodes raw observations into a latent
       :class:`~worldflux.core.state.State`.
    2. **action_conditioner** -- fuses action and condition information into
       the dynamics input representation.
    3. **dynamics_model** -- predicts the next latent state given the current
       state and conditioned inputs.
    4. **decoder_module** -- maps a latent state back to observable
       predictions (reconstructed observations, rewards, continuation flags).
    5. **rollout_executor** -- executes multi-step open-loop rollouts by
       chaining ``transition`` and ``decode``.

    Subclasses must implement :meth:`loss` (training objective).  The default
    implementations of :meth:`encode`, :meth:`transition`, :meth:`decode`, and
    :meth:`rollout` delegate to the pluggable components listed above; they
    raise :class:`NotImplementedError` when the corresponding component is
    ``None``.

    Attributes:
        capabilities: Set of :class:`~worldflux.core.spec.Capability` flags
            advertised by this model (e.g. ``REWARD_PRED``, ``PLANNING``).
        observation_encoder: Pluggable encoder component.
        action_conditioner: Pluggable action/condition fusion component.
        dynamics_model: Pluggable latent dynamics component.
        decoder_module: Pluggable decoder component.
        rollout_executor: Pluggable rollout executor component.
        composable_support: Set of component slot names that are effective
            in runtime execution paths for this model.

    Example::

        from worldflux import create_world_model

        model = create_world_model("dreamerv3:size12m", obs_shape=(3, 64, 64), action_dim=6)
        state = model.encode(obs)
        next_state = model.transition(state, action)
        output = model.decode(next_state)

    Note:
        When subclassing, implement :meth:`loss` at a minimum.  Override
        :meth:`encode`, :meth:`transition`, and :meth:`decode` only when the
        default component-delegation behaviour is insufficient.  Attach
        concrete component instances in ``__init__`` and declare supported
        :class:`~worldflux.core.spec.Capability` flags in
        ``self.capabilities``.
    """

    capabilities: set[Capability]

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = set()

        # Universal pluggable components (v0.2).
        self.observation_encoder: ObservationEncoder | None = None
        self.action_conditioner: ActionConditioner | None = None
        self.dynamics_model: DynamicsModel | None = None
        self.decoder_module: Decoder | None = None
        self.rollout_executor: RolloutExecutor | None = None
        # Deprecated compatibility alias retained in v0.2.
        self.rollout_engine: RolloutEngine | None = None
        # Slots where component overrides are effective in runtime execution paths.
        self.composable_support: set[str] = set()

    def _get_api_version(self) -> str:
        return str(getattr(self, "_wf_api_version", "v0.2"))

    def supports(self, capability: Capability) -> bool:
        """Return True if the model advertises a capability."""
        return capability in self.capabilities

    def require(self, capability: Capability, message: str | None = None) -> None:
        """Raise if the model does not support a capability."""
        if capability not in self.capabilities:
            raise CapabilityError(message or f"Model lacks capability: {capability.value}")

    @property
    def supports_reward(self) -> bool:
        return Capability.REWARD_PRED in self.capabilities

    @property
    def supports_continue(self) -> bool:
        return Capability.CONTINUE_PRED in self.capabilities

    @property
    def supports_planning(self) -> bool:
        return Capability.PLANNING in self.capabilities

    @staticmethod
    def _split_batch_key(key: str) -> list[str]:
        normalized = key.replace(":", ".")
        return [part for part in normalized.split(".") if part]

    @classmethod
    def _resolve_batch_key(cls, batch: Batch, key: str) -> Any:
        parts = cls._split_batch_key(key)
        if not parts:
            return None

        root = parts[0]

        # Primary lookup across legacy and v0.2 generic containers.
        if hasattr(batch, root):
            current = getattr(batch, root)
        elif root == "inputs":
            current = batch.inputs
        elif root == "targets":
            current = batch.targets
        elif root == "conditions":
            current = batch.conditions
        elif root == "extras":
            current = batch.extras
        elif root in batch.inputs:
            current = batch.inputs.get(root)
        elif root in batch.targets:
            current = batch.targets.get(root)
        elif root in batch.conditions:
            current = batch.conditions.get(root)
        else:
            current = batch.extras.get(root)

        for part in parts[1:]:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
        return current

    @classmethod
    def has_batch_key(cls, batch: Batch, key: str) -> bool:
        return cls._resolve_batch_key(batch, key) is not None

    @staticmethod
    def coerce_action_payload(
        action: ActionPayload | Tensor | None,
        *,
        kind: str = "continuous",
    ) -> ActionPayload | None:
        if action is None:
            return None
        if isinstance(action, ActionPayload):
            return action
        if isinstance(action, Tensor):
            valid_kinds = {"none", "continuous", "discrete", "token", "latent", "text", "hybrid"}
            normalized_kind: ActionKind = (
                cast(ActionKind, kind) if kind in valid_kinds else "continuous"
            )
            return ActionPayload(kind=normalized_kind, tensor=action)
        raise TypeError(f"Unsupported action type: {type(action)}")

    @staticmethod
    def coerce_condition_payload(
        conditions: ConditionPayload | dict[str, Tensor] | None,
    ) -> ConditionPayload:
        if conditions is None:
            return ConditionPayload()
        if isinstance(conditions, ConditionPayload):
            return conditions
        if isinstance(conditions, dict):
            return ConditionPayload(extras=dict(conditions))
        raise TypeError(f"Unsupported conditions type: {type(conditions)}")

    def _validate_condition_payload(self, conditions: ConditionPayload) -> None:
        contract = self.io_contract()
        allowed = set(contract.effective_condition_extra_keys)
        extras_schema = contract.condition_extras_schema_dict()
        api_version = self._get_api_version()
        strict = api_version == "v3"
        try:
            conditions.validate(
                strict=strict,
                allowed_extra_keys=allowed if (strict or allowed) else None,
                extra_schema=extras_schema if extras_schema else None,
                api_version=api_version,
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e

    def _validate_action_payload(self, payload: ActionPayload) -> None:
        contract = self.io_contract()
        try:
            validate_action_payload_against_union(
                payload,
                contract.effective_action_specs,
                api_version=self._get_api_version(),
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e

    def action_tensor_or_none(
        self,
        action: ActionPayload | Tensor | None,
        *,
        validate_contract: bool = True,
    ) -> Tensor | None:
        contract = self.io_contract()
        if validate_contract:
            action_spec = contract.action_spec
            payload = self.coerce_action_payload(action, kind=action_spec.kind)
            if payload is not None:
                self._validate_action_payload(payload)
        else:
            payload = self.coerce_action_payload(action)
        if payload is None:
            return None
        if payload.tensor is not None:
            return payload.tensor
        if payload.tokens is not None:
            return payload.tokens
        if payload.latent is not None:
            return payload.latent
        return None

    @staticmethod
    def _coerce_world_input(obs: Tensor | dict[str, Tensor] | WorldModelInput) -> WorldModelInput:
        if isinstance(obs, WorldModelInput):
            return obs
        if isinstance(obs, dict):
            return WorldModelInput(observations=obs)
        return WorldModelInput(observations={"obs": obs})

    def validate_batch_contract(self, batch: Batch) -> None:
        """Validate batch keys/layouts against model I/O contract."""
        contract = self.io_contract()
        contract.validate()
        missing = [
            key for key in contract.required_batch_keys if not self.has_batch_key(batch, key)
        ]
        if missing:
            raise ValidationError(
                f"Batch is missing required batch keys: {missing}. "
                f"Required: {list(contract.required_batch_keys)}"
            )

        if batch.strict_layout and batch.conditions:
            declared = set(contract.condition_spec.modalities.keys()) | set(
                contract.condition_spec.allowed_extra_keys
            )
            unknown = [k for k in batch.conditions.keys() if k not in declared]
            if unknown:
                raise ValidationError(
                    "Batch.conditions contains undeclared keys under strict layout mode: "
                    f"{sorted(unknown)}"
                )

    def validate_state_contract(self, state: State) -> None:
        """Validate state tensor keys/shapes against model I/O contract."""
        contract = self.io_contract()
        contract.validate()
        missing = [k for k in contract.required_state_keys if k not in state.tensors]
        if missing:
            raise ValidationError(
                f"State is missing required state tensors: {missing}. "
                f"Required: {list(contract.required_state_keys)}"
            )
        for key, spec in contract.state_spec.tensors.items():
            tensor = state.tensors.get(key)
            if tensor is None:
                continue
            if spec.shape and tensor.shape[1:] != spec.shape:
                raise ValidationError(
                    f"State tensor '{key}' has shape {tuple(tensor.shape[1:])}, "
                    f"expected {spec.shape}"
                )

    def io_contract(self) -> ModelIOContract:
        """
        Return runtime I/O contract.

        Subclasses should override this when they have richer modality/state specs.
        The default keeps backward compatibility for existing models.
        """
        config = getattr(self, "config", None)
        obs_shape = tuple(getattr(config, "obs_shape", ()))
        action_dim = int(getattr(config, "action_dim", 0))
        action_type = str(getattr(config, "action_type", "continuous"))

        obs_kind = ModalityKind.IMAGE if len(obs_shape) == 3 else ModalityKind.VECTOR
        obs_spec = ObservationSpec(
            modalities={"obs": ModalitySpec(kind=obs_kind, shape=obs_shape or (1,))}
        )
        action_spec = ActionSpec(
            kind=action_type,
            dim=action_dim,
            discrete=action_type == "discrete",
            num_actions=action_dim if action_type == "discrete" else None,
        )

        state_spec = StateSpec(tensors={})
        prediction_tensors: dict[str, ModalitySpec] = {}
        if Capability.OBS_DECODER in self.capabilities:
            prediction_tensors["obs"] = ModalitySpec(kind=obs_kind, shape=obs_shape or (1,))
        if Capability.REWARD_PRED in self.capabilities:
            prediction_tensors["reward"] = ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,))
        if Capability.CONTINUE_PRED in self.capabilities:
            prediction_tensors["continue"] = ModalitySpec(kind=ModalityKind.VECTOR, shape=(1,))

        return ModelIOContract(
            observation_spec=obs_spec,
            input_spec=obs_spec,
            target_spec=ObservationSpec(
                modalities={"next_obs": ModalitySpec(kind=obs_kind, shape=obs_shape or (1,))}
            ),
            action_spec=action_spec,
            state_spec=state_spec,
            prediction_spec=PredictionSpec(tensors=prediction_tensors),
            sequence_layout=SequenceLayout(
                axes_by_field={
                    "obs": "BT...",
                    "actions": "BT...",
                    "rewards": "BT",
                    "terminations": "BT",
                    "next_obs": "BT...",
                    "mask": "BT...",
                }
            ),
            required_batch_keys=("obs",),
            required_state_keys=(),
        )

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        """Encode observations into a latent :class:`~worldflux.core.state.State`.

        Delegates to the attached ``observation_encoder`` component.  Input
        observations are first coerced to :class:`WorldModelInput` and
        validated against the model's I/O contract.

        Args:
            obs: Raw observation tensor, a dict of named modality tensors,
                or a :class:`WorldModelInput` wrapping both.
            deterministic: If ``True``, use deterministic encoding (e.g.
                posterior mean rather than a sample).

        Returns:
            A :class:`~worldflux.core.state.State` containing the latent
            representation.

        Raises:
            NotImplementedError: If no ``observation_encoder`` is attached.
            ValidationError: If required input modalities are missing.
        """
        if self.observation_encoder is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.encode(...) is not implemented and no observation_encoder is attached"
            )
        world_input = self._coerce_world_input(obs)
        self._validate_input_modalities(world_input.observations)
        return self.observation_encoder.encode(world_input.observations)

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        """Predict the next latent state given current state and action.

        Performs a single imagination step through the dynamics model.  The
        action is coerced to :class:`ActionPayload` and validated against the
        I/O contract.  If an ``action_conditioner`` is attached, it fuses the
        action and condition information before passing them to the dynamics
        model.

        Args:
            state: Current latent state.
            action: Action to condition on.  Accepts a raw tensor, an
                :class:`ActionPayload`, or ``None`` for unconditional
                transition.
            conditions: Optional auxiliary condition signals
                (e.g. goal embeddings, context vectors).
            deterministic: If ``True``, use deterministic dynamics (e.g.
                mean prediction rather than a sample).

        Returns:
            The predicted next :class:`~worldflux.core.state.State`.

        Raises:
            NotImplementedError: If no ``dynamics_model`` is attached.
            ValidationError: If action or conditions violate the I/O contract.
        """
        if self.dynamics_model is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transition(...) is not implemented and no dynamics_model is attached"
            )

        contract = self.io_contract()
        action_spec = contract.action_spec
        action_payload = self.coerce_action_payload(action, kind=action_spec.kind)
        if action_payload is not None:
            self._validate_action_payload(action_payload)
        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        conditioned: dict[str, Tensor] = {}

        if self.action_conditioner is not None:
            conditioned.update(
                self.action_conditioner.condition(state, action_payload, condition_payload)
            )

        action_tensor = self.action_tensor_or_none(action_payload, validate_contract=False)
        if action_tensor is not None and "action" not in conditioned:
            conditioned["action"] = action_tensor

        return self.dynamics_model.transition(state, conditioned, deterministic=deterministic)

    @staticmethod
    async def _run_component_async(
        component: object,
        async_name: str,
        sync_name: str,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        async_fn = getattr(component, async_name, None)
        if callable(async_fn):
            maybe_awaitable = async_fn(*args, **kwargs)
            if inspect.isawaitable(maybe_awaitable):
                return await maybe_awaitable
            return maybe_awaitable

        sync_fn = getattr(component, sync_name, None)
        if callable(sync_fn):
            return await asyncio.to_thread(sync_fn, *args, **kwargs)
        raise NotImplementedError(
            f"Component {type(component).__name__} must implement {async_name} or {sync_name}"
        )

    def _validate_input_modalities(self, observations: dict[str, Tensor]) -> None:
        contract = self.io_contract()
        required = tuple(contract.effective_input_spec.modalities.keys())
        missing = [key for key in required if key not in observations]
        if missing:
            raise ValidationError(
                f"Input observations are missing required modalities: {missing}. "
                f"Declared modalities: {list(required)}"
            )

    async def async_encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        """Asynchronous non-blocking variant of ``encode``."""
        del deterministic
        if self.observation_encoder is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.async_encode(...) requires observation_encoder"
            )
        world_input = self._coerce_world_input(obs)
        self._validate_input_modalities(world_input.observations)
        encoder = self.observation_encoder
        if isinstance(encoder, AsyncObservationEncoder) or hasattr(encoder, "encode_async"):
            return await self._run_component_async(
                encoder, "encode_async", "encode", world_input.observations
            )
        return await asyncio.to_thread(encoder.encode, world_input.observations)

    async def async_transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        """Asynchronous non-blocking variant of ``transition``."""
        if self.dynamics_model is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.async_transition(...) requires dynamics_model"
            )

        contract = self.io_contract()
        action_spec = contract.action_spec
        action_payload = self.coerce_action_payload(action, kind=action_spec.kind)
        if action_payload is not None:
            self._validate_action_payload(action_payload)
        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        conditioned: dict[str, Tensor] = {}

        if self.action_conditioner is not None:
            conditioned.update(
                self.action_conditioner.condition(state, action_payload, condition_payload)
            )

        action_tensor = self.action_tensor_or_none(action_payload, validate_contract=False)
        if action_tensor is not None and "action" not in conditioned:
            conditioned["action"] = action_tensor

        dynamics_model = self.dynamics_model
        if isinstance(dynamics_model, AsyncDynamicsModel) or hasattr(
            dynamics_model, "transition_async"
        ):
            return await self._run_component_async(
                dynamics_model,
                "transition_async",
                "transition",
                state,
                conditioned,
                deterministic=deterministic,
            )
        return await asyncio.to_thread(dynamics_model.transition, state, conditioned, deterministic)

    async def async_decode(
        self,
        state: State,
        conditions: ConditionPayload | None = None,
    ) -> ModelOutput:
        """Asynchronous non-blocking variant of ``decode``."""
        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        if self.decoder_module is None:
            raise CapabilityError(f"{self.__class__.__name__} does not expose a decoder component")
        decoder_module = self.decoder_module
        if isinstance(decoder_module, AsyncDecoder) or hasattr(decoder_module, "decode_async"):
            preds = await self._run_component_async(
                decoder_module,
                "decode_async",
                "decode",
                state,
                conditions=condition_payload,
            )
        else:
            preds = await asyncio.to_thread(decoder_module.decode, state, condition_payload)
        return ModelOutput(predictions=preds, state=state)

    def update(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        conditions: ConditionPayload | None = None,
    ) -> State:
        """Update state with observation (posterior)."""
        del state, action, conditions
        warnings.warn(
            "Calling default WorldModel.update(...). This fallback re-encodes observations and "
            "should be overridden by stateful posterior models.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.encode(obs)

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput:
        """Decode a latent state into observable predictions.

        Maps the latent representation back to the observation space and any
        auxiliary prediction heads (reward, continuation flag).  Delegates to
        the attached ``decoder_module`` component.

        Args:
            state: Latent state to decode.
            conditions: Optional auxiliary condition signals.

        Returns:
            A :class:`~worldflux.core.output.ModelOutput` containing the
            ``predictions`` dict and the originating ``state``.

        Raises:
            CapabilityError: If no ``decoder_module`` is attached.
            ValidationError: If conditions violate the I/O contract.
        """
        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)
        if self.decoder_module is None:
            raise CapabilityError(f"{self.__class__.__name__} does not expose a decoder component")
        preds = self.decoder_module.decode(state, conditions=condition_payload)
        return ModelOutput(predictions=preds, state=state)

    def plan_step(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        """Optional planner hook. Default delegates to transition()."""
        try:
            return self.transition(
                state,
                action,
                conditions=conditions,
                deterministic=deterministic,
            )
        except TypeError:
            # Backward compatibility for subclasses overriding transition(state, action, deterministic)
            return self.transition(state, action, deterministic=deterministic)  # type: ignore[misc]

    def sample_step(
        self,
        state: State,
        action: ActionPayload | Tensor | None = None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> ModelOutput:
        """
        Optional sampler hook for generative families.

        If an action is provided, transition first then decode. Otherwise decode state.
        """
        if action is None:
            try:
                return self.decode(state, conditions=conditions)
            except TypeError:
                # Backward compatibility for subclasses overriding decode(state).
                return self.decode(state)  # type: ignore[misc]
        next_state = self.transition(
            state, action, conditions=conditions, deterministic=deterministic
        )
        try:
            return self.decode(next_state, conditions=conditions)
        except TypeError:
            # Backward compatibility for subclasses overriding decode(state).
            return self.decode(next_state)  # type: ignore[misc]

    def teacher_forcing_step(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        conditions: ConditionPayload | None = None,
    ) -> State:
        """Optional training hook. Default delegates to update()."""
        return self.update(state, action, obs, conditions=conditions)

    @staticmethod
    def _action_from_sequence(
        action_sequence: ActionSequence | Tensor | None,
        step: int,
    ) -> ActionPayload | Tensor | None:
        if action_sequence is None:
            return None
        if isinstance(action_sequence, Tensor):
            return action_sequence[step]
        if action_sequence.tensor is not None:
            return action_sequence.tensor[step]
        if action_sequence.payloads is not None:
            return action_sequence.payloads[step]
        return None

    def rollout(
        self,
        initial_state: State,
        action_sequence: ActionSequence | ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
        mode: str = "autoregressive",
    ) -> Trajectory:
        """Execute a multi-step open-loop rollout from an initial state.

        Starting from ``initial_state``, the method iteratively applies
        :meth:`transition` and :meth:`decode` for each action in the
        sequence, collecting states, predicted rewards, and continuation
        flags into a :class:`~worldflux.core.trajectory.Trajectory`.

        If a ``rollout_executor`` component is attached, execution is
        delegated to it; otherwise the default loop is used.

        Args:
            initial_state: Starting latent state for the rollout.
            action_sequence: Sequence of actions to apply.  Accepts a
                :class:`Tensor` of shape ``(horizon, ...)``, an
                :class:`ActionSequence`, a single :class:`ActionPayload`,
                or ``None``.
            conditions: Optional auxiliary condition signals applied at each
                step.
            deterministic: If ``True``, use deterministic transitions.
            mode: Rollout mode.  Only ``"autoregressive"`` is supported in
                v0.2+; other values emit a deprecation warning.

        Returns:
            A :class:`~worldflux.core.trajectory.Trajectory` containing
            the collected states, actions, rewards, and continuation flags.
        """
        api_version = self._get_api_version()
        if mode != "autoregressive":
            msg = (
                "rollout(..., mode=...) is deprecated in v0.2 and will be removed in v0.3. "
                "Use planner strategies for re-planning/tree-search."
            )
            if api_version == "v3":
                raise ValueError(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)

        if isinstance(action_sequence, ActionPayload):
            action_sequence = normalize_planned_action(action_sequence, api_version=api_version)

        if self.rollout_executor is not None:
            return self.rollout_executor.rollout_open_loop(
                self,
                initial_state,
                action_sequence,
                conditions=condition_payload,
                deterministic=deterministic,
            )
        if self.rollout_engine is not None:
            warnings.warn(
                "rollout_engine is deprecated in v0.2; attach rollout_executor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return self.rollout_engine.rollout(
                self,
                initial_state,
                action_sequence.tensor
                if isinstance(action_sequence, ActionSequence)
                else action_sequence,
                conditions=condition_payload,
                deterministic=deterministic,
                mode="autoregressive",
            )

        if isinstance(action_sequence, Tensor):
            horizon = int(action_sequence.shape[0])
            actions_tensor: Tensor | None = action_sequence
        elif isinstance(action_sequence, ActionSequence) and action_sequence.tensor is not None:
            horizon = int(action_sequence.tensor.shape[0])
            actions_tensor = action_sequence.tensor
        elif isinstance(action_sequence, ActionSequence) and action_sequence.payloads is not None:
            horizon = len(action_sequence.payloads)
            actions_tensor = None
        else:
            horizon = 0
            actions_tensor = None

        states = [initial_state]
        rewards = []
        continues = []

        state = initial_state
        for t in range(horizon):
            action_t = self._action_from_sequence(action_sequence, t)
            state = self.transition(
                state, action_t, conditions=condition_payload, deterministic=deterministic
            )
            states.append(state)
            decoded = self.decode(state, conditions=condition_payload)
            if "reward" in decoded.predictions:
                rewards.append(decoded.predictions["reward"])
            if "continue" in decoded.predictions:
                continues.append(decoded.predictions["continue"])

        rewards_t = torch.stack(rewards, dim=0).squeeze(-1) if rewards else None
        continues_t = torch.stack(continues, dim=0).squeeze(-1) if continues else None

        if actions_tensor is None:
            batch = initial_state.batch_size
            actions_tensor = torch.zeros(horizon, batch, 0, device=initial_state.device)

        return Trajectory(
            states=states,
            actions=actions_tensor,
            rewards=rewards_t,
            continues=continues_t,
        )

    async def async_rollout(
        self,
        initial_state: State,
        action_sequence: ActionSequence | ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
        mode: str = "autoregressive",
    ) -> Trajectory:
        """Asynchronous non-blocking variant of ``rollout``."""
        api_version = self._get_api_version()
        if mode != "autoregressive":
            msg = (
                "rollout(..., mode=...) is deprecated in v0.2 and will be removed in v0.3. "
                "Use planner strategies for re-planning/tree-search."
            )
            if api_version == "v3":
                raise ValueError(msg)
            warnings.warn(msg, DeprecationWarning, stacklevel=2)

        condition_payload = self.coerce_condition_payload(conditions)
        self._validate_condition_payload(condition_payload)

        if isinstance(action_sequence, ActionPayload):
            action_sequence = normalize_planned_action(action_sequence, api_version=api_version)

        if self.rollout_executor is not None:
            executor = self.rollout_executor
            if isinstance(executor, AsyncRolloutExecutor) or hasattr(
                executor, "rollout_open_loop_async"
            ):
                return await self._run_component_async(
                    executor,
                    "rollout_open_loop_async",
                    "rollout_open_loop",
                    self,
                    initial_state,
                    action_sequence,
                    conditions=condition_payload,
                    deterministic=deterministic,
                )
            return await asyncio.to_thread(
                executor.rollout_open_loop,
                self,
                initial_state,
                action_sequence,
                condition_payload,
                deterministic,
            )
        if self.rollout_engine is not None:
            warnings.warn(
                "rollout_engine is deprecated in v0.2; attach rollout_executor instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            engine_sequence = (
                action_sequence.tensor
                if isinstance(action_sequence, ActionSequence)
                else action_sequence
            )
            return await asyncio.to_thread(
                self.rollout_engine.rollout,
                self,
                initial_state,
                engine_sequence,
                condition_payload,
                deterministic,
                "autoregressive",
            )

        if isinstance(action_sequence, Tensor):
            horizon = int(action_sequence.shape[0])
            actions_tensor: Tensor | None = action_sequence
        elif isinstance(action_sequence, ActionSequence) and action_sequence.tensor is not None:
            horizon = int(action_sequence.tensor.shape[0])
            actions_tensor = action_sequence.tensor
        elif isinstance(action_sequence, ActionSequence) and action_sequence.payloads is not None:
            horizon = len(action_sequence.payloads)
            actions_tensor = None
        else:
            horizon = 0
            actions_tensor = None

        states = [initial_state]
        rewards = []
        continues = []

        state = initial_state
        for t in range(horizon):
            action_t = self._action_from_sequence(action_sequence, t)
            state = await self.async_transition(
                state, action_t, conditions=condition_payload, deterministic=deterministic
            )
            states.append(state)
            decoded = await self.async_decode(state, conditions=condition_payload)
            if "reward" in decoded.predictions:
                rewards.append(decoded.predictions["reward"])
            if "continue" in decoded.predictions:
                continues.append(decoded.predictions["continue"])

        rewards_t = torch.stack(rewards, dim=0).squeeze(-1) if rewards else None
        continues_t = torch.stack(continues, dim=0).squeeze(-1) if continues else None

        if actions_tensor is None:
            batch = initial_state.batch_size
            actions_tensor = torch.zeros(horizon, batch, 0, device=initial_state.device)

        return Trajectory(
            states=states,
            actions=actions_tensor,
            rewards=rewards_t,
            continues=continues_t,
        )

    @abstractmethod
    def loss(self, batch: Batch) -> LossOutput:
        """Compute the training loss for a single batch.

        This is the only method that subclasses **must** implement.  It
        should run the full forward pass (encode, transition/predict, decode)
        and return a :class:`~worldflux.core.output.LossOutput` containing
        the scalar loss and per-component breakdowns.

        Args:
            batch: A :class:`~worldflux.core.batch.Batch` of training data
                that conforms to the model's I/O contract.

        Returns:
            A :class:`~worldflux.core.output.LossOutput` with ``loss``
            (scalar tensor), ``components`` (dict of named sub-losses),
            and optional ``metrics``.
        """
        ...

    def save_pretrained(self, path: str) -> None:
        """Save model weights and config using a unified directory layout."""
        os.makedirs(path, exist_ok=True)

        config = getattr(self, "config", None)
        if config is None or not hasattr(config, "save"):
            raise ValidationError(
                f"{self.__class__.__name__} cannot be saved because config.save(...) is unavailable."
            )
        config.save(os.path.join(path, "config.json"))
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
        metadata = {
            "save_format_version": 1,
            "worldflux_version": self._resolve_worldflux_version(),
            "api_version": self._get_api_version(),
            "model_type": str(getattr(config, "model_type", self.__class__.__name__)),
            "contract_fingerprint": self.contract_fingerprint(),
            "created_at_utc": datetime.now(timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
        }
        with open(os.path.join(path, "worldflux_meta.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

    def push_to_hub(
        self,
        repo_id: str,
        *,
        token: str | None = None,
        private: bool | None = None,
        commit_message: str | None = None,
    ) -> str:
        """Upload this model to the Hugging Face Hub.

        Requires optional dependency group ``worldflux[hub]``.
        """
        try:
            from huggingface_hub import HfApi
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency guard
            raise ValidationError(
                "Hugging Face Hub support requires optional dependency "
                '`huggingface_hub`. Install with: uv pip install "worldflux[hub]"'
            ) from exc

        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, private=private, repo_type="model", exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="worldflux-hub-") as tmpdir:
            self.save_pretrained(tmpdir)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=tmpdir,
                commit_message=commit_message or f"Upload {self.__class__.__name__} from WorldFlux",
                token=token,
            )

        return f"https://huggingface.co/{repo_id}"

    @staticmethod
    def _normalize_contract_value(value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if is_dataclass(value) and not isinstance(value, type):
            return {k: WorldModel._normalize_contract_value(v) for k, v in asdict(value).items()}
        if isinstance(value, dict):
            return {str(k): WorldModel._normalize_contract_value(v) for k, v in value.items()}
        if isinstance(value, tuple):
            return [WorldModel._normalize_contract_value(v) for v in value]
        if isinstance(value, list):
            return [WorldModel._normalize_contract_value(v) for v in value]
        return value

    @staticmethod
    def _resolve_worldflux_version() -> str:
        try:
            from importlib.metadata import PackageNotFoundError, version

            return version("worldflux")
        except PackageNotFoundError:
            return "0.1.0.dev0"
        except (ImportError, ValueError):
            return "0.1.0.dev0"

    def contract_fingerprint(self) -> str:
        """Return a stable fingerprint for this model's declared IO contract."""
        contract = self._normalize_contract_value(self.io_contract())
        serialized = json.dumps(contract, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> WorldModel:
        from .registry import WorldModelRegistry

        model = WorldModelRegistry.from_pretrained(name_or_path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(model).__name__}. "
                f"Check that the model type in the config matches."
            )
        return model
