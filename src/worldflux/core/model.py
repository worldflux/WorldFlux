"""Abstract base class for world models."""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor

from .batch import Batch
from .exceptions import CapabilityError, ValidationError
from .interfaces import (
    ActionConditioner,
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
    validate_action_payload_against_spec,
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
    """Base class for all world models."""

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
    def _resolve_batch_key(cls, batch: Batch, key: str):
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
        allowed = set(contract.condition_spec.allowed_extra_keys)
        api_version = self._get_api_version()
        strict = api_version == "v3"
        try:
            conditions.validate(
                strict=strict,
                allowed_extra_keys=allowed if (strict or allowed) else None,
                api_version=api_version,
            )
        except ValueError as e:
            raise ValidationError(str(e)) from e

    def action_tensor_or_none(
        self,
        action: ActionPayload | Tensor | None,
        *,
        validate_contract: bool = True,
    ) -> Tensor | None:
        if validate_contract:
            action_spec = self.io_contract().action_spec
            payload = self.coerce_action_payload(action, kind=action_spec.kind)
            if payload is not None:
                try:
                    validate_action_payload_against_spec(
                        payload,
                        action_spec,
                        api_version=self._get_api_version(),
                    )
                except ValueError as e:
                    raise ValidationError(str(e)) from e
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
        """Encode observation to latent state."""
        if self.observation_encoder is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.encode(...) is not implemented and no observation_encoder is attached"
            )
        world_input = self._coerce_world_input(obs)
        return self.observation_encoder.encode(world_input.observations)

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        """Predict next state (prior/imagination)."""
        if self.dynamics_model is None:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transition(...) is not implemented and no dynamics_model is attached"
            )

        action_spec = self.io_contract().action_spec
        action_payload = self.coerce_action_payload(action, kind=action_spec.kind)
        if action_payload is not None:
            try:
                validate_action_payload_against_spec(
                    action_payload,
                    action_spec,
                    api_version=self._get_api_version(),
                )
            except ValueError as e:
                raise ValidationError(str(e)) from e
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
        """Decode latent state to predictions."""
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
        """Default rollout implementation using transition + decode."""
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
                action_sequence.tensor
                if isinstance(action_sequence, ActionSequence)
                else action_sequence,
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

    @abstractmethod
    def loss(self, batch: Batch) -> LossOutput:
        """Compute training loss."""
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

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> WorldModel:
        from .registry import WorldModelRegistry

        model = WorldModelRegistry.from_pretrained(name_or_path, **kwargs)
        if not isinstance(model, cls):
            raise TypeError(
                f"Expected {cls.__name__}, got {type(model).__name__}. "
                f"Check that the model type in the config matches."
            )
        return model
