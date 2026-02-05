"""Abstract base class for world models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from .batch import Batch
from .exceptions import CapabilityError
from .output import LossOutput, ModelOutput
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

    @abstractmethod
    def encode(self, obs: Tensor | dict[str, Tensor], deterministic: bool = False) -> State:
        """Encode observation to latent state."""
        ...

    @abstractmethod
    def transition(self, state: State, action: Tensor, deterministic: bool = False) -> State:
        """Predict next state (prior/imagination)."""
        ...

    @abstractmethod
    def update(self, state: State, action: Tensor, obs: Tensor | dict[str, Tensor]) -> State:
        """Update state with observation (posterior)."""
        ...

    @abstractmethod
    def decode(self, state: State) -> ModelOutput:
        """Decode latent state to predictions."""
        ...

    def plan_step(self, state: State, action: Tensor, deterministic: bool = False) -> State:
        """Optional planner hook. Default delegates to transition()."""
        return self.transition(state, action, deterministic=deterministic)

    def sample_step(
        self,
        state: State,
        action: Tensor | None = None,
        deterministic: bool = False,
    ) -> ModelOutput:
        """
        Optional sampler hook for generative families.

        If an action is provided, transition first then decode. Otherwise decode state.
        """
        if action is None:
            return self.decode(state)
        next_state = self.transition(state, action, deterministic=deterministic)
        return self.decode(next_state)

    def teacher_forcing_step(
        self,
        state: State,
        action: Tensor,
        obs: Tensor | dict[str, Tensor],
    ) -> State:
        """Optional training hook. Default delegates to update()."""
        return self.update(state, action, obs)

    def rollout(
        self, initial_state: State, actions: Tensor, deterministic: bool = False
    ) -> Trajectory:
        """Default rollout implementation using transition + decode."""
        horizon = actions.shape[0]
        states = [initial_state]
        rewards = []
        continues = []

        state = initial_state
        for t in range(horizon):
            state = self.transition(state, actions[t], deterministic=deterministic)
            states.append(state)
            decoded = self.decode(state)
            if "reward" in decoded.preds:
                rewards.append(decoded.preds["reward"])
            if "continue" in decoded.preds:
                continues.append(decoded.preds["continue"])

        rewards_t = torch.stack(rewards, dim=0).squeeze(-1) if rewards else None
        continues_t = torch.stack(continues, dim=0).squeeze(-1) if continues else None

        return Trajectory(
            states=states,
            actions=actions,
            rewards=rewards_t,
            continues=continues_t,
        )

    @abstractmethod
    def loss(self, batch: Batch) -> LossOutput:
        """Compute training loss."""
        ...

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
