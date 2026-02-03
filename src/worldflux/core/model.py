"""Abstract base class for world models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor

from .batch import Batch
from .output import LossOutput, ModelOutput
from .spec import Capability
from .state import State
from .trajectory import Trajectory


class WorldModel(nn.Module, ABC):
    """Base class for all world models."""

    capabilities: set[Capability]

    def __init__(self) -> None:
        super().__init__()
        self.capabilities = set()

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
