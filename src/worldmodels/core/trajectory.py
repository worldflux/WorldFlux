"""Trajectory representation for imagination rollouts."""

from dataclasses import dataclass

import torch
from torch import Tensor

from .state import LatentState


@dataclass
class Trajectory:
    """
    Imagination rollout trajectory in latent space.
    
    Attributes:
        states: List of latent states [T+1] (initial + T steps)
        actions: Action tensor [T, batch, action_dim]
        rewards: Predicted rewards [T, batch] (optional)
        values: Predicted values [T+1, batch] (optional)
        continues: Continue probabilities [T, batch] (optional)
    """
    states: list[LatentState]
    actions: Tensor
    rewards: Tensor | None = None
    values: Tensor | None = None
    continues: Tensor | None = None

    def __len__(self) -> int:
        return len(self.states)

    @property
    def horizon(self) -> int:
        """Prediction horizon (number of actions)."""
        return self.actions.shape[0]

    @property
    def batch_size(self) -> int:
        return self.states[0].batch_size

    def to_features_tensor(self) -> Tensor:
        """Stack all state features [T+1, batch, feature_dim]."""
        return torch.stack([s.features for s in self.states], dim=0)

    def to(self, device: torch.device) -> "Trajectory":
        return Trajectory(
            states=[s.to(device) for s in self.states],
            actions=self.actions.to(device),
            rewards=self.rewards.to(device) if self.rewards is not None else None,
            values=self.values.to(device) if self.values is not None else None,
            continues=self.continues.to(device) if self.continues is not None else None,
        )

    def detach(self) -> "Trajectory":
        return Trajectory(
            states=[s.detach() for s in self.states],
            actions=self.actions.detach(),
            rewards=self.rewards.detach() if self.rewards is not None else None,
            values=self.values.detach() if self.values is not None else None,
            continues=self.continues.detach() if self.continues is not None else None,
        )
