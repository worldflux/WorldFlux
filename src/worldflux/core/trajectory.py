"""Trajectory representation for imagination rollouts."""

from dataclasses import dataclass, field

import torch
from torch import Tensor

from .exceptions import StateError
from .state import State


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

    The trajectory maintains the invariant that len(states) == actions.shape[0] + 1,
    representing the initial state plus one state per action taken.
    """

    states: list[State]
    actions: Tensor
    rewards: Tensor | None = None
    values: Tensor | None = None
    continues: Tensor | None = None
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Validate trajectory consistency after initialization."""
        self._validate()
        self._validated = True

    def _validate(self) -> None:
        """
        Validate trajectory consistency.

        Raises:
            StateError: If the trajectory is inconsistent.
        """
        if not self.states:
            raise StateError("Trajectory must have at least one state")

        num_states = len(self.states)
        num_actions = self.actions.shape[0]

        # States should be one more than actions (initial + T steps)
        if num_states != num_actions + 1:
            raise StateError(
                f"Trajectory state/action mismatch: {num_states} states but {num_actions} actions. "
                f"Expected {num_actions + 1} states (initial + one per action)."
            )

        # Validate batch sizes are consistent across states
        batch_size = self.states[0].batch_size
        for i, state in enumerate(self.states[1:], 1):
            if state.batch_size != batch_size:
                raise StateError(
                    f"Inconsistent batch size in trajectory: state[0] has batch_size={batch_size}, "
                    f"state[{i}] has batch_size={state.batch_size}"
                )

        # Validate actions batch size
        if self.actions.shape[1] != batch_size:
            raise StateError(
                f"Actions batch size ({self.actions.shape[1]}) doesn't match "
                f"states batch size ({batch_size})"
            )

        # Validate optional tensors if present
        if self.rewards is not None:
            if self.rewards.shape[0] != num_actions:
                raise StateError(
                    f"Rewards length ({self.rewards.shape[0]}) doesn't match "
                    f"actions length ({num_actions})"
                )

        if self.values is not None:
            if self.values.shape[0] != num_states:
                raise StateError(
                    f"Values length ({self.values.shape[0]}) doesn't match "
                    f"states length ({num_states})"
                )

        if self.continues is not None:
            if self.continues.shape[0] != num_actions:
                raise StateError(
                    f"Continues length ({self.continues.shape[0]}) doesn't match "
                    f"actions length ({num_actions})"
                )

    def __len__(self) -> int:
        return len(self.states)

    @property
    def horizon(self) -> int:
        """Prediction horizon (number of actions)."""
        return self.actions.shape[0]

    @property
    def batch_size(self) -> int:
        return self.states[0].batch_size

    def to_tensor(self, key: str) -> Tensor:
        """Stack a specific state tensor key across time [T+1, batch, ...]."""
        return torch.stack([s.tensors[key] for s in self.states], dim=0)

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
