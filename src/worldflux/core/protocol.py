"""World model protocol definition."""

from typing import Protocol, runtime_checkable

import torch
from torch import Tensor

from .config import WorldModelConfig
from .state import LatentState
from .trajectory import Trajectory


@runtime_checkable
class WorldModel(Protocol):
    """
    Unified interface for all world models.

    Core operations:
    1. encode: observation -> initial latent state
    2. predict: (state, action) -> next state (prior/imagination)
    3. observe: (state, action, observation) -> next state (posterior)
    4. decode: state -> predictions (obs, reward, continue)

    Compound operations:
    5. imagine: multi-step imagination rollout
    6. initial_state: create initial state for new episode
    7. compute_loss: compute training losses
    """

    config: WorldModelConfig

    def encode(self, obs: Tensor, deterministic: bool = False) -> LatentState:
        """Encode observation to latent state."""
        ...

    def predict(
        self, state: LatentState, action: Tensor, deterministic: bool = False
    ) -> LatentState:
        """Predict next state given action (prior/imagination mode)."""
        ...

    def observe(self, state: LatentState, action: Tensor, obs: Tensor) -> LatentState:
        """Update state with observation (posterior)."""
        ...

    def decode(self, state: LatentState) -> dict[str, Tensor]:
        """Decode latent state to predictions."""
        ...

    def imagine(
        self, initial_state: LatentState, actions: Tensor, deterministic: bool = False
    ) -> Trajectory:
        """Multi-step imagination rollout."""
        ...

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LatentState:
        """Create initial latent state for new episode."""
        ...

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute training losses from batch."""
        ...

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "WorldModel":
        """Load pretrained model or config."""
        ...

    def save_pretrained(self, path: str) -> None:
        """Save model and config."""
        ...
