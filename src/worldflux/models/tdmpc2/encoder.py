"""TD-MPC2 Encoder."""

import torch.nn as nn
from torch import Tensor


class MLPEncoder(nn.Module):
    """MLP encoder for state-based observations.

    Encodes observations into a latent representation using a 3-layer MLP
    with LayerNorm and Mish activations.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ):
        """Initialize MLP encoder.

        Args:
            obs_dim: Observation dimension (flattened if multi-dimensional).
            hidden_dim: Hidden layer dimension.
            latent_dim: Output latent dimension.
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.mlp = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, obs: Tensor) -> Tensor:
        """Encode observation to latent representation.

        Args:
            obs: Observation tensor of shape [batch, obs_dim] or [batch, *obs_shape].

        Returns:
            Latent representation of shape [batch, latent_dim].
        """
        if obs.dim() > 2:
            obs = obs.flatten(start_dim=1)
        return self.mlp(obs)
