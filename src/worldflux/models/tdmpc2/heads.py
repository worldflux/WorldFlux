"""TD-MPC2 Prediction Heads."""

import torch
import torch.nn as nn
from torch import Tensor


class RewardHead(nn.Module):
    """Reward prediction head.

    Predicts scalar reward given concatenated latent state and action.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        """Initialize reward head.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict reward.

        Args:
            z: Latent state of shape [batch, latent_dim].
            action: Action of shape [batch, action_dim].

        Returns:
            Predicted reward of shape [batch, 1].
        """
        za = torch.cat([z, action], dim=-1)
        return self.mlp(za)


class QNetwork(nn.Module):
    """Single Q-network.

    Predicts Q-value given concatenated latent state and action.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        """Initialize Q-network.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict Q-value.

        Args:
            z: Latent state of shape [batch, latent_dim].
            action: Action of shape [batch, action_dim].

        Returns:
            Predicted Q-value of shape [batch, 1].
        """
        za = torch.cat([z, action], dim=-1)
        return self.mlp(za)


class QEnsemble(nn.Module):
    """Q-function ensemble.

    Maintains multiple Q-networks for uncertainty estimation and
    conservative Q-learning.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_q_networks: int = 5,
    ):
        """Initialize Q-ensemble.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            num_q_networks: Number of Q-networks in ensemble.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_q_networks = num_q_networks

        self.q_networks = nn.ModuleList(
            [QNetwork(latent_dim, action_dim, hidden_dim) for _ in range(num_q_networks)]
        )

    def forward(self, z: Tensor, action: Tensor) -> Tensor:
        """Predict Q-values from all networks.

        Args:
            z: Latent state of shape [batch, latent_dim].
            action: Action of shape [batch, action_dim].

        Returns:
            Q-values of shape [num_q_networks, batch, 1].
        """
        q_values = torch.stack([q(z, action) for q in self.q_networks], dim=0)
        return q_values


class PolicyHead(nn.Module):
    """Policy head for MPC warm-start.

    Outputs actions bounded to [-1, 1] via tanh activation.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dim: int,
    ):
        """Initialize policy head.

        Args:
            latent_dim: Latent state dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Predict action.

        Args:
            z: Latent state of shape [batch, latent_dim].

        Returns:
            Action of shape [batch, action_dim], bounded to [-1, 1].
        """
        return self.mlp(z)
