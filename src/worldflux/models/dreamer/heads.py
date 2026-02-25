"""Prediction heads for DreamerV3."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def symlog(x: Tensor) -> Tensor:
    """Symlog transform: sign(x) * ln(|x| + 1)"""
    return torch.sign(x) * torch.log(torch.abs(x) + 1)


def symexp(x: Tensor) -> Tensor:
    """Inverse symlog: sign(x) * (exp(|x|) - 1) with overflow protection.

    The exponential function overflows for inputs > ~88 (float32).
    We clamp the absolute value to prevent NaN propagation.
    """
    x_clipped = torch.clamp(torch.abs(x), max=88.0)
    return torch.sign(x) * (torch.exp(x_clipped) - 1)


def twohot_encode(x: Tensor, bins: Tensor) -> Tensor:
    """Encode scalar values as twohot distributions over bins.

    Args:
        x: Scalar values, shape ``(*batch,)``. Must already be in symlog space.
        bins: Bin centers, shape ``(num_bins,)``. Sorted ascending.

    Returns:
        Twohot soft targets, shape ``(*batch, num_bins)``.
    """
    x = x.clamp(bins[0], bins[-1])
    below = torch.searchsorted(bins, x.contiguous(), right=False) - 1
    below = below.clamp(0, len(bins) - 2)
    above = below + 1
    weight_above = (x - bins[below]) / (bins[above] - bins[below] + 1e-8)
    weight_below = 1.0 - weight_above
    target = torch.zeros(*x.shape, len(bins), device=x.device, dtype=x.dtype)
    target.scatter_(-1, below.unsqueeze(-1), weight_below.unsqueeze(-1))
    target.scatter_(-1, above.unsqueeze(-1), weight_above.unsqueeze(-1))
    return target


def twohot_expected_value(logits: Tensor, bins: Tensor) -> Tensor:
    """Compute expected value from categorical logits over bins.

    Args:
        logits: Raw logits, shape ``(*batch, num_bins)``.
        bins: Bin centers, shape ``(num_bins,)``.

    Returns:
        Expected value in symlog space, shape ``(*batch,)``.
    """
    return (torch.softmax(logits, dim=-1) * bins).sum(dim=-1)


class RewardHead(nn.Module):
    """Reward prediction head.

    When ``use_twohot=True`` (DreamerV3 default), predicts a categorical
    distribution over symlog-spaced bins.  Otherwise, predicts a scalar.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        use_symlog: bool = True,
        use_twohot: bool = False,
        num_bins: int = 255,
        bin_min: float = -20.0,
        bin_max: float = 20.0,
    ):
        super().__init__()
        self.use_symlog = use_symlog
        self.use_twohot = use_twohot
        self.num_bins = num_bins
        self.bins: Tensor

        output_dim = num_bins if use_twohot else 1

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        if use_twohot:
            self.register_buffer("bins", torch.linspace(bin_min, bin_max, num_bins))

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features)

    def predict(self, features: Tensor) -> Tensor:
        """Get scalar reward prediction."""
        out = self.mlp(features)
        if self.use_twohot:
            symlog_value = twohot_expected_value(out, self.bins)
            return symexp(symlog_value) if self.use_symlog else symlog_value
        out = out.squeeze(-1)
        if self.use_symlog:
            out = symexp(out)
        return out


class ContinueHead(nn.Module):
    """Continue probability prediction head."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        return self.mlp(features)

    def predict(self, features: Tensor) -> Tensor:
        """Get continue probability."""
        return torch.sigmoid(self.mlp(features).squeeze(-1))


class DiscreteActorHead(nn.Module):
    """Discrete action policy head using straight-through gradients."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        entropy_coef: float = 3e-4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.entropy_coef = entropy_coef
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return action logits, shape ``(B, action_dim)``."""
        return self.mlp(features)

    def sample(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action with straight-through gradient.

        Returns:
            (action_onehot, log_prob) both shape ``(B, action_dim)`` / ``(B,)``.
        """
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()
        action_onehot = torch.nn.functional.one_hot(idx, self.action_dim).float()
        # Straight-through: forward uses one-hot, backward uses probs
        action_onehot = action_onehot + probs - probs.detach()
        log_prob = dist.log_prob(idx)
        return action_onehot, log_prob

    def entropy(self, features: Tensor) -> Tensor:
        """Per-sample entropy, shape ``(B,)``."""
        logits = self.forward(features)
        probs = torch.softmax(logits, dim=-1)
        return torch.distributions.Categorical(probs=probs).entropy()


class ContinuousActorHead(nn.Module):
    """Continuous action policy head using tanh-squashed normal."""

    def __init__(
        self,
        feature_dim: int,
        action_dim: int,
        hidden_dim: int = 512,
        min_std: float = 0.1,
        entropy_coef: float = 1e-4,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.min_std = min_std
        self.entropy_coef = entropy_coef
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim * 2),
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(mean, std)`` each shape ``(B, action_dim)``."""
        out = self.mlp(features)
        mean, raw_std = out.chunk(2, dim=-1)
        std = torch.nn.functional.softplus(raw_std) + self.min_std
        return mean, std

    def sample(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Sample action via tanh-normal with reparameterised gradient.

        Returns:
            (action, log_prob) shapes ``(B, action_dim)`` / ``(B,)``.
        """
        mean, std = self.forward(features)
        dist = torch.distributions.Normal(mean, std)
        raw = dist.rsample()
        action = torch.tanh(raw)
        # Log-prob with tanh correction
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1)
        return action, log_prob

    def entropy(self, features: Tensor) -> Tensor:
        """Approximate entropy, shape ``(B,)``."""
        _, std = self.forward(features)
        # Gaussian entropy (approximate, ignoring tanh squashing)
        return 0.5 * torch.log(2 * torch.pi * torch.e * std.pow(2)).sum(dim=-1)


class CriticHead(nn.Module):
    """Value function head using twohot categorical (same as RewardHead)."""

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 512,
        num_bins: int = 255,
        bin_min: float = -20.0,
        bin_max: float = 20.0,
    ):
        super().__init__()
        self.num_bins = num_bins
        self.bins: Tensor
        self.register_buffer("bins", torch.linspace(bin_min, bin_max, num_bins))
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Return logits, shape ``(B, num_bins)``."""
        return self.mlp(features)

    def predict(self, features: Tensor) -> Tensor:
        """Scalar value prediction (symexp'd), shape ``(B,)``."""
        logits = self.forward(features)
        symlog_value = twohot_expected_value(logits, self.bins)
        return symexp(symlog_value)


def compute_td_lambda(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    gamma: float = 0.997,
    lambda_: float = 0.95,
) -> Tensor:
    """Compute TD-lambda returns.

    Args:
        rewards: Shape ``(H, N)``.
        values: Shape ``(H+1, N)``.
        continues: Shape ``(H, N)``.

    Returns:
        TD-lambda returns, shape ``(H, N)``.
    """
    horizon = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    # Bootstrap from last value
    last = values[horizon]
    for t in reversed(range(horizon)):
        last = rewards[t] + gamma * continues[t] * ((1 - lambda_) * values[t + 1] + lambda_ * last)
        returns[t] = last
    return returns
