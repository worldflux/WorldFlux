"""Latent space implementations for different world model architectures."""

from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LatentSpace(ABC, nn.Module):
    """Base class for latent space sampling and KL computation."""

    @abstractmethod
    def sample(self, params: Tensor, deterministic: bool = False) -> Tensor:
        """Sample from distribution given parameters."""
        ...

    @abstractmethod
    def kl_divergence(self, posterior_params: Tensor, prior_params: Tensor) -> Tensor:
        """Compute KL(posterior || prior)."""
        ...

    @property
    @abstractmethod
    def param_dim(self) -> int:
        """Required parameter dimension."""
        ...


class GaussianLatentSpace(LatentSpace):
    """Gaussian latent space with reparameterization trick."""

    def __init__(self, dim: int, min_std: float = 0.1, max_std: float = 2.0):
        super().__init__()
        self.dim = dim
        self.min_std = min_std
        self.max_std = max_std

    @property
    def param_dim(self) -> int:
        return self.dim * 2

    def sample(self, params: Tensor, deterministic: bool = False) -> Tensor:
        mean, raw_std = params.chunk(2, dim=-1)
        std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(raw_std)

        if deterministic:
            return mean

        eps = torch.randn_like(std)
        return mean + std * eps

    def kl_divergence(self, posterior_params: Tensor, prior_params: Tensor) -> Tensor:
        p_mean, p_raw_std = posterior_params.chunk(2, dim=-1)
        q_mean, q_raw_std = prior_params.chunk(2, dim=-1)

        p_std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(p_raw_std)
        q_std = self.min_std + (self.max_std - self.min_std) * torch.sigmoid(q_raw_std)

        p_var, q_var = p_std ** 2, q_std ** 2

        kl = 0.5 * (
            torch.log(q_var / p_var)
            + (p_var + (p_mean - q_mean) ** 2) / q_var
            - 1
        )
        return kl.sum(dim=-1)


class CategoricalLatentSpace(LatentSpace):
    """Categorical latent space with straight-through estimator."""

    def __init__(
        self,
        num_categoricals: int = 32,
        num_classes: int = 32,
        temperature: float = 1.0,
        straight_through: bool = True
    ):
        super().__init__()
        self.num_categoricals = num_categoricals
        self.num_classes = num_classes
        self.temperature = temperature
        self.straight_through = straight_through

    @property
    def param_dim(self) -> int:
        return self.num_categoricals * self.num_classes

    @property
    def sample_dim(self) -> int:
        return self.num_categoricals * self.num_classes

    def sample(self, logits: Tensor, deterministic: bool = False) -> Tensor:
        if logits.dim() == 2:
            logits = logits.view(-1, self.num_categoricals, self.num_classes)

        if deterministic:
            indices = logits.argmax(dim=-1)
            return F.one_hot(indices, self.num_classes).float()

        if self.straight_through:
            return F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
        else:
            return F.gumbel_softmax(logits, tau=self.temperature, hard=False, dim=-1)

    def kl_divergence(
        self,
        posterior_logits: Tensor,
        prior_logits: Tensor,
        free_nats: float = 0.0
    ) -> Tensor:
        if posterior_logits.dim() == 2:
            posterior_logits = posterior_logits.view(-1, self.num_categoricals, self.num_classes)
            prior_logits = prior_logits.view(-1, self.num_categoricals, self.num_classes)

        posterior = F.softmax(posterior_logits, dim=-1)

        kl = (posterior * (
            F.log_softmax(posterior_logits, dim=-1) -
            F.log_softmax(prior_logits, dim=-1)
        )).sum(dim=-1)

        if free_nats > 0:
            kl = torch.clamp(kl - free_nats, min=0.0)

        return kl.sum(dim=-1)


class SimNormLatentSpace(LatentSpace):
    """SimNorm (Simplicial Normalization) latent space for TD-MPC2."""

    def __init__(self, dim: int = 512, simnorm_dim: int = 8):
        super().__init__()
        self.dim = dim
        self.simnorm_dim = simnorm_dim

        if dim % simnorm_dim != 0:
            raise ValueError(f"dim ({dim}) must be divisible by simnorm_dim ({simnorm_dim})")

        self.num_simplices = dim // simnorm_dim

    @property
    def param_dim(self) -> int:
        return self.dim

    def sample(self, params: Tensor, deterministic: bool = False) -> Tensor:
        reshaped = params.view(-1, self.num_simplices, self.simnorm_dim)
        normalized = F.softmax(reshaped, dim=-1)
        return normalized.view(-1, self.dim)

    def kl_divergence(self, posterior_params: Tensor, prior_params: Tensor) -> Tensor:
        return torch.zeros(posterior_params.shape[0], device=posterior_params.device)
