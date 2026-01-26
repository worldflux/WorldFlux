"""Latent state representation for world models."""

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import Tensor

from .exceptions import StateError


@dataclass
class LatentState:
    """
    Universal latent state representation for all world models.

    This dataclass provides a unified interface for storing latent representations
    across different world model architectures. It supports various state types
    and distribution parameters needed for training and inference.

    Attributes:
        deterministic: Deterministic state component (h in RSSM).
            Shape: [batch_size, deter_dim]
        stochastic: Stochastic state component (z in RSSM).
            Shape: [batch_size, stoch_dim] or [batch_size, num_categories, num_classes]
        logits: Raw logits for categorical distributions.
            Shape: [batch_size, num_categories, num_classes]
        mean: Mean for Gaussian distributions.
            Shape: [batch_size, latent_dim]
        std: Standard deviation for Gaussian distributions.
            Shape: [batch_size, latent_dim]
        prior_logits: Prior distribution logits (for KL computation).
            Shape: [batch_size, num_categories, num_classes]
        posterior_logits: Posterior distribution logits (for KL computation).
            Shape: [batch_size, num_categories, num_classes]
        codebook_indices: VQ-VAE codebook indices.
            Shape: [batch_size, num_tokens]
        commitment_loss: VQ-VAE commitment loss.
            Shape: scalar or [batch_size]
        latent_type: Type identifier ("deterministic", "categorical", "gaussian", "vq").
        metadata: Additional model-specific metadata.

    Supported Architectures:
        - DreamerV3: deterministic (h) + stochastic (z) + prior/posterior_logits
        - TD-MPC2: deterministic only (SimNorm embedding)
        - V-JEPA: deterministic only (ViT features)
        - IRIS: codebook_indices (VQ-VAE)

    Example:
        >>> # DreamerV3 state
        >>> state = LatentState(
        ...     deterministic=torch.randn(32, 512),
        ...     stochastic=torch.randn(32, 32, 32),
        ...     posterior_logits=torch.randn(32, 32, 32),
        ...     latent_type="categorical",
        ... )
        >>> features = state.features  # [32, 512 + 32*32]

        >>> # TD-MPC2 state
        >>> state = LatentState(
        ...     deterministic=torch.randn(32, 256),
        ...     latent_type="deterministic",
        ... )
        >>> features = state.features  # [32, 256]
    """

    # Primary components
    deterministic: Tensor | None = None
    stochastic: Tensor | None = None

    # Distribution parameters (for loss computation)
    logits: Tensor | None = None
    mean: Tensor | None = None
    std: Tensor | None = None
    prior_logits: Tensor | None = None
    posterior_logits: Tensor | None = None

    # VQ-VAE
    codebook_indices: Tensor | None = None
    commitment_loss: Tensor | None = None

    # Metadata
    latent_type: str = "deterministic"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate state after initialization."""
        # At least one state component must be present
        has_state = (
            self.deterministic is not None
            or self.stochastic is not None
            or self.codebook_indices is not None
        )
        if not has_state:
            raise StateError(
                "LatentState requires at least one of: deterministic, stochastic, "
                "or codebook_indices"
            )

    @property
    def features(self) -> Tensor:
        """
        Combined feature vector for downstream prediction heads.

        Concatenates deterministic and stochastic components into a single
        feature vector. For categorical stochastic states (3D tensors),
        the last two dimensions are flattened.

        Returns:
            Combined features of shape [batch_size, feature_dim].

        Raises:
            StateError: If no valid state components are available.

        Example:
            >>> state = LatentState(
            ...     deterministic=torch.randn(32, 512),
            ...     stochastic=torch.randn(32, 32, 32),
            ... )
            >>> state.features.shape
            torch.Size([32, 1536])  # 512 + 32*32
        """
        components: list[Tensor] = []

        if self.deterministic is not None:
            components.append(self.deterministic)

        if self.stochastic is not None:
            stoch = self.stochastic
            if stoch.dim() == 3:  # [batch, num_categories, num_classes]
                stoch = stoch.flatten(start_dim=1)
            elif stoch.dim() > 3:
                raise StateError(
                    f"Stochastic tensor has unsupported shape: {stoch.shape}. "
                    f"Expected 2D [batch, dim] or 3D [batch, num_cat, num_class]."
                )
            components.append(stoch)

        if not components:
            raise StateError(
                "Cannot compute features: no deterministic or stochastic component. "
                f"State has: deterministic={self.deterministic is not None}, "
                f"stochastic={self.stochastic is not None}, "
                f"codebook_indices={self.codebook_indices is not None}"
            )

        return torch.cat(components, dim=-1) if len(components) > 1 else components[0]

    @property
    def batch_size(self) -> int:
        """
        Batch size of the state.

        Returns:
            The batch size (first dimension) of the state tensors.

        Raises:
            StateError: If no valid tensor is found to determine batch size.
        """
        for name, tensor in [
            ("deterministic", self.deterministic),
            ("stochastic", self.stochastic),
            ("codebook_indices", self.codebook_indices),
        ]:
            if tensor is not None:
                return tensor.shape[0]

        raise StateError(
            "Cannot determine batch_size: no valid tensor found in state. "
            "Ensure at least one of deterministic, stochastic, or codebook_indices is set."
        )

    @property
    def device(self) -> torch.device:
        """
        Device of the state tensors.

        Returns:
            The device of the first valid tensor found.

        Raises:
            StateError: If no valid tensor is found to determine device.
        """
        for name, tensor in [
            ("deterministic", self.deterministic),
            ("stochastic", self.stochastic),
            ("codebook_indices", self.codebook_indices),
        ]:
            if tensor is not None:
                return tensor.device

        raise StateError(
            "Cannot determine device: no valid tensor found in state. "
            "Ensure at least one of deterministic, stochastic, or codebook_indices is set."
        )

    @property
    def feature_dim(self) -> int:
        """
        Dimension of the combined feature vector.

        Returns:
            The total feature dimension.
        """
        return self.features.shape[-1]

    def to(self, device: torch.device | str) -> "LatentState":
        """
        Move all tensors to the specified device.

        Args:
            device: Target device (e.g., "cuda", "cpu", torch.device("cuda:0")).

        Returns:
            New LatentState with all tensors on the target device.
        """
        if isinstance(device, str):
            device = torch.device(device)

        def move(t: Tensor | None) -> Tensor | None:
            return t.to(device) if t is not None else None

        return LatentState(
            deterministic=move(self.deterministic),
            stochastic=move(self.stochastic),
            logits=move(self.logits),
            mean=move(self.mean),
            std=move(self.std),
            prior_logits=move(self.prior_logits),
            posterior_logits=move(self.posterior_logits),
            codebook_indices=move(self.codebook_indices),
            commitment_loss=move(self.commitment_loss),
            latent_type=self.latent_type,
            metadata=self.metadata,
        )

    def detach(self) -> "LatentState":
        """
        Detach all tensors from the computation graph.

        Returns:
            New LatentState with all tensors detached.
        """

        def detach_tensor(t: Tensor | None) -> Tensor | None:
            return t.detach() if t is not None else None

        return LatentState(
            deterministic=detach_tensor(self.deterministic),
            stochastic=detach_tensor(self.stochastic),
            logits=detach_tensor(self.logits),
            mean=detach_tensor(self.mean),
            std=detach_tensor(self.std),
            prior_logits=detach_tensor(self.prior_logits),
            posterior_logits=detach_tensor(self.posterior_logits),
            codebook_indices=detach_tensor(self.codebook_indices),
            commitment_loss=detach_tensor(self.commitment_loss),
            latent_type=self.latent_type,
            metadata=self.metadata,
        )

    def clone(self) -> "LatentState":
        """
        Create a deep copy of the state.

        Returns:
            New LatentState with cloned tensors and metadata.
        """

        def clone_tensor(t: Tensor | None) -> Tensor | None:
            return t.clone() if t is not None else None

        return LatentState(
            deterministic=clone_tensor(self.deterministic),
            stochastic=clone_tensor(self.stochastic),
            logits=clone_tensor(self.logits),
            mean=clone_tensor(self.mean),
            std=clone_tensor(self.std),
            prior_logits=clone_tensor(self.prior_logits),
            posterior_logits=clone_tensor(self.posterior_logits),
            codebook_indices=clone_tensor(self.codebook_indices),
            commitment_loss=clone_tensor(self.commitment_loss),
            latent_type=self.latent_type,
            metadata=dict(self.metadata),
        )

    def __repr__(self) -> str:
        """Return a detailed string representation."""
        parts = [f"LatentState(type={self.latent_type!r}"]

        if self.deterministic is not None:
            parts.append(f"deter={list(self.deterministic.shape)}")
        if self.stochastic is not None:
            parts.append(f"stoch={list(self.stochastic.shape)}")
        if self.codebook_indices is not None:
            parts.append(f"codebook={list(self.codebook_indices.shape)}")

        try:
            parts.append(f"device={self.device}")
        except StateError:
            pass

        return ", ".join(parts) + ")"
