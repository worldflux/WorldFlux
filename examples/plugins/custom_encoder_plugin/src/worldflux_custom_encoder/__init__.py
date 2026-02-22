"""Custom observation encoder plugin example.

Demonstrates how to register a custom ObservationEncoder component
that replaces the default encoder in any WorldFlux model family.
"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from worldflux import PluginManifest
from worldflux.core import ComponentSpec, WorldModelRegistry
from worldflux.core.exceptions import ConfigurationError
from worldflux.core.state import State

COMPONENT_ID = "custom_encoder.mlp_encoder"


class MLPObservationEncoder(nn.Module):
    """Simple MLP-based observation encoder.

    This encoder flattens the input observations and passes them through
    a two-layer MLP to produce a latent state representation.

    Args:
        input_dim: Total input dimension (product of obs_shape).
        latent_dim: Dimension of the output latent representation.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, input_dim: int = 64, latent_dim: int = 256, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        self.latent_dim = latent_dim

    def encode(self, observations: dict[str, Tensor]) -> State:
        """Encode observations into a latent state.

        Args:
            observations: Dictionary of observation tensors. The "obs" key
                is expected to contain the primary observation.

        Returns:
            State with a "latent" tensor of shape [batch, latent_dim].
        """
        obs = observations.get("obs")
        if obs is None:
            obs = next(iter(observations.values()))
        flat = obs.reshape(obs.shape[0], -1)
        latent = self.net(flat)
        return State(tensors={"latent": latent})


def register_components() -> PluginManifest:
    """Entry point called by WorldFlux plugin discovery."""
    try:
        WorldModelRegistry.register_component(
            COMPONENT_ID,
            MLPObservationEncoder,
            ComponentSpec(
                name="Custom MLP Observation Encoder",
                component_type="observation_encoder",
            ),
        )
    except ConfigurationError:
        pass
    return PluginManifest(
        plugin_api_version="0.x-experimental",
        worldflux_version_range=">=0.1.0,<0.2.0",
        capabilities=("observation-encoder",),
        experimental=True,
    )
