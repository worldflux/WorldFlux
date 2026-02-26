"""Custom dynamics model plugin example.

Demonstrates how to register a custom DynamicsModel component
that provides an alternative transition function for latent states.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from worldflux.core import ComponentSpec, PluginManifest, WorldModelRegistry
from worldflux.core.exceptions import ConfigurationError
from worldflux.core.state import State

COMPONENT_ID = "custom_dynamics.residual_dynamics"


class ResidualDynamicsModel(nn.Module):
    """Residual MLP dynamics model.

    Predicts state deltas instead of absolute next states, which can
    improve learning stability for small state changes.

    Args:
        latent_dim: Dimension of the latent state.
        action_dim: Dimension of the action space.
        hidden_dim: Hidden layer dimension.
    """

    def __init__(self, latent_dim: int = 256, action_dim: int = 6, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def transition(
        self,
        state: State,
        conditioned: dict[str, Tensor],
        deterministic: bool = False,
    ) -> State:
        """Predict next state via residual connection.

        Args:
            state: Current latent state (must have "latent" tensor).
            conditioned: Conditioning tensors (must have "action" tensor).
            deterministic: Ignored for this simple model.

        Returns:
            Next State with updated "latent" tensor.
        """
        latent = state.tensors["latent"]
        action = conditioned.get("action", torch.zeros(latent.shape[0], 1, device=latent.device))
        x = torch.cat([latent, action], dim=-1)
        delta = self.net(x)
        next_latent = latent + delta  # Residual connection
        return State(tensors={"latent": next_latent}, meta=state.meta)


def register_components() -> PluginManifest:
    """Entry point called by WorldFlux plugin discovery."""
    try:
        WorldModelRegistry.register_component(
            COMPONENT_ID,
            ResidualDynamicsModel,
            ComponentSpec(
                name="Custom Residual Dynamics Model",
                component_type="dynamics_model",
            ),
        )
    except ConfigurationError:
        pass
    return PluginManifest(
        plugin_api_version="0.x-experimental",
        worldflux_version_range=">=0.1.0,<0.2.0",
        capabilities=("dynamics-model",),
        experimental=True,
    )
