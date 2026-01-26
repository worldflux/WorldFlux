"""Recurrent State-Space Model for DreamerV3."""

import torch
import torch.nn as nn
from torch import Tensor

from ...core.latent_space import CategoricalLatentSpace
from ...core.state import LatentState


class RSSM(nn.Module):
    """
    Recurrent State-Space Model (DreamerV3).

    Combines deterministic state (h) and stochastic state (z):
        h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}])  # Sequence Model
        z_t_prior = prior(h_t)                   # Prior (imagination)
        z_t_post = posterior(h_t, embed(x_t))    # Posterior (learning)
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        deter_dim: int = 4096,
        stoch_discrete: int = 32,
        stoch_classes: int = 32,
        hidden_dim: int = 640,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.deter_dim = deter_dim
        self.stoch_discrete = stoch_discrete
        self.stoch_classes = stoch_classes
        self.stoch_dim = stoch_discrete * stoch_classes
        self.hidden_dim = hidden_dim

        self.latent_space = CategoricalLatentSpace(
            num_categoricals=stoch_discrete,
            num_classes=stoch_classes,
        )

        # Sequence Model (GRU)
        gru_input_dim = self.stoch_dim + action_dim
        self.gru = nn.GRUCell(gru_input_dim, deter_dim)

        # Prior: h_t -> z_t_prior
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_dim),
        )

        # Posterior: [h_t, embed] -> z_t_post
        self.posterior_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_dim),
        )

    def initial_state(self, batch_size: int, device: torch.device) -> LatentState:
        """Create initial state."""
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_discrete, self.stoch_classes, device=device)
        z[..., 0] = 1.0  # One-hot first class

        return LatentState(
            deterministic=h,
            stochastic=z,
            latent_type="categorical",
        )

    def prior_step(
        self, state: LatentState, action: Tensor, deterministic: bool = False
    ) -> LatentState:
        """Prior transition (no observation, for imagination)."""
        assert state.stochastic is not None, "RSSM requires stochastic state"
        z_flat = state.stochastic.flatten(start_dim=1)

        gru_input = torch.cat([z_flat, action], dim=-1)
        h_next = self.gru(gru_input, state.deterministic)

        prior_logits = self.prior_net(h_next)
        prior_logits = prior_logits.view(-1, self.stoch_discrete, self.stoch_classes)

        z_next = self.latent_space.sample(prior_logits, deterministic=deterministic)

        return LatentState(
            deterministic=h_next,
            stochastic=z_next,
            logits=prior_logits,
            latent_type="categorical",
        )

    def posterior_step(self, state: LatentState, action: Tensor, obs_embed: Tensor) -> LatentState:
        """Posterior transition (with observation, for learning)."""
        assert state.stochastic is not None, "RSSM requires stochastic state"
        z_flat = state.stochastic.flatten(start_dim=1)
        gru_input = torch.cat([z_flat, action], dim=-1)
        h_next = self.gru(gru_input, state.deterministic)

        prior_logits = self.prior_net(h_next)
        prior_logits = prior_logits.view(-1, self.stoch_discrete, self.stoch_classes)

        posterior_input = torch.cat([h_next, obs_embed], dim=-1)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(-1, self.stoch_discrete, self.stoch_classes)

        z_next = self.latent_space.sample(posterior_logits, deterministic=False)

        return LatentState(
            deterministic=h_next,
            stochastic=z_next,
            logits=posterior_logits,
            prior_logits=prior_logits,
            posterior_logits=posterior_logits,
            latent_type="categorical",
        )

    @property
    def feature_dim(self) -> int:
        """Feature dimension for downstream heads."""
        return self.deter_dim + self.stoch_dim
