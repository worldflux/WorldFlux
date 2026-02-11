"""Recurrent State-Space Model for DreamerV3."""

import torch
import torch.nn as nn
from torch import Tensor

from ...core.latent_space import CategoricalLatentSpace
from ...core.state import State

_RSSM_NAN_FILL = 0.0
_RSSM_INF_CLAMP = 1e4
_RSSM_LOGIT_CLAMP = 30.0


def _stabilize_tensor(x: Tensor, *, clamp: float) -> Tensor:
    return torch.nan_to_num(x, nan=_RSSM_NAN_FILL, posinf=clamp, neginf=-clamp).clamp(-clamp, clamp)


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

    def initial_state(self, batch_size: int, device: torch.device) -> State:
        """Create initial state."""
        h = torch.zeros(batch_size, self.deter_dim, device=device)
        z = torch.zeros(batch_size, self.stoch_discrete, self.stoch_classes, device=device)
        z[..., 0] = 1.0  # One-hot first class

        return State(
            tensors={
                "deter": h,
                "stoch": z,
            },
            meta={"latent_type": "categorical"},
        )

    def prior_step(self, state: State, action: Tensor, deterministic: bool = False) -> State:
        """Prior transition (no observation, for imagination)."""
        z = state.tensors.get("stoch")
        h = state.tensors.get("deter")
        if z is None or h is None:
            raise ValueError("RSSM requires 'deter' and 'stoch' tensors in State")
        z_flat = _stabilize_tensor(z.flatten(start_dim=1), clamp=_RSSM_INF_CLAMP)
        action = _stabilize_tensor(action, clamp=_RSSM_INF_CLAMP)

        gru_input = torch.cat([z_flat, action], dim=-1)
        gru_input = _stabilize_tensor(gru_input, clamp=_RSSM_INF_CLAMP)
        h_next = self.gru(gru_input, h)
        h_next = _stabilize_tensor(h_next, clamp=_RSSM_INF_CLAMP)

        prior_logits = self.prior_net(h_next)
        prior_logits = prior_logits.view(-1, self.stoch_discrete, self.stoch_classes)
        prior_logits = _stabilize_tensor(prior_logits, clamp=_RSSM_LOGIT_CLAMP)

        z_next = self.latent_space.sample(prior_logits, deterministic=deterministic)

        return State(
            tensors={
                "deter": h_next,
                "stoch": z_next,
                "logits": prior_logits,
                "prior_logits": prior_logits,
            },
            meta={"latent_type": "categorical"},
        )

    def posterior_step(self, state: State, action: Tensor, obs_embed: Tensor) -> State:
        """Posterior transition (with observation, for learning)."""
        z = state.tensors.get("stoch")
        h = state.tensors.get("deter")
        if z is None or h is None:
            raise ValueError("RSSM requires 'deter' and 'stoch' tensors in State")
        z_flat = _stabilize_tensor(z.flatten(start_dim=1), clamp=_RSSM_INF_CLAMP)
        action = _stabilize_tensor(action, clamp=_RSSM_INF_CLAMP)
        obs_embed = _stabilize_tensor(obs_embed, clamp=_RSSM_INF_CLAMP)
        gru_input = torch.cat([z_flat, action], dim=-1)
        gru_input = _stabilize_tensor(gru_input, clamp=_RSSM_INF_CLAMP)
        h_next = self.gru(gru_input, h)
        h_next = _stabilize_tensor(h_next, clamp=_RSSM_INF_CLAMP)

        prior_logits = self.prior_net(h_next)
        prior_logits = prior_logits.view(-1, self.stoch_discrete, self.stoch_classes)
        prior_logits = _stabilize_tensor(prior_logits, clamp=_RSSM_LOGIT_CLAMP)

        posterior_input = torch.cat([h_next, obs_embed], dim=-1)
        posterior_input = _stabilize_tensor(posterior_input, clamp=_RSSM_INF_CLAMP)
        posterior_logits = self.posterior_net(posterior_input)
        posterior_logits = posterior_logits.view(-1, self.stoch_discrete, self.stoch_classes)
        posterior_logits = _stabilize_tensor(posterior_logits, clamp=_RSSM_LOGIT_CLAMP)

        z_next = self.latent_space.sample(posterior_logits, deterministic=False)

        return State(
            tensors={
                "deter": h_next,
                "stoch": z_next,
                "logits": posterior_logits,
                "prior_logits": prior_logits,
                "posterior_logits": posterior_logits,
            },
            meta={"latent_type": "categorical"},
        )

    @property
    def feature_dim(self) -> int:
        """Feature dimension for downstream heads."""
        return self.deter_dim + self.stoch_dim
