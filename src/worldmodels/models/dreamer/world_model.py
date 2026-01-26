"""DreamerV3 World Model implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.config import DreamerV3Config
from ...core.registry import WorldModelRegistry
from ...core.state import LatentState
from ...core.trajectory import Trajectory
from .decoder import CNNDecoder, MLPDecoder
from .encoder import CNNEncoder, MLPEncoder
from .heads import ContinueHead, RewardHead, symlog
from .rssm import RSSM


@WorldModelRegistry.register("dreamer", DreamerV3Config)
class DreamerV3WorldModel(nn.Module):
    """
    DreamerV3 world model implementation.

    Components:
        - Encoder: observation -> embedding
        - RSSM: latent state transitions
        - Decoder: latent state -> observation reconstruction
        - Reward Head: reward prediction
        - Continue Head: continue probability prediction
    """

    def __init__(self, config: DreamerV3Config):
        super().__init__()
        self.config = config

        # Encoder
        self.encoder: CNNEncoder | MLPEncoder
        if config.encoder_type == "cnn":
            self.encoder = CNNEncoder(
                obs_shape=config.obs_shape,
                depth=config.cnn_depth,
                kernels=config.cnn_kernels,
            )
        else:
            obs_dim = (
                config.obs_shape[0]
                if len(config.obs_shape) == 1
                else int(torch.prod(torch.tensor(config.obs_shape)).item())
            )
            self.encoder = MLPEncoder(
                input_dim=obs_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.deter_dim,
            )

        # RSSM
        self.rssm = RSSM(
            embed_dim=self.encoder.output_dim,
            action_dim=config.action_dim,
            deter_dim=config.deter_dim,
            stoch_discrete=config.stoch_discrete,
            stoch_classes=config.stoch_classes,
            hidden_dim=config.hidden_dim,
        )

        # Decoder
        feature_dim = self.rssm.feature_dim
        self.decoder: CNNDecoder | MLPDecoder
        if config.decoder_type == "cnn":
            self.decoder = CNNDecoder(
                feature_dim=feature_dim,
                obs_shape=config.obs_shape,
                depth=config.cnn_depth,
                kernels=config.cnn_kernels,
            )
        else:
            self.decoder = MLPDecoder(
                feature_dim=feature_dim,
                output_shape=config.obs_shape,
                hidden_dim=config.hidden_dim,
            )

        # Prediction heads
        self.reward_head = RewardHead(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
            use_symlog=config.use_symlog,
        )
        self.continue_head = ContinueHead(
            feature_dim=feature_dim,
            hidden_dim=config.hidden_dim,
        )

        self.register_buffer("_device_tracker", torch.empty(0))

    @property
    def device(self) -> torch.device:
        device = self._device_tracker.device
        assert isinstance(device, torch.device)
        return device

    def encode(self, obs: Tensor, deterministic: bool = False) -> LatentState:
        """Encode observation to latent state."""
        embed = self.encoder(obs)

        batch_size = obs.shape[0]
        initial = self.rssm.initial_state(batch_size, obs.device)
        zero_action = torch.zeros(batch_size, self.config.action_dim, device=obs.device)

        state = self.rssm.posterior_step(initial, zero_action, embed)
        return state

    def predict(
        self, state: LatentState, action: Tensor, deterministic: bool = False
    ) -> LatentState:
        """Predict next state (prior, for imagination)."""
        return self.rssm.prior_step(state, action, deterministic=deterministic)

    def observe(self, state: LatentState, action: Tensor, obs: Tensor) -> LatentState:
        """Update state with observation (posterior)."""
        embed = self.encoder(obs)
        return self.rssm.posterior_step(state, action, embed)

    def decode(self, state: LatentState) -> dict[str, Tensor]:
        """Decode latent state to predictions."""
        features = state.features

        return {
            "obs": self.decoder(features),
            "reward": self.reward_head(features),
            "continue": self.continue_head(features),
        }

    def imagine(
        self, initial_state: LatentState, actions: Tensor, deterministic: bool = False
    ) -> Trajectory:
        """Multi-step imagination rollout."""
        horizon = actions.shape[0]
        states = [initial_state]
        rewards = []
        continues = []

        state = initial_state
        for t in range(horizon):
            state = self.predict(state, actions[t], deterministic=deterministic)
            states.append(state)

            decoded = self.decode(state)
            rewards.append(decoded["reward"])
            continues.append(decoded["continue"])

        return Trajectory(
            states=states,
            actions=actions,
            rewards=torch.stack(rewards, dim=0).squeeze(-1),
            continues=torch.stack(continues, dim=0).squeeze(-1),
        )

    def initial_state(self, batch_size: int, device: torch.device | None = None) -> LatentState:
        """Create initial state."""
        if device is None:
            device = self.device
        return self.rssm.initial_state(batch_size, device)

    def compute_loss(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        """
        Compute training losses.

        Args:
            batch:
                - obs: [batch, seq_len, *obs_shape]
                - actions: [batch, seq_len, action_dim]
                - rewards: [batch, seq_len]
                - continues: [batch, seq_len]
        """
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        continues = batch["continues"]

        batch_size, seq_len = obs.shape[:2]
        device = obs.device

        # Process sequence
        state = self.initial_state(batch_size, device)

        states = []
        for t in range(seq_len):
            if t == 0:
                action = torch.zeros(batch_size, self.config.action_dim, device=device)
            else:
                action = actions[:, t - 1]

            state = self.observe(state, action, obs[:, t])
            states.append(state)

        losses: dict[str, Tensor] = {}

        # KL loss
        kl_loss = torch.tensor(0.0, device=device)
        for state in states:
            if state.posterior_logits is not None and state.prior_logits is not None:
                kl = self.rssm.latent_space.kl_divergence(
                    state.posterior_logits,
                    state.prior_logits,
                    free_nats=self.config.kl_free,
                )
                kl_loss = kl_loss + kl.mean()
        losses["kl"] = kl_loss / seq_len

        # Reconstruction loss
        recon_loss = torch.tensor(0.0, device=device)
        for t, state in enumerate(states):
            decoded = self.decode(state)

            if self.config.use_symlog:
                target = symlog(obs[:, t])
            else:
                target = obs[:, t]
            recon_loss = recon_loss + F.mse_loss(decoded["obs"], target)
        losses["reconstruction"] = recon_loss / seq_len

        # Reward loss
        reward_loss = torch.tensor(0.0, device=device)
        for t in range(1, seq_len):
            decoded = self.decode(states[t])
            target = rewards[:, t]
            if self.config.use_symlog:
                target = symlog(target)
            reward_loss = reward_loss + F.mse_loss(decoded["reward"].squeeze(-1), target)
        losses["reward"] = reward_loss / max(seq_len - 1, 1)

        # Continue loss
        continue_loss = torch.tensor(0.0, device=device)
        for t in range(1, seq_len):
            decoded = self.decode(states[t])
            target = continues[:, t]
            continue_loss = continue_loss + F.binary_cross_entropy_with_logits(
                decoded["continue"].squeeze(-1), target
            )
        losses["continue"] = continue_loss / max(seq_len - 1, 1)

        # Total loss
        losses["loss"] = (
            self.config.loss_scales["reconstruction"] * losses["reconstruction"]
            + self.config.loss_scales["kl"] * losses["kl"]
            + self.config.loss_scales["reward"] * losses["reward"]
            + self.config.loss_scales["continue"] * losses["continue"]
        )

        return losses

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> "DreamerV3WorldModel":
        from ...core.registry import WorldModelRegistry

        model = WorldModelRegistry.from_pretrained(name_or_path, **kwargs)
        assert isinstance(model, cls)
        return model

    def save_pretrained(self, path: str) -> None:
        import os

        os.makedirs(path, exist_ok=True)
        self.config.save(os.path.join(path, "config.json"))
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))
