"""GAN skeleton world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import GANSkeletonConfig
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import Capability
from ...core.state import State


@WorldModelRegistry.register("gan", GANSkeletonConfig)
class GANSkeletonWorldModel(WorldModel):
    """Lightweight GAN-style skeleton for API coverage."""

    def __init__(self, config: GANSkeletonConfig):
        super().__init__()
        self.config = config
        self.capabilities = {Capability.LATENT_DYNAMICS, Capability.OBS_DECODER}

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        self.encoder = nn.Linear(obs_dim, config.generator_dim)
        self.generator = nn.Sequential(
            nn.Linear(config.generator_dim + config.action_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, obs_dim),
        )
        self.discriminator = nn.Sequential(
            nn.Linear(obs_dim, config.discriminator_dim),
            nn.ReLU(),
            nn.Linear(config.discriminator_dim, 1),
        )

    def _obs_tensor(self, obs: Tensor | dict[str, Tensor] | WorldModelInput) -> Tensor:
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            tensor = obs.get("obs")
            if tensor is None:
                raise ValueError("GAN skeleton expects 'obs' in input dictionary")
            return tensor
        return obs

    @staticmethod
    def _flatten(obs: Tensor) -> Tensor:
        if obs.dim() > 2:
            return obs.reshape(obs.shape[0], -1)
        return obs

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        obs_tensor = self._obs_tensor(obs)
        z = self.encoder(self._flatten(obs_tensor))
        return State(tensors={"latent": z})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del conditions, deterministic
        z = state.tensors["latent"]
        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            action_tensor = torch.zeros(z.shape[0], self.config.action_dim, device=z.device)
        if action_tensor.dim() > 2:
            action_tensor = action_tensor.reshape(action_tensor.shape[0], -1)
        fake = self.generator(torch.cat([z, action_tensor], dim=-1))
        return State(tensors={"latent": self.encoder(fake)})

    def update(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        conditions: ConditionPayload | None = None,
    ) -> State:
        del state, action, conditions
        return self.encode(obs)

    def decode(self, state: State, conditions: ConditionPayload | None = None) -> ModelOutput:
        del conditions
        z = state.tensors["latent"]
        zeros = torch.zeros(z.shape[0], self.config.action_dim, device=z.device)
        fake = self.generator(torch.cat([z, zeros], dim=-1)).reshape(
            z.shape[0], *self.config.obs_shape
        )
        return ModelOutput(predictions={"obs": fake})

    def loss(self, batch: Batch) -> LossOutput:
        obs = self._obs_tensor(batch.obs if batch.obs is not None else batch.inputs)
        if obs.dim() == 2:
            real_flat = obs
            z = self.encoder(real_flat)
            action = batch.actions
            if action is None:
                action = torch.zeros(obs.shape[0], self.config.action_dim, device=obs.device)
            if action.dim() > 2:
                action = action.reshape(action.shape[0], -1)
            fake_flat = self.generator(torch.cat([z, action], dim=-1))
            d_real = self.discriminator(real_flat)
            d_fake = self.discriminator(fake_flat.detach())
            adv_d = F.binary_cross_entropy_with_logits(
                d_real, torch.ones_like(d_real)
            ) + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake))
            adv_g = F.binary_cross_entropy_with_logits(
                self.discriminator(fake_flat), torch.ones_like(d_fake)
            )
            recon = F.mse_loss(fake_flat, real_flat)
            total = adv_g + recon + 0.1 * adv_d
            return LossOutput(
                loss=total, components={"gan_adv_g": adv_g, "gan_recon": recon, "gan_adv_d": adv_d}
            )

        # Sequence fallback: use next-frame reconstruction objective.
        bsz, seq_len = obs.shape[:2]
        obs_flat = obs.reshape(bsz * seq_len, -1)
        z_all = self.encoder(obs_flat).reshape(bsz, seq_len, -1)
        actions = batch.actions
        if actions is None:
            actions = torch.zeros(bsz, seq_len, self.config.action_dim, device=obs.device)

        pred = self.generator(
            torch.cat([z_all[:, :-1], actions[:, :-1]], dim=-1).reshape(
                -1, z_all.shape[-1] + actions.shape[-1]
            )
        )
        target = obs[:, 1:].reshape(-1, obs_flat.shape[-1])
        loss = F.mse_loss(pred, target)
        return LossOutput(loss=loss, components={"gan_skeleton": loss})
