"""DiT skeleton world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import DiTSkeletonConfig
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import Capability
from ...core.state import State


@WorldModelRegistry.register("dit", DiTSkeletonConfig)
class DiTSkeletonWorldModel(WorldModel):
    """Minimal DiT-style skeleton for contract and pipeline validation."""

    def __init__(self, config: DiTSkeletonConfig):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.DIFFUSION,
        }

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, config.hidden_dim),
            nn.GELU(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(config.hidden_dim + config.action_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.decoder = nn.Linear(config.hidden_dim, obs_dim)

    def _flatten(self, obs: Tensor) -> Tensor:
        if obs.dim() > 2:
            return obs.reshape(obs.shape[0], -1)
        return obs

    def _obs_tensor(self, obs: Tensor | dict[str, Tensor] | WorldModelInput) -> Tensor:
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            tensor = obs.get("obs")
            if tensor is None:
                raise ValueError("DiT skeleton expects 'obs' in input dictionary")
            return tensor
        return obs

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        obs_tensor = self._obs_tensor(obs)
        latent = self.encoder(self._flatten(obs_tensor))
        return State(tensors={"latent": latent})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del conditions, deterministic
        latent = state.tensors["latent"]
        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            action_tensor = torch.zeros(
                latent.shape[0], self.config.action_dim, device=latent.device
            )
        if action_tensor.dim() > 2:
            action_tensor = action_tensor.reshape(action_tensor.shape[0], -1)
        z_next = self.dynamics(torch.cat([latent, action_tensor], dim=-1))
        return State(tensors={"latent": z_next})

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
        obs_flat = self.decoder(state.tensors["latent"])
        preds = {"obs": obs_flat.reshape(obs_flat.shape[0], *self.config.obs_shape)}
        return ModelOutput(predictions=preds)

    def loss(self, batch: Batch) -> LossOutput:
        obs = self._obs_tensor(batch.obs if batch.obs is not None else batch.inputs)
        if obs.dim() < 3:
            encoded = self.encode(obs)
            pred = self.decode(encoded).predictions["obs"]
            loss = F.mse_loss(pred, obs)
            return LossOutput(loss=loss, components={"reconstruction": loss})

        bsz, seq_len = obs.shape[:2]
        obs_flat = obs.reshape(bsz * seq_len, *obs.shape[2:])
        z = self.encoder(self._flatten(obs_flat)).reshape(bsz, seq_len, -1)

        if batch.actions is None:
            action = torch.zeros(bsz, seq_len, self.config.action_dim, device=obs.device)
        else:
            action = batch.actions
            if action.dim() == 2:
                action = action.unsqueeze(1).expand(-1, seq_len, -1)

        pred_latents = self.dynamics(torch.cat([z[:, :-1], action[:, :-1]], dim=-1))
        pred_obs = self.decoder(pred_latents.reshape(-1, pred_latents.shape[-1]))
        pred_obs = pred_obs.reshape(bsz, seq_len - 1, *self.config.obs_shape)
        target = obs[:, 1:]

        loss = F.mse_loss(pred_obs, target)
        return LossOutput(loss=loss, components={"dit_skeleton": loss})
