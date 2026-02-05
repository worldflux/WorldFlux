"""Renderer3D skeleton world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import Renderer3DSkeletonConfig
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import Capability
from ...core.state import State


@WorldModelRegistry.register("renderer3d", Renderer3DSkeletonConfig)
class Renderer3DSkeletonWorldModel(WorldModel):
    """Minimal 3D-renderer style skeleton for camera/pose conditioned contracts."""

    def __init__(self, config: Renderer3DSkeletonConfig):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.VIDEO_PRED,
        }

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        self.encoder = nn.Sequential(nn.Linear(obs_dim, config.hidden_dim), nn.GELU())
        self.pose_adapter = nn.Linear(max(1, config.ray_dim), config.hidden_dim)
        self.action_adapter = nn.Linear(config.action_dim, config.hidden_dim)
        self.dynamics = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim), nn.GELU()
        )
        self.decoder = nn.Linear(config.hidden_dim, obs_dim)
        self.depth_head = nn.Linear(config.hidden_dim, 1)

    @staticmethod
    def _match_feature_dim(tensor: Tensor, feature_dim: int) -> Tensor:
        last_dim = tensor.shape[-1]
        if last_dim == feature_dim:
            return tensor
        if last_dim > feature_dim:
            return tensor[..., :feature_dim]
        pad_shape = (*tensor.shape[:-1], feature_dim - last_dim)
        pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=-1)

    def _obs_tensor(self, obs: Tensor | dict[str, Tensor] | WorldModelInput) -> Tensor:
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            tensor = obs.get("obs")
            if tensor is None:
                raise ValueError("Renderer3D skeleton expects 'obs' in input dictionary")
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
        tensor = self._obs_tensor(obs)
        latent = self.encoder(self._flatten(tensor))
        return State(tensors={"latent": latent})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        latent = state.tensors["latent"]
        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            action_tensor = torch.zeros(
                latent.shape[0], self.config.action_dim, device=latent.device
            )
        if action_tensor.dim() > 2:
            action_tensor = action_tensor.reshape(action_tensor.shape[0], -1)
        action_tensor = self._match_feature_dim(action_tensor, self.action_adapter.in_features)
        conditioned = self.action_adapter(action_tensor)

        if conditions is not None and conditions.camera_pose is not None:
            pose = conditions.camera_pose
            if pose.dim() > 2:
                pose = pose.reshape(pose.shape[0], -1)
            pose = self._match_feature_dim(pose, self.pose_adapter.in_features)
            conditioned = conditioned + self.pose_adapter(pose.to(dtype=conditioned.dtype))

        next_latent = self.dynamics(torch.cat([latent, conditioned], dim=-1))
        return State(tensors={"latent": next_latent})

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
        latent = state.tensors["latent"]
        obs_flat = self.decoder(latent)
        depth = self.depth_head(latent)
        preds = {
            "obs": obs_flat.reshape(obs_flat.shape[0], *self.config.obs_shape),
            "depth": depth,
        }
        return ModelOutput(predictions=preds)

    def loss(self, batch: Batch) -> LossOutput:
        obs = self._obs_tensor(batch.obs if batch.obs is not None else batch.inputs)
        if obs.dim() < 3:
            encoded = self.encode(obs)
            preds = self.decode(encoded).predictions
            recon = F.mse_loss(preds["obs"], obs)
            depth_reg = preds["depth"].pow(2).mean()
            total = recon + 0.01 * depth_reg
            return LossOutput(
                loss=total, components={"renderer3d_recon": recon, "depth_reg": depth_reg}
            )

        bsz, seq_len = obs.shape[:2]
        actions = batch.actions
        if actions is None:
            actions = torch.zeros(bsz, seq_len, self.config.action_dim, device=obs.device)

        latent = self.encode(obs[:, 0]).tensors["latent"]
        losses = []
        for t in range(seq_len - 1):
            latent = self.transition(State(tensors={"latent": latent}), actions[:, t]).tensors[
                "latent"
            ]
            pred = self.decoder(latent).reshape(bsz, *self.config.obs_shape)
            losses.append(F.mse_loss(pred, obs[:, t + 1]))

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=obs.device)
        return LossOutput(loss=loss, components={"renderer3d_skeleton": loss})
