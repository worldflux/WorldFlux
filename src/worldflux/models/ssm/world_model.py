"""SSM/Mamba skeleton world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import SSMSkeletonConfig
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import Capability
from ...core.state import State


@WorldModelRegistry.register("ssm", SSMSkeletonConfig)
class SSMSkeletonWorldModel(WorldModel):
    """Minimal SSM-style skeleton for long-context interfaces."""

    def __init__(self, config: SSMSkeletonConfig):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.REPRESENTATION,
        }

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        self.encoder = nn.Linear(obs_dim, config.hidden_dim)
        self.transition_core = nn.GRUCell(config.hidden_dim + config.action_dim, config.hidden_dim)
        self.decoder = nn.Linear(config.hidden_dim, obs_dim)

    def _obs_tensor(self, obs: Tensor | dict[str, Tensor] | WorldModelInput) -> Tensor:
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            tensor = obs.get("obs")
            if tensor is None:
                raise ValueError("SSM skeleton expects 'obs' in input dictionary")
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
        latent = torch.tanh(self.encoder(self._flatten(obs_tensor)))
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
        latent_next = self.transition_core(torch.cat([latent, action_tensor], dim=-1), latent)
        return State(tensors={"latent": latent_next})

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
        obs = obs_flat.reshape(obs_flat.shape[0], *self.config.obs_shape)
        return ModelOutput(predictions={"obs": obs, "representation": state.tensors["latent"]})

    def loss(self, batch: Batch) -> LossOutput:
        obs = self._obs_tensor(batch.obs if batch.obs is not None else batch.inputs)
        if obs.dim() < 3:
            encoded = self.encode(obs)
            pred = self.decode(encoded).predictions["obs"]
            loss = F.mse_loss(pred, obs)
            return LossOutput(loss=loss, components={"ssm_reconstruction": loss})

        bsz, seq_len = obs.shape[:2]
        if batch.actions is None:
            actions = torch.zeros(bsz, seq_len, self.config.action_dim, device=obs.device)
        else:
            actions = batch.actions
            if actions.dim() == 2:
                actions = actions.unsqueeze(1).expand(-1, seq_len, -1)

        latent = self.encode(obs[:, 0]).tensors["latent"]
        losses = []
        for t in range(seq_len - 1):
            latent = self.transition(State(tensors={"latent": latent}), actions[:, t]).tensors[
                "latent"
            ]
            pred = self.decoder(latent).reshape(bsz, *self.config.obs_shape)
            losses.append(F.mse_loss(pred, obs[:, t + 1]))

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=obs.device)
        return LossOutput(loss=loss, components={"ssm_skeleton": loss})
