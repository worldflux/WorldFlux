"""Physics skeleton world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import PhysicsSkeletonConfig
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import Capability
from ...core.state import State


@WorldModelRegistry.register("physics", PhysicsSkeletonConfig)
class PhysicsSkeletonWorldModel(WorldModel):
    """Differentiable-physics style skeleton."""

    def __init__(self, config: PhysicsSkeletonConfig):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.REWARD_PRED,
            Capability.CONTINUE_PRED,
        }

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        self.encoder = nn.Linear(obs_dim, config.state_dim)
        self.force_model = nn.Sequential(
            nn.Linear(config.state_dim + config.action_dim, config.hidden_dim),
            nn.Tanh(),
            nn.Linear(config.hidden_dim, config.state_dim),
        )
        self.reward_head = nn.Linear(config.state_dim + config.action_dim, 1)

    def _obs_tensor(self, obs: Tensor | dict[str, Tensor] | WorldModelInput) -> Tensor:
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            tensor = obs.get("obs")
            if tensor is None:
                raise ValueError("Physics skeleton expects 'obs' in input dictionary")
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
        latent = self.encoder(self._flatten(obs_tensor))
        return State(tensors={"state": latent})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del conditions, deterministic
        s = state.tensors["state"]
        action_tensor = self.action_tensor_or_none(action)
        if action_tensor is None:
            action_tensor = torch.zeros(s.shape[0], self.config.action_dim, device=s.device)
        if action_tensor.dim() > 2:
            action_tensor = action_tensor.reshape(action_tensor.shape[0], -1)

        force = self.force_model(torch.cat([s, action_tensor], dim=-1))
        dt = 0.1
        next_s = s + dt * force
        return State(tensors={"state": next_s})

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
        s = state.tensors["state"]
        zeros_action = torch.zeros(s.shape[0], self.config.action_dim, device=s.device)
        reward = self.reward_head(torch.cat([s, zeros_action], dim=-1))
        cont = torch.ones_like(reward)
        return ModelOutput(predictions={"reward": reward, "continue": cont})

    def loss(self, batch: Batch) -> LossOutput:
        obs = self._obs_tensor(batch.obs if batch.obs is not None else batch.inputs)
        actions = batch.actions
        rewards = batch.rewards

        if obs.dim() < 3:
            state = self.encode(obs)
            out = self.decode(state).predictions["reward"]
            target = (
                rewards.unsqueeze(-1)
                if rewards is not None and rewards.dim() == 1
                else torch.zeros_like(out)
            )
            loss = F.mse_loss(out, target)
            return LossOutput(loss=loss, components={"physics_reward": loss})

        bsz, seq_len = obs.shape[:2]
        if actions is None:
            actions = torch.zeros(bsz, seq_len, self.config.action_dim, device=obs.device)
        if rewards is None:
            rewards = torch.zeros(bsz, seq_len, device=obs.device)

        state = self.encode(obs[:, 0])
        losses = []
        for t in range(seq_len - 1):
            state = self.transition(state, actions[:, t])
            s = state.tensors["state"]
            pred_reward = self.reward_head(torch.cat([s, actions[:, t]], dim=-1)).squeeze(-1)
            losses.append(F.mse_loss(pred_reward, rewards[:, t + 1]))

        loss = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=obs.device)
        return LossOutput(loss=loss, components={"physics_skeleton": loss})
