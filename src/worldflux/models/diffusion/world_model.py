"""Minimal diffusion-based world model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import DiffusionWorldModelConfig
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.registry import WorldModelRegistry
from ...core.spec import (
    ActionSpec,
    Capability,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from ...core.state import State
from ...samplers.diffusion import DiffusionSampler, DiffusionScheduler


@WorldModelRegistry.register("diffusion", DiffusionWorldModelConfig)
class DiffusionWorldModel(WorldModel):
    """Minimal diffusion-style world model."""

    def __init__(self, config: DiffusionWorldModelConfig):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.DIFFUSION,
            Capability.OBS_DECODER,
        }

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        input_dim = obs_dim + config.action_dim

        self.denoise_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, obs_dim),
        )
        self.scheduler = DiffusionScheduler(
            num_train_steps=max(10, config.diffusion_steps * 25),
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            prediction_target=config.prediction_target,
        )
        self.sampler = DiffusionSampler(self.scheduler)

    def io_contract(self) -> ModelIOContract:
        obs_dim = int(torch.prod(torch.tensor(self.config.obs_shape)).item())
        obs_kind = ModalityKind.IMAGE if len(self.config.obs_shape) == 3 else ModalityKind.VECTOR
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={"obs": ModalitySpec(kind=obs_kind, shape=self.config.obs_shape)}
            ),
            action_spec=ActionSpec(
                kind=self.config.action_type,
                dim=self.config.action_dim,
                discrete=self.config.action_type == "discrete",
                num_actions=self.config.action_dim
                if self.config.action_type == "discrete"
                else None,
            ),
            state_spec=StateSpec(
                tensors={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(obs_dim,))}
            ),
            prediction_spec=PredictionSpec(
                tensors={"obs": ModalitySpec(kind=ModalityKind.VECTOR, shape=(obs_dim,))}
            ),
            sequence_layout=SequenceLayout(
                axes_by_field={
                    "obs": "B...",
                    "actions": "B...",
                    "target": "B...",
                    "next_obs": "B...",
                }
            ),
            required_batch_keys=("obs",),
            required_state_keys=("obs",),
        )

    def _extract_obs(self, obs: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(obs, dict):
            if "obs" in obs:
                return obs["obs"]
            raise ValueError("Diffusion model expects 'obs' in observation dict")
        return obs

    def _flatten_obs(self, obs: Tensor | dict[str, Tensor]) -> Tensor:
        tensor = self._extract_obs(obs)
        if tensor.dim() == 2:
            return tensor
        if tensor.dim() >= 3:
            return tensor.view(tensor.shape[0], -1)
        raise ValueError("Diffusion model expects obs with batch dimension")

    def _action_or_zeros(self, obs: Tensor, action: Tensor | None) -> Tensor:
        if action is None:
            return torch.zeros(obs.shape[0], self.config.action_dim, device=obs.device)
        return action

    def denoise(
        self,
        x: Tensor,
        action: Tensor | None = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        action_t = self._action_or_zeros(x, action)
        inp = torch.cat([x, action_t], dim=-1)
        return self.denoise_net(inp)

    def encode(self, obs: Tensor | dict[str, Tensor], deterministic: bool = False) -> State:
        flat = self._flatten_obs(obs)
        return State(tensors={"obs": flat})

    def transition(self, state: State, action: Tensor, deterministic: bool = False) -> State:
        obs = state.tensors["obs"]
        start_timestep = max(self.config.diffusion_steps - 1, 0)
        next_obs = self.sampler.sample(
            self, obs, action, steps=self.config.diffusion_steps, start_timestep=start_timestep
        )
        timestep = torch.full((obs.shape[0],), start_timestep, device=obs.device, dtype=torch.long)
        return State(tensors={"obs": next_obs}, meta={"timestep": timestep})

    def update(self, state: State, action: Tensor, obs: Tensor | dict[str, Tensor]) -> State:
        return self.encode(obs)

    def decode(self, state: State) -> ModelOutput:
        return ModelOutput(preds={"obs": state.tensors["obs"]})

    def loss(self, batch: Batch) -> LossOutput:
        obs = self._flatten_obs(batch.obs)
        if batch.target is not None:
            target = self._flatten_obs(batch.target)
        elif batch.next_obs is not None:
            target = self._flatten_obs(batch.next_obs)
        else:
            target = obs

        action = batch.actions
        action_t = self._action_or_zeros(
            obs, action[:, 0] if action is not None and action.dim() > 2 else action
        )

        noise = torch.randn_like(obs)
        timesteps = self.scheduler.sample_timesteps(obs.shape[0], device=obs.device)
        noisy = self.scheduler.add_noise(obs, noise, timesteps)
        pred_noise = self.denoise(noisy, action_t, timestep=timesteps)
        denoised = self.scheduler.step(pred_noise, noisy, timesteps)

        recon_loss = F.mse_loss(denoised, target)
        noise_loss = F.mse_loss(pred_noise, noise)
        loss = 0.5 * (recon_loss + noise_loss)
        return LossOutput(loss=loss, components={"diffusion_mse": loss})
