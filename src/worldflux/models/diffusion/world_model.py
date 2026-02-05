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
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
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
        self.action_embed = nn.Linear(config.action_dim, config.hidden_dim)
        self.ada_scale = nn.Linear(config.hidden_dim, obs_dim)
        self.ada_shift = nn.Linear(config.hidden_dim, obs_dim)
        self.action_conditioner_kind = config.action_conditioner
        self.scheduler = DiffusionScheduler(
            num_train_steps=max(10, config.diffusion_steps * 25),
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            prediction_target=config.prediction_target,
        )
        self.sampler = DiffusionSampler(self.scheduler)
        self._obs_rank = len(config.obs_shape)
        self._obs_dim = obs_dim

    def set_action_conditioner(self, kind: str) -> None:
        if kind not in {"none", "adaln", "adagn"}:
            raise ValueError(f"Unsupported action conditioner kind: {kind}")
        self.action_conditioner_kind = kind

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
        if self._obs_rank == 0:
            raise ValueError("Diffusion model requires non-empty obs_shape")
        if tensor.dim() < self._obs_rank + 1:
            raise ValueError(
                f"Diffusion model expects rank >= {self._obs_rank + 1}, got {tuple(tensor.shape)}"
            )
        feature_shape = tuple(tensor.shape[-self._obs_rank :])
        if feature_shape != self.config.obs_shape:
            raise ValueError(
                f"Diffusion model expected trailing shape {self.config.obs_shape}, got {feature_shape}"
            )

        if tensor.dim() == self._obs_rank + 1:
            return tensor.reshape(tensor.shape[0], self._obs_dim)

        if tensor.dim() == self._obs_rank + 2:
            return tensor[:, 0].reshape(tensor.shape[0], self._obs_dim)

        lead = int(torch.prod(torch.tensor(tensor.shape[: -self._obs_rank])).item())
        return tensor.reshape(lead, self._obs_dim)

    def _match_action_dim(self, action: Tensor) -> Tensor:
        in_features = self.config.action_dim
        last_dim = action.shape[-1]
        if last_dim == in_features:
            return action
        if last_dim > in_features:
            return action[..., :in_features]
        pad_shape = (*action.shape[:-1], in_features - last_dim)
        pad = torch.zeros(pad_shape, device=action.device, dtype=action.dtype)
        return torch.cat([action, pad], dim=-1)

    def _select_batch_action(self, action: Tensor) -> Tensor:
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        if action.dim() == 2:
            return action
        if action.dim() >= 3:
            return action[:, 0].reshape(action.shape[0], -1)
        raise ValueError("Diffusion model expects obs with batch dimension")

    def _action_or_zeros(self, obs: Tensor, action: Tensor | None) -> Tensor:
        if action is None:
            return torch.zeros(obs.shape[0], self.config.action_dim, device=obs.device)
        action_2d = self._select_batch_action(action)
        return self._match_action_dim(action_2d)

    def _apply_action_conditioning(self, denoised: Tensor, action_t: Tensor) -> Tensor:
        if self.action_conditioner_kind == "none":
            return denoised

        hidden = torch.tanh(self.action_embed(action_t))
        scale = self.ada_scale(hidden)
        shift = self.ada_shift(hidden)

        if self.action_conditioner_kind == "adaln":
            normed = F.layer_norm(denoised, denoised.shape[-1:])
            return normed * (1.0 + scale) + shift

        # adagn approximation in vectorized latent space.
        channels = denoised.shape[-1]
        groups = 1
        for candidate in (8, 4, 2):
            if channels % candidate == 0:
                groups = candidate
                break
        normed = F.group_norm(denoised.unsqueeze(-1), num_groups=groups).squeeze(-1)
        return normed * (1.0 + scale) + shift

    def denoise(
        self,
        x: Tensor,
        action: ActionPayload | Tensor | None = None,
        timestep: Tensor | None = None,
    ) -> Tensor:
        del timestep
        action_tensor = self.action_tensor_or_none(action)
        action_t = self._action_or_zeros(x, action_tensor)
        inp = torch.cat([x, action_t], dim=-1)
        denoised = self.denoise_net(inp)
        return self._apply_action_conditioning(denoised, action_t)

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        flat = self._flatten_obs(obs)
        return State(tensors={"obs": flat})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del conditions, deterministic
        obs = state.tensors["obs"]
        action_tensor = self.action_tensor_or_none(action)
        start_timestep = max(self.config.diffusion_steps - 1, 0)
        next_obs = self.sampler.sample(
            self,
            obs,
            action_tensor,
            steps=self.config.diffusion_steps,
            start_timestep=start_timestep,
        )
        timestep = torch.full((obs.shape[0],), start_timestep, device=obs.device, dtype=torch.long)
        return State(tensors={"obs": next_obs}, meta={"timestep": timestep})

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
        return ModelOutput(predictions={"obs": state.tensors["obs"]})

    def loss(self, batch: Batch) -> LossOutput:
        if batch.obs is None:
            raise ValueError("Diffusion loss requires obs")
        obs = self._flatten_obs(batch.obs)
        if batch.target is not None:
            target = self._flatten_obs(batch.target)
        elif batch.next_obs is not None:
            target = self._flatten_obs(batch.next_obs)
        else:
            target = obs

        action = batch.actions
        action_tensor = self._select_batch_action(action) if action is not None else None
        action_t = self._action_or_zeros(obs, action_tensor)

        noise = torch.randn_like(obs)
        timesteps = self.scheduler.sample_timesteps(obs.shape[0], device=obs.device)
        noisy = self.scheduler.add_noise(obs, noise, timesteps)
        pred_noise = self.denoise(noisy, action_t, timestep=timesteps)
        denoised = self.scheduler.step(pred_noise, noisy, timesteps)

        recon_loss = F.mse_loss(denoised, target)
        noise_loss = F.mse_loss(pred_noise, noise)
        loss = 0.5 * (recon_loss + noise_loss)
        return LossOutput(loss=loss, components={"diffusion_mse": loss})
