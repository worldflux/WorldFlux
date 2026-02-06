"""V-JEPA2-style predictive representation world model."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import VJEPA2Config
from ...core.model import WorldModel
from ...core.output import LossOutput, ModelOutput
from ...core.payloads import ActionPayload, ConditionPayload, WorldModelInput
from ...core.registry import WorldModelRegistry
from ...core.spec import (
    ActionSpec,
    Capability,
    ConditionSpec,
    ModalityKind,
    ModalitySpec,
    ModelIOContract,
    ObservationSpec,
    PredictionSpec,
    SequenceLayout,
    StateSpec,
)
from ...core.state import State


@WorldModelRegistry.register("vjepa2", VJEPA2Config)
class VJEPA2WorldModel(WorldModel):
    """Minimal V-JEPA2-style model focused on context-target representation learning."""

    def __init__(self, config: VJEPA2Config):
        super().__init__()
        self.config = config
        self.capabilities = {Capability.REPRESENTATION}

        self._obs_rank = len(config.obs_shape)
        self._obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())

        self.encoder = nn.Sequential(
            nn.Linear(self._obs_dim, config.encoder_dim),
            nn.LayerNorm(config.encoder_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.encoder_dim, config.predictor_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_dim, config.projection_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(config.encoder_dim, config.projection_dim),
            nn.GELU(),
        )
        self.action_predictor = nn.Linear(max(1, config.action_dim), config.projection_dim)
        self.action_conditioner_mode = "none"

    def io_contract(self) -> ModelIOContract:
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
                tensors={
                    "rep": ModalitySpec(kind=ModalityKind.VECTOR, shape=(self.config.encoder_dim,))
                }
            ),
            prediction_spec=PredictionSpec(
                tensors={
                    "representation": ModalitySpec(
                        kind=ModalityKind.VECTOR,
                        shape=(self.config.projection_dim,),
                    )
                }
            ),
            condition_spec=ConditionSpec(allowed_extra_keys=()),
            sequence_layout=SequenceLayout(
                axes_by_field={
                    "obs": "B...",
                    "context": "B...",
                    "target": "B...",
                    "mask": "B...",
                }
            ),
            required_batch_keys=("obs",),
            required_state_keys=("rep",),
        )

    def _extract_tensor(
        self,
        value: Tensor | dict[str, Tensor],
        *,
        keys: tuple[str, ...],
        error_label: str,
    ) -> Tensor:
        if isinstance(value, dict):
            for key in keys:
                tensor = value.get(key)
                if tensor is not None:
                    return tensor
            raise ValueError(f"V-JEPA2 expects one of {keys} in '{error_label}' dict")
        return value

    def _flatten_observation(self, obs: Tensor) -> tuple[Tensor, tuple[int, ...]]:
        if self._obs_rank == 0:
            raise ValueError("V-JEPA2 requires non-empty obs_shape")
        if obs.dim() < self._obs_rank + 1:
            raise ValueError(
                f"V-JEPA2 expects rank >= {self._obs_rank + 1}, got {tuple(obs.shape)}"
            )

        feature_shape = tuple(obs.shape[-self._obs_rank :])
        if feature_shape != self.config.obs_shape:
            raise ValueError(
                f"V-JEPA2 expected trailing observation shape {self.config.obs_shape}, got {feature_shape}"
            )

        lead_shape = tuple(obs.shape[: -self._obs_rank])
        if not lead_shape:
            raise ValueError("V-JEPA2 expects at least one leading batch axis")

        return obs.reshape(-1, self._obs_dim), lead_shape

    @staticmethod
    def _mask_to_flat(mask: Tensor, lead_shape: tuple[int, ...], width: int) -> Tensor:
        candidates = {
            lead_shape,
            (*lead_shape, 1),
            (*lead_shape, width),
        }
        if tuple(mask.shape) not in candidates:
            raise ValueError(
                f"mask shape mismatch: expected one of {sorted(candidates)}, got {tuple(mask.shape)}"
            )

        if tuple(mask.shape) == (*lead_shape, width):
            flat = mask.reshape(-1, width)
        else:
            flat = mask.reshape(-1, 1)
        return flat.to(dtype=torch.float32)

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        obs_tensor = self._extract_tensor(obs, keys=("obs", "context"), error_label="obs")
        flat, _ = self._flatten_observation(obs_tensor)
        rep = self.encoder(flat)
        return State(tensors={"rep": rep})

    def encode_representation(self, obs: Tensor | dict[str, Tensor]) -> Tensor:
        """Encode context observations into representation vectors."""
        return self.encode(obs).tensors["rep"]

    def set_encoder_trainable(self, trainable: bool) -> None:
        """Freeze/unfreeze encoder parameters for staged control integration."""
        for param in self.encoder.parameters():
            param.requires_grad = trainable

    def set_action_conditioner(self, mode: str) -> None:
        """Switch action conditioning mode for representation prediction."""
        if mode not in {"none", "latent"}:
            raise ValueError(f"Unsupported action conditioner mode: {mode}")
        self.action_conditioner_mode = mode

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        # Action-conditioned transitions are intentionally deferred for this experimental release.
        del action, deterministic
        self._validate_condition_payload(self.coerce_condition_payload(conditions))
        return state

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
        rep = state.tensors.get("rep")
        if rep is None:
            raise ValueError("V-JEPA2 state requires 'rep'")
        projected = self.projector(rep)
        return ModelOutput(predictions={"representation": projected})

    @staticmethod
    def _flatten_action_like(action: Tensor, lead_shape: tuple[int, ...]) -> Tensor:
        if tuple(action.shape[: len(lead_shape)]) != lead_shape:
            raise ValueError(
                f"latent action lead-shape mismatch: expected {lead_shape}, got {tuple(action.shape)}"
            )
        tail = action.shape[len(lead_shape) :]
        if not tail:
            return action.reshape(-1, 1).to(dtype=torch.float32)
        feat_dim = int(torch.prod(torch.tensor(tail)).item())
        return action.reshape(-1, feat_dim).to(dtype=torch.float32)

    def _match_action_features(self, action: Tensor) -> Tensor:
        in_features = self.action_predictor.in_features
        last_dim = action.shape[-1]
        if last_dim == in_features:
            return action
        if last_dim > in_features:
            return action[..., :in_features]
        pad_shape = (*action.shape[:-1], in_features - last_dim)
        pad = torch.zeros(pad_shape, device=action.device, dtype=action.dtype)
        return torch.cat([action, pad], dim=-1)

    def loss(self, batch: Batch) -> LossOutput:
        context_raw = batch.context if batch.context is not None else batch.obs
        if context_raw is None:
            raise ValueError("V-JEPA2 loss requires context or obs")

        if batch.target is not None:
            target_raw = batch.target
        elif batch.next_obs is not None:
            target_raw = batch.next_obs
        elif batch.obs is not None:
            target_raw = batch.obs
        else:
            raise ValueError("V-JEPA2 loss requires target, next_obs, or obs")

        context_tensor = self._extract_tensor(
            context_raw,
            keys=("context", "obs"),
            error_label="context",
        )
        target_tensor = self._extract_tensor(
            target_raw,
            keys=("target", "obs", "context"),
            error_label="target",
        )

        context_flat, context_lead = self._flatten_observation(context_tensor)
        target_flat, target_lead = self._flatten_observation(target_tensor)

        if context_lead != target_lead:
            raise ValueError(f"context/target lead-shape mismatch: {context_lead} vs {target_lead}")

        context_rep = self.encoder(context_flat)
        pred = self.predictor(context_rep)

        with torch.no_grad():
            target_rep = self.projector(self.encoder(target_flat))

        if self.action_conditioner_mode == "latent":
            latent_action = None
            if batch.actions is not None:
                latent_action = batch.actions
            elif "latent_action" in batch.conditions and isinstance(
                batch.conditions["latent_action"], Tensor
            ):
                latent_action = batch.conditions["latent_action"]
            if latent_action is not None:
                action_flat = self._flatten_action_like(latent_action, context_lead).to(
                    device=pred.device
                )
                pred = pred + self.action_predictor(self._match_action_features(action_flat))

        if batch.mask is not None:
            mask = self._mask_to_flat(batch.mask, context_lead, pred.shape[1]).to(
                device=pred.device
            )
            pred = pred * mask
            target_rep = target_rep * mask

        loss = F.mse_loss(pred, target_rep)
        components = {"vjepa2": loss}
        return LossOutput(loss=loss, components=components, metrics={"vjepa2": loss.item()})
