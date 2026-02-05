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

    def encode(self, obs: Tensor | dict[str, Tensor], deterministic: bool = False) -> State:
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

    def transition(self, state: State, action: Tensor, deterministic: bool = False) -> State:
        # Action-conditioned transitions are intentionally deferred for this experimental release.
        return state

    def update(self, state: State, action: Tensor, obs: Tensor | dict[str, Tensor]) -> State:
        return self.encode(obs)

    def decode(self, state: State) -> ModelOutput:
        rep = state.tensors.get("rep")
        if rep is None:
            raise ValueError("V-JEPA2 state requires 'rep'")
        projected = self.projector(rep)
        return ModelOutput(preds={"representation": projected})

    def loss(self, batch: Batch) -> LossOutput:
        context_raw = batch.context if batch.context is not None else batch.obs
        target_raw = batch.target if batch.target is not None else batch.next_obs or batch.obs

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

        if batch.mask is not None:
            mask = self._mask_to_flat(batch.mask, context_lead, pred.shape[1]).to(
                device=pred.device
            )
            pred = pred * mask
            target_rep = target_rep * mask

        loss = F.mse_loss(pred, target_rep)
        components = {"vjepa2": loss}
        return LossOutput(loss=loss, components=components, metrics={"vjepa2": loss.item()})
