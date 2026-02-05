"""Minimal JEPA-style world model skeleton."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import JEPABaseConfig
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


@WorldModelRegistry.register("jepa", JEPABaseConfig)
class JEPABaseWorldModel(WorldModel):
    """JEPA-style representation predictor (minimal skeleton)."""

    def __init__(self, config: JEPABaseConfig):
        super().__init__()
        self.config = config
        self.capabilities = {Capability.REPRESENTATION}

        input_dim = (
            config.obs_shape[0]
            if len(config.obs_shape) == 1
            else int(torch.prod(torch.tensor(config.obs_shape)).item())
        )

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, config.encoder_dim),
            nn.LayerNorm(config.encoder_dim),
            nn.GELU(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.encoder_dim, config.predictor_dim),
            nn.GELU(),
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

    def _flatten_obs(self, obs: Tensor) -> Tensor:
        if obs.dim() > 2:
            return obs.view(obs.shape[0], -1)
        return obs

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        obs_tensor: Tensor | None
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        if isinstance(obs, dict):
            obs_tensor = obs.get("obs")
            if obs_tensor is None:
                obs_tensor = obs.get("context")
        else:
            obs_tensor = obs
        if obs_tensor is None:
            raise ValueError("JEPA requires obs tensor or dict with 'obs'/'context'")
        obs_flat = self._flatten_obs(obs_tensor)
        rep = self.encoder(obs_flat)
        return State(tensors={"rep": rep})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del action, conditions, deterministic
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
            raise ValueError("JEPA state requires 'rep'")
        proj = self.projector(rep)
        return ModelOutput(predictions={"representation": proj})

    def loss(self, batch: Batch) -> LossOutput:
        context_raw = batch.context if batch.context is not None else batch.obs
        target_raw = batch.target if batch.target is not None else batch.next_obs or batch.obs

        context: Tensor | None
        target: Tensor | None

        if isinstance(context_raw, dict):
            context = context_raw.get("context")
            if context is None:
                context = context_raw.get("obs")
        else:
            context = context_raw

        if isinstance(target_raw, dict):
            target = target_raw.get("target")
            if target is None:
                target = target_raw.get("obs")
        else:
            target = target_raw

        if context is None or target is None:
            raise ValueError("JEPA requires context and target tensors in batch")

        context_rep = self.encoder(self._flatten_obs(context))
        pred = self.predictor(context_rep)

        with torch.no_grad():
            target_rep = self.projector(self.encoder(self._flatten_obs(target)))

        if batch.mask is not None:
            mask = batch.mask
            if mask.shape[0] != pred.shape[0]:
                raise ValueError(
                    f"mask shape mismatch: batch dimension {mask.shape[0]} != {pred.shape[0]}"
                )
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)
            if mask.dim() != 2:
                raise ValueError(f"mask shape mismatch: expected rank-2 mask, got {mask.shape}")
            if mask.shape[1] not in (1, pred.shape[1]):
                raise ValueError(
                    f"mask shape mismatch: second dimension must be 1 or {pred.shape[1]}, "
                    f"got {mask.shape[1]}"
                )
            pred = pred * mask
            target_rep = target_rep * mask

        loss = F.mse_loss(pred, target_rep)
        components = {"jepa": loss}
        return LossOutput(loss=loss, components=components, metrics={"jepa": loss.item()})
