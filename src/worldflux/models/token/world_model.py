"""Minimal token-based world model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ...core.batch import Batch
from ...core.config import TokenWorldModelConfig
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
from ...samplers.token import TokenSampler


class TokenActionConditioner(nn.Module):
    """Explicit token action-conditioning module."""

    def __init__(self, token_dim: int, action_dim: int):
        super().__init__()
        self.projector = nn.Linear(max(1, action_dim), token_dim)

    def _match_in_features(self, tensor: Tensor) -> Tensor:
        in_features = self.projector.in_features
        last_dim = tensor.shape[-1]
        if last_dim == in_features:
            return tensor
        if last_dim > in_features:
            return tensor[..., :in_features]
        pad_shape = (*tensor.shape[:-1], in_features - last_dim)
        pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=-1)

    def _tensor_from_payload(self, action: ActionPayload | Tensor | None) -> Tensor | None:
        if action is None:
            return None
        if isinstance(action, Tensor):
            return action
        if action.tokens is not None:
            return action.tokens
        if action.tensor is not None:
            return action.tensor
        if action.latent is not None:
            return action.latent
        return None

    def condition(
        self, action: ActionPayload | Tensor | None, seq_len: int, device: torch.device
    ) -> Tensor | None:
        tensor = self._tensor_from_payload(action)
        if tensor is None:
            return None

        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)

        if tensor.dim() == 2:
            base_in = self._match_in_features(tensor.to(device=device, dtype=torch.float32))
            base = self.projector(base_in)
            return base.unsqueeze(1).expand(-1, seq_len, -1)

        # [B, T, ...] -> flatten tail dims.
        if tensor.dim() >= 3:
            lead = tensor.shape[:2]
            flat = tensor.reshape(lead[0], lead[1], -1).to(device=device, dtype=torch.float32)
            cond = self.projector(self._match_in_features(flat))
            if cond.shape[1] == seq_len:
                return cond
            if cond.shape[1] == 1:
                return cond.expand(-1, seq_len, -1)
            raise ValueError(
                f"Action sequence length mismatch for token conditioning: got {cond.shape[1]}, expected {seq_len}"
            )

        return None


@WorldModelRegistry.register("token", TokenWorldModelConfig)
class TokenWorldModel(WorldModel):
    """Minimal token-based world model with transformer dynamics."""

    def __init__(self, config: TokenWorldModelConfig):
        super().__init__()
        self.config = config
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.TOKEN_MODEL,
            Capability.REPRESENTATION,
        }

        self.embedding = nn.Embedding(config.vocab_size, config.token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.token_dim,
            nhead=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.logit_head = nn.Linear(config.token_dim, config.vocab_size)

        obs_dim = int(torch.prod(torch.tensor(config.obs_shape)).item())
        self.tokenizer = nn.Linear(obs_dim, config.vocab_size)
        self.sampler = TokenSampler()
        self.token_action_conditioner = TokenActionConditioner(config.token_dim, config.action_dim)

    def io_contract(self) -> ModelIOContract:
        token_shape = self.config.obs_shape if self.config.obs_shape else (1,)
        return ModelIOContract(
            observation_spec=ObservationSpec(
                modalities={"tokens": ModalitySpec(kind=ModalityKind.TOKENS, shape=token_shape)}
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
                    "tokens": ModalitySpec(kind=ModalityKind.TOKENS, shape=token_shape),
                    "emb": ModalitySpec(kind=ModalityKind.VECTOR, shape=(self.config.token_dim,)),
                }
            ),
            prediction_spec=PredictionSpec(
                tensors={
                    "logits": ModalitySpec(
                        kind=ModalityKind.VECTOR, shape=(self.config.vocab_size,)
                    ),
                    "tokens": ModalitySpec(kind=ModalityKind.TOKENS, shape=token_shape),
                }
            ),
            sequence_layout=SequenceLayout(
                axes_by_field={
                    "obs": "BT",
                    "target": "BT",
                    "next_obs": "BT",
                    "mask": "BT",
                }
            ),
            required_batch_keys=("obs",),
            required_state_keys=("tokens",),
        )

    def _extract_obs(self, obs: Tensor | dict[str, Tensor]) -> Tensor:
        if isinstance(obs, dict):
            if "tokens" in obs:
                return obs["tokens"]
            if "obs" in obs:
                return obs["obs"]
            raise ValueError("Token model expects 'tokens' or 'obs' in observation dict")
        return obs

    def _ensure_token_shape(self, tokens: Tensor) -> Tensor:
        if tokens.dim() == 1:
            return tokens.unsqueeze(1)
        if tokens.dim() == 3 and tokens.shape[-1] == 1:
            return tokens.squeeze(-1)
        return tokens

    def _tokenize(self, obs: Tensor | dict[str, Tensor]) -> Tensor:
        obs_tensor = self._extract_obs(obs)
        if obs_tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            tokens = obs_tensor
        else:
            if obs_tensor.dim() == 2:
                flat = obs_tensor
            elif obs_tensor.dim() >= 3:
                flat = obs_tensor.view(obs_tensor.shape[0], obs_tensor.shape[1], -1)
            else:
                raise ValueError("Token model expects obs with batch dimension")
            logits = self.tokenizer(flat)
            tokens = logits.argmax(dim=-1)
        tokens = self._ensure_token_shape(tokens)
        return tokens.long()

    def _forward_tokens(
        self, tokens: Tensor, action: ActionPayload | Tensor | None = None
    ) -> Tensor:
        emb = self.embedding(tokens)
        action_bias = self.token_action_conditioner.condition(
            action,
            seq_len=tokens.shape[1],
            device=tokens.device,
        )
        if action_bias is not None:
            emb = emb + action_bias
        hidden = self.transformer(emb)
        return self.logit_head(hidden)

    def encode(
        self,
        obs: Tensor | dict[str, Tensor] | WorldModelInput,
        deterministic: bool = False,
    ) -> State:
        del deterministic
        if isinstance(obs, WorldModelInput):
            obs = obs.observations
        tokens = self._tokenize(obs)
        emb = self.embedding(tokens)
        return State(tensors={"tokens": tokens, "emb": emb})

    def transition(
        self,
        state: State,
        action: ActionPayload | Tensor | None,
        conditions: ConditionPayload | None = None,
        deterministic: bool = False,
    ) -> State:
        del conditions
        tokens = state.tensors["tokens"]
        logits = self._forward_tokens(tokens, action=action)
        next_tokens = logits.argmax(dim=-1) if deterministic else self.sampler.sample(logits)
        next_emb = self.embedding(next_tokens)
        return State(tensors={"tokens": next_tokens, "emb": next_emb})

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
        emb = state.tensors.get("emb")
        if emb is None:
            tokens = state.tensors["tokens"]
            emb = self.embedding(tokens)
        logits = self.logit_head(emb)
        preds: dict[str, Tensor] = {"logits": logits}
        if "tokens" in state.tensors:
            preds["tokens"] = state.tensors["tokens"]
        return ModelOutput(predictions=preds)

    def loss(self, batch: Batch) -> LossOutput:
        if batch.obs is None:
            raise ValueError("Token loss requires obs")
        obs_has_tokens = isinstance(batch.obs, dict) and "tokens" in batch.obs
        tokens = self._tokenize(batch.obs)
        if batch.target is not None:
            target = batch.target
        elif batch.next_obs is not None:
            target = batch.next_obs
        elif batch.obs is not None:
            target = batch.obs
        else:
            raise ValueError("Token loss requires target, next_obs, or obs")
        target_has_tokens = isinstance(target, dict) and "tokens" in target
        if obs_has_tokens and not target_has_tokens:
            raise ValueError("Token loss requires target tokens when obs contains 'tokens'")
        target_tokens = self._tokenize(target)

        if tokens.shape != target_tokens.shape:
            raise ValueError(
                f"Token loss requires matching shapes, got {tokens.shape} vs {target_tokens.shape}"
            )

        action_payload: ActionPayload | Tensor | None = None
        if batch.actions is not None:
            action_payload = batch.actions
        logits = self._forward_tokens(tokens, action=action_payload)
        vocab = logits.shape[-1]
        loss = F.cross_entropy(logits.view(-1, vocab), target_tokens.view(-1))
        return LossOutput(loss=loss, components={"token_ce": loss})
