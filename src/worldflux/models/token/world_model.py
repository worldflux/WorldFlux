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
from ...core.registry import WorldModelRegistry
from ...core.spec import Capability
from ...core.state import State
from ...samplers.token import TokenSampler


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

    def _forward_tokens(self, tokens: Tensor) -> Tensor:
        emb = self.embedding(tokens)
        hidden = self.transformer(emb)
        return self.logit_head(hidden)

    def encode(self, obs: Tensor | dict[str, Tensor], deterministic: bool = False) -> State:
        tokens = self._tokenize(obs)
        emb = self.embedding(tokens)
        return State(tensors={"tokens": tokens, "emb": emb})

    def transition(self, state: State, action: Tensor, deterministic: bool = False) -> State:
        tokens = state.tensors["tokens"]
        logits = self._forward_tokens(tokens)
        next_tokens = logits.argmax(dim=-1) if deterministic else self.sampler.sample(logits)
        next_emb = self.embedding(next_tokens)
        return State(tensors={"tokens": next_tokens, "emb": next_emb})

    def update(self, state: State, action: Tensor, obs: Tensor | dict[str, Tensor]) -> State:
        return self.encode(obs)

    def decode(self, state: State) -> ModelOutput:
        emb = state.tensors.get("emb")
        if emb is None:
            tokens = state.tensors["tokens"]
            emb = self.embedding(tokens)
        logits = self.logit_head(emb)
        preds = {"logits": logits}
        if "tokens" in state.tensors:
            preds["tokens"] = state.tensors["tokens"]
        return ModelOutput(preds=preds)

    def loss(self, batch: Batch) -> LossOutput:
        tokens = self._tokenize(batch.obs)
        if batch.target is not None:
            target = batch.target
        elif batch.next_obs is not None:
            target = batch.next_obs
        else:
            target = batch.obs
        target_tokens = self._tokenize(target)

        if tokens.shape != target_tokens.shape:
            raise ValueError(
                f"Token loss requires matching shapes, got {tokens.shape} vs {target_tokens.shape}"
            )

        logits = self._forward_tokens(tokens)
        vocab = logits.shape[-1]
        loss = F.cross_entropy(logits.view(-1, vocab), target_tokens.view(-1))
        return LossOutput(loss=loss, components={"token_ce": loss})
