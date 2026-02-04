"""Integration tests for reference model families."""

import torch

from worldflux.core.batch import Batch
from worldflux.core.config import DiffusionWorldModelConfig, TokenWorldModelConfig
from worldflux.models.diffusion import DiffusionWorldModel
from worldflux.models.token import TokenWorldModel
from worldflux.training import Trainer, TrainingConfig


class RandomTokenProvider:
    def __init__(self, vocab_size: int, seq_len: int):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ):
        length = seq_len or self.seq_len
        tokens = torch.randint(self.vocab_size, (batch_size, length), device=device)
        target = torch.randint(self.vocab_size, (batch_size, length), device=device)
        return Batch(obs=tokens, target=target)


class RandomDiffusionProvider:
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ):
        obs = torch.randn(batch_size, self.obs_dim, device=device)
        target = torch.randn(batch_size, self.obs_dim, device=device)
        actions = torch.randn(batch_size, self.action_dim, device=device)
        return Batch(obs=obs, actions=actions, target=target)


def test_token_training_short():
    config = TokenWorldModelConfig(
        obs_shape=(8,),
        action_dim=1,
        vocab_size=64,
        token_dim=32,
        num_layers=1,
        num_heads=1,
    )
    model = TokenWorldModel(config)
    trainer = Trainer(model, TrainingConfig(total_steps=3, batch_size=4, sequence_length=8))
    provider = RandomTokenProvider(64, 8)
    trainer.train(provider)


def test_diffusion_training_short():
    config = DiffusionWorldModelConfig(
        obs_shape=(4,), action_dim=2, hidden_dim=32, diffusion_steps=2
    )
    model = DiffusionWorldModel(config)
    trainer = Trainer(model, TrainingConfig(total_steps=3, batch_size=4, sequence_length=1))
    provider = RandomDiffusionProvider(4, 2)
    trainer.train(provider)
