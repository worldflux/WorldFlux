"""Tests for token sampler utilities."""

from __future__ import annotations

import torch

from worldflux.samplers.token import TokenSampler


def test_token_sampler_temperature_non_positive_returns_argmax() -> None:
    sampler = TokenSampler()
    logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, -1.0, 0.0]])

    sampled = sampler.sample(logits, temperature=0.0)
    expected = logits.argmax(dim=-1)
    torch.testing.assert_close(sampled, expected)


def test_token_sampler_temperature_positive_samples_valid_indices() -> None:
    torch.manual_seed(7)
    sampler = TokenSampler()
    logits = torch.randn(8, 5)

    sampled = sampler.sample(logits, temperature=0.7)
    assert sampled.shape == (8,)
    assert sampled.dtype == torch.int64
    assert int(sampled.min().item()) >= 0
    assert int(sampled.max().item()) < 5
