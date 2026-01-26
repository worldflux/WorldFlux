"""Tests for latent space implementations."""

import pytest
import torch

from worldmodels.core.latent_space import (
    GaussianLatentSpace,
    CategoricalLatentSpace,
    SimNormLatentSpace,
)


class TestGaussianLatentSpace:
    """Gaussian latent space tests."""
    
    def test_sample_shape(self):
        space = GaussianLatentSpace(dim=256)
        params = torch.randn(4, 512)  # mean + std
        sample = space.sample(params)
        assert sample.shape == (4, 256)
    
    def test_deterministic_returns_mean(self):
        space = GaussianLatentSpace(dim=256)
        params = torch.randn(4, 512)
        mean = params[:, :256]
        sample = space.sample(params, deterministic=True)
        assert torch.allclose(sample, mean)
    
    def test_kl_positive(self):
        space = GaussianLatentSpace(dim=256)
        posterior = torch.randn(4, 512)
        prior = torch.randn(4, 512)
        kl = space.kl_divergence(posterior, prior)
        assert kl.shape == (4,)
        assert (kl >= 0).all()
    
    def test_kl_same_distribution_near_zero(self):
        space = GaussianLatentSpace(dim=256)
        params = torch.randn(4, 512)
        kl = space.kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros(4), atol=1e-5)


class TestCategoricalLatentSpace:
    """Categorical latent space tests."""
    
    def test_sample_shape(self):
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        logits = torch.randn(4, 32, 32)
        sample = space.sample(logits)
        assert sample.shape == (4, 32, 32)
    
    def test_sample_from_flat(self):
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        logits = torch.randn(4, 32 * 32)
        sample = space.sample(logits)
        assert sample.shape == (4, 32, 32)
    
    def test_one_hot(self):
        """Each categorical is one-hot."""
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        logits = torch.randn(4, 32, 32)
        sample = space.sample(logits)
        assert torch.allclose(sample.sum(dim=-1), torch.ones(4, 32))
    
    def test_deterministic_argmax(self):
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        logits = torch.randn(4, 32, 32)
        sample = space.sample(logits, deterministic=True)
        
        # Should be argmax
        expected_indices = logits.argmax(dim=-1)
        actual_indices = sample.argmax(dim=-1)
        assert torch.equal(expected_indices, actual_indices)
    
    def test_straight_through_gradient(self):
        """Straight-through gradient flows."""
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        logits = torch.randn(4, 32, 32, requires_grad=True)
        sample = space.sample(logits)
        loss = sample.sum()
        loss.backward()
        assert logits.grad is not None
    
    def test_kl_positive(self):
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        posterior = torch.randn(4, 32, 32)
        prior = torch.randn(4, 32, 32)
        kl = space.kl_divergence(posterior, prior)
        assert kl.shape == (4,)
        assert (kl >= 0).all()
    
    def test_kl_with_free_nats(self):
        space = CategoricalLatentSpace(num_categoricals=32, num_classes=32)
        posterior = torch.randn(4, 32, 32)
        prior = torch.randn(4, 32, 32)
        
        kl = space.kl_divergence(posterior, prior, free_nats=0.0)
        kl_free = space.kl_divergence(posterior, prior, free_nats=10.0)
        
        # Free nats should reduce KL
        assert (kl_free <= kl).all()


class TestSimNormLatentSpace:
    """SimNorm latent space tests."""
    
    def test_simplex_constraint(self):
        space = SimNormLatentSpace(dim=256, simnorm_dim=8)
        params = torch.randn(4, 256)
        output = space.sample(params)
        
        reshaped = output.view(4, 32, 8)
        sums = reshaped.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4, 32), atol=1e-6)
    
    def test_deterministic(self):
        """SimNorm is deterministic transform."""
        space = SimNormLatentSpace(dim=256, simnorm_dim=8)
        params = torch.randn(4, 256)
        out1 = space.sample(params)
        out2 = space.sample(params)
        assert torch.allclose(out1, out2)
    
    def test_kl_zero(self):
        """SimNorm has zero KL (deterministic)."""
        space = SimNormLatentSpace(dim=256, simnorm_dim=8)
        posterior = torch.randn(4, 256)
        prior = torch.randn(4, 256)
        kl = space.kl_divergence(posterior, prior)
        assert torch.allclose(kl, torch.zeros(4))
    
    def test_invalid_dim_raises(self):
        """Dimension must be divisible by simnorm_dim."""
        with pytest.raises(ValueError):
            SimNormLatentSpace(dim=257, simnorm_dim=8)
    
    def test_gradient_flows(self):
        """Gradient flows through SimNorm."""
        space = SimNormLatentSpace(dim=256, simnorm_dim=8)
        params = torch.randn(4, 256, requires_grad=True)
        output = space.sample(params)
        loss = output.sum()
        loss.backward()
        assert params.grad is not None
