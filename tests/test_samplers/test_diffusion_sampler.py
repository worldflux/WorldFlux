"""Tests for diffusion scheduler and sampler behavior."""

from __future__ import annotations

import pytest
import torch

from worldflux.samplers.diffusion import DiffusionSampler, DiffusionScheduler


def test_diffusion_scheduler_rejects_invalid_init_arguments() -> None:
    with pytest.raises(ValueError):
        DiffusionScheduler(num_train_steps=0)
    with pytest.raises(ValueError):
        DiffusionScheduler(beta_start=0.02, beta_end=0.01)
    with pytest.raises(ValueError):
        DiffusionScheduler(prediction_target="velocity")


def test_sample_timesteps_are_long_and_in_range() -> None:
    scheduler = DiffusionScheduler(num_train_steps=10)
    timesteps = scheduler.sample_timesteps(batch_size=32, device=torch.device("cpu"))

    assert timesteps.dtype == torch.long
    assert timesteps.shape == (32,)
    assert int(timesteps.min().item()) >= 0
    assert int(timesteps.max().item()) < 10


def test_add_noise_validates_timestep_dtype_and_range() -> None:
    scheduler = DiffusionScheduler(num_train_steps=5)
    clean = torch.zeros(2, 3)
    noise = torch.ones(2, 3)

    with pytest.raises(ValueError, match="torch.long"):
        scheduler.add_noise(clean, noise, torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="out of range"):
        scheduler.add_noise(clean, noise, torch.tensor([0, 7], dtype=torch.long))


def test_scheduler_step_x0_target_returns_prediction() -> None:
    scheduler = DiffusionScheduler(num_train_steps=8, prediction_target="x0")
    prediction = torch.randn(4, 6)
    x_t = torch.randn(4, 6)
    timesteps = torch.full((4,), 3, dtype=torch.long)

    out = scheduler.step(prediction, x_t, timesteps)
    torch.testing.assert_close(out, prediction)


def test_scheduler_step_noise_target_runs_without_nan() -> None:
    scheduler = DiffusionScheduler(num_train_steps=8, prediction_target="noise")
    prediction = torch.randn(4, 6)
    x_t = torch.randn(4, 6)
    timesteps = torch.full((4,), 2, dtype=torch.long)

    out = scheduler.step(prediction, x_t, timesteps)
    assert out.shape == x_t.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_diffusion_sampler_step_supports_optional_timestep_arg() -> None:
    class ModelWithTimestep:
        def denoise(self, x, action=None, timestep=None):
            assert timestep is not None
            return torch.zeros_like(x)

    scheduler = DiffusionScheduler(num_train_steps=4)
    sampler = DiffusionSampler(scheduler=scheduler)
    x = torch.randn(3, 2)

    out = sampler.step(ModelWithTimestep(), x)
    assert out.shape == x.shape


def test_diffusion_sampler_step_falls_back_when_model_has_no_timestep_parameter() -> None:
    class ModelWithoutTimestep:
        def denoise(self, x, action=None):
            return torch.zeros_like(x)

    scheduler = DiffusionScheduler(num_train_steps=4)
    sampler = DiffusionSampler(scheduler=scheduler)
    x = torch.randn(3, 2)

    out = sampler.step(ModelWithoutTimestep(), x)
    assert out.shape == x.shape


def test_diffusion_sampler_sample_decrements_timesteps() -> None:
    class EchoModel:
        def __init__(self) -> None:
            self.timesteps: list[int] = []

        def denoise(self, x, action=None, timestep=None):
            assert timestep is not None
            self.timesteps.extend(timestep.tolist())
            return torch.zeros_like(x)

    scheduler = DiffusionScheduler(num_train_steps=6, prediction_target="x0")
    sampler = DiffusionSampler(scheduler=scheduler)
    x = torch.randn(2, 4)
    model = EchoModel()

    out = sampler.sample(model, x, steps=3, start_timestep=5)
    assert out.shape == x.shape
    assert model.timesteps[:2] == [5, 5]
    assert model.timesteps[2:4] == [4, 4]
    assert model.timesteps[4:6] == [3, 3]
