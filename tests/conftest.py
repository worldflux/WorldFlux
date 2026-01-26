"""Pytest configuration and fixtures."""

import pytest
import torch


@pytest.fixture
def device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def batch_size():
    return 4


@pytest.fixture
def seq_len():
    return 16


@pytest.fixture
def action_dim():
    return 6


@pytest.fixture
def obs_shape():
    return (3, 64, 64)


@pytest.fixture
def vector_obs_shape():
    return (39,)
