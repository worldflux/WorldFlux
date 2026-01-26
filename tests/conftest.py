"""Pytest configuration and fixtures."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seeds for reproducibility across all tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
