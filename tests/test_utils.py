"""Tests for worldflux.utils helpers."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from worldflux.utils import set_seed


def test_set_seed_reproducibility_across_python_numpy_torch() -> None:
    set_seed(123)
    a_py = random.random()
    a_np = np.random.rand(3)
    a_torch = torch.randn(2, 3)

    set_seed(123)
    b_py = random.random()
    b_np = np.random.rand(3)
    b_torch = torch.randn(2, 3)

    assert a_py == b_py
    np.testing.assert_allclose(a_np, b_np)
    torch.testing.assert_close(a_torch, b_torch)


def test_set_seed_deterministic_mode_sets_runtime_flags(monkeypatch) -> None:
    old_deterministic = torch.backends.cudnn.deterministic
    old_benchmark = torch.backends.cudnn.benchmark
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    try:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        set_seed(7, deterministic=True)
        assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark
