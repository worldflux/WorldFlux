# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for gradient clip unification between model and trainer configs."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from worldflux.training.config import TrainingConfig


class TestGradClipConfig:
    def test_grad_clip_none_is_valid(self):
        config = TrainingConfig(grad_clip=None)
        assert config.grad_clip is None

    def test_grad_clip_default_is_none(self):
        config = TrainingConfig()
        assert config.grad_clip is None

    def test_explicit_grad_clip(self):
        config = TrainingConfig(grad_clip=50.0)
        assert config.grad_clip == 50.0

    def test_negative_grad_clip_rejected(self):
        from worldflux.core.exceptions import ConfigurationError

        with pytest.raises(ConfigurationError):
            TrainingConfig(grad_clip=-1.0)


class _DummyModel(nn.Module):
    """Minimal nn.Module that satisfies Trainer construction."""

    def __init__(self, grad_clip=None, has_config=True):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        if has_config and grad_clip is not None:
            self.config = MagicMock()
            self.config.grad_clip = grad_clip
            # Ensure to_dict is not called unexpectedly
            self.config.to_dict = MagicMock(return_value={"grad_clip": grad_clip})
        elif not has_config:
            # No config attribute at all
            if hasattr(self, "config"):
                delattr(self, "config")
        else:
            # has_config=True but grad_clip=None: config exists but has no grad_clip
            cfg = MagicMock()
            del cfg.grad_clip
            cfg.to_dict = MagicMock(return_value={})
            self.config = cfg


class TestTrainerGradClipResolution:
    def test_none_trainer_uses_model_config(self):
        from worldflux.training.trainer import Trainer

        model = _DummyModel(grad_clip=1000.0)
        config = TrainingConfig(grad_clip=None, device="cpu")
        trainer = Trainer(model, config)
        assert trainer._resolved_grad_clip == 1000.0

    def test_explicit_trainer_overrides_model(self):
        from worldflux.training.trainer import Trainer

        model = _DummyModel(grad_clip=1000.0)
        config = TrainingConfig(grad_clip=50.0, device="cpu")
        trainer = Trainer(model, config)
        assert trainer._resolved_grad_clip == 50.0

    def test_no_model_config_uses_fallback(self):
        from worldflux.training.trainer import Trainer

        model = _DummyModel(has_config=False)
        config = TrainingConfig(grad_clip=None, device="cpu")
        trainer = Trainer(model, config)
        assert trainer._resolved_grad_clip == 100.0

    def test_warn_on_mismatch(self, caplog):
        from worldflux.training.trainer import Trainer

        model = _DummyModel(grad_clip=1000.0)
        config = TrainingConfig(grad_clip=100.0, device="cpu")
        with caplog.at_level(logging.WARNING):
            Trainer(model, config)
        assert "grad_clip" in caplog.text.lower() or "differs" in caplog.text.lower()

    def test_no_grad_clip_attr_on_model_config_uses_fallback(self):
        """Model has config but no grad_clip attribute - should use fallback."""
        from worldflux.training.trainer import Trainer

        model = _DummyModel(grad_clip=None, has_config=True)
        config = TrainingConfig(grad_clip=None, device="cpu")
        trainer = Trainer(model, config)
        assert trainer._resolved_grad_clip == 100.0

    def test_explicit_trainer_no_model_config(self):
        """Trainer has explicit grad_clip, model has no config - use trainer value."""
        from worldflux.training.trainer import Trainer

        model = _DummyModel(has_config=False)
        config = TrainingConfig(grad_clip=42.0, device="cpu")
        trainer = Trainer(model, config)
        assert trainer._resolved_grad_clip == 42.0
