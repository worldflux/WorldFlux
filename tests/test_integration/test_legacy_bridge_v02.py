"""Integration tests for v0.2 legacy bridge compatibility."""

from __future__ import annotations

import pytest
import torch

from worldflux import create_world_model
from worldflux.core.batch import Batch
from worldflux.core.payloads import PLANNER_HORIZON_KEY, ActionPayload, normalize_planned_action
from worldflux.training import Trainer, TrainingConfig


class _LegacySequenceProvider:
    def __init__(self, obs_dim: int, action_dim: int):
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def sample(
        self, batch_size: int, seq_len: int | None = None, device: str | torch.device = "cpu"
    ) -> Batch:
        t = seq_len or 4
        obs = torch.randn(batch_size, t, self.obs_dim, device=device)
        actions = torch.randn(batch_size, t, self.action_dim, device=device)
        rewards = torch.randn(batch_size, t, device=device)
        terminations = torch.zeros(batch_size, t, device=device)
        return Batch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            layouts={"obs": "BT...", "actions": "BT...", "rewards": "BT", "terminations": "BT"},
            strict_layout=True,
        )


def test_create_world_model_v02_emits_deprecation_warning():
    with pytest.warns(DeprecationWarning):
        model = create_world_model("tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="v0.2")
    assert model.config.model_type == "tdmpc2"


def test_legacy_batch_bridge_trains_one_step(tmp_path):
    with pytest.warns(DeprecationWarning):
        model = create_world_model("tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="v0.2")

    provider = _LegacySequenceProvider(obs_dim=4, action_dim=2)
    cfg = TrainingConfig(
        total_steps=1,
        batch_size=4,
        sequence_length=4,
        output_dir=str(tmp_path / "legacy-v02"),
        device="cpu",
    )
    trainer = Trainer(model, cfg)
    trained = trainer.train(provider, num_steps=1)
    assert trained is model


def test_planner_horizon_inference_warns_in_v02():
    payload = ActionPayload(kind="continuous", tensor=torch.randn(3, 2))
    with pytest.warns(DeprecationWarning, match="missing extras"):
        seq = normalize_planned_action(payload, api_version="v0.2")
    assert seq.tensor is not None


def test_planner_horizon_is_required_in_v3():
    payload = ActionPayload(kind="continuous", tensor=torch.randn(3, 2))
    with pytest.raises(ValueError, match="Missing required planner metadata"):
        normalize_planned_action(payload, api_version="v3")


def test_planner_sequence_field_errors_in_v3():
    payload = ActionPayload(
        kind="continuous",
        tensor=torch.randn(2, 3),
        extras={"wf.planner.sequence": True, PLANNER_HORIZON_KEY: 2},
    )
    with pytest.raises(ValueError, match="removed in v0.3"):
        normalize_planned_action(payload, api_version="v3")
