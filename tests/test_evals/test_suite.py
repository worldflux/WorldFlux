# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for evaluation suite runner."""

from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.spec import ActionSpec, Capability, ModelIOContract
from worldflux.core.state import State
from worldflux.evals.result import EvalReport
from worldflux.evals.suite import SUITE_CONFIGS, run_eval_suite


class _MockWorldModel(WorldModel):
    """Minimal mock world model for suite tests."""

    def __init__(self):
        super().__init__()
        self._encoder = nn.Linear(8, 32)
        self._decoder = nn.Linear(32, 8)
        self._reward_head = nn.Linear(32, 1)
        self.capabilities = {
            Capability.LATENT_DYNAMICS,
            Capability.OBS_DECODER,
            Capability.REWARD_PRED,
        }
        self.config = type("Config", (), {"obs_shape": (8,), "action_dim": 4})()

    def io_contract(self) -> ModelIOContract:
        return ModelIOContract(
            action_spec=ActionSpec(kind="continuous", dim=4),
        )

    def encode(self, obs, deterministic: bool = False) -> State:
        if isinstance(obs, dict):
            obs = obs.get("obs", obs)
        return State(tensors={"latent": self._encoder(obs)})

    def decode(self, state, conditions=None) -> ModelOutput:
        latent = state.tensors["latent"]
        return ModelOutput(
            predictions={
                "obs": self._decoder(latent),
                "reward": self._reward_head(latent),
            },
            state=state,
        )

    def transition(self, state, action, conditions=None, deterministic=False) -> State:
        latent = state.tensors["latent"]
        if isinstance(action, torch.Tensor):
            new_latent = latent + 0.01 * action.sum(dim=-1, keepdim=True).expand_as(latent)
        else:
            new_latent = latent + 0.01
        return State(tensors={"latent": new_latent})

    def loss(self, batch) -> LossOutput:
        return LossOutput(loss=torch.tensor(0.0))


class TestRunEvalSuite:
    @pytest.fixture
    def model(self):
        return _MockWorldModel()

    def test_quick_suite(self, model):
        report = run_eval_suite(model, suite="quick", model_id="test")

        assert isinstance(report, EvalReport)
        assert report.suite == "quick"
        assert report.model_id == "test"
        assert len(report.results) == len(SUITE_CONFIGS["quick"])
        assert report.wall_time_sec >= 0.0

    def test_standard_suite(self, model):
        report = run_eval_suite(model, suite="standard", model_id="test")

        assert isinstance(report, EvalReport)
        assert len(report.results) == len(SUITE_CONFIGS["standard"])

    def test_comprehensive_suite(self, model):
        report = run_eval_suite(model, suite="comprehensive", model_id="test")

        assert isinstance(report, EvalReport)
        assert len(report.results) == len(SUITE_CONFIGS["comprehensive"])

    def test_invalid_suite_raises(self, model):
        with pytest.raises(ValueError, match="Unknown suite"):
            run_eval_suite(model, suite="nonexistent")

    def test_output_saving(self, model, tmp_path):
        output_path = tmp_path / "eval_report.json"
        run_eval_suite(model, suite="quick", output_path=output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert data["suite"] == "quick"
        assert len(data["results"]) == len(SUITE_CONFIGS["quick"])

    def test_report_to_dict_roundtrip(self, model):
        report = run_eval_suite(model, suite="quick", model_id="test")
        d = report.to_dict()

        assert d["suite"] == "quick"
        assert d["model_id"] == "test"
        assert isinstance(d["results"], list)
        assert d["wall_time_sec"] >= 0.0
        assert d["mode"] == "synthetic"
        assert "synthetic_provenance" in d

    def test_dataset_replay_mode_requires_explicit_data(self, model):
        with pytest.raises(ValueError, match="real evaluation data"):
            run_eval_suite(model, suite="quick", mode="dataset_replay")

    def test_dataset_replay_mode_records_new_and_legacy_provenance(self, model):
        data = {
            "obs": torch.randn(4, 8),
            "actions": torch.randn(6, 4, 4),
            "rewards": torch.randn(6, 4),
        }
        report = run_eval_suite(
            model,
            suite="quick",
            mode="dataset_replay",
            data=data,
            provenance={"kind": "dataset_manifest", "env_id": "mujoco/HalfCheetah-v5"},
        )

        payload = report.to_dict()
        assert payload["mode"] == "dataset_replay"
        assert payload["dataset_replay_provenance"]["kind"] == "dataset_manifest"
        assert payload["real_provenance"]["kind"] == "dataset_manifest"
        assert "synthetic_provenance" not in payload

    def test_env_policy_mode_records_env_policy_provenance(self, model):
        data = {
            "obs": torch.randn(4, 8),
            "actions": torch.randn(6, 4, 4),
            "rewards": torch.randn(6, 4),
        }
        report = run_eval_suite(
            model,
            suite="quick",
            mode="env_policy",
            data=data,
            provenance={"kind": "env_policy", "env_id": "ALE/Breakout-v5"},
        )

        payload = report.to_dict()
        assert payload["mode"] == "env_policy"
        assert payload["env_policy_provenance"]["kind"] == "env_policy"
        assert payload["real_provenance"]["kind"] == "env_policy"
