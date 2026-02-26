"""Tests for EvalCallback."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from worldflux.core.model import WorldModel
from worldflux.core.output import LossOutput, ModelOutput
from worldflux.core.spec import ActionSpec, Capability, ModelIOContract
from worldflux.core.state import State
from worldflux.evals.result import EvalReport, EvalResult
from worldflux.training.callbacks import EvalCallback


class _MockWorldModel(WorldModel):
    """Minimal mock for callback tests."""

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


def _make_trainer_mock(model, step: int = 0):
    """Create a mock trainer with the given model and step."""
    trainer = MagicMock()
    trainer.model = model
    trainer.state.global_step = step
    trainer.state.epoch = 0
    trainer.state.ttfi_sec = 0.0
    trainer.state.train_start_time = None
    return trainer


class TestEvalCallback:
    def test_init_validation(self):
        with pytest.raises(ValueError, match="eval_interval must be positive"):
            EvalCallback(eval_interval=0)

    def test_skips_non_interval_steps(self):
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=50)

        with patch("worldflux.training.callbacks.write_event") as mock_write:
            cb.on_step_end(trainer)
            mock_write.assert_not_called()

    def test_runs_at_interval(self):
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=100)
        cb.on_train_begin(trainer)

        with patch("worldflux.training.callbacks.write_event") as mock_write:
            cb.on_step_end(trainer)
            mock_write.assert_called_once()
            call_kwargs = mock_write.call_args[1]
            assert call_kwargs["event"] == "eval.quick"
            assert call_kwargs["step"] == 100

    def test_skips_step_zero(self):
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=0)

        with patch("worldflux.training.callbacks.write_event") as mock_write:
            cb.on_step_end(trainer)
            mock_write.assert_not_called()

    def test_success_true_when_all_passed_true(self):
        """write_event receives success=True when report.all_passed is True."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=100)
        cb.on_train_begin(trainer)

        report = EvalReport(
            suite="quick",
            model_id="mock",
            results=(
                EvalResult(
                    suite="quick",
                    metric="m1",
                    value=0.9,
                    threshold=0.5,
                    passed=True,
                    timestamp=0.0,
                    model_id="mock",
                ),
            ),
            timestamp=0.0,
            wall_time_sec=0.1,
            all_passed=True,
        )
        with (
            patch("worldflux.evals.suite.run_eval_suite", return_value=report),
            patch("worldflux.training.callbacks.write_event") as mock_write,
        ):
            cb.on_step_end(trainer)
            mock_write.assert_called_once()
            assert mock_write.call_args[1]["success"] is True

    def test_success_false_when_all_passed_none(self):
        """write_event receives success=False when report.all_passed is None."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=100)
        cb.on_train_begin(trainer)

        report = EvalReport(
            suite="quick",
            model_id="mock",
            results=(),
            timestamp=0.0,
            wall_time_sec=0.1,
            all_passed=None,
        )
        with (
            patch("worldflux.evals.suite.run_eval_suite", return_value=report),
            patch("worldflux.training.callbacks.write_event") as mock_write,
        ):
            cb.on_step_end(trainer)
            mock_write.assert_called_once()
            assert mock_write.call_args[1]["success"] is False

    def test_consecutive_failures_increment_and_reset(self):
        """Failure counter increments on exception and resets on success."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        cb.on_train_begin(_make_trainer_mock(model))

        report = EvalReport(
            suite="quick",
            model_id="mock",
            results=(),
            timestamp=0.0,
            wall_time_sec=0.0,
            all_passed=True,
        )

        with patch("worldflux.training.callbacks.write_event"):
            # Two failures
            with patch(
                "worldflux.evals.suite.run_eval_suite",
                side_effect=RuntimeError("boom"),
            ):
                cb.on_step_end(_make_trainer_mock(model, step=100))
                assert cb._consecutive_failures == 1
                cb.on_step_end(_make_trainer_mock(model, step=200))
                assert cb._consecutive_failures == 2

            # Success resets counter
            with patch(
                "worldflux.evals.suite.run_eval_suite",
                return_value=report,
            ):
                cb.on_step_end(_make_trainer_mock(model, step=300))
                assert cb._consecutive_failures == 0

    def test_escalation_warning_to_error_at_threshold(self, caplog):
        """First 2 failures log WARNING, 3rd+ logs ERROR."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        cb.on_train_begin(_make_trainer_mock(model))

        with (
            patch("worldflux.training.callbacks.write_event"),
            patch(
                "worldflux.evals.suite.run_eval_suite",
                side_effect=RuntimeError("boom"),
            ),
        ):
            # Failures 1-2 → WARNING
            for step in (100, 200):
                with caplog.at_level(logging.WARNING):
                    cb.on_step_end(_make_trainer_mock(model, step=step))
            assert cb._consecutive_failures == 2

            # Failure 3 → ERROR
            with caplog.at_level(logging.ERROR):
                cb.on_step_end(_make_trainer_mock(model, step=300))
            assert cb._consecutive_failures == 3
            assert any("3 consecutive times" in r.message for r in caplog.records)

    def test_circuit_breaker_disables_eval(self, caplog):
        """After MAX_CONSECUTIVE_EVAL_FAILURES, eval is permanently disabled."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        cb.on_train_begin(_make_trainer_mock(model))

        with (
            patch("worldflux.training.callbacks.write_event"),
            patch(
                "worldflux.evals.suite.run_eval_suite",
                side_effect=RuntimeError("boom"),
            ) as mock_eval,
        ):
            # Trigger MAX failures
            for i in range(EvalCallback.MAX_CONSECUTIVE_EVAL_FAILURES):
                cb.on_step_end(_make_trainer_mock(model, step=(i + 1) * 100))

            assert cb._eval_disabled is True
            assert cb._consecutive_failures == EvalCallback.MAX_CONSECUTIVE_EVAL_FAILURES

            # Next call should skip eval entirely
            call_count_before = mock_eval.call_count
            cb.on_step_end(_make_trainer_mock(model, step=99900))
            assert mock_eval.call_count == call_count_before  # not called again

    def test_success_false_when_all_passed_false(self):
        """write_event receives success=False when report.all_passed is explicitly False."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        trainer = _make_trainer_mock(model, step=100)
        cb.on_train_begin(trainer)

        report = EvalReport(
            suite="quick",
            model_id="mock",
            results=(
                EvalResult(
                    suite="quick",
                    metric="m1",
                    value=0.1,
                    threshold=0.5,
                    passed=False,
                    timestamp=0.0,
                    model_id="mock",
                ),
            ),
            timestamp=0.0,
            wall_time_sec=0.1,
            all_passed=False,
        )
        with (
            patch("worldflux.evals.suite.run_eval_suite", return_value=report),
            patch("worldflux.training.callbacks.write_event") as mock_write,
        ):
            cb.on_step_end(trainer)
            mock_write.assert_called_once()
            assert mock_write.call_args[1]["success"] is False

    def test_circuit_breaker_not_re_enabled_by_success(self):
        """Once the circuit breaker trips, a subsequent success does not re-enable eval."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        cb.on_train_begin(_make_trainer_mock(model))

        report = EvalReport(
            suite="quick",
            model_id="mock",
            results=(),
            timestamp=0.0,
            wall_time_sec=0.0,
            all_passed=True,
        )

        with patch("worldflux.training.callbacks.write_event"):
            # Trip the circuit breaker
            with patch(
                "worldflux.evals.suite.run_eval_suite",
                side_effect=RuntimeError("boom"),
            ):
                for i in range(EvalCallback.MAX_CONSECUTIVE_EVAL_FAILURES):
                    cb.on_step_end(_make_trainer_mock(model, step=(i + 1) * 100))
            assert cb._eval_disabled is True

            # Even if we could succeed, eval stays disabled (run_eval_suite is not called)
            with patch(
                "worldflux.evals.suite.run_eval_suite",
                return_value=report,
            ) as mock_eval:
                cb.on_step_end(_make_trainer_mock(model, step=99900))
                mock_eval.assert_not_called()
            assert cb._eval_disabled is True

    def test_circuit_breaker_emits_telemetry_event(self):
        """Circuit breaker trip emits an eval.circuit_break write_event."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        cb.on_train_begin(_make_trainer_mock(model))

        with (
            patch("worldflux.training.callbacks.write_event") as mock_write,
            patch(
                "worldflux.evals.suite.run_eval_suite",
                side_effect=RuntimeError("boom"),
            ),
        ):
            for i in range(EvalCallback.MAX_CONSECUTIVE_EVAL_FAILURES):
                cb.on_step_end(_make_trainer_mock(model, step=(i + 1) * 100))

        assert cb._eval_disabled is True
        # Find the circuit_break event among all write_event calls
        circuit_break_calls = [
            c for c in mock_write.call_args_list if c[1].get("event") == "eval.circuit_break"
        ]
        assert len(circuit_break_calls) == 1
        assert circuit_break_calls[0][1]["success"] is False

    def test_failure_does_not_emit_write_event(self):
        """A single eval failure should not emit any write_event."""
        cb = EvalCallback(eval_interval=100)
        model = _MockWorldModel()
        cb.on_train_begin(_make_trainer_mock(model))

        with (
            patch("worldflux.training.callbacks.write_event") as mock_write,
            patch(
                "worldflux.evals.suite.run_eval_suite",
                side_effect=RuntimeError("boom"),
            ),
        ):
            cb.on_step_end(_make_trainer_mock(model, step=100))
            mock_write.assert_not_called()
