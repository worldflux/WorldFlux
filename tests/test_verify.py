"""Tests for worldflux.verify and the CLI verify command."""

from __future__ import annotations

import dataclasses

import pytest
from typer.testing import CliRunner

from worldflux.verify import ParityVerifier, VerifyResult

# ---------------------------------------------------------------------------
# VerifyResult
# ---------------------------------------------------------------------------


class TestVerifyResult:
    def test_frozen(self) -> None:
        result = VerifyResult(
            passed=True,
            target="model.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=True,
            elapsed_seconds=0.5,
            stats={"samples": 500},
            verdict_reason="ok",
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.passed = False  # type: ignore[misc]

    def test_fields_stored(self) -> None:
        result = VerifyResult(
            passed=False,
            target="t",
            baseline="b",
            env="e",
            demo=False,
            elapsed_seconds=1.23,
            stats={"k": "v"},
            verdict_reason="reason",
        )
        assert result.passed is False
        assert result.target == "t"
        assert result.baseline == "b"
        assert result.env == "e"
        assert result.demo is False
        assert result.elapsed_seconds == 1.23
        assert result.stats == {"k": "v"}
        assert result.verdict_reason == "reason"

    def test_default_stats_and_reason(self) -> None:
        result = VerifyResult(
            passed=True,
            target="t",
            baseline="b",
            env="e",
            demo=True,
            elapsed_seconds=0.0,
        )
        assert result.stats == {}
        assert result.verdict_reason == ""


# ---------------------------------------------------------------------------
# ParityVerifier
# ---------------------------------------------------------------------------


class TestParityVerifier:
    def test_demo_mode_passes(self) -> None:
        result = ParityVerifier.run(
            target="my_model.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=True,
            device="cpu",
        )
        assert result.passed is True
        assert result.demo is True
        assert result.target == "my_model.pt"
        assert result.elapsed_seconds >= 2.5
        assert "samples" in result.stats
        assert "mean_drop_ratio" in result.stats
        assert "ci_upper_ratio" in result.stats
        assert "margin_ratio" in result.stats
        assert "bayesian_equivalence_hdi" in result.stats
        assert "tost_p_value" in result.stats
        assert 0.0 < result.stats["bayesian_equivalence_hdi"] <= 1.0
        assert 0.0 < result.stats["tost_p_value"] < 1.0

    def test_real_mode_raises(self) -> None:
        with pytest.raises(NotImplementedError, match="Real parity verification"):
            ParityVerifier.run(
                target="my_model.pt",
                demo=False,
            )


# ---------------------------------------------------------------------------
# CLI verify command
# ---------------------------------------------------------------------------


class TestVerifyCLI:
    @pytest.fixture()
    def runner(self) -> CliRunner:
        return CliRunner()

    def test_demo_pass(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result = VerifyResult(
            passed=True,
            target="m.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=True,
            elapsed_seconds=3.1,
            stats={
                "samples": 500,
                "mean_drop_ratio": 0.01,
                "ci_upper_ratio": 0.03,
                "margin_ratio": 0.05,
                "bayesian_equivalence_hdi": 0.985,
                "tost_p_value": 0.012,
            },
            verdict_reason="Demo mode: synthetic pass",
        )
        monkeypatch.setattr(ParityVerifier, "run", classmethod(lambda cls, **kw: fake_result))

        from worldflux.cli import app

        result = runner.invoke(app, ["verify", "--target", "m.pt", "--demo"])
        assert result.exit_code == 0
        assert "PASS" in result.output
        assert "Mathematically Guaranteed Parity" in result.output
        assert "Bayesian Equivalence HDI" in result.output
        assert "TOST p-value" in result.output

    def test_fail_exit_code(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result = VerifyResult(
            passed=False,
            target="m.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=True,
            elapsed_seconds=3.1,
            stats={
                "samples": 500,
                "mean_drop_ratio": 0.1,
                "ci_upper_ratio": 0.2,
                "margin_ratio": 0.05,
                "bayesian_equivalence_hdi": 0.42,
                "tost_p_value": 0.31,
            },
            verdict_reason="threshold exceeded",
        )
        monkeypatch.setattr(ParityVerifier, "run", classmethod(lambda cls, **kw: fake_result))

        from worldflux.cli import app

        result = runner.invoke(app, ["verify", "--target", "m.pt", "--demo"])
        assert result.exit_code == 1
        assert "FAIL" in result.output

    def test_target_required(self, runner: CliRunner) -> None:
        from worldflux.cli import app

        result = runner.invoke(app, ["verify"])
        assert result.exit_code != 0

    def test_real_mode_unavailable(self, runner: CliRunner) -> None:
        from worldflux.cli import app

        result = runner.invoke(app, ["verify", "--target", "m.pt"])
        assert result.exit_code == 1
        assert "unavailable" in result.output.lower() or "not yet" in result.output.lower()


# ---------------------------------------------------------------------------
# parity proof command existence
# ---------------------------------------------------------------------------


class TestParityProofCombined:
    def test_proof_command_exists(self, runner: CliRunner | None = None) -> None:
        """Verify that `parity proof` is registered as a CLI command."""
        from worldflux.cli import parity_app

        command_names = [cmd.name for cmd in parity_app.registered_commands]
        assert "proof" in command_names

    def test_proof_help(self) -> None:
        runner = CliRunner()
        from worldflux.cli import app

        result = runner.invoke(app, ["parity", "proof", "--help"])
        assert result.exit_code == 0
        assert "proof-grade" in result.output.lower() or "proof" in result.output.lower()
