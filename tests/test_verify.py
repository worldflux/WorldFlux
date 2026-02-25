"""Tests for worldflux.verify and the CLI verify command."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from worldflux.verify import PROTOCOL_VERSION, ParityVerifier, QuickVerifyResult, VerifyResult
from worldflux.verify.protocol import PROTOCOL_VERSION as PROTO_VER

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

    def test_real_mode_delegates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result = VerifyResult(
            passed=True,
            target="my_model.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=False,
            elapsed_seconds=1.2,
            stats={"samples": 3},
            verdict_reason="ok",
        )
        monkeypatch.setattr(ParityVerifier, "_run_real", classmethod(lambda cls, **kw: fake_result))
        result = ParityVerifier.run(target="my_model.pt", demo=False)
        assert result is fake_result


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

    def test_real_mode_unavailable(
        self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_runtime_error(cls, **kw):
            raise RuntimeError("boom")

        monkeypatch.setattr(
            ParityVerifier,
            "run",
            classmethod(_raise_runtime_error),
        )
        from worldflux.cli import app

        result = runner.invoke(app, ["verify", "--target", "m.pt"])
        assert result.exit_code == 1
        assert "unavailable" in result.output.lower()

    def test_real_mode_success(self, runner: CliRunner, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_result = VerifyResult(
            passed=True,
            target="m.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=False,
            elapsed_seconds=12.0,
            stats={
                "samples": 20,
                "mean_drop_ratio": 0.01,
                "ci_upper_ratio": 1.03,
                "margin_ratio": 0.05,
                "bayesian_equivalence_hdi": 0.98,
                "tost_p_value": 0.01,
            },
            verdict_reason="proof pass",
        )
        monkeypatch.setattr(ParityVerifier, "run", classmethod(lambda cls, **kw: fake_result))

        from worldflux.cli import app

        result = runner.invoke(app, ["verify", "--target", "m.pt"])
        assert result.exit_code == 0
        assert "PASS" in result.output

    def test_verify_writes_evidence_bundle_for_proof_mode(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        runs = tmp_path / "parity_runs.jsonl"
        eq_json = tmp_path / "equivalence_report.json"
        eq_md = tmp_path / "equivalence_report.md"
        runs.write_text("{}\n", encoding="utf-8")
        eq_json.write_text("{}\n", encoding="utf-8")
        eq_md.write_text("# report\n", encoding="utf-8")

        fake_result = VerifyResult(
            passed=True,
            target="m.pt",
            baseline="official/dreamerv3",
            env="atari/pong",
            demo=False,
            elapsed_seconds=1.0,
            stats={
                "runs_jsonl": str(runs),
                "equivalence_report_json": str(eq_json),
                "equivalence_report_md": str(eq_md),
                "device": "cpu",
            },
            verdict_reason="ok",
        )
        monkeypatch.setattr(ParityVerifier, "run", classmethod(lambda cls, **kw: fake_result))

        evidence_dir = tmp_path / "evidence"
        from worldflux.cli import app

        result = runner.invoke(
            app,
            [
                "verify",
                "--target",
                "m.pt",
                "--mode",
                "proof",
                "--evidence-bundle",
                str(evidence_dir),
            ],
        )
        assert result.exit_code == 0
        manifest_path = evidence_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["mode"] == "proof"
        assert len(manifest["artifacts"]) >= 1

    def test_verify_cloud_mode_json_success(
        self,
        runner: CliRunner,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        class _FakeClient:
            api_key = "wf_test_key"

            def verify_cloud(self, payload):
                return {
                    "passed": True,
                    "status": "completed",
                    "verdict_reason": "cloud pass",
                    "target": payload["target"],
                    "baseline": payload["baseline"],
                    "env": payload["env"],
                    "stats": {"samples": 12},
                }

        monkeypatch.setattr(
            "worldflux.cloud.client.WorldFluxCloudClient.from_env",
            classmethod(lambda cls: _FakeClient()),
        )

        output_path = tmp_path / "cloud-verify.json"
        from worldflux.cli import app

        result = runner.invoke(
            app,
            [
                "verify",
                "--target",
                "m.pt",
                "--mode",
                "cloud",
                "--format",
                "json",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["passed"] is True
        assert payload["status"] == "completed"


# ---------------------------------------------------------------------------
# parity proof command existence
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# QuickVerifyResult
# ---------------------------------------------------------------------------


class TestQuickVerifyResult:
    def test_frozen(self) -> None:
        result = QuickVerifyResult(
            passed=True,
            target="outputs/",
            env="atari/pong",
            episodes=10,
            mean_score=0.85,
            baseline_mean=0.80,
            elapsed_seconds=1.0,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.passed = False  # type: ignore[misc]

    def test_fields_stored(self) -> None:
        result = QuickVerifyResult(
            passed=True,
            target="outputs/",
            env="atari/pong",
            episodes=10,
            mean_score=0.85,
            baseline_mean=0.80,
            elapsed_seconds=1.5,
            stats={"k": "v"},
            verdict_reason="pass",
        )
        assert result.passed is True
        assert result.target == "outputs/"
        assert result.env == "atari/pong"
        assert result.episodes == 10
        assert result.mean_score == 0.85
        assert result.baseline_mean == 0.80
        assert result.elapsed_seconds == 1.5
        assert result.protocol_version == PROTO_VER
        assert result.stats == {"k": "v"}
        assert result.verdict_reason == "pass"

    def test_to_dict(self) -> None:
        result = QuickVerifyResult(
            passed=True,
            target="t",
            env="e",
            episodes=5,
            mean_score=1.0,
            baseline_mean=0.9,
            elapsed_seconds=0.5,
        )
        d = result.to_dict()
        assert d["passed"] is True
        assert d["protocol_version"] == PROTO_VER
        assert d["episodes"] == 5

    def test_default_stats_and_reason(self) -> None:
        result = QuickVerifyResult(
            passed=True,
            target="t",
            env="e",
            episodes=5,
            mean_score=1.0,
            baseline_mean=0.9,
            elapsed_seconds=0.0,
        )
        assert result.stats == {}
        assert result.verdict_reason == ""


class TestProtocolVersion:
    def test_protocol_version_is_string(self) -> None:
        assert isinstance(PROTOCOL_VERSION, str)

    def test_protocol_version_format(self) -> None:
        parts = PROTOCOL_VERSION.split(".")
        assert len(parts) >= 1
        assert all(p.isdigit() for p in parts)


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


# ---------------------------------------------------------------------------
# Phase 6: QualityTier and quality_check
# ---------------------------------------------------------------------------


class TestQualityTier:
    def test_enum_values(self) -> None:
        from worldflux.verify.quick import QualityTier

        assert QualityTier.SMOKE.value == "smoke"
        assert QualityTier.BASELINE.value == "baseline"
        assert QualityTier.PRODUCTION.value == "production"

    def test_is_str_enum(self) -> None:
        from worldflux.verify.quick import QualityTier

        assert isinstance(QualityTier.SMOKE, str)
        assert QualityTier.SMOKE == "smoke"

    def test_exports(self) -> None:
        from worldflux.verify import QualityCheckResult, QualityTier, quality_check

        assert QualityTier is not None
        assert QualityCheckResult is not None
        assert callable(quality_check)


class TestQualityCheck:
    def test_smoke_check_passes_with_valid_model(self) -> None:
        from worldflux import create_world_model
        from worldflux.verify.quick import QualityTier, quality_check

        model = create_world_model("dreamer:ci", obs_shape=(3, 64, 64), action_dim=6)
        result = quality_check(model, tier=QualityTier.SMOKE, device="cpu")
        assert result.passed is True
        assert result.achieved_tier == QualityTier.SMOKE
        assert result.score > 0

    def test_quality_check_result_frozen(self) -> None:
        from worldflux.verify.quick import QualityCheckResult, QualityTier

        result = QualityCheckResult(
            tier=QualityTier.SMOKE,
            achieved_tier=QualityTier.SMOKE,
            score=0.33,
            passed=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            result.passed = False  # type: ignore[misc]


class TestAutoQualityCheck:
    def test_config_has_auto_quality_check(self) -> None:
        from worldflux.training.config import TrainingConfig

        config = TrainingConfig()
        assert hasattr(config, "auto_quality_check")
        assert config.auto_quality_check is True

    def test_config_auto_quality_check_disabled(self) -> None:
        from worldflux.training.config import TrainingConfig

        config = TrainingConfig(auto_quality_check=False)
        assert config.auto_quality_check is False
