"""Tests for worldflux.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldflux.config_loader import (
    ArchitectureConfig,
    CloudSectionConfig,
    FlywheelSectionConfig,
    ProjectConfig,
    TrainingSectionConfig,
    VerifySectionConfig,
    load_config,
)


@pytest.fixture()
def sample_toml(tmp_path: Path) -> Path:
    """Write a minimal worldflux.toml and return its path."""
    content = """\
project_name = "test-project"
environment = "atari"
model = "dreamer:ci"
model_type = "dreamer"

[architecture]
obs_shape = [3, 64, 64]
action_dim = 6
hidden_dim = 32

[training]
total_steps = 5000
batch_size = 8
sequence_length = 25
learning_rate = 1e-3
device = "cpu"
output_dir = "./test-outputs"

[verify]
baseline = "official/dreamerv3"
env = "atari/pong"

[cloud]
gpu_type = "a10g"
spot = false
region = "us-west-2"
timeout_hours = 12

[flywheel]
opt_in = true
privacy_epsilon = 0.5
privacy_delta = 1e-6
"""
    toml_path = tmp_path / "worldflux.toml"
    toml_path.write_text(content, encoding="utf-8")
    return toml_path


class TestLoadConfig:
    def test_loads_valid_toml(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg, ProjectConfig)
        assert cfg.project_name == "test-project"
        assert cfg.environment == "atari"
        assert cfg.model == "dreamer:ci"
        assert cfg.model_type == "dreamer"

    def test_architecture_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.architecture, ArchitectureConfig)
        assert cfg.architecture.obs_shape == (3, 64, 64)
        assert cfg.architecture.action_dim == 6
        assert cfg.architecture.hidden_dim == 32

    def test_training_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.training, TrainingSectionConfig)
        assert cfg.training.total_steps == 5000
        assert cfg.training.batch_size == 8
        assert cfg.training.sequence_length == 25
        assert cfg.training.learning_rate == 1e-3
        assert cfg.training.device == "cpu"
        assert cfg.training.output_dir == "./test-outputs"

    def test_verify_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.verify, VerifySectionConfig)
        assert cfg.verify.baseline == "official/dreamerv3"
        assert cfg.verify.env == "atari/pong"

    def test_raw_dict_preserved(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.raw, dict)
        assert cfg.raw["project_name"] == "test-project"

    def test_cloud_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.cloud, CloudSectionConfig)
        assert cfg.cloud.gpu_type == "a10g"
        assert cfg.cloud.spot is False
        assert cfg.cloud.region == "us-west-2"
        assert cfg.cloud.timeout_hours == 12

    def test_flywheel_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.flywheel, FlywheelSectionConfig)
        assert cfg.flywheel.opt_in is True
        assert cfg.flywheel.privacy_epsilon == 0.5
        assert cfg.flywheel.privacy_delta == 1e-6

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_config(tmp_path / "missing.toml")

    def test_defaults_for_missing_sections(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text('project_name = "minimal"\n', encoding="utf-8")
        cfg = load_config(toml_path)
        assert cfg.project_name == "minimal"
        assert cfg.architecture.obs_shape == (3, 64, 64)
        assert cfg.training.total_steps == 100_000
        assert cfg.verify.env == "atari/pong"
        assert cfg.cloud.gpu_type == "a100"
        assert cfg.cloud.spot is True
        assert cfg.flywheel.opt_in is False
        assert cfg.flywheel.privacy_epsilon == 1.0

    def test_infers_model_type_from_model(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            'project_name = "infer"\nmodel = "tdmpc2:5m"\n',
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.model_type == "tdmpc2"

    def test_infers_model_from_environment(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            'project_name = "infer-env"\nenvironment = "mujoco"\n',
            encoding="utf-8",
        )
        cfg = load_config(toml_path)
        assert cfg.model == "tdmpc2:ci"
