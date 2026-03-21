# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for worldflux.config_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldflux.config_loader import (
    ArchitectureConfig,
    CloudSectionConfig,
    DataSectionConfig,
    FlywheelSectionConfig,
    GameplaySectionConfig,
    InferenceSectionConfig,
    OnlineCollectionSectionConfig,
    ProjectConfig,
    TrainingSectionConfig,
    VerifySectionConfig,
    VisualizationSectionConfig,
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
backend = "native_torch"
backend_profile = "local"
output_dir = "./test-outputs"

[data]
source = "gym"
num_episodes = 12
episode_length = 34
buffer_capacity = 4096
gym_env = "ALE/Breakout-v5"

[gameplay]
enabled = true
fps = 12
max_frames = 256

[online_collection]
enabled = true
warmup_transitions = 1024
collect_steps_per_update = 128
max_episode_steps = 77

[inference]
horizon = 15
checkpoint = "./outputs/checkpoint_best.pt"

[visualization]
enabled = true
host = "127.0.0.1"
port = 8765
refresh_ms = 750
history_max_points = 1234
open_browser = false

[verify]
baseline = "official/dreamerv3"
env = "atari/pong"
backend = "native_torch"
backend_profile = "official_xl"
mode = "proof"
proof_claim = "compare"
allow_official_only = true

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
        assert cfg.training.backend == "native_torch"
        assert cfg.training.backend_profile == "local"
        assert cfg.training.output_dir == "./test-outputs"

    def test_verify_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.verify, VerifySectionConfig)
        assert cfg.verify.baseline == "official/dreamerv3"
        assert cfg.verify.env == "atari/pong"
        assert cfg.verify.backend == "native_torch"
        assert cfg.verify.backend_profile == "official_xl"
        assert cfg.verify.mode == "proof"
        assert cfg.verify.proof_claim == "compare"
        assert cfg.verify.allow_official_only is True

    def test_data_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.data, DataSectionConfig)
        assert cfg.data.source == "gym"
        assert cfg.data.num_episodes == 12
        assert cfg.data.episode_length == 34
        assert cfg.data.buffer_capacity == 4096
        assert cfg.data.gym_env == "ALE/Breakout-v5"

    def test_gameplay_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.gameplay, GameplaySectionConfig)
        assert cfg.gameplay.enabled is True
        assert cfg.gameplay.fps == 12
        assert cfg.gameplay.max_frames == 256

    def test_online_collection_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.online_collection, OnlineCollectionSectionConfig)
        assert cfg.online_collection.enabled is True
        assert cfg.online_collection.warmup_transitions == 1024
        assert cfg.online_collection.collect_steps_per_update == 128
        assert cfg.online_collection.max_episode_steps == 77

    def test_inference_and_visualization_parsed(self, sample_toml: Path) -> None:
        cfg = load_config(sample_toml)
        assert isinstance(cfg.inference, InferenceSectionConfig)
        assert cfg.inference.horizon == 15
        assert cfg.inference.checkpoint == "./outputs/checkpoint_best.pt"

        assert isinstance(cfg.visualization, VisualizationSectionConfig)
        assert cfg.visualization.enabled is True
        assert cfg.visualization.host == "127.0.0.1"
        assert cfg.visualization.port == 8765
        assert cfg.visualization.refresh_ms == 750
        assert cfg.visualization.history_max_points == 1234
        assert cfg.visualization.open_browser is False

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
        assert cfg.data.source == "random"
        assert cfg.gameplay.enabled is False
        assert cfg.online_collection.enabled is False
        assert cfg.inference.horizon == 15
        assert cfg.visualization.enabled is False
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

    def test_rejects_unknown_top_level_keys(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            'project_name = "bad"\nextra = "nope"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown top-level key"):
            load_config(toml_path)

    def test_rejects_unknown_section_keys(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "bad-training"
model = "dreamer:ci"

[training]
total_steps = 10
ema_decay = 0.99
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Unknown key 'training.ema_decay'"):
            load_config(toml_path)

    def test_rejects_non_supported_model_family(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            'project_name = "bad-model"\nmodel = "jepa:base"\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="supported newcomer schema"):
            load_config(toml_path)

    def test_rejects_non_table_sections(self, tmp_path: Path) -> None:
        toml_path = tmp_path / "worldflux.toml"
        toml_path.write_text(
            """\
project_name = "bad-section"
model = "tdmpc2:5m"
training = "oops"
""",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Section 'training' must be a TOML table"):
            load_config(toml_path)
