"""Tests for registry and alias resolution."""

import json

import pytest

from worldflux.core.config import DreamerV3Config, TDMPC2Config, WorldModelConfig
from worldflux.core.exceptions import ConfigurationError
from worldflux.core.registry import AutoConfig, AutoWorldModel, ConfigRegistry, WorldModelRegistry


class TestWorldModelRegistryAliases:
    """WorldModelRegistry.from_pretrained alias resolution tests."""

    def test_dreamerv3_alias(self):
        """'dreamerv3:size12m' resolves correctly."""
        model = WorldModelRegistry.from_pretrained("dreamerv3:size12m")
        assert model.config.model_type == "dreamer"

    def test_dreamer_alias(self):
        """'dreamer:size12m' resolves correctly."""
        model = WorldModelRegistry.from_pretrained("dreamer:size12m")
        assert model.config.model_type == "dreamer"

    def test_tdmpc2_alias(self):
        """'tdmpc2:5m' resolves correctly."""
        model = WorldModelRegistry.from_pretrained("tdmpc2:5m")
        assert model.config.model_type == "tdmpc2"

    def test_tdmpc_alias(self):
        """'tdmpc:5m' resolves correctly."""
        model = WorldModelRegistry.from_pretrained("tdmpc:5m")
        assert model.config.model_type == "tdmpc2"

    def test_case_insensitive(self):
        """Aliases are case insensitive."""
        model1 = WorldModelRegistry.from_pretrained("DreamerV3:size12m")
        model2 = WorldModelRegistry.from_pretrained("TDMPC2:5m")
        assert model1.config.model_type == "dreamer"
        assert model2.config.model_type == "tdmpc2"

    def test_unknown_type_raises(self):
        """Unknown model type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model type"):
            WorldModelRegistry.from_pretrained("unknown:size")

    def test_invalid_format_raises(self):
        """Invalid format without colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid model identifier"):
            WorldModelRegistry.from_pretrained("invalid_format")


class TestAutoConfigAliases:
    """AutoConfig.from_pretrained alias resolution tests."""

    def test_dreamerv3_alias(self):
        """'dreamerv3:size12m' returns DreamerV3Config."""
        config = AutoConfig.from_pretrained("dreamerv3:size12m")
        assert isinstance(config, DreamerV3Config)
        assert config.model_type == "dreamer"

    def test_dreamer_alias(self):
        """'dreamer:size12m' returns DreamerV3Config."""
        config = AutoConfig.from_pretrained("dreamer:size12m")
        assert isinstance(config, DreamerV3Config)
        assert config.model_type == "dreamer"

    def test_tdmpc2_alias(self):
        """'tdmpc2:5m' returns TDMPC2Config."""
        config = AutoConfig.from_pretrained("tdmpc2:5m")
        assert isinstance(config, TDMPC2Config)
        assert config.model_type == "tdmpc2"

    def test_tdmpc_alias(self):
        """'tdmpc:5m' returns TDMPC2Config (bug fix verification)."""
        config = AutoConfig.from_pretrained("tdmpc:5m")
        assert isinstance(config, TDMPC2Config)
        assert config.model_type == "tdmpc2"

    def test_case_insensitive(self):
        """Aliases are case insensitive."""
        config1 = AutoConfig.from_pretrained("DreamerV3:size12m")
        config2 = AutoConfig.from_pretrained("TDMPC:5m")
        assert isinstance(config1, DreamerV3Config)
        assert isinstance(config2, TDMPC2Config)

    def test_unknown_type_fallback(self):
        """Unknown type falls back to base WorldModelConfig."""
        config = AutoConfig.from_pretrained("unknown:size")
        assert isinstance(config, WorldModelConfig)
        assert config.model_name == "size"

    def test_invalid_format_raises(self):
        """Invalid format without colon raises ValueError."""
        with pytest.raises(ValueError, match="Invalid config identifier"):
            AutoConfig.from_pretrained("invalid_format")


class TestAutoWorldModel:
    """AutoWorldModel.from_pretrained tests."""

    def test_delegates_to_registry(self):
        """AutoWorldModel delegates to WorldModelRegistry."""
        model = AutoWorldModel.from_pretrained("dreamer:size12m")
        assert model.config.model_type == "dreamer"


class TestRegistryConsistency:
    """Verify registry consistency."""

    def test_all_registered_types_have_configs(self):
        """All registered model types have configs registered."""
        for model_type in WorldModelRegistry.list_models().keys():
            config = AutoConfig.from_pretrained(f"{model_type}:ci")
            assert isinstance(config, WorldModelConfig)


class TestConfigRegistryErrors:
    """ConfigRegistry error handling tests."""

    def test_missing_model_type_raises(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_name": "test"}))
        with pytest.raises(ConfigurationError):
            ConfigRegistry.from_pretrained(str(tmp_path))

    def test_invalid_json_raises(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text("{invalid-json")
        with pytest.raises(ConfigurationError):
            ConfigRegistry.from_pretrained(str(tmp_path))

    def test_unregistered_model_type_raises(self, tmp_path):
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps({"model_type": "unknown"}))
        with pytest.raises(ConfigurationError):
            WorldModelRegistry.from_pretrained(str(tmp_path))
