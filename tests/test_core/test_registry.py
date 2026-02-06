"""Tests for registry and alias resolution."""

import json

import pytest

from worldflux.core.config import DreamerV3Config, TDMPC2Config, VJEPA2Config, WorldModelConfig
from worldflux.core.exceptions import ConfigurationError
from worldflux.core.interfaces import ComponentSpec
from worldflux.core.registry import (
    AutoConfig,
    AutoWorldModel,
    ConfigRegistry,
    PluginManifest,
    WorldModelRegistry,
)


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

    def test_vjepa2_alias(self):
        """'vjepa2:base' resolves correctly."""
        model = WorldModelRegistry.from_pretrained("vjepa2:base")
        assert model.config.model_type == "vjepa2"

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

    def test_vjepa2_config(self):
        config = AutoConfig.from_pretrained("vjepa2:base")
        assert isinstance(config, VJEPA2Config)
        assert config.model_type == "vjepa2"

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


class TestComponentOverrides:
    def test_apply_component_overrides_rejects_unknown_slot(self):
        model = WorldModelRegistry.from_pretrained("tdmpc2:ci", obs_shape=(4,), action_dim=2)
        with pytest.raises(ConfigurationError, match="Unknown component slot"):
            WorldModelRegistry.apply_component_overrides(model, {"unknown_slot": object()})

    def test_apply_component_overrides_rejects_component_type_mismatch(self):
        class _DummyDynamics:
            def transition(self, state, conditioned, deterministic: bool = False):
                del conditioned, deterministic
                return state

        component_id = "tests.registry.dummy_dynamics"
        WorldModelRegistry.register_component(
            component_id,
            _DummyDynamics,
            ComponentSpec(name="Dummy Dynamics", component_type="dynamics_model"),
        )
        try:
            model = WorldModelRegistry.from_pretrained("tdmpc2:ci", obs_shape=(4,), action_dim=2)
            with pytest.raises(ConfigurationError, match="expected 'action_conditioner'"):
                WorldModelRegistry.apply_component_overrides(
                    model, {"action_conditioner": component_id}
                )
        finally:
            WorldModelRegistry.unregister_component(component_id)

    def test_apply_component_overrides_rejects_non_composable_family(self):
        model = WorldModelRegistry.from_pretrained("jepa:base", obs_shape=(4,), action_dim=2)
        with pytest.raises(ConfigurationError, match="not supported by model"):
            WorldModelRegistry.apply_component_overrides(model, {"action_conditioner": object()})


class _FakeEntryPoint:
    def __init__(self, name: str, group: str, loader):
        self.name = name
        self.group = group
        self._loader = loader

    def load(self):
        return self._loader


class _FakeEntryPoints(list):
    def select(self, *, group: str):
        return [entry for entry in self if entry.group == group]


class TestPluginDiscovery:
    def test_load_entrypoint_plugins_registers_model_and_component_hooks(self, monkeypatch):
        model_type = "extplugin_model"
        model_id = "extplugin:demo"
        alias = "extplugin"
        component_id = "extplugin.zero_action"

        class _PluginConfig(WorldModelConfig):
            pass

        def _register_models():
            @WorldModelRegistry.register(model_type, _PluginConfig)
            class _PluginModel:
                def __init__(self, config):
                    self.config = config

            WorldModelRegistry.register_alias(alias, model_id)
            WorldModelRegistry.register_catalog_entry(
                model_id,
                {
                    "description": "External plugin demo model",
                    "params": "~0M",
                    "type": model_type,
                    "default_obs": "vector",
                    "maturity": "experimental",
                },
            )

        def _register_components():
            class _PluginActionConditioner:
                def condition(self, state, action, conditions=None):
                    del state, action, conditions
                    return {}

            WorldModelRegistry.register_component(
                component_id,
                _PluginActionConditioner,
                ComponentSpec(name="Plugin Zero Action", component_type="action_conditioner"),
            )

        fake_eps = _FakeEntryPoints(
            [
                _FakeEntryPoint("extplugin_models", "worldflux.models", _register_models),
                _FakeEntryPoint(
                    "extplugin_components", "worldflux.components", _register_components
                ),
            ]
        )
        monkeypatch.setattr(
            "worldflux.core.registry.importlib_metadata.entry_points",
            lambda: fake_eps,
        )

        WorldModelRegistry.load_entrypoint_plugins(force=True)
        try:
            assert model_type in WorldModelRegistry.list_models()
            assert component_id in WorldModelRegistry.list_components()
            assert WorldModelRegistry.resolve_alias(alias) == model_id
        finally:
            WorldModelRegistry.unregister_plugin_manifest("extplugin_models")
            WorldModelRegistry.unregister_plugin_manifest("extplugin_components")
            WorldModelRegistry.unregister_component(component_id)
            WorldModelRegistry.unregister_alias(alias)
            WorldModelRegistry.unregister_catalog_entry(model_id)
            WorldModelRegistry.unregister(model_type)

    def test_register_plugin_manifest_rejects_incompatible_worldflux_range(self):
        with pytest.raises(ConfigurationError, match="EXPERIMENTAL_PLUGIN_INCOMPATIBLE"):
            WorldModelRegistry.register_plugin_manifest(
                "tests.incompatible",
                PluginManifest(worldflux_version_range="<0.0.1"),
            )

    def test_load_entrypoint_plugins_rejects_incompatible_manifest(self, monkeypatch):
        def _register_with_bad_manifest():
            return PluginManifest(worldflux_version_range="<0.0.1")

        fake_eps = _FakeEntryPoints(
            [_FakeEntryPoint("extplugin_bad", "worldflux.models", _register_with_bad_manifest)]
        )
        monkeypatch.setattr(
            "worldflux.core.registry.importlib_metadata.entry_points",
            lambda: fake_eps,
        )

        with pytest.raises(ConfigurationError, match="EXPERIMENTAL_PLUGIN_INCOMPATIBLE"):
            WorldModelRegistry.load_entrypoint_plugins(force=True)

        WorldModelRegistry.unregister_plugin_manifest("extplugin_bad")
