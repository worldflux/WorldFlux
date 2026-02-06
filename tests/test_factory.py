"""Tests for the simple factory API."""

import warnings

import pytest
import torch

from worldflux import (
    MODEL_ALIASES,
    MODEL_CATALOG,
    create_world_model,
    get_config,
    get_model_info,
    list_models,
)
from worldflux.core.registry import WorldModelRegistry
from worldflux.models.dreamer import DreamerV3WorldModel
from worldflux.models.tdmpc2 import TDMPC2WorldModel
from worldflux.models.vjepa2 import VJEPA2WorldModel


class TestListModels:
    """Tests for list_models function."""

    def test_list_models_simple(self):
        """list_models returns list of model IDs."""
        models = list_models()
        assert isinstance(models, list)
        assert "dreamerv3:size12m" in models
        assert "tdmpc2:5m" in models
        assert "jepa:base" in models
        assert "vjepa2:base" in models
        assert "dit:base" in models
        assert "ssm:base" in models

    def test_list_models_verbose(self):
        """list_models with verbose returns detailed info."""
        models = list_models(verbose=True)
        assert isinstance(models, dict)
        assert "dreamerv3:size12m" in models
        assert "description" in models["dreamerv3:size12m"]
        assert "params" in models["dreamerv3:size12m"]

    def test_list_models_with_maturity_filter(self):
        ref_models = list_models(maturity="reference")
        exp_models = list_models(maturity="experimental")
        assert "dreamerv3:size12m" in ref_models
        assert "jepa:base" in exp_models
        assert "vjepa2:base" in exp_models


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_get_model_info_direct(self):
        """Get info for direct model ID."""
        info = get_model_info("dreamerv3:size12m")
        assert info["model_id"] == "dreamerv3:size12m"
        assert "description" in info
        assert "params" in info

    def test_get_model_info_alias(self):
        """Get info for model alias."""
        info = get_model_info("dreamer-large")
        assert info["model_id"] == "dreamerv3:size200m"
        assert info["alias"] == "dreamer-large"

    def test_get_model_info_unknown(self):
        """Unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            get_model_info("unknown-model")

    def test_get_model_info_jepa_alias(self):
        info = get_model_info("jepa")
        assert info["model_id"] == "jepa:base"

    def test_get_model_info_vjepa2_alias(self):
        info = get_model_info("vjepa2")
        assert info["model_id"] == "vjepa2:base"


class TestGetConfig:
    """Tests for get_config function."""

    def test_get_config_dreamer(self):
        """Get DreamerV3 config."""
        config = get_config("dreamerv3:size12m")
        assert config.model_type == "dreamer"
        assert config.deter_dim == 2048

    def test_get_config_tdmpc(self):
        """Get TD-MPC2 config."""
        config = get_config("tdmpc2:5m")
        assert config.model_type == "tdmpc2"
        assert config.latent_dim == 256

    def test_get_config_with_overrides(self):
        """Config with custom overrides."""
        config = get_config("tdmpc2:5m", obs_shape=(100,), action_dim=8)
        assert config.obs_shape == (100,)
        assert config.action_dim == 8
        assert config.observation_modalities["obs"]["shape"] == (100,)
        assert config.action_spec["dim"] == 8

    def test_get_config_alias(self):
        """Get config using alias."""
        config = get_config("dreamer-medium")
        assert config.model_name == "size50m"


class TestCreateWorldModel:
    """Tests for create_world_model function."""

    def test_create_dreamer_basic(self):
        """Create basic DreamerV3 model."""
        model = create_world_model("dreamerv3:size12m")
        assert isinstance(model, DreamerV3WorldModel)

    def test_create_tdmpc_basic(self):
        """Create basic TD-MPC2 model."""
        model = create_world_model("tdmpc2:5m", obs_shape=(39,))
        assert isinstance(model, TDMPC2WorldModel)

    def test_create_with_alias(self):
        """Create model using alias."""
        model = create_world_model("dreamer")
        assert isinstance(model, DreamerV3WorldModel)
        assert model.config.model_name == "size12m"

    def test_create_with_custom_config(self):
        """Create model with custom configuration."""
        model = create_world_model(
            "tdmpc2:5m",
            obs_shape=(50,),
            action_dim=4,
        )
        assert model.config.obs_shape == (50,)
        assert model.config.action_dim == 4

    def test_create_and_encode_dreamer(self):
        """Create DreamerV3 and run encode."""
        model = create_world_model("dreamer-small")
        obs = torch.randn(2, 3, 64, 64)
        state = model.encode(obs)
        assert state.tensors["deter"].shape[0] == 2

    def test_create_and_encode_tdmpc(self):
        """Create TD-MPC2 and run encode."""
        model = create_world_model("tdmpc-small", obs_shape=(39,))
        obs = torch.randn(2, 39)
        state = model.encode(obs)
        assert state.tensors["latent"].shape[0] == 2

    def test_create_with_device(self):
        """Create model on specific device."""
        model = create_world_model("dreamer", device="cpu")
        assert model.device == torch.device("cpu")

    def test_create_vjepa2_basic(self):
        model = create_world_model("vjepa2:base", obs_shape=(4,), action_dim=1)
        assert isinstance(model, VJEPA2WorldModel)

    def test_create_defaults_to_api_version_v3_without_warning(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model = create_world_model("tdmpc2:ci", obs_shape=(4,), action_dim=2)
        assert model.config.model_type == "tdmpc2"
        assert getattr(model, "_wf_api_version", None) == "v3"
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)

    def test_create_with_api_version_v3(self):
        model = create_world_model("tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="v3")
        assert model.config.model_type == "tdmpc2"
        assert getattr(model, "_wf_api_version", None) == "v3"

    def test_create_with_api_version_v02_warns(self):
        with pytest.warns(DeprecationWarning, match="api_version='v0.2'"):
            model = create_world_model(
                "tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="v0.2"
            )
        assert model.config.model_type == "tdmpc2"
        assert getattr(model, "_wf_api_version", None) == "v0.2"

    def test_create_with_hybrid_action_rejected_in_v3(self):
        with pytest.raises(ValueError, match="hybrid"):
            create_world_model(
                "tdmpc2:ci",
                obs_shape=(4,),
                action_dim=2,
                api_version="v3",
                action_spec={"kind": "hybrid", "dim": 2},
            )

    def test_create_with_invalid_api_version_raises(self):
        with pytest.raises(ValueError, match="Unsupported api_version"):
            create_world_model("tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="vX")


class TestModelAliases:
    """Tests for model aliases."""

    def test_all_aliases_resolve(self):
        """All aliases resolve to valid models."""
        for alias, target in MODEL_ALIASES.items():
            assert target in MODEL_CATALOG, f"Alias {alias} points to unknown model {target}"

    def test_dreamer_aliases(self):
        """DreamerV3 aliases are correct."""
        assert MODEL_ALIASES["dreamer"] == "dreamerv3:size12m"
        assert MODEL_ALIASES["dreamer-small"] == "dreamerv3:size12m"
        assert MODEL_ALIASES["dreamer-medium"] == "dreamerv3:size50m"
        assert MODEL_ALIASES["dreamer-large"] == "dreamerv3:size200m"

    def test_tdmpc_aliases(self):
        """TD-MPC2 aliases are correct."""
        assert MODEL_ALIASES["tdmpc"] == "tdmpc2:5m"
        assert MODEL_ALIASES["tdmpc-small"] == "tdmpc2:5m"
        assert MODEL_ALIASES["tdmpc-medium"] == "tdmpc2:48m"
        assert MODEL_ALIASES["tdmpc-large"] == "tdmpc2:317m"

    def test_jepa_alias(self):
        """JEPA alias is correct."""
        assert MODEL_ALIASES["jepa"] == "jepa:base"

    def test_vjepa2_alias(self):
        assert MODEL_ALIASES["vjepa2"] == "vjepa2:base"


class TestModelCatalog:
    """Tests for model catalog."""

    def test_catalog_has_all_dreamer_sizes(self):
        """Catalog has all DreamerV3 size presets."""
        expected = ["size12m", "size25m", "size50m", "size100m", "size200m"]
        for size in expected:
            assert f"dreamerv3:{size}" in MODEL_CATALOG

    def test_catalog_has_all_tdmpc_sizes(self):
        """Catalog has all TD-MPC2 size presets."""
        expected = ["5m", "19m", "48m", "317m"]
        for size in expected:
            assert f"tdmpc2:{size}" in MODEL_CATALOG

    def test_catalog_has_jepa(self):
        assert "jepa:base" in MODEL_CATALOG

    def test_catalog_has_vjepa2(self):
        assert "vjepa2:base" in MODEL_CATALOG

    def test_catalog_entries_have_required_fields(self):
        """All catalog entries have required fields."""
        required_fields = ["description", "params", "type", "default_obs"]
        for model_id, info in MODEL_CATALOG.items():
            for field in required_fields:
                assert field in info, f"Model {model_id} missing field {field}"

    def test_catalog_entries_have_maturity(self):
        for model_id, info in MODEL_CATALOG.items():
            assert info.get("maturity") in {"reference", "experimental", "skeleton"}, model_id

    def test_list_models_reflects_dynamic_registry_catalog_entries(self):
        model_id = "extplugin:demo"
        try:
            WorldModelRegistry.register_catalog_entry(
                model_id,
                {
                    "description": "External plugin demo model",
                    "params": "~0M",
                    "type": "extplugin",
                    "default_obs": "vector",
                    "maturity": "experimental",
                },
            )
            models = list_models()
            assert model_id in models
            assert get_model_info(model_id)["type"] == "extplugin"
        finally:
            WorldModelRegistry.unregister_catalog_entry(model_id)
