"""Tests for modality-centric config normalization."""

from __future__ import annotations

from worldflux.core.config import WorldModelConfig


def test_base_config_normalizes_modalities_from_legacy_fields():
    config = WorldModelConfig(obs_shape=(8,), action_dim=3, action_type="continuous")
    assert config.observation_modalities["obs"]["shape"] == (8,)
    assert config.observation_modalities["obs"]["kind"] == "vector"
    assert config.action_spec["dim"] == 3
    assert config.action_spec["kind"] == "continuous"


def test_base_config_respects_explicit_modality_spec():
    config = WorldModelConfig(
        obs_shape=(3, 64, 64),
        action_dim=6,
        observation_modalities={
            "image": {"shape": (3, 64, 64), "kind": "image", "dtype": "float32"}
        },
        action_spec={"kind": "continuous", "dim": 6, "discrete": False},
    )
    assert config.observation_modalities["image"]["shape"] == (3, 64, 64)
    assert config.obs_shape == (3, 64, 64)
    assert config.action_dim == 6


def test_from_dict_converts_nested_shapes_to_tuple():
    config = WorldModelConfig.from_dict(
        {
            "model_type": "base",
            "model_name": "test",
            "obs_shape": [4],
            "action_dim": 2,
            "action_type": "continuous",
            "observation_modalities": {"obs": {"shape": [4], "kind": "vector"}},
            "action_spec": {"kind": "continuous", "dim": 2, "discrete": False},
        }
    )
    assert config.obs_shape == (4,)
    assert config.observation_modalities["obs"]["shape"] == (4,)
