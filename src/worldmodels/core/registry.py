"""Model registry and auto-loading utilities."""

import os

import torch

from .config import DreamerV3Config, TDMPC2Config, WorldModelConfig
from .protocol import WorldModel


class WorldModelRegistry:
    """
    World model auto-registration and resolution system.

    Usage:
        @WorldModelRegistry.register("dreamer")
        class DreamerWorldModel(nn.Module):
            ...

        model = WorldModelRegistry.from_pretrained("dreamerv3:size12m")
    """

    _model_registry: dict[str, type] = {}
    _config_registry: dict[str, type[WorldModelConfig]] = {}
    _presets: dict[str, dict] = {}

    @classmethod
    def register(cls, model_type: str, config_class: type[WorldModelConfig] | None = None):
        """Register a model class with decorator."""

        def decorator(model_class: type):
            cls._model_registry[model_type] = model_class
            if config_class is not None:
                cls._config_registry[model_type] = config_class
            return model_class

        return decorator

    @classmethod
    def register_preset(cls, name: str, config: dict):
        """Register a preset configuration."""
        cls._presets[name] = config

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> WorldModel:
        """
        Load model from preset or path.

        Args:
            name_or_path:
                - "dreamerv3:size12m" - registered preset
                - "tdmpc2:5m" - size preset
                - "./path/to/model" - local path
        """
        # Local path
        if os.path.exists(name_or_path):
            config_path = os.path.join(name_or_path, "config.json")
            config = WorldModelConfig.load(config_path)
            model_class = cls._model_registry[config.model_type]
            model = model_class(config)

            weights_path = os.path.join(name_or_path, "model.pt")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, weights_only=True))

            return model

        # Preset format "model:size"
        if ":" in name_or_path:
            model_type, size = name_or_path.split(":", 1)

            type_map = {
                "dreamerv3": "dreamer",
                "dreamer": "dreamer",
                "tdmpc2": "tdmpc2",
                "tdmpc": "tdmpc2",
            }
            normalized_type = type_map.get(model_type.lower(), model_type.lower())

            if normalized_type not in cls._model_registry:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available: {list(cls._model_registry.keys())}"
                )

            model_class = cls._model_registry[normalized_type]
            config_class = cls._config_registry.get(normalized_type, WorldModelConfig)

            if hasattr(config_class, "from_size"):
                config = config_class.from_size(size, **kwargs)
            else:
                config = config_class(model_name=size, **kwargs)

            return model_class(config)

        raise ValueError(f"Invalid model identifier: {name_or_path}")

    @classmethod
    def list_models(cls) -> dict[str, type]:
        """List registered models."""
        return dict(cls._model_registry)


class AutoWorldModel:
    """HuggingFace AutoModel-style alias."""

    @staticmethod
    def from_pretrained(name_or_path: str, **kwargs) -> WorldModel:
        return WorldModelRegistry.from_pretrained(name_or_path, **kwargs)


class AutoConfig:
    """HuggingFace AutoConfig-style alias."""

    @staticmethod
    def from_pretrained(name_or_path: str) -> WorldModelConfig:
        if os.path.exists(name_or_path):
            config_path = os.path.join(name_or_path, "config.json")
            return WorldModelConfig.load(config_path)

        if ":" in name_or_path:
            model_type, size = name_or_path.split(":", 1)
            type_map = {"dreamerv3": "dreamer", "tdmpc2": "tdmpc2"}
            normalized = type_map.get(model_type.lower(), model_type.lower())

            config_map = {
                "dreamer": DreamerV3Config,
                "tdmpc2": TDMPC2Config,
            }
            config_class = config_map.get(normalized, WorldModelConfig)

            if hasattr(config_class, "from_size"):
                return config_class.from_size(size)
            return config_class(model_name=size)

        raise ValueError(f"Invalid config identifier: {name_or_path}")
