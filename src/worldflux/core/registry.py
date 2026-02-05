"""Model registry and auto-loading utilities."""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from .config import WorldModelConfig
from .exceptions import ConfigurationError
from .interfaces import ComponentSpec
from .model import WorldModel


def _validate_config_json(config_path: str) -> dict[str, Any]:
    """
    Validate and load a config.json file.

    Args:
        config_path: Path to the config.json file.

    Returns:
        Validated configuration dictionary.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ConfigurationError: If config is invalid or missing required fields.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    try:
        with open(config_path) as f:
            config_dict = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigurationError(
            f"Invalid JSON in config file: {e}",
            config_name=config_path,
        ) from e

    # Validate required fields
    required_fields = ["model_type"]
    for field in required_fields:
        if field not in config_dict:
            raise ConfigurationError(
                f"Config missing required field: {field}",
                config_name=config_path,
            )

    return config_dict


class ConfigRegistry:
    """Registry for model configuration classes."""

    _registry: dict[str, type[WorldModelConfig]] = {}

    @classmethod
    def register(cls, model_type: str, config_class: type[WorldModelConfig]) -> None:
        cls._registry[model_type] = config_class

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> WorldModelConfig:
        if os.path.exists(name_or_path):
            config_path = os.path.join(name_or_path, "config.json")
            _validate_config_json(config_path)
            with open(config_path) as f:
                config_dict = json.load(f)
            model_type = config_dict.get("model_type", "base")
            config_class = cls._registry.get(model_type, WorldModelConfig)
            return config_class.from_dict(config_dict)  # type: ignore[arg-type]

        if ":" in name_or_path:
            model_type, size = name_or_path.split(":", 1)
            model_type = model_type.lower()
            if model_type not in cls._registry:
                alias_map = {"dreamerv3": "dreamer", "tdmpc": "tdmpc2"}
                model_type = alias_map.get(model_type, model_type)
            config_class = cls._registry.get(model_type, WorldModelConfig)
            if hasattr(config_class, "from_size"):
                return config_class.from_size(size, **kwargs)
            return config_class(model_name=size, **kwargs)

        raise ValueError(f"Invalid config identifier: {name_or_path}")


class WorldModelRegistry:
    """World model auto-registration and resolution system."""

    _model_registry: dict[str, type] = {}
    _aliases: dict[str, str] = {}
    _catalog: dict[str, dict[str, Any]] = {}
    _component_registry: dict[str, tuple[ComponentSpec, type]] = {}

    @classmethod
    def register(
        cls,
        model_type: str,
        config_class: type[WorldModelConfig] | None = None,
    ):
        def decorator(model_class: type):
            if model_type in cls._model_registry:
                existing_class = cls._model_registry[model_type]
                raise ConfigurationError(
                    f"Model type '{model_type}' is already registered to {existing_class.__name__}. "
                    f"Cannot re-register to {model_class.__name__}. "
                    "Use a different model_type or unregister the existing model first."
                )
            cls._model_registry[model_type] = model_class
            if config_class is not None:
                ConfigRegistry.register(model_type, config_class)
            return model_class

        return decorator

    @classmethod
    def register_alias(cls, name: str, target: str) -> None:
        cls._aliases[name.lower()] = target

    @classmethod
    def unregister_alias(cls, name: str) -> bool:
        return cls._aliases.pop(name.lower(), None) is not None

    @classmethod
    def register_catalog_entry(cls, model_id: str, info: dict[str, Any]) -> None:
        cls._catalog[model_id] = dict(info)

    @classmethod
    def unregister_catalog_entry(cls, model_id: str) -> bool:
        return cls._catalog.pop(model_id, None) is not None

    @classmethod
    def unregister(cls, model_type: str) -> bool:
        was_registered = model_type in cls._model_registry
        cls._model_registry.pop(model_type, None)
        return was_registered

    @classmethod
    def resolve_alias(cls, name: str) -> str:
        return cls._aliases.get(name.lower(), name)

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> WorldModel:
        if not cls._model_registry:
            try:
                import worldflux.models  # noqa: F401
            except Exception:
                pass
        if os.path.exists(name_or_path):
            config = ConfigRegistry.from_pretrained(name_or_path, **kwargs)
            if config.model_type not in cls._model_registry:
                raise ConfigurationError(
                    f"Model type '{config.model_type}' not registered. "
                    f"Available: {list(cls._model_registry.keys())}",
                    config_name=str(name_or_path),
                )
            model_class = cls._model_registry[config.model_type]
            model = model_class(config)
            weights_path = os.path.join(name_or_path, "model.pt")
            if os.path.exists(weights_path):
                model.load_state_dict(torch.load(weights_path, weights_only=True))
            return model

        resolved = cls.resolve_alias(name_or_path)
        if ":" in resolved:
            model_type, size = resolved.split(":", 1)
            model_type = model_type.lower()
            if model_type not in cls._model_registry:
                alias_map = {"dreamerv3": "dreamer", "tdmpc": "tdmpc2"}
                model_type = alias_map.get(model_type, model_type)
            if model_type not in cls._model_registry:
                raise ValueError(
                    f"Unknown model type: {model_type}. "
                    f"Available: {list(cls._model_registry.keys())}"
                )
            config = ConfigRegistry.from_pretrained(f"{model_type}:{size}", **kwargs)
            model_class = cls._model_registry[model_type]
            return model_class(config)

        raise ValueError(f"Invalid model identifier: {name_or_path}")

    @classmethod
    def list_models(cls) -> dict[str, type]:
        if not cls._model_registry:
            try:
                import worldflux.models  # noqa: F401
            except Exception:
                pass
        return dict(cls._model_registry)

    @classmethod
    def list_catalog(cls) -> dict[str, dict[str, Any]]:
        return dict(cls._catalog)

    @classmethod
    def register_component(
        cls,
        component_id: str,
        component_class: type,
        spec: ComponentSpec,
    ) -> None:
        if component_id in cls._component_registry:
            existing = cls._component_registry[component_id][1]
            raise ConfigurationError(
                f"Component '{component_id}' is already registered to {existing.__name__}. "
                f"Cannot re-register to {component_class.__name__}."
            )
        cls._component_registry[component_id] = (spec, component_class)

    @classmethod
    def unregister_component(cls, component_id: str) -> bool:
        return cls._component_registry.pop(component_id, None) is not None

    @classmethod
    def get_component(cls, component_id: str) -> tuple[ComponentSpec, type]:
        if component_id not in cls._component_registry:
            raise ConfigurationError(
                f"Unknown component id '{component_id}'. "
                f"Available: {sorted(cls._component_registry.keys())}"
            )
        return cls._component_registry[component_id]

    @classmethod
    def list_components(cls) -> dict[str, ComponentSpec]:
        return {component_id: spec for component_id, (spec, _) in cls._component_registry.items()}


class AutoWorldModel:
    """HuggingFace AutoModel-style alias."""

    @staticmethod
    def from_pretrained(name_or_path: str, **kwargs) -> WorldModel:
        return WorldModelRegistry.from_pretrained(name_or_path, **kwargs)


class AutoConfig:
    """HuggingFace AutoConfig-style alias."""

    @staticmethod
    def from_pretrained(name_or_path: str) -> WorldModelConfig:
        return ConfigRegistry.from_pretrained(name_or_path)
