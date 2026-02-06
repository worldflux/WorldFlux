"""Model registry and auto-loading utilities."""

from __future__ import annotations

import inspect
import json
import os
import warnings
from importlib import metadata as importlib_metadata
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
    _plugins_loaded: bool = False
    _component_slots: dict[str, tuple[str, str]] = {
        "observation_encoder": ("observation_encoder", "observation_encoder"),
        "action_conditioner": ("action_conditioner", "action_conditioner"),
        "dynamics_model": ("dynamics_model", "dynamics_model"),
        "decoder": ("decoder_module", "decoder"),
        "decoder_module": ("decoder_module", "decoder"),
        "rollout_executor": ("rollout_executor", "rollout_executor"),
        # v0.2 compatibility alias.
        "rollout_engine": ("rollout_engine", "rollout_executor"),
    }

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

    @staticmethod
    def _load_saved_metadata(path: str) -> dict[str, Any] | None:
        meta_path = os.path.join(path, "worldflux_meta.json")
        if not os.path.exists(meta_path):
            return None
        with open(meta_path, encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ConfigurationError(
                "worldflux_meta.json must contain a JSON object.",
                config_name=meta_path,
            )
        return loaded

    @staticmethod
    def _validate_saved_metadata(meta: dict[str, Any], model: WorldModel, path: str) -> None:
        save_format_version = meta.get("save_format_version")
        if save_format_version not in (None, 1, "1"):
            raise ConfigurationError(
                f"Unsupported save_format_version in worldflux_meta.json: {save_format_version!r}",
                config_name=path,
            )

        expected_model_type = str(getattr(getattr(model, "config", None), "model_type", ""))
        saved_model_type = str(meta.get("model_type", ""))
        if expected_model_type and saved_model_type and expected_model_type != saved_model_type:
            raise ConfigurationError(
                "Saved metadata model_type mismatch: "
                f"expected {expected_model_type!r}, got {saved_model_type!r}.",
                config_name=path,
            )

        saved_fingerprint = meta.get("contract_fingerprint")
        if saved_fingerprint is not None:
            if not hasattr(model, "contract_fingerprint"):
                raise ConfigurationError(
                    "Model does not expose contract_fingerprint() required for compatibility check.",
                    config_name=path,
                )
            current_fingerprint = str(model.contract_fingerprint())  # type: ignore[attr-defined]
            if str(saved_fingerprint) != current_fingerprint:
                raise ConfigurationError(
                    "Saved metadata contract_fingerprint does not match current model contract. "
                    "This artifact is not compatible with the current runtime.",
                    config_name=path,
                )

    @classmethod
    def from_pretrained(cls, name_or_path: str, **kwargs) -> WorldModel:
        cls.load_entrypoint_plugins()
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
            metadata = cls._load_saved_metadata(name_or_path)
            if metadata is not None:
                cls._validate_saved_metadata(metadata, model, name_or_path)
                saved_api_version = metadata.get("api_version")
                if saved_api_version is not None:
                    setattr(model, "_wf_api_version", str(saved_api_version))
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
        cls.load_entrypoint_plugins()
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
        cls.load_entrypoint_plugins()
        if component_id not in cls._component_registry:
            raise ConfigurationError(
                f"Unknown component id '{component_id}'. "
                f"Available: {sorted(cls._component_registry.keys())}"
            )
        return cls._component_registry[component_id]

    @classmethod
    def list_components(cls) -> dict[str, ComponentSpec]:
        cls.load_entrypoint_plugins()
        return {component_id: spec for component_id, (spec, _) in cls._component_registry.items()}

    @staticmethod
    def _iter_entry_points(group: str) -> list[Any]:
        try:
            entries = importlib_metadata.entry_points()
        except Exception:
            return []
        if hasattr(entries, "select"):
            return list(entries.select(group=group))
        group_entries: Any = entries.get(group, [])  # pragma: no cover - legacy importlib API
        return list(group_entries)

    @classmethod
    def _load_entrypoint_group(cls, group: str) -> None:
        for entry_point in cls._iter_entry_points(group):
            try:
                loaded = entry_point.load()
                if callable(loaded):
                    loaded()
            except Exception as exc:
                warnings.warn(
                    f"Failed to load entry point '{entry_point.name}' from group '{group}': {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    @classmethod
    def load_entrypoint_plugins(cls, *, force: bool = False) -> None:
        """Load external plugin registration hooks from package entry points."""
        if cls._plugins_loaded and not force:
            return
        cls._load_entrypoint_group("worldflux.models")
        cls._load_entrypoint_group("worldflux.components")
        cls._plugins_loaded = True

    @staticmethod
    def _instantiate_component(component_class: type, model: WorldModel) -> Any:
        """Instantiate a component class with either ``(model)`` or ``()`` signature."""
        init_method = getattr(component_class, "__init__", None)
        if init_method is object.__init__:
            return component_class()

        try:
            init_sig = inspect.signature(component_class.__init__)  # type: ignore[misc]
        except (TypeError, ValueError):
            init_sig = None

        if init_sig is not None:
            params = list(init_sig.parameters.values())[1:]  # skip `self`
            required_positional = [
                p
                for p in params
                if p.default is inspect._empty
                and p.kind
                in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            has_var_positional = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            if not required_positional and not has_var_positional:
                return component_class()

        try:
            return component_class(model)
        except TypeError:
            return component_class()

    @classmethod
    def build_component(
        cls,
        override: object,
        *,
        expected_component_type: str,
        model: WorldModel,
    ) -> Any:
        """Build a component override from registry id, class, or pre-built instance."""
        if isinstance(override, str):
            spec, component_class = cls.get_component(override)
            if spec.component_type != expected_component_type:
                raise ConfigurationError(
                    f"Component '{override}' has type {spec.component_type!r}, "
                    f"expected {expected_component_type!r}."
                )
            return cls._instantiate_component(component_class, model)
        if isinstance(override, type):
            return cls._instantiate_component(override, model)
        return override

    @classmethod
    def apply_component_overrides(
        cls,
        model: WorldModel,
        component_overrides: dict[str, object],
    ) -> None:
        """Apply component overrides to a world model instance."""
        for slot, override in component_overrides.items():
            slot_meta = cls._component_slots.get(slot)
            if slot_meta is None:
                raise ConfigurationError(
                    f"Unknown component slot '{slot}'. "
                    f"Available: {sorted(cls._component_slots.keys())}"
                )
            attr_name, expected_type = slot_meta
            built_component = cls.build_component(
                override,
                expected_component_type=expected_type,
                model=model,
            )
            setattr(model, attr_name, built_component)
            if slot == "rollout_engine":
                # Keep deprecated alias and new slot in sync.
                setattr(model, "rollout_executor", built_component)


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
