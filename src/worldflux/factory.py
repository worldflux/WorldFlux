# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""
Simple factory API for creating world models.

This module provides a LangChain/HuggingFace-style interface for easily
creating and switching between different world model implementations.

Example:
    from worldflux import create_world_model

    # Simple usage
    model = create_world_model("dreamerv3:size12m")

    # With custom configuration
    model = create_world_model(
        "tdmpc2:5m",
        obs_shape=(39,),
        action_dim=6,
    )

    # List available models
    from worldflux import list_models
    print(list_models())
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any, cast

from .core.backend_bridge import (
    NATIVE_TORCH_BACKEND,
    canonical_backend_profile,
    normalize_backend_hint,
    resolve_backend_execution,
)
from .core.backend_handle import OfficialBackendHandle
from .core.config import WorldModelConfig
from .core.exceptions import ConfigurationError
from .core.model import WorldModel
from .core.registry import ConfigRegistry, WorldModelRegistry
from .core.spec import ModelMaturity

# Model aliases for user convenience
MODEL_ALIASES: dict[str, str] = {
    # DreamerV3 aliases
    "dreamer": "dreamerv3:size12m",
    "dreamer-ci": "dreamer:ci",
    "dreamerv3": "dreamerv3:size12m",
    "dreamer-small": "dreamerv3:size12m",
    "dreamer-medium": "dreamerv3:size50m",
    "dreamer-large": "dreamerv3:size200m",
    "dreamerv3:official": "dreamerv3:official_xl",
    # TD-MPC2 aliases
    "tdmpc": "tdmpc2:5m",
    "tdmpc2-ci": "tdmpc2:ci",
    "tdmpc2": "tdmpc2:5m",
    "tdmpc-small": "tdmpc2:5m",
    "tdmpc-proof": "tdmpc2:proof_5m",
    "tdmpc2:proof": "tdmpc2:proof_5m",
    "tdmpc-legacy": "tdmpc2:5m_legacy",
    "tdmpc-medium": "tdmpc2:48m",
    "tdmpc-large": "tdmpc2:317m",
    # JEPA aliases
    "jepa": "jepa:base",
    # V-JEPA2 aliases
    "vjepa2": "vjepa2:base",
    "v-jepa2": "vjepa2:base",
    # Token model aliases
    "token": "token:base",
    # Diffusion model aliases
    "diffusion": "diffusion:base",
    # Skeleton families (cross-category adapters)
    "dit": "dit:base",
    "ssm": "ssm:base",
    "renderer3d": "renderer3d:base",
    "physics": "physics:base",
    "gan": "gan:base",
}

# Available model presets with descriptions
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "dreamer:ci": {
        "description": "Dreamer CI preset - small profile for quick validation and scaffolds (not proof-eligible)",
        "params": "~0.1M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "ci_only",
    },
    "dreamerv3:size12m": {
        "description": "DreamerV3 12M params - Good for simple environments",
        "params": "~12M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "dreamerv3:size25m": {
        "description": "DreamerV3 25M params - Balanced performance",
        "params": "~25M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "dreamerv3:size50m": {
        "description": "DreamerV3 50M params - Strong performance",
        "params": "~50M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "dreamerv3:size100m": {
        "description": "DreamerV3 100M params - High capacity",
        "params": "~100M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "dreamerv3:size200m": {
        "description": "DreamerV3 200M params - Maximum capacity",
        "params": "~200M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "dreamerv3:official_xl": {
        "description": "DreamerV3 official XL - canonical profile for Dreamer proof JAX backends",
        "params": "~200-300M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "proof_canonical",
    },
    "tdmpc2:5m": {
        "description": "TD-MPC2 5M params - Compatibility preset",
        "params": "~5M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "canonical_display_name": "TD-MPC2 5M Compatibility",
        "parity_role": "reference_family",
    },
    "tdmpc2:proof_5m": {
        "description": "TD-MPC2 proof canonical 5M params - official compare target",
        "params": "~5M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "canonical_display_name": "TD-MPC2 Proof 5M",
        "parity_role": "proof_canonical",
    },
    "tdmpc2:5m_legacy": {
        "description": "TD-MPC2 legacy 5M params - retained for compatibility",
        "params": "~5M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "canonical_display_name": "TD-MPC2 5M Legacy",
        "parity_role": "reference_family",
    },
    "tdmpc2:ci": {
        "description": "TD-MPC2 CI preset - small profile for quick validation and scaffolds (not proof-eligible)",
        "params": "~0.1M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "ci_only",
    },
    "tdmpc2:19m": {
        "description": "TD-MPC2 19M params - Balanced",
        "params": "~19M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "tdmpc2:48m": {
        "description": "TD-MPC2 48M params - Strong performance",
        "params": "~48M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "tdmpc2:317m": {
        "description": "TD-MPC2 317M params - Maximum capacity",
        "params": "~317M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
        "parity_role": "reference_family",
    },
    "jepa:base": {
        "description": "JEPA base model - Representation prediction",
        "params": "~1M+",
        "type": "jepa",
        "default_obs": "image",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
    "vjepa2:ci": {
        "description": "V-JEPA2 CI model - Fast representation smoke tests",
        "params": "~0.2M+",
        "type": "vjepa2",
        "default_obs": "image",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
    "vjepa2:tiny": {
        "description": "V-JEPA2 tiny model - Lightweight representation learning",
        "params": "~1M+",
        "type": "vjepa2",
        "default_obs": "image",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
    "vjepa2:base": {
        "description": "V-JEPA2 base model - Predictive representation learning",
        "params": "~4M+",
        "type": "vjepa2",
        "default_obs": "image",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
    "token:base": {
        "description": "Token world model - Discrete sequence modeling",
        "params": "~1M+",
        "type": "token",
        "default_obs": "token",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
    "diffusion:base": {
        "description": "Diffusion world model - Generative dynamics",
        "params": "~1M+",
        "type": "diffusion",
        "default_obs": "vector",
        "maturity": ModelMaturity.EXPERIMENTAL.value,
    },
    "dit:base": {
        "description": "DiT skeleton - Transformer diffusion interface validation",
        "params": "~1M+",
        "type": "dit",
        "default_obs": "image",
        "maturity": ModelMaturity.SKELETON.value,
    },
    "ssm:base": {
        "description": "SSM skeleton - Long-context latent dynamics interface validation",
        "params": "~1M+",
        "type": "ssm",
        "default_obs": "vector",
        "maturity": ModelMaturity.SKELETON.value,
    },
    "renderer3d:base": {
        "description": "Renderer3D skeleton - 3D/camera-conditioned contract validation",
        "params": "~1M+",
        "type": "renderer3d",
        "default_obs": "image",
        "maturity": ModelMaturity.SKELETON.value,
    },
    "physics:base": {
        "description": "Physics skeleton - differentiable transition/reward interface validation",
        "params": "~1M+",
        "type": "physics",
        "default_obs": "vector",
        "maturity": ModelMaturity.SKELETON.value,
    },
    "gan:base": {
        "description": "GAN skeleton - adversarial generative interface validation",
        "params": "~1M+",
        "type": "gan",
        "default_obs": "image",
        "maturity": ModelMaturity.SKELETON.value,
    },
}

_FACTORY_BOOTSTRAPPED = False


def _bootstrap_factory_registry() -> None:
    """Register bundled aliases/catalog entries only when the factory is used."""
    global _FACTORY_BOOTSTRAPPED
    if _FACTORY_BOOTSTRAPPED:
        return

    for alias, target in MODEL_ALIASES.items():
        WorldModelRegistry.register_alias(alias, target)
    for model_id, info in MODEL_CATALOG.items():
        WorldModelRegistry.register_catalog_entry(model_id, info)
    _FACTORY_BOOTSTRAPPED = True


# HuggingFace Hub kwargs that are NOT config fields and should be excluded
# from strict validation.
_HF_HUB_KWARGS: frozenset[str] = frozenset(
    {
        "revision",
        "token",
        "cache_dir",
        "local_files_only",
        "force_download",
        "allow_patterns",
        "ignore_patterns",
    }
)

# Internal factory kwargs that are consumed before reaching config.
_FACTORY_INTERNAL_KWARGS: frozenset[str] = frozenset(
    {
        "backend",
        "action_type",
    }
)


def _levenshtein(a: str, b: str) -> int:
    """Compute the Levenshtein (edit) distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr_row = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr_row.append(
                min(
                    curr_row[j] + 1,  # insert
                    prev_row[j + 1] + 1,  # delete
                    prev_row[j] + cost,  # replace
                )
            )
        prev_row = curr_row
    return prev_row[-1]


def _validate_kwargs(kwargs: dict[str, Any], config_cls: type) -> None:
    """Validate kwargs keys against dataclass fields of *config_cls*.

    Raises :class:`ConfigurationError` for unknown keys, with "Did you mean?"
    suggestions when the Levenshtein distance to a valid field is <= 2.

    HuggingFace Hub kwargs (``revision``, ``token``, etc.) and internal
    factory kwargs (``backend``) are silently excluded from validation.
    """
    valid_fields = {f.name for f in dataclasses.fields(config_cls)}
    excluded = _HF_HUB_KWARGS | _FACTORY_INTERNAL_KWARGS

    for key in kwargs:
        if key in excluded:
            continue
        if key not in valid_fields:
            suggestions = sorted(f for f in valid_fields if _levenshtein(key, f) <= 2)
            msg = f"Unknown parameter '{key}' for {config_cls.__name__}."
            if suggestions:
                msg += f" Did you mean: {', '.join(suggestions)}?"
            raise ConfigurationError(msg)


def _resolve_config_class(model_identifier: str) -> type:
    """Return the Config class for a model identifier like ``dreamerv3:size12m``.

    Falls back to :class:`WorldModelConfig` when no specific class is
    registered.
    """
    if ":" not in model_identifier:
        return WorldModelConfig
    model_type = model_identifier.split(":", 1)[0].lower()
    alias_map = {"dreamerv3": "dreamer", "tdmpc": "tdmpc2"}
    model_type = alias_map.get(model_type, model_type)
    # Ensure model modules are imported so config classes are registered.
    if not ConfigRegistry._registry:
        WorldModelRegistry._load_builtin_models()
    config_class = ConfigRegistry._registry.get(model_type, WorldModelConfig)
    return config_class


def _resolved_catalog() -> dict[str, dict[str, Any]]:
    """Return the current catalog view, including dynamically registered entries."""
    _bootstrap_factory_registry()
    # Ensure model modules are imported so plugin registrations run.
    WorldModelRegistry.list_models()
    catalog = dict(MODEL_CATALOG)
    catalog.update(WorldModelRegistry.list_catalog())
    return catalog


def create_world_model(
    model: str,
    *,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    observation_modalities: dict[str, dict[str, Any]] | None = None,
    action_spec: dict[str, Any] | None = None,
    component_overrides: dict[str, object] | None = None,
    device: str = "cpu",
    api_version: str = "v3",
    **kwargs: Any,
) -> WorldModel:
    """
    Create a world model with a simple, unified interface.

    This is the recommended way to create world models. It provides a clean,
    LangChain-style API that abstracts away implementation details.

    Args:
        model: Model identifier. Can be:
            - Full preset: "dreamerv3:size12m", "tdmpc2:5m"
            - Alias: "dreamer", "tdmpc", "dreamer-large"
            - Local path: "./my_trained_model"
        obs_shape: Observation shape. Default depends on model type:
            - DreamerV3: (3, 64, 64) for images
            - TD-MPC2: Must be specified for vector observations
        action_dim: Action dimension. Default: 6
        observation_modalities: Optional dict describing multi-modal
            observation inputs.  Keys are modality names and values are
            dicts with ``"kind"`` and ``"shape"`` entries, e.g.
            ``{"image": {"kind": "image", "shape": (3, 64, 64)}}``.
        action_spec: Optional dict overriding the default action
            specification.  Recognized keys include ``"kind"``
            (``"continuous"``, ``"discrete"``, etc.), ``"dim"``, and
            ``"num_actions"``.
        device: Device to place model on. Default: "cpu"
        component_overrides: Optional component-slot overrides. Values may be:
            - Registered component id (str)
            - Component class
            - Pre-built component instance
        **kwargs: Additional model-specific configuration

    Returns:
        WorldModel: Configured world model instance

    Examples:
        # Basic usage
        model = create_world_model("dreamerv3:size12m")

        # With custom observation space
        model = create_world_model(
            "tdmpc2:5m",
            obs_shape=(39,),
            action_dim=4,
        )

        # Using aliases
        model = create_world_model("dreamer-large")  # dreamerv3:size200m

        # Load trained model
        model = create_world_model("./checkpoints/my_model")
    """
    import torch

    _bootstrap_factory_registry()
    if api_version not in {"v0.2", "v3"}:
        raise ValueError(f"Unsupported api_version: {api_version}. Expected 'v0.2' or 'v3'.")
    if api_version == "v0.2":
        warnings.warn(
            "create_world_model(..., api_version='v0.2') enables legacy compatibility adapters. "
            "This mode is temporary and will be removed in v0.3.",
            DeprecationWarning,
            stacklevel=2,
        )
    else:
        action_kind = None
        if action_spec is not None:
            action_kind = action_spec.get("kind")
        if action_kind is None:
            action_kind = kwargs.get("action_type")
        if action_kind == "hybrid":
            raise ValueError("action kind 'hybrid' is not supported when api_version='v3'.")

    # Resolve aliases
    resolved_model = WorldModelRegistry.resolve_alias(model)

    # Validate kwargs before consuming them (API-01: strict validation)
    config_cls = _resolve_config_class(resolved_model)
    _validate_kwargs(kwargs, config_cls)

    backend_name = normalize_backend_hint(str(kwargs.pop("backend", NATIVE_TORCH_BACKEND)))

    # Build kwargs
    config_kwargs: dict[str, Any] = dict(kwargs)

    if obs_shape is not None:
        config_kwargs["obs_shape"] = obs_shape

    if action_dim is not None:
        config_kwargs["action_dim"] = action_dim

    if observation_modalities is not None:
        config_kwargs["observation_modalities"] = observation_modalities

    if action_spec is not None:
        config_kwargs["action_spec"] = action_spec

    if backend_name != NATIVE_TORCH_BACKEND:
        ConfigRegistry.from_pretrained(resolved_model, **config_kwargs)
    backend_execution = resolve_backend_execution(resolved_model, backend_name)

    if isinstance(backend_execution, OfficialBackendHandle):
        return cast(
            WorldModel,
            backend_execution.with_metadata(
                execution_kind="backend_handle",
                requested_device=str(device),
                api_version=api_version,
                backend_profile=canonical_backend_profile(resolved_model, backend_name),
            ),
        )

    # Create model
    world_model = WorldModelRegistry.from_pretrained(resolved_model, **config_kwargs)
    if component_overrides:
        WorldModelRegistry.apply_component_overrides(world_model, component_overrides)

    # Move to device
    if hasattr(world_model, "to"):
        world_model = world_model.to(torch.device(device))
    setattr(world_model, "_wf_api_version", api_version)
    setattr(
        world_model,
        "_wf_backend_handle",
        OfficialBackendHandle(
            backend=backend_name,
            model_id=resolved_model,
            metadata={
                "execution_kind": "native_model",
                "device": str(device),
                "api_version": api_version,
            },
        ),
    )
    setattr(world_model, "_wf_backend", NATIVE_TORCH_BACKEND)

    return world_model


def list_models(
    verbose: bool = False,
    maturity: str | None = None,
) -> list[str] | dict[str, dict[str, Any]]:
    """
    List all available world model presets.

    Args:
        verbose: If True, return detailed model information
        maturity: Optional maturity filter ("reference", "reference-family",
            "experimental", "skeleton")

    Returns:
        List of model names, or dict with detailed info if verbose=True

    Examples:
        # Simple list
        >>> list_models()
        ['dreamerv3:size12m', 'dreamerv3:size25m', ..., 'tdmpc2:317m']

        # With details
        >>> list_models(verbose=True)
        {
            'dreamerv3:size12m': {
                'description': 'DreamerV3 12M params - Good for simple environments',
                'params': '~12M',
                ...
            },
            ...
        }
    """
    catalog = _resolved_catalog()
    if maturity is not None:
        maturity = maturity.lower().replace("_", "-")
        if maturity == "reference-family":
            maturity = ModelMaturity.REFERENCE.value
        catalog = {k: v for k, v in catalog.items() if v.get("maturity") == maturity}
    else:
        # Default: exclude skeleton models (use maturity="skeleton" to see them)
        catalog = {
            k: v for k, v in catalog.items() if v.get("maturity") != ModelMaturity.SKELETON.value
        }
    if verbose:
        return catalog
    return list(catalog.keys())


def get_model_info(model: str) -> dict[str, Any]:
    """
    Get detailed information about a specific model.

    Args:
        model: Model identifier or alias

    Returns:
        Dictionary with model information

    Raises:
        ValueError: If model is not found
    """
    _bootstrap_factory_registry()
    resolved = WorldModelRegistry.resolve_alias(model)

    catalog = _resolved_catalog()
    if resolved in catalog:
        info = dict(catalog[resolved])
        info["model_id"] = resolved
        if model in MODEL_ALIASES:
            info["alias"] = model
        return info

    raise ValueError(f"Unknown model: {model}. Available models: {list(catalog.keys())}")


def get_config(
    model: str,
    *,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    **kwargs: Any,
) -> WorldModelConfig:
    """
    Get a configuration object without creating the model.

    Useful for inspecting or modifying configuration before model creation.

    Args:
        model: Model identifier or alias
        obs_shape: Override observation shape
        action_dim: Override action dimension
        **kwargs: Additional configuration overrides

    Returns:
        WorldModelConfig: Configuration object

    Examples:
        # Get config and inspect
        config = get_config("dreamerv3:size12m")
        print(config.deter_dim)  # 2048

        # Modify and create
        config = get_config("tdmpc2:5m", obs_shape=(100,))
        config.num_q_networks = 10  # Custom Q ensemble size
        model = DreamerV3WorldModel(config)  # or use registry
    """
    _bootstrap_factory_registry()
    resolved = WorldModelRegistry.resolve_alias(model)

    if ":" not in resolved:
        raise ValueError(
            f"Invalid model format: {model}. Expected 'type:size' format like 'dreamerv3:size12m'"
        )

    model_type, size = resolved.split(":", 1)

    # Build config kwargs
    config_kwargs = dict(kwargs)
    if obs_shape is not None:
        config_kwargs["obs_shape"] = obs_shape
    if action_dim is not None:
        config_kwargs["action_dim"] = action_dim

    # Create config from size preset
    return ConfigRegistry.from_pretrained(resolved, **config_kwargs)


# ---------------------------------------------------------------------------
# API-04: Builder Pattern (Fluent API)
# ---------------------------------------------------------------------------


class WorldModelBuilder:
    """Fluent builder for creating world models.

    Provides a step-by-step, method-chaining API as an alternative to the
    single-call :func:`create_world_model`.  Each setter validates its
    argument eagerly so errors surface at the point of misconfiguration.

    Example::

        model = (
            WorldModelBuilder("dreamerv3:size12m")
            .with_obs_shape((3, 64, 64))
            .with_action_dim(6)
            .with_device("cuda")
            .with_component("observation_encoder", MyEncoder)
            .build()
        )
    """

    def __init__(self, model: str) -> None:
        if not model or not isinstance(model, str):
            raise ConfigurationError("model identifier must be a non-empty string.")
        self._model = model
        self._obs_shape: tuple[int, ...] | None = None
        self._action_dim: int | None = None
        self._observation_modalities: dict[str, dict[str, Any]] | None = None
        self._action_spec: dict[str, Any] | None = None
        self._device: str = "cpu"
        self._api_version: str = "v3"
        self._component_overrides: dict[str, object] = {}
        self._extra_kwargs: dict[str, Any] = {}

    # -- Setters (return self for chaining) ---------------------------------

    def with_obs_shape(self, obs_shape: tuple[int, ...]) -> WorldModelBuilder:
        """Set observation shape, e.g. ``(3, 64, 64)``."""
        if not isinstance(obs_shape, tuple) or len(obs_shape) == 0:
            raise ConfigurationError("obs_shape must be a non-empty tuple of ints.")
        if any(not isinstance(d, int) or d <= 0 for d in obs_shape):
            raise ConfigurationError(
                f"obs_shape dimensions must be positive integers, got {obs_shape}."
            )
        self._obs_shape = obs_shape
        return self

    def with_action_dim(self, action_dim: int) -> WorldModelBuilder:
        """Set action dimension."""
        if not isinstance(action_dim, int) or action_dim <= 0:
            raise ConfigurationError(f"action_dim must be a positive integer, got {action_dim}.")
        self._action_dim = action_dim
        return self

    def with_device(self, device: str) -> WorldModelBuilder:
        """Set target device (``'cpu'``, ``'cuda'``, ``'cuda:0'``, etc.)."""
        if not isinstance(device, str) or not device:
            raise ConfigurationError("device must be a non-empty string.")
        self._device = device
        return self

    def with_api_version(self, api_version: str) -> WorldModelBuilder:
        """Set API version (``'v3'`` or ``'v0.2'``)."""
        if api_version not in {"v0.2", "v3"}:
            raise ConfigurationError(
                f"Unsupported api_version: {api_version}. Expected 'v0.2' or 'v3'."
            )
        self._api_version = api_version
        return self

    def with_observation_modalities(
        self, modalities: dict[str, dict[str, Any]]
    ) -> WorldModelBuilder:
        """Set multi-modal observation specification."""
        if not isinstance(modalities, dict) or len(modalities) == 0:
            raise ConfigurationError("observation_modalities must be a non-empty dict.")
        self._observation_modalities = modalities
        return self

    def with_action_spec(self, action_spec: dict[str, Any]) -> WorldModelBuilder:
        """Set action specification override."""
        if not isinstance(action_spec, dict):
            raise ConfigurationError("action_spec must be a dict.")
        self._action_spec = action_spec
        return self

    def with_component(self, slot: str, component: object) -> WorldModelBuilder:
        """Override a component slot (e.g. ``'observation_encoder'``)."""
        if not isinstance(slot, str) or not slot:
            raise ConfigurationError("component slot must be a non-empty string.")
        self._component_overrides[slot] = component
        return self

    def with_config(self, **kwargs: Any) -> WorldModelBuilder:
        """Pass arbitrary model-specific config overrides."""
        self._extra_kwargs.update(kwargs)
        return self

    # -- Terminal operation --------------------------------------------------

    def build(self) -> WorldModel:
        """Build and return the configured :class:`WorldModel`.

        Delegates to :func:`create_world_model` with all accumulated
        parameters.
        """
        return create_world_model(
            self._model,
            obs_shape=self._obs_shape,
            action_dim=self._action_dim,
            observation_modalities=self._observation_modalities,
            action_spec=self._action_spec,
            component_overrides=self._component_overrides or None,
            device=self._device,
            api_version=self._api_version,
            **self._extra_kwargs,
        )
