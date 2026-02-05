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

from typing import Any

from .core.config import WorldModelConfig
from .core.model import WorldModel
from .core.registry import ConfigRegistry, WorldModelRegistry
from .core.spec import ModelMaturity

# Model aliases for user convenience
MODEL_ALIASES: dict[str, str] = {
    # DreamerV3 aliases
    "dreamer": "dreamerv3:size12m",
    "dreamerv3": "dreamerv3:size12m",
    "dreamer-small": "dreamerv3:size12m",
    "dreamer-medium": "dreamerv3:size50m",
    "dreamer-large": "dreamerv3:size200m",
    # TD-MPC2 aliases
    "tdmpc": "tdmpc2:5m",
    "tdmpc2": "tdmpc2:5m",
    "tdmpc-small": "tdmpc2:5m",
    "tdmpc-medium": "tdmpc2:48m",
    "tdmpc-large": "tdmpc2:317m",
    # JEPA aliases
    "jepa": "jepa:base",
    # Token model aliases
    "token": "token:base",
    # Diffusion model aliases
    "diffusion": "diffusion:base",
}

# Available model presets with descriptions
MODEL_CATALOG: dict[str, dict[str, Any]] = {
    "dreamerv3:size12m": {
        "description": "DreamerV3 12M params - Good for simple environments",
        "params": "~12M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "dreamerv3:size25m": {
        "description": "DreamerV3 25M params - Balanced performance",
        "params": "~25M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "dreamerv3:size50m": {
        "description": "DreamerV3 50M params - Strong performance",
        "params": "~50M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "dreamerv3:size100m": {
        "description": "DreamerV3 100M params - High capacity",
        "params": "~100M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "dreamerv3:size200m": {
        "description": "DreamerV3 200M params - Maximum capacity",
        "params": "~200M",
        "type": "dreamer",
        "default_obs": "image",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "tdmpc2:5m": {
        "description": "TD-MPC2 5M params - Fast planning",
        "params": "~5M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "tdmpc2:19m": {
        "description": "TD-MPC2 19M params - Balanced",
        "params": "~19M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "tdmpc2:48m": {
        "description": "TD-MPC2 48M params - Strong performance",
        "params": "~48M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "tdmpc2:317m": {
        "description": "TD-MPC2 317M params - Maximum capacity",
        "params": "~317M",
        "type": "tdmpc2",
        "default_obs": "vector",
        "maturity": ModelMaturity.REFERENCE.value,
    },
    "jepa:base": {
        "description": "JEPA base model - Representation prediction",
        "params": "~1M+",
        "type": "jepa",
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
}

# Register aliases and catalog entries on import
for alias, target in MODEL_ALIASES.items():
    WorldModelRegistry.register_alias(alias, target)
for model_id, info in MODEL_CATALOG.items():
    WorldModelRegistry.register_catalog_entry(model_id, info)


def create_world_model(
    model: str,
    *,
    obs_shape: tuple[int, ...] | None = None,
    action_dim: int | None = None,
    observation_modalities: dict[str, dict[str, Any]] | None = None,
    action_spec: dict[str, Any] | None = None,
    device: str = "cpu",
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
        device: Device to place model on. Default: "cpu"
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

    # Resolve aliases
    resolved_model = WorldModelRegistry.resolve_alias(model)

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

    # Create model
    world_model = WorldModelRegistry.from_pretrained(resolved_model, **config_kwargs)

    # Move to device
    if hasattr(world_model, "to"):
        world_model = world_model.to(torch.device(device))

    return world_model


def list_models(
    verbose: bool = False,
    maturity: str | None = None,
) -> list[str] | dict[str, dict[str, Any]]:
    """
    List all available world model presets.

    Args:
        verbose: If True, return detailed model information
        maturity: Optional maturity filter ("reference", "experimental", "skeleton")

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
    catalog = dict(MODEL_CATALOG)
    if maturity is not None:
        maturity = maturity.lower()
        catalog = {k: v for k, v in catalog.items() if v.get("maturity") == maturity}
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
    resolved = WorldModelRegistry.resolve_alias(model)

    if resolved in MODEL_CATALOG:
        info = dict(MODEL_CATALOG[resolved])
        info["model_id"] = resolved
        if model in MODEL_ALIASES:
            info["alias"] = model
        return info

    raise ValueError(f"Unknown model: {model}. Available models: {list(MODEL_CATALOG.keys())}")


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
