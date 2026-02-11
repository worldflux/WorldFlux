"""Minimal installable plugin showing model and component extension hooks."""

from __future__ import annotations

from worldflux import PluginManifest
from worldflux.core import ComponentSpec, WorldModelRegistry
from worldflux.core.exceptions import ConfigurationError

MODEL_ALIAS = "minimalplugin-dreamer"
COMPONENT_ID = "minimal_plugin.zero_action_conditioner"


class ZeroActionConditioner:
    """Example action conditioner that discards action influence."""

    def condition(self, state, action, conditions=None):
        del state, action, conditions
        return {}


def register_models() -> PluginManifest:
    WorldModelRegistry.register_alias(MODEL_ALIAS, "dreamerv3:ci")
    WorldModelRegistry.register_catalog_entry(
        MODEL_ALIAS,
        {
            "description": "Minimal plugin alias for Dreamer CI preset",
            "params": "~0M",
            "type": "plugin-alias",
            "default_obs": "vector",
            "maturity": "experimental",
        },
    )
    return PluginManifest(
        plugin_api_version="0.x-experimental",
        worldflux_version_range=">=0.1.0,<0.2.0",
        capabilities=("model-alias",),
        experimental=True,
    )


def register_components() -> PluginManifest:
    try:
        WorldModelRegistry.register_component(
            COMPONENT_ID,
            ZeroActionConditioner,
            ComponentSpec(
                name="Minimal Zero Action Conditioner", component_type="action_conditioner"
            ),
        )
    except ConfigurationError:
        pass
    return PluginManifest(
        plugin_api_version="0.x-experimental",
        worldflux_version_range=">=0.1.0,<0.2.0",
        capabilities=("action-conditioner",),
        experimental=True,
    )
