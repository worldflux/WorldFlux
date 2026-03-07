"""Lazy exports for bundled world model implementations."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, str] = {
    "DreamerV3WorldModel": "worldflux.models.dreamer",
    "TDMPC2WorldModel": "worldflux.models.tdmpc2",
    "JEPABaseWorldModel": "worldflux.models.jepa",
    "VJEPA2WorldModel": "worldflux.models.vjepa2",
    "TokenWorldModel": "worldflux.models.token",
    "DiffusionWorldModel": "worldflux.models.diffusion",
    "DiTSkeletonWorldModel": "worldflux.models.dit",
    "SSMSkeletonWorldModel": "worldflux.models.ssm",
    "Renderer3DSkeletonWorldModel": "worldflux.models.renderer3d",
    "PhysicsSkeletonWorldModel": "worldflux.models.physics",
    "GANSkeletonWorldModel": "worldflux.models.gan",
}

__all__ = [
    "DreamerV3WorldModel",
    "TDMPC2WorldModel",
    "JEPABaseWorldModel",
    "VJEPA2WorldModel",
    "TokenWorldModel",
    "DiffusionWorldModel",
]


def __getattr__(name: str) -> Any:
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module = import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(_EXPORTS))
