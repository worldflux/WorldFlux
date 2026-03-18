"""Runtime components for parity native online runners."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AtariEnv": ("runtime.atari_env", "AtariEnv"),
    "AtariEnvError": ("runtime.atari_env", "AtariEnvError"),
    "build_atari_env": ("runtime.atari_env", "build_atari_env"),
    "DMControlEnv": ("runtime.dmcontrol_env", "DMControlEnv"),
    "DMControlEnvError": ("runtime.dmcontrol_env", "DMControlEnvError"),
    "build_dmcontrol_env": ("runtime.dmcontrol_env", "build_dmcontrol_env"),
    "DreamerNativeRunConfig": ("runtime.dreamer_native_agent", "DreamerNativeRunConfig"),
    "run_dreamer_native": ("runtime.dreamer_native_agent", "run_dreamer_native"),
    "TDMPC2NativeRunConfig": ("runtime.tdmpc2_native_agent", "TDMPC2NativeRunConfig"),
    "run_tdmpc2_native": ("runtime.tdmpc2_native_agent", "run_tdmpc2_native"),
}


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))


__all__ = sorted(_EXPORTS)
