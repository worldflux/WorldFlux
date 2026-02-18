"""Tests for parity family plugin registry."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "parity" / "suite_registry.py"
    spec = importlib.util.spec_from_file_location("suite_registry", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load suite_registry")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_default_registry_contains_core_families() -> None:
    mod = _load_module()
    registry = mod.build_default_registry()

    assert set(registry.families()) == {"dreamerv3", "tdmpc2"}


def test_plugins_normalize_tasks() -> None:
    mod = _load_module()
    registry = mod.build_default_registry()

    dreamer = registry.require("dreamerv3")
    tdmpc2 = registry.require("tdmpc2")

    assert dreamer.normalize_task("atari_pong") == "atari100k_pong"
    assert tdmpc2.normalize_task("dog_run") == "dog-run"


def test_registry_require_raises_for_unknown_family() -> None:
    mod = _load_module()
    registry = mod.build_default_registry()

    with pytest.raises(KeyError, match="No plugin registered"):
        registry.require("new_family")
