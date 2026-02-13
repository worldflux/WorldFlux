"""Public contract snapshot capture and diff classification helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import worldflux
from worldflux import create_world_model, get_config, get_model_info, list_models
from worldflux.core.model import WorldModel


@dataclass(frozen=True)
class PublicContractDiff:
    """Classification result for a contract snapshot diff."""

    classification: str
    additive_changes: tuple[str, ...]
    breaking_reasons: tuple[str, ...]


def capture_public_contract_snapshot() -> dict[str, Any]:
    """Capture the current public contract surface used by freeze tests."""
    return {
        "worldflux_all": sorted(worldflux.__all__),
        "function_signatures": {
            "create_world_model": str(inspect.signature(create_world_model)),
            "list_models": str(inspect.signature(list_models)),
            "get_model_info": str(inspect.signature(get_model_info)),
            "get_config": str(inspect.signature(get_config)),
        },
        "worldmodel_signatures": {
            "encode": str(inspect.signature(WorldModel.encode)),
            "transition": str(inspect.signature(WorldModel.transition)),
            "update": str(inspect.signature(WorldModel.update)),
            "decode": str(inspect.signature(WorldModel.decode)),
            "rollout": str(inspect.signature(WorldModel.rollout)),
            "loss": str(inspect.signature(WorldModel.loss)),
            "io_contract": str(inspect.signature(WorldModel.io_contract)),
            "save_pretrained": str(inspect.signature(WorldModel.save_pretrained)),
            "from_pretrained": str(inspect.signature(WorldModel.from_pretrained)),
        },
    }


def classify_public_contract_diff(
    previous: dict[str, Any],
    current: dict[str, Any],
) -> PublicContractDiff:
    """Classify contract delta as none, additive, or breaking."""
    additive_changes: list[str] = []
    breaking_reasons: list[str] = []

    old_symbols = {
        str(symbol) for symbol in previous.get("worldflux_all", []) if isinstance(symbol, str)
    }
    new_symbols = {
        str(symbol) for symbol in current.get("worldflux_all", []) if isinstance(symbol, str)
    }

    removed_symbols = sorted(old_symbols - new_symbols)
    added_symbols = sorted(new_symbols - old_symbols)
    if removed_symbols:
        breaking_reasons.extend(f"Removed public symbol: {symbol}" for symbol in removed_symbols)
    if added_symbols:
        additive_changes.extend(f"Added public symbol: {symbol}" for symbol in added_symbols)

    for key in ("function_signatures", "worldmodel_signatures"):
        old_map_raw = previous.get(key, {})
        new_map_raw = current.get(key, {})
        old_map = (
            {str(name): str(sig) for name, sig in old_map_raw.items()}
            if isinstance(old_map_raw, dict)
            else {}
        )
        new_map = (
            {str(name): str(sig) for name, sig in new_map_raw.items()}
            if isinstance(new_map_raw, dict)
            else {}
        )

        missing_keys = sorted(set(old_map) - set(new_map))
        for name in missing_keys:
            breaking_reasons.append(f"Removed {key} entry: {name}")

        for name in sorted(set(old_map) & set(new_map)):
            if old_map[name] != new_map[name]:
                breaking_reasons.append(
                    f"Changed {key} signature for {name}: {old_map[name]!r} -> {new_map[name]!r}"
                )

        added_keys = sorted(set(new_map) - set(old_map))
        for name in added_keys:
            additive_changes.append(f"Added {key} entry: {name}")

    if breaking_reasons:
        classification = "breaking"
    elif additive_changes:
        classification = "additive"
    else:
        classification = "none"

    return PublicContractDiff(
        classification=classification,
        additive_changes=tuple(additive_changes),
        breaking_reasons=tuple(breaking_reasons),
    )
