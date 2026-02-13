"""Tests for public contract automation helpers."""

from __future__ import annotations

from worldflux._internal.public_contract import classify_public_contract_diff


def test_classify_public_contract_diff_additive_when_symbol_added() -> None:
    previous = {
        "worldflux_all": ["create_world_model"],
        "function_signatures": {"create_world_model": "(model: str)"},
        "worldmodel_signatures": {"encode": "(self, obs)"},
    }
    current = {
        "worldflux_all": ["create_world_model", "new_symbol"],
        "function_signatures": {
            "create_world_model": "(model: str)",
            "new_fn": "(x: int)",
        },
        "worldmodel_signatures": {"encode": "(self, obs)"},
    }

    diff = classify_public_contract_diff(previous, current)
    assert diff.classification == "additive"
    assert any("new_symbol" in change for change in diff.additive_changes)
    assert diff.breaking_reasons == ()


def test_classify_public_contract_diff_breaking_on_signature_change() -> None:
    previous = {
        "worldflux_all": ["create_world_model"],
        "function_signatures": {"create_world_model": "(model: str)"},
        "worldmodel_signatures": {"encode": "(self, obs)"},
    }
    current = {
        "worldflux_all": ["create_world_model"],
        "function_signatures": {"create_world_model": "(model: str, strict: bool)"},
        "worldmodel_signatures": {"encode": "(self, obs)"},
    }

    diff = classify_public_contract_diff(previous, current)
    assert diff.classification == "breaking"
    assert any(
        "Changed function_signatures signature" in reason for reason in diff.breaking_reasons
    )
