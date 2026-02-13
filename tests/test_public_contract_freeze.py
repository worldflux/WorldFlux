"""Freeze tests for public API symbols and core signatures."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from worldflux import create_world_model
from worldflux._internal.public_contract import capture_public_contract_snapshot

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PATH = REPO_ROOT / "tests" / "fixtures" / "public_contract_snapshot.json"


def _load_snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_public_contract_symbols_and_signatures_are_frozen() -> None:
    assert capture_public_contract_snapshot() == _load_snapshot()


def test_api_version_bridge_remains_v3_and_v02_compatible() -> None:
    model_v3 = create_world_model("tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="v3")
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        model_v02 = create_world_model(
            "tdmpc2:ci", obs_shape=(4,), action_dim=2, api_version="v0.2"
        )

    assert getattr(model_v3, "_wf_api_version", None) == "v3"
    assert getattr(model_v02, "_wf_api_version", None) == "v0.2"

    contract_v3 = model_v3.io_contract()
    contract_v02 = model_v02.io_contract()

    assert contract_v3.required_batch_keys == contract_v02.required_batch_keys
    assert contract_v3.required_state_keys == contract_v02.required_state_keys
    assert model_v3.contract_fingerprint() == model_v02.contract_fingerprint()
