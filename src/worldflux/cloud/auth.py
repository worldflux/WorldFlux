"""Credential helpers for WorldFlux Cloud authentication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def credentials_path() -> Path:
    """Return the local credential file path."""
    root = Path(os.environ.get("WORLDFLUX_HOME", Path.home() / ".worldflux")).expanduser()
    return root / "credentials.json"


def load_credentials() -> dict[str, Any]:
    """Load credentials from disk."""
    path = credentials_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def save_credentials(payload: dict[str, Any]) -> None:
    """Persist credentials to disk."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def get_api_key() -> str | None:
    """Return stored API key if available."""
    key = load_credentials().get("api_key")
    if isinstance(key, str) and key.strip():
        return key.strip()
    return None


def set_api_key(api_key: str) -> None:
    """Store API key in local credentials file."""
    payload = load_credentials()
    payload["api_key"] = api_key.strip()
    save_credentials(payload)
