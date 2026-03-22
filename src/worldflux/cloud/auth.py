# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Credential helpers for WorldFlux Cloud authentication."""

from __future__ import annotations

import json
import os
import stat
import warnings
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
    _warn_if_permissions_are_too_open(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def save_credentials(payload: dict[str, Any]) -> None:
    """Persist credentials to disk."""
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _chmod_if_posix(path.parent, 0o700)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _chmod_if_posix(path, 0o600)


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


def _chmod_if_posix(path: Path, mode: int) -> None:
    if os.name == "posix":
        path.chmod(mode)


def _warn_if_permissions_are_too_open(path: Path) -> None:
    if os.name != "posix":
        return
    current_mode = stat.S_IMODE(path.stat().st_mode)
    if current_mode & 0o077:
        warnings.warn(
            f"WorldFlux Cloud credential file permissions are too open: {oct(current_mode)}",
            UserWarning,
            stacklevel=2,
        )
