# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Tests for WorldFlux Cloud credential storage."""

from __future__ import annotations

import os
import stat

import pytest

from worldflux.cloud.auth import credentials_path, load_credentials, set_api_key


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-specific")
def test_set_api_key_persists_with_restricted_permissions(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("WORLDFLUX_HOME", str(tmp_path / "wf-home"))

    set_api_key("wf_test_key")

    path = credentials_path()
    assert path.exists()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-specific")
def test_load_credentials_warns_when_file_permissions_are_too_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("WORLDFLUX_HOME", str(tmp_path / "wf-home"))
    path = credentials_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"api_key":"wf_test_key"}\n', encoding="utf-8")
    path.chmod(0o644)

    with pytest.warns(UserWarning, match="permissions"):
        payload = load_credentials()

    assert payload["api_key"] == "wf_test_key"
