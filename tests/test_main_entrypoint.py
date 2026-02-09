"""Tests for `python -m worldflux` entrypoint behavior."""

from __future__ import annotations

import builtins
import sys
import types

import pytest

from worldflux import __main__ as worldflux_main


def test_main_invokes_cli_app(monkeypatch: pytest.MonkeyPatch) -> None:
    called: dict[str, str] = {}
    dummy_cli = types.ModuleType("worldflux.cli")

    def _app(*, prog_name: str) -> None:
        called["prog_name"] = prog_name

    dummy_cli.app = _app  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "worldflux.cli", dummy_cli)

    worldflux_main.main()
    assert called["prog_name"] == "worldflux"


def test_main_exits_with_hint_when_cli_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delitem(sys.modules, "worldflux.cli", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"worldflux.cli", "cli"}:
            err = ModuleNotFoundError("No module named 'typer'")
            err.name = "typer"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(SystemExit) as exc:
        worldflux_main.main()

    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "WorldFlux CLI dependencies are missing from this environment" in out
    assert "uv pip install -U worldflux" in out
    assert ".[cli]" not in out


def test_main_reraises_unexpected_module_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, "worldflux.cli", raising=False)
    real_import = builtins.__import__

    def _fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"worldflux.cli", "cli"}:
            err = ModuleNotFoundError("No module named 'yaml'")
            err.name = "yaml"
            raise err
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _fake_import)

    with pytest.raises(ModuleNotFoundError):
        worldflux_main.main()
