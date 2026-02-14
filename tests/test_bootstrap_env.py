"""Tests for init dependency bootstrap runtime."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import worldflux.bootstrap_env as bootstrap_env


def test_resolve_bootstrap_runtime_uses_override_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(bootstrap_env.BOOTSTRAP_HOME_ENV, str(tmp_path / "bootstrap-home"))
    runtime = bootstrap_env.resolve_bootstrap_runtime()
    assert runtime.home_dir == (tmp_path / "bootstrap-home")
    assert (
        runtime.venv_dir == runtime.home_dir / f"py{sys.version_info.major}{sys.version_info.minor}"
    )
    assert runtime.python_executable == bootstrap_env._runtime_python_executable(runtime.venv_dir)


def test_bootstrap_home_dir_windows_defaults_to_localappdata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(bootstrap_env.BOOTSTRAP_HOME_ENV, raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "LocalAppData"))
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: True)
    assert (
        bootstrap_env._bootstrap_home_dir() == tmp_path / "LocalAppData" / "WorldFlux" / "bootstrap"
    )


def test_bootstrap_home_dir_posix_defaults_to_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv(bootstrap_env.BOOTSTRAP_HOME_ENV, raising=False)
    monkeypatch.delenv("LOCALAPPDATA", raising=False)
    monkeypatch.setattr(bootstrap_env, "_is_windows", lambda: False)
    monkeypatch.setattr(bootstrap_env.Path, "home", staticmethod(lambda: tmp_path / "home"))
    assert bootstrap_env._bootstrap_home_dir() == tmp_path / "home" / ".worldflux" / "bootstrap"


def test_verify_modules_reports_missing_module() -> None:
    missing = bootstrap_env.verify_modules(
        Path(sys.executable), ("json", "worldflux", "__missing_module_for_worldflux_tests__")
    )
    assert missing == ("__missing_module_for_worldflux_tests__",)


def test_ensure_init_dependencies_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap_env.INIT_ENSURE_DEPS_ENV, "0")
    result = bootstrap_env.ensure_init_dependencies({"environment": "atari"})
    assert result.success is True
    assert result.skipped is True
    assert result.runtime is None
    assert result.launcher is None


def test_ensure_init_dependencies_installs_worldflux_and_env_specs(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(bootstrap_env.INIT_ENSURE_DEPS_ENV, raising=False)

    runtime = bootstrap_env.BootstrapRuntime(
        home_dir=tmp_path / "bootstrap",
        venv_dir=tmp_path / "bootstrap" / "py311",
        python_executable=tmp_path / "bootstrap" / "py311" / "bin" / "python",
    )
    commands: list[list[str]] = []
    stream_flags: list[bool] = []
    monkeypatch.setattr(bootstrap_env, "resolve_bootstrap_runtime", lambda: runtime)
    monkeypatch.setattr(bootstrap_env, "_ensure_virtualenv", lambda _runtime, **_kwargs: (True, ()))
    monkeypatch.setattr(bootstrap_env, "_discover_local_worldflux_source_root", lambda: None)
    monkeypatch.setattr(bootstrap_env, "verify_modules", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        bootstrap_env,
        "_run_command",
        lambda args, **kwargs: (
            stream_flags.append(bool(kwargs.get("stream_output", False)))
            or commands.append(list(args))
            or subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        ),
    )

    progress_messages: list[str] = []
    result = bootstrap_env.ensure_init_dependencies(
        {"environment": "atari"},
        progress_callback=progress_messages.append,
    )

    assert result.success is True
    assert result.launcher == str(runtime.python_executable)
    install_cmd = commands[-1]
    assert install_cmd[:5] == [
        str(runtime.python_executable),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
    ]
    assert "worldflux" in install_cmd
    assert "gymnasium" in install_cmd
    assert "ale-py" in install_cmd
    assert stream_flags[-1] is True
    assert any("Installing bootstrap dependencies" in message for message in progress_messages)


def test_ensure_init_dependencies_installs_only_worldflux_for_custom(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(bootstrap_env.INIT_ENSURE_DEPS_ENV, raising=False)

    runtime = bootstrap_env.BootstrapRuntime(
        home_dir=tmp_path / "bootstrap",
        venv_dir=tmp_path / "bootstrap" / "py311",
        python_executable=tmp_path / "bootstrap" / "py311" / "bin" / "python",
    )
    commands: list[list[str]] = []
    monkeypatch.setattr(bootstrap_env, "resolve_bootstrap_runtime", lambda: runtime)
    monkeypatch.setattr(bootstrap_env, "_ensure_virtualenv", lambda _runtime, **_kwargs: (True, ()))
    monkeypatch.setattr(bootstrap_env, "_discover_local_worldflux_source_root", lambda: None)
    monkeypatch.setattr(bootstrap_env, "verify_modules", lambda *_args, **_kwargs: ())
    monkeypatch.setattr(
        bootstrap_env,
        "_run_command",
        lambda args, **_kwargs: (
            commands.append(list(args))
            or subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")
        ),
    )

    result = bootstrap_env.ensure_init_dependencies({"environment": "custom"})

    assert result.success is True
    install_cmd = commands[-1]
    assert "worldflux" in install_cmd
    assert "gymnasium" not in install_cmd
    assert "ale-py" not in install_cmd
    assert "mujoco" not in install_cmd


def test_ensure_init_dependencies_failure_on_venv_creation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv(bootstrap_env.INIT_ENSURE_DEPS_ENV, raising=False)
    runtime = bootstrap_env.BootstrapRuntime(
        home_dir=tmp_path / "bootstrap",
        venv_dir=tmp_path / "bootstrap" / "py311",
        python_executable=tmp_path / "bootstrap" / "py311" / "bin" / "python",
    )
    monkeypatch.setattr(bootstrap_env, "resolve_bootstrap_runtime", lambda: runtime)
    monkeypatch.setattr(
        bootstrap_env,
        "_ensure_virtualenv",
        lambda _runtime, **_kwargs: (False, ("failed to create venv",)),
    )

    result = bootstrap_env.ensure_init_dependencies({"environment": "mujoco"})
    assert result.success is False
    assert result.skipped is False
    assert "Unable to prepare the bootstrap virtual environment" in result.message
    assert result.retry_commands
