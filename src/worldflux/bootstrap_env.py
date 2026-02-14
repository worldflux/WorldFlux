"""Cross-platform dependency bootstrap helpers for ``worldflux init``."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BOOTSTRAP_HOME_ENV = "WORLDFLUX_BOOTSTRAP_HOME"
INIT_ENSURE_DEPS_ENV = "WORLDFLUX_INIT_ENSURE_DEPS"
_DISABLED_VALUES = {"0", "false", "no", "off"}
_MAX_DIAGNOSTIC_LINES = 40
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class DependencyProfile:
    """Dependency requirements needed for a selected environment."""

    environment: str
    pip_specs: tuple[str, ...]
    import_modules: tuple[str, ...]


@dataclass(frozen=True)
class BootstrapRuntime:
    """Resolved location for the bootstrap virtual environment."""

    home_dir: Path
    venv_dir: Path
    python_executable: Path


@dataclass(frozen=True)
class EnsureDepsResult:
    """Result payload for dependency bootstrap attempts."""

    success: bool
    profile: DependencyProfile
    runtime: BootstrapRuntime | None
    launcher: str | None
    skipped: bool
    message: str
    retry_commands: tuple[str, ...]
    diagnostics: tuple[str, ...]


def _normalize_environment(value: Any) -> str:
    environment = str(value or "custom").strip().lower()
    if environment not in {"atari", "mujoco", "custom"}:
        return "custom"
    return environment


def _resolve_dependency_profile(environment: str) -> DependencyProfile:
    if environment == "atari":
        return DependencyProfile(
            environment="atari",
            pip_specs=("gymnasium", "ale-py"),
            import_modules=("gymnasium", "ale_py"),
        )
    if environment == "mujoco":
        return DependencyProfile(
            environment="mujoco",
            pip_specs=("gymnasium[mujoco]", "mujoco"),
            import_modules=("gymnasium", "mujoco"),
        )
    return DependencyProfile(environment="custom", pip_specs=(), import_modules=())


def _is_windows() -> bool:
    return os.name == "nt"


def _bootstrap_home_dir() -> Path:
    override = os.environ.get(BOOTSTRAP_HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser()

    if _is_windows():
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "WorldFlux" / "bootstrap"
        return Path.home() / "AppData" / "Local" / "WorldFlux" / "bootstrap"

    return Path.home() / ".worldflux" / "bootstrap"


def _runtime_python_executable(venv_dir: Path) -> Path:
    if _is_windows():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _command_to_text(args: list[str]) -> str:
    if _is_windows():
        return subprocess.list2cmdline(args)
    return " ".join(shlex.quote(token) for token in args)


def _trim_output(text: str) -> str:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if len(lines) <= _MAX_DIAGNOSTIC_LINES:
        return "\n".join(lines)
    return "\n".join(lines[-_MAX_DIAGNOSTIC_LINES:])


def _emit_progress(callback: ProgressCallback | None, message: str) -> None:
    if callback is None:
        return
    callback(message)


def _run_command(
    args: list[str],
    *,
    stream_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    if not stream_output:
        return subprocess.run(args, check=False, text=True, capture_output=True)

    try:
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as exc:
        return subprocess.CompletedProcess(args=args, returncode=1, stdout="", stderr=str(exc))

    if process.stdout is None:
        returncode = process.wait()
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout="", stderr="")

    lines: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    returncode = process.wait()
    return subprocess.CompletedProcess(
        args=args,
        returncode=returncode,
        stdout="".join(lines),
        stderr="",
    )


def _is_bootstrap_disabled() -> bool:
    raw = os.environ.get(INIT_ENSURE_DEPS_ENV, "1").strip().lower()
    return raw in _DISABLED_VALUES


def resolve_bootstrap_runtime() -> BootstrapRuntime:
    """Resolve the bootstrap virtual environment path for the current OS."""
    home_dir = _bootstrap_home_dir()
    py_tag = f"py{sys.version_info.major}{sys.version_info.minor}"
    venv_dir = home_dir / py_tag
    python_executable = _runtime_python_executable(venv_dir)
    return BootstrapRuntime(
        home_dir=home_dir,
        venv_dir=venv_dir,
        python_executable=python_executable,
    )


def _discover_local_worldflux_source_root() -> Path | None:
    candidates: list[Path] = []
    module_root_candidate = Path(__file__).resolve()
    if len(module_root_candidate.parents) >= 3:
        candidates.append(module_root_candidate.parents[2])
    candidates.append(Path.cwd().resolve())

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        pyproject = candidate / "pyproject.toml"
        if not pyproject.exists():
            continue
        try:
            content = pyproject.read_text(encoding="utf-8")
        except OSError:
            continue
        if 'name = "worldflux"' in content or "name = 'worldflux'" in content:
            return candidate
    return None


def _install_targets(profile: DependencyProfile) -> tuple[tuple[str, ...], Path | None]:
    source_root = _discover_local_worldflux_source_root()
    targets: list[str] = []
    if source_root is not None:
        targets.extend(["-e", str(source_root)])
    else:
        targets.append("worldflux")
    targets.extend(profile.pip_specs)
    return tuple(targets), source_root


def _ensure_virtualenv(
    runtime: BootstrapRuntime,
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[bool, tuple[str, ...]]:
    if runtime.python_executable.exists():
        _emit_progress(
            progress_callback, f"Using existing bootstrap environment: {runtime.venv_dir}"
        )
        return True, ()

    runtime.venv_dir.parent.mkdir(parents=True, exist_ok=True)
    create_cmd = [str(sys.executable), "-m", "venv", str(runtime.venv_dir)]
    _emit_progress(
        progress_callback,
        "Creating bootstrap environment (first run can take a bit): "
        + _command_to_text(create_cmd),
    )
    create_result = _run_command(create_cmd)
    if create_result.returncode == 0 and runtime.python_executable.exists():
        _emit_progress(progress_callback, f"Created bootstrap environment: {runtime.venv_dir}")
        return True, ()

    diagnostics = [
        f"Failed to create bootstrap venv: {runtime.venv_dir}",
        _trim_output(create_result.stdout),
        _trim_output(create_result.stderr),
    ]
    return False, tuple(item for item in diagnostics if item)


def verify_modules(python_executable: Path, modules: tuple[str, ...]) -> tuple[str, ...]:
    """Verify that each module can be imported by the given Python executable."""
    if not modules:
        return ()

    verify_code = "\n".join(
        [
            "import importlib.util",
            "import json",
            f"modules = {list(modules)!r}",
            "missing = [name for name in modules if importlib.util.find_spec(name) is None]",
            "print(json.dumps(missing))",
            "raise SystemExit(0 if not missing else 1)",
        ]
    )
    verify_cmd = [str(python_executable), "-c", verify_code]
    verify_result = _run_command(verify_cmd)
    if verify_result.returncode == 0:
        return ()

    raw = verify_result.stdout.strip().splitlines()
    if raw:
        last = raw[-1].strip()
        try:
            payload = json.loads(last)
            if isinstance(payload, list):
                missing = [str(item) for item in payload if str(item)]
                return tuple(missing)
        except json.JSONDecodeError:
            pass
    return modules


def ensure_init_dependencies(
    context: Mapping[str, Any],
    *,
    progress_callback: ProgressCallback | None = None,
) -> EnsureDepsResult:
    """Ensure worldflux + selected environment dependencies before project generation."""
    environment = _normalize_environment(context.get("environment", "custom"))
    profile = _resolve_dependency_profile(environment)

    if _is_bootstrap_disabled():
        return EnsureDepsResult(
            success=True,
            profile=profile,
            runtime=None,
            launcher=None,
            skipped=True,
            message=(
                f"Dependency bootstrap skipped because {INIT_ENSURE_DEPS_ENV}=0. "
                "Proceeding without pre-init dependency assurance."
            ),
            retry_commands=(),
            diagnostics=(),
        )

    runtime = resolve_bootstrap_runtime()
    _emit_progress(progress_callback, f"Resolved bootstrap runtime: {runtime.venv_dir}")
    install_targets, _source_root = _install_targets(profile)

    create_venv_cmd = [str(sys.executable), "-m", "venv", str(runtime.venv_dir)]
    install_cmd = [
        str(runtime.python_executable),
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *install_targets,
    ]
    retry_commands = (_command_to_text(create_venv_cmd), _command_to_text(install_cmd))

    created, creation_diagnostics = _ensure_virtualenv(
        runtime,
        progress_callback=progress_callback,
    )
    if not created:
        return EnsureDepsResult(
            success=False,
            profile=profile,
            runtime=runtime,
            launcher=None,
            skipped=False,
            message="Unable to prepare the bootstrap virtual environment.",
            retry_commands=retry_commands,
            diagnostics=creation_diagnostics,
        )

    package_summary = (
        ", ".join(profile.pip_specs) if profile.pip_specs else "core dependencies only"
    )
    _emit_progress(
        progress_callback,
        f"Installing bootstrap dependencies ({package_summary}). This may take a few minutes...",
    )
    _emit_progress(progress_callback, "Install command: " + _command_to_text(install_cmd))
    install_result = _run_command(install_cmd, stream_output=progress_callback is not None)
    if install_result.returncode != 0:
        diagnostics = tuple(
            item
            for item in (
                _trim_output(install_result.stdout),
                _trim_output(install_result.stderr),
            )
            if item
        )
        return EnsureDepsResult(
            success=False,
            profile=profile,
            runtime=runtime,
            launcher=None,
            skipped=False,
            message=(
                "Dependency installation failed while preparing the bootstrap environment "
                f"for {environment!r}."
            ),
            retry_commands=retry_commands,
            diagnostics=diagnostics,
        )

    required_modules = ("worldflux", *profile.import_modules)
    _emit_progress(progress_callback, "Verifying imports: " + ", ".join(required_modules))
    missing_modules = verify_modules(runtime.python_executable, required_modules)
    if missing_modules:
        return EnsureDepsResult(
            success=False,
            profile=profile,
            runtime=runtime,
            launcher=None,
            skipped=False,
            message=(
                "Dependency installation completed but required modules are still missing: "
                + ", ".join(missing_modules)
            ),
            retry_commands=retry_commands,
            diagnostics=(),
        )

    _emit_progress(progress_callback, "Dependency bootstrap completed successfully.")
    return EnsureDepsResult(
        success=True,
        profile=profile,
        runtime=runtime,
        launcher=str(runtime.python_executable),
        skipped=False,
        message=f"Bootstrap dependencies are ready ({package_summary}).",
        retry_commands=retry_commands,
        diagnostics=(),
    )


__all__ = [
    "DependencyProfile",
    "BootstrapRuntime",
    "EnsureDepsResult",
    "ensure_init_dependencies",
    "resolve_bootstrap_runtime",
    "verify_modules",
]
