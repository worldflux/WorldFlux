"""Tests for the WorldFlux Dreamer JAX proof runner."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_module(name: str, relative: str):
    script_path = Path(__file__).resolve().parents[2] / relative
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {relative}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_uses_official_checkout_with_required_flags(tmp_path: Path) -> None:
    mod = _load_module(
        "dreamer_worldflux_jax_runner", "scripts/parity/runtime/dreamer_worldflux_jax_runner.py"
    )

    class _Args:
        repo_root = tmp_path / "worldflux"
        official_repo_root = None
        run_dir = tmp_path / "run"
        task_id = "atari100k_pong"
        seed = 7
        steps = 110000
        device = "cpu"
        python_executable = "python3"

    expected_vendor_root = _Args.repo_root / "third_party" / "dreamerv3_official"
    resolved = mod._resolve_official_repo_root(_Args())
    assert resolved == expected_vendor_root.resolve()

    command = mod._build_command(_Args())
    assert str(expected_vendor_root / "dreamerv3" / "main.py") in command
    assert "--logdir" in command
    assert "--task" in command
    assert "atari100k_pong" in command
    assert "--seed" in command
    assert "7" in command
    assert "--run.steps" in command
    assert "110000" in command


def test_vendor_checkout_exists_in_repo() -> None:
    root = Path(__file__).resolve().parents[2]
    vendor_root = root / "third_party" / "dreamerv3_official"
    assert (vendor_root / "dreamerv3" / "main.py").exists()
    assert (vendor_root / "dreamerv3" / "agent.py").exists()
    assert (vendor_root / "embodied" / "__init__.py").exists()
    assert (vendor_root / "LICENSE").exists()


def test_vendored_dreamer_debug_dummy_cpu_smoke_runs_when_jax_available(tmp_path: Path) -> None:
    if sys.version_info >= (3, 13):
        return
    try:
        import jax  # noqa: F401
    except ImportError:
        return

    root = Path(__file__).resolve().parents[2]
    vendor_root = root / "third_party" / "dreamerv3_official"
    logdir = tmp_path / "smoke"
    cmd = [
        sys.executable,
        str(vendor_root / "dreamerv3" / "main.py"),
        "--configs",
        "debug",
        "--task",
        "dummy_disc",
        "--run.steps",
        "20",
        "--jax.platform",
        "cpu",
        "--logger.outputs",
        "jsonl",
        "--logdir",
        str(logdir),
    ]
    env = {"PYTHONPATH": str(vendor_root), **dict()}
    completed = subprocess.run(
        cmd,
        cwd=str(root),
        text=True,
        capture_output=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
