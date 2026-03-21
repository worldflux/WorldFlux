# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 WorldFlux Contributors
"""Consistency checks for documentation and quality gate policy."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_public_docs_include_cpu_success_and_benchmark_paths():
    readme = _read("README.md")
    index = _read("docs/index.md")
    parity = _read("docs/reference/parity.md")
    tutorial_policy = _read("docs/reference/tutorial-policy.md")

    assert "CPU-First Success Path" in readme
    assert "quickstart_cpu_success.py" in readme
    assert "(tutorials/train-first-model.md)" in index
    assert "(getting-started/cpu-success.md)" in index
    assert "(reference/benchmarks.md)" in index
    assert "(reference/parity.md)" in index
    assert "worldflux parity run" in parity
    assert "Promoted tutorials must contain runnable guidance" in tutorial_policy


def test_installation_docs_bound_windows_mvp_claims() -> None:
    readme = _read("README.md")
    installation = _read("docs/getting-started/installation.md")

    assert "supported newcomer path" in readme
    assert "Windows bootstrap support exists in implementation" in installation
    assert "not yet part of the current newcomer E2E guarantee" in installation


def test_quality_docs_reference_current_gate_commands():
    gates = _read("docs/reference/quality-gates.md")
    checklist = _read("docs/reference/release-checklist.md")
    publishing = _read("docs/reference/publishing.md")

    assert "uvx ruff check src/ tests/ examples/ benchmarks/ scripts/" in gates
    assert "uvx ruff format --check src/ tests/ examples/ benchmarks/ scripts/" in gates
    assert "uv run mypy src/worldflux/" in gates
    assert "uv run pytest tests/" in gates
    assert "npm audit --audit-level=high" in gates
    assert "npm run build" in gates
    assert "scripts/generate_release_parity_fixtures.py" in gates

    assert "scripts/run_release_dry_run.py" in checklist
    assert "scripts/generate_release_parity_fixtures.py" in checklist
    assert "examples/quickstart_cpu_success.py --quick" in checklist
    assert "examples/compare_unified_training.py --quick" in checklist
    assert "benchmarks/benchmark_dreamerv3_atari.py --quick" in checklist
    assert "scripts/run_release_dry_run.py" in publishing
    assert "scripts/generate_release_parity_fixtures.py" in publishing


def test_ci_includes_strict_docs_gate_and_new_smokes():
    ci = _read(".github/workflows/ci.yml")
    assert "docs:" in ci
    assert "npm ci" in ci
    assert "npm audit --audit-level=high" in ci
    assert "npm run build" in ci
    assert "--root-dir website/static" in ci
    assert "examples/quickstart_cpu_success.py --quick" in ci
    assert "examples/compare_unified_training.py --quick" in ci


def test_cli_installation_docs_prefer_direct_worldflux_command():
    readme = _read("README.md")
    installation = _read("docs/getting-started/installation.md")
    index = _read("docs/index.md")

    assert "uv tool install worldflux" in readme
    assert "worldflux init my-world-model" in readme

    assert "uv tool install worldflux" in installation
    assert "worldflux init my-world-model" in installation

    assert "uv tool install worldflux" in index
    assert "worldflux init my-world-model" in index

    assert "uv run worldflux init" not in readme
    assert "uv run worldflux init" not in installation


def test_pyproject_uses_default_cli_dependencies_and_optional_inquirer():
    pyproject = tomllib.loads(_read("pyproject.toml"))
    dependencies = pyproject["project"]["dependencies"]
    cli_extra = pyproject["project"]["optional-dependencies"]["cli"]

    assert any(dep.startswith("typer") for dep in dependencies)
    assert any(dep.startswith("rich") for dep in dependencies)
    assert cli_extra == ["inquirerpy>=0.3.4"]


def test_public_surfaces_use_conservative_claim_language():
    readme = _read("README.md")
    index = _read("docs/index.md")
    cpu = _read("docs/getting-started/cpu-success.md")
    comparison = _read("docs/reference/unified-comparison.md")
    comparison_tutorial = _read("docs/tutorials/dreamer-vs-tdmpc2.md")
    parity = _read("docs/reference/parity.md")

    assert "Infinite Imagination" not in readme
    assert "Infinite Imagination" not in index
    assert "parity-verified" not in readme
    assert "docs/assets/dogrun.gif" not in readme
    assert "Reference-family" in readme
    assert "Reference-family" in index
    assert "random replay" in cpu.lower()
    assert "smoke test" in cpu.lower()
    assert "random `ReplayBuffer` source" in comparison
    assert "same quick verification flow" in comparison
    assert "quick_verify.json" in comparison
    assert "contract demonstration" in comparison
    assert "same quick verification flow" in comparison_tutorial
    assert "published evidence bundle" in parity
    assert "not a public proof claim" in parity


def test_hyperparameter_sensitivity_doc_is_no_longer_template() -> None:
    sensitivity = _read("docs/reference/hyperparameter-sensitivity.md")
    assert "Template - to be populated" not in sensitivity
    assert "Pending experiment" not in sensitivity
    assert "atari100k_pong" in sensitivity


def test_training_reference_isolates_unsupported_placeholders() -> None:
    training_ref = _read("docs/api/training-reference.md")
    assert "Advanced/Internal placeholders" in training_ref
    assert "| `ema_decay` |" not in training_ref
    assert "| `model_overrides` |" not in training_ref
