"""Consistency checks for stealth-safe documentation policy."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_roadmap_file_is_removed_for_stealth_sharing():
    assert not (REPO_ROOT / "ROADMAP.md").exists()


def test_public_docs_omit_benchmark_and_comparison_headings():
    readme = _read("README.md")
    index = _read("docs/index.md")

    assert "## Benchmarks" not in readme
    assert "## Model Maturity Policy" not in readme
    assert "DreamerV3 vs TD-MPC2" not in index
    assert "Reproduce Dreamer/TD-MPC2" not in index


def test_quality_docs_remain_operational_but_neutral():
    gates = _read("docs/reference/quality-gates.md")
    checklist = _read("docs/reference/release-checklist.md")

    assert "uvx ruff check src/ tests/ examples/" in gates
    assert "uvx ruff format --check src/ tests/ examples/" in gates
    assert "uv run mypy src/worldflux/" in gates
    assert "uv run pytest tests/" in gates
    assert "uv run mkdocs build --strict" in gates

    assert "Benchmark Gates" not in gates
    assert "Reproducibility Gates" not in gates
    assert "measure_quality_gates.py" not in gates
    assert "benchmark/repro gate runs" not in checklist


def test_ci_includes_strict_docs_gate():
    ci = _read(".github/workflows/ci.yml")
    assert "docs:" in ci
    assert "uv sync --extra docs" in ci
    assert "uv run mkdocs build --strict" in ci
    assert "ROADMAP.md" not in ci


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
