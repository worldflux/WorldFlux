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
    assert "(getting-started/cpu-success.md)" in index
    assert "(reference/benchmarks.md)" in index
    assert "(reference/parity.md)" in index
    assert "worldflux parity run" in parity
    assert "Placeholder tutorial pages remain published" in tutorial_policy


def test_quality_docs_reference_current_gate_commands():
    gates = _read("docs/reference/quality-gates.md")
    checklist = _read("docs/reference/release-checklist.md")

    assert "uvx ruff check src/ tests/ examples/ benchmarks/ scripts/" in gates
    assert "uvx ruff format --check src/ tests/ examples/ benchmarks/ scripts/" in gates
    assert "uv run mypy src/worldflux/" in gates
    assert "uv run pytest tests/" in gates
    assert "uv run mkdocs build --strict" in gates

    assert "examples/quickstart_cpu_success.py --quick" in checklist
    assert "examples/compare_unified_training.py --quick" in checklist
    assert "benchmarks/benchmark_dreamerv3_atari.py --quick" in checklist


def test_ci_includes_strict_docs_gate_and_new_smokes():
    ci = _read(".github/workflows/ci.yml")
    assert "docs:" in ci
    assert "uv sync --extra docs" in ci
    assert "uv run mkdocs build --strict" in ci
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
