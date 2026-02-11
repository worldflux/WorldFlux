"""Policy checks for current public documentation wording and links."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_readme_uses_custom_docs_host():
    readme = _read("README.md")
    assert "https://worldflux.ai/" in readme
    assert "https://github.com/worldflux/WorldFlux/tree/main/docs" not in readme
    assert "https://worldflux.readthedocs.io/en/latest/" not in readme


def test_public_docs_include_benchmark_and_comparison_guides():
    index = _read("docs/index.md")
    benchmarks = _read("docs/reference/benchmarks.md")
    comparison = _read("docs/reference/unified-comparison.md")

    assert "(reference/benchmarks.md)" in index
    assert "(reference/unified-comparison.md)" in index
    assert "DreamerV3" in benchmarks
    assert "TD-MPC2" in benchmarks
    assert "same ReplayBuffer source" in comparison


def test_cpu_success_and_wasr_docs_exist_and_are_actionable():
    cpu = _read("docs/getting-started/cpu-success.md")
    wasr = _read("docs/reference/wasr.md")

    assert "uv run python examples/quickstart_cpu_success.py --quick" in cpu
    assert ".worldflux/metrics.jsonl" in wasr
    assert "scripts/compute_wasr.py" in wasr
