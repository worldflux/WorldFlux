"""Policy checks for stealth-safe public documentation wording."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

PUBLIC_DOC_PATHS = (
    "README.md",
    "docs/index.md",
    "docs/getting-started/quickstart.md",
    "docs/getting-started/concepts.md",
    "docs/tutorials/dreamer-vs-tdmpc2.md",
    "docs/tutorials/reproduce-dreamer-tdmpc2.md",
    "docs/tutorials/train-first-model.md",
    "docs/reference/quality-gates.md",
    "docs/reference/release-checklist.md",
)

FORBIDDEN_PATTERNS = (
    r"##\s*Benchmarks",
    r"DreamerV3\s+vs\s+TD-MPC2",
    r"Reproduce\s+Dreamer/TD-MPC2",
    r"Benchmark\s+Gates",
    r"Reproducibility\s+Gates",
    r"Results\s+on\s+standard\s+benchmarks",
)


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def test_public_docs_exclude_forbidden_stealth_phrases():
    hits: list[str] = []
    for rel in PUBLIC_DOC_PATHS:
        text = _read(rel)
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, text, flags=re.IGNORECASE):
                hits.append(f"{rel}: matches '{pattern}'")

    assert not hits, "Forbidden stealth phrases found:\n" + "\n".join(hits)
