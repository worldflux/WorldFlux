"""Documentation integrity checks against implementation contracts."""

from __future__ import annotations

import ast
import importlib
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

MARKDOWN_IGNORE_DIRS = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".mypy_cache",
    "site",
    "build",
    "dist",
}

FORBIDDEN_CODE_SNIPPETS = (
    "model.q(",
    "enable_gradient_checkpointing(",
    "WORLDLOOM_",
)

FORBIDDEN_DOC_URL_PATTERNS = (
    r"https://worldflux\.github\.io/WorldFlux(?:/|$)",
    r"https://github\.com/worldflux/WorldFlux/tree/main/docs(?:/|$)",
)

PY_FENCE_RE = re.compile(r"```(?:python|py)\s*\n(.*?)```", flags=re.DOTALL | re.IGNORECASE)
IMPORT_RE = re.compile(
    r"^\s*import\s+(worldflux(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\b(?:\s+as\s+\w+)?",
    flags=re.MULTILINE,
)
FROM_RE = re.compile(
    r"^\s*from\s+(worldflux(?:\.[A-Za-z_][A-Za-z0-9_]*)*)\s+import\s+(.+)$",
    flags=re.MULTILINE,
)


def _markdown_files() -> list[Path]:
    files: list[Path] = []
    for path in REPO_ROOT.rglob("*.md"):
        rel = path.relative_to(REPO_ROOT)
        if any(part in MARKDOWN_IGNORE_DIRS for part in rel.parts):
            continue
        files.append(path)
    return sorted(files)


def _python_code_blocks(markdown: str) -> list[str]:
    return [block.strip() for block in PY_FENCE_RE.findall(markdown)]


def _extract_import_targets_from_ast(code: str) -> list[tuple[str, list[str] | None]]:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return _extract_import_targets_with_regex(code)

    targets: list[tuple[str, list[str] | None]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("worldflux"):
                    targets.append((alias.name, None))
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0 and node.module.startswith("worldflux"):
                names = [alias.name for alias in node.names]
                targets.append((node.module, names))
    return targets


def _extract_import_targets_with_regex(code: str) -> list[tuple[str, list[str] | None]]:
    targets: list[tuple[str, list[str] | None]] = []
    for match in IMPORT_RE.finditer(code):
        targets.append((match.group(1), None))
    for match in FROM_RE.finditer(code):
        module = match.group(1)
        raw_names = match.group(2)
        names = []
        for name in raw_names.split(","):
            token = name.strip()
            if not token:
                continue
            token = token.split(" as ", 1)[0].strip()
            token = token.strip("()")
            if token:
                names.append(token)
        if names:
            targets.append((module, names))
    return targets


def _resolve_module(module_name: str) -> str | None:
    try:
        importlib.import_module(module_name)
        return None
    except Exception as exc:  # pragma: no cover - exact import failures vary by runtime
        return f"cannot import module '{module_name}': {exc}"


def _resolve_from_import(module_name: str, symbol: str) -> str | None:
    if symbol == "*":
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover
        return f"cannot import module '{module_name}': {exc}"

    if hasattr(module, symbol):
        return None

    try:
        importlib.import_module(f"{module_name}.{symbol}")
        return None
    except Exception:
        return f"'{symbol}' is not exported from '{module_name}'"


def test_markdown_worldflux_imports_are_resolvable():
    errors: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for block_idx, block in enumerate(_python_code_blocks(text), start=1):
            for module_name, symbols in _extract_import_targets_from_ast(block):
                if symbols is None:
                    err = _resolve_module(module_name)
                    if err:
                        errors.append(f"{path}: block {block_idx}: {err}")
                    continue
                for symbol in symbols:
                    err = _resolve_from_import(module_name, symbol)
                    if err:
                        errors.append(f"{path}: block {block_idx}: {err}")

    assert not errors, "Unresolvable worldflux imports in markdown:\n" + "\n".join(errors)


def test_markdown_does_not_reference_removed_or_unimplemented_apis():
    hits: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for token in FORBIDDEN_CODE_SNIPPETS:
            if token in text:
                hits.append(f"{path}: contains forbidden token '{token}'")
    assert not hits, "Forbidden API snippets found:\n" + "\n".join(hits)


def test_markdown_does_not_include_deprecated_doc_hosts():
    hits: list[str] = []
    for path in _markdown_files():
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_DOC_URL_PATTERNS:
            if re.search(pattern, text):
                hits.append(f"{path}: matches forbidden URL pattern '{pattern}'")
    assert not hits, "Forbidden documentation hosts found:\n" + "\n".join(hits)


def test_pyproject_documentation_url_matches_policy():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'Documentation = "https://worldflux.readthedocs.io/en/latest/"' in pyproject
