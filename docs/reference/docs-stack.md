# Docs Stack

This page defines the documentation tooling used by WorldFlux and explains why each library is included.

## Purpose

- Keep docs authoring simple (Docusaurus + Markdown / MDX).
- Catch documentation regressions early in CI (`onBrokenLinks: 'throw'`).
- Stay pragmatic: adopt only what is needed for the current project size.

## Adopted Libraries (WorldFlux)

| Library | Why it is used |
|---|---|
| `@docusaurus/core` v3 | Static docs site generator with React-based architecture |
| `@docusaurus/preset-classic` | Standard theme, search, and navigation UX |
| `@docusaurus/theme-mermaid` | Mermaid diagram rendering in Markdown |
| `prism-react-renderer` | Syntax highlighting for code blocks |
| `React` | Docusaurus rendering engine |

## Previous Stack (deprecated)

| Library | Status |
|---|---|
| `mkdocs` | Removed — replaced by Docusaurus |
| `mkdocs-material` | Removed — Docusaurus preset-classic replaces theme |
| `mkdocstrings[python]` | Removed — API reference tables maintained manually |
| `mkdocs-autorefs` | Removed — Docusaurus handles cross-references natively |
| `mkdocs-redirects` | Removed — not needed at current scale |
| `mkdocs-git-revision-date-localized-plugin` | Removed — not needed at current scale |
| `mkdocs-minify-plugin` | Removed — Docusaurus production build handles optimization |
| `pymdown-extensions` | Removed — Docusaurus MDX handles admonitions, tabs, etc. |

## Not Adopted in This Revision (and why)

| Candidate | Deferred reason |
|---|---|
| Vale style linting | Useful but out of scope for the current minimal change set |
| Snippet execution pipeline | Valuable, but deferred to keep CI/runtime cost low |
| `@docusaurus/plugin-ideal-image` | Deferred until image-heavy pages are added |

## Standard Commands

```bash
cd website && npm install
cd website && npm audit --audit-level=high
cd website && npm start
cd website && npm run build
```

## Policy

- Runtime API/architecture is unchanged by docs-tooling updates.
- Docs dependencies are managed via `website/package.json`, not `pyproject.toml`.
- Docs dependency health is enforced with `npm audit --audit-level=high` in CI and release dry-runs.
- New docs tooling should be added only when a concrete docs quality gap exists.
