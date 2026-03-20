# WorldFlux S-Grade Program Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install the Phase 0 program structure required by the 2026-03-19 S-grade master spec so later correctness and production work can run against explicit, verifiable contracts.

**Architecture:** Start with repository-level process artifacts before touching runtime behavior. Phase 0 is split into three bounded chunks: agent-readable documentation skeleton, a collect-only CI audit path, and release metadata alignment checks so later PRDs and P0 fixes have a stable operational frame.

**Tech Stack:** Python, pytest, Markdown docs, GitHub Actions, uv

---

## Chunk 1: Program Docs Skeleton

### Task 1: Add S-Grade program structure documents

**Files:**
- Create: `AGENTS.md`
- Create: `docs/roadmap/2026-q2-worldflux-quality-program.md`
- Create: `docs/prd/architecture-s-grade.md`
- Create: `docs/prd/api-s-grade.md`
- Create: `docs/prd/ml-correctness-s-grade.md`
- Create: `docs/prd/code-quality-s-grade.md`
- Create: `docs/prd/differentiation-s-grade.md`
- Create: `docs/prd/production-maturity-s-grade.md`
- Create: `docs/prd/oss-readiness-s-grade.md`
- Create: `docs/prd/scalability-s-grade.md`
- Create: `docs/tasks/README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `GOVERNANCE.md`
- Modify: `docs/roadmap.md`
- Test: `tests/test_docs/test_s_grade_program_docs.py`

- [ ] **Step 1: Write the failing documentation structure test**

```python
def test_s_grade_program_docs_exist_and_reference_required_phase_zero_artifacts() -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs/test_s_grade_program_docs.py -v`
Expected: FAIL because the new roadmap / PRD / AGENTS artifacts do not exist yet

- [ ] **Step 3: Add the minimal docs and cross-references**

Create the required roadmap and PRD skeleton files using the spec-defined YAML shape and section order. Add a root `AGENTS.md` that encodes the task envelope rules from the master spec. Update contributor-facing docs to point at the new roadmap and program structure without removing existing operational guidance.

- [ ] **Step 4: Run targeted tests to verify the docs pass**

Run: `uv run pytest tests/test_docs/test_s_grade_program_docs.py tests/test_docs/test_oss_operations.py -v`
Expected: PASS

## Chunk 2: Collect-Only CI Gate

### Task 2: Add a non-blocking Phase 0 audit collector

**Files:**
- Create: `scripts/collect_s_grade_program_status.py`
- Create: `.github/workflows/s-grade-collect.yml`
- Test: `tests/test_scripts/test_collect_s_grade_program_status.py`

- [ ] **Step 1: Write the failing script test**

```python
def test_collect_s_grade_program_status_reports_missing_and_present_artifacts() -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scripts/test_collect_s_grade_program_status.py -v`
Expected: FAIL because the collector script does not exist yet

- [ ] **Step 3: Implement the minimal collector**

Write a script that inspects the Phase 0 required artifacts, emits a JSON summary, and exits 0 even when gaps remain. Add a workflow that uploads the report artifact and is clearly marked collect-only.

- [ ] **Step 4: Run targeted verification**

Run: `uv run pytest tests/test_scripts/test_collect_s_grade_program_status.py -v`
Expected: PASS

## Chunk 3: Release Metadata Alignment

### Task 3: Add release metadata alignment policy checks

**Files:**
- Create: `tests/test_docs/test_release_metadata_alignment.py`
- Modify: `README.md`
- Modify: `docs/reference/release-checklist.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing alignment test**

```python
def test_release_metadata_documents_reference_consistent_versioning_and_release_sources() -> None:
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_docs/test_release_metadata_alignment.py -v`
Expected: FAIL because the release metadata references are not yet aligned to the new program sources

- [ ] **Step 3: Implement the minimal metadata alignment updates**

Ensure the public release checklist, README release status language, and package metadata point at the same release authority and validation path. Keep changes additive and avoid unrelated copy edits.

- [ ] **Step 4: Run targeted verification**

Run: `uv run pytest tests/test_docs/test_release_metadata_alignment.py tests/test_docs/test_oss_operations.py -v`
Expected: PASS
