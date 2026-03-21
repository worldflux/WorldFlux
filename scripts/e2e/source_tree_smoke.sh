#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Lightweight e2e smoke test that runs in the source tree (no wheel build).
# Mirrors the pip_flow_smoke.sh flow but uses uv run from the repo root.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

echo "[e2e-source] root: ${ROOT_DIR}"
echo "[e2e-source] work: ${WORK_DIR}"

cd "${WORK_DIR}"

# Non-interactive wizard answers (same as pip_flow_smoke.sh):
# project, env(1=atari), model(1=dreamer:ci), steps(1=50k), batch(1=8), device(2=cpu), confirm
printf '%s\n' \
  "smoke-project" \
  "1" \
  "1" \
  "1" \
  "1" \
  "2" \
  "y" \
  | uv run --project "${ROOT_DIR}" worldflux init smoke --force

cd smoke
uv run --project "${ROOT_DIR}" worldflux train --steps 2 --device cpu

set +e
uv run --project "${ROOT_DIR}" worldflux verify \
  --target ./outputs --mode quick --episodes 2 \
  --format json --output verify-quick.json
quick_status=$?
set -e

if [[ "${quick_status}" -ne 0 && "${quick_status}" -ne 1 ]]; then
  echo "[e2e-source] unexpected quick verify exit code: ${quick_status}" >&2
  exit 1
fi

uv run --project "${ROOT_DIR}" python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

quick = json.loads(Path("verify-quick.json").read_text(encoding="utf-8"))
assert "passed" in quick, f"missing 'passed' key: {list(quick.keys())}"
assert quick["env"] == "atari/pong", f"unexpected env: {quick['env']}"
assert "stats" in quick and isinstance(quick["stats"], dict)
print("[e2e-source] verification payload validated")
PY

echo "[e2e-source] source tree smoke passed"
