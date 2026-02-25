#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$(mktemp -d)"
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "[e2e] python interpreter not found" >&2
    exit 1
  fi
fi

cleanup() {
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

echo "[e2e] root: ${ROOT_DIR}"
echo "[e2e] work: ${WORK_DIR}"

"${PYTHON_BIN}" -m venv "${WORK_DIR}/venv"
# shellcheck disable=SC1091
source "${WORK_DIR}/venv/bin/activate"

python -m pip install --upgrade pip build
python -m build --wheel "${ROOT_DIR}"

WHEEL_PATH="$(ls -t "${ROOT_DIR}"/dist/worldflux-*.whl | head -n 1)"
if [[ -z "${WHEEL_PATH}" ]]; then
  echo "[e2e] no wheel found in dist/" >&2
  exit 1
fi

python -m pip install "${WHEEL_PATH}"

export WORLDFLUX_INIT_ENSURE_DEPS=0

cd "${WORK_DIR}"
# Non-interactive wizard answers (Rich numbered fallback):
# project, env(1=atari), model(1=dreamer:ci), steps(1=50k), batch(1=8), device(2=cpu), confirm
printf '%s\n' \
  "demo-project" \
  "1" \
  "1" \
  "1" \
  "1" \
  "2" \
  "y" \
  | worldflux init demo --force

cd demo
worldflux train --steps 2 --device cpu
set +e
worldflux verify --target ./outputs --mode quick --episodes 2 --format json --output verify-quick.json
quick_status=$?
worldflux verify --target ./outputs --demo --format json --output verify-demo.json
demo_status=$?
set -e

if [[ "${quick_status}" -ne 0 && "${quick_status}" -ne 1 ]]; then
  echo "[e2e] unexpected quick verify exit code: ${quick_status}" >&2
  exit 1
fi
if [[ "${demo_status}" -ne 0 && "${demo_status}" -ne 1 ]]; then
  echo "[e2e] unexpected demo verify exit code: ${demo_status}" >&2
  exit 1
fi

python - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

quick = json.loads(Path("verify-quick.json").read_text(encoding="utf-8"))
demo = json.loads(Path("verify-demo.json").read_text(encoding="utf-8"))

assert "passed" in quick
assert quick["env"] == "atari/pong"
assert "stats" in quick and isinstance(quick["stats"], dict)

assert "passed" in demo
assert "stats" in demo and isinstance(demo["stats"], dict)
print("[e2e] verification payloads validated")
PY

echo "[e2e] pip flow smoke passed"
