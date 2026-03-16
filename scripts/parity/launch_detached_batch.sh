#!/usr/bin/env bash
set -euo pipefail

# Launch a long-running parity batch independently from the caller session.
# This prevents parent SSM session timeouts from killing the actual batch job.

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <state-dir> <command> [args...]" >&2
  exit 2
fi

STATE_DIR="$1"
shift

mkdir -p "$STATE_DIR"

STDOUT_LOG="$STATE_DIR/launcher.stdout.log"
STDERR_LOG="$STATE_DIR/launcher.stderr.log"
PID_FILE="$STATE_DIR/launcher.pid"
META_FILE="$STATE_DIR/launcher.meta.json"

python3 - "$META_FILE" "$@" <<'PY'
import json
import pathlib
import sys

meta_path = pathlib.Path(sys.argv[1])
cmd = sys.argv[2:]
meta = {"command": cmd}
meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
PY

nohup "$@" >"$STDOUT_LOG" 2>"$STDERR_LOG" < /dev/null &
PID="$!"
echo "$PID" > "$PID_FILE"
echo "launched detached pid=$PID"
