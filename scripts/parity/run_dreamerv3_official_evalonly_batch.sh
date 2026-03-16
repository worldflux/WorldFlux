#!/usr/bin/env bash
set -euo pipefail

# Evaluate existing official DreamerV3 checkpoints for a fixed number of episodes.
# This uses the official repo's eval_only entrypoint and stops each run once
# the requested number of scores has been written to scores.jsonl.

BASE_DIR="${BASE_DIR:-/opt/parity/bootstrap/i-03558d158797a5f2f}"
OFFICIAL_REPO="${OFFICIAL_REPO:-$BASE_DIR/dreamerv3-official}"
PYTHON_BIN="${PYTHON_BIN:-$BASE_DIR/.venv/bin/python}"
SITE_PACKAGES="${SITE_PACKAGES:-$BASE_DIR/.venv/lib/python3.10/site-packages}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-$BASE_DIR/worldflux-current/reports/parity/dreamer_official_batch10_managed}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASE_DIR/worldflux-current/reports/parity/dreamer_official_batch10_eval10}"
S3_ROOT="${S3_ROOT:-}"
TASK_ID="${TASK_ID:-atari100k_pong}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2,3,4,5,6,7,8,9}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
RUN_STEPS="${RUN_STEPS:-100000000}"
POLL_SECONDS="${POLL_SECONDS:-5}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.80}"

mkdir -p "$OUTPUT_ROOT"

IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

declare -a PIDS=()
declare -A PID_TO_SEED=()
declare -A PID_TO_GPU=()

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUTPUT_ROOT/batch.log"
}

checkpoint_dir_path() {
  local seed="$1"
  local ckpt_dir="$CHECKPOINT_ROOT/seed_${seed}/official/dreamerv3_logdir/ckpt"
  local latest
  latest="$(cat "$ckpt_dir/latest")"
  printf '%s\n' "$ckpt_dir/$latest"
}

launch_seed() {
  local seed="$1"
  local gpu="$2"
  local checkpoint_dir="$3"
  local run_dir="$OUTPUT_ROOT/seed_${seed}"
  mkdir -p "$run_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
    export PYTHONPATH="$SITE_PACKAGES:$OFFICIAL_REPO:$OFFICIAL_REPO/embodied"
    "$PYTHON_BIN" "$OFFICIAL_REPO/dreamerv3/main.py" \
      --script eval_only \
      --logdir "$run_dir" \
      --configs atari100k \
      --task "$TASK_ID" \
      --seed "$seed" \
      --run.from_checkpoint "$checkpoint_dir" \
      --run.steps "$RUN_STEPS" \
      --run.envs 1 \
      --run.log_every 1 \
      --logger.outputs jsonl \
      --jax.platform cuda \
      >"$run_dir/runner.stdout.log" 2>"$run_dir/runner.stderr.log" &
    eval_pid=$!
    echo "$eval_pid" > "$run_dir/eval.pid"
    scores="$run_dir/scores.jsonl"
    while true; do
      count=0
      if [ -f "$scores" ]; then
        count="$(wc -l < "$scores" | tr -d ' ')"
      fi
      if [ "$count" -ge "$EVAL_EPISODES" ]; then
        break
      fi
      if ! kill -0 "$eval_pid" 2>/dev/null; then
        break
      fi
      sleep "$POLL_SECONDS"
    done
    if kill -0 "$eval_pid" 2>/dev/null; then
      kill -TERM "$eval_pid" >/dev/null 2>&1 || true
      sleep 3
    fi
    if kill -0 "$eval_pid" 2>/dev/null; then
      kill -KILL "$eval_pid" >/dev/null 2>&1 || true
    fi
    wait "$eval_pid" || true
    "$PYTHON_BIN" - "$run_dir" "$EVAL_EPISODES" <<'PY'
import json
import pathlib
import statistics
import sys

run_dir = pathlib.Path(sys.argv[1])
target = int(sys.argv[2])
scores_path = run_dir / "scores.jsonl"
rows = []
if scores_path.exists():
    for line in scores_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
scores = [float(row["episode/score"]) for row in rows[:target] if "episode/score" in row]
payload = {
    "episodes_requested": target,
    "episodes_observed": len(scores),
    "scores": scores,
    "mean_score": statistics.fmean(scores) if scores else None,
    "median_score": statistics.median(scores) if scores else None,
    "min_score": min(scores) if scores else None,
    "max_score": max(scores) if scores else None,
}
(run_dir / "eval10_summary.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    if [ -n "$S3_ROOT" ]; then
      prefix="${S3_ROOT%/}/seed_${seed}"
      aws s3 cp "$run_dir/scores.jsonl" "$prefix/scores.jsonl" || true
      aws s3 cp "$run_dir/metrics.jsonl" "$prefix/metrics.jsonl" || true
      aws s3 cp "$run_dir/eval10_summary.json" "$prefix/eval10_summary.json" || true
      aws s3 cp "$run_dir/runner.stdout.log" "$prefix/runner.stdout.log" || true
      aws s3 cp "$run_dir/runner.stderr.log" "$prefix/runner.stderr.log" || true
    fi
  ) &
  local pid=$!
  PIDS+=("$pid")
  PID_TO_SEED["$pid"]="$seed"
  PID_TO_GPU["$pid"]="$gpu"
  log "launched seed=${seed} gpu=${gpu} pid=${pid}"
}

wait_for_slot() {
  while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
    for i in "${!PIDS[@]}"; do
      local pid="${PIDS[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
        log "finished seed=${PID_TO_SEED[$pid]} gpu=${PID_TO_GPU[$pid]} pid=${pid}"
        unset 'PIDS[$i]'
        PIDS=("${PIDS[@]}")
        unset 'PID_TO_SEED[$pid]'
        unset 'PID_TO_GPU[$pid]'
        break
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

gpu_idx=0
for seed in "${SEEDS[@]}"; do
  checkpoint_dir="$(checkpoint_dir_path "$seed")"
  wait_for_slot
  gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
  launch_seed "$seed" "$gpu" "$checkpoint_dir"
  gpu_idx=$((gpu_idx + 1))
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || true
  log "finished seed=${PID_TO_SEED[$pid]} gpu=${PID_TO_GPU[$pid]} pid=${pid}"
done

log "batch_complete"
