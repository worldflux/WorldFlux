#!/usr/bin/env bash
set -euo pipefail

# Generate official DreamerV3 checkpoints for an arbitrary seed list and upload
# normalized artifacts to a chosen S3 prefix.

BASE_DIR="${BASE_DIR:-/opt/parity/bootstrap/i-03558d158797a5f2f}"
REPO_DIR="${REPO_DIR:-$BASE_DIR/worldflux-current/third_party/dreamerv3_official}"
PYTHON_BIN="${PYTHON_BIN:-$BASE_DIR/.venv/bin/python}"
SITE_PACKAGES="${SITE_PACKAGES:-$BASE_DIR/.venv/lib/python3.10/site-packages}"
WRAPPER_PATH="${WRAPPER_PATH:-$BASE_DIR/worldflux-current/scripts/parity/wrappers/official_dreamerv3.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$BASE_DIR/worldflux-current/reports/parity/dreamer_official_seed_batch}"
S3_ROOT="${S3_ROOT:?S3_ROOT is required}"
TASK_ID="${TASK_ID:-atari100k_pong}"
SEEDS_CSV="${SEEDS_CSV:?SEEDS_CSV is required}"
GPUS_CSV="${GPUS_CSV:-0,1,2,3}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
STEPS="${STEPS:-110000}"
EVAL_EPISODES="${EVAL_EPISODES:-1}"
XLA_MEM_FRACTION="${XLA_MEM_FRACTION:-0.85}"

mkdir -p "$OUTPUT_ROOT"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

declare -a PIDS=()
declare -A PID_SEED=()
declare -A PID_GPU=()

log() {
  printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$OUTPUT_ROOT/batch.log"
}

launch_seed() {
  local seed="$1"
  local gpu="$2"
  local run_id="seed_${seed}"
  local run_dir="$OUTPUT_ROOT/$run_id"
  mkdir -p "$run_dir"
  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    export XLA_PYTHON_CLIENT_MEM_FRACTION="$XLA_MEM_FRACTION"
    export PYTHONPATH="$SITE_PACKAGES:$REPO_DIR:$REPO_DIR/embodied"
    "$PYTHON_BIN" "$WRAPPER_PATH" \
      --repo-root "$REPO_DIR" \
      --task-id "$TASK_ID" \
      --seed "$seed" \
      --steps "$STEPS" \
      --device cuda \
      --run-dir "$run_dir/official" \
      --metrics-out "$run_dir/official/metrics.json" \
      --eval-episodes "$EVAL_EPISODES" \
      --python-executable "$PYTHON_BIN" \
      >"$run_dir/runner.stdout.log" 2>"$run_dir/runner.stderr.log"
    rc=$?
    echo "$rc" > "$run_dir/exit_code.txt"
    if [ "$rc" -eq 0 ]; then
      logdir="$run_dir/official/dreamerv3_logdir"
      latest="$(cat "$logdir/ckpt/latest")"
      prefix="${S3_ROOT%/}/$run_id"
      aws s3 cp "$logdir/config.yaml" "$prefix/config.yaml"
      aws s3 cp "$run_dir/official/metrics.json" "$prefix/metrics.json"
      aws s3 cp "$logdir/metrics.jsonl" "$prefix/metrics.jsonl"
      aws s3 cp "$logdir/scores.jsonl" "$prefix/scores.jsonl"
      aws s3 cp "$run_dir/runner.stdout.log" "$prefix/runner.stdout.log" || true
      aws s3 cp "$run_dir/runner.stderr.log" "$prefix/runner.stderr.log" || true
      aws s3 cp "$logdir/ckpt/latest" "$prefix/latest"
      aws s3 cp "$logdir/ckpt/$latest/agent.pkl" "$prefix/agent.pkl"
      aws s3 cp "$logdir/ckpt/$latest/replay.pkl" "$prefix/replay.pkl"
      aws s3 cp "$logdir/ckpt/$latest/step.pkl" "$prefix/step.pkl"
      aws s3 cp "$logdir/ckpt/$latest/done" "$prefix/done"
      if [ -f "$run_dir/component_match_report.json" ]; then
        aws s3 cp "$run_dir/component_match_report.json" "$prefix/component_match_report.json" || true
      fi
    fi
    exit "$rc"
  ) &
  local pid=$!
  PIDS+=("$pid")
  PID_SEED["$pid"]="$seed"
  PID_GPU["$pid"]="$gpu"
  log "launched seed=$seed gpu=$gpu pid=$pid"
}

wait_for_slot() {
  while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
    for i in "${!PIDS[@]}"; do
      local pid="${PIDS[$i]}"
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" || true
        log "finished seed=${PID_SEED[$pid]} gpu=${PID_GPU[$pid]} pid=$pid"
        unset 'PIDS[$i]'
        PIDS=("${PIDS[@]}")
        unset 'PID_SEED[$pid]'
        unset 'PID_GPU[$pid]'
        break
      fi
    done
    sleep 30
  done
}

gpu_idx=0
for seed in "${SEEDS[@]}"; do
  wait_for_slot
  gpu="${GPUS[$((gpu_idx % ${#GPUS[@]}))]}"
  launch_seed "$seed" "$gpu"
  gpu_idx=$((gpu_idx + 1))
done

for pid in "${PIDS[@]}"; do
  wait "$pid" || true
  log "finished seed=${PID_SEED[$pid]} gpu=${PID_GPU[$pid]} pid=$pid"
done

log "batch_complete"
