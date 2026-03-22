#!/usr/bin/env bash
# launch_tdmpc2_parity.sh
#
# TD-MPC2-specific parity verification on AWS.

set -euo pipefail

MODE="${1:-pilot}"
shift || true

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

case "$MODE" in
    pilot|full) ;;
    *) echo "Usage: $0 {pilot|full} [--dry-run]"; exit 1 ;;
esac

REGION="us-west-2"
RUN_TAG="tdmpc2_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="s3://worldflux-parity/${RUN_TAG}"
MANIFEST="scripts/parity/manifests/official_vs_worldflux_full_v2.yaml"
SEEDS="0,1,2,3,4,5,6,7,8,9"
if [ "$MODE" = "full" ]; then
    SEEDS="$(python3 - <<'PY'
print(",".join(str(i) for i in range(20)))
PY
)"
fi

CMD=(
    python3 scripts/parity/aws_distributed_orchestrator.py
    --region "${REGION}"
    --manifest "${MANIFEST}"
    --full-manifest "${MANIFEST}"
    --run-id "${RUN_TAG}"
    --s3-prefix "${S3_PREFIX}"
    --phase-plan two_stage_proof
    --phase-gate strict_pass
    --sharding-mode seed_system
    --seed-shard-unit pair
    --systems official,worldflux
    --seed-list "${SEEDS}"
)

echo "Run tag: ${RUN_TAG}"
echo "S3 prefix: ${S3_PREFIX}"
echo "Canonical profile: proof_5m"

if [ "$DRY_RUN" = true ]; then
    printf '[DRY RUN] %q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

"${CMD[@]}"
