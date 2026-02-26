#!/usr/bin/env bash
# launch_dreamerv3_parity.sh
#
# DreamerV3-specific parity verification on AWS using g5.xlarge instances.
# Based on launch_cloud_orchestrator.sh but tuned for DreamerV3-only runs
# with cost-efficient g5.xlarge (A10G) GPU fleet.
#
# Usage:
#   bash scripts/parity/launch_dreamerv3_parity.sh smoke      # 2 seeds × 3 tasks × 2 systems
#   bash scripts/parity/launch_dreamerv3_parity.sh stage-a     # 4→10-seed auto_power
#   bash scripts/parity/launch_dreamerv3_parity.sh full        # smoke → stage-a sequential
#   bash scripts/parity/launch_dreamerv3_parity.sh [mode] --dry-run
#
# Prerequisites:
#   - AWS CLI configured with credentials for us-west-2
#   - G instance quota ≥ 48 vCPU in us-west-2
#   - The IAM instance profile must allow ec2/ssm/s3 access
#
# Cost estimates (g5.xlarge ~$1.01/hr):
#   smoke:   ~$6   (~30 min, 12 parallel)
#   stage-a: ~$12  (~1 hr,   12 parallel × 2 batches)
#   full:    ~$48  (~2.5 hr, smoke + stage-a 10-seed)

set -euo pipefail

# ── Mode selection ─────────────────────────────────────────────────
MODE="${1:-smoke}"
shift || true

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# Validate mode
case "$MODE" in
    smoke|stage-a|full) ;;
    *) echo "Usage: $0 {smoke|stage-a|full} [--dry-run]"; exit 1 ;;
esac

# ── Configuration ────────────────────────────────────────────────────
REGION="us-west-2"
CONTROL_PLANE_TYPE="t3.medium"
CONTROL_PLANE_AMI="ami-0e5e1413a3bf2d262"  # Ubuntu 24.04 LTS us-west-2

# GPU worker configuration — g5.xlarge (4 vCPU, 16GB RAM, 1× A10G 24GB)
GPU_INSTANCE_TYPE="g5.xlarge"
GPU_AMI="ami-068674ce56829a0ea"  # Deep Learning AMI with CUDA
FLEET_SIZE=11
FLEET_SPLIT="11,0"  # all official-side, each runs both systems (44/48 vCPU headroom)

# Networking / IAM (reuse existing parity infrastructure)
SUBNET_ID="subnet-017e9b9db46658c71"
SECURITY_GROUP_IDS="sg-0b6167f7cb8009323"
IAM_INSTANCE_PROFILE="worldflux-ec2-ssm-role"
KEY_NAME="worldflux"

# S3 and run configuration
S3_BUCKET="worldflux-parity"
RUN_TAG="dreamerv3_${MODE}_$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="s3://${S3_BUCKET}/${RUN_TAG}"
WORLDFLUX_BRANCH="main"

# Manifest paths
SMOKE_MANIFEST="scripts/parity/manifests/dreamerv3_smoke_v2.yaml"
STAGE_A_MANIFEST="scripts/parity/manifests/dreamerv3_stage_a_v2.yaml"

# Mode-specific orchestrator flags
case "$MODE" in
    smoke)
        MANIFEST="$SMOKE_MANIFEST"
        FULL_MANIFEST="$SMOKE_MANIFEST"
        PHASE_PLAN="single"
        ;;
    stage-a)
        MANIFEST="$STAGE_A_MANIFEST"
        FULL_MANIFEST="$STAGE_A_MANIFEST"
        PHASE_PLAN="two_stage_proof"
        ;;
    full)
        MANIFEST="$SMOKE_MANIFEST"
        FULL_MANIFEST="$STAGE_A_MANIFEST"
        PHASE_PLAN="two_stage_proof"
        ;;
esac

# ── User-data script (runs on the control-plane at boot) ────────────
USERDATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/orchestrator-boot.log) 2>&1

echo "=== DreamerV3 Parity Control-plane bootstrap: $(date -u) ==="

# Install essentials
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git python3 python3-pip python3-venv jq unzip curl

# Install latest AWS CLI v2 (the apt awscli is v1)
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -qo /tmp/awscliv2.zip -d /tmp/
/tmp/aws/install --update || true
rm -rf /tmp/aws /tmp/awscliv2.zip

# Fetch configuration from instance metadata (IMDSv2)
IMDS_TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $IMDS_TOKEN" http://169.254.169.254/latest/meta-data/placement/region)

# Read orchestrator config from instance tags
get_tag() { aws ec2 describe-tags --region "$REGION" --filters "Name=resource-id,Values=$INSTANCE_ID" "Name=key,Values=$1" --query "Tags[0].Value" --output text; }

S3_PREFIX=$(get_tag "ParityS3Prefix")
RUN_TAG=$(get_tag "ParityRunTag")
WF_BRANCH=$(get_tag "ParityBranch")
GPU_AMI=$(get_tag "ParityGpuAmi")
GPU_INSTANCE_TYPE=$(get_tag "ParityGpuInstanceType")
FLEET_SIZE=$(get_tag "ParityFleetSize")
FLEET_SPLIT=$(get_tag "ParityFleetSplit")
SUBNET_ID=$(get_tag "ParitySubnetId")
SG_IDS=$(get_tag "ParitySecurityGroupIds")
IAM_PROFILE=$(get_tag "ParityIamProfile")
KEY_NAME=$(get_tag "ParityKeyName")
MANIFEST=$(get_tag "ParityManifest")
FULL_MANIFEST=$(get_tag "ParityFullManifest")
PHASE_PLAN=$(get_tag "ParityPhasePlan")

# Clone the repo
cd /opt
git clone --depth 1 --branch "${WF_BRANCH}" https://github.com/worldflux/WorldFlux.git worldflux
cd worldflux

echo "=== Starting DreamerV3 parity orchestrator: $(date -u) ==="
echo "Run tag: ${RUN_TAG}"
echo "S3 prefix: ${S3_PREFIX}"
echo "Manifest: ${MANIFEST}"
echo "Full manifest: ${FULL_MANIFEST}"
echo "Phase plan: ${PHASE_PLAN}"

# Run the orchestrator
export PYTHONUNBUFFERED=1

stdbuf -oL -eL python3 scripts/parity/aws_distributed_orchestrator.py \
    --region "${REGION}" \
    --manifest "${MANIFEST}" \
    --full-manifest "${FULL_MANIFEST}" \
    --run-id "${RUN_TAG}" \
    --s3-prefix "${S3_PREFIX}" \
    --phase-plan "${PHASE_PLAN}" \
    --phase-gate strict_pass \
    --auto-provision \
    --auto-terminate \
    --fleet-size "${FLEET_SIZE}" \
    --fleet-split "${FLEET_SPLIT}" \
    --instance-type "${GPU_INSTANCE_TYPE}" \
    --image-id "${GPU_AMI}" \
    --subnet-id "${SUBNET_ID}" \
    --security-group-ids "${SG_IDS}" \
    --iam-instance-profile "${IAM_PROFILE}" \
    --key-name "${KEY_NAME}" \
    --gpu-slots-per-instance 1 \
    --sharding-mode seed_system \
    --seed-shard-unit pair \
    --wait \
    --output-dir reports/parity \
    2>&1 | tee "/var/log/orchestrator-${RUN_TAG}.log"

ORCH_RC=$?
echo "=== Orchestrator finished: RC=${ORCH_RC} $(date -u) ==="

# Upload orchestrator log and final results to S3
aws s3 cp "/var/log/orchestrator-${RUN_TAG}.log" \
    "${S3_PREFIX}/orchestrator.log" --region "${REGION}"
aws s3 sync "reports/parity/${RUN_TAG}" \
    "${S3_PREFIX}/results/" --region "${REGION}" || true

echo "=== Results uploaded to ${S3_PREFIX}/ ==="

# Self-terminate the control-plane instance
echo "=== Self-terminating control-plane $(date -u) ==="
aws ec2 terminate-instances --instance-ids "${INSTANCE_ID}" --region "${REGION}"
USERDATA_EOF
)

USERDATA_B64=$(echo "$USERDATA" | base64)

# ── Print summary ────────────────────────────────────────────────────
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DreamerV3 Parity Verification Launch                      ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Mode           : ${MODE}"
echo "║  Control plane  : ${CONTROL_PLANE_TYPE} (${CONTROL_PLANE_AMI})"
echo "║  GPU workers    : ${GPU_INSTANCE_TYPE} × ${FLEET_SIZE}"
echo "║  GPU slots/inst : 1 (A10G 24GB)"
echo "║  Region         : ${REGION}"
echo "║  Run tag        : ${RUN_TAG}"
echo "║  S3 prefix      : ${S3_PREFIX}"
echo "║  Branch         : ${WORLDFLUX_BRANCH}"
echo "║  Manifest       : ${MANIFEST}"
echo "║  Full manifest  : ${FULL_MANIFEST}"
echo "║  Phase plan     : ${PHASE_PLAN}"
echo "╚══════════════════════════════════════════════════════════════╝"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "[DRY RUN] Would launch control-plane instance with the above config."
    echo "[DRY RUN] User-data script:"
    echo "$USERDATA"
    exit 0
fi

# ── Launch the control-plane ─────────────────────────────────────────
echo ""
echo "Launching control-plane instance..."

RESULT=$(aws ec2 run-instances \
    --region "${REGION}" \
    --image-id "${CONTROL_PLANE_AMI}" \
    --instance-type "${CONTROL_PLANE_TYPE}" \
    --subnet-id "${SUBNET_ID}" \
    --security-group-ids "${SECURITY_GROUP_IDS}" \
    --iam-instance-profile "Name=${IAM_INSTANCE_PROFILE}" \
    --key-name "${KEY_NAME}" \
    --user-data "${USERDATA_B64}" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3","DeleteOnTermination":true}}]' \
    --metadata-options '{"HttpTokens":"required","HttpPutResponseHopLimit":2,"HttpEndpoint":"enabled"}' \
    --tag-specifications "$(python3 - "${RUN_TAG}" "${S3_PREFIX}" "${WORLDFLUX_BRANCH}" "${GPU_AMI}" "${GPU_INSTANCE_TYPE}" "${FLEET_SIZE}" "${FLEET_SPLIT}" "${SUBNET_ID}" "${SECURITY_GROUP_IDS}" "${IAM_INSTANCE_PROFILE}" "${KEY_NAME}" "${MANIFEST}" "${FULL_MANIFEST}" "${PHASE_PLAN}" <<'PYTAGS'
import json, sys
a = sys.argv[1:]
tags = dict(zip(
    ["ParityRunTag","ParityS3Prefix","ParityBranch","ParityGpuAmi",
     "ParityGpuInstanceType","ParityFleetSize","ParityFleetSplit",
     "ParitySubnetId","ParitySecurityGroupIds","ParityIamProfile","ParityKeyName",
     "ParityManifest","ParityFullManifest","ParityPhasePlan"], a))
tags["Name"] = "worldflux-dreamerv3-parity-orchestrator"
tags["ManagedBy"] = "worldflux-parity-cloud"
spec = [{"ResourceType": "instance", "Tags": [{"Key": k, "Value": v} for k, v in tags.items()]}]
print(json.dumps(spec))
PYTAGS
)" \
    --instance-initiated-shutdown-behavior terminate \
    --output json)

CP_INSTANCE_ID=$(echo "$RESULT" | python3 -c "import json,sys; print(json.load(sys.stdin)['Instances'][0]['InstanceId'])")
echo ""
echo "Control-plane launched: ${CP_INSTANCE_ID}"
echo ""
echo "Monitor commands:"
echo "  # Watch boot log (after SSM agent starts, ~60s):"
echo "  aws ssm start-session --target ${CP_INSTANCE_ID} --region ${REGION}"
echo ""
echo "  # Check orchestrator log:"
echo "  aws ssm send-command --instance-ids ${CP_INSTANCE_ID} \\"
echo "    --document-name AWS-RunShellScript \\"
echo "    --parameters '{\"commands\":[\"tail -50 /var/log/orchestrator-boot.log\"]}' \\"
echo "    --region ${REGION}"
echo ""
echo "  # Check S3 for results:"
echo "  aws s3 ls ${S3_PREFIX}/ --recursive"
echo ""
echo "  # Check instance status:"
echo "  aws ec2 describe-instances --instance-ids ${CP_INSTANCE_ID} \\"
echo "    --query 'Reservations[0].Instances[0].State.Name' --output text"
