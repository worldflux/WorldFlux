#!/usr/bin/env bash
# launch_cloud_orchestrator.sh
#
# Launch a cheap t3.medium "control-plane" instance that runs the parity
# orchestrator entirely in the cloud.  The orchestrator auto-provisions
# GPU worker instances and auto-terminates them when finished.
#
# Usage:
#   bash scripts/parity/launch_cloud_orchestrator.sh [--dry-run]
#
# Prerequisites:
#   - AWS CLI configured with credentials for us-west-2
#   - The IAM instance profile must allow ec2/ssm/s3 access
#
# The script:
#   1. Launches a t3.medium control-plane instance
#   2. Bootstraps git, python3, awscli via user-data
#   3. Clones the worldflux repo (main branch)
#   4. Runs the orchestrator with --auto-provision --auto-terminate
#   5. Uploads results to S3
#   6. Self-terminates the control-plane when done
#
# Monitor progress:
#   aws ssm start-session --target <control-plane-instance-id>
#   tail -f /var/log/cloud-init-output.log
#   # or check S3 directly for results

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────
REGION="us-west-2"
CONTROL_PLANE_TYPE="t3.medium"
CONTROL_PLANE_AMI="ami-0e5e1413a3bf2d262"  # Ubuntu 24.04 LTS us-west-2

# GPU worker configuration (passed to orchestrator --auto-provision)
GPU_INSTANCE_TYPE="p4d.24xlarge"
GPU_AMI="ami-068674ce56829a0ea"  # Deep Learning AMI with CUDA
FLEET_SIZE=1
FLEET_SPLIT="1,0"  # 1 official, 0 worldflux (same instance runs both)

# Networking / IAM (reuse existing parity infrastructure)
SUBNET_ID="subnet-017e9b9db46658c71"
SECURITY_GROUP_IDS="sg-0b6167f7cb8009323"
IAM_INSTANCE_PROFILE="worldflux-ec2-ssm-role"
KEY_NAME="worldflux"

# S3 and run configuration
S3_BUCKET="worldflux-parity"
RUN_TAG="cloud_proof_$(date -u +%Y%m%dT%H%M%SZ)"
S3_PREFIX="s3://${S3_BUCKET}/${RUN_TAG}"
WORLDFLUX_BRANCH="main"

# ── Parse flags ──────────────────────────────────────────────────────
DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── User-data script (runs on the control-plane at boot) ────────────
USERDATA=$(cat <<'USERDATA_EOF'
#!/bin/bash
set -euxo pipefail
exec > >(tee /var/log/orchestrator-boot.log) 2>&1

echo "=== Control-plane bootstrap: $(date -u) ==="

# Install essentials
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y git python3 python3-pip python3-venv awscli jq unzip

# Install latest AWS CLI v2 (the apt awscli is v1)
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -qo /tmp/awscliv2.zip -d /tmp/
/tmp/aws/install --update || true
rm -rf /tmp/aws /tmp/awscliv2.zip

# Fetch configuration from instance tags
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)

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

# Clone the repo
cd /opt
git clone --depth 1 --branch "${WF_BRANCH}" https://github.com/yoshinorisano/worldflux.git
cd worldflux

echo "=== Starting orchestrator: $(date -u) ==="
echo "Run tag: ${RUN_TAG}"
echo "S3 prefix: ${S3_PREFIX}"

# Run the orchestrator
export PYTHONUNBUFFERED=1

stdbuf -oL -eL python3 scripts/parity/aws_distributed_orchestrator.py \
    --region "${REGION}" \
    --manifest scripts/parity/manifests/official_vs_worldflux_v1.yaml \
    --full-manifest scripts/parity/manifests/official_vs_worldflux_full_v1.yaml \
    --run-id "${RUN_TAG}" \
    --s3-prefix "${S3_PREFIX}" \
    --phase-plan two_stage_proof \
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
    --gpu-slots-per-instance 8 \
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
echo "║  Cloud Parity Orchestrator Launch                          ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Control plane : ${CONTROL_PLANE_TYPE} (${CONTROL_PLANE_AMI})"
echo "║  GPU workers   : ${GPU_INSTANCE_TYPE} × ${FLEET_SIZE}"
echo "║  Region        : ${REGION}"
echo "║  Run tag       : ${RUN_TAG}"
echo "║  S3 prefix     : ${S3_PREFIX}"
echo "║  Branch        : ${WORLDFLUX_BRANCH}"
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
    --tag-specifications "$(python3 - "${RUN_TAG}" "${S3_PREFIX}" "${WORLDFLUX_BRANCH}" "${GPU_AMI}" "${GPU_INSTANCE_TYPE}" "${FLEET_SIZE}" "${FLEET_SPLIT}" "${SUBNET_ID}" "${SECURITY_GROUP_IDS}" "${IAM_INSTANCE_PROFILE}" "${KEY_NAME}" <<'PYTAGS'
import json, sys
a = sys.argv[1:]
tags = dict(zip(
    ["ParityRunTag","ParityS3Prefix","ParityBranch","ParityGpuAmi",
     "ParityGpuInstanceType","ParityFleetSize","ParityFleetSplit",
     "ParitySubnetId","ParitySecurityGroupIds","ParityIamProfile","ParityKeyName"], a))
tags["Name"] = "worldflux-parity-orchestrator"
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
