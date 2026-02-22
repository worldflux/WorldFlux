#!/usr/bin/env bash
# setup_cloud_iam.sh
#
# Add EC2/SSM orchestration permissions to the worldflux-ec2-ssm-role
# so the control-plane instance can auto-provision and manage GPU workers.
#
# Usage:
#   bash scripts/parity/setup_cloud_iam.sh [--dry-run]
#
# This is idempotent - safe to run multiple times.

set -euo pipefail

ROLE_NAME="worldflux-ec2-ssm-role"
POLICY_NAME="WorldfluxParityOrchestration"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
    esac
done

POLICY_DOC=$(cat <<'POLICY_EOF'
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "EC2InstanceManagement",
            "Effect": "Allow",
            "Action": [
                "ec2:RunInstances",
                "ec2:TerminateInstances",
                "ec2:StopInstances",
                "ec2:StartInstances",
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceTypes",
                "ec2:DescribeInstanceStatus",
                "ec2:DescribeTags",
                "ec2:CreateTags"
            ],
            "Resource": "*",
            "Condition": {
                "StringEquals": {
                    "aws:RequestedRegion": "us-west-2"
                }
            }
        },
        {
            "Sid": "SSMCommandDispatch",
            "Effect": "Allow",
            "Action": [
                "ssm:SendCommand",
                "ssm:GetCommandInvocation",
                "ssm:ListCommands",
                "ssm:ListCommandInvocations",
                "ssm:DescribeInstanceInformation"
            ],
            "Resource": "*"
        },
        {
            "Sid": "IAMPassRole",
            "Effect": "Allow",
            "Action": "iam:PassRole",
            "Resource": "arn:aws:iam::730335551417:role/worldflux-ec2-ssm-role"
        },
        {
            "Sid": "EC2NetworkForRunInstances",
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeSubnets",
                "ec2:DescribeSecurityGroups",
                "ec2:DescribeImages",
                "ec2:DescribeKeyPairs"
            ],
            "Resource": "*"
        }
    ]
}
POLICY_EOF
)

echo "Policy to apply to role '${ROLE_NAME}':"
echo "$POLICY_DOC" | python3 -m json.tool
echo ""

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would put-role-policy '${POLICY_NAME}' on role '${ROLE_NAME}'"
    exit 0
fi

echo "Applying policy '${POLICY_NAME}' to role '${ROLE_NAME}'..."
aws iam put-role-policy \
    --role-name "${ROLE_NAME}" \
    --policy-name "${POLICY_NAME}" \
    --policy-document "${POLICY_DOC}"

echo "Done. Verifying..."
aws iam list-role-policies --role-name "${ROLE_NAME}" --output text
echo ""
echo "Policy '${POLICY_NAME}' applied successfully."
