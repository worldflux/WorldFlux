# Proof Parity Runbook

This runbook defines the minimum operator sequence for proof-grade parity runs
on AWS for both DreamerV3 and TD-MPC2.

## Preconditions

- AWS credentials are configured and `aws sts get-caller-identity` succeeds
- required subnet / security groups / IAM instance profile are available
- parity manifests and canonical backend profiles are confirmed
- S3 prefix for the run is chosen before provisioning workers

## Standard Flow

1. Run AWS identity and quota preflight.
2. Choose a run id and S3 prefix.
3. Launch the family-specific parity helper.
4. Monitor shard progress with `scripts/parity/aws_progress_audit.py`.
5. Resume missing work with the rerun manifest when completeness fails.
6. Collect `equivalence_report.json`, `stability_report.json`, and `evidence_bundle.zip`.

## DreamerV3

- pilot: 10 seeds
- full proof compare: 20+ seeds
- rerun twice after the first successful proof verdict

Entry point:

```bash
bash scripts/parity/launch_dreamerv3_parity.sh stage-a --version=v3
```

## TD-MPC2

- alignment report must be `aligned` before compare
- pilot: 10 seeds
- full proof compare: 20+ seeds
- rerun twice after the first successful proof verdict

Entry point:

```bash
bash scripts/parity/launch_tdmpc2_parity.sh pilot
```

## Acceptance Artifacts

- `coverage_report.json`
- `validity_report.json`
- `equivalence_report.json`
- `equivalence_report.md`
- `stability_report.json`
- `evidence_bundle.zip`

Proof completion requires:

- `equivalence_report.json` final verdict pass
- `stability_report.json` with no verdict flip or metric sign flip
