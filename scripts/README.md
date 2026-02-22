# Scripts

Developer and CI utility scripts for WorldFlux.

## Top-Level Scripts

| Script | Description | Usage |
|--------|-------------|-------|
| `check_critical_coverage.py` | Enforce per-file coverage thresholds for critical runtime modules | `python scripts/check_critical_coverage.py` |
| `check_docs_domain_tls.py` | Validate docs-domain TLS SAN entries and HTTPS reachability | `python scripts/check_docs_domain_tls.py` |
| `check_parity_suite_coverage.py` | Validate parity suite coverage policy and required-suite lock alignment | `python scripts/check_parity_suite_coverage.py` |
| `check_release_checklist_gate.py` | Ensure CI captures the documented release checklist gates | `python scripts/check_release_checklist_gate.py` |
| `check_release_metadata.py` | Validate release tag/version/changelog consistency | `python scripts/check_release_metadata.py` |
| `classify_public_contract_diff.py` | Classify public contract snapshot diff as none/additive/breaking | `python scripts/classify_public_contract_diff.py` |
| `compute_wasr.py` | Compute simple WASR metrics from local JSONL telemetry | `python scripts/compute_wasr.py` |
| `create_demo_gif.py` | Create demo GIFs for WorldFlux documentation | `python scripts/create_demo_gif.py` |
| `estimate_quality_gate_runtime.py` | Estimate runtime for quality-gate measurements using existing seed run results | `python scripts/estimate_quality_gate_runtime.py` |
| `export_public_contract_snapshot.py` | Export current public contract snapshot to JSON | `python scripts/export_public_contract_snapshot.py` |
| `generate_verification_report.py` | Generate machine + markdown release verification report artifacts | `python scripts/generate_verification_report.py` |
| `measure_quality_gates.py` | Measure reproducibility gates and loss drop thresholds for WorldFlux | `python scripts/measure_quality_gates.py` |
| `parity/aws_progress_audit.py` | Audit AWS proof-run pilot progress from SSM + S3 (official/worldflux split) | `python scripts/parity/aws_progress_audit.py --run-id <RUN_ID>` |
| `run_local_ci_gate.py` | Run local commands that mirror repository CI gates | `python scripts/run_local_ci_gate.py` |
| `update_public_contract_snapshot.py` | Update public contract snapshot in additive mode and block breaking changes | `python scripts/update_public_contract_snapshot.py` |
| `validate_parity_artifacts.py` | Validate fixed parity artifacts for release gating | `python scripts/validate_parity_artifacts.py` |

## Subdirectories

### `parity/`

Parity proof infrastructure for verifying WorldFlux model equivalence against reference implementations.

| Script | Description |
|--------|-------------|
| `run_campaign.py` | Run a parity proof campaign |
| `run_parity_matrix.py` | Run the full parity matrix across families and environments |
| `run_full_proof.py` | End-to-end parity proof execution |
| `merge_parity_runs.py` | Merge results from multiple parity runs |
| `stats_bayesian.py` | Bayesian statistical analysis for parity evaluation |
| `stats_equivalence.py` | Equivalence testing statistics |
| `metric_transforms.py` | Metric transformation utilities |
| `validity_gate.py` | Validity gate checks for parity runs |
| `suite_registry.py` | Parity suite registration and discovery |
| `contract_schema.py` | Contract schema definitions for parity proofs |
| `validate_matrix_completeness.py` | Validate completeness of the parity matrix |
| `report_markdown.py` | Generate markdown parity reports |
| `paper_comparison_report.py` | Generate paper-comparison reports |
| `plot_learning_curves.py` | Plot learning curves from parity runs |
| `aws_quota_planner.py` | Plan AWS quota requirements for cloud parity runs |
| `aws_distributed_orchestrator.py` | Distributed orchestrator for AWS-based parity runs |
| `fetch_oracles.sh` | Fetch oracle reference data |
| `setup_cloud_iam.sh` | Set up IAM roles for cloud parity infrastructure |
| `launch_cloud_orchestrator.sh` | Launch the cloud-based parity orchestrator |

### `demo/`

Demo recording scripts for documentation and social media.

| Script | Description |
|--------|-------------|
| `record_demo.sh` | 90-second terminal demo recording via asciinema |
| `record_short.sh` | 30-second compact demo for GitHub README embed and social media |
