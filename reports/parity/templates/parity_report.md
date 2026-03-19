# Parity Verification Report

Generated at: {{ generated_at_utc }}
Schema version: worldflux.parity.report.v1

## Campaign Overview

| Property | Value |
|----------|-------|
| Suite ID | {{ suite_id }} |
| Family | {{ family }} |
| Seeds | {{ seeds }} |
| Total Tasks | {{ total_tasks }} |
| Parity Threshold | {{ margin_ratio }} |
| Confidence Level | {{ confidence }} |

## Results Summary

| Metric | Value |
|--------|-------|
| Overall Verdict | **{{ verdict }}** |
| Tasks Passing | {{ tasks_pass }} / {{ total_tasks }} |
| Mean Drop Ratio | {{ mean_drop_ratio }} |
| Non-inferiority Upper CI | {{ ci_upper }} |
| Margin | {{ margin_ratio }} |

## Per-Task Results

| Task | WorldFlux Mean | Oracle Mean | Drop (%) | p-value (Welch) | Cohen's d | Within CI |
|------|---------------|-------------|----------|-----------------|-----------|-----------|
{% for task in tasks %}
| {{ task.name }} | {{ task.wf_mean }} | {{ task.oracle_mean }} | {{ task.drop_pct }} | {{ task.p_value }} | {{ task.cohens_d }} | {{ task.within_ci }} |
{% endfor %}

## Statistical Tests

### Non-inferiority Test (Bootstrap)
- Sample size: {{ ni_sample_size }}
- Mean drop ratio: {{ ni_mean_drop }}
- One-sided upper CI: {{ ni_ci_upper }}
- Margin: {{ ni_margin }}
- Verdict: {{ ni_verdict }}

### Multiple Testing Correction (Benjamini-Hochberg)
- Alpha: {{ bh_alpha }}
- Rejected (after FDR): {{ bh_rejected_count }} / {{ total_tasks }}

## Environment

| Component | Version |
|-----------|---------|
| Python | {{ python_version }} |
| PyTorch | {{ torch_version }} |
| CUDA | {{ cuda_version }} |
| Platform | {{ platform }} |

## Badge

{{ badge_svg }}
