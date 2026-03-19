# Learning Curve Report: {{ suite_id }}

Generated at: {{ generated_at_utc }}
Family: {{ family }}
Seeds: {{ seeds }}

## Summary

| Metric | Value |
|--------|-------|
| Total Tasks | {{ total_tasks }} |
| Seeds per Task | {{ n_seeds }} |
| Environment Steps | {{ env_steps }} |
| Parity Target | {{ parity_target }} |

## Per-Task Learning Curves

<!-- Filled by automated reporting. Each task gets a plot with:
     - X-axis: environment steps
     - Y-axis: episode return
     - Lines: mean across seeds with std shading
     - Comparison: WorldFlux (blue) vs Oracle (orange)
     - Reference: paper-reported final score (dashed horizontal line) -->

{% for task in tasks %}
### {{ task.name }}

| Metric | WorldFlux | Oracle | Delta (%) |
|--------|-----------|--------|-----------|
| Mean (final) | {{ task.wf_mean }} | {{ task.oracle_mean }} | {{ task.delta_pct }} |
| Std | {{ task.wf_std }} | {{ task.oracle_std }} | - |
| 95% CI | [{{ task.wf_ci_low }}, {{ task.wf_ci_high }}] | [{{ task.oracle_ci_low }}, {{ task.oracle_ci_high }}] | - |

![{{ task.name }} learning curve](./curves/{{ task.name }}.png)

{% endfor %}

## Statistical Summary

| Test | Result |
|------|--------|
| Non-inferiority (5% margin) | {{ ni_verdict }} |
| Mean drop ratio | {{ mean_drop_ratio }} |
| Upper CI (one-sided) | {{ ci_upper }} |
| Tasks within 95% CI | {{ tasks_within_ci }} / {{ total_tasks }} |
