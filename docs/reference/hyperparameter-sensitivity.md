# Hyperparameter Sensitivity Analysis

Generated at: 2026-03-21T20:52:01Z
Family: dreamerv3
Environment: atari100k_pong
Task ID: atari100k_pong
Env backend: gymnasium
Model profile: ci
Seeds: [0]
Steps per run: 12

This is an initial measured Dreamer sensitivity report. It is not a proof claim or a benchmark claim.

## Sensitivity Ranking

| Rank | Parameter | Sensitivity Score | Default | Default Safe |
| ---: | --- | ---: | ---: | --- |
| 1 | kl_dynamics | 0.0000 | 0.5 | No |
| 2 | kl_representation | 0.0000 | 0.1 | No |
| 3 | free_nats | 0.0000 | 1.0 | No |
| 4 | learning_rate | 0.0000 | 0.0001 | No |
| 5 | imagination_horizon | 0.0000 | 15.0 | No |

## Per-Parameter Details

### kl_dynamics

| Value | Mean Reward | Std |
| ---: | ---: | ---: |
| 0.1 | 0.00 | 0.00 |
| 0.3 | 0.00 | 0.00 |
| 0.5 **(default)** | 0.00 | 0.00 |
| 1.0 | 0.00 | 0.00 |
| 2.0 | 0.00 | 0.00 |

### kl_representation

| Value | Mean Reward | Std |
| ---: | ---: | ---: |
| 0.01 | 0.00 | 0.00 |
| 0.05 | 0.00 | 0.00 |
| 0.1 **(default)** | 0.00 | 0.00 |
| 0.5 | 0.00 | 0.00 |
| 1.0 | 0.00 | 0.00 |

### free_nats

| Value | Mean Reward | Std |
| ---: | ---: | ---: |
| 0.0 | 0.00 | 0.00 |
| 0.5 | 0.00 | 0.00 |
| 1.0 **(default)** | 0.00 | 0.00 |
| 2.0 | 0.00 | 0.00 |
| 5.0 | 0.00 | 0.00 |

### learning_rate

| Value | Mean Reward | Std |
| ---: | ---: | ---: |
| 3e-05 | 0.00 | 0.00 |
| 0.0001 **(default)** | 0.00 | 0.00 |
| 0.0003 | 0.00 | 0.00 |
| 0.001 | 0.00 | 0.00 |

### imagination_horizon

| Value | Mean Reward | Std |
| ---: | ---: | ---: |
| 5.0 | 0.00 | 0.00 |
| 10.0 | 0.00 | 0.00 |
| 15.0 **(default)** | 0.00 | 0.00 |
| 20.0 | 0.00 | 0.00 |
| 30.0 | 0.00 | 0.00 |

## Reproducing

```bash
python scripts/run_sensitivity.py --dry-run

# smoke / deterministic CI-sized execution
python scripts/run_sensitivity.py \
    --task-id atari100k_pong \
    --env-backend gymnasium \
    --model-profile ci \
    --seeds 0 \
    --steps 12 \
    --output reports/parity/sensitivity/dreamerv3_sensitivity.json

# render markdown from existing results
python scripts/run_sensitivity.py \
    --report-from reports/parity/sensitivity/dreamerv3_sensitivity.json \
    --output-md docs/reference/hyperparameter-sensitivity.md
```
