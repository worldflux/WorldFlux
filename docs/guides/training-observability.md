# Training Observability

Guide for monitoring training health and generating structured training reports.

## Overview

WorldFlux automatically generates structured training reports with health signals, loss curve analysis, and actionable recommendations.

## Training Reports

After training completes, a `training_report.json` is saved to the output directory:

```python
from worldflux.training import Trainer

trainer = Trainer(model, config=config)
trainer.train(data)
# → outputs/training_report.json generated automatically
```

## Health Signals

| Signal | What it Monitors | Severity Levels |
|--------|-----------------|-----------------|
| Loss convergence | Slope of recent loss values | healthy / warning / critical |
| Gradient health | NaN/Inf gradient events | healthy / critical |
| Numerical stability | Non-finite values | healthy / warning / critical |
| Throughput stability | Steps/sec degradation | healthy / warning |
| Latent health | Latent collapse indicators | healthy / warning |

## Health Score

The overall health score (0.0–1.0) is a weighted average of all health signals. A score above 0.8 indicates healthy training.

## WASR Integration

Training reports emit a `run.summary` event to WASR telemetry for centralized monitoring.
