# WorldFlux Strategy

## Overview

WorldFlux fundraising strategy, go-to-market plan, and market analysis.

**Status**: Pre-Seed preparation | **Target**: Q2-Q3 2026 launch

**Core thesis**: World Models investment is growing 5x YoY ($1.4B in 2024 to $6.9B in 2025). There is zero direct competition for a unified World Model API. Window of 12-18 months.

## Documents

| Document | Contents |
|---|---|
| [Fundraising Strategy](fundraising.md) | Round design, investor targeting, moat analysis, pitch narrative, financial model |
| [Go-to-Market Strategy](gtm.md) | Phase 1-3 GTM, developer adoption, enterprise SaaS, Physical AI platform |
| [Market Analysis](market-analysis.md) | Market sizing, competitive positioning, timing analysis |
| [Action Items](action-items.md) | Immediate technical, GTM, and fundraising priorities |
| [Prompt Engineering Guide](prompt-engineering.md) | Prompt techniques for engineering/business analysis with Agent Teams |

## Key File References

| File | Strategic Importance |
|---|---|
| `src/worldflux/factory.py` | Core API. 1-line DX. Every pitch demo point |
| `src/worldflux/verify/runner.py` | `worldflux verify` real mode implementation (`_run_real`) + proof artifact wiring |
| `src/worldflux/parity/stats.py` | Non-inferiority bootstrap test. Most defensible technical asset |
| `src/worldflux/parity/harness.py` | Proof-grade parity harness |
| `src/worldflux/core/interfaces.py` | 5-layer plugin Protocol. Network effects foundation |
| `src/worldflux/core/registry.py` | Plugin system. Ecosystem growth foundation |
| `src/worldflux/_internal/public_contract.py` | API contract freeze. Switching cost generator |

## Strategic Constraints (Do NOT)

- Accept strategic investment from NVIDIA/Google before Series A (preserves optionality)
- Open-source the distributed AWS orchestration part of parity pipeline (Enterprise feature)
- Pivot toward "AI agents" buzzwords. Physical AI thesis is most defensible
- Accept funding from crypto-related funds (credibility cost within ML community)
