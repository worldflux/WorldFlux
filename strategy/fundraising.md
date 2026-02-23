# Fundraising Strategy

## Round Design

### Pre-Seed: $500K-$1.5M | Q2-Q3 2026

- **Valuation**: $5M-$10M post-money (SAFE + valuation cap)
- **Triggers**:
  - PyPI published
  - CLI operational
  - DreamerV3/TD-MPC2 parity proof CI passing
  - HF Spaces/Discord live
  - 3-5 lighthouse users
- **Use of funds**: 12-18 months runway, GPU credits (parity campaigns $5-15K/model), brand/design, conferences (NeurIPS/ICML/CoRL)
- **Structure**: Post-money SAFE (YC standard), MFN clause, no board seat. Option pool NOT created (defer to Seed to avoid pre-seed dilution)

### Seed: $3M-$6M | Q1-Q2 2027

- **Valuation**: $20M-$35M post-money (priced round recommended)
- **Triggers**:
  - GitHub 500+ stars
  - Monthly 1,000+ PyPI installs
  - 5+ third-party plugins
  - `worldflux verify` real mode operational
  - 3+ paper citations
  - 1-2 enterprise LOIs / design partnerships
- **Use of funds**: 3-4 engineers (ML infra, cloud backend, DevRel), cloud infra build-out, enterprise pilots, JEPA/V-JEPA2 production-ization
- **Structure**: Series Seed Preferred, 1x non-participating LP, 15% option pool, 5-seat board (2 founders + 1 investor + 2 independent)

### Series A: $15M-$25M | Q4 2027-Q2 2028

- **Valuation**: $80M-$150M post-money
- **Triggers**:
  - $500K+ ARR
  - 5,000+ stars
  - Monthly 10,000+ installs
  - 3+ paying enterprise customers
  - Data flywheel demonstrated
  - 10+ paper citations
  - Robot deployment demo
- **Use of funds**: 15-20 person team, sales team (2-3 AE, 1 SE), GPU cluster, Physics/Renderer3D/SSM family development, SOC 2 compliance

---

## Investor Targeting

### Tier 1: AI-Native VC

| VC | Fit Rationale | Narrative |
|---|---|---|
| **a16z** (Infra/AI) | Martin Casado's infra thesis (Replit, Vercel, Anyscale, Databricks) | "What HuggingFace did for Transformers in 2019, WorldFlux does for World Models" |
| **Lux Capital** | "Atoms + Bits" thesis, Physical AI fit | "The infra layer that makes Physical AI safe and verifiable" |
| **Radical Ventures** | AI-focused, deep tech DD, Cohere etc. | Pure technical depth: parity proofs, 5-layer architecture, contract freeze |
| **Index Ventures** | HashiCorp (Terraform) track record. "Protocol to platform" understanding | "`worldflux verify` is `terraform plan` for World Models" |

### Tier 2: Robotics / Physical AI Strategic

- **Toyota Ventures**: Automotive + robotics. Verifiable World Model training
- **Samsung NEXT**: Consumer robotics, smart home
- **Qualcomm Ventures**: Edge deployment (5M-317M params are edge-friendly)
- **NVIDIA NVentures**: Series A and beyond. Establish position first, then approach

### Tier 3: Accelerators / Angels

- **Y Combinator**: Optimal for pre-seed fast ramp (7% dilution)
- **Angel candidates**:
  - Pieter Abbeel (RL/robotics authority)
  - Andrej Karpathy (World Model advocate)
  - Soumith Chintala (PyTorch creator, design validation)

---

## Moat Analysis

| Moat | Strength | Mechanism |
|---|---|---|
| **Parity proof protocol** | Strongest, most unique | TOST equivalence + Holm correction + Bayesian HDI. Zero competitors have this level of statistical rigor. If 20+ papers cite it, switching costs become massive |
| **Public API contract freeze** | Strong | `public_contract.py` freezes all public symbols and function signatures in CI. Auto-blocks breaking changes. Ecosystem stability guarantee |
| **5-layer plugin architecture** | Medium-strong | Protocol-based (structural typing). Entry-point plugins for third-party extension create bidirectional network effects |
| **Data flywheel** (Phase 2+) | Potentially strongest | Trajectory accumulation improves pretraining, attracts users, generates more trajectories. Physical world data is scarce and expensive ($100-1000+/hr) |
| **Protocol standardization** | Strongest long-term | Contract freeze + parity proofs + CLI = protocol. Same dynamic as HTTP replacing Gopher |

---

## Risk Mitigation for Investors

| Risk | Response |
|---|---|
| **Solo founder / bus factor** | ~58,700 lines of source, 304 test files (0.74:1 test ratio), contract freeze, comprehensive docs. Pre-seed funds immediately hire engineer #2 |
| **Big tech enters** | Multi-architecture design (11 families). If DeepMind ships SDK, we wrap it as another backend. Historically big tech publishes papers but not production tooling (PyTorch -> HuggingFace pattern) |
| **World Models still research-stage** | Tesla FSD, Waymo already use learned dynamics models. Phase 1 is pure devtool revenue. Parity proofs provide immediate value for academic reproducibility |
| **OSS monetization is hard** | HashiCorp $5.8B exit, Databricks $134B valuation, HuggingFace $4.5B. Phase 2 is cloud services, Phase 3 is usage-based |
| **PyTorch dependency** | 5-layer Protocol interfaces are framework-agnostic by design. JAX backend is addable as plugin. PyTorch holds 85%+ research market share |

---

## Pitch Narratives

### Pre-Seed

> "There are 847 World Models papers on arXiv. Zero production frameworks. We're building that infra layer. `pip install worldflux` gives you a mathematically verified World Model in one line."

### Seed

> "Since launch, [X] researchers are using WorldFlux, [Y] papers cite our parity proofs, and [Z] companies are requesting the managed cloud version. We're building the trajectory data flywheel."

### Series A

> "WorldFlux is the dominant World Model framework. We're building the cloud imagination -> safe distillation -> robot deployment loop, targeting $10M+ ARR."

---

## Financial Model

| Phase | Revenue Model | Year 1 ARR Target | Gross Margin |
|---|---|---|---|
| Phase 1 (DevTool) | Free OSS (adoption metrics focus) | $0 | N/A |
| Phase 2 (SaaS) | Cloud training ($2-5/GPU-hr) + Enterprise verification ($5-25K/mo) | $600K-$1M | 40-90% |
| Phase 3 (Platform) | Imagination compute ($0.01-0.10/1K steps) + Robot license ($100-1K/unit/mo) | $10M+ | 75%+ |
