# Immediate Action Items

## Technical Priorities (Pre-Launch)

### 1. Validate `worldflux verify` real mode on GPU

- **File**: `src/worldflux/verify/runner.py:216`
- **Status**: `_run_real()` is implemented and wired to the proof pipeline
- **Gap**: End-to-end CUDA smoke run is not yet recorded
- **Why**: This is the core of the "safety protocol" narrative. Every pitch demo should show a verified path
- **Approach**: Run `worldflux verify --target <ckpt> --device cuda` on a rented GPU instance and archive artifacts
- **Priority**: P0-soft

### 2. Production-ize JEPA / V-JEPA2

- **Files**: `src/worldflux/models/jepa/`, `src/worldflux/models/vjepa2/`
- **Why**: LeCun's AMI Labs wave. JEPA demand will spike
- **Target**: Move from EXPERIMENTAL to REFERENCE maturity with parity proofs
- **Priority**: P1

### 3. HuggingFace Hub integration

- **Pattern**: `save_pretrained` / `from_pretrained` for model distribution
- **Why**: Model distribution channel, familiar API for ML community
- **Priority**: P1

### 4. Parity proof methodology paper

- **Target**: NeurIPS / ICML workshop submission
- **Content**: TOST + Holm correction + Bayesian HDI methodology (sufficient novelty)
- **Assets**: `src/worldflux/parity/stats.py`, `src/worldflux/parity/harness.py`
- **Priority**: P1

---

## GTM Priorities

1. **Launch preparation**: README optimization, SEO targeting ("world models python", "dreamerv3 pytorch")
2. **HN / Reddit / Twitter(X) launch posts**
3. **5-post launch blog series** (see content strategy in [gtm.md](gtm.md))
4. **ICML 2026 (July) workshop submission deadline** -- prepare paper
5. **DevRel team planning**: 1 research-focused + 1 practitioner-focused

---

## Fundraising Priorities

1. **Pre-Seed SAFE document preparation** (YC standard post-money SAFE)
2. **YC application** (evaluate: distribution speed + network value)
3. **Angel investor approach list** (Pieter Abbeel, Andrej Karpathy, Soumith Chintala)
4. **Pitch deck production** (build around `worldflux verify --demo` demo sequence)

---

## Timeline

```
Q2 2026:  Public launch + Pre-Seed raise
          |- PyPI publish, CLI operational
          |- Parity proof CI passing
          |- HF Spaces / Discord live
          |- 3-5 lighthouse users
          |- ICRA attendance (May)

Q3 2026:  Growth phase
          |- 500 GitHub stars target
          |- ICML exhibition (July)
          |- worldflux verify real mode (Month 9 target)
          |- JEPA/V-JEPA2 production push

Q4 2026:  Community building
          |- CoRL sponsorship (November)
          |- NeurIPS presence (December)
          |- 2,000 stars target
          |- 5+ third-party plugins

Q1 2027:  Seed preparation
          |- 1,000+ monthly PyPI installs
          |- 3+ paper citations
          |- 1-2 enterprise LOIs
          |- Seed raise begins

Q2-Q3 2027:  Seed close + enterprise
              |- $3-6M Seed closed
              |- Team expansion (3-4 engineers)
              |- Enterprise pilot begins

Q4 2027+:  Series A preparation
           |- $500K+ ARR target
           |- 5,000+ stars
           |- Robot deployment demo
```
