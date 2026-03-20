# Visual Benchmark Studio Design

Date: 2026-03-20
Owner: WorldFlux
Status: Draft

## 1. Context

WorldFlux already has a strong contract-first architecture, a unified factory surface, and a proof-oriented verification culture. However, current differentiation is stronger in architecture and validation than in immediately legible product output. DreamerV3 and TD-MPC2 are the mature cores, while Diffusion, Token, JEPA, and SSM are present but uneven in implementation maturity.

The strategic goal is not to maximize the number of model families exposed in the catalog. The goal is to turn WorldFlux into a system that can generate high-signal, technically differentiated, shareable outputs that create momentum with developers, social media, and investors.

The highest-priority success metric for the next iteration is the strength of the investor-facing demo. The user also explicitly prefers a technically differentiated demo over a purely flashy one. That means the first milestone should showcase Diffusion and SSM, not only DreamerV3.

## 2. Product Goal

Build a `Visual Benchmark Studio` for WorldFlux that runs multiple world model families under the same scenario and produces comparison-ready artifacts:

- rollout videos
- side-by-side comparison panels
- difference heatmaps
- compact benchmark summaries
- a shareable artifact pack suitable for GitHub, X, blog posts, and investor decks

The core message is:

`WorldFlux is not just a model implementation. It is a benchmarkable, visualizable operating system for world models.`

## 3. Non-Goals

- A full interactive hosted platform in the first release
- Broad support for every catalog family from day one
- Claiming proof-grade superiority without evidence-backed artifacts
- Building domain-specific robotics or driving stacks before the cross-family comparison core exists

## 4. Target Users

Primary users:

- ML researchers working on world models
- research engineers and applied engineers evaluating model families
- founders and technical storytellers preparing demos, benchmarks, and technical positioning

Secondary users:

- investors evaluating technical differentiation
- open-source contributors looking for a visible entry point into WorldFlux

## 5. Recommended Approach

Three approaches were considered:

1. Visual Benchmark Studio
2. Playable World Demo
3. Proof-First Research Console

The recommended approach is `Visual Benchmark Studio`.

Reasoning:

- It aligns with WorldFlux's actual strength: unified model interfaces and shared evaluation structure.
- It allows Diffusion and SSM to become visible differentiators immediately.
- It creates outputs that are inherently shareable.
- It is more defensible than a single flashy demo because the comparison itself is the product.

## 6. User Experience

The minimum successful experience is:

1. User runs a single command such as `worldflux demo benchmark ...`
2. WorldFlux executes the same scenario across DreamerV3, SSM, and Diffusion
3. WorldFlux generates:
   - comparison video
   - static panel image
   - per-model and cross-model metrics summary
   - artifact bundle in a single output directory
4. User can directly attach or publish those outputs

The first release should optimize for "I can generate an impressive, technically differentiated result in one command" rather than "I can explore every possible setting interactively."

## 7. Functional Scope

### 7.1 Visual Benchmark Studio v0

This is the phase-4 target if implementation proceeds.

Required capabilities:

- run a controlled scenario across multiple models
- normalize outputs into a shared comparison format
- generate videos and comparison visuals automatically
- export shareable static artifacts
- include a concise machine-readable summary

### 7.2 First-Class Model Set

The initial supported set should be limited to:

- DreamerV3
- SSM
- Diffusion

This is intentionally narrow.

DreamerV3 provides the stable baseline.
SSM and Diffusion provide the differentiation story.
Token and JEPA should wait until the comparison pipeline is stable.

## 8. System Design

### 8.1 Core Components

#### A. Scenario Runner

Purpose:
Run the same scenario, seed, observation stream, and action conditions across multiple models.

Responsibilities:

- resolve model profiles
- standardize scenario inputs
- execute rollout loops under a shared contract
- persist intermediate outputs for downstream rendering

Why this matters:
This is the layer that converts WorldFlux's unified API from an architectural claim into a visible benchmark product.

#### B. Model Adapter Profiles

Purpose:
Provide thin normalization layers for families with different output maturity.

Responsibilities:

- define how each model emits rollout-ready outputs
- normalize latent/output tensors into a shared render schema
- declare capability support and fallback behaviors

Initial adapters:

- DreamerV3 adapter
- SSM adapter
- Diffusion adapter

#### C. Visual Artifact Generator

Purpose:
Convert rollouts into comparison media.

Responsibilities:

- generate side-by-side rollout videos
- generate GT vs prediction grids
- generate difference heatmaps
- generate short GIF clips

This is the main viral surface.

#### D. Benchmark Summary Layer

Purpose:
Create compact, legible summaries that add technical credibility without overwhelming the visual story.

Initial metrics:

- rollout horizon degradation
- reward prediction error where available
- pixel or latent reconstruction difference
- basic runtime metadata

#### E. Shareable Output Pack

Purpose:
Emit a publishable output directory from a single run.

Output targets:

- `comparison.mp4`
- `comparison.gif`
- `comparison_grid.png`
- `diff_heatmap.png`
- `summary.json`
- `summary.md`

Future extension:

- static HTML report

## 9. Data Flow

1. CLI accepts scenario + model set + output directory
2. Scenario Runner resolves models and seeds
3. Each model runs under a normalized adapter profile
4. Outputs are converted into a shared artifact schema
5. Visual Artifact Generator renders visual outputs
6. Benchmark Summary Layer writes metrics summaries
7. Shareable Output Pack assembles final outputs

## 10. Error Handling

The system should fail in a way that preserves partial artifacts whenever possible.

Required behavior:

- if one model fails, the report should mark it as failed rather than aborting the whole bundle
- unsupported capability should degrade to "comparison unavailable" instead of a hard crash when feasible
- artifact generation errors should be isolated from rollout execution errors
- output directories must always contain a manifest describing what succeeded and failed

This is critical because demo workflows are time-sensitive. A user should not lose the entire run because one model profile is incomplete.

## 11. Testing Strategy

Testing should follow three layers.

### 11.1 Unit Tests

- adapter normalization logic
- artifact manifest generation
- summary metric calculation
- output pack assembly

### 11.2 Integration Tests

- DreamerV3 vs SSM comparison flow
- DreamerV3 vs Diffusion comparison flow
- partial-failure behavior
- deterministic artifact naming and manifest shape

### 11.3 Smoke Tests

- CPU-first short comparison path
- one-command artifact generation path

The same philosophy as `examples/quickstart_cpu_success.py` should apply here: there must be a short path that proves the feature works end-to-end.

## 12. Trade-Offs

### Chosen Trade-Offs

- Prefer static artifact quality over early browser interactivity
- Prefer 3 strong model families over broad weak family coverage
- Prefer comparison-driven differentiation over single-model showmanship
- Prefer artifact reproducibility over custom one-off demo scripts

### Rejected Trade-Offs

- Shipping a broad UI without stable normalized outputs
- Treating the entire catalog as equally mature
- Building investor-only demo code that does not help community adoption

## 13. Roadmap

### 13.1 3-Month Roadmap

Goal:
Investor-ready demo strength with real technical differentiation.

Deliverables:

- Scenario Runner
- DreamerV3 / SSM / Diffusion adapter profiles
- Visual Artifact Generator
- Shareable Output Pack
- compact benchmark summary
- CLI entry point for benchmark demo generation

Expected outcome:

- strongest immediate fundraising demo
- high shareability
- first public evidence of cross-family comparison value

### 13.2 6-Month Roadmap

Goal:
Turn the demo engine into a reusable comparison workflow for developers.

Deliverables:

- Token adapter support
- static HTML report generation
- multi-seed batch comparison
- local leaderboard generation
- short-form social export workflow

Expected outcome:

- stronger developer adoption
- more repeatable community usage
- better public comparison content

### 13.3 12-Month Roadmap

Goal:
Establish WorldFlux as the category-defining benchmark-and-demo layer for world models.

Deliverables:

- JEPA and V-JEPA2 comparison support
- browser playground
- hosted artifact gallery
- external plugin submission path
- evidence-backed public benchmark hub
- domain scenario packs for robotics and driving

Expected outcome:

- stronger community flywheel
- platform positioning rather than library positioning

## 14. Prioritized Feature Table

| Priority | Feature | Effort | Adoption Driver | Viral Potential | Technical Differentiation | Fundraising Contribution |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Visual Benchmark Studio v0 | M-L | Very High | Very High | High | Very High |
| 2 | Diffusion adapter hardening | M-L | High | Medium | Very High | High |
| 3 | SSM adapter hardening | M | High | Medium | High | High |
| 4 | Shareable Output Pack | S-M | High | Very High | Medium | High |
| 5 | Static HTML report | S-M | Medium | High | Medium | Medium |
| 6 | Token adapter support | M | Medium | Medium | High | Medium |
| 7 | Browser playground | L | High | High | Medium | Medium |
| 8 | Public benchmark hub | L | High | High | High | High |

## 15. Recommendation

The implementation target for the next phase should be:

`Visual Benchmark Studio v0`

with the narrowest viable scope:

`DreamerV3 vs SSM vs Diffusion comparison, exported as video + static artifact pack from a single CLI command`

This is the highest-leverage feature because it:

- matches the real architectural strengths of WorldFlux
- surfaces Diffusion and SSM as visible differentiators
- creates assets that travel outside the repository
- improves both fundraising narrative and open-source momentum

## 16. Open Risks

- Diffusion and SSM adapter quality may not be strong enough for compelling first outputs without targeted hardening
- artifact rendering may expose inconsistencies in current family outputs
- public messaging must not imply proof-grade maturity for families that are still experimental
- there is risk of over-building reporting before the core visual comparison path feels excellent

## 17. Phase 4 Entry Criterion

Proceed to implementation only if the first milestone stays narrow:

- one CLI path
- three model families
- static artifacts first
- no browser-first productization

If scope expands beyond that, the first release risks becoming broad but weak.
