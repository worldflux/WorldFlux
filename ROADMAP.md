# Roadmap

This roadmap outlines planned milestones for WorldFlux. Dates are targets, not
commitments. Priorities may shift based on community feedback and upstream
ecosystem changes.

For current work in progress, see the
[GitHub Project board](https://github.com/orgs/worldflux/projects).

## v0.2.0 - Foundation Hardening (Q2 2026)

**Theme**: Distributed training readiness and actor-critic integration.

| Area | Items |
| --- | --- |
| Training | Native DDP support, actor-critic enabled by default |
| Parity | DreamerV3 actor-critic parity proof |
| API | Deprecation warnings for v0.2 bridge mode |
| Quality | 80%+ unit test coverage on core modules |
| OSS | SPDX headers, CodeQL, Dependabot, DCO enforcement |

**Status**: In Progress

## v0.3.0 - Scale and Prove (Q3 2026)

**Theme**: Long-horizon parity evidence and environment parallelization.

| Area | Items |
| --- | --- |
| Parity | Long-duration parity proofs (1M+ steps) for DreamerV3 and TD-MPC2 |
| Training | Environment parallelization (vectorized envs) |
| Training | FSDP integration (experimental) |
| Models | V-JEPA2 promoted to reference-family (conditional on upstream) |
| Infra | Multi-GPU CI smoke tests |

**Status**: Not Started

## v0.4.0 - S-Tier Polish (Q4 2026)

**Theme**: API cleanup, full distributed support, production hardening.

| Area | Items |
| --- | --- |
| API | v0.2 bridge mode removed (breaking) |
| Training | FSDP production-ready |
| Quality | 90%+ test coverage, mutation testing on critical paths |
| Docs | Complete API reference with runnable examples |
| OSS | 2+ active maintainers, public governance review |

**Status**: Not Started

## v1.0.0 - Production Ready (Q1 2027)

**Theme**: Stable public API, long-term support commitment.

| Area | Items |
| --- | --- |
| API | Public API freeze (semver stability guarantee) |
| Parity | All reference-family models with published evidence bundles |
| Quality | Security audit, performance benchmarks published |
| Packaging | Stable PyPI releases with LTS policy |
| Community | 5+ external contributors, plugin ecosystem established |

**Status**: Not Started

## How to Contribute

If a roadmap item interests you, check the corresponding GitHub Issues or open
a discussion. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup.

Roadmap feedback is welcome via GitHub Discussions or Discord.
