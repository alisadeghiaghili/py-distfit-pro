# Architecture Decision Records

ADRs record decisions that affect correctness, public claims, or long-lived
interfaces. `Proposed` ADRs are working designs, not authority to claim the
capability publicly. `Accepted` ADRs are binding until superseded.

## Required record schema

Every ADR must contain: Status; Owner; Context; Decision; Scope; Evidence;
Test implications; Dependencies; Consequences; and, for multi-wave work,
measurable exit criteria and effort/budget class. Superseding ADRs link to the
prior record; accepted records are not silently rewritten. A Proposed record
may contain recommendation but must identify the decision owner and approval
needed.

| ADR | Title | Status |
| --- | --- | --- |
| 0001 | Greenfield Python-first and legacy policy | Accepted |
| 0002 | Statistical correctness and capability matrix | Proposed |
| 0003 | Functional core and immutable results | Proposed |
| 0004 | Floating-point reproducibility contract | Proposed |
| 0005 | Out-of-core backends and scale tiers | Accepted |
| 0006 | TDD, coverage, mutation and calibration gates | Accepted |
| 0007 | GoF and large-n reporting | Proposed |
| 0008 | Censoring, truncation and weights semantics | Proposed |
| 0009 | Dependencies, extras and lazy imports | Proposed |
| 0010 | v1 scope, non-goals and release gates | Accepted |
| 0011 | i18n and multilingual documentation | Accepted |
| 0012 | Competitive coverage strategy | Accepted |
| 0013 | Documentation toolchain | Accepted |
| 0014 | Evidence freezing and competitive claims | Proposed |
| 0015 | Retry/checkpoint transactional guarantees | Accepted |
| 0016 | Evidence-gated legacy salvage and migration ledger | Accepted; partially supersedes ADR-0001's no-port prohibition only |

## Dependency notes

- ADR-0015 depends on Accepted ADR-0005, ADR-0006 and ADR-0010. Any public
  durability, security or encryption claim also depends on a separate Accepted
  persistent-backend/security ADR; that record does not yet exist.
- ADR-0016 partially supersedes ADR-0001 only by permitting evidence-gated
  legacy salvage. ADR-0001's Python-first, independent-golden, and
  no-legacy-oracle decisions remain binding.
