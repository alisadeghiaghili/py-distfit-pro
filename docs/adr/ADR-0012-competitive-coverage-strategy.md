# ADR-0012: Competitive coverage strategy

Status: Accepted

Owner: product owner (Ali) with named statistical, numerical/data-systems,
documentation/community and domain leads per accepted wave.

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- Python-first/R
deferred and reliability + censoring + big-data as the first domain vertical.
Acceptance fixes the platform ordering and guardrails; each future capability
cell still requires its own evidence and, where needed, an ADR.

## Context

The market spans scientific libraries, reliability tools, hydrology/EVT,
enterprise software and GUI simulation products. Family-count parity alone
would create a large, unvalidated and unmaintainable surface.

## Decision

Use `docs/competitive-*` as a source-led baseline. **Pareto-leading** means a
small target workflow set beats its stated baseline on published trust criteria:
support semantics, reproducible diagnostics, calibrated inference where
claimed, bounded scale behaviour and time-to-first-fit. It does not mean most
families or an unsourced overall ranking. Pursue it through evidence waves:

1. Evidence baseline: capability matrix, migration examples, source-backed
   competitor audit and reproducible calibration/benchmark site.
2. First domain vertical -- reliability + censoring + big-data: narrow native
   reliability families/estimators, cited censored likelihoods, reference
   validation, domain diagnostics and external-memory contracts.
3. Broaden the statistical core only after the first vertical is evidenced.
4. Hydrology/extremes/L-moments: native or carefully integrated methods with
   non-silent approximation/error evidence.
5. Mixtures/zero-inflation: identifiability, convergence and selection/failure
   contracts.
6. Bayesian integration before a competing inference stack.
7. Enterprise/GUI/export: tested adapters, versioned report schema and limited
   dialect-specific SQL.
8. R cross-language after Python stabilizes; agreement is tolerance-based, not
   an independent truth oracle.

Integration is preferable only when it preserves provenance and semantics. No
empty wrappers: every integration declares materialization, versions, errors,
support matrix, tests and ownership.

## Per-wave exits and dependencies

| Wave | Measurable exit | Key dependencies | Effort/budget |
| --- | --- | --- | --- |
| Evidence baseline | source-schema validation, source lock under ADR-0014 if accepted, reproducible migration/benchmark/calibration run | docs toolchain, CI storage, ADR-0014 decision | small/medium |
| First domain vertical | cited reliability/censoring cells agree with reference cases, pass calibration scope and scale contract, and report failures | statistical + numerical + reliability leads, datasets | large |
| Broader statistical core | every added advertised cell passes conformance/reference/calibration scope and scale contract | statistical + numerical leads | medium/large |
| Hydrology/extremes/L-moments | selected workflows have source, error contract and domain review | hydrology/EVT lead | medium/large |
| Mixtures/zero-inflation | convergence, identifiability and recovery/failure evidence published | optimization expertise | large |
| Bayesian integration | versioned adapter and end-to-end reproduction | external API stability | medium |
| Enterprise/GUI/export | export schema, privacy review, dialect tests and support owner | product/security budget | large |
| R cross-language | shared spec, independent goldens and tolerance matrix | R maintainer/CRAN capacity | large |

## Consequences

This is multi-year work, not a 90-day v1. A credible path needs a statistical
engineer, a numerical/data-systems engineer and sustained documentation/community
capacity; reliability, GUI and R waves need additional domain owners and
budget. Adoption work is first-class but never overrides ADR-0010 correctness,
scale or release gates.
