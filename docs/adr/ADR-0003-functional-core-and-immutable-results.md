# ADR-0003: Functional core and immutable results

Status: Proposed

## Decision

Families are stateless mathematical declarations.  Estimators, criteria and
GoF calculations are pure functions over explicit data/statistics and settings.
Public fit/report results are frozen value objects containing diagnostics and
provenance.  IO, visualization, localization and logging are outer adapters.

## Consequences

Pure functions make oracle, property and mutation tests practical and keep
streaming schedulers independent of statistical semantics.  Optimizer state
and data sources remain implementation details but their outcomes must be
captured in the immutable result.
