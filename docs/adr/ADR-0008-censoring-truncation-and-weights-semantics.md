# ADR-0008: Censoring-truncation-and-weights-semantics

Status: Proposed

## Decision

Represent exact, left-, right- and interval-censored observations explicitly,
separately from observation-specific truncation intervals and from weights.
Censoring contributes probability mass/survival terms; truncation conditions
each observation on its observable interval.  Frequency weights and analytic
weights have distinct declared semantics.

## Consequences

Each supported combination gets a tested log-likelihood, numerical tail-stable
implementation and reference comparison.  Unsupported combinations fail
loudly.  Integer frequency weights are tested against replicated data within
the reproducibility tolerance; arbitrary survey/importance weights do not
inherit that claim automatically.
