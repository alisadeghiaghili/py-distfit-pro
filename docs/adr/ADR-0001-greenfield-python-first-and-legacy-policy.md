# ADR-0001: Greenfield Python-first and legacy policy

Status: Accepted

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- Python-first;
R deferred.

## Context

The legacy package has reported correctness and quality risks.  Reusing its
statistical core would make its behaviour the accidental specification.  v1
must become reliable before a second language multiplies the surface area.

## Decision

Build a new Python-first v1 with specifications and golden data outside package
code.  Retain legacy read-only as evidence of intended user scenarios and as
regression-case input; do not port its core or treat its results as an oracle.
R/CRAN work is deferred until Python has accepted capability and reference
evidence.

## Consequences

Short-term feature count falls, but every v1 claim can be scoped and tested.
Migration/compatibility policy, package name availability and the eventual R
implementation require separate decisions.
