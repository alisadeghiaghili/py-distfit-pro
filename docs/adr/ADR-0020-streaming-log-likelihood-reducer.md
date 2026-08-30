# ADR-0020: Exact-state streaming log-likelihood reduction

Status: Proposed

Owner: Ali Sadeghi Aghili

## Context

ADR-0019 provides five finite scalar log-density evaluators.  Callers also
need a one-pass way to reduce those scalar outputs without retaining raw
observations or making partition-dependent floating-point claims.

## Decision

Add a generic streaming reducer for the five canonical `FamilyId` values.  It
validates canonical parameters once through the registry, evaluates each
observation through `evaluate_log_density`, and stores the exact sum of every
successful binary64 log-density in integer units of `2^-1074`.  A finite
binary64 value is converted with `float.as_integer_ratio`; this is an exact
conversion of the already-rounded scalar output.

The contract is **not** exact real-arithmetic density or likelihood.  It is
exact summation of the successful binary64 log-density outputs, followed by
one correctly rounded binary64 finalization.  Scalar evaluation remains bound
by ADR-0019 and its platform/input contract.

The observation cap is `2^64 - 1`.  Every finite binary64 contribution has
absolute integer-unit magnitude at most

`(2^53 - 1) * 2^2045`.

Consequently the absolute exact total is at most

`(2^64 - 1) * (2^53 - 1) * 2^2045`,

whose integer bit length is 2162.  The state rejects any count or exact-unit
total outside those bounds.  This is a fixed-field state with a bounded
integer bit length under the declared count cap.  `math.fsum` is not used for
this claim because it has no public worst-case fixed-state bound.

State is frozen and slotted, binds a canonical `FamilyId` and an exact,
deterministic canonical-parameter signature, and retains no observations,
chunks, paths, localized text, or legacy objects.  State merge adds exact
integer units, is associative and commutative at state level, rejects family
or signature mismatches, and enforces the same count/total invariants.

Scalar typed failures produce a locale-neutral likelihood failure without a
partial total or raw value leakage.  Count-cap and non-representable final
total are separate closed failure codes.  Processed count, when reported on a
failure, is only the count completed before the failing observation and is not
a complete-input count.

## Scope

This authorizes exact-observation streaming log-likelihood only.  It does not
authorize fitting, censored likelihoods, arrays, NumPy/SciPy, legacy runtime
imports, out-of-core adapters, throughput claims, or process-memory claims.

## Evidence

Independent `Fraction`/integer reference checks will cover cancellation,
subnormals, maximum finite outputs, signed zeros, one-rounding cases,
overflow, count-boundary construction, all five dispatches, scalar failures,
ragged/lazy inputs, and merge/order/tree identity.  A retained generated-stream
matrix separately measures a bounded reducer state for 10k/100k/1m rows and
does not claim a process-memory or throughput guarantee.

## Test implications

- `LLR-01`: exact-integer state and one correctly rounded finalization.
- `LLR-02`: merge incompatibility/count/total boundaries and closed failures.
- `LLR-03`: order, chunks, and merge trees are bit-identical for identical
  scalar results.
- `LLR-04`: all five evaluator dispatches and scalar failures are closed.
- `LLR-05`: lazy/ragged streams are one-pass and retain no observations.
- `LLR-06`: a future retained scale artifact will be fail-closed and scoped to
  reducer state.

## Dependencies

ADR-0003, ADR-0004, ADR-0005, ADR-0006, ADR-0011, ADR-0016, and ADR-0019.

## Consequences

The reducer trades a small bounded big-integer state for deterministic exact
state arithmetic.  Its result is reproducible across input order, chunking,
and merge trees only when the scalar evaluator produces identical binary64
outputs; it does not overclaim cross-platform scalar equivalence.

## Exit criteria and effort class

Acceptance requires RED/green separation, independent reference and scale
evidence, coverage/watchdog updates, full local gates, and capability/readiness
reconciliation.  EN/FA/DE production documentation is explicitly a later
stage.  Effort class: medium.
