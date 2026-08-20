# ADR-0005: Out-of-core backends and scale tiers

Status: Accepted

Owner: numerical/data-systems lead (TBD)

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- the first domain
vertical is reliability + censoring + big-data; multipass work on a
`single_pass` source is rejected and spooling is explicit opt-in.

## Decision

Expose in-memory, chunked and parallel schedulers behind one data-source
protocol. The protocol declares `single_pass`, `replayable` or
`checkpoint_replayable`; a planner enforces `max_passes` before reading. Spool
is an explicit opt-in adapter with disk budget and cleanup policy. Classify each
operation, not merely each family: A-reg (one-pass,
fixed-state regular model), A-boundary (one-pass support/min/max statistics),
A-frequency (`O(unique)` state), B (adaptive multipass likelihood), C
(external-memory objective), or D (explicit approximation).

Every source declares stable identity, schema/version and redaction policy.
Adapters declare replayability plus ordering and offset guarantees. Chunks use
stable IDs and offset ranges; empty chunks are valid. Duplicate, lost or
out-of-order chunks fail with distinct typed errors unless declared semantics
and an idempotency key permit them. `chunk_bytes` and `max_inflight_bytes` are
hard observable budgets with backpressure and cancellation semantics.

## Consequences

Each matrix cell publishes accumulator state, pass budget, replayability,
chunk/row-ID semantics, memory/inflight-byte bounds, checkpoint/retry policy,
backpressure and cancellation behaviour. Missing/duplicate/out-of-order chunks
are typed failures; there is no unlabeled partial result. PKD informs regular
fixed-support models but does not license an iff claim for
parameter-dependent-support families. Tier B/C claims are about objective
evaluation and convergence contract, not a guarantee of global MLE.

Retries require idempotent updates. Checksum mismatch, source mutation, retry
exhaustion, cancellation and missing data are distinct typed causes.
Checkpoints retain accumulator schema/version and source range; incompatible
resume is rejected. Any partial result includes `complete=False`, exact missing
ranges and its typed cause. Provenance includes its schema/version, source
identity/hash or redaction, ranges/counts, adapter/library versions,
chunk/pass/budget policy, estimator/RNG/approximation and checkpoint metadata.

## Evidence and test implications

The first vertical slice must continuously test empty/ragged chunks, IDs and
offsets, duplicate/lost/out-of-order delivery, single-pass rejection of
multipass plans, explicit spool, pass enforcement, retry idempotency,
checkpoint compatibility, bounded state/inflight bytes, cancellation and
redacted provenance. The DS-01--DS-12 test harness specified in the v1 test
plan is **NOT IMPLEMENTED**; a scale claim is not deferred to a late
performance phase and cannot be advertised before the relevant tests exist.
