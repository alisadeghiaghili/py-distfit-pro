# ADR-0022: Public stream sources and lease-accounted delivery bounds

Status: Accepted

Owner: Ali Sadeghi Aghili

## Context

The initial CSV/exponential vertical and log-likelihood reducer both consume
streams, but exposed different input conventions.  The delivery buffer also
stopped charging bytes once a consumer removed an item from its queue, even
though the consumer still retained the payload lease.

## Decision

Expose a small stdlib-only `IterableDataSource` for a caller-owned iterable
and a `StreamSource` protocol for adapters.  A source always has immutable
`DataSourceMetadata`; its replayability declaration controls acquisition.
`SINGLE_PASS` sources can be acquired once and a further acquisition raises
the existing typed `PASS_BUDGET_EXCEEDED` failure.  `REPLAYABLE` and
`CHECKPOINT_REPLAYABLE` sources require an explicit iterator factory, so the
library never infers replayability from an arbitrary iterable.  The strict
CSV lifetime adapter remains the only shipped file adapter and declares
`SINGLE_PASS`.

Existing reducers keep accepting legacy nested iterables.  They also accept a
`StreamSource`, acquired through the common helper, so the generic source is
used end-to-end without widening claims to unimplemented adapters.

`BoundedChunkBuffer` charges a chunk from successful `put` until the consumer
calls `BufferedChunk.release()`. `get` transfers, rather than removes, that
charge.  The buffer composes its accounting callback with a caller callback;
release is idempotent and unblocks producers only after that accounting runs.
Cancellation drains queued leases and attempts every callback even when one
raises.  The first callback exception is re-raised after the drain; the buffer
remains cancelled and its byte observation is exact.

## Scope

This adds generic iterable sources and lease-safe bounded delivery. It does
not add Parquet, Arrow, pandas, Polars, Dask, database, generic CSV, RSS,
throughput, or broad out-of-core support.

## Test implications

- a generic source is consumed by both current reducers;
- single-pass acquisition has a typed failure;
- active consumer leases remain charged;
- a blocked producer proceeds only after release;
- cancellation invokes all queued callbacks and has a deterministic first-error policy.

## Consequences

Consumers must release every received `BufferedChunk`, normally in `finally`.
This is a deliberate correctness requirement: retaining a payload while not
charging it would make the advertised bound false.
