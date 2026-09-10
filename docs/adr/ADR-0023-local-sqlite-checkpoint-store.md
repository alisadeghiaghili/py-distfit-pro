# ADR-0023: Local SQLite checkpoint store

Status: Accepted

Owner: Ali Sadeghi Aghili

Decision scope: persistent checkpoint storage for the 0.6 milestone.

## Context

ADR-0015 intentionally limits the current implementation to an in-memory
checkpoint test double. A file-replacement prototype was reviewed and rejected:
per-instance locks do not protect multiple processes, a shared temporary name
can collide, and atomic replacement alone does not establish crash durability.

The 0.6 milestone needs one local, dependency-free persistence backend that
supports a sequential pure-reducer/CAS protocol across separate processes on
supported Windows and Linux environments. It must retain no raw input in its
public provenance and must distinguish integrity failures from storage failures.

## Decision

Use the Python standard-library `sqlite3` module for one local checkpoint
database. The backend supports local filesystems only; network filesystems,
replication, encryption, authentication, hostile-writer resistance, and
distributed coordination are outside this decision.

The database contains one versioned checkpoint row. It stores serialized state
as a BLOB and all `CheckpointRecord` fields needed to recompute its SHA-256
integrity checksum. Connections set `PRAGMA journal_mode=WAL` and
`PRAGMA synchronous=FULL`. Creation uses `BEGIN IMMEDIATE` and fails when a
checkpoint already exists. Compare-and-swap uses `BEGIN IMMEDIATE`, validates
the candidate before mutation, and updates only when the stored generation is
the expected generation. A zero-row update is a typed conflict.

The backend gives a local atomic-commit guarantee within SQLite's documented
failure model. It does not claim survival of every hardware, filesystem, or
power-loss failure. If the caller cannot determine whether `COMMIT` completed,
it reconnects and compares generation, checksum, operation token and operation
digest. It returns the candidate only on an exact match, returns conflict for a
definitive different committed state, and otherwise raises a typed uncertain
commit failure.

The backend rejects unsupported formats, malformed storage records and checksum
mismatches without migration or checkpoint mutation. File paths, source
revisions, state bytes, SQLite messages and exception messages never enter
public error context or provenance.

## Scope

This adds a single-host checkpoint store for sequential pure reducers. It does
not make CSV acquisition checkpoint-replayable, introduce a public resume API,
permit parallel unresolved updates, add an idempotent external sink, or extend
the 0.5 release claim.

## Evidence

Required executable evidence:

- `CKPT-SQL-01`: create, reopen and binary-state round trip.
- `CKPT-SQL-02`: independent instances and spawned processes race one expected
  generation; exactly one commits and the other returns `CHECKPOINT_CONFLICT`.
- `CKPT-SQL-03`: malformed row, unsupported format and checksum corruption
  return distinct declared typed errors without mutation.
- `CKPT-SQL-04`: create collision, read-only storage, lock timeout and write
  failure return typed storage failures without leaking private details.
- `CKPT-SQL-05`: fault injection before transaction, before update, before
  commit and after uncertain commit response preserves either old or whole new
  state; reconciliation is deterministic.
- `CKPT-SQL-06`: an abruptly terminated writer leaves an openable database
  whose row is complete old or complete new state, never a partial record.
- `CKPT-SQL-07`: public provenance and errors exclude source revision, path,
  state and SQLite exception text.
- `CKPT-SQL-08`: the same suite runs on Windows and Linux supported Python
  versions; retained evidence binds the candidate SHA and platform facts.

## Test implications

Add RED tests before implementation for every evidence ID, including process
races and injected transaction boundaries. Use a test-only seam for uncertain
commit responses; it must not expose a production bypass. Retain the existing
DS-08 and DS-09 contracts and run them against the SQLite backend once it
exists. Update coverage and mutation evidence only after behavior tests pass;
changing the coverage denominator is not a substitute for tests.

## Dependencies

ADR-0005, ADR-0006, ADR-0015 and ADR-0022 remain binding. This proposal does
not supersede their claim limits.

## Consequences

SQLite provides cross-process transactions without separate platform-specific
file-lock implementations, while deliberately limiting the first persistence
backend to a local single-host deployment. The public API must state those
limits. SQLite lock contention is observable and typed, never silently retried
as a second application of an operation.

## Exit criteria and effort class

This record was accepted by Ali Sadeghi Aghili on 2026-09-10. Acceptance
authorizes only the scoped local backend. The 0.6 checkpoint milestone
additionally requires every evidence ID above, quality gates,
three-locale documentation, and an end-to-end resumable operation before any
public persistence claim. Effort class: medium, one focused engineering wave.
