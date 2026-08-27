# ADR-0018: CSV lifetime adapter and one-pass exponential orchestrator

Status: Proposed

Owner: Ali Sadeghi Aghili

## Context

ADR-0017 proves the exponential sufficient-statistic reducer for an in-memory
iterable, but it deliberately makes no claim about a production adapter or
bounded ingestion.  ADR-0005 requires a real adapter to expose its identity,
replayability, chunk and byte contracts, delivery order, failure surface, and
provenance before a scale claim can be made.

The first adapter must be narrower than a convenience CSV reader.  Flexible
inference, coercion, and row skipping make a reliability analysis
non-reproducible and can turn malformed source data into an apparently valid
fit.  The first CSV vertical therefore needs a closed dialect, a visible
schema, and an end-to-end orchestration outcome that never releases a partial
scientific fit.

## Decision

The initial stdlib-only adapter is `veridist.adapters.csv_lifetimes`.  It reads
one UTF-8-sig, comma-delimited CSV dialect through Python's `csv` module.  It
does not infer encoding, dialect, headers, types, missing values, or schema.
`CsvLifetimeSchema(time_column, event_observed_column)` names the two required
columns exactly.  The header contains each declared column exactly once and no
additional columns.  The event token is exactly `1` for an event and exactly
`0` for independent right censoring.  Times and tokens are parsed under a
published strict grammar; there is no coercion, NA policy, blank-row skipping,
or silent repair.  Empty input, header-only input, blank records, duplicate,
missing, extra, malformed, and nonconforming rows receive distinct typed,
locale-neutral adapter failures.

Logical record offsets are zero-based data-record indices: the header is not a
record, and a quoted embedded newline is one logical record.  Chunk IDs and
row ranges are deterministic functions of this logical sequence, not physical
file lines.  The adapter is replayable, but the exposed `fit_exponential_csv`
operation obtains exactly one source iterator and requires one pass.  It does
not spool, retry, checkpoint, or parallelize; DS-08 and DS-09 are not
applicable to this cell.

The public caller supplies an opaque `PublicSourceId`; the adapter never
derives it from a path.  Public errors and result/provenance fields contain no
path, raw cell, parser exception text, or private file revision.  File identity
is sampled before and after the single read to detect observable mutations on a
best-effort basis.  This cannot prove a stable snapshot where a filesystem
does not expose a reliable identity or mutation occurs without a distinguishable
identity change.  For this cell, identity sampling itself is mandatory: an
unavailable initial or final sample is the typed fail-closed
`SOURCE_REVISION_UNAVAILABLE` / `identity_unavailable` failure, rather than a
successful run with an unverified identity.  An optional SHA-256 may cover only bytes actually consumed;
it does not prove an independently snapshotted source.

`CsvLifetimeLimits(chunk_bytes, max_inflight_bytes)` accepts positive built-in
integers only; booleans are rejected and `max_inflight_bytes >= chunk_bytes`.
For this sequential adapter, `chunk_bytes` is the deterministic Python
owned-object accounting unit defined below.  It is not serialized file bytes,
allocator usage, or RSS.  Each emitted record has a defined retained payload
cost, chunks stay within the declared logical bound, and one oversized record
produces a typed failure.  RSS and `tracemalloc` are separate measured
evidence, not substitutions for this accounting invariant.

The planned entry point is
`fit_exponential_csv(path, *, schema, source_id, limits)`.  A tested
source-opening protocol seam permits deterministic fault and mutation tests;
the public function does not yet join a registry or package top-level export.
Its frozen, slotted combined result is closed: a fit is non-`None` only after
complete execution.  A typed statistical non-estimate is still a complete
execution; any adapter or engine failure has `fit=None`.  No partial scientific
fit is exposed.

Cancellation is unsupported by this adapter API.  The adapter guarantees only
that a successfully opened stream is closed exactly once on success, typed
failure, early iterator close, or an unexpected exception.  The execution
outcome and provenance mapping is the closed mapping specified below.

`max_inflight_bytes` is deliberately not enforced by this sequential adapter:
the adapter has no producer/consumer buffer and therefore cannot honestly make
a peak-inflight claim.  The value is retained in `CsvLifetimeLimits` as the
future execution-layer `BoundedChunkBuffer` budget.  That layer must enforce
backpressure and prove the peak before claiming the bound; this adapter
enforces only `chunk_bytes`.

## Test implications

- `CSV-01`: exact schema, header, event-token, and strict time grammar.
- `CSV-02`: logical offsets and chunk IDs remain stable across chunk budgets,
  including quoted embedded newlines.
- `CSV-03`: logical retained-payload bounds, oversized-record rejection, and
  no whole-file materialization.
- `CSV-04`: exactly one replayable source pass for the one-pass fit.
- `CSV-05`: typed redacted open, decode, schema, row, and mutation failures.
- `CSV-06`: exact empty/header-only/blank/duplicate/malformed/extra-column
  policy.

The red tests precede the adapter and orchestration implementation.  They use
committed fixtures and test doubles, not legacy code or production code as a
numeric oracle.

## Dependencies

ADR-0002, ADR-0003, ADR-0004, ADR-0005, ADR-0006, ADR-0010, ADR-0011,
ADR-0013, ADR-0016, ADR-0017.

## References

- Python standard library, [`csv` module documentation](https://docs.python.org/3/library/csv.html), moving-current documentation; retrieval date and Python version must be recorded with implementation evidence.
- RFC 4180, [Common Format and MIME Type for CSV Files](https://www.rfc-editor.org/rfc/rfc4180), informational format reference.

## Consequences

This is a narrow, evidence-first adapter, not broad CSV compatibility.  It
adds the first genuine source-to-fit seam and bounded logical ingestion
contract, while withholding production-scale, cancellation, checkpoint,
retry, SHA snapshot, and RSS claims until their specific evidence exists.

## Closeout reconciliation -- 2026-08-27

The narrow wrapper is now exported at the package top level together with only
its schema, limits, opaque source identifier, and closed result type. The
retained `SCALE-CSV-EXP-01` artifact supplies a frozen 10k/100k/1m by
32KiB/64KiB/128KiB trace, checked against independent generated bytes and
Decimal sufficient facts. This ADR remains **Proposed**, not Accepted: the
formal mutation runner is absent, full documentation table/code/formula RTL
render evidence is not complete, and the retained trace does not prove a
portable RSS ceiling, general big-data support, cancellation, retry,
checkpointing, or another adapter.

## Corrective decision record -- 2026-08-25

The exact ASCII time grammar is
'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?'.  It is parsed with
Decimal before construction of a lifetime observation.  Exact zero is valid.
Any finite non-negative Decimal whose float conversion is non-finite, or whose
value is positive but converts to float zero, is invalid time input.  There is
no silent floating-point overflow or underflow conversion.

CSV failures extend the closed FailureCode enum only with
SOURCE_OPEN_FAILED, SOURCE_DECODE_FAILED, SOURCE_SCHEMA_INVALID, and
SOURCE_ROW_INVALID.  The adapter reuses CHUNK_TOO_LARGE for an oversized
record and SOURCE_REVISION_MISMATCH for detected source mutation.
CsvLifetimeAdapterError is an EngineContractError subclass.  Its allowlisted
immutable context distinguishes only closed non-sensitive reasons:
SOURCE_SCHEMA_INVALID uses header_missing, header_duplicate, or
header_columns_mismatch; SOURCE_ROW_INVALID uses blank_record,
malformed_record, invalid_time, or invalid_event_token; SOURCE_DECODE_FAILED
uses invalid_utf8; SOURCE_OPEN_FAILED uses open_failed; CHUNK_TOO_LARGE uses
record_too_large; and SOURCE_REVISION_MISMATCH uses source_mutated.  It never
exposes a path, cell, parser exception text, or private file revision.

The caller supplies a typed PublicSourceId, not an arbitrary string.  It must
satisfy the existing opaque src_<32 lowercase hexadecimal> public contract.
An emitted row identity remains source-ID plus logical row offset.  Chunk IDs
are opaque chk_<32 lowercase hexadecimal> values derived from the public source
ID and each chunk's row range.  They may change as a chunk budget changes, so
only row identity, not chunk identity, is stable across budgets.

The empty byte stream and a UTF-8 BOM-only stream fail before a header as
SOURCE_SCHEMA_INVALID with reason header_missing.  A header-only source is
valid data and its orchestrated result is a CompleteOutcome with the existing
EMPTY_SAMPLE statistical non-estimate.  A blank data record is
SOURCE_ROW_INVALID with reason blank_record.  Internal BOMs, missing, extra,
or wrong-order header names are SOURCE_SCHEMA_INVALID with reason
header_columns_mismatch.  Extra or missing data fields and malformed CSV syntax
are delivery failures with SOURCE_ROW_INVALID and reason malformed_record.
A syntactically valid quoted multiline time field is one logical record; its
newline makes it SOURCE_ROW_INVALID with reason invalid_time and its
zero-based logical record offset.  Invalid time/event values are
SOURCE_ROW_INVALID with reason invalid_time or invalid_event_token.

The retained-payload byte contract is closed and Python-specific: a chunk's
retained_payload_bytes is the recursive sum of sys.getsizeof over its declared
owned retained object graph, with object identities de-duplicated.  The graph
is exactly the chunk object, its frozen ChunkEnvelope, its observations tuple,
each observation, and their owned scalar objects.  Shared class objects,
Enum members, and static module constants are excluded.  Every dynamically
reachable string, including chunk and public-source IDs, is counted once by
identity.  The adapter records CPython,
Python-version, and platform labels with a measured evidence artifact.  This
is a deterministic accounting model only within a declared interpreter build.
Parser buffers, temporary builders, file/kernel caches, allocator overhead,
tracemalloc, and RSS are excluded from this logical accounting model. Separate
`SCALE-CSV-EXP-01` evidence now records descriptive tracemalloc peaks; RSS is
explicitly unknown on its Windows stdlib environment. Tests derive
observed costs from the accounting function rather than hard-code a byte
budget.  A guarded stream that rejects read(-1), readall, and unbounded
iteration is separate evidence against whole-file materialization; a
slope/peak benchmark or portable process-memory ceiling is not claimed by this
contract.

The adapter remains family-agnostic and has no fit_exponential method.
veridist.execution.fit_exponential_source(adapter) is the one-pass execution
orchestrator and fit_exponential_csv(path, *, schema, source_id, limits) is
its convenience wrapper.  Their frozen, slotted combined result has a
non-None fit if and only if ExecutionReport.outcome is CompleteOutcome; an
ExponentialFitFailure is therefore a complete execution, while a caught
EngineContractError becomes a FailedOutcome with fit=None.  Unexpected
exceptions propagate.

For a successful run, provenance is an exact known coverage [0, N), pass count
1/1, CSV adapter version 1, bounded-buffer observation, no spool, no
checkpoint, the exponential settings hash, and no-randomness policy.  It
records a consumed-byte SHA-256 only when it was actually calculated; otherwise
it uses the existing hash-unavailable redaction.  It records a best-effort
mutation status.  Open and header failures are preflight failures with unknown
extent.  Decode and row failures are delivery failures with processed prefix
and unknown final extent.  A detected post-read mutation is a finalization
failure.  Cancellation is explicitly outside this PR and API; it makes no
cancellation or backpressure guarantee beyond closing/releasing resources on
success, typed failure, early iterator close, and unexpected exception.

Failure precedence is frozen as follows: transport corruption (UTF-8 decode or
CSV parser failure) dominates semantic row validation; absent transport
corruption, the first semantic row failure is retained while the remainder is
scanned; a detected post-read source mutation dominates any semantic failure,
because source integrity invalidates the interpretation.  Cleanup never masks
a primary failure; an unexpected cleanup exception propagates only when no
primary failure exists.
