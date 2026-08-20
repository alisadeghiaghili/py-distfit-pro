# veridist v1 test plan

Every production change starts with a failing test that fails for the intended
reason.  Expected values come from a derivation, primary source, or independent
versioned oracle--never from legacy implementation behaviour.

## Taxonomy

| Suite | Purpose | Default CI |
| --- | --- | --- |
| `unit` | pure math, error paths, diagnostics | required |
| `property` | support, monotonicity, invariance, reduction laws | required |
| `conformance` | every registered family | required |
| `reference` | committed data/golden files with provenance | required |
| `statistical` | calibration, recovery, interval coverage, selection | nightly/release |
| `scale` | streaming, pass count, state/memory integration | required from first slice; extended nightly |
| `docs` | Sphinx/MyST links, examples, localized rendering | required from first slice |

## Core and numerical tests

- Conformance covers `log_density`, support, CDF/SF/PPF identities, discrete
  mass, declared free-parameter count, undefined moments, and sample support.
- Test integer frequency weights against replicated observations within the
  documented numerical tolerance.  Test zero censoring against ordinary MLE.
- Test optimizer non-convergence, invalid support, all-censored data and failed
  restarts: diagnostics/failure counts must be returned or raised, never
  substituted.
- Tolerances state their source: high-precision oracle error, published
  precision, convergence tolerance, or Monte-Carlo error.  No tuned tolerance.

## Out-of-core tests

For each advertised tier, test empty chunks, one chunk, ragged chunks, changing
boundaries, stable chunk IDs/row offsets, deterministic shuffled schedule,
non-finite input, duplicate/lost/out-of-order chunks, source mutation and
restartable/non-restartable errors. Test planning rejects multipass work on
`single_pass`; test explicit spool opt-in separately. Assert state bound,
`chunk_bytes`, `max_inflight_bytes`, backpressure, cancellation, no retained
chunk, no full-source materialization and declared `max_passes` with an
instrumented pass-enforcer. Retry tests prove idempotent checkpoint recovery or
return a labelled incomplete/failure result.

Compare to an in-memory canonical reduction under the floating-point
reproducibility contract: tolerance-based across ordinary schedules; bitwise
only under the pinned reproducible schedule.  Integration benchmarks measure
peak memory and pass count outside ordinary PR CI; no portable RSS ratio or
parallel speedup is a universal unit-test assertion.

## DataSource contract-to-test matrix

This is the binding test design for the DataSource rules in
[conventions.md](conventions.md) and ADR-0005. IDs are stable: an ADR or
implementation may add tests, but may not weaken or rename an ID without a
superseding ADR. The harness and fixtures below are **NOT IMPLEMENTED**; no
current command, adapter or production result is implied.

| ID | Contract | Planned test fixture | Observable assertion |
| --- | --- | --- | --- |
| DS-01 | Identity, schema and redaction | metadata-contract fixture | Source ID is stable; schema/provenance versions are non-empty and supported; either a source hash is present or an explicit redaction reason is present. |
| DS-02 | Replayability plan | one `single_pass`, one `replayable`, one `checkpoint_replayable` fake source | Planner accepts within-budget plans only; it rejects a second pass on `single_pass` before iteration; checkpoint replay is accepted only with compatible checkpoint metadata. |
| DS-03 | Explicit spooling | single-pass fixture plus opt-in/off plans | A multipass plan with spooling off returns `SPOOL_REQUIRED`/capability error without reading; opt-in records disk budget, retention and cleanup outcome in provenance. |
| DS-04 | Adapter ordering, offsets and boundaries | CSV, Parquet, Arrow, pandas, Polars, Dask and database adapter contract fakes | Each adapter declares replayability and ordering; emitted chunk IDs are stable, offsets are contiguous and non-overlapping, and boundary changes preserve row identity. |
| DS-05 | Empty, duplicate, lost and out-of-order delivery | adversarial chunk scheduler | Empty chunks complete normally; undeclared duplicate/lost/out-of-order delivery returns its distinct typed code and neither double-counts nor silently reorders input. |
| DS-06 | Byte bounds, inflight bound, backpressure and cancellation | instrumented producer/consumer with oversized and blocked chunks | Observed buffered bytes never exceed `max_inflight_bytes`; each chunk honours `chunk_bytes`; producer blocks under backpressure; cancellation stops new reads, releases resources and reports a typed cancelled outcome. |
| DS-07 | Pass budget | counting source plus pass-enforcer | Iteration count never exceeds declared `max_passes`; attempt `max_passes + 1` fails before the next source read and provenance records the actual pass count. |
| DS-08 | Retry, idempotency, checksum and source mutation | fault-injecting checkpointed source | Retry only replays an idempotency-keyed update; checksum mismatch and source mutation receive distinct typed errors; no accumulator update is applied twice. |
| DS-09 | Checkpoint resume | interrupted run with compatible and incompatible accumulator/source schema | Compatible resume equals the declared canonical reduction within its contract; incompatible schema/version/range rejects without reuse and records the reason. |
| DS-10 | Typed failure surface | parameterized invalid-delivery/source faults | Every documented failure returns/raises the declared stable error code; no generic string-only failure is accepted. |
| DS-11 | Partial-result labelling | failure after a known offset range | A returned partial result has `complete=False`, exact missing ranges and a typed cause; a complete result cannot contain missing ranges or a partial cause. |
| DS-12 | Full provenance | completed, cancelled, retried and redacted runs | Provenance validates schema/version, source ID/hash-or-redaction, row ranges/count, chunk/pass/budget settings, adapter/library versions, estimator/RNG/approximation and checkpoint metadata; raw input is absent under redaction. |

The first vertical slice must implement this matrix as executable tests before
advertising a DataSource adapter or a scale tier. The matrix harness is also a
release artifact: it must emit fixture version, observed pass/byte traces and
the typed-result/provenance record needed to audit each assertion.

## Statistical evidence

For each supported family/estimator/test scenario, simulate a declared grid of
parameters and sample sizes.  Fit every simulated sample again before GoF.
With nominal alpha `a` and `R` replicates, record rejection rate and binomial
standard error `sqrt(a*(1-a)/R)`; at `a=.05, R=2000`, this is .00487 and a
three-standard-error descriptive band is .05 +/- .0146.  Publish confidence
intervals and the full grid; this is evidence for that grid, not proof for all
families.

Bootstrap coverage tests declare the data-generating model, parameter regime,
sample size, interval type and failures.  At 95% nominal coverage, report its
binomial uncertainty; do not promise BCa/CRB behaviour in boundary or
nonregular models.  Selection tests measure algorithmic stability and recovery
over a stated candidate set; they never interpret resampling frequency as the
probability that a model is true.

## Localization and documentation tests

- Verify English/Persian/German key parity, permitted explicit fallback rules,
  placeholder/format compatibility, Unicode normalization, and locale-aware
  numeral/date formatting.
- Render Persian pages in RTL and perform a visual smoke test for direction,
  code blocks, formulae, tables and mixed Latin identifiers.
- Build every locale with warnings treated as errors, run link checking, and
  execute every public code example from a canonical source or parity harness.

## CI tiers and checker contract

PR CI will run core, DS-01--DS-12 scale-contract, docs/i18n and example tests
from the first vertical slice on Python 3.10, 3.11, 3.12 and 3.13 on Linux;
release CI adds supported Windows and macOS. Pin dependencies and record
Python, NumPy, SciPy and locale/rendering versions. CI configuration, commands,
coverage parser, mutation runner and documentation toolchain are all **Planned
/ NOT IMPLEMENTED**. No executable command is claimed here.

The planned checker contract is exact: its coverage input must identify every
production file and report separate line and branch percentages; it fails when
global line <95%, global branch <95%, a numerical production path in
`domain`, `statistics`, `families` or `engine` has line or branch <98%, or any
production file has line or branch <90% without an accepted exact-path ADR
exception. It must reject unlisted files and denominator changes. Its mutation
input must enumerate every eligible mutant in those same critical numerical
paths and fail unless `killed / (killed + survived) >= 0.80`; timeout, harness
error, unclassified result, excluded mutant without an accepted exact-path
expiry, or a missing module report fails the run. Generated code, `pragma: no
cover`, import-only execution, test-only helpers, and configuration exclusions
cannot reduce a denominator unless an accepted ADR supplies an exact path,
rationale, owner and expiry.

The planned docs checker builds EN, FA and DE with warnings fatal, checks links,
executes canonical examples/parity tests and captures an RTL smoke artifact.
The planned scale checker executes DS-01--DS-12 and retains pass/byte traces.
Calibration, scale, docs/i18n/example and rendered-RTL artifacts are continuous
gates, not skipped placeholders or late-release work.
