# veridist v1 conventions

Status: **mixed authority**. The DataSource planner/scale rules are binding
under Accepted ADR-0005; quality gates under Accepted ADR-0006; and
localization/documentation rules under Accepted ADR-0011 and ADR-0013.
Correctness/API, floating-point reproducibility and competitive-evidence rules
remain Draft pending their Proposed ADRs. This document is never authority to
claim an unimplemented capability publicly.

## Correctness and API -- Draft pending ADR-0002/0003/0004

- A result is immutable and contains parameters, estimator, diagnostics,
  capability metadata, provenance, and any approximation/error contract.
- A family declares its support, canonical free parameters, estimator and data
  capability matrix.  Unsupported combinations raise a named capability error;
  they never silently fall back to another estimator or to uncensored data.
- Likelihoods use log space.  Tail probabilities use survival/log-survival
  functions.  Parameter count comes from declared free parameters, never from a
  mapping length.
- Randomness is supplied as an explicit `numpy.random.Generator`; library code
  never reseeds or consumes a global random state.

## Data-source and scale contract -- Binding under ADR-0005

- A `DataSource` declares `replayability` as `single_pass`, `replayable`, or
  `checkpoint_replayable`, a stable source ID, schema version, and a privacy
  redaction policy. A planner rejects a multipass operation on `single_pass`;
  it may spool only through an explicit user opt-in with declared disk budget,
  retention and cleanup policy.
- Chunks have stable chunk IDs and row-offset ranges. Empty chunks are valid;
  duplicate, lost or out-of-order chunks are errors unless the declared source
  semantics and idempotency key explicitly permit them. A partial result is
  never returned without `complete=False`, missing ranges and a typed cause.
- A source is materialized only by an explicitly in-memory API; no streaming
  path may call `np.asarray` on the whole source. Adapters for CSV, Parquet,
  Arrow, pandas, Polars, Dask and database readers declare whether they are
  replayable and what ordering/offset guarantee they provide.
- Each declared streaming capability records: required accumulator state,
  maximum state complexity, `max_passes`, chunk policy, `chunk_bytes`,
  `max_inflight_bytes`, backpressure/cancellation behaviour, and whether the
  result is exact in real arithmetic or approximate. A pass-enforcer counts
  source iterations and fails before exceeding `max_passes`. A frequency table
  is `O(unique_values)`, not constant memory.
- **Draft pending ADR-0004:** reductions have a deterministic canonical
  order/tree in reproducible mode.
  Floating-point results are compared with documented numerical tolerances;
  bit-identical output is promised only for an explicitly pinned platform,
  input partition, and reduction tree.
- Retry is legal only for an idempotent chunk/update checkpoint. Cancellation,
  retry exhaustion, checksum mismatch, source mutation and missing chunks use
  distinct failure codes. Checkpoints record accumulator schema/version and
  source range; they are never silently reused after incompatible changes.
- A fit records provenance schema version, source ID/hash (or explicit
  redaction), row count/ranges, non-finite handling, chunk policy, pass count,
  library/adapter versions, estimator settings, RNG policy, approximation and
  checkpoint metadata. Provenance must reproduce the analysis without exposing
  raw data.

## Quality gates -- Binding under ADR-0006

- Global release coverage is **>=95% line and >=95% branch**. The critical
  statistical core is only the numerical production paths in
  `domain`, `statistics`, `families`, and `engine`; each requires >=98% line
  **and** branch coverage. Every production file requires >=90% line and
  branch coverage unless an accepted ADR names owner, reason, expiry and
  compensating evidence.
- `pragma: no cover`, generated-code exclusions, and denominator-changing test
  configuration are prohibited unless an accepted ADR lists the exact path and
  rationale. Tests that merely import/execute code do not satisfy behaviour
  evidence.
- Mutation is an executable release gate for that same critical statistical
  core: eligible mutants killed divided by eligible killed plus survived
  mutants is >=80%. A timeout, harness error, unclassified mutant, missing
  module report or score below threshold fails the gate. A survivor remains in
  the denominator unless an accepted ADR classifies an exact-path, expiring
  exclusion.

## Documentation and localization -- Binding under ADR-0011/0013

- The statistical core emits structured diagnostic codes and data only.  It
  contains no translated strings or logging.
- English is the canonical documentation source; Persian and German maintain
  translation parity through a manifest.  Stable anchors and API identifiers do
  not change by locale.  Missing locale keys fail CI; fallback behaviour is
  explicit and visible to callers.
- Persian documentation is rendered RTL and visually smoke-tested.  Examples
  are executable from one canonical source or parity-tested across locales.
- Every Persian HTML/PDF report must declare `lang="fa"` and `dir="rtl"` at
  its document boundary. Code, formulae, URLs, API names and other Latin runs
  require explicit LTR isolation; direction alone is not adequate mixed-script
  layout evidence. A browser screenshot review of tables, code and formulae is
  required before a rendered-RTL claim. The present structural checks are not
  that browser evidence.

## Legacy migration boundary -- Binding under ADR-0016

- `veridist` is the sole runtime namespace and release artifact. Source code,
  build metadata, wheel and sdist must not import, depend on, dynamically load,
  or ship `distfit_pro`; legacy results are not an oracle or fallback.
- Reuse starts with evidence, not copying. Every considered legacy component
  has a Draft 2020-12 ledger entry with source commit/path/blob/SHA-256,
  license, independent evidence, reviewers, limits and exactly one
  disposition: `modify_port`, `rewrite`, or `archive`. A changed `origin/main`
  source hash makes the evidence stale.
- A `modify_port` entry is only a candidate; it needs an independent v1
  specification and RED tests before implementation. Translated legacy content
  is reviewed-content input, never behavioural authority. Exponential is a
  rewrite.

## Git discipline -- Binding repository process

- Use short, descriptive kebab-case feature-branch names such as
  `v1-foundation` or `data-source-contracts`; no branch prefix is mandatory.
  Keep commits atomic and use Conventional Commit style with a CI prefix where
  applicable.
- On a feature branch, keep documentation, test-red (the intentionally failing
  specification test), and implementation changes in separate commits. The PR
  head must nevertheless be green; do not open a PR with a deliberately
  failing test as its final state.
- Before committing, inspect status and diff; run the relevant tests, lint,
  type checks, secret scan and generated-artifact check. Record unavailable
  checks honestly rather than claiming they passed.
- Do not mix unrelated legacy changes into a feature branch. Open a PR only
  for a coherent milestone of roughly one week of human work; smaller changes
  may stay as reviewed commits, larger work must be split.
- Each PR supplies a proper Conventional/CI-prefixed title and message, scope,
  linked decision/capability cells, tests and evidence run, calibration/scale
  impact, documentation/i18n impact, migration impact, and explicit known
  limits. A green check alone is not sufficient evidence.

## Evidence language -- Draft pending ADR-0014

Use **verified** only with a linked test, benchmark, calibration table, or
primary reference.  Say **unverified** for a hypothesis, legacy observation not
re-run in this repository, or a claim inherited from documentation.  Never use
"exact", "calibrated", "constant memory", or "better" without its stated
scope and evidence.
