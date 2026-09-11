# veridist v1 roadmap

This is a 90-day delivery plan.  It intentionally narrows scope so that v1 can
prove a few difficult capabilities rather than advertise many unvalidated ones.

**Status at 0.9.1 (2026-09-11):** foundations, the three-family reliability
vertical, strict CSV execution, local SQLite resume, scalar family operations,
and the declared exponential inference cell are implemented. The final 1.0
stage is evidence and release work: retained calibration and portable scale
reports, release metadata, multilingual parity for the expanded scope, and the
fail-closed v1 release gates.

## Days 1-14: foundations

Create Python-only greenfield package boundaries, spec/provenance schema,
family-estimator capability matrix, immutable result/diagnostic objects,
reproducible RNG contract, TDD harness, and English/Persian/German document
structure with a translation-parity manifest.  Freeze legacy as an audited
reference; do not port its statistical core. Establish DataSource replayability,
chunk-ID/offset, max-pass, bounded-inflight, failure/checkpoint and redacted
provenance contracts. Start scale-contract, docs/i18n and example CI gates now.
The DS-01--DS-12 harness, strict CSV adapter, persistent local SQLite
checkpoints, orchestration, and retained pass/byte evidence now exist. No
portable process-memory or general production-scale claim follows. The
competitive claim-cell registry/source-lock design remains required; no
competitive landscape is publishable
until its 100% required lock coverage checker exists and passes.

## Days 15-35: first domain vertical -- reliability, censoring and big data

Release status on 2026-09-11: fixed-location Exponential, Weibull-minimum, and
Lognormal MLE cells for exact and independently right-censored lifetimes are
implemented with typed failures. The Exponential path has strict CSV and local
SQLite resume. Broader adapters and portable RSS evidence remain future work.

Implement and cite the narrow reliability set needed for the vertical
(Exponential, Weibull and Lognormal unless a capability review narrows it
further). Deliver right-censored likelihood cells, visible failure diagnostics,
declared accumulator state/pass budgets, bounded chunked data paths, reference
goldens and conformance/property tests. Publish no blanket bit-identical claim
and do not add broad general-family catalogue work before this vertical has
scale and reference evidence.

## Days 36-52: hard censoring and data semantics

Complete the selected right-censored Exponential/Weibull cells only after
reference agreement. Specify left/interval censoring, truncation, frequency
versus analytic weights and unsupported combinations before implementing them.
Exercise replayability, explicit spool rejection/opt-in, checkpoint/retry and
bounded-inflight data semantics on this vertical. Keep an explicit capability
error where the matrix says no.

## Days 53-68: honest inference

Implement refit Monte-Carlo KS/AD/CvM for the declared uncensored capability
matrix; record Monte-Carlo uncertainty, fit failures, seed/provenance and
calibration scope.  Add AIC/BIC and bootstrap selection stability with a
decision-threshold-based `NONE_ADEQUATE` report.  For large n, lead reports
with effect sizes and decision thresholds while retaining p-values as labelled
evidence.

## Days 69-80: deepen continuous scale and multilingual evidence

Run reproducible chunked/parallel scale experiments, checkpoint/retry/cancel
audits, state/pass/inflight-byte evidence and nightly statistical coverage.
Complete EN/FA/DE narratives, locale parity,
RTL render QA, localized reports, link checking and executable example tests.
Run the DataSource matrix continuously rather than treating it as a benchmark
week; retain observed byte/pass traces. Finalize a source-lock manifest only
for externally publishable competitive claims; the source-lock tool remains
**NOT IMPLEMENTED** until built and validated.

## Days 81-90: release candidate

Run the release matrix, coverage/mutation gates, build artifacts, public
calibration and benchmark reports, limitations/capability matrices, and an
external adversarial review.  Release only if every advertised cell has its
evidence; otherwise publish an alpha with the unsupported cells visibly absent.
PyPI artifact reproducibility, conda-forge publication readiness, a valid
`CITATION.cff`, and Zenodo DOI release metadata are mandatory gates.

## Delivery hygiene throughout

Use short descriptive kebab-case feature branches such as `v1-foundation` or
`data-source-contracts`; no branch prefix is mandatory. Keep atomic
Conventional/CI-prefixed commits and separate docs, test-red and implementation
commits on each feature branch; the PR's final head must be green. Before a
commit inspect status/diff and run available tests, lint, type, secret and
generated-artifact checks. A PR is for a coherent roughly week-sized milestone
only, never unrelated legacy cleanup, and carries the evidence/checklist
required by `conventions.md`.

## Deferred deliberately

R/CRAN, L-moment sketches, BLB, arbitrary censoring/truncation, custom family
DSL, 25+ families, warehouse SQL export, GUI/plotting breadth, and universal
model-selection claims are post-v1 decisions.  Three-language documentation is
not deferred: it is a v1 release gate.

## Beyond v1: breadth without wrapper theatre

ADR-0012 sequences the long-term platform: evidence baseline; statistical core;
censoring/reliability; hydrology/extremes/L-moments; mixtures/zero-inflation;
Bayesian integration; enterprise/GUI/export; then R cross-language work. Each
wave needs an owner, capability matrix, references, calibration/reference tests
and scale semantics. Adoption comes from migration guides, EN/FA/DE
discoverability, reproducible public evidence and open release governance--not
from unsourced vanity metrics.
