# Veridist changelog

This changelog covers only the nested `veridist` package. The repository-root
legacy changelog describes the frozen `distfit_pro` history and is not a
Veridist release record.

## [0.9.1] - 2026-09-11

### Changed

- Checkpointed CSV execution commits bounded adapter chunks instead of opening
  a SQLite transaction for every row. Cancellation persists the completed
  prefix for nonzero-cursor resume.
- Candidate-bound 1.0 evidence collection verifies complete, resumed, and
  cancelled 10k, 100k, and 1m-row runs on Linux, macOS, and Windows and rejects
  digest drift.
- Citation File Format, Zenodo, and conda-forge release metadata is
  version-aligned and validated in CI.

### Quality

- The 95% global line and branch coverage contract remains enforced; the
  execution-module denominator is frozen at 179 statements and 70 branches.
- Competitive source-lock validation remains outside the package release gate
  while ADR-0014 is Proposed; competitive drafts remain non-publishable.

## [0.9.0] - 2026-09-11

### Added

- Durable local SQLite checkpoints with generation-based compare-and-swap,
  corruption detection, source-revision checks, and resumable exponential CSV
  and canonical-chunk execution.
- Fixed-location Weibull-minimum and lognormal MLE cells for exact and
  independently right-censored lifetimes, including typed failures and
  frequency-weight contracts.
- Scalar CDF, survival, quantile, and caller-owned RNG sampling operations for
  the declared continuous-family registry.
- AIC/BIC, adequacy-gated model selection, and refit Monte Carlo KS, AD, and
  CvM goodness-of-fit for the uncensored exponential cell. Monte Carlo output
  reports requested, successful, and failed refits plus sampling uncertainty.

### Quality

- Python 3.11 through 3.14 CI, package, documentation, RTL browser, coverage,
  and fail-closed mutation gates pass on the release line.
- The inference module is registered with 129 statements and 46 branches and
  is fully exercised by its contract and calibration tests.

### Known limits

See [Known limits](KNOWN_LIMITS.md). Inference remains limited to uncensored
exponential samples, checkpoints remain local SQLite, operations are scalar,
and no general performance, RSS, or universal best-fit claim is made.

## [0.5.0] - 2026-09-10

### Available in the candidate scope

- A strict UTF-8 CSV adapter for fixed-location, rate-only exponential MLE
  with exact and independently right-censored lifetimes.
- Scalar log-density evaluation for normal, gamma, Weibull-minimum,
  lognormal, and right-Gumbel families.
- Exact-state streaming log-likelihood reduction through the public
  `IterableDataSource` contract.
- Bounded delivery that charges queued payload and active consumer leases.
- English, Persian, and German package documentation with Persian RTL checks.
- Candidate-bound coverage, mutation, scale-evidence, and release-validation
  workflows.

### Release status

The package version is `0.5.0`. Its candidate-specific ADR-0020 gates passed
on the exact release candidate; retained CI and scale artifacts bind to that
candidate SHA.

### Known limits

See [Known limits](KNOWN_LIMITS.md). These limits are part of the release
contract and constrain every public 0.5 claim.
