# Veridist changelog

This changelog covers only the nested `veridist` package. The repository-root
legacy changelog describes the frozen `distfit_pro` history and is not a
Veridist release record.

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
