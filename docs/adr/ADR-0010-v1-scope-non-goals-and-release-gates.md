# ADR-0010: v1 scope, non-goals and release gates

Status: Accepted

Decision evidence: direct Ali Sadeghi decision, 2026-08-20 -- Python-first
with R deferred; first vertical reliability + censoring + big-data; no
telemetry in v1; and PyPI, conda-forge, `CITATION.cff` and Zenodo DOI are v1
release gates.

## Decision

v1 prioritizes a reliability + censoring + big-data first vertical: a small
cited reliability-family matrix, stable chunked fitting, selected censored MLE,
refit Monte-Carlo GoF, honest selection reports, and complete
English/Persian/German documentation. It does not compete on catalogue size,
Bayesian modelling, multivariate/copula fitting, regression, GUI breadth,
arbitrary SQL, R parity, telemetry, or unvalidated approximate inference.

## Release gates

All advertised cells must have capability docs, conformance/reference tests,
diagnostics, provenance, localization coverage, coverage/mutation gates and
appropriate calibration/scale evidence.  The release report publishes known
limits and failed/unsupported scenarios. A v1 release also requires a
reproducible PyPI artifact, conda-forge publication readiness, a valid
`CITATION.cff`, and Zenodo DOI release metadata. These are release gates, not
aspirational distribution channels.

From the first vertical slice, CI continuously runs scale-contract,
documentation, i18n and executable-example gates; they are not a late
roadmap phase. ADR-0013 is Accepted, but the documentation toolchain remains
NOT IMPLEMENTED until it is configured.

## Relationship to platform breadth and adoption

ADR-0012 governs post-v1 parity waves. Integration badges, package listings,
benchmarks, community examples and adoption claims require reproducible
evidence. No reputation objective relaxes this ADR's correctness, scale or
multilingual release gates. ADR-0014 additionally blocks publication of any
competitive matrix until every `supported`/`not_supported` claim cell has its
required source lock; the registry/checker is currently NOT IMPLEMENTED.
