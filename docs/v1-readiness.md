# veridist v1 readiness ledger

This is a planning ledger, not a release checklist for `distfit_pro`. v1 uses
the `veridist` namespace. Legacy assets count only after an explicit,
evidence-backed disposition and validation under the v1 contracts.

## Evidence already obtained

- The reviewed legacy package demonstrated the value of explicit failure,
  canonical parameter counts, log-domain likelihoods, injected RNGs, and
  parameter-refit Monte-Carlo GoF.
- Independent reproduction described in the supplied review is consistent with
  the mechanism that a normal-only AD correction can miscalibrate another
  family and that a known-parameter KS null is invalid after fitting.
- Current public documentation establishes that SciPy offers refit
  Monte-Carlo GoF; fitdistrplus and SurPyval establish that censoring,
  truncation, bootstrap and broad estimator support are table stakes in parts
  of the market.  These are positioning inputs, not v1 implementation evidence.

## Static verified legacy defects

The supplied read-only review reports: discrete likelihood dispatch failures;
duplicate parameter aliases corrupting `k`; duplicated/miscalibrated AD and KS
paths; silent bootstrap substitution; global RNG reseeding; unsupported
censoring and method-name drift; weak test/CI gates; and documentation claims
that did not execute, including six syntax-invalid legacy examples. Treat these
as regression blockers/scenarios to specify, **not**
as a statement about uninspected current files.

## Runtime claims still UNVERIFIED

- On 2026-08-22, local generic discovery executed 156 tests. The deterministic
  coverage checker accepted all 14 enumerated production files with 100%
  observed coverage, against frozen denominators of 1,293 statements and 444
  branches and no accepted exceptions. This is structural and contract
  evidence, not statistical correctness or scale evidence.
- The DS-01--DS-12 contract suite now exercises in-memory sources, bounded
  delivery, pass enforcement, retry/checkpoint boundaries, typed failures,
  outcome classification and closed redacted provenance. The implementation
  span is `d846fc8` through `1345666`.
- On 2026-08-22, local PEP 517 builds produced an sdist and wheel. Inspection
  found the MIT license and package metadata in the wheel and the MIT license,
  manifest and all three package landing pages in the sdist. A clean wheel
  environment passed dependency, import, version, `py.typed` and `pip check`.
- CI lane isolation and the Linux Python 3.11--3.14 quality, package and
  documentation jobs are configured in `e23cda5` through `89202c6` and pushed.
  Their remote execution result remains **UNVERIFIED**.
- Structural documentation tests and EN/FA/DE catalog parity pass locally.
  A complete local Sphinx build remains **UNVERIFIED**: on 2026-08-22, the
  available Sphinx 8.2.3 environment failed before gettext because it did not
  contain the declared `myst_parser` dependency. The configured CI job installs
  the declared documentation extra, but its remote result is also unverified.
- The formal mutation runner and score, retained DS pass/byte trace artifact,
  browser screenshot of Persian rendering, actual source adapters, persistent
  checkpoint backend and production orchestrator are **NOT IMPLEMENTED**.
- Every exact numeric defect magnitude, benchmark, release status, competitor
  feature count, name/trademark availability, and SQL-export novelty remains
  unverified unless a current command or primary source is attached.
- Statistical streaming equivalence, calibration, bootstrap coverage,
  reference agreement, production-scale memory bounds, PyPI publication and
  source-lock checking remain unverified.

## Remaining v1 work and release evidence

| Area | Required v1 evidence |
| --- | --- |
| Core | Immutable result schema, declared capability matrix, canonical parameter map, and conformance tests |
| First vertical | Reliability + censoring + big-data: selected reliability families, cited censored likelihood cells, DS-01--DS-12 evidence and visible failure diagnostics |
| Families | Any family beyond the first reliability vertical passes support/CDF/PPF/log-density/reference tests with cited specifications |
| Estimation | MLE applicability, convergence/restart diagnostics, and failures that remain visible |
| Scale | Promote the implemented DS-01--DS-12 contract fixtures into actual adapters and an orchestrator; retain measured pass/byte traces and production-scale memory evidence |
| Inference | Refit Monte-Carlo GoF and published calibration scope; no analytic p-value without a family/estimator source |
| Uncertainty | Bootstrap failure accounting and coverage evidence only for declared regular scenarios |
| Censoring | Explicit likelihood semantics and reference tests; unsupported family combinations fail loudly |
| Localization | Keep implemented EN/FA/DE parity and canonical example checks; verify remote `-W` builds/linkcheck and add browser-rendered RTL QA |
| Quality | Global >=95% line and branch; numerical `domain`/`statistics`/`families`/`engine` paths >=98% line and branch; every production file >=90% of both absent ADR; executable mutation >=80% for that scope |
| Competitive evidence | Every public `supported`/`not_supported` claim cell source-locked at 100% coverage; source-lock registry/checker currently NOT IMPLEMENTED |
| Release | Reproducible PyPI artifact, conda-forge publication readiness, valid `CITATION.cff`, Zenodo DOI release metadata, all CI tiers, changelog, security/license review, and public limitations/calibration report |

## Explicitly not done

Legacy API compatibility, an R implementation, 25+ families, KLL-based GoF,
Bag of Little Bootstraps, arbitrary SQL export, and a universal "best fit"
claim are not v1-complete merely because they appear in older plans.  They need
separate accepted ADRs and evidence. Telemetry is not a v1 feature: collection,
upload and auto-reporting are prohibited.
