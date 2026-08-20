# veridist v1 readiness ledger

This is a planning ledger, not a release checklist for `distfit_pro`.  v1 is a
greenfield product; nothing in the legacy package counts as v1 complete.

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

- Current repository test, coverage, mutation, packaging, PyPI, license, and
  CI state have not been re-executed for this v1 work.
- Every exact numeric defect magnitude, benchmark, release status, competitor
  feature count, name/trademark availability, and SQL-export novelty remains
  unverified unless a current command or primary source is attached.
- No streaming equivalence, calibration, bootstrap coverage, language render,
  reference agreement, or memory/pass budget has been run for v1 because v1
  production code does not yet exist.
- The Sphinx/MyST/gettext documentation toolchain, locale builds, doctest,
  linkcheck, RTL screenshots, source adapters, DataSource DS-01--DS-12 test
  harness, pass-enforcer and source-lock tool are **NOT IMPLEMENTED**. The
  checker contracts in the test plan and evidence policy are planned, not
  existing commands.

## Greenfield v1 TODO and release evidence

| Area | Required v1 evidence |
| --- | --- |
| Core | Immutable result schema, declared capability matrix, canonical parameter map, and conformance tests |
| First vertical | Reliability + censoring + big-data: selected reliability families, cited censored likelihood cells, DS-01--DS-12 evidence and visible failure diagnostics |
| Families | Any family beyond the first reliability vertical passes support/CDF/PPF/log-density/reference tests with cited specifications |
| Estimation | MLE applicability, convergence/restart diagnostics, and failures that remain visible |
| Scale | DataSource replayability, explicit spool, chunk IDs/offsets, pass-enforcer, retry/checkpoint and bounded-memory evidence from first slice; executable DS-01--DS-12 harness |
| Inference | Refit Monte-Carlo GoF and published calibration scope; no analytic p-value without a family/estimator source |
| Uncertainty | Bootstrap failure accounting and coverage evidence only for declared regular scenarios |
| Censoring | Explicit likelihood semantics and reference tests; unsupported family combinations fail loudly |
| Localization | ADR-0013 toolchain, EN/FA/DE parity, RTL QA, `-W` builds, linkcheck and canonical executable examples |
| Quality | Global >=95% line and branch; numerical `domain`/`statistics`/`families`/`engine` paths >=98% line and branch; every production file >=90% of both absent ADR; executable mutation >=80% for that scope |
| Competitive evidence | Every public `supported`/`not_supported` claim cell source-locked at 100% coverage; source-lock registry/checker currently NOT IMPLEMENTED |
| Release | Reproducible PyPI artifact, conda-forge publication readiness, valid `CITATION.cff`, Zenodo DOI release metadata, all CI tiers, changelog, security/license review, and public limitations/calibration report |

## Explicitly not done

Legacy API compatibility, an R implementation, 25+ families, KLL-based GoF,
Bag of Little Bootstraps, arbitrary SQL export, and a universal "best fit"
claim are not v1-complete merely because they appear in older plans.  They need
separate accepted ADRs and evidence. Telemetry is not a v1 feature: collection,
upload and auto-reporting are prohibited.
