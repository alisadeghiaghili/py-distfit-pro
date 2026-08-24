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

## Current verified and unverified status

- On 2026-08-24, the branch-local coverage run executed 221 passing tests with
  one opt-in browser test skipped. The deterministic checker accepted all 22
  enumerated production files with 100% observed coverage: 1,578 statements and
  542 branches, with no accepted exceptions. This is structural and contract
  evidence, not a formal mutation score, calibration result or production-scale
  benchmark.
- ADR-0017's first callable cell is implemented on the development branch: a
  fixed-location, rate-only exponential MLE for exact and independently
  right-censored lifetimes. It returns a finite point estimate or a typed
  statistical failure; `inference=not_provided`. Reference, property,
  constructor-boundary, reducer, merge, underflow/overflow and report contracts
  pass. This is not a 0.1.0a1 release claim.
- The DS-01--DS-12 contract suite now exercises in-memory sources, bounded
  delivery, pass enforcement, retry/checkpoint boundaries, typed failures,
  outcome classification and closed redacted provenance. The implementation
  span is `d846fc8` through `1345666`.
- A branch-local PEP 517 sdist/wheel build and Twine check pass. A clean wheel
  environment outside the checkout passes dependency/import/version/`py.typed`
  checks and calls the 0.5-rate exponential fit plus a Persian RTL report.
- CI lane isolation and the Linux Python 3.11--3.14 quality, package and
  documentation jobs are configured in `e23cda5` through `89202c6` and pushed.
  Their remote execution result remains **UNVERIFIED**.
- Local Sphinx gettext, EN/FA/DE HTML and English linkcheck complete with
  warnings fatal. Exact real POT/catalog parity, translated rendered semantics,
  locale direction and canonical examples pass. Remote CI remains unverified.
- The opt-in browser contract passes locally with Edge 151 and exactly two
  nonempty Persian HTML screenshots; computed RTL/right alignment and
  LTR/isolate facts pass for success and failure reports. Exact Playwright
  1.62.0 bundled-Chromium execution remains required and unverified in remote CI.
  PDF, network-font and pixel-baseline rendering are not claimed.
- Eight temporary, targeted manual mutations were genuinely killed: input
  materialization, lost merge compensation, allowed underflow, constructor
  spoof acceptance, English Persian-catalog fallback, removed report key,
  overwritten machine facts and forced LTR Persian output. This is diagnostic
  evidence only. The formal mutation runner and score are **NOT IMPLEMENTED**.
- Retained DS pass/byte trace artifacts, actual source adapters, persistent
  checkpoint backend and production orchestrator are **NOT IMPLEMENTED**.
- ADR-0016 now provides an evidence-gated migration ledger, a dependency-free
  semantic/hash checker, AST/dynamic-import isolation checks and built-artifact
  payload inspection in the configured package job. LM-002 records the
  independent exponential rewrite and reviewed statistical evidence but remains
  `review_pending`; no legacy runtime code, compatibility surface or numerical
  oracle is used. LM-003 remains pending and does not claim legacy phrase reuse.
- Every exact numeric defect magnitude, benchmark, release status, competitor
  feature count, name/trademark availability, and SQL-export novelty remains
  unverified unless a current command or primary source is attached.
- Statistical calibration, bootstrap coverage, production-adapter streaming
  equivalence, production-scale memory bounds, PyPI publication and source-lock
  checking remain unverified. Reference agreement is scoped only to the current
  exponential point-estimation cell.

## Remaining v1 work and release evidence

| Area | Required v1 evidence |
| --- | --- |
| Core | Keep the current immutable exponential result/capability facts; extend the [capability matrix](capability-matrix.md) only with cited conformance evidence |
| First vertical | Current exponential point-estimation cell is callable; complete production adapter/pass-byte/scale evidence before calling the reliability + big-data vertical complete |
| Families | Any family beyond the first reliability vertical passes support/CDF/PPF/log-density/reference tests with cited specifications |
| Estimation | MLE applicability, convergence/restart diagnostics, and failures that remain visible |
| Scale | Promote the implemented DS-01--DS-12 contract fixtures into actual adapters and an orchestrator; retain measured pass/byte traces and production-scale memory evidence |
| Inference | Refit Monte-Carlo GoF and published calibration scope; no analytic p-value without a family/estimator source |
| Uncertainty | Bootstrap failure accounting and coverage evidence only for declared regular scenarios |
| Censoring | Explicit likelihood semantics and reference tests; unsupported family combinations fail loudly |
| Localization | Keep implemented EN/FA/DE parity, examples and local `-W`/linkcheck; require remote pinned-Chromium HTML RTL evidence and separately specify PDF before claiming it |
| Migration | Keep ADR-0016 ledger hashes current; require independent specs/RED tests per candidate and retain `distfit_pro` import/package isolation |
| Quality | Global >=95% line and branch; numerical `domain`/`statistics`/`families`/`engine` paths >=98% line and branch; every production file >=90% of both absent ADR; executable mutation >=80% for that scope |
| Competitive evidence | Every public `supported`/`not_supported` claim cell source-locked at 100% coverage; source-lock registry/checker currently NOT IMPLEMENTED |
| Release | Reproducible PyPI artifact, conda-forge publication readiness, valid `CITATION.cff`, Zenodo DOI release metadata, all CI tiers, changelog, security/license review, and public limitations/calibration report |

## Explicitly not done

Legacy API compatibility, an R implementation, 25+ families, KLL-based GoF,
Bag of Little Bootstraps, arbitrary SQL export, and a universal "best fit"
claim are not v1-complete merely because they appear in older plans.  They need
separate accepted ADRs and evidence. Telemetry is not a v1 feature: collection,
upload and auto-reporting are prohibited.
