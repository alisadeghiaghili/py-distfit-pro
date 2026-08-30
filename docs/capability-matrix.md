# veridist capability matrix

## Exact-state streaming log likelihood

Five validated scalar-density families support an exact binary64-output
streaming reducer. The retained 10k/100k/1m generated-stream matrix at
`python/evidence/scale-log-likelihood-v1.json` is checker-validated only for
one-pass traversal and bounded reducer state (2162 bits under the uint64 count
cap). It is not a process-memory, throughput, out-of-core, fitting, or
cross-platform scalar-equivalence claim.

This matrix records callable behavior on the current development branch. It is
not a release announcement: the package version remains `0.0.0.dev0` and the
remote pull-request workflow has not yet verified this branch.

| Family | Estimator and parameterization | Accepted data semantics | Result | Inference | Scale boundary | Status/evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Exponential, fixed `loc=0` | rate-only exponential MLE | strict UTF-8 CSV with exact `time,event_observed` header; event `1` is exact; `0` is independent right-censoring; finite times `>= 0` | finite point estimate, typed statistical non-estimate, or typed failed execution | `inference=not_provided` | one CSV iterator and one pass; retained logical payload bound per 32/64/128KiB budget; no RSS ceiling | Experimental callable cell; ADR-0017/0018, `EXP-01`--`EXP-14`, `CSV-01`--`CSV-06`, `SCALE-CSV-EXP-01` |
| Normal, Gamma, Weibull-min, Lognormal, Gumbel-right | exact scalar log-density only; canonical two-parameter forms in ADR-0019 | one finite built-in scalar; Gamma/Weibull-min/Lognormal require strict `x>0`; no aliases | finite log-density or closed typed evaluation failure | none | scalar only; no array, fitting, censoring, ranking, or streaming claim | Experimental internal module; ADR-0019 addendum, independent 100-digit oracle and metamorphic contracts |

This callable cell accepts exact and independent right-censoring observations;
the censoring independence is a declared modeling assumption, not a property
verified from the CSV values.

The estimator retains fixed O(1) reducer state; that algorithmic fact is
separate from the measured CSV retained-payload boundary and is not an RSS or
generic big-data claim.

The exponential cell does not accept weights, covariates, truncation,
left/interval censoring, free location, model selection, goodness-of-fit,
confidence intervals, or another estimator. No other distribution family is a
callable v1 statistical cell yet. EN/FA/DE reports are presentation adapters for
the same locale-neutral result facts; they do not expand statistical support.

The five scalar log-density evaluators are intentionally not package-top-level
exports and are not a generic distribution-fitting API.  They have no fitting,
CDF/SF/PPF, inference, reports, weights, or big-data claim.  Their registry
availability is all-or-nothing and checked against the exact dispatch table at
module import.

Traceability for `EXP-01`--`EXP-14` is split across
`python/tests/reference/test_exponential_mle_reference.py` and
`python/tests/contract/test_exponential_reducer_contract.py`. Report semantics
and browser rendering are exercised by
`python/tests/docs/test_exponential_report_i18n.py` and
`python/tests/browser/test_exponential_report_rtl.py`.

The public package surface exposes only `fit_exponential_csv`,
`CsvLifetimeSchema`, `CsvLifetimeLimits`, `PublicSourceId`, and its closed
result. It has no legacy runtime import. The retained scale artifact is limited
to the strict CSV/exponential matrix; it does not establish generic big-data,
portable RSS, throughput, retry/checkpoint, cancellation, or other-adapter
support. The formal mutation runner remains **NOT IMPLEMENTED**.
