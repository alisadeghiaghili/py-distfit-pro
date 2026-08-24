# veridist capability matrix

This matrix records callable behavior on the current development branch. It is
not a release announcement: the package version remains `0.0.0.dev0` and the
remote pull-request workflow has not yet verified this branch.

| Family | Estimator and parameterization | Accepted data semantics | Result | Inference | Scale boundary | Status/evidence |
| --- | --- | --- | --- | --- | --- | --- |
| Exponential, fixed `loc=0` | rate-only exponential MLE | exact and independent right-censoring; finite times `>= 0` | finite point estimate or typed statistical failure | `inference=not_provided` | fixed O(1) reducer state; no production adapter or out-of-core claim | Experimental callable cell; ADR-0017, `EXP-01`--`EXP-14`, NIST/R/SciPy references |

The exponential cell does not accept weights, covariates, truncation,
left/interval censoring, free location, model selection, goodness-of-fit,
confidence intervals, or another estimator. No other distribution family is a
callable v1 statistical cell yet. EN/FA/DE reports are presentation adapters for
the same locale-neutral result facts; they do not expand statistical support.

Traceability for `EXP-01`--`EXP-14` is split across
`python/tests/reference/test_exponential_mle_reference.py` and
`python/tests/contract/test_exponential_reducer_contract.py`. Report semantics
and browser rendering are exercised by
`python/tests/docs/test_exponential_report_i18n.py` and
`python/tests/browser/test_exponential_report_rtl.py`.

Contract fixtures exercise bounded delivery, replay, retry, checkpoint and
provenance semantics. They are not production data adapters, persistent storage,
an orchestrator, or measured external-memory execution.
