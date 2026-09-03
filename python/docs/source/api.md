(veridist-api)=
# CSV exponential API

`fit_exponential_csv(path, *, schema, source_id, limits)` is the public source
entry point. `CsvLifetimeSchema("time", "event_observed")` selects the only
accepted header; `PublicSourceId` is an opaque identifier safe for returned
provenance; and `CsvLifetimeLimits` supplies positive byte budgets. The result
is always an `ExponentialSourceFitResult`: a finite rate-only estimate, a typed
statistical non-estimate, or a typed execution failure.

Times must be finite and non-negative. Event token `1` denotes an exact event;
token `0` denotes independent right censoring. The CSV contract is deliberately
strict rather than a permissive spreadsheet reader: wrong encoding, headers,
tokens, fields, budgets, or source access produce typed failures.

The vertical fixes `location=0`, has no confidence interval or goodness-of-fit
claim, and supplies no weights, covariates, truncation, left/interval censoring,
free location, model selection, retry, checkpointing, or cancellation.

## Evaluated scalar primitives

`veridist.families.registry` exports immutable `FAMILY_REGISTRY` metadata and
`FamilyId` for five evaluated families. `veridist.statistics.log_density`
exports scalar `evaluate_log_density`; `veridist.statistics.log_likelihood`
exports `LogLikelihoodState` and `reduce_log_likelihood_chunks`. These are
separate, finite scalar contracts: they are not generic fitting, inference,
goodness-of-fit, ranking, array, or censoring APIs.
