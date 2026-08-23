(veridist-api)=
# Exponential MLE API

`fit_exponential(observations)` and `fit_exponential_chunks(chunks)` return a
finite rate-only estimate or a typed statistical non-estimate. Observations are
`ExactLifetime(time)` or `RightCensoredLifetime(time)` with finite,
non-negative times. A wrong observation object is a programmer-input error,
not a fitting failure.

`render_exponential_report(result, ReportLocale.EN|FA|DE)` renders safe pure
HTML for one explicit locale. It has no silent locale fallback. Every successful
result declares `inference=not_provided` and
`censoring_assumption=independent_right_censoring`.

The vertical fixes `location=0`, has no confidence interval or goodness-of-fit
claim, and does not claim a production out-of-core adapter or benchmarked
large-data tier.
