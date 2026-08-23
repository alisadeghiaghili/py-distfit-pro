(veridist-exponential-right-censoring)=
# Exponential right-censoring tutorial

## Model and estimate

For exact-event count `r` and total time on test `tau`, this vertical uses
`r*log(rate)-rate*tau` and returns `rate=r/tau` only when both values permit a
finite estimate. It accepts exact lifetimes and independent right censoring.
Independent right censoring is an assumption visible in the result; it cannot
be established from the observations by this library.

## Failure cases and limits

An empty sample, no observed events, zero total time with an event, and numeric
overflow produce typed non-estimates. There is no confidence interval,
goodness-of-fit test, truncation, left or interval censoring, weights,
covariates, or free location parameter in this vertical.

## Streaming scope

The reducer retains fixed O(1) accumulator state: counts and a compensated
total-time sum. That algorithmic scope is not a claim of a production
out-of-core adapter, bounded ingestion, or a benchmarked large-data tier.
