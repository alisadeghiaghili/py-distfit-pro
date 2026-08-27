(veridist-exponential-right-censoring)=
# Exponential right-censoring tutorial

## Model and estimate

For exact-event count `r` and total time on test `tau`, this vertical uses
`r*log(rate)-rate*tau` and returns `rate=r/tau` only when both values permit a
finite estimate. It accepts exact lifetimes and independent right censoring.
Independent right censoring is an assumption visible in the result; it cannot
be established from the observations by this library.

## Strict CSV input

The public adapter accepts UTF-8 CSV with exactly `time,event_observed` in that
order. Each data row is one finite non-negative time and token `1` (event) or
`0` (right censored). A caller supplies a non-secret `PublicSourceId` instead
of relying on a local path in returned facts. The adapter never guesses columns,
headers, encodings, delimiters, missing values, or censoring conventions.

The schema declaration below is executable Python. It names the only accepted
header pair and intentionally keeps machine-readable identifiers left to right.

```python
schema = CsvLifetimeSchema("time", "event_observed")
```

| CSV field | Meaning |
| --- | --- |
| `time` | finite non-negative lifetime |
| `event_observed` | `1` for an exact event; `0` for independent right censoring |

The displayed equation fixes the model parameterization used throughout this
vertical.

```{math}
\widehat{rate} = r / tau
```

## Failure cases and limits

An empty sample, no observed events, zero total time with an event, and numeric
overflow produce typed non-estimates. There is no confidence interval,
goodness-of-fit test, truncation, left or interval censoring, weights,
covariates, or free location parameter in this vertical.

## One-pass and scale scope

The CSV adapter consumes one iterator pass. Its declared chunk byte budget is a
logical bound on retained payload, not a portable process-memory or RSS limit.
`SCALE-CSV-EXP-01` retains a checked 10k/100k/1m by 32KiB/64KiB/128KiB matrix
for this strict adapter and estimator only. It demonstrates bounded internal
payload for that matrix; it does not establish throughput, generic big-data
support, another adapter, cancellation, retry, or checkpointing.
