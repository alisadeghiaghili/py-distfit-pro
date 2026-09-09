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

## Generic stream source API

`DataSourceMetadata`, `Replayability`, `IterableDataSource`, `FamilyId`, and
`reduce_log_likelihood_chunks` are public so a caller can construct and consume
a generic in-memory or caller-owned stream. A `SINGLE_PASS` source receives an
iterable and is acquired once; `REPLAYABLE` requires a zero-argument iterator
factory. `CHECKPOINT_REPLAYABLE` is not yet implemented by this adapter and
fails immediately with typed `CHECKPOINT_REQUIRED`; do not treat it as a
promise of checkpoint recovery.

```python
from veridist import (
    DataSourceMetadata,
    FamilyId,
    IterableDataSource,
    Replayability,
    reduce_log_likelihood_chunks,
)

metadata = DataSourceMetadata(
    source_id="public-stream",
    schema_version="1",
    provenance_schema_version="1",
    replayability=Replayability.SINGLE_PASS,
    redaction_reason="example",
)
source = IterableDataSource(((0.0, 1.0),), metadata)
result = reduce_log_likelihood_chunks(FamilyId.NORMAL, source, mu=0.0, sigma=1.0)
```
