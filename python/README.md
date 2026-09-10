# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

## Status

`veridist` 0.5.0 is the first evidence-backed public contract release.
It specifies and tests bounded delivery, replayability, pass budgets,
transactional retry, checkpoint compatibility, typed failures, execution
outcomes, and redacted provenance.

This build includes an experimental rate-only exponential MLE for exact and
independently right-censored lifetimes. It provides a point estimate when a
finite MLE exists and typed failures otherwise. Inference is not provided.
Its public CSV path is strict: UTF-8 CSV with exactly `time,event_observed`,
event token `1`, and right-censoring token `0`. It executes one iterator pass
with a declared logical retained-payload chunk budget and returns a closed,
typed execution result. This is not a generic CSV reader or a portable RSS,
throughput, cancellation, retry, checkpoint, or broad out-of-core claim.
Retained evidence establishes bounded internal payload only for the measured
10k/100k/1m by 32KiB/64KiB/128KiB matrix; it does not establish a general
big-data or high-throughput capability.

`IterableDataSource` is the reusable public stream adapter for caller-owned
chunk iterables. Its immutable metadata explicitly declares one-pass or
replayable acquisition: a single-pass source fails with a typed pass-budget
error if acquired twice, while a replayable source requires an iterator
factory. `BoundedChunkBuffer` charges queued and consumer-held chunks until
`BufferedChunk.release()`; callers must release received chunks, normally in
`finally`. The only shipped file adapter remains the strict CSV lifetime
adapter—this does not add generic CSV, Parquet, Arrow, dataframe, database,
or broad out-of-core adapters.

The separate scalar surface exposes immutable `FAMILY_REGISTRY` metadata for
normal, gamma, Weibull-minimum, lognormal, and right-Gumbel; scalar
`evaluate_log_density`; and exact-state `reduce_log_likelihood_chunks`.
It is not generic fitting, inference, goodness-of-fit, ranking, arrays, or
censoring. The reducer represents successful binary64 terms exactly and rounds
the final total once; its unsigned-64 count cap implies a 2162-bit exact-total
bound. Its retained 10k/100k/1m evidence is scoped to tested normal streams.

Candidate scale measurement is deliberately manual: the `veridist-scale-evidence`
workflow first binds a clean, full candidate SHA and runs the evidence contracts,
then measures the public iterable likelihood and strict CSV/exponential paths on
Linux and Windows. It retains artifacts only after their own fail-closed SHA and
schema validation. Merely having this workflow, or a historical artifact, is not
evidence for a new candidate and is not a throughput or RSS claim.

## Install an evaluation build

Install from the nested source project after cloning the repository:

```console
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro/python
python -m pip install .
```

Or install a wheel that you built or obtained from a specific verified run:

```console
python -m pip install /path/to/veridist-0.0.0.dev0-py3-none-any.whl
```

The project does not direct users to install an unreleased package name from a
public index.

## Try the experimental vertical

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path, schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
```

See the [documentation toolchain](docs/README.md) and the repository's
[evidence ledger](../docs/v1-readiness.md) for implemented checks and explicit
limits.

## License

BUSL-1.1 with an Apache-2.0 additional-use grant for personal,
non-commercial use; see [LICENSE](LICENSE).
