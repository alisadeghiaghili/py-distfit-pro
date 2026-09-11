# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

`veridist` 1.0.0 is an evidence-first distribution-fitting package whose stated
scope is bound to executable CI, coverage, mutation, release, and scale
contracts.

The current public scope includes strict UTF-8 CSV and resumable local SQLite
execution; fixed-location Exponential, Weibull-minimum, and Lognormal MLE cells
for exact and independently right-censored lifetimes; scalar operations for the
declared continuous families; and refit Monte Carlo goodness-of-fit plus
adequacy-gated model selection for uncensored exponential samples. See the
known-limits documents for the exact boundaries; this release makes no general
RSS, throughput, broad censoring, or universal best-fit claim.

## Install

Install the published package from PyPI:

```console
python -m pip install veridist
```

For development from a repository checkout:

```console
cd python
python -m pip install .
```

## Quick start

The public CSV vertical accepts UTF-8 data with the exact columns
`time,event_observed`: use `1` for an observed event and `0` for independent
right-censoring.

```python
from pathlib import Path
from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

csv_path = Path("lifetimes.csv")
result = fit_exponential_csv(
    csv_path,
    schema=CsvLifetimeSchema("time", "event_observed"),
    source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
    limits=CsvLifetimeLimits(32_768, 32_768),
)
assert isinstance(result.fit, ExponentialFitSuccess)
print(result.fit.rate)
```

For large CSV lifetimes, use `fit_exponential_checkpointed_csv` with a local
`SQLiteCheckpointStore`. A cancellation commits the completed chunk prefix;
the next compatible run resumes from that nonzero cursor. The package guide
defines the source-revision, cancellation, and retry contracts.

## What is supported

| Area | Public contract |
| --- | --- |
| Lifetime fitting | Fixed-location Exponential, Weibull-minimum, and Lognormal MLE for exact and independently right-censored observations |
| Model assessment | AIC/BIC, adequacy-gated selection, and refit Monte Carlo KS, AD, and CvM for uncensored exponential samples |
| Execution | Strict UTF-8 lifetime CSV; one-pass iterable streams; bounded delivery; local SQLite checkpoint and resume |
| Distribution operations | Scalar log-density, CDF, survival, quantile, and caller-owned RNG sampling for the declared registry |

## Boundaries

The package does not claim generic dataframe, Parquet, Arrow, database, or
distributed adapters; a portable checkpoint store; general throughput or RSS
guarantees; broad censoring support; vectorized distribution operations; or a
universal best-fit recommendation. Read [known limits](python/KNOWN_LIMITS.md)
before adopting it in a production decision path.

Start with the executable [package guide](python/README.md). It documents the
CSV schema, checkpoint and resume contract, and a minimal fit. Localized guides
are available in [Persian](python/README.fa.md) and [German](python/README.de.md).

## Release evidence

The `1.0.0` release boundary requires a candidate-bound 27-cell execution
matrix: complete, retry-resume, and cancellation scenarios at 10k, 100k, and
1m rows on Linux, macOS, and Windows. Release artifacts are built twice and
must match byte-for-byte before publication. The [changelog](python/CHANGELOG.md),
[known limits](python/KNOWN_LIMITS.md), and [v1 roadmap](docs/v1-roadmap.md)
record the public boundary and remaining work.

## Security and license

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
The repository and nested package use BUSL-1.1 with the Apache-2.0
additional-use grant stated in [LICENSE](LICENSE).
