# veridist

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python](https://img.shields.io/pypi/pyversions/veridist.svg)](https://pypi.org/project/veridist/)
[![CI](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage](https://codecov.io/gh/alisadeghiaghili/veridist/graph/badge.svg)](https://app.codecov.io/gh/alisadeghiaghili/veridist)
[![Mutation gate](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml)
[![Release evidence](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-7b1fa2.svg)](LICENSE)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

## Distribution fitting with a visible evidence trail

`veridist` 1.0.0 is an evidence-first distribution-fitting package for teams
that need a lifetime-fit result they can inspect, reproduce, and constrain. It
turns a strict CSV of event times into a typed result and execution record,
instead of silently guessing input semantics or presenting a generic
"best-fit" answer.

## Why Veridist

| What you need | What Veridist provides |
| --- | --- |
| A clear starting point | Fixed-location Exponential, Weibull-minimum, and Lognormal MLE cells for exact and independently right-censored lifetimes. |
| Inputs you can trust | A strict UTF-8 `time,event_observed` CSV contract: `1` is an event and `0` is independent right censoring. |
| Results you can audit | Typed estimates and failures, one-pass execution facts, and redacted source provenance. |
| Evidence before adoption | CI coverage gates, a critical mutation gate, reproducible-release checks, and retained release evidence. |

It is aimed at reliability, engineering, and data-science teams working with a
declared lifetime model. It is not a broad exploratory fitting workbench.

## Start in 60 seconds

Install the published package:

```console
python -m pip install veridist
```

Then run this complete example. It writes a tiny censored-lifetime CSV, fits
the supported exponential vertical, and checks the returned model assumptions.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    result = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32_768, 32_768),
    )

fit = result.fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

For a checkout instead, use `cd veridist/python` followed by `python -m pip
install .`. The package page has the same executable example in
[English](python/README.md), [Persian](python/README.fa.md), and
[German](python/README.de.md).

## Choose a workflow

| Goal | Start here |
| --- | --- |
| Fit a small, strict lifetime CSV | [First fit and CSV contract](python/docs/source/exponential-right-censoring.md) |
| Fit data with independent right censoring | [Censoring model and failure cases](python/docs/source/exponential-right-censoring.md#model-and-estimate) |
| Inspect a result or a typed failure | [CSV exponential API](python/docs/source/api.md) |
| Evaluate a declared distribution or reduce caller-owned chunks | [Scalar families and streaming likelihood](python/docs/source/families-log-density-likelihood.md) |
| Plan local checkpoint and resume work | [Checkpoint/resume contract tests](python/tests/contract/test_v1_checkpointed_csv.py) |

The checkpoint store is local SQLite state. A compatible run may resume its
committed prefix; it is not a distributed checkpoint service. Read the linked
contract before putting it on an operational path.

## Validate before you trust a result

Read the returned fit together with its declared model assumptions and execution
record. A finite point estimate is not a confidence interval or a goodness-of-
fit conclusion. The uncensored exponential cell has refit Monte Carlo KS, AD,
and CvM assessment plus AIC/BIC and adequacy-gated selection; that inference is
limited to its declared cell and caller-owned generator.

The main CI runs supported Python versions, enforces global line and branch
coverage of at least 95%, and publishes the Python 3.11 coverage report behind
the Coverage badge. The badge is a live Codecov result, not a README value.
The [quality contract](python/tools/check_coverage.py) rejects weak or
incomplete coverage evidence.

## Evidence, scale, and production boundaries

Release validation binds a candidate SHA, rebuilds the package twice, and
requires byte-for-byte matching artifacts. The release boundary includes a
27-cell execution matrix across complete, retry-resume, and cancellation
scenarios at 10k, 100k, and 1m rows on Linux, macOS, and Windows.

Those checks do not establish a general throughput, RSS, distributed, Parquet,
Arrow, dataframe, database, broad censoring, vectorized, or universal best-fit
capability. Review [known limits](python/KNOWN_LIMITS.md) before using a result
in a production decision path, and consult the [evidence ledger](docs/v1-readiness.md)
for the exact scope of retained evidence.

The repository still contains `distfit_pro` material for its historical and
audited context. It is not a compatibility promise for `veridist`; see the
[legacy migration ledger](docs/migration/README.md) for its evidence-gated
status.

## Find the right documentation

| You are… | Use this |
| --- | --- |
| Trying the package | [Package guide](python/README.md) and [CSV tutorial](python/docs/source/exponential-right-censoring.md) |
| Integrating an API | [API reference](python/docs/source/api.md) and [known limits](python/KNOWN_LIMITS.md) |
| Assessing the statistical surface | [Family and likelihood guide](python/docs/source/families-log-density-likelihood.md) |
| Reviewing quality or release evidence | [Test plan](docs/v1-test-plan.md), [readiness ledger](docs/v1-readiness.md), and [ADRs](docs/adr/README.md) |
| Contributing | [Contributing guide](CONTRIBUTING.md), [engineering conventions](docs/conventions.md), and [documentation toolchain](python/docs/README.md) |

## Support, security, and license

Use [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) for
reproducible defects and feature proposals. Report vulnerabilities through the
process in [SECURITY.md](SECURITY.md). Release history is in
[python/CHANGELOG.md](python/CHANGELOG.md). The repository and nested package
use BUSL-1.1 with the Apache-2.0 additional-use grant stated in [LICENSE](LICENSE).
