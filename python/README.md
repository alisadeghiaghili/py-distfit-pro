# veridist

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python](https://img.shields.io/pypi/pyversions/veridist.svg)](https://pypi.org/project/veridist/)
[![CI](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage](https://codecov.io/gh/alisadeghiaghili/veridist/graph/badge.svg)](https://app.codecov.io/gh/alisadeghiaghili/veridist)
[![Mutation gate](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml)
[![Release evidence](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-7b1fa2.svg)](LICENSE)

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

## Fit declared lifetime models without hiding the contract

`veridist` 1.0.0 is an evidence-backed public contract release. It helps a
reliability or data-science team turn a strict lifetime CSV into a typed fit and
an execution record that makes the model, input, and operational boundaries
visible.

## Install and first success

Install the released package:

```console
python -m pip install veridist
```

For a source checkout:

```console
git clone https://github.com/alisadeghiaghili/veridist.git
cd veridist/python
python -m pip install .
```

Or install a wheel from a verified run:

```console
python -m pip install /path/to/veridist-1.0.0-py3-none-any.whl
```

Run this complete CSV fit after installation:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
```

## Pick the right path

| Need | Use |
| --- | --- |
| A strict CSV lifetime fit | `fit_exponential_csv` and the [CSV tutorial](docs/source/exponential-right-censoring.md) |
| A declared scalar distribution operation | `FAMILY_REGISTRY` and `evaluate_log_density`; see the [family guide](docs/source/families-log-density-likelihood.md) |
| An exact-state reducer over caller-owned chunks | `reduce_log_likelihood_chunks`; see the [stream source API](docs/source/api.md#generic-stream-source-api) |
| Local checkpoint and resume design | `SQLiteCheckpointStore` plus the [checkpoint CSV contract](tests/contract/test_v1_checkpointed_csv.py) |

## Capability and evidence

The fitting surface contains fixed-location Exponential, Weibull-minimum, and
Lognormal MLE cells for exact and independently right-censored lifetimes. A
finite solution yields a point estimate; invalid statistical or operational
conditions yield typed failures. Inference is restricted to that declared cell:
the uncensored exponential cell supports refit Monte Carlo KS, AD, and CvM,
AIC/BIC, and adequacy-gated selection with a caller-owned generator.

The public CSV path is strict: UTF-8 with exactly `time,event_observed`, event
token `1`, and right-censoring token `0`. It uses one iterator pass. It is not
a generic CSV reader. A successful call should be read with its execution
record and model assumptions.

The CI gate checks supported Python versions, at least 95% global line and
branch coverage, quality checks, package installation, and documentation.
Coverage is published from the Python 3.11 run to the live Codecov badge;
there is no static coverage number in this page. The mutation and release-
evidence badges link to their own verifiable workflows.

## Scale and production boundaries

Retained evidence covers a measured 10k/100k/1m by 32KiB/64KiB/128KiB matrix
for the declared paths. It does not establish general big-data support,
throughput, portable RSS, dataframe, Parquet, Arrow, database, distributed
execution, broad censoring, vectorized operations, or universal model choice.
`SQLiteCheckpointStore` is durable local state, not a distributed service.

Read [KNOWN_LIMITS.md](KNOWN_LIMITS.md) and the repository
[evidence ledger](../docs/v1-readiness.md) before production use.

## Documentation, contribution, and support

Use the [API reference](docs/source/api.md) to integrate, the
[family guide](docs/source/families-log-density-likelihood.md) to assess the
statistical surface, and the [documentation toolchain](docs/README.md) to work
on docs. For changes, start with the repository [contribution guide](../CONTRIBUTING.md)
and [engineering conventions](../docs/conventions.md). Report reproducible
defects through [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues)
and vulnerabilities through [SECURITY.md](../SECURITY.md).

The package uses BUSL-1.1 with an Apache-2.0 additional-use grant for personal,
non-commercial use; see [LICENSE](LICENSE).
