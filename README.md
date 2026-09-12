# Veridist

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python](https://img.shields.io/pypi/pyversions/veridist.svg)](https://pypi.org/project/veridist/)
[![CI](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=coverage%20%E2%89%A595%25)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Mutation gate](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml)
[![Release evidence](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-7b1fa2.svg)](LICENSE)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

## Turn lifetime data into a result you can review

Veridist is a Python package for fitting declared lifetime models to event-time
data, such as component failure times. It validates a small, explicit CSV
format and returns both the fit and facts about how the calculation ran. Use it
when another person needs to understand the data assumptions behind the result,
not only see a curve.

```console
python -m pip install veridist
```

[Run the first example](#run-your-first-fit) · [Choose your next task](#what-you-can-do-today) · [Read the package guide](python/README.md)

## Run your first fit

The example below creates its own CSV, so it works immediately after
installation. `1` means that the event was observed. `0` means that the event
had not happened by the end of observation; this is called right censoring.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32_768, 32_768),
    ).fit

assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

```text
rate=0.5; events=1; censored=1
```

`rate` is expressed in the inverse of the time unit used in your CSV. Here the
time values have no stated unit, so the rate is `0.5` per input-time unit. A
successful fit means the declared calculation completed; it does not by itself
prove that the exponential model is suitable. Review the model assumptions
before using a result in a decision.

## What you can do today

| Your task | Start here |
| --- | --- |
| Estimate a reliability model from a small lifetime CSV | [Fit a strict CSV](python/docs/source/exponential-right-censoring.md) |
| Work with observations that have not yet failed | [Understand right censoring and its assumptions](python/docs/source/exponential-right-censoring.md#model-and-estimate) |
| Inspect a result or handle an input problem | [Read the CSV API and failure cases](python/docs/source/api.md) |
| Evaluate a supported distribution or process your own chunks | [Use scalar families and streaming likelihood](python/docs/source/families-log-density-likelihood.md) |
| Continue a compatible local run after interruption | [Follow the checkpoint and resume recipe](python/examples/checkpoint_resume.py) |

Veridist currently supports fixed-location Exponential, Weibull-minimum, and
Lognormal fits for exact and independently right-censored lifetime data. The
CSV entry point uses the Exponential workflow. Scalar distribution operations
are available for the declared registry; they are separate from the fitting
surface.

## When a run is interrupted

For a single machine and local filesystem, `SQLiteCheckpointStore` can retain a
committed prefix and continue a compatible CSV reduction. Keep the source
revision stable and reopen the same local store. The [executable checkpoint
recipe](python/examples/checkpoint_resume.py) shows the setup and a second pass.

This is a local restart mechanism, not a distributed job system. Use the
[known limits](python/KNOWN_LIMITS.md) to decide whether its boundaries match
your workload.

## Why you can trust what is shown here

| Evidence | What it means |
| --- | --- |
| PyPI and Python | The published package and its declared Python compatibility |
| CI | Tests, package installation, documentation, and browser checks run on `main` |
| Coverage ≥95% | CI enforces at least 95% global line and branch coverage; the badge reports that gate's pass/fail state, not a made-up percentage |
| Mutation gate | The critical statistical core is tested against code mutations |
| Release evidence | A release candidate is rebuilt and its artifacts are checked for reproducibility |

The release evidence covers the stated paths at 10k, 100k, and 1m rows under
defined test conditions. It is not a general promise about throughput, memory,
distributed execution, Parquet, dataframes, or a universal “best” model.

## Before production use

Read [known limits](python/KNOWN_LIMITS.md), then review the [API reference](python/docs/source/api.md)
and [release evidence](docs/v1-readiness.md) for the exact scope. In particular,
the input is strict UTF-8 CSV with `time,event_observed`, checkpoints are local
SQLite state, and inference is narrower than the full distribution registry.

The repository retains `distfit_pro` material for historical context. It is not
a compatibility promise for Veridist; its status is recorded in the [legacy
migration ledger](docs/migration/README.md).

## Cite Veridist

Cite the release that produced your result.

> A. S. Aghili, “Veridist,” ver. 1.0.1, Sep. 2026. [Online]. Available:
> https://github.com/alisadeghiaghili/veridist/releases/tag/v1.0.1

The [citation guide](docs/citing-veridist.md) also provides APA 7, BibTeX, RIS, EndNote XML, CSL-JSON, Chicago, MLA 9, Harvard, and Vancouver. The canonical machine-readable record is [CITATION.cff](CITATION.cff).
## Help and further reading

Start with the [package guide](python/README.md) for installation and examples,
the [statistics guide](python/docs/source/families-log-density-likelihood.md)
for the supported model surface, and [known limits](python/KNOWN_LIMITS.md) for
adoption decisions. Report reproducible defects through [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues)
and security problems through [SECURITY.md](SECURITY.md). Contributors should
read [CONTRIBUTING.md](CONTRIBUTING.md) and [engineering conventions](docs/conventions.md).

The package is licensed under BUSL-1.1 with the Apache-2.0 additional-use grant
in [LICENSE](LICENSE). Release history is in [python/CHANGELOG.md](python/CHANGELOG.md).
