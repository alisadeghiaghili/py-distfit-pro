# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

`veridist` 0.9.1 is an evidence-first distribution-fitting package whose stated
scope is bound to executable CI, coverage, mutation, release, and scale
contracts.

The current public scope includes strict UTF-8 CSV and resumable local SQLite
execution; fixed-location Exponential, Weibull-minimum, and Lognormal MLE cells
for exact and independently right-censored lifetimes; scalar operations for the
declared continuous families; and refit Monte Carlo goodness-of-fit plus
adequacy-gated model selection for uncensored exponential samples. See the
known-limits documents for the exact boundaries; this release makes no general
RSS, throughput, broad censoring, or universal best-fit claim.

Install an evaluation build from the nested project after cloning:

```console
cd veridist/python
python -m pip install .
```

The package landing pages give the executable example, adapter contracts, and
limits in [English](python/README.md), [Persian](python/README.fa.md), and
[German](python/README.de.md). The [changelog](python/CHANGELOG.md),
[known limits](python/KNOWN_LIMITS.md), and [v1 roadmap](docs/v1-roadmap.md)
define the release boundary and the remaining work toward 1.0.

## Security and license

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
The repository and nested package use BUSL-1.1 with the Apache-2.0
additional-use grant stated in [LICENSE](LICENSE). No license change is made
by this pre-release documentation.
