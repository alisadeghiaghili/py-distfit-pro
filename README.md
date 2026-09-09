# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

`veridist` is an evidence-first, pre-alpha distribution-fitting package. Its
formal package version is `0.0.0.dev0`; it is not a released distribution and
is not available from a public package index.

The current public scope is deliberately narrow: a strict UTF-8 CSV,
fixed-location, rate-only exponential MLE for exact and independently
right-censored lifetimes; five scalar log-density evaluators; and exact-state
streaming likelihood reduction. It does not provide generic fitting,
inference, goodness-of-fit, ranking, broad censoring, a generic CSV reader, or
a general out-of-core or performance claim.

Install an evaluation build from the nested project after cloning:

```console
cd py-distfit-pro/python
python -m pip install .
```

The package landing pages give the executable example, adapter contracts, and
limits in [English](python/README.md), [Persian](python/README.fa.md), and
[German](python/README.de.md). The candidate [changelog](python/CHANGELOG.md)
and [known limits](python/KNOWN_LIMITS.md) are maintained separately. The binding release contract is
[ADR-0020](docs/adr/ADR-0020-veridist-0.5-release-contract.md); it keeps the
version at `0.0.0.dev0` until its candidate-specific gates pass.

## Security and license

Report vulnerabilities through the process in [SECURITY.md](SECURITY.md).
The repository and nested package use BUSL-1.1 with the Apache-2.0
additional-use grant stated in [LICENSE](LICENSE). No license change is made
by this pre-release documentation.
