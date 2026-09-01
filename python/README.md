# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

## Status

`veridist` 0.0.0.dev0 is a pre-alpha contract kernel under active development.
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

The separate scalar surface exposes immutable `FAMILY_REGISTRY` metadata for
normal, gamma, Weibull-minimum, lognormal, and right-Gumbel; scalar
`evaluate_log_density`; and exact-state `reduce_log_likelihood_chunks`.
It is not generic fitting, inference, goodness-of-fit, ranking, arrays, or
censoring. The reducer represents successful binary64 terms exactly and rounds
the final total once; its unsigned-64 count cap implies a 2162-bit exact-total
bound. Its retained 10k/100k/1m evidence is scoped to tested normal streams.

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

MIT; see [LICENSE](LICENSE).
