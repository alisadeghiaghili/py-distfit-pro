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
Its reducer has fixed O(1) algorithmic state, but the package does not ship
production data adapters and does not claim production out-of-core execution
or persistent checkpoint durability. In-memory sources and checkpoint stores
remain contract fixtures, not production storage or orchestration components.

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
from veridist.domain import ExactLifetime, RightCensoredLifetime
from veridist.families import ExponentialFitSuccess, fit_exponential
from veridist.reporting import ReportLocale, render_exponential_report

fit = fit_exponential([ExactLifetime(1.0), RightCensoredLifetime(1.0)])
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"

report = render_exponential_report(fit, ReportLocale.FA)
assert 'lang="fa" dir="rtl"' in report
```

See the [documentation toolchain](docs/README.md) and the repository's
[evidence ledger](../docs/v1-readiness.md) for implemented checks and explicit
limits.

## License

MIT; see [LICENSE](LICENSE).
