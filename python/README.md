# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

## Status

`veridist` 0.0.0.dev0 is a pre-alpha contract kernel under active development.
It specifies and tests bounded delivery, replayability, pass budgets,
transactional retry, checkpoint compatibility, typed failures, execution
outcomes, and redacted provenance.

This build does not provide a distribution-fitting API. It does not ship
production data adapters. It does not claim persistent checkpoint durability.
The in-memory sources and checkpoint stores are contract fixtures, not
production storage or orchestration components.

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

## Verify the package boundary

```python
import veridist

assert veridist.__version__ == "0.0.0.dev0"
```

See the [documentation toolchain](docs/README.md) and the repository's
[evidence ledger](../docs/v1-readiness.md) for implemented checks and explicit
limits.

## License

MIT; see [LICENSE](LICENSE).
