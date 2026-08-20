# Documentation toolchain

English under `docs/source` is canonical. Persian and German use Sphinx gettext
catalogs under `docs/locales`; `docs/i18n/parity-manifest.json` owns stable page,
message and example IDs. Missing or source-identical translations fail the
dependency-free parity check instead of silently falling back to English.

Install the exact documentation extra declared by the project:

```console
python -m pip install -e ".[docs,test]"
```

From `python/`, run the structural and parity gate:

```console
python -m unittest discover -s tests/docs -p "test_*.py" -v
python docs/toolchain.py check
```

Build gettext and all locales with warnings fatal:

```console
sphinx-build -b gettext -W -n docs/source docs/_build/gettext
sphinx-intl update -p docs/_build/gettext -d docs/locales -l fa -l de
sphinx-build -b html -W -n docs/source docs/_build/en/html -D language=en
sphinx-build -b html -W -n docs/source docs/_build/fa/html -D language=fa
sphinx-build -b html -W -n docs/source docs/_build/de/html -D language=de
sphinx-build -b linkcheck -W -n docs/source docs/_build/linkcheck -D language=en
python docs/toolchain.py render docs/_build/en/html en
python docs/toolchain.py render docs/_build/fa/html fa
python docs/toolchain.py render docs/_build/de/html de
```

`toolchain.py render` requires an explicit `lang` and `dir` on every HTML page.
For Persian it also requires the RTL stylesheet and at least one rendered code
block. Generated `_build` files are evidence artifacts and are not committed.
