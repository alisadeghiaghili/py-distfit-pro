(veridist-home)=
# Veridist documentation

Evidence-first distribution fitting.

## First callable vertical

The first callable vertical fits a fixed-location exponential lifetime model
with exact and independently right-censored observations. Its canonical
parameter is a positive rate; the reported mean is derived as its reciprocal.

## Limits

This vertical provides a point estimate only. `inference=not_provided` means no
confidence interval, goodness-of-fit result, truncation, weights, covariates,
or free location parameter is supplied.

## Executable example

The example below has one canonical executable source and reports only stable
machine facts. Use an explicit report locale; no report-language fallback is
performed.

Translations for this vertical are owner-reviewed provisional text; no external
native-speaker review is claimed.

```{literalinclude} examples/quickstart.py
:language: python
:caption: Canonical executable example
```

```{toctree}
:maxdepth: 2

api
exponential-right-censoring
```
