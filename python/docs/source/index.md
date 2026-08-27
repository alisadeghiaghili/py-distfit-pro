(veridist-home)=
# Veridist documentation

Evidence-first distribution fitting.

## First CSV vertical

The first public vertical fits a fixed-location exponential lifetime model from
a strict UTF-8 CSV source. Its canonical parameter is a positive rate; the
reported mean is derived as its reciprocal.

## Limits

The CSV schema is exactly `time,event_observed`; event `1` is exact and `0` is
independent right censoring. The adapter makes one iterator pass and retains at
most its declared logical payload budget. `inference=not_provided` means no
confidence interval, goodness-of-fit result, truncation, weights, covariates,
or free location parameter is supplied.

## Executable example

The example below has one canonical executable source and reports only stable
machine facts. Use an opaque public source identifier; file paths are not part
of the returned provenance.

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
