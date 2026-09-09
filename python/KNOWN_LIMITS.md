# Known limits for Veridist 0.5

This document describes the intended 0.5 release boundary. While the package
version is `0.0.0.dev0`, the release remains a candidate and these statements
must not be read as a publication claim.

- `FIT-CSV-EXP`: fitting is limited to a fixed-location, rate-only
  exponential model over exact and independently right-censored lifetimes.
  There is no inference, confidence interval, goodness-of-fit test, model
  ranking, weights, covariates, truncation, left censoring, interval censoring,
  or free location parameter.
- `CSV-STRICT`: the bundled file adapter accepts only UTF-8 CSV with exactly
  `time,event_observed`, where `1` is an exact event and `0` is independent
  right censoring. It is not a general CSV or spreadsheet reader.
- `SCALAR-FAMILIES`: normal, gamma, Weibull-minimum, lognormal, and
  right-Gumbel expose finite scalar log-density evaluation. They do not expose
  array evaluation, fitting, CDF, PPF, inference, censoring likelihood, or
  ranking.
- `STREAM-SOURCE`: `IterableDataSource` adapts caller-owned chunk iterables.
  The package bundles no Parquet, Arrow, dataframe, database, or network
  adapter. Checkpoint-replayable acquisition is rejected by this adapter.
- `MEMORY-BOUND`: the delivery bound covers queued payload and active consumer
  leases until explicit release. It is a logical retained-payload bound, not a
  portable RSS ceiling.
- `SCALE-EVIDENCE`: measurements apply only to their exact adapter, family,
  workload, platform, Python version, chunk limit, and candidate SHA. They do
  not establish universal throughput, generic big-data support, or a broad
  out-of-core capability.
- `LICENSE`: the package uses BUSL-1.1 with the Apache-2.0 additional-use grant
  stated in `LICENSE`; it changes to Apache-2.0 on 2030-09-05.

[فارسی](KNOWN_LIMITS.fa.md) | [Deutsch](KNOWN_LIMITS.de.md)
