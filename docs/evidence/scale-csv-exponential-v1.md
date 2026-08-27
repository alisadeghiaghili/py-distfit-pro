# SCALE-CSV-EXP-01 retained evidence

The machine-readable artifact is
[`python/evidence/scale-csv-exponential-v1.json`](../../python/evidence/scale-csv-exponential-v1.json).

```powershell
python python/tools/check_scale_csv_exponential_evidence.py `
  --artifact python/evidence/scale-csv-exponential-v1.json
```

It records a clean CPython 3.14.6 / Windows 11 run at commit
`4490c9eb08e9ed5e420a2b677d9de843fdf66a5d`. The deterministic generator uses
Decimal arithmetic independently of the production fit and writes raw CSV only
to a private temporary directory before measurement. The artifact stores source
byte counts and SHA-256 values, never raw input paths.

| Logical rows | Chunk budgets | Passes | Internal payload result | Fit result |
| ---: | --- | --- | --- | --- |
| 10,000 | 32KiB, 64KiB, 128KiB | 1/1 in every cell | peak retained/inflight <= configured cap | Decimal facts agree |
| 100,000 | 32KiB, 64KiB, 128KiB | 1/1 in every cell | peak retained/inflight <= configured cap | Decimal facts agree |
| 1,000,000 | 32KiB, 64KiB, 128KiB | 1/1 in every cell | peak retained/inflight <= configured cap | Decimal facts agree |

The one-million-row cells took approximately 1,253--1,285 seconds each in
three isolated Windows processes. This is descriptive evidence, not a
performance target or a claim of fast large-data fitting. Per-cell tracemalloc
peaks are retained. RSS is absent because this Windows stdlib environment has
no supported `resource` RSS interface; it must not be read as zero.

This evidence establishes only bounded internal retained-payload behavior for
the strict CSV/exponential path on this matrix. It does not claim a portable
RSS ceiling, generic large-data support, backpressure stress behavior, parallel
speed-up, cancellation, checkpoint/retry, or support for other input formats.
