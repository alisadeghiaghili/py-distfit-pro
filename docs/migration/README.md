# Legacy migration ledger

This directory records evidence-gated decisions about legacy material. It is
not a compatibility layer and does not make legacy code part of `veridist`.

`legacy-salvage-ledger.schema.json` defines the portable Draft 2020-12 shape.
`legacy-salvage-ledger.json` is the reviewed decision register. Run:

```powershell
python python/tools/check_migration_ledger.py
```

The checker validates both JSON Schema-shaped fields and cross-field policy,
then recomputes each recorded legacy file hash from the working tree. A hash
mismatch means the entry is stale and must not be used as reuse evidence.

The only allowed dispositions are:

- `modify_port`: an identified non-runtime asset may be adapted after its own
  specification and tests exist;
- `rewrite`: retain scenario/evidence value while implementing independently;
- `archive`: retain read-only historical context with no implementation use.

No entry permits importing, packaging, executing, or using `distfit_pro` as a
runtime fallback or numerical oracle. The first exponential vertical is
explicitly a rewrite.
