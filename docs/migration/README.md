# Legacy migration ledger

This directory records evidence-gated decisions about legacy material. It is
not a compatibility layer and does not make legacy code part of `veridist`.

`legacy-salvage-ledger.schema.json` defines the portable Draft 2020-12 shape.
`legacy-salvage-ledger.json` is the reviewed decision register. Run:

```powershell
python python/tools/check_migration_ledger.py
```

The checker validates both JSON Schema-shaped fields and cross-field policy,
then requires the recorded immutable source *commit* to exist and be a commit,
verifies `commit:path` is exactly the recorded blob, recomputes its SHA-256,
and validates closed, ordered, in-bounds line ranges tied to that same path.
Missing history is a failure, not a fallback to the working tree. A mismatch
means the entry is stale and must not be used as reuse evidence.

The only allowed dispositions are:

- `modify_port`: an identified non-runtime asset may be adapted after its own
  specification and tests exist;
- `rewrite`: retain scenario/evidence value while implementing independently;
- `archive`: retain read-only historical context with no implementation use.

No entry permits importing, packaging, executing, or using `distfit_pro` as a
runtime fallback or numerical oracle. The first exponential vertical is
explicitly an independent rewrite. Its statistical evidence is recorded under
LM-002, but that entry remains `review_pending` until the migration review is
closed. LM-003 remains a translation candidate; current Persian content does
not claim reuse of a legacy phrase or a completed migration review.
