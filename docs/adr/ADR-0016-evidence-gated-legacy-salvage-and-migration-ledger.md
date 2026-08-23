# ADR-0016: Evidence-gated legacy salvage and migration ledger

Status: Accepted

Owner: Ali Sadeghi Aghili

## Context

ADR-0001 correctly prohibited treating the legacy implementation as a
specification or runtime authority.  Some legacy assets may still contain
useful scenario descriptions, tests, or non-statistical infrastructure.
Discarding them without inspection is wasteful; copying them without evidence
would reintroduce the exact risk ADR-0001 was written to avoid.

## Decision

`veridist` is the only release name and runtime namespace.  Legacy is never a
runtime dependency, import target, dynamic fallback, packaged payload, or
numerical oracle.  A legacy component may be considered only when an entry in
the migration ledger identifies its immutable source commit/blob and SHA-256,
license status, independent evidence, review record, limits, and one of these
dispositions: `modify_port`, `rewrite`, or `archive`.

This partially supersedes ADR-0001 only by permitting evidence-gated salvage.
Its Python-first rule, independent goldens, and prohibition on treating legacy
results as an oracle remain binding.  Exponential fitting is a rewrite, not a
port.  Translations are reviewed-content candidates only; they are not an
authority for behaviour.

## Scope

This record governs a ledger and enforcement boundary only.  It does not
approve a statistical family, legacy API compatibility, adapter, report, or
translation.  Core diagnostics remain locale-neutral structured data.  Any
user-facing EN/FA/DE surface requires parity review.  Persian HTML/PDF must
declare `lang=fa` and `dir=rtl`, preserve code/formula/Latin identifiers with
LTR isolation, and pass a browser screenshot gate before a rendering claim.

## Evidence

The checked-in ledger is schema-validated and semantically checked by a
stdlib-only tool.  Its hashes are computed from files at the recorded
`origin/main` legacy revision; hash mismatch is stale evidence, not a warning.
The isolation tests inspect AST import forms and package artifacts so a simple
string workaround cannot bypass the rule.

## Test implications

- `ML-01` validates schema and required ledger fields.
- `ML-02` rejects stale source hashes and invalid cross-field combinations.
- `LI-01` rejects static, dynamic, and string-mediated imports of legacy.
- `LI-02` inspects built wheel/sdist contents when build tools are available.
- `I18N-RTL-01` remains a required future browser screenshot gate; this ADR
  records it but does not claim it is implemented.

## Dependencies

ADR-0001, ADR-0005, ADR-0006, ADR-0010, ADR-0011, ADR-0013; a later accepted family ADR is needed
before any statistical implementation is advertised.

## Consequences

Salvage is slower than copying, but each reuse decision is auditable and
reversible.  The ledger is a governance artifact, not proof of correctness.

## Exit criteria and effort class

Governance is complete when the ledger checker and import/package isolation
tests are green.  A component is eligible for implementation only after its
own independent specification and RED tests exist.  Effort class: small.
