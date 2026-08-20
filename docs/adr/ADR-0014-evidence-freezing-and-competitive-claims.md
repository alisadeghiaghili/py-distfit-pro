# ADR-0014: Evidence freezing and competitive claims

Status: Proposed

Owner: competitive-research lead (TBD); publication approval: Ali Sadeghi
Aghili / release owner

Decision note: 2026-08-20 decisions did not approve a source-lock storage
location, archive-retention/privacy policy, or public-claim workflow. This ADR
therefore remains Proposed.

## Context

Competitive URLs are mutable and a feature cell can be narrower than its label.
The current landscape and CSV are internal drafts with no fabricated archive,
hash, snapshot or source-lock coverage claim. A public comparative claim needs
an auditable source boundary without copying material beyond lawful limits.

## Decision

Adopt `competitive-evidence-policy.md` as the proposed release-control model.
Prefer immutable, tagged or versioned primary sources. When only mutable
official material exists, retain permissible metadata and a minimal locating
excerpt, retrieval timestamp, cited version, archive/snapshot locator where
lawful, and normalized excerpt/content hash where lawful. Respect copyright,
licence, robots and terms; record why a lawful field is `not_captured` rather
than fabricate it.

The planned source-lock registry derives stable claim-cell IDs from normalized
CSV fields and records source/scope/reviewer/revalidation metadata. No public
competitive comparison may ship until lock coverage is 100% for every CSV cell
asserting `supported` or `not_supported`. A negative assertion is bounded to
the documented API/version/scope; silence in a page is not proof of absence.

## Scope

Applies to public competitive matrices, feature comparisons, benchmarks,
release notes, presentations, papers and adoption material. It does not make
the census exhaustive, validate a competitor's statistical correctness, or
replace veridist's own calibration evidence.

## Evidence

Current evidence consists only of draft URLs with `cited_version` and
`retrieved_at` in the CSV. No source-lock manifest, archive locator, excerpt
or hash has been captured.

## Unresolved before acceptance

Choose a licence-aware storage location, retention/deletion rules, access
control, lawful archive/snapshot policy, and owner for any potentially
identifying or vendor-controlled evidence metadata. Until then, no source-lock
record or publication gate may be represented as implemented.

## Test implications

A **NOT IMPLEMENTED** validator must parse the CSV, derive unique claim-cell
IDs, validate lock records and lawful `not_captured` reasons, reject stale or
scope-mismatched locks, calculate required coverage, and fail any public-build
flag below 1.00. It must not require or manufacture unlawful content copies.

## Dependencies

ADR-0012, a licence-aware archive/metadata store, a registry schema, CI/release
tooling, and named research/statistical/release reviewers.

## Consequences

The current competitive documents are marked Internal Draft / not publishable;
their URLs remain research pointers until the lock gate is met. This slows
marketing comparisons, deliberately, but makes each eventual claim disputable
and maintainable rather than promotional folklore.
