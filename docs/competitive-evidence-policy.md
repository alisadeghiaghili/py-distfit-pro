# Competitive evidence policy

**Status: Internal Draft / not publishable.** This policy defines the proposed
evidence freeze required before publishing any competitive comparison. No lock,
archive, excerpt, hash or snapshot is asserted to exist yet.

## Purpose and boundary

The competitive landscape and feature matrix are bounded research aids, not an
exhaustive survey and not a claim that one tool is statistically superior.
Their current URLs remain draft research pointers. A capability status is
publishable only when its source and claim are reproducibly auditable under
this policy.

## Claim-cell record

The planned source-lock registry assigns each CSV row a deterministic
claim-cell ID derived from the normalized tuple
`tool | ecosystem | feature | capability_status | cited_version`. It records:

- claim-cell ID, CSV row identity and exact status asserted;
- canonical evidence URL, preferred immutable/tagged/versioned URL when
  available, cited version and retrieval timestamp;
- minimal claim excerpt or metadata sufficient to locate the assertion, kept
  within copyright and licence limits;
- archive/snapshot locator when legally permissible; normalized excerpt hash or
  content hash when lawful; and an explicit `not_captured` reason otherwise;
- reviewer, lock date, revalidation due date, and any scope qualification that
  prevents a broad inference from a narrow feature page.

Do not invent a hash, archive locator, version or excerpt. An absent lawful
archive/hash is evidence incomplete, not a reason to silently weaken the
record.

## Evidence hierarchy and freezing rules

1. Prefer immutable primary evidence: a tagged manual, released API reference,
   versioned source/doc artifact, official release note, or a primary paper.
2. If only a mutable official page exists, retain permissible metadata and a
   short locating excerpt, record the retrieval timestamp, and retain an
   archive/snapshot locator and normalized excerpt/content hash where lawful.
   Respect robots, copyright, licences and terms; never bulk-copy a vendor
   page merely to freeze it.
3. Secondary/vendor summaries can identify a research lead but cannot alone
   upgrade an `unverified` capability to `supported` or `not_supported`.
4. A `not_supported` claim must state its bounded API/version/scope. Absence
   from a page is not proof of absence from the product.

Before a public website, release note, paper, benchmark or sales comparison
uses the matrix, source-lock coverage must be **100% of all CSV rows whose
`capability_status` is `supported` or `not_supported`**. The planned checker
calculates `locked required cells / all required cells` and fails on any
missing, duplicate, stale, unresolvable or scope-mismatched cell. `partial`,
`unverified` and `not_applicable` rows remain clearly labelled and may not be
used to imply a locked affirmative or negative claim.

## Revalidation and review

Proposed cadence is before every public release/paper, and at least every 90
days for mutable sources. Immutable versioned records are rechecked before a
new census date or when their cited product version changes. The competitive
research lead owns intake; a statistical reviewer checks scope, and the release
owner approves publication. Cadence, owners and tooling remain Proposed until
ADR-0014 is accepted.

## Planned checks -- NOT IMPLEMENTED

The future validator must parse the long-form CSV, derive and de-duplicate
claim-cell IDs, validate required lock metadata, distinguish lawful
`not_captured` reasons from blank fields, calculate required lock coverage, and
emit an audit report without storing prohibited excerpts. It must reject a
public-build flag unless required coverage equals 1.00. This document does not
name an executable command because no registry, validator or CI configuration
exists yet.
