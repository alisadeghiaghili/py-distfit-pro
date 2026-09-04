# ADR-0020: Veridist 0.5 release contract and fail-closed publication

Status: Accepted

Owner: Ali Sadeghi Aghili

## Context

The repository contains both a frozen legacy package and the new `veridist`
package.  A root legacy workflow previously reacted to GitHub release events
and could publish the repository-root legacy artifact.  That is incompatible
with the binding rule that `veridist` is the sole new runtime namespace and
release artifact.  It is also unsafe to attach a 0.5 tag to a development
version while the current evidence covers only narrow callable cells.

Earlier ADRs set v1 ambitions.  They do not by themselves define what a real,
honest 0.5 release may claim.  This record establishes a deliberately narrow
0.5 contract, a fail-closed release posture until a dedicated publisher exists,
and measurable gates that must pass before the version changes from
`0.0.0.dev0`.

## Decision

The legacy `.github/workflows/ci.yml` must neither subscribe to GitHub release
events nor contain a publication job, publication action, upload command, or
PyPI environment.  It may validate legacy code only.  No GitHub release may
publish `distfit_pro` from this repository.

No publishing workflow is authorized by this ADR.  Publication remains
fail-closed until a separate dedicated `veridist` workflow is implemented and
tested.  That workflow must build only from `python/`, verify the tag equals
the package version, inspect both sdist and wheel metadata and payloads, prove
the installed artifact imports `veridist` rather than legacy code, and publish
only after every required gate succeeds.  A repository release event alone is
never sufficient authority to publish.

The authoritative 0.5 scope is limited to these current cells:

1. The selected five scalar log-density families under their closed parameter,
   support, and typed-failure contracts.
2. Exact-state streaming likelihood reduction of those scalar binary64 outputs.
3. The strict CSV, independently right-censored, fixed-location exponential
   point-estimation vertical.
4. The transactional in-memory execution engine contracts that support those
   cells.

`0.5.0` may describe only behavior backed by retained, reproducible evidence
for those cells.  It must not imply a general distribution-fitting package,
general out-of-core engine, calibrated inference, broad censoring, or a
performance result beyond the declared measurement.

The package version remains `0.0.0.dev0` until every 0.5 exit criterion below
passes on the candidate being tagged.  An ADR, local test, branch name, or
green historical run is not a substitute for that evidence.

## Scope

Before a `0.5.0` tag, all of the following are mandatory:

- The delivery bound accounts for queued payload bytes **and active consumer
  leases**; a producer cannot exceed the stated hard byte bound by handing a
  chunk to a consumer.  The implementation, contracts, and retained evidence
  must use the corrected definition consistently.
- Every execution path, including cancellation, retry exhaustion, parser
  failure, consumer failure, and normal completion, has exhaustive resource
  cleanup evidence.  No claim of cleanup may rely solely on garbage
  collection.
- A narrow, reproducible million-row measurement for an in-scope CSV/
  exponential or streaming-likelihood cell records platform, Python version,
  input generation, chunk limit, elapsed time, throughput, and process RSS.
  Its checker must reject altered facts.  The report must state its exact
  platform and workload limits; it establishes neither universal throughput
  nor a generic out-of-core claim.
- Required portability evidence includes Linux and at least one Windows lane
  for the supported Python contract.  Any platform-dependent measurement is
  labelled as such.
- EN, FA, and DE documentation have semantic parity for the supported cells.
  The Persian rendered documentation and reports have retained RTL screenshots
  covering ordinary prose plus tables, inline/code blocks, and formulae, with
  required LTR isolates where applicable.  This ADR is English canonical and
  does not falsely assert translated-ADR parity.
- Candidate coverage is at least 95% line and 95% branch globally, and at
  least 98% line and 98% branch for the binding critical scope.  The manifest,
  denominator, and checker are fail-closed.
- Mutation evidence for the binding critical scope has a score of at least
  80% and zero unresolved mutants.  No exclusion, pragma, timeout
  reclassification, or threshold reduction may substitute for a killed mutant.
- The dedicated `veridist` release workflow, tag/version/artifact checks, and
  installed-artifact smoke contract pass.  The legacy workflow remains
  incapable of publication.
- Changelog, security policy, license and repository landing material are
  consistent with the release, and the published limitations are as visible as
  the supported cells.
- Final independent Sol Medium audits cover code/statistics, release safety,
  artifacts, localization, and public claims; all blocking findings are fixed
  and rerun on the exact tag candidate.

Inference, goodness-of-fit, model ranking, broad family catalogues,
conda-forge publication, Zenodo metadata, and claims of superiority are
explicitly deferred to later milestones or v1.  They are not backfilled into
0.5 by documentation or naming.

## Evidence

The release-safety RED contract reads the legacy workflow and fails if a
`release` trigger, `publish` job, PyPI publishing action, or `twine upload`
appears.  The corresponding green change removes the legacy release trigger
and publisher.  This proves only that the legacy path is disabled; it does not
prove a future publisher safe.

Evidence for each remaining gate is candidate-specific and must be retained
with its command, immutable source revision, environment, checker result, and
limits.  A passing result from another revision is historical context only.

## Test implications

- `REL-LEGACY-01`: legacy CI cannot trigger or publish a release artifact.
- `REL-BOUND-01`: active leases participate in the byte-bound invariant.
- `REL-CLEANUP-01`: each terminal execution path closes its owned resources.
- `REL-SCALE-01`: the million-row evidence checker rejects altered RSS,
  throughput, platform, workload, and limit facts.
- `REL-PORT-01`: Linux and Windows required lanes execute the declared
  portability contract.
- `REL-I18N-01`: EN/FA/DE rendered semantics and Persian RTL/LTR-isolate
  screenshot evidence cover prose, tables, code, and formulae.
- `REL-QUALITY-01`: global and critical line/branch coverage gates are
  fail-closed, as is mutation score with zero unresolved mutants.
- `REL-PUBLISH-01`: the future dedicated publisher rejects a nonmatching tag,
  version, metadata, payload, or installed-artifact namespace.

These are acceptance tests, not placeholders.  They are written before the
corresponding production or workflow changes and remain required on the exact
release candidate.

## Dependencies

ADR-0005, ADR-0006, ADR-0010, ADR-0011, ADR-0013, ADR-0015, ADR-0016,
ADR-0017, ADR-0019, and the deferred ADR-0021.  This record narrows the
release claim without superseding their individual behavioral contracts.

## Consequences

The next release cannot be rushed by creating a GitHub release or changing a
badge.  The temporary absence of any publishing workflow is intentional: it
prevents an artifact-name error while the new publisher is designed.  The 0.5
scope is narrower than the long-term ambition, but its claims can be tested
and falsified.  Feature breadth without the listed evidence is progress toward
later work, not eligibility for a 0.5 tag.

## Exit criteria and effort class

All scope bullets are binary release gates for the exact tagged revision, with
the retained evidence and final Sol Medium audit results available for review.
Until then, version `0.0.0.dev0` is mandatory.  Effort class: milestone;
estimated work spans release engineering, portability, measurement,
documentation, and adversarial quality review.
