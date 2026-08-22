# ADR-0015: Retry/checkpoint transactional guarantees

Status: Accepted

Owner: Ali Sadeghi Aghili

Decision scope: DS-08--DS-09

## Context

ADR-0005 requires retries to be idempotent and checkpoints to reject an
incompatible resume. Those requirements do not by themselves define the
transaction boundary between accumulator progress, checkpoint advancement and
an external side effect. Without that boundary, a failure between two writes
can either apply an update twice or acknowledge progress that was never
applied. Broad "exactly-once" language would hide this unresolved failure
surface.

The v1 design also needs bounded orchestration memory, a way to distinguish a
changed source from an intact checkpoint, and a test double that does not
accidentally become evidence for durability or security claims.

## Decision

Retry is prohibited for an arbitrary callback. A retried update is legal only
through one of these explicitly declared protocols:

1. A pure reducer produces a new immutable accumulator value, which is
   committed atomically with checkpoint advancement through compare-and-swap
   (CAS).
2. A durable `IdempotentSink` applies an operation through `apply_once`, keyed
   by a stable operation ID, and durably deduplicates the effect. The
   checkpoint may lag the sink. On retry, an `ALREADY_APPLIED` result for the
   same operation ID instructs the coordinator to reconcile by advancing the
   checkpoint through CAS. This protocol makes no atomicity claim across the
   sink and checkpoint stores.

For v1, at most one sequential update may be unresolved. The orchestrator
retains `O(1)` state with respect to input rows and chunk count: current
checkpoint identity, one pending operation identity, retry counters and
bounded integrity metadata. This is an orchestration-state claim only; it does
not imply that a selected statistical accumulator, source adapter or sink uses
constant memory.

Resume requires a private, stable source revision in addition to the source
identity, checkpoint schema/version and committed range. The revision is used
only for resume validation and must never be emitted in public provenance.
Public provenance records the permitted identity/hash-or-redaction fields
defined by ADR-0005.

SHA-256 may detect accidental checkpoint or payload corruption. It is an
integrity checksum, not authentication, authorization, freshness, secrecy or
proof against an active attacker. No security claim may be inferred from it.

The in-memory checkpoint store is a deterministic test double only. It makes
no durability, crash-recovery, cross-process, concurrency or security claim.
A persistent backend and any public durability/security/encryption claim
require a separate Accepted ADR defining the storage threat model, atomicity,
locking, authentication, key management, recovery and evidence requirements.

Checkpoint schemas are versioned and incompatible versions are rejected. No
automatic migration is performed. A migration facility, if introduced, must
be explicit, independently tested and governed by a later ADR.

The strongest permitted wording is conditional: an update is
"logical-once under the declared pure-reducer/CAS or durable-`apply_once`
protocol and its tested failure model." Broad "exactly-once" claims are
prohibited.

## Scope

This ADR governs retry admission, checkpoint commit order, resume validation,
in-memory checkpoint test doubles, and claim language for DS-08 and DS-09. It
does not select a persistent database, distributed-consensus algorithm,
encryption scheme, secret store, network protocol or automatic schema
migration system.

## Evidence

Decision evidence: Ali Sadeghi Aghili directly approved this decision on
2026-08-20.

Required evidence identifiers are:

- `DS-08-RETRY-PURE-CAS`: injected failures before, during and after atomic
  pure-reducer/checkpoint commit demonstrate no duplicate committed update.
- `DS-08-RETRY-SINK-APPLY-ONCE`: repeated stable operation IDs demonstrate one
  durable sink application within the declared backend failure model, and an
  `ALREADY_APPLIED` result reconciles a lagging checkpoint through CAS.
- `DS-08-ARBITRARY-CALLBACK-REJECTED`: retry admission rejects an unclassified
  callback before invoking it.
- `DS-08-INTEGRITY-FAILURE`: checksum mismatch has a distinct typed failure and
  carries no authentication claim.
- `DS-08-SOURCE-REVISION-MISMATCH`: a changed private source revision rejects
  resume before accumulator or sink mutation.
- `DS-09-COMPATIBLE-RESUME`: interruption at every supported commit boundary
  resumes to the declared canonical reduction result.
- `DS-09-INCOMPATIBLE-RESUME`: source identity/revision, checkpoint schema,
  accumulator schema and committed-range mismatches reject without reuse.
- `DS-09-ONE-UNRESOLVED-BOUND`: instrumentation demonstrates at most one
  unresolved sequential update and bounded orchestration state as input grows.
- `DS-09-PRIVATE-REVISION`: public provenance snapshots contain no private
  source revision.
- `DS-09-NO-AUTOMIGRATION`: an unknown checkpoint version is rejected without
  transformation or partial mutation.

No item above is implementation evidence until its executable test and retained
artifact exist. An in-memory run cannot satisfy persistent-backend evidence.

## Test implications

DS-08 and DS-09 tests must enumerate the transaction boundaries and inject a
failure on each side of them. They must distinguish checksum mismatch, source
mutation, incompatible schema/range, retry exhaustion and sink/reducer failure
with typed causes. Tests must prove callback rejection is preflight-only,
operation IDs remain stable across retries, CAS detects stale writers, public
provenance redacts the private revision, and incompatible resume performs no
write.

Property/state-machine tests must explore duplicate delivery and repeated
recovery. Scale-contract tests must measure orchestration state across growing
row and chunk counts while separately reporting accumulator and backend state.
Persistent durability, adversarial security and encryption tests remain out of
scope until their separate ADR is Accepted and a backend is selected.

## Dependencies

- ADR-0005: source identity, replayability, checkpoints, retry and provenance.
- ADR-0006: coverage, mutation, conformance and retained-evidence gates.
- ADR-0010: v1 reliability/big-data scope and release-claim constraints.
- A future Accepted persistent-backend/security ADR before any public
  durability, security or encryption claim.

## Consequences

Retry APIs are deliberately narrower than general callback APIs. A user who
needs retriable external effects must supply a conforming durable
`IdempotentSink`; otherwise execution fails before the callback is invoked.
Sequential processing limits throughput in v1 but makes the single unresolved
update boundary testable and keeps orchestration state bounded.

Private source revisions improve mutation detection but cannot appear in
public provenance, so internal resume records and public audit records require
separate schemas. Refusing automatic migration makes upgrades less convenient
but prevents silent reinterpretation of saved state. Durable and security
claims remain blocked until a separate ADR and backend evidence exist.

## Non-goals

- General retry safety for arbitrary or impure callbacks.
- Broad exactly-once delivery, execution or side-effect semantics.
- Parallel unresolved updates in v1.
- A claim that the full statistical computation is constant-memory.
- Authentication, authorization, encryption, tamper resistance or key
  management from SHA-256 checksums.
- Production durability from the in-memory checkpoint store.
- Automatic checkpoint or accumulator migration.

## Exit criteria and effort class

This ADR may be considered for acceptance once Ali Sadeghi Aghili approves the
transaction protocol and public vocabulary and confirms that this schema and
test plan are complete. Acceptance authorizes implementation; it does not
prove the capability. A public conditional logical-once claim additionally
requires all listed DS-08 and DS-09 evidence IDs to be executable and retained,
the v1 sequential bound to be measured, and every public surface to avoid broad
exactly-once wording. Persistent backend or security claims additionally
require their own Accepted ADR and are not an acceptance shortcut for this
record.

Effort/budget class: medium, one focused engineering wave for the in-memory
protocol and evidence; persistent backend and security work are separate,
unbudgeted waves.
