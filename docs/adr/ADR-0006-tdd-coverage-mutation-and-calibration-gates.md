# ADR-0006: TDD, coverage, mutation and calibration gates

Status: Accepted

Decision evidence: direct Ali Sadeghi decision, 2026-08-20, confirms the v1
global 95% line-and-branch coverage requirement. The stricter critical-module,
per-file and mutation gates remain part of this accepted release-control
decision.

## Decision

Use test-first development, global >=95% line **and** branch coverage for v1
code, >=98% of both in the critical statistical core (numerical production
paths in `domain`, `statistics`, `families` and `engine`), and no production
file below 90% line or branch without an accepted exception ADR. Mutation score
is >=80% for that same critical statistical core, defined as killed / (killed +
survived) for eligible mutants; timeout, harness error, unclassified result or
missing module report fails rather than disappears. Conformance/reference tests
run on each PR. Statistical calibration, DataSource scale-contract coverage
and scale tests run nightly and are release evidence with retained artifacts.

## Consequences

The planned checker is **NOT IMPLEMENTED**. When implemented, it must fail
global line <95%, global branch <95%, a critical-path line/branch <98%, or any
production-file line/branch <90% absent an accepted exact-path exception ADR.
Coverage exclusions require exact paths, owner, expiry and rationale; pragma,
generated-code, import-only execution, test-only helpers and denominator
manipulation cannot game either gate. Coverage is not treated as correctness
proof.
Mutation eligibility must enumerate all mutation points in those four critical
path groups. Its planned checker fails score <80%, timeout, harness error,
unclassified mutant, unapproved exclusion, or missing module report; no runner
or command is currently configured.
Monte-Carlo gates report replicate count, seed and uncertainty; a failing
calibration changes procedure/claim rather than being hidden by a looser
tolerance or placeholder skip.
