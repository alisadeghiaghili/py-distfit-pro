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

## Implementation evidence addendum -- 2026-08-22

The deterministic coverage checker is now implemented. On 2026-08-22, local
generic discovery executed 156 tests and the checker accepted 14 enumerated
production files with 100% observed coverage against frozen denominators of
1,293 statements and 444 branches, no accepted exceptions, and the binding
95/98/90 thresholds intact. CI configuration for
Ruff, strict source type checking, Python 3.11--3.14 tests, coverage, package
inspection and documentation checks is present in `e23cda5` through `89202c6`
and pushed; its remote result remains **UNVERIFIED**.

## Formal mutation implementation addendum -- 2026-09-01

Formal mutation infrastructure is implemented, but it is **NOT YET EXECUTED**
on GitHub Linux and therefore establishes no mutation score or release pass.
Its binding scope is every Python file in all four critical production
directories: `src/veridist/domain`, `statistics`, `families`, and `engine`.
There are no hand-picked mutation sites, `pragma: no mutate`, `do_not_mutate`,
or exclusion patterns. The fail-closed schema/checker binds the checked-out
commit and source-tree digest, the pinned `mutmut==3.7.0` configuration, the
curated behavioral mutation selection (`tests/contract`, `tests/reference`,
and `tests/unit`), baseline result, environment, every mutant identity and
per-file/per-module totals. It rejects missing files/modules, duplicate IDs,
booleans disguised as counts, source/config drift, empty scored denominator,
score below 80%, and suspicious, timeout, error, or unclassified results.

Mutmut 3.7.0 generates function/method mutants, not module-level executable
code. Module-level lines remain in the coverage and contract-test gates but
are honestly outside this tool-generated mutant denominator. A critical
executable file with zero generated function/method mutants must still appear
in evidence with `generated=0` and an empty identity list; it cannot be
silently omitted. If every executable critical file has zero mutants, the
checker rejects the evidence because there is no scored denominator. Native
Windows is deliberately refused: mutmut 3.7.0 requires POSIX/fork. The
versioned GitHub Linux workflow runs the complete `tests` baseline, then the
curated behavioral oracle selection for each mutation run, export and gate
export and gate on Ubuntu/Python 3.13 after verifying the pinned PyPI wheel
SHA-256. It retains evidence, both logs and raw cache metadata even on failure,
but it has not yet produced a remote result or score.
Temporary targeted probes remain diagnostic only. Statistical calibration and
retained production-scale evidence remain unimplemented.
