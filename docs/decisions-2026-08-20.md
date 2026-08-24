# Decisions recorded 2026-08-20

This record translates direct Ali Sadeghi decisions into ADR status changes. It
is decision evidence, not runtime evidence that any capability is implemented
or release-ready.

| Direct decision | ADR effect | Operational consequence |
| --- | --- | --- |
| Python-first; R deferred | ADR-0001 Accepted; ADR-0010 and ADR-0012 Accepted | v1 is Python-only. R/CRAN comes after Python capability/reference evidence. |
| First domain vertical: reliability + censoring + big-data | ADR-0005, ADR-0010 and ADR-0012 Accepted | Roadmap and release evidence prioritize bounded reliability-family censoring and DataSource scale semantics before catalogue breadth. |
| Sphinx + MyST + gettext/`sphinx-intl` | ADR-0013 Accepted; supports ADR-0011 Accepted | EN/FA/DE builds, parity, link checking and RTL QA are binding release work; configuration was not implemented at this decision time. See ADR-0013's dated implementation addenda for later evidence. |
| Reject multipass work on `single_pass`; spool only by opt-in | ADR-0005 Accepted | Planner rejection and declared spool budget/retention/cleanup are binding contract requirements. |
| 95% coverage requirement | ADR-0006 Accepted | Global line and branch coverage >=95% is binding; the stricter documented critical/per-file/mutation gates are retained. |
| No telemetry in v1 | ADR-0010 Accepted | No collection, upload, auto-reporting or analytics beacon ships in v1. A future proposal needs a separate privacy/security ADR. |
| PyPI + conda-forge + `CITATION.cff` + Zenodo DOI as release gates | ADR-0010 Accepted | A v1 release requires reproducible PyPI artifact, conda-forge publication readiness, valid citation metadata and DOI release metadata. |

## Status outcome

Accepted: ADR-0001, ADR-0005, ADR-0006, ADR-0010, ADR-0011, ADR-0012 and
ADR-0013.

Proposed: ADR-0002, ADR-0003, ADR-0004, ADR-0007, ADR-0008, ADR-0009 and
ADR-0014. ADR-0014 remains Proposed because source-lock storage,
archive-retention/privacy, access control and publication workflow were not
decided.
