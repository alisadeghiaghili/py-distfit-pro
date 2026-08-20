# Adoption and reputation strategy

Trust is the product. A package with many features but uncalibrated inference
will not earn durable references, industrial use or contributors.

## What we measure--and what we do not pretend to measure

| Funnel stage | Measurable signal | Guardrail |
| --- | --- | --- |
| Discovery | indexed EN/FA/DE pages, broken-link rate, search landing conversion | no fabricated stars/download claims |
| First fit | clean-install time-to-first-fit and example pass rate | not only happy-path notebooks |
| Trust | calibration/benchmark artifact freshness and reproducibility pass rate | no unscoped "accurate" claim |
| Retention | migration-guide feedback and repeat issue categories | no telemetry collection in v1 |
| Community | reviewed external PRs, contributor-ladder progression, citations/use cases | popularity is not validation |

Targets are set after a baseline release; this document intentionally has no
invented download, star or conversion target.

## KPI dictionary and governance

| KPI | Formula | Owner role | Cadence | Privacy / target |
| --- | --- | --- | --- | --- |
| Executable-example evidence | passing public examples / registered public examples | docs lead | each PR | 100% engineering acceptance target; no user telemetry |
| Documentation-link evidence | passing internal+external checked links / registered checked links | docs lead | each PR/nightly | 100% engineering acceptance target; network failures classified, not hidden |
| Reproducibility freshness | required benchmark/calibration artifacts within published refresh window / required artifacts | statistical lead | release | baseline first, then target set by ADR |
| Time-to-first-fit | median observed task time in scripted usability study from clean install to valid report | product lead | alpha/beta | prospective study with consent; target set after baseline |
| Migration completion | participants completing stated migration task / enrolled consented participants | product lead | beta/release | prospective; no telemetry substitute |
| Triage quality | issues receiving label/owner within published policy window / eligible new issues | maintainer | monthly | public tracker data only |
| Community contribution | merged external PRs with review record / eligible external PRs | maintainer | release | report context, never as quality proxy |

There is **no telemetry in v1**: no collection, upload, auto-reporting or
analytics beacon. Any post-v1 telemetry proposal requires a separate accepted
privacy/security ADR; it is not implied by this strategy.

## Product routes to adoption

1. **Time to first trustworthy fit:** one installation command, a 10-line
   example, visible support errors and migration from `scipy.stats`.
2. **Interoperability earned by tests:** begin with NumPy/pandas; add
   Polars/Arrow/DuckDB/Dask only when adapters preserve streaming, provenance
   and failure semantics. Add scikit-learn only with a real estimator contract.
   Publish Colab/notebooks only if CI executes them.
3. **Discoverable multilingual knowledge:** EN/FA/DE docs, stable URLs,
   locale-aware search metadata, RTL screenshots and a shared glossary.
4. **Reputation artifacts:** PyPI and conda-forge require reproducible build
   evidence; `CITATION.cff` and Zenodo DOI require versioned release metadata;
   JOSS requires a documented, tested research-software release; a methods or
   calibration paper requires a public replication package and stable evidence.
5. **Open maintenance:** support/security/release policy, changelog,
   deprecation window, issue templates and a contributor/reviewer ladder.
   v1 collects no telemetry and raw data never leave the process.
6. **Proposed community channel:** submit PyData and relevant community-talk
   proposals only when a talk can show a reproducible calibration, failure or
   migration artifact. A talk is outreach, not evidence of adoption or
   correctness, and carries no promised audience/outcome target.

## Phased launch

| Phase | Deliverable | Reputation test |
| --- | --- | --- |
| Alpha | narrow matrix, SciPy/fitter/distfit migration notes, executable notebooks, limitations | outsider reproduces one fit and one failure |
| Beta | calibration table, goldens, streaming benchmark, issue templates | outsider can dispute/re-run each headline claim |
| 1.0 | reproducible PyPI artifact, conda-forge readiness, valid `CITATION.cff`, Zenodo DOI metadata, EN/FA/DE docs, security/release policy | each release channel gate and evidence artifact passes |
| Post-1.0 | fitdistrplus guide, domain cases, JOSS/methods submission | adoption never outruns capability matrix |

## Non-negotiable anti-patterns

- No star/download comparison without source, time window and definition.
- No notebook that CI has not run; no benchmark without code/data/version.
- No hidden telemetry, customer-data upload or auto-reporting.
- No integration badge before reference and scale tests.
- No compatibility promise before an explicit migration/error map exists.
