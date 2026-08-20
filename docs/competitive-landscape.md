# Competitive landscape: distribution fitting census

**Status: Internal Draft / not publishable.** Source URLs and capability cells
remain working research notes until the evidence-freezing policy's required
claim-cell locks reach 100% coverage.

**Census boundary:** 2026-08-19. This is a deliberately bounded, source-led
feature census of general univariate fitting, reliability/survival fitting,
extremes and adjacent platforms. It is not exhaustive and a vendor statement is
not independent statistical validation.

## Evidence rules

- CSV statuses are `supported`, `partial`, `not_supported`, `unverified` and
  `not_applicable`; evidence quality is `primary`, `official_secondary`,
  `paper` or `unverified`. These terms are defined by the CSV schema, not by
  marketing language.
- A versioned/tagged URL is preferred. Every row records `cited_version` and
  `retrieved_at`; publication is blocked until the claim-cell locks required by
  [competitive-evidence-policy.md](competitive-evidence-policy.md) exist. No
  snapshot or hash has yet been captured for this draft.
- `GoF-refit` means that the null simulation re-fits parameters for every
  simulated sample; printing KS/AD/CvM alone does not meet this definition.

The machine-readable long-form evidence is in
[competitive-feature-matrix.csv](competitive-feature-matrix.csv); the grouped
human summary is [competitive-feature-matrix.md](competitive-feature-matrix.md).

## What the evidence says

### There is no empty market

Current SciPy documentation retrieved 2026-08-20 documents MLE/MM fitting, left/right/interval-censored continuous
MLE through `CensoredData`, and refit parametric Monte-Carlo GoF for finite
uncensored data. It also represents mixtures, but its manual does not establish
generic mixture fitting in that API. [fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.fit.html),
[CensoredData](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.CensoredData.html),
[GoF](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html),
[Mixture](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Mixture.html)

R's `fitdistrplus` is a major general-fitting reference for custom distributions,
MLE/MME/QME/MGE, weighted non-censored fits, diagnostics and bootstrap; exact
MSE availability must be cited per package version rather than inferred; it
supports left/right/interval-censored MLE. Its own 2026 FAQ says `gofstat` is
not available for censored fits, a genuine but narrow opportunity.
[vignette](https://lbbe-software.github.io/fitdistrplus/articles/fitdistrplus_vignette.html),
[FAQ](https://lbbe-software.github.io/fitdistrplus/articles/FAQ.html)

SurPyval, MATLAB, SAS PROC SEVERITY and JMP Life Distribution document subsets
of censoring/truncation, mixtures, covariates, diagnostics and GUI workflows.
Coverage is heterogeneous and must not be summarized as category-wide parity. SAS
PROC SEVERITY documents custom continuous distributions, censored/truncated
likelihoods, BY groups, scale covariates, selection, multithreading and
scoring. [SurPyval](https://surpyval.readthedocs.io/),
[MATLAB](https://www.mathworks.com/help/stats/fitdist.html),
[SAS](https://support.sas.com/documentation/cdl/en/etsug/68148/HTML/default/etsug_severity_overview.htm),
[JMP](https://www.jmp.com/support/help/en/19.0/jmp/life-distribution.shtml)

### Honest GoF and reproducible scale are plausible wedges

The census found a documented refit Monte-Carlo GoF implementation in SciPy; it
did not find public per-family calibration tables, failure-rate accounting,
decision thresholds, or an out-of-core correctness contract among direct
fitters examined. That is an opportunity only after veridist publishes its own
evidence. "Nobody has this" is not an admissible claim.

Julia `Distributions.jl` exposes `suffstats`, weighted `fit_mle` and a defined
family list, but documented `fit` takes an array rather than advertising an
external-memory backend. [Distributions.jl (generated 2026-06-26)](https://juliastats.org/Distributions.jl/latest/fit/)

### L-moments and extremes are capability clusters

`lmomco` is a mature R ecosystem. Python `lmoments3` 1.0.8 also computes sample
L-moments and fits selected distributions. Thus "no Python equivalent" is
false. The defensible target is audited, scalable L-moment and extreme-value
workflows with explicit approximation errors, not a first-ever implementation.
[lmomco](https://cran.r-project.org/web/packages/lmomco/citation.html),
[lmoments3 1.0.8](https://pypi.org/project/lmoments3/1.0.8/)

### Strategic conclusions

1. Do not pursue catalogue parity first. Wrapping a catalogue without
   support/estimator/GoF/scale evidence recreates silent capability ambiguity.
2. Own a narrow trust contract first: family × data × estimator matrix,
   calibration grid, failure accounting and reproducible streaming benchmark.
3. Treat reliability as a later serious product line. SurPyval/SAS/JMP/MATLAB
   parity requires domain expertise and reference validation, not API stubs.
4. Integrate before reimplementing. Ingest pandas/Polars/Arrow/Dask only when
   the source contract, provenance and failure tests exist.
5. Reputation is a technical deliverable; see
   [adoption-and-reputation-strategy.md](adoption-and-reputation-strategy.md).

## Deliberate research gaps

SPSS, Stata, Java, .NET, C++, EasyFit, ExpertFit, Stat::Fit, Arena Input
Analyzer, @RISK/BestFit, Crystal Ball, Statgraphics and XLSTAT are represented
only where this census found a primary source. Several current generic
fitting/censoring/GoF semantics remain **Unverified**. Before a release-note or
paper comparison, audit exact product version, license tier and workflow.
