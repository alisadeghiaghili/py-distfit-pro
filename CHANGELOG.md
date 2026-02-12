# Changelog

All notable changes to DistFit Pro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.0.0] - 2026-02-12

### Phase 1: Core Distribution Fitting (Complete)

#### Added
- 30 statistical distributions (25 continuous + 5 discrete)
- Three parameter estimation methods: MLE, Method of Moments, Quantile Matching
- Complete statistical functions for all distributions:
  - PDF/PMF, CDF, PPF (inverse CDF)
  - Mean, variance, standard deviation
  - Median, mode, skewness, kurtosis
  - Quantiles and survival functions
- Model selection criteria: AIC, BIC, log-likelihood
- Visualization module with matplotlib and plotly:
  - PDF/PMF plots
  - CDF plots  
  - Q-Q plots
  - P-P plots
  - Histograms with fitted distributions
- Multilingual support (English, Farsi, German)
- Self-explanatory distribution classes with use cases
- Comprehensive summary() and explain() methods

### Phase 2: Advanced Statistical Testing (Complete)

#### Added
- **Goodness-of-Fit Tests Module** (`gof_tests.py`):
  - Kolmogorov-Smirnov (KS) test
  - Anderson-Darling (AD) test
  - Chi-Square (χ²) test
  - Cramér-von Mises (CvM) test
  - `run_all_tests()` convenience method
  - Summary table generator
  - Human-readable interpretations with p-values

- **Bootstrap Confidence Intervals** (`bootstrap.py`):
  - Parametric bootstrap (sample from fitted distribution)
  - Non-parametric bootstrap (resample from data)
  - Percentile-based CI
  - Bias-Corrected and Accelerated (BCa) bootstrap
  - Parallel processing with joblib
  - Progress bars with tqdm
  - `BootstrapResult` dataclass for clean output

- **Enhanced Diagnostics** (`diagnostics.py`):
  - Residual analysis (4 types):
    - Quantile residuals
    - Pearson residuals
    - Deviance residuals  
    - Standardized residuals
  - Influence diagnostics:
    - Cook's distance
    - Leverage values
    - DFFITS
  - Outlier detection (4 methods):
    - Z-score method
    - IQR (Interquartile Range)
    - Likelihood-based
    - Mahalanobis distance
  - Q-Q plot diagnostics
  - P-P plot diagnostics
  - Worm plot (detrended Q-Q)

- **Weighted Data Support** (`weighted.py`):
  - Weighted Maximum Likelihood Estimation
  - Weighted Method of Moments
  - Weighted quantile calculation
  - Weighted statistics (mean, variance, std, median)
  - Effective Sample Size (ESS) calculation
  - Support for 14 distributions with weighted fitting

### Phase 3: Documentation & Testing (In Progress)

#### Added
- **Sphinx Documentation**:
  - Installation guide
  - Quick start guide
  - 9 comprehensive tutorials:
    1. Basics - Introduction to distribution fitting
    2. Distributions - Guide to all 30 distributions
    3. Fitting Methods - MLE, Moments, Quantile
    4. GOF Tests - Statistical testing
    5. Bootstrap - Confidence intervals
    6. Diagnostics - Residuals and outliers
    7. Weighted Data - Weighted fitting
    8. Visualization - Plotting guide
    9. Advanced - Complex workflows
  - Real-world examples (customer wait times, defects, etc.)
  - Detailed README with usage examples

#### To Be Added
- Unit test suite with pytest
- API reference documentation (auto-generated)
- User guides (6 chapters)
- Advanced examples
- FAQ section
- Contributing guidelines

## Distribution List

### Continuous Distributions (25)
1. Normal (Gaussian)
2. Lognormal
3. Weibull
4. Gamma
5. Exponential
6. Beta
7. Uniform
8. Triangular
9. Logistic
10. Gumbel
11. Fréchet
12. Pareto
13. Cauchy
14. Student's t
15. Chi-squared
16. F-distribution
17. Rayleigh
18. Laplace
19. Inverse Gamma
20. Log-Logistic
21. Inverse Gaussian
22. Generalized Extreme Value (GEV)
23. Generalized Pareto (GPD)
24. Nakagami
25. Rice

### Discrete Distributions (5)
1. Poisson
2. Binomial
3. Negative Binomial
4. Geometric  
5. Hypergeometric

## Contributors

- Ali Sadeghi Aghili (@alisadeghiaghili) - Creator & Maintainer

## License

MIT License - see LICENSE file for details.
