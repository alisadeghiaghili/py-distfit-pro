# Changelog

All notable changes to DistFit Pro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-12

### Added - Phase 1: Core Distribution Fitting

#### Distributions (30 total)
- **25 Continuous Distributions:**
  - Normal (Gaussian)
  - Lognormal
  - Weibull
  - Gamma
  - Exponential
  - Beta
  - Uniform
  - Triangular
  - Logistic
  - Gumbel (Extreme Value Type I)
  - Frechet (Extreme Value Type II)
  - Pareto
  - Cauchy
  - Student's t
  - Chi-squared
  - F-distribution
  - Rayleigh
  - Laplace (Double Exponential)
  - Inverse Gamma
  - Log-Logistic
  - Burr Type XII
  - Generalized Extreme Value (GEV)
  - Nakagami
  - Rice
  - Wald (Inverse Gaussian)

- **5 Discrete Distributions:**
  - Poisson
  - Binomial
  - Negative Binomial
  - Geometric
  - Hypergeometric

#### Fitting Methods
- Maximum Likelihood Estimation (MLE)
- Method of Moments
- Quantile Matching
- Automatic method selection with fallback

#### Model Selection
- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC)
- Log-likelihood calculation
- Automatic best distribution finder

#### Visualization
- PDF/PMF plots with data histogram
- CDF plots with empirical CDF
- Q-Q plots for goodness-of-fit assessment
- P-P plots
- Customizable matplotlib and plotly backends
- Interactive plots with plotly

#### Internationalization (i18n)
- English (EN) - complete
- Farsi/Persian (FA) - complete
- German (DE) - complete
- Easy language switching
- Translated distribution names and explanations

### Added - Phase 2: Advanced Statistical Features

#### Goodness-of-Fit Tests
- **Kolmogorov-Smirnov Test**
  - Most widely used GOF test
  - Measures maximum distance between empirical and theoretical CDFs
  - P-value calculation
  - Critical value tables

- **Anderson-Darling Test**
  - More sensitive to tail differences
  - Weighted squared distance measure
  - Better for extreme value distributions

- **Chi-Square Test**
  - Frequency-based comparison
  - Configurable number of bins
  - Works with both discrete and continuous data

- **Cramér-von Mises Test**
  - Uses squared differences over entire distribution
  - More sensitive to middle deviations than KS

- **Utilities:**
  - Run all tests at once
  - Summary table generation
  - Automatic interpretation
  - Configurable significance levels

#### Bootstrap Confidence Intervals
- **Parametric Bootstrap**
  - Samples from fitted distribution
  - Refits parameters on bootstrap samples
  - Percentile-based confidence intervals

- **Non-parametric Bootstrap**
  - Resamples from original data with replacement
  - More robust for model misspecification

- **BCa Bootstrap**
  - Bias-corrected and accelerated method
  - More accurate than simple percentile method
  - Jackknife-based acceleration

- **Performance:**
  - Parallel processing with joblib
  - Progress bars with tqdm
  - Customizable number of bootstrap samples
  - Random state control for reproducibility

#### Enhanced Diagnostics
- **Residual Analysis (4 types):**
  - Quantile residuals (randomized)
  - Pearson residuals
  - Deviance residuals
  - Standardized residuals

- **Influence Diagnostics:**
  - Cook's distance
  - Leverage values
  - DFFITS
  - Automatic identification of influential observations

- **Outlier Detection (4 methods):**
  - Z-score method (standard deviations)
  - IQR method (interquartile range)
  - Likelihood-based method
  - Mahalanobis distance

- **Diagnostic Plots:**
  - Q-Q plot data generation
  - P-P plot data generation
  - Worm plot (detrended Q-Q)
  - All with correlation metrics

#### Weighted Data Support
- **Weighted Fitting:**
  - Weighted Maximum Likelihood Estimation
  - Weighted Method of Moments
  - Support for 14+ distributions

- **Weighted Statistics:**
  - Weighted mean, variance, std
  - Weighted quantiles
  - Weighted median
  - Effective sample size (ESS) calculation

- **Use Cases:**
  - Survey data with sampling weights
  - Stratified sampling
  - Reliability data with different qualities
  - Aggregated data with frequency counts

### Added - Phase 3: Documentation & Testing

#### Documentation
- **Sphinx Documentation:**
  - Complete HTML documentation
  - Read the Docs theme
  - Auto-generated from docstrings

- **Tutorials (9 complete):**
  1. Basics - Introduction to distribution fitting
  2. Distributions - Guide to all 30 distributions
  3. Fitting Methods - MLE, Moments, Quantile
  4. GOF Tests - Complete testing guide
  5. Bootstrap - Confidence interval estimation
  6. Diagnostics - Residuals and influence
  7. Weighted Data - Working with weights
  8. Visualization - All plotting features
  9. Advanced - Complex workflows

- **Examples:**
  - Basic examples with real data
  - Advanced use cases
  - Real-world applications

- **README Files:**
  - English (comprehensive)
  - Farsi (complete translation)
  - German (complete translation)

## [Unreleased]

### Planned Features

#### Distribution Features
- Truncated distributions
- Mixture distributions
- Custom distribution support

#### Advanced Methods
- Bayesian parameter estimation
- Cross-validation for model selection
- Profile likelihood confidence intervals

#### Performance
- Numba JIT compilation for critical paths
- GPU acceleration option
- Caching for repeated computations

#### Visualization
- Animated distribution evolution
- 3D parameter space visualization
- Dashboard for model comparison

---

## Version History Summary

- **1.0.0** (2026-02-12) - Initial release
  - 30 distributions
  - 3 fitting methods
  - 4 GOF tests
  - Bootstrap CI
  - Complete diagnostics
  - Weighted data support
  - Multilingual (EN/FA/DE)
  - Full documentation
