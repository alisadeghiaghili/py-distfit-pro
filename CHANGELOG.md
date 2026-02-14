# Changelog

All notable changes to DistFit Pro will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

## [1.0.0] - 2026-02-14

### Phase 4: Comprehensive Examples & Documentation (Complete) 🎉

#### Added
- **Complete Examples Package** (`examples/`) with 20+ files across 7 folders:
  
  **01_basics/** (Introduction)
  - `basic_fitting.py`: Introduction to distribution fitting (400+ lines)
  - `common_distributions.py`: Overview of standard distributions (500+ lines)
  
  **02_advanced_fitting/** (Advanced Estimation)
  - `maximum_likelihood.py`: MLE with custom optimization (450+ lines)
  - `method_of_moments.py`: Method of Moments estimation (400+ lines)
  
  **03_model_selection/** (Choosing Best Distribution)
  - `aic_bic_comparison.py`: Information criteria comparison (550+ lines)
  - `cross_validation.py`: CV for distribution selection (500+ lines)
  - `hypothesis_testing.py`: Statistical hypothesis tests (450+ lines)
  
  **04_goodness_of_fit/** (Validation)
  - `ks_test.py`: Kolmogorov-Smirnov test (400+ lines)
  - `chi_square_test.py`: Chi-square test (450+ lines)
  - `anderson_darling.py`: Anderson-Darling test (400+ lines)
  
  **05_visualization/** (Beautiful Plots)
  - `pdf_cdf_plots.py`: Probability plots (500+ lines)
  - `qq_pp_plots.py`: Q-Q and P-P plots (450+ lines)
  - `interactive_plots.py`: Plotly dashboards (400+ lines)
  
  **06_real_world/** (Practical Applications)
  - `finance_analysis.py`: Risk analysis, VaR, portfolios (550+ lines)
  - `reliability_engineering.py`: Failure analysis, MTBF, Weibull (500+ lines)
  - `quality_control.py`: SPC, Cp/Cpk, Six Sigma (600+ lines)
  
  **07_advanced_topics/** (Expert Techniques)
  - `mixture_models.py`: Gaussian Mixture Models, EM algorithm (550+ lines)
  - `bootstrap_confidence.py`: Bootstrap CI, uncertainty quantification (500+ lines)
  - `custom_distributions.py`: Create custom distributions, KDE (550+ lines)

- **Comprehensive Examples README** (`examples/README.md`):
  - Quick start guide with code snippets
  - Complete table of contents
  - Structured learning path (beginner → advanced)
  - Real-world application guides
  - Examples organized by topic (tables)
  - Best practices section
  - Troubleshooting guide with solutions
  - Quick reference for common distributions
  - Key metrics and formulas
  - Running examples guide
  - Contributing guidelines
  - Additional resources and references

- **Package Integration** (`examples/__init__.py`):
  - Package structure documentation
  - Import convenience
  - Version tracking
  - Quick start code
  - Learning path guidance

#### Features
- **8,500+ lines** of production-ready example code
- **Beginner to advanced** coverage
- **Real-world scenarios** from finance, manufacturing, reliability
- **Publication-quality** visualizations
- **Best practices** and troubleshooting included
- **Complete documentation** with detailed explanations
- **Code quality**: Clean, well-commented, educational

#### Real-World Applications Covered
1. **Finance & Risk Management**:
   - Stock returns distribution analysis
   - Value at Risk (VaR) calculation (95%, 99%)
   - Portfolio optimization and efficient frontier
   - Fat-tail analysis with t-distribution

2. **Reliability Engineering**:
   - Weibull failure time analysis
   - Mean Time Between Failures (MTBF)
   - Hazard rate calculation
   - Maintenance scheduling (B10, B50 life)
   - System reliability (series vs parallel)

3. **Quality Control & Manufacturing**:
   - Process capability analysis (Cp, Cpk)
   - Statistical Process Control (SPC)
   - Control charts (X-bar, R charts)
   - Six Sigma methodology
   - Defect rate estimation

#### Advanced Techniques Covered
1. **Mixture Models**:
   - Gaussian Mixture Models (GMM)
   - EM algorithm implementation
   - Model selection (BIC/AIC for number of components)
   - Soft clustering and component assignment

2. **Bootstrap Methods**:
   - Non-parametric bootstrap for CI
   - Parametric bootstrap for distribution parameters
   - Uncertainty quantification
   - Parameter stability analysis

3. **Custom Distributions**:
   - Extending scipy.stats.rv_continuous
   - Kernel Density Estimation (KDE)
   - Truncated distributions
   - Domain-specific distributions

### Phase 3: Documentation & Testing (Complete)

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

## Project Statistics

- **Total Code Lines**: ~15,000+
- **Example Files**: 22
- **Distribution Classes**: 30
- **Fitting Methods**: 3 (MLE, MoM, Quantile)
- **Goodness-of-Fit Tests**: 4
- **Visualization Types**: 6+
- **Real-World Applications**: 3 domains
- **Advanced Techniques**: 3 (Mixture, Bootstrap, Custom)
- **Documentation Pages**: 50+
- **Code Quality**: Production-ready

## Contributors

- Ali Sadeghi Aghili (@alisadeghiaghili) - Creator & Maintainer

## License

MIT License - see LICENSE file for details.

---

## Acknowledgments

Special thanks to the scientific Python community for foundational libraries:
- NumPy & SciPy for numerical computing
- Matplotlib & Plotly for visualization
- scikit-learn for machine learning utilities
- Sphinx for documentation generation
