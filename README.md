# DistFit Pro

**Professional Distribution Fitting for Python**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)](https://github.com/alisadeghiaghili/py-distfit-pro)

DistFit Pro is a comprehensive Python library for statistical distribution fitting with advanced features for data analysis, reliability engineering, quality control, and risk assessment.

## 🌟 Features

### Core Capabilities

- **30 Statistical Distributions**
  - 25 Continuous: Normal, Lognormal, Weibull, Gamma, Exponential, Beta, and more
  - 5 Discrete: Poisson, Binomial, Negative Binomial, Geometric, Hypergeometric

- **Multiple Estimation Methods**
  - Maximum Likelihood Estimation (MLE)
  - Method of Moments
  - Quantile Matching

- **Goodness-of-Fit Tests**
  - Kolmogorov-Smirnov (KS)
  - Anderson-Darling (AD)
  - Chi-Square (χ²)
  - Cramér-von Mises (CvM)

- **Bootstrap Confidence Intervals**
  - Parametric bootstrap
  - Non-parametric bootstrap
  - BCa (Bias-Corrected and Accelerated) method
  - Parallel processing support

- **Enhanced Diagnostics**
  - Residual analysis (4 types)
  - Influence diagnostics (Cook's D, leverage, DFFITS)
  - Outlier detection (4 methods)
  - Q-Q, P-P, and Worm plots

- **Weighted Data Support**
  - Weighted MLE and moments
  - Weighted statistics
  - Effective sample size calculation

- **Multilingual Support**
  - English, Farsi (فارسی), German (Deutsch)

## 🚀 Quick Start

### Installation

```bash
pip install distfit-pro
```

### Basic Usage

```python
from distfit_pro import get_distribution
import numpy as np

# Generate sample data
data = np.random.normal(10, 2, 1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data, method='mle')

# View summary
print(dist.summary())
```

### Goodness-of-Fit Testing

```python
from distfit_pro.core.gof_tests import GOFTests

# Run all GOF tests
results = GOFTests.run_all_tests(data, dist)
print(GOFTests.summary_table(results))
```

### Bootstrap Confidence Intervals

```python
from distfit_pro.core.bootstrap import Bootstrap

# Parametric bootstrap with parallel processing
ci_results = Bootstrap.parametric(
    data, 
    dist, 
    n_bootstrap=1000,
    n_jobs=-1  # Use all CPU cores
)

for param, result in ci_results.items():
    print(result)
```

### Diagnostics

```python
from distfit_pro.core.diagnostics import Diagnostics

# Residual analysis
residuals = Diagnostics.residual_analysis(data, dist)
print(residuals.summary())

# Detect outliers
outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
print(outliers.summary())
```

### Weighted Fitting

```python
from distfit_pro.core.weighted import WeightedFitting

# Data with different reliabilities
weights = np.random.uniform(0.5, 1.5, len(data))

# Weighted MLE
params = WeightedFitting.fit_weighted_mle(data, weights, dist)
dist.params = params
dist.fitted = True
```

## 📚 Documentation

Comprehensive documentation available at: [Read the Docs](https://distfit-pro.readthedocs.io)

- **Tutorials**: Step-by-step guides from basics to advanced
- **API Reference**: Complete function and class documentation
- **Examples**: Real-world applications
- **User Guide**: Detailed feature explanations

## 📊 Available Distributions

### Continuous Distributions

| Distribution | Use Case | Parameters |
|-------------|----------|------------|
| Normal | Symmetric data, measurement errors | μ (mean), σ (std) |
| Lognormal | Positive, right-skewed data (income, file sizes) | s (shape), scale |
| Weibull | Reliability, time-to-failure | c (shape), scale |
| Gamma | Waiting times, sum of exponentials | a (shape), scale |
| Exponential | Time between events (memoryless) | scale |
| Beta | Proportions, probabilities [0,1] | a, b (shapes) |
| Uniform | Equal probability | loc, scale |
| Triangular | Three-point estimates | c (mode), loc, scale |
| Logistic | Growth models | loc, scale |
| Gumbel | Extreme values (maximum) | loc, scale |
| Pareto | Power law, 80-20 rule | b (shape), scale |
| Student's t | Heavy tails, small samples | df (degrees of freedom) |
| Chi-squared | Variance tests | df |
| F | Variance ratio | dfn, dfd |
| Rayleigh | Signal processing | scale |
| Laplace | Sparse data | loc, scale |
| Cauchy | Undefined mean/variance | loc, scale |

### Discrete Distributions

| Distribution | Use Case | Parameters |
|-------------|----------|------------|
| Poisson | Rare event counts | μ (rate) |
| Binomial | n independent trials | n, p |
| Negative Binomial | Overdispersed counts | n, p |
| Geometric | Trials to first success | p |
| Hypergeometric | Sampling without replacement | M, n, N |

## 🔧 Advanced Features

### Model Selection

```python
# Compare multiple distributions
candidates = ['normal', 'lognormal', 'gamma', 'weibull']

for dist_name in candidates:
    dist = get_distribution(dist_name)
    dist.fit(data)
    
    # Calculate AIC
    n = len(data)
    k = len(dist.params)
    log_lik = np.sum(dist.logpdf(data))
    aic = 2 * k - 2 * log_lik
    
    print(f"{dist_name}: AIC = {aic:.2f}")
```

### Reliability Analysis

```python
# Weibull reliability
dist = get_distribution('weibull')
dist.fit(lifetime_data)

# Reliability at time t
R_t = dist.reliability(1000)  # hours
print(f"Reliability at 1000h: {R_t:.2%}")

# Hazard rate
h_t = dist.hazard_rate(1000)
print(f"Hazard rate: {h_t:.6f}")

# MTTF
mttf = dist.mean_time_to_failure()
print(f"MTTF: {mttf:.1f} hours")
```

### Risk Metrics

```python
# Value at Risk (VaR)
var_95 = dist.ppf(0.95)
print(f"VaR (95%): {var_95:.2f}")

# Conditional VaR (Expected Shortfall)
cvar_95 = dist.conditional_var(0.95)
print(f"CVaR (95%): {cvar_95:.2f}")
```

## 🎯 Real-World Applications

### Quality Control

```python
# Statistical process control
dist = get_distribution('normal')
dist.fit(baseline_measurements)

ucl = dist.mean() + 3 * dist.std()  # Upper control limit
lcl = dist.mean() - 3 * dist.std()  # Lower control limit

out_of_control = (new_measurements > ucl) | (new_measurements < lcl)
```

### A/B Testing

```python
# Bayesian A/B test with Beta distributions
from scipy.stats import beta

# Posterior distributions
samples_A = beta.rvs(successes_A + 1, failures_A + 1, size=10000)
samples_B = beta.rvs(successes_B + 1, failures_B + 1, size=10000)

# Probability B > A
prob_B_better = np.mean(samples_B > samples_A)
print(f"P(B > A): {prob_B_better:.1%}")
```

### Insurance/Finance

```python
# Fit claim amounts
dist = get_distribution('lognormal')
dist.fit(claim_amounts)

# Risk assessment
var_99 = dist.ppf(0.99)  # 99th percentile
expected_loss = dist.conditional_var(0.95)
```

## 🛠️ Development

### Requirements

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.3
- Plotly >= 5.0
- joblib >= 1.0
- tqdm >= 4.60

### Installation from Source

```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e .
```

### Running Tests

```bash
pytest tests/
```

## 📝 Citation

If you use DistFit Pro in your research, please cite:

```bibtex
@software{distfitpro2026,
  author = {Sadeghi Aghili, Ali},
  title = {DistFit Pro: Professional Distribution Fitting for Python},
  year = {2026},
  url = {https://github.com/alisadeghiaghili/py-distfit-pro},
  version = {1.0.0}
}
```

## 🔗 Links

- **GitHub**: [https://github.com/alisadeghiaghili/py-distfit-pro](https://github.com/alisadeghiaghili/py-distfit-pro)
- **Documentation**: [https://distfit-pro.readthedocs.io](https://distfit-pro.readthedocs.io)
- **PyPI**: [https://pypi.org/project/distfit-pro](https://pypi.org/project/distfit-pro)
- **Issues**: [https://github.com/alisadeghiaghili/py-distfit-pro/issues](https://github.com/alisadeghiaghili/py-distfit-pro/issues)

## 👥 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

MIT License - see [LICENSE](LICENSE) for details.

## ✨ What's New in v1.0.0

### Major Features

- ✅ **30 Statistical Distributions** with self-explanatory behavior
- ✅ **3 Estimation Methods**: MLE, Moments, Quantile
- ✅ **4 GOF Tests**: KS, AD, Chi-Square, CvM
- ✅ **Bootstrap CI**: Parametric, non-parametric, BCa
- ✅ **Enhanced Diagnostics**: Residuals, influence, outliers
- ✅ **Weighted Data Support**: MLE and moments for weighted observations
- ✅ **Multilingual**: English, Farsi, German
- ✅ **Parallel Processing**: Fast bootstrap with joblib

### Performance

- Optimized numerical algorithms
- Parallel bootstrap (5-10x speedup)
- Efficient weighted fitting

### Documentation

- 9 comprehensive tutorials
- Complete API reference
- Real-world examples
- Publication-quality figures

## 🚀 Roadmap

### v1.1.0 (Planned)

- [ ] Mixture distributions
- [ ] Truncated distributions
- [ ] Copulas
- [ ] More diagnostics

### v1.2.0 (Future)

- [ ] Time-varying parameters
- [ ] Bayesian inference
- [ ] Interactive dashboards

## 💬 Contact

**Ali Sadeghi Aghili**

- GitHub: [@alisadeghiaghili](https://github.com/alisadeghiaghili)
- Website: [zil.ink/thedatascientist](https://zil.ink/thedatascientist)
- LinkTree: [linktr.ee/aliaghili](https://linktr.ee/aliaghili)

## 🔥 Acknowledgments

- Built on top of NumPy and SciPy
- Inspired by fitdistrplus (R) and reliability (Python)
- Thanks to the open-source community

---

**Made with ❤️ by Ali Sadeghi Aghili**

*"Make data analysis simple, powerful, and accessible to everyone."*
