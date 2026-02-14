# DistFit-Pro Examples

<div align="center">

**Comprehensive Guide to Distribution Fitting in Python**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[Quick Start](#quick-start) • [Examples](#examples-overview) • [Learning Path](#learning-path) • [Real-World Applications](#real-world-applications)

</div>

---

## 📚 Table of Contents

- [Quick Start](#quick-start)
- [Examples Overview](#examples-overview)
- [Learning Path](#learning-path)
- [Installation](#installation)
- [Examples by Topic](#examples-by-topic)
- [Real-World Applications](#real-world-applications)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🚀 Quick Start

### Basic Distribution Fitting

```python
from distfit_pro import get_distribution
import numpy as np

# Generate sample data
data = np.random.normal(100, 15, 1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data)

# Get statistics
print(f"Mean: {dist.mean():.2f}")
print(f"Std: {dist.std():.2f}")
print(f"AIC: {dist.aic():.2f}")
```

### Finding Best Distribution

```python
from distfit_pro import find_best_distribution

# Try multiple distributions
candidates = ['normal', 'lognormal', 'gamma', 'weibull_min']
best = find_best_distribution(data, candidates)

print(f"Best distribution: {best.name}")
print(f"AIC: {best.aic():.2f}")
```

---

## 📖 Examples Overview

This repository contains **20+ comprehensive examples** organized into 7 categories:

### 📁 Folder Structure

```
examples/
├── 01_basics/                    # Start here!
│   ├── basic_fitting.py
│   └── common_distributions.py
├── 02_advanced_fitting/          # Advanced estimation
│   ├── maximum_likelihood.py
│   └── method_of_moments.py
├── 03_model_selection/           # Choose best distribution
│   ├── aic_bic_comparison.py
│   ├── cross_validation.py
│   └── hypothesis_testing.py
├── 04_goodness_of_fit/           # Validate fits
│   ├── ks_test.py
│   ├── chi_square_test.py
│   └── anderson_darling.py
├── 05_visualization/             # Beautiful plots
│   ├── pdf_cdf_plots.py
│   ├── qq_pp_plots.py
│   └── interactive_plots.py
├── 06_real_world/               # Practical applications
│   ├── finance_analysis.py
│   ├── reliability_engineering.py
│   └── quality_control.py
└── 07_advanced_topics/          # Expert techniques
    ├── mixture_models.py
    ├── bootstrap_confidence.py
    └── custom_distributions.py
```

---

## 🎓 Learning Path

### Beginner (Start Here!)

1. **01_basics/basic_fitting.py** ⭐
   - First steps with distribution fitting
   - Understanding parameters
   - Simple visualizations

2. **01_basics/common_distributions.py**
   - Normal, Exponential, Gamma, Weibull
   - When to use each distribution
   - Parameter interpretation

3. **05_visualization/pdf_cdf_plots.py**
   - Visualize fitted distributions
   - Compare multiple fits
   - Publication-quality plots

### Intermediate

4. **03_model_selection/aic_bic_comparison.py**
   - AIC vs BIC
   - Model comparison
   - Avoiding overfitting

5. **04_goodness_of_fit/ks_test.py**
   - Validate your fits
   - Kolmogorov-Smirnov test
   - Statistical significance

6. **05_visualization/qq_pp_plots.py**
   - Q-Q plots for diagnostics
   - Identify distribution issues
   - Tail behavior analysis

### Advanced

7. **06_real_world/** (Choose your domain)
   - **finance_analysis.py**: Risk, VaR, portfolios
   - **reliability_engineering.py**: Failure analysis, MTBF
   - **quality_control.py**: SPC, Cp/Cpk

8. **07_advanced_topics/mixture_models.py**
   - Gaussian mixture models
   - Multiple populations
   - EM algorithm

9. **07_advanced_topics/bootstrap_confidence.py**
   - Uncertainty quantification
   - Confidence intervals
   - Parameter stability

10. **07_advanced_topics/custom_distributions.py**
    - Create your own distributions
    - Kernel Density Estimation
    - Truncated distributions

---

## 💼 Real-World Applications

### Finance & Risk Management

**File:** `06_real_world/finance_analysis.py`

```python
# Value at Risk (VaR) calculation
returns = load_stock_returns()
dist = get_distribution('t')  # Fat-tailed
dist.fit(returns)

var_95 = dist.ppf(0.05)  # 95% VaR
print(f"Maximum expected loss: {var_95*100:.2f}%")
```

**Use Cases:**
- Portfolio risk assessment
- VaR calculation
- Stress testing
- Option pricing

### Manufacturing & Quality Control

**File:** `06_real_world/quality_control.py`

```python
# Process capability analysis
measurements = load_process_data()
dist = get_distribution('normal')
dist.fit(measurements)

# Calculate Cpk
USL, LSL = 10.5, 9.5
Cpk = calculate_cpk(dist, USL, LSL)
print(f"Process capability: Cpk = {Cpk:.3f}")
```

**Use Cases:**
- Process capability (Cp/Cpk)
- Control charts
- Six Sigma analysis
- Defect rate estimation

### Reliability Engineering

**File:** `06_real_world/reliability_engineering.py`

```python
# Weibull failure analysis
failure_times = load_failure_data()
dist = get_distribution('weibull_min')
dist.fit(failure_times)

# Mean Time Between Failures
mtbf = dist.mean()
print(f"MTBF: {mtbf:.0f} hours")

# Reliability at time t
R_1000 = dist.sf(1000)  # Survival function
print(f"Reliability at 1000h: {R_1000:.3f}")
```

**Use Cases:**
- Failure time analysis
- Maintenance scheduling
- Reliability prediction
- Warranty analysis

---

## 📊 Examples by Topic

### Distribution Fitting Methods

| Method | File | Complexity | Use When |
|--------|------|------------|----------|
| Maximum Likelihood | `02_advanced_fitting/maximum_likelihood.py` | ⭐⭐⭐ | Standard approach, works well |
| Method of Moments | `02_advanced_fitting/method_of_moments.py` | ⭐⭐ | Fast, simple parameters |
| Kernel Density | `07_advanced_topics/custom_distributions.py` | ⭐⭐⭐ | Non-parametric, complex data |

### Model Selection Criteria

| Criterion | File | Pros | Cons |
|-----------|------|------|------|
| AIC | `03_model_selection/aic_bic_comparison.py` | Balances fit & complexity | Can overfit |
| BIC | `03_model_selection/aic_bic_comparison.py` | Penalizes complexity more | May underfit |
| Cross-Validation | `03_model_selection/cross_validation.py` | Data-driven | Computationally expensive |
| Hypothesis Tests | `03_model_selection/hypothesis_testing.py` | Statistical rigor | Binary decision |

### Goodness-of-Fit Tests

| Test | File | Best For | Limitations |
|------|------|----------|-------------|
| Kolmogorov-Smirnov | `04_goodness_of_fit/ks_test.py` | Overall fit | Sensitive to middle |
| Chi-Square | `04_goodness_of_fit/chi_square_test.py` | Categorical data | Requires binning |
| Anderson-Darling | `04_goodness_of_fit/anderson_darling.py` | Tail behavior | Specific distributions |

### Visualization Types

| Plot | File | Purpose |
|------|------|----------|
| PDF/CDF | `05_visualization/pdf_cdf_plots.py` | See distribution shape |
| Q-Q Plot | `05_visualization/qq_pp_plots.py` | Diagnose fit quality |
| P-P Plot | `05_visualization/qq_pp_plots.py` | Check probability match |
| Interactive | `05_visualization/interactive_plots.py` | Explore data dynamically |

---

## ✅ Best Practices

### 1. Always Visualize First

```python
import matplotlib.pyplot as plt

# Look at your data!
plt.hist(data, bins=50, edgecolor='black')
plt.show()

# Check for:
# - Outliers
# - Multimodality
# - Skewness
# - Bounded ranges
```

### 2. Try Multiple Distributions

```python
candidates = ['normal', 'lognormal', 'gamma', 'weibull_min']
results = {}

for dist_name in candidates:
    dist = get_distribution(dist_name)
    dist.fit(data)
    results[dist_name] = dist.aic()

# Choose best by AIC
best = min(results, key=results.get)
print(f"Best: {best} (AIC={results[best]:.2f})")
```

### 3. Validate with Q-Q Plots

```python
from scipy import stats
import numpy as np

# Fit distribution
dist.fit(data)

# Q-Q plot
percentiles = np.linspace(0.01, 0.99, len(data))
theoretical = dist.ppf(percentiles)
empirical = np.sort(data)

plt.scatter(theoretical, empirical)
plt.plot([data.min(), data.max()], [data.min(), data.max()], 'r--')
plt.show()

# Points should fall on diagonal line!
```

### 4. Report Uncertainty

```python
# Use bootstrap for confidence intervals
from examples.advanced_topics.bootstrap_confidence import bootstrap_ci

ci_mean = bootstrap_ci(data, statistic=np.mean, n_bootstrap=1000)
print(f"Mean: {data.mean():.2f} [95% CI: {ci_mean[0]:.2f}, {ci_mean[1]:.2f}]")
```

### 5. Check Assumptions

```python
# For normal distribution:
# 1. Check normality
from scipy.stats import shapiro
stat, p_value = shapiro(data)
print(f"Shapiro-Wilk p-value: {p_value:.4f}")

# 2. Check for outliers
z_scores = np.abs(stats.zscore(data))
outliers = data[z_scores > 3]
print(f"Outliers: {len(outliers)}")
```

---

## 🔧 Troubleshooting

### Common Issues

#### "Distribution doesn't fit well"

**Solution 1:** Try different distributions
```python
# Your data might not be normally distributed
candidates = ['lognormal', 'gamma', 'weibull_min', 'beta']
```

**Solution 2:** Check for mixture
```python
# Multiple populations?
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2)
```

**Solution 3:** Use non-parametric
```python
# KDE doesn't assume a distribution
from scipy.stats import gaussian_kde
kde = gaussian_kde(data)
```

#### "Fitting fails with error"

**Solution 1:** Check data range
```python
# Some distributions require positive data
if (data <= 0).any():
    data = data[data > 0]  # Filter
    # Or shift: data = data - data.min() + 0.01
```

**Solution 2:** Scale data
```python
# Large values can cause numerical issues
data_scaled = (data - data.mean()) / data.std()
```

**Solution 3:** Use different method
```python
# Try Method of Moments instead of MLE
dist.fit(data, method='MoM')
```

#### "How many parameters?"

**Rule of thumb:**
- n < 50: Use 1-2 parameter distributions
- n = 50-200: Up to 3 parameters OK
- n > 200: Can use complex distributions

```python
# Check sample size
if len(data) < 50:
    candidates = ['normal', 'exponential']  # Simple
else:
    candidates = ['normal', 'gamma', 'weibull_min']  # More complex
```

---

## 🎯 Quick Reference

### Common Distributions

| Distribution | Use When | Parameters | Domain |
|--------------|----------|------------|--------|
| Normal | Symmetric, bell-shaped | μ, σ | (-∞, ∞) |
| Lognormal | Right-skewed, positive | μ, σ | (0, ∞) |
| Exponential | Time between events | λ | (0, ∞) |
| Gamma | Positive, flexible shape | α, β | (0, ∞) |
| Weibull | Failure times | β (shape), η (scale) | (0, ∞) |
| Beta | Bounded [0,1] | α, β | [0, 1] |
| Uniform | Equal probability | a, b | [a, b] |

### Key Metrics

```python
# After fitting
dist.mean()          # Expected value
dist.std()           # Standard deviation
dist.var()           # Variance
dist.median()        # 50th percentile

dist.aic()           # Akaike Information Criterion
dist.bic()           # Bayesian Information Criterion

dist.pdf(x)          # Probability density at x
dist.cdf(x)          # Cumulative probability at x
dist.ppf(q)          # Quantile (inverse CDF)
dist.sf(x)           # Survival function (1 - CDF)
```

---

## 📝 Running Examples

### Run Individual Example

```bash
# Navigate to examples directory
cd examples/

# Run any example
python 01_basics/basic_fitting.py
python 06_real_world/finance_analysis.py
```

### Run All Examples

```bash
# Run all examples in a folder
for file in 01_basics/*.py; do python "$file"; done
```

### Import in Your Code

```python
# Import example utilities
from examples.model_selection.aic_bic_comparison import compare_models
from examples.visualization.pdf_cdf_plots import plot_fit

# Use example functions
results = compare_models(data, ['normal', 'lognormal'])
plot_fit(data, best_dist)
```

---

## 🤝 Contributing

We welcome contributions! Here's how:

1. **Report Issues**: Found a bug? Open an issue
2. **Suggest Examples**: Have a use case? Share it
3. **Submit PR**: Improved code? Send a pull request

### Guidelines

- Follow existing code style
- Add docstrings and comments
- Include example output
- Test with different data

---

## 📚 Additional Resources

### Documentation
- [SciPy Stats](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [DistFit-Pro API](../README.md)

### Books
- *Statistical Distributions* by Forbes et al.
- *Probability and Statistics for Engineers* by Montgomery

### Papers
- Akaike (1974) - AIC
- Schwarz (1978) - BIC
- Shapiro & Wilk (1965) - Normality test

---

## 📄 License

MIT License - see [LICENSE](../LICENSE) for details

---

## 👤 Author

**Ali Sadeghi Aghili**
- GitHub: [@alisadeghiaghili](https://github.com/alisadeghiaghili)
- Website: [zil.ink/thedatascientist](https://zil.ink/thedatascientist)

---

## 🌟 Star History

If these examples helped you, please ⭐ star this repo!

---

<div align="center">

**Happy Distribution Fitting!** 📊

Made with ❤️ by [Ali Sadeghi Aghili](https://github.com/alisadeghiaghili)

</div>
