# Getting Started with DistFit Pro

## Installation

### From PyPI (Recommended)

```bash
pip install distfit-pro
```

### From Source

```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e ".[dev]"
```

## Your First Fit

### 1. Import and Generate Data

```python
from distfit_pro import get_distribution
import numpy as np

# Generate sample data from normal distribution
np.random.seed(42)
data = np.random.normal(loc=10, scale=2, size=1000)
```

### 2. Fit a Distribution

```python
# Get normal distribution
dist = get_distribution('normal')

# Fit using Maximum Likelihood Estimation (MLE)
dist.fit(data, method='mle')
```

### 3. View Results

```python
# Summary of fitted parameters
print(dist.summary())

# Output:
# Distribution: Normal
# ─────────────────────────────────────
# Parameter    | Value      | SE
# ─────────────────────────────────────
# loc (μ)      | 10.05      | 0.063
# scale (σ)    | 1.98       | 0.045
# ─────────────────────────────────────
```

### 4. Plot the Fit

```python
# Visualize the fit
dist.plot()  # Shows histogram + fitted curve
```

## Understanding Estimation Methods

### Maximum Likelihood (MLE) - **Default**
- **Best for:** Most situations
- **Advantage:** Statistically optimal
- **Disadvantage:** Can fail for heavy-tailed data

```python
dist.fit(data, method='mle')
```

### Method of Moments
- **Best for:** Quick estimates, robust to outliers
- **Advantage:** Always converges
- **Disadvantage:** Less efficient than MLE

```python
dist.fit(data, method='moments')
```

### Quantile Matching
- **Best for:** Data with outliers
- **Advantage:** Very robust
- **Disadvantage:** Less efficient

```python
dist.fit(data, method='quantile', quantiles=[0.25, 0.5, 0.75])
```

## Common Workflows

### Workflow 1: Quick Exploration

```python
from distfit_pro import get_distribution
import numpy as np

data = np.random.weibull(2, 1000)

# Try multiple distributions
for dist_name in ['weibull', 'gamma', 'lognormal']:
    dist = get_distribution(dist_name)
    dist.fit(data)
    print(f"{dist_name}: {dist.aic:.2f}")  # Compare AIC
```

### Workflow 2: Statistical Testing

```python
from distfit_pro.core.gof_tests import GOFTests

# Run goodness-of-fit tests
results = GOFTests.run_all_tests(data, dist)

# View summary
print(GOFTests.summary_table(results))
```

### Workflow 3: Uncertainty Quantification

```python
from distfit_pro.core.bootstrap import Bootstrap

# Get confidence intervals
ci = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=-1)

for param, result in ci.items():
    print(f"{param}: {result.ci_lower:.3f} - {result.ci_upper:.3f}")
```

## Next Steps

👉 **Ready for more?** Check out [Tutorial 1: Basic Fitting](../tutorials/01_basic_fitting.md)

## Troubleshooting

### Issue: "Fit did not converge"
- Try `method='moments'` first
- Check for outliers: `dist.detect_outliers(data)`
- Verify data is numeric

### Issue: "Distribution not found"
- Check supported distributions: `from distfit_pro import list_distributions; print(list_distributions())`
- Spelling matters! Use lowercase

### Issue: "Too slow"
- Use `method='moments'` (faster)
- Reduce data size for exploration
- Check CPU usage (parallel processing uses all cores)

---

**Need help?** See [FAQ](../faq.md) or open an [issue](https://github.com/alisadeghiaghili/py-distfit-pro/issues)
