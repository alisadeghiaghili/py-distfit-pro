# Frequently Asked Questions

## Installation & Setup

### Q: How do I install DistFit Pro?
**A:**
```bash
pip install distfit-pro
```

For development:
```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e ".[dev]"
```

### Q: What Python versions are supported?
**A:** Python 3.8, 3.9, 3.10, 3.11, 3.12

### Q: Do I need to install SciPy/NumPy separately?
**A:** No, they're installed automatically as dependencies.

## Basic Usage

### Q: What's the simplest way to fit a distribution?
**A:**
```python
from distfit_pro import get_distribution
import numpy as np

data = np.random.normal(10, 2, 1000)
dist = get_distribution('normal')
dist.fit(data)
print(dist.summary())
```

### Q: How do I see what distributions are available?
**A:**
```python
from distfit_pro import list_distributions
all_dists = list_distributions()
print(f"Continuous: {all_dists['continuous']}")
print(f"Discrete: {all_dists['discrete']}")
```

### Q: How do I compare multiple distributions?
**A:**
```python
for name in ['normal', 'weibull', 'gamma']:
    dist = get_distribution(name)
    dist.fit(data)
    print(f"{name}: AIC={dist.aic:.1f}")
```

## Estimation Methods

### Q: Which method should I use - MLE, Moments, or Quantile?
**A:** 
- **MLE (default):** Best for most cases
- **Moments:** Fast, good for exploration
- **Quantile:** Robust to outliers

Start with MLE. If it fails, try Moments. If you have outliers, use Quantile.

### Q: My fit didn't converge. What do I do?
**A:**
```python
# Try different methods
dist.fit(data, method='moments')       # Usually works
dist.fit(data, method='quantile')      # Very robust

# Check for outliers
outliers = np.abs(data - np.mean(data)) > 3*np.std(data)
print(f"Outliers: {np.sum(outliers)}")

# Remove outliers and retry
clean_data = data[~outliers]
dist.fit(clean_data, method='mle')
```

### Q: What do parameters like 'loc' and 'scale' mean?
**A:**
- **loc (location):** Where distribution is centered (mean for Normal)
- **scale:** Spread/width (std dev for Normal)
- **shape:** How "peaked" or "flat" (for Weibull, Gamma, etc.)

```python
print(dist.params)
# {'loc': 10.05, 'scale': 1.98}  # for Normal
```

## Goodness-of-Fit

### Q: How do I know if my fit is good?
**A:**
```python
from distfit_pro.core.gof_tests import GOFTests

results = GOFTests.run_all_tests(data, dist)
print(GOFTests.summary_table(results))

# If p-value > 0.05 for most tests, fit is good
```

### Q: What do the different GOF tests measure?
**A:**
- **KS:** General overall fit
- **AD:** Emphasizes tails (more sensitive)
- **Chi-Square:** For binned/discrete data
- **CVM:** Alternative to KS, balanced

### Q: What if different tests give different results?
**A:** Look for consensus. If 3/4 tests pass (p > 0.05), the fit is probably good.

## Visualization

### Q: How do I plot my fit?
**A:**
```python
dist.plot()  # Default: histogram + curve + stats
```

### Q: What's a Q-Q plot and when do I need it?
**A:** Q-Q plot shows how well quantiles match:
```python
dist.plot(plot_type='qq')

# Perfect fit = diagonal line
# Points above line = heavier tails in data
# Points below line = lighter tails in data
```

### Q: How do I save a plot?
**A:**
```python
import matplotlib.pyplot as plt

dist.plot()
plt.savefig('my_distribution.png', dpi=300)
plt.savefig('my_distribution.pdf')  # Vector format
```

## Advanced

### Q: How do I fit with weighted data?
**A:**
```python
from distfit_pro.core.weighted import WeightedFitting

weights = np.array([1.0, 2.0, 0.5])  # Different weights
params = WeightedFitting.fit_weighted_mle(data, weights, dist)
```

### Q: Can I get confidence intervals?
**A:**
```python
from distfit_pro.core.bootstrap import Bootstrap

ci = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=-1)
for param, result in ci.items():
    print(f"{param}: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### Q: What's a mixture model and when do I need it?
**A:** When data has multiple modes (bumps):
```python
from distfit_pro.core.mixture import GaussianMixture

mixture = GaussianMixture(n_components=2)
mixture.fit(data)
```

### Q: How do I detect outliers?
**A:**
```python
from distfit_pro.core.diagnostics import Diagnostics

outliers = Diagnostics.detect_outliers(data, method='mad')  # Robust
print(outliers.summary())
```

### Q: Can I use custom distributions?
**A:** Yes! See [User Guide 06: Advanced Topics](docs/user_guide/06_advanced_topics.md#custom-distributions)

## Performance

### Q: My fitting is slow. How do I speed it up?
**A:**
```python
# Use faster method
dist.fit(data, method='moments')  # Fastest

# Reduce data size for exploration
sample = np.random.choice(data, 10000)
dist.fit(sample)

# Check system resources
import psutil
print(psutil.cpu_count())  # Available CPU cores
```

### Q: How do I use multiple CPU cores?
**A:**
```python
from distfit_pro.core.bootstrap import Bootstrap

# n_jobs=-1 uses all cores
ci = Bootstrap.parametric(data, dist, n_bootstrap=10000, n_jobs=-1)
```

### Q: How do I work with large datasets?
**A:**
```python
# Use float32 to save memory
data = np.random.normal(10, 2, 10_000_000).astype(np.float32)

# Fit subset for exploration
sample = np.random.choice(data, 100_000)
dist.fit(sample)
```

## Troubleshooting

### Q: I get "Distribution not found" error
**A:**
```python
from distfit_pro import list_distributions
print(list_distributions())  # Check exact name

# Names are lowercase
dist = get_distribution('normal')  # ✓ Correct
dist = get_distribution('Normal')  # ✗ Wrong
```

### Q: My fit looks wrong visually
**A:**
```python
# 1. Check for outliers
print(f"Min: {np.min(data)}, Max: {np.max(data)}")

# 2. Try different distribution
for name in ['normal', 'lognormal', 'weibull']:
    dist = get_distribution(name)
    dist.fit(data)
    print(f"{name}: AIC={dist.aic:.1f}")

# 3. Check Q-Q plot
dist.plot(plot_type='qq')
```

### Q: I have negative values but fit is failing
**A:**
```python
# Some distributions require positive data (Lognormal, Weibull, Gamma)
# Either:
# 1. Use normal, student's t, or laplace (allow negatives)
dist = get_distribution('normal')

# 2. Or transform data
data_positive = data - np.min(data) + 1
```

### Q: Bootstrap is taking too long
**A:**
```python
# Reduce bootstrap samples for testing
ci = Bootstrap.parametric(data, dist, n_bootstrap=100)  # Quick test

# Then increase for final result
ci = Bootstrap.parametric(data, dist, n_bootstrap=10000, n_jobs=-1)
```

## Data Requirements

### Q: How much data do I need?
**A:**
- **Minimum:** n >= 20 (but results unreliable)
- **Good:** n >= 100
- **Excellent:** n >= 1000

Smaller samples = wider confidence intervals

### Q: Can I use integer data?
**A:** Yes, but convert to float:
```python
data = np.array([1, 2, 3, 4, 5], dtype=float)
```

### Q: Can I use categorical data?
**A:** No, only numerical data. For categories, use frequency counts:
```python
# Instead of: ['red', 'red', 'blue', 'red']
# Use Poisson/Binomial with counts
```

## Documentation

### Q: Where's the full API documentation?
**A:** See [API Reference](api/)

### Q: Are there tutorials?
**A:** Yes! Check [Tutorials](docs/tutorials/)

### Q: Where can I find examples?
**A:** See `examples/` directory in repository

## Support

### Q: How do I report a bug?
**A:** Open an issue: [GitHub Issues](https://github.com/alisadeghiaghili/py-distfit-pro/issues)

### Q: Can I request a feature?
**A:** Yes! Open a feature request issue

### Q: Is there a community forum?
**A:** Not yet, but check GitHub Discussions

### Q: Who maintains this project?
**A:** Ali Sadeghi Aghili. See [Contributing](CONTRIBUTING.md) to help!

---

**Can't find your question?** Open an [issue](https://github.com/alisadeghiaghili/py-distfit-pro/issues) and I'll add it here!
