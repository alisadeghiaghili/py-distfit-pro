# Estimation Methods Deep Dive

## Overview

DistFit Pro supports 3 statistical methods to estimate parameters. Each has tradeoffs:

| Method | Speed | Accuracy | Robustness | When to Use |
|--------|-------|----------|-----------|-------------|
| **MLE** | Medium | ⭐⭐⭐⭐⭐ | Medium | Default, well-behaved data |
| **Moments** | ⭐⭐⭐⭐⭐ | Medium | ⭐⭐⭐⭐ | Quick exploration, outliers |
| **Quantile** | ⭐⭐⭐⭐ | Good | ⭐⭐⭐⭐⭐ | Heavy outliers, robust fit |

## 1. Maximum Likelihood Estimation (MLE)

### What It Does
Finds parameters that make your data **most likely** to occur.

### How It Works

```
Likelihood = P(data | parameters)

Find parameters that maximize this probability
```

### When to Use
- ✅ Well-behaved data
- ✅ Most statistical tests work best with MLE
- ✅ Larger sample sizes (n > 30)
- ❌ Heavy outliers
- ❌ Small sample sizes with extreme values

### Code

```python
from distfit_pro import get_distribution
import numpy as np

data = np.random.normal(10, 2, 100)
dist = get_distribution('normal')

# MLE (default)
dist.fit(data, method='mle')
print(f"loc: {dist.params['loc']:.2f}")
print(f"scale: {dist.params['scale']:.2f}")
```

### Advantages
- ✅ Statistically optimal (most efficient)
- ✅ Works with any distribution
- ✅ Standard errors automatically calculated
- ✅ Best for hypothesis testing

### Disadvantages
- ❌ Can fail on heavy-tailed data
- ❌ May not converge with bad starting values
- ❌ Sensitive to outliers

### Troubleshooting

**Problem:** "Fit did not converge"
```python
# Solution 1: Try moments first
dist.fit(data, method='moments')

# Solution 2: Use robust method
dist.fit(data, method='quantile')

# Solution 3: Check for outliers
outliers = np.abs(data - np.mean(data)) > 3 * np.std(data)
print(f"Outliers found: {np.sum(outliers)}")

# Remove and retry
clean_data = data[~outliers]
dist.fit(clean_data, method='mle')
```

## 2. Method of Moments (MM)

### What It Does
Matches **sample moments** (mean, variance) to **theoretical moments**.

### How It Works

```
Sample mean = Theoretical mean
Sample variance = Theoretical variance

Solve for parameters
```

### When to Use
- ✅ Quick exploration
- ✅ Rough initial estimates
- ✅ Data with outliers
- ✅ Small samples
- ❌ Very skewed distributions (may be inaccurate)

### Code

```python
from distfit_pro import get_distribution
import numpy as np

data = np.random.gamma(2, 2, 100)
dist = get_distribution('gamma')

# Method of Moments (fast!)
dist.fit(data, method='moments')
print(dist.summary())
```

### Advantages
- ✅ **Always converges** (no numerical issues)
- ✅ Very fast
- ✅ Good for quick estimates
- ✅ Less sensitive to outliers than MLE

### Disadvantages
- ❌ Less efficient than MLE
- ❌ May produce invalid parameters
- ❌ Not ideal for hypothesis testing

### Example: Comparing to MLE

```python
import numpy as np
from distfit_pro import get_distribution

data = np.random.weibull(1.5, 100)  # Weibull data

# Method of Moments
dist_mm = get_distribution('weibull')
dist_mm.fit(data, method='moments')

# MLE
dist_mle = get_distribution('weibull')
dist_mle.fit(data, method='mle')

# Compare
print("Method of Moments:", dist_mm.params)
print("MLE:", dist_mle.params)
print(f"AIC: MM={dist_mm.aic:.1f}, MLE={dist_mle.aic:.1f}")
```

## 3. Quantile Matching (QM)

### What It Does
Matches **empirical quantiles** (percentiles) to **theoretical quantiles**.

### How It Works

```
Sample 25th percentile = Theoretical 25th percentile
Sample 50th percentile = Theoretical 50th percentile
Sample 75th percentile = Theoretical 75th percentile

Solve for parameters
```

### When to Use
- ✅ Data with significant outliers
- ✅ Robust estimation needed
- ✅ Heavy-tailed distributions
- ✅ Non-normal data
- ❌ Small samples (< 20)

### Code

```python
from distfit_pro import get_distribution
import numpy as np

data = np.random.lognormal(0, 1, 100)  # Right-skewed
data = np.concatenate([data, [1000, 2000]])  # Add outliers

dist = get_distribution('lognormal')

# Robust quantile matching
dist.fit(data, method='quantile', quantiles=[0.25, 0.5, 0.75])
print(dist.summary())
```

### Advantages
- ✅ **Most robust** to outliers
- ✅ Works with heavy-tailed data
- ✅ Always converges
- ✅ Good for non-normal distributions

### Disadvantages
- ❌ Less efficient than MLE (needs more data)
- ❌ Ignores outer tails
- ❌ Choice of quantiles matters

### Choosing Quantiles

```python
# Default: quartiles (25%, 50%, 75%)
dist.fit(data, method='quantile')

# Custom: focus on extremes
dist.fit(data, method='quantile', quantiles=[0.05, 0.5, 0.95])

# Custom: focus on center
dist.fit(data, method='quantile', quantiles=[0.35, 0.5, 0.65])
```

## Decision Tree

```
Choosing estimation method:

Does data have extreme outliers?
  YES → Use Quantile Matching
  NO  → Continue

Small sample (n < 30)?
  YES → Use Method of Moments
  NO  → Continue

Well-behaved, normal-like data?
  YES → Use MLE
  NO  → Try Quantile Matching
```

## Complete Example

```python
import numpy as np
from distfit_pro import get_distribution

# Generate data with outliers
np.random.seed(42)
data = np.random.lognormal(0, 1, 500)
data = np.concatenate([data, np.random.uniform(500, 1000, 10))])

# Try all three methods
for method in ['moments', 'quantile', 'mle']:
    dist = get_distribution('lognormal')
    try:
        dist.fit(data, method=method)
        print(f"{method:10} | AIC={dist.aic:7.1f} | params={dist.params}")
    except Exception as e:
        print(f"{method:10} | FAILED: {e}")
```

## Next Steps

👉 See [Tutorial 2: Comparison](../tutorials/02_compare_distributions.md)

👉 See [Tutorial 5: Robustness](../tutorials/05_robust_fitting.md)
