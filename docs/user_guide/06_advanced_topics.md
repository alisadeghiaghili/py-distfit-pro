# Advanced Topics

## Weighted Data Fitting

### When to Use Weights

**Survey sampling:** Each observation represents different population sizes
```python
# Survey data: some regions have more respondents
weights = np.array([1.0, 0.8, 1.2, 0.9, 1.1])  # Adjustment factors
```

**Frequency data:** Some values repeated multiple times
```python
# Instead of: [1, 1, 1, 2, 2, 5]
# Use: values=[1, 2, 5], weights=[3, 2, 1]
```

**Risk-weighted data:** Different confidence in observations
```python
# Precise measurements vs estimates
weights = np.array([1.0, 1.0, 0.5, 0.5])  # Precise, precise, estimate, estimate
```

### Fitting with Weights

```python
from distfit_pro.core.weighted import WeightedFitting
import numpy as np

data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
weights = np.array([1.0, 2.0, 3.0, 2.0, 1.0])  # More weight on middle values

# Weighted MLE
params = WeightedFitting.fit_weighted_mle(data, weights, dist)

# Compare to unweighted
unweighted = dist.fit(data)  # Treats all equal
print(f"Weighted params: {params}")
print(f"Unweighted params: {unweighted.params}")
```

## Mixture Models

### When Data Has Multiple Modes

```python
import numpy as np
from distfit_pro import get_distribution

# Bimodal data: two groups
group1 = np.random.normal(2, 0.5, 500)
group2 = np.random.normal(8, 1.0, 500)
mixed = np.concatenate([group1, group2])

# Fit single distribution (won't work well)
dist_single = get_distribution('normal')
dist_single.fit(mixed)
print(f"Single: AIC={dist_single.aic:.1f}")

# Use mixture model
from distfit_pro.core.mixture import GaussianMixture

mixture = GaussianMixture(n_components=2)
mixture.fit(mixed)
print(f"Mixture: AIC={mixture.aic:.1f}")
```

## Transformed Data

### When to Transform

```python
import numpy as np
from distfit_pro import get_distribution

# Right-skewed income data
income = np.random.lognormal(10, 1, 1000)

# Log transform
log_income = np.log(income)

# Fit to transformed data
dist_log = get_distribution('normal')
dist_log.fit(log_income)

print(f"Normal fit to log(income): AIC={dist_log.aic:.1f}")
```

### Box-Cox Transformation

```python
from scipy.stats import boxcox
import numpy as np
from distfit_pro import get_distribution

# Automatic optimal transformation
data_transformed, lambda_param = boxcox(data)

dist = get_distribution('normal')
dist.fit(data_transformed)
print(f"Box-Cox lambda: {lambda_param:.3f}")
```

## Outlier Detection and Handling

### Automatic Detection

```python
from distfit_pro.core.diagnostics import Diagnostics
import numpy as np

data = np.array([1, 2, 3, 4, 5, 100])  # 100 is outlier

# Detect using multiple methods
outliers_zscore = Diagnostics.detect_outliers(data, method='zscore')
outliers_iqr = Diagnostics.detect_outliers(data, method='iqr')
outliers_mad = Diagnostics.detect_outliers(data, method='mad')  # Robust

print(f"Z-score: {outliers_zscore}")
print(f"IQR: {outliers_iqr}")
print(f"MAD: {outliers_mad}")
```

### Robust Fitting

```python
# Option 1: Remove outliers
clean_data = data[~outliers_mad.flag]
dist.fit(clean_data)

# Option 2: Use robust method (quantile matching)
dist.fit(data, method='quantile')

# Option 3: Downweight outliers
weights = np.ones(len(data))
weights[outliers_mad.flag] = 0.1  # Downweight
WeightedFitting.fit_weighted_mle(data, weights, dist)
```

## Truncated Distributions

### Fitting to Range-Limited Data

```python
from distfit_pro.core.truncated import TruncatedDistribution
import numpy as np

# Data constrained to [0, 10]
data = np.random.beta(2, 5, 1000) * 10  # Bounded [0, 10]

# Fit truncated normal
trunc_dist = TruncatedDistribution('normal', a=0, b=10)
trunc_dist.fit(data)

print(trunc_dist.summary())
```

## Censored Data

### When Some Values Are Unknown

```python
from distfit_pro.core.censored import CensoredFitting
import numpy as np

# Survival data: some observations censored
values = np.array([1.5, 2.3, 5.0, 5.0, 5.0])  # Last three censored at 5.0
censoring = np.array([False, False, True, True, True])

# Fit with censoring
dist = get_distribution('exponential')
CensoredFitting.fit_censored_mle(values, censoring, dist)

print(dist.summary())
```

## Custom Distributions

### Define Your Own

```python
from distfit_pro.core.base import BaseDistribution
import numpy as np
from scipy.special import gamma

class MyCustomDist(BaseDistribution):
    name = 'mycustom'
    params = ['alpha', 'beta']
    
    def pdf(self, x, alpha, beta):
        """Probability density function"""
        return (beta ** alpha) / gamma(alpha) * x ** (alpha - 1) * np.exp(-beta * x)
    
    def cdf(self, x, alpha, beta):
        """Cumulative distribution function"""
        from scipy.special import gammainc
        return gammainc(alpha, beta * x)
    
    def fit(self, data, method='mle'):
        """Fit parameters to data"""
        # Your fitting logic here
        pass

# Use it
custom = MyCustomDist()
custom.fit(data)
print(custom.summary())
```

## Bayesian Fitting

### Incorporate Prior Knowledge

```python
from distfit_pro.core.bayesian import BayesianFitting
import numpy as np

data = np.random.normal(10, 2, 100)

# Define priors (expert knowledge)
priors = {
    'loc': {'mean': 10, 'std': 1},      # Expert expects mean ≈ 10
    'scale': {'mean': 2, 'std': 0.5}    # Expert expects σ ≈ 2
}

# Fit with priors
result = BayesianFitting.fit_with_priors(data, dist, priors)
print(result.posterior_mean)
print(result.posterior_samples)  # MCMC samples
```

## Parallel Processing

### Speed Up Bootstrap

```python
from distfit_pro.core.bootstrap import Bootstrap

# Use all CPU cores (default: -1 uses all)
ci = Bootstrap.parametric(
    data, dist,
    n_bootstrap=10000,  # 10k bootstrap samples
    n_jobs=-1           # All cores
)

# With progress bar
ci = Bootstrap.parametric(
    data, dist,
    n_bootstrap=10000,
    n_jobs=-1,
    verbose=True  # Shows progress
)
```

## Memory Optimization

### For Large Datasets

```python
import numpy as np
from distfit_pro import get_distribution

# Use float32 instead of float64
data = np.random.normal(10, 2, 1_000_000).astype(np.float32)

# Fit on subset for exploration
sample = np.random.choice(data, 10000)
dist = get_distribution('normal')
dist.fit(sample)

# Then fit full dataset
dist.fit(data)  # Uses all data
```

## Next Steps

🚀 See [Tutorial 5: Robust Fitting](../tutorials/05_robust_fitting.md)

🚀 See [Tutorial 7: Advanced Inference](../tutorials/07_advanced_inference.md)
