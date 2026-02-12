# API Reference

## Main Module

### `distfit_pro.get_distribution(name)`

Get a distribution object by name.

**Parameters:**
- `name` (str): Distribution name (lowercase, e.g., 'normal', 'weibull')

**Returns:**
- Distribution object with `fit()`, `plot()`, and `summary()` methods

**Example:**
```python
from distfit_pro import get_distribution

dist = get_distribution('normal')
```

### `distfit_pro.list_distributions()`

List all supported distributions.

**Returns:**
- dict with 'continuous' and 'discrete' keys

**Example:**
```python
from distfit_pro import list_distributions

all_dists = list_distributions()
print(all_dists['continuous'])
```

---

## Distribution Objects

### `Distribution.fit(data, method='mle', **kwargs)`

Fit distribution to data.

**Parameters:**
- `data` (array): Data to fit
- `method` (str): 'mle', 'moments', or 'quantile'
- `**kwargs`: Method-specific parameters

**Returns:**
- self (for chaining)

**Example:**
```python
dist.fit(data, method='mle')
```

### `Distribution.plot(figsize=(10,6), title=None, **kwargs)`

Plot fitted distribution.

**Parameters:**
- `figsize` (tuple): Figure size
- `title` (str): Plot title
- `plot_type` (str): 'default', 'qq', 'pp'
- `ax` (Axes): Matplotlib axes

**Example:**
```python
dist.plot(figsize=(12, 8))
```

### `Distribution.summary()`

Print summary of fitted parameters.

**Returns:**
- None (prints to stdout)

**Example:**
```python
dist.summary()
```

### `Distribution.params`

Get fitted parameters as dictionary.

**Returns:**
- dict of parameter names and values

**Example:**
```python
print(dist.params)  # {'loc': 10.05, 'scale': 1.98}
```

### `Distribution.aic`

Akaike Information Criterion (for model comparison).

**Returns:**
- float

### `Distribution.bic`

Bayesian Information Criterion (for model comparison).

**Returns:**
- float

---

## Goodness-of-Fit Tests

### `GOFTests.run_all_tests(data, dist)`

Run all 4 GOF tests.

**Parameters:**
- `data` (array): Data
- `dist` (Distribution): Fitted distribution

**Returns:**
- dict with test results

**Example:**
```python
from distfit_pro.core.gof_tests import GOFTests

results = GOFTests.run_all_tests(data, dist)
print(GOFTests.summary_table(results))
```

### `KolmogorovSmirnov.test(data, dist)`

Kolmogorov-Smirnov test.

**Returns:**
- result object with `statistic`, `pvalue`, `critical_value`

### `AndersonDarling.test(data, dist)`

Anderson-Darling test.

**Returns:**
- result object with `statistic`, `critical_values`, `pvalue`

### `ChiSquare.test(data, dist, bins=10)`

Chi-square test.

**Parameters:**
- `bins` (int): Number of bins

**Returns:**
- result object with `statistic`, `pvalue`, `dof`

### `CramerVonMises.test(data, dist)`

Cramér-von Mises test.

**Returns:**
- result object with `statistic`, `pvalue`

---

## Bootstrap

### `Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=-1)`

Parametric bootstrap confidence intervals.

**Parameters:**
- `data` (array): Original data
- `dist` (Distribution): Fitted distribution
- `n_bootstrap` (int): Number of samples
- `n_jobs` (int): Number of CPU cores (-1 = all)

**Returns:**
- dict of parameter results with CI

**Example:**
```python
from distfit_pro.core.bootstrap import Bootstrap

ci = Bootstrap.parametric(data, dist, n_bootstrap=1000, n_jobs=-1)
for param, result in ci.items():
    print(f"{param}: [{result.ci_lower:.3f}, {result.ci_upper:.3f}]")
```

### `Bootstrap.nonparametric(data, dist, n_bootstrap=1000, n_jobs=-1)`

Non-parametric (resampling) bootstrap.

**Returns:**
- Same as parametric

---

## Diagnostics

### `Diagnostics.detect_outliers(data, method='mad', threshold=3.0)`

Detect outliers in data.

**Parameters:**
- `data` (array): Data
- `method` (str): 'zscore', 'iqr', or 'mad'
- `threshold` (float): Sensitivity threshold

**Returns:**
- result object with `flag` (bool array), `count`, `indices`

**Example:**
```python
from distfit_pro.core.diagnostics import Diagnostics

outliers = Diagnostics.detect_outliers(data, method='mad')
print(f"Found {outliers.count} outliers")
clean_data = data[~outliers.flag]
```

### `Diagnostics.residual_analysis(data, dist)`

Analyze residuals from fitted distribution.

**Returns:**
- result object with residuals and diagnostics

---

## Weighted Fitting

### `WeightedFitting.fit_weighted_mle(data, weights, dist)`

Fit with observation weights.

**Parameters:**
- `data` (array): Data
- `weights` (array): Weight for each observation
- `dist` (Distribution): Distribution to fit

**Returns:**
- dict of fitted parameters

**Example:**
```python
from distfit_pro.core.weighted import WeightedFitting

weights = np.array([1.0, 2.0, 0.5])
params = WeightedFitting.fit_weighted_mle(data, weights, dist)
```

---

## Mixture Models

### `GaussianMixture(n_components=2)`

Gaussian mixture model for multimodal data.

**Parameters:**
- `n_components` (int): Number of mixture components

**Methods:**
- `fit(data)`
- `plot()`
- `summary()`

**Example:**
```python
from distfit_pro.core.mixture import GaussianMixture

mixture = GaussianMixture(n_components=2)
mixture.fit(data)
mixture.plot()
```

---

## For More Details

- **User Guides:** [User Guide Index](../user_guide/)
- **Tutorials:** [Tutorial Index](../tutorials/)
- **Source Code:** [GitHub Repository](https://github.com/alisadeghiaghili/py-distfit-pro)

---

*API documentation auto-generated from docstrings*
