# Core Distribution Classes

## Base Class: `BaseDistribution`

All distributions inherit from `BaseDistribution`.

**Common Attributes:**
- `name` (str): Distribution name
- `params` (dict): Fitted parameters
- `data` (array): Original data
- `fitted` (bool): Whether distribution is fitted
- `aic` (float): Akaike Information Criterion
- `bic` (float): Bayesian Information Criterion

**Common Methods:**
- `fit(data, method='mle')`: Fit to data
- `pdf(x)`: Probability density function
- `cdf(x)`: Cumulative distribution function
- `ppf(q)`: Quantile/percent point function
- `plot()`: Visualization
- `summary()`: Print summary

---

## Continuous Distributions (25)

### Normal Distribution
```python
dist = get_distribution('normal')
params: loc (mean), scale (std dev)
```

### Lognormal Distribution
```python
dist = get_distribution('lognormal')
params: s (shape), loc, scale
use: Prices, income (right-skewed)
```

### Weibull Distribution
```python
dist = get_distribution('weibull')
params: c (shape), loc, scale
use: Reliability, lifetimes
```

### Gamma Distribution
```python
dist = get_distribution('gamma')
params: a (shape), loc, scale
use: Waiting times, rainfall
```

### Exponential Distribution
```python
dist = get_distribution('exponential')
params: loc, scale
use: Time between events
```

### Beta Distribution
```python
dist = get_distribution('beta')
params: a, b (shapes), loc, scale
use: Proportions [0,1]
```

### Student's t Distribution
```python
dist = get_distribution('t')
params: df (degrees of freedom), loc, scale
use: Small samples, heavy tails
```

### Pareto Distribution
```python
dist = get_distribution('pareto')
params: b (shape), loc, scale
use: Wealth (80-20 rule)
```

### Gumbel Distribution
```python
dist = get_distribution('gumbel')
params: loc, scale
use: Extreme values (floods)
```

### Laplace Distribution
```python
dist = get_distribution('laplace')
params: loc, scale
use: Differences, errors
```

**And 15 more:** Uniform, Triangular, Logistic, Cauchy, Frechet, Chi-Square, F, Rayleigh, Inverse Gamma, Log-Logistic, etc.

---

## Discrete Distributions (5)

### Poisson Distribution
```python
dist = get_distribution('poisson')
params: mu (mean)
use: Count of rare events
```

### Binomial Distribution
```python
dist = get_distribution('binom')
params: n (trials), p (probability)
use: Success/failure
```

### Negative Binomial Distribution
```python
dist = get_distribution('nbinom')
params: n, p
use: Overdispersed counts
```

### Geometric Distribution
```python
dist = get_distribution('geom')
params: p (probability)
use: Trials to first success
```

### Hypergeometric Distribution
```python
dist = get_distribution('hypergeom')
params: M (pop size), n (success count), N (draw)
use: Sampling without replacement
```

---

## See Also

- [Distribution Selection Guide](../user_guide/02_distributions.md)
- [API Reference](index.md)
