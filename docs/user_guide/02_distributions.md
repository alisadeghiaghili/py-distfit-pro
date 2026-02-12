# Understanding Distributions

## What is a Distribution?

A probability distribution describes how likely different outcomes are in a random process.

**Example:** Heights of people follow roughly a "normal" distribution - most are average, fewer are very tall or very short.

## Continuous vs Discrete

### Continuous Distributions
- Can take any value in a range (including decimals)
- **Examples:** Normal, Lognormal, Weibull
- **Use for:** Measurements, prices, time intervals

```python
# Heights, temperatures, stock prices
data = np.random.normal(170, 10, 1000)  # Heights in cm
```

### Discrete Distributions  
- Can only take specific values (usually integers)
- **Examples:** Poisson, Binomial, Geometric
- **Use for:** Counts, number of events

```python
# Number of customers, defects, phone calls
data = np.random.poisson(5, 1000)  # ~5 events on average
```

## 30 Supported Distributions

### Top 10 Continuous

| Distribution | Best For | Key Feature |
|--------------|----------|-------------|
| **Normal** | General purpose | Bell curve, symmetric |
| **Lognormal** | Prices, income | Right-skewed, positive |
| **Weibull** | Reliability, lifetimes | Flexible shape |
| **Gamma** | Waiting times | Sum of exponentials |
| **Exponential** | Time between events | Memoryless |
| **Beta** | Proportions [0,1] | Bounded range |
| **Pareto** | Wealth, power law | Few large + many small |
| **Gumbel** | Extremes (floods) | Skewed maxima |
| **Student's t** | Small samples | Heavy tails |
| **Laplace** | Differences | Double exponential |

### Top 5 Discrete

| Distribution | Best For | Example |
|--------------|----------|----------|
| **Poisson** | Rare events | Phone calls per hour |
| **Binomial** | Yes/no outcomes | Heads in 10 coin flips |
| **Neg. Binomial** | Overdispersed counts | Failed trials before success |
| **Geometric** | Trials to first success | First defect in batch |
| **Hypergeometric** | Sampling w/o replacement | Lottery wins |

## Quick Selection Guide

### "My data is..."

**Symmetric & bell-shaped?**
→ Try **Normal** or **Student's t**

**Skewed right (long tail)?**
→ Try **Lognormal**, **Gamma**, or **Weibull**

**Skewed left?**
→ Try **Gumbel** or **Pareto**

**Always positive & heavy-tailed?**
→ Try **Lognormal** or **Exponential**

**Bounded between 0 and 1?**
→ Try **Beta**

**Count data (0, 1, 2, ...)?**
→ Try **Poisson** or **Binomial**

## Working with Distributions

### List All Available

```python
from distfit_pro import list_distributions

all_dists = list_distributions()
print(f"Continuous: {len(all_dists['continuous'])}")
print(f"Discrete: {len(all_dists['discrete'])}")
```

### Get Distribution Object

```python
from distfit_pro import get_distribution

# By name
normal = get_distribution('normal')
lognormal = get_distribution('lognormal')
weibull = get_distribution('weibull')

# Check if exists
try:
    dist = get_distribution('mycustomdist')
except ValueError:
    print("Distribution not found")
```

### Fit and Compare

```python
import numpy as np
from distfit_pro import get_distribution

data = np.random.weibull(2, 1000)

# Try multiple
for name in ['normal', 'weibull', 'gamma', 'lognormal']:
    dist = get_distribution(name)
    try:
        dist.fit(data)
        print(f"{name:12} AIC={dist.aic:8.1f} BIC={dist.bic:8.1f}")
    except:
        print(f"{name:12} Failed to fit")
```

## Parameters

Each distribution has **shape, location, and scale parameters**:

```python
from distfit_pro import get_distribution

dist = get_distribution('normal')
dist.fit(data)

# After fitting:
print(dist.params)  # {'loc': 10.05, 'scale': 1.98}

# Access individual parameters
print(dist.params['loc'])    # mean
print(dist.params['scale'])  # standard deviation
```

## Next Steps

👉 See [Tutorial 1: Basic Fitting](../tutorials/01_basic_fitting.md) for practical examples

👉 See [Tutorial 3: GOF Tests](../tutorials/03_goodness_of_fit.md) for testing if fit is good
