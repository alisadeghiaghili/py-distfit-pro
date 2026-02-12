# DistFit Pro - Complete Distribution Fitting Library

**Professional Python library for statistical distribution fitting.**

## Features

✅ **30 Distributions** - 25 continuous + 5 discrete  
✅ **GOF Tests** - KS, Anderson-Darling, Chi-Square, CvM  
✅ **Bootstrap CI** - Parametric & non-parametric  
✅ **Diagnostics** - Residuals, influence, outliers  
✅ **Weighted Data** - Full support  
✅ **Multilingual** - EN/FA/DE

## Quick Start

```python
from distfit_pro import get_distribution
import numpy as np

# Generate data
data = np.random.normal(10, 2, 1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data)
print(dist.summary())
```

## Goodness-of-Fit Tests

```python
from distfit_pro.core.gof_tests import GOFTests

# Run all tests
results = GOFTests.run_all_tests(data, dist)
print(GOFTests.summary_table(results))
```

## Bootstrap Confidence Intervals

```python
from distfit_pro.core.bootstrap import Bootstrap

ci = Bootstrap.parametric(data, dist, n_bootstrap=1000)
for param, result in ci.items():
    print(result)
```

## Diagnostics

```python
from distfit_pro.core.diagnostics import Diagnostics

# Residuals
residuals = Diagnostics.residual_analysis(data, dist)

# Outliers
outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
```

## Documentation

See `docs/` folder for complete tutorials.

## License

MIT License
