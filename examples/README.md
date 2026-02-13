# Examples

Practical examples demonstrating various features of distfit-pro.

## Quick Start

### Basic Distribution Fitting
```python
import numpy as np
from distfit_pro import get_distribution

# Generate sample data
data = np.random.normal(10, 2, 1000)

# Fit a distribution
dist = get_distribution('normal')
dist.fit(data, method='mle')

# Check results
print(dist.summary())
print(f"Mean: {dist.mean():.2f}")
print(f"AIC: {dist.aic():.2f}")
```

### Automatic Model Selection
```python
from distfit_pro import DistributionFitter

# Fit multiple distributions and pick best
fitter = DistributionFitter(data)
results = fitter.fit()

# Get best model
best = results.get_best()
print(f"Best distribution: {best.info.display_name}")

# Visualize
results.plot(kind='comparison')
```

### Multilingual Verbose Mode
```python
from distfit_pro.core.config import config

# English with detailed explanations
config.set_language('en')
config.set_verbosity('verbose')

dist.fit(data)
# Prints detailed explanations of:
# - Data characteristics
# - Fitting process
# - Parameter meanings
# - Statistical properties

# Persian output
config.set_language('fa')
dist.fit(data)
# Same information in Persian
```

## Examples

- `quick_test.py` - Simple smoke test
- `complete_workflow.py` - End-to-end realistic workflow
- `verbose_demo.py` - Verbosity and language features

## Running Examples

```bash
python examples/complete_workflow.py
```

Or from Python:
```python
import sys
sys.path.insert(0, '../')  # if running from examples dir
exec(open('complete_workflow.py').read())
```
