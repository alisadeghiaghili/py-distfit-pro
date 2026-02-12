# Visualization Guide

## Quick Start

```python
from distfit_pro import get_distribution
import numpy as np

# Fit a distribution
data = np.random.normal(10, 2, 1000)
dist = get_distribution('normal')
dist.fit(data)

# Plot everything
dist.plot()  # Beautiful default plot
```

## Built-in Plot Types

### 1. Default Plot (Recommended)

```python
dist.plot()  # Shows:
             # - Histogram with data
             # - Fitted curve
             # - Statistics
             # - GOF test result
```

**Output:**
- Histogram of data (semi-transparent)
- Fitted PDF curve (bright line)
- Statistics box (mean, std, AIC, BIC)
- Goodness-of-fit result

### 2. Q-Q Plot (Diagnostic)

```python
# Quantile-Quantile plot
# Perfect fit = straight diagonal line
dist.plot(plot_type='qq')

# Interpretation:
# - Points on line = perfect fit
# - Points above line = data has heavier tail
# - Points below line = data has lighter tail
```

### 3. P-P Plot (Another Diagnostic)

```python
# Probability-Probability plot
dist.plot(plot_type='pp')

# Similar to Q-Q but for cumulative probability
```

### 4. Residuals Plot

```python
from distfit_pro.core.diagnostics import Diagnostics

# Analyze residuals
residuals = Diagnostics.residual_analysis(data, dist)
residuals.plot()  # Residuals vs quantiles
```

### 5. Multiple Distributions

```python
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

# Create comparison plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for ax, dist_name in zip(axes, ['normal', 'lognormal', 'weibull']):
    dist = get_distribution(dist_name)
    dist.fit(data)
    dist.plot(ax=ax)
```

## Customization

### Colors

```python
# Change colors
dist.plot(color_data='blue', color_fit='red')

# Using Matplotlib colors
dist.plot(color_data='#1f77b4', color_fit='#ff7f0e')
```

### Size and Style

```python
# Larger figure
dist.plot(figsize=(12, 6))

# Custom title
dist.plot(title='My Distribution Fit')

# No title
dist.plot(title=None)
```

### Legend

```python
# Show/hide legend
dist.plot(show_legend=True)   # Default
dist.plot(show_legend=False)

# Custom labels
dist.plot(label_data='Observed', label_fit='Fitted Model')
```

## Advanced: Interactive Plots

### Using Plotly (Interactive)

```python
# Create interactive plot
dist.plot_interactive()

# Features:
# - Zoom and pan
# - Hover for values
# - Download as PNG
# - Show/hide traces
```

### Export to File

```python
import matplotlib.pyplot as plt

dist.plot()
plt.savefig('my_distribution.png', dpi=300, bbox_inches='tight')
plt.savefig('my_distribution.pdf')  # Vector format
```

## Comparison Plots

### Side-by-Side

```python
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Plot 1: Normal fit
dist_normal = get_distribution('normal')
dist_normal.fit(data)
dist_normal.plot(ax=ax1)
ax1.set_title('Normal Distribution')

# Plot 2: Weibull fit
dist_weibull = get_distribution('weibull')
dist_weibull.fit(data)
dist_weibull.plot(ax=ax2)
ax2.set_title('Weibull Distribution')

plt.tight_layout()
plt.show()
```

### Overlay Multiple

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 6))

# Plot data histogram once
ax.hist(data, bins=50, density=True, alpha=0.5, label='Data')

# Overlay multiple fits
for name, color in [('normal', 'red'), ('weibull', 'blue'), ('lognormal', 'green')]:
    dist = get_distribution(name)
    dist.fit(data)
    dist.plot_pdf(ax=ax, color=color, label=name)

ax.legend()
plt.show()
```

## Statistical Plots

### GOF Test Results

```python
from distfit_pro.core.gof_tests import GOFTests

# Run tests
results = GOFTests.run_all_tests(data, dist)

# Plot test statistics
GOFTests.plot_results(results)
```

### Bootstrap Confidence Intervals

```python
from distfit_pro.core.bootstrap import Bootstrap

# Get CI
ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)

# Plot CI for each parameter
for param, result in ci_results.items():
    result.plot()  # Histogram + CI interval
```

## Jupyter Integration

```python
# In Jupyter notebooks
%matplotlib inline

dist.plot()  # Displays inline

# Interactive mode
%matplotlib widget
dist.plot_interactive()  # Interactive plot
```

## Common Issues

### Issue: Plot looks blank
```python
# Solution: Add plt.show()
import matplotlib.pyplot as plt

dist.plot()
plt.show()
```

### Issue: Cannot save plot
```python
# Solution: Use full path
import os
path = os.path.expanduser('~/plots/distribution.png')
dist.plot()
plt.savefig(path)
```

### Issue: Text overlaps
```python
# Solution: Increase figure size
dist.plot(figsize=(12, 8))
plt.tight_layout()
```

## Next Steps

👉 See [Tutorial 3: GOF Tests](../tutorials/03_goodness_of_fit.md)

👉 See [Tutorial 8: Advanced Visualization](../tutorials/08_advanced_visualization.md)
