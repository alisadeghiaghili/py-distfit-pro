# 📚 DistFit-Pro Examples

Comprehensive, copy-paste ready examples organized by use case.

## 🎯 Quick Navigation

### 01. Basic Usage
**Start here if you're new!**
- [`01_quick_start.py`](01_basic/01_quick_start.py) - 5-minute introduction to core features
- [`02_fitting_single_dist.py`](01_basic/02_fitting_single_dist.py) - Complete workflow for one distribution
- [`03_comparing_methods.py`](01_basic/03_comparing_methods.py) - MLE vs Method of Moments

### 02. Distribution Types
**Learn about available distributions**
- [`01_continuous.py`](02_distributions/01_continuous.py) - Normal, Exponential, Lognormal, etc.
- [`02_discrete.py`](02_distributions/02_discrete.py) - Poisson, Binomial, Geometric
- [`03_specialized.py`](02_distributions/03_specialized.py) - Weibull, Gamma for reliability analysis

### 03. Advanced Fitting
**Handle complex data scenarios**
- [`01_weighted_data.py`](03_advanced_fitting/01_weighted_data.py) - Survey sampling weights
- [`02_censored_data.py`](03_advanced_fitting/02_censored_data.py) - Survival/reliability data
- [`03_custom_optimization.py`](03_advanced_fitting/03_custom_optimization.py) - Custom parameter bounds

### 04. Model Selection
**Choose the best distribution**
- [`01_aic_bic_comparison.py`](04_model_selection/01_aic_bic_comparison.py) - Compare multiple distributions
- [`02_goodness_of_fit.py`](04_model_selection/02_goodness_of_fit.py) - KS test, Q-Q plots
- [`03_cross_validation.py`](04_model_selection/03_cross_validation.py) - Validate on hold-out data

### 05. Visualization
**Beautiful, informative plots**
- [`01_basic_plots.py`](05_visualization/01_basic_plots.py) - PDF/CDF overlays
- [`02_diagnostics.py`](05_visualization/02_diagnostics.py) - Residuals, Q-Q, P-P plots
- [`03_interactive.py`](05_visualization/03_interactive.py) - Interactive Plotly dashboards

### 06. Real-World Applications
**Industry use cases**
- [`01_failure_analysis.py`](06_real_world/01_failure_analysis.py) - Manufacturing defect rates
- [`02_financial_returns.py`](06_real_world/02_financial_returns.py) - Stock market returns
- [`03_queue_modeling.py`](06_real_world/03_queue_modeling.py) - Service center wait times

### 07. Integration
**Use with other tools**
- [`01_pandas_workflow.py`](07_integration/01_pandas_workflow.py) - DataFrame integration
- [`02_sklearn_pipeline.py`](07_integration/02_sklearn_pipeline.py) - Scikit-learn pipelines
- [`03_parallel_fitting.py`](07_integration/03_parallel_fitting.py) - Multi-core batch processing

---

## 🚀 Running Examples

### Prerequisites
```bash
pip install distfit-pro matplotlib scipy pandas
```

### Run Any Example
```bash
python examples/01_basic/01_quick_start.py
```

### In Jupyter/Colab
```python
# Copy-paste the code directly into a cell
# All examples are self-contained
```

---

## 📖 Example Structure

Each example follows this pattern:

```python
"""
Clear title and purpose
========================

What you'll learn:
- Point 1
- Point 2

Real-world context: When to use this
"""

# 1. Generate realistic data
# 2. Fit distribution
# 3. Evaluate results
# 4. Visualize (optional)
```

---

## 💡 Tips

1. **Start simple**: Begin with `01_quick_start.py`
2. **Copy-paste**: All examples run standalone
3. **Modify**: Change data to match your use case
4. **Combine**: Mix techniques from multiple examples
5. **Ask**: Open an issue if something's unclear

---

## 📚 More Resources

- [API Documentation](../docs/api/)
- [User Guides](../docs/guides/)
- [GitHub Issues](https://github.com/alisadeghiaghili/py-distfit-pro/issues)

---

**Made with ❤️ by the DistFit-Pro team**
