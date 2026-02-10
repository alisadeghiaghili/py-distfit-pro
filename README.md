# DistFit Pro 🎯

**Professional Distribution Fitting for Python**

A comprehensive, production-ready package that combines the best features of EasyFit and R's fitdistrplus, with modern improvements in statistical methodology, user experience, and software engineering.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Why DistFit Pro?

### Better Statistical Philosophy
- ✅ **Model selection via AIC/BIC/WAIC/LOO-CV** instead of just p-values
- ✅ **Bayesian model averaging** for robust inference
- ✅ **Automatic tail behavior detection** and outlier diagnosis
- ✅ **Multiple testing correction** to avoid false positives

### Better User Experience
- ✅ **Scikit-learn-like API** - intuitive and consistent
- ✅ **Rich visualizations** with matplotlib/seaborn/plotly
- ✅ **Self-explanatory outputs** - every step is documented
- ✅ **Comprehensive documentation** and tutorials

### Better Extensibility
- ✅ **Custom distributions** made easy
- ✅ **Mixture models** built-in
- ✅ **Hierarchical/multilevel fitting** support
- ✅ **Modular architecture** for easy extension

### Better Performance
- ✅ **Optimized for large datasets**
- ✅ **Parallel processing** via joblib
- ✅ **GPU acceleration** (optional, via CuPy)
- ✅ **Efficient algorithms** with numba JIT

### Better Software Engineering
- ✅ **Comprehensive test suite** (90%+ coverage)
- ✅ **CI/CD ready** with GitHub Actions
- ✅ **Version controlled** and reproducible
- ✅ **Type hints** and mypy compatible

---

## 📦 Installation

```bash
pip install distfit-pro
```

For development:
```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e ".[dev]"
```

---

## 🎯 Quick Start

```python
import numpy as np
from distfit_pro import DistributionFitter

# Generate sample data
np.random.seed(42)
data = np.random.lognormal(mean=2, sigma=0.5, size=1000)

# Fit distributions
fitter = DistributionFitter(data)
results = fitter.fit(
    distributions=['lognormal', 'gamma', 'weibull', 'normal'],
    method='mle',  # or 'moments', 'quantile', 'bayesian'
    n_jobs=-1  # parallel processing
)

# Print self-explanatory results
print(results.summary())

# Visualize
results.plot(kind='comparison')  # P-P, Q-Q, PDF, CDF
results.plot(kind='diagnostics')  # Residuals, tail behavior

# Get best model with explanation
best = results.get_best(criterion='aic')
print(best.explanation)
```

---

## 📚 Core Features

### Comprehensive Distribution Support

**Continuous (30+):** Normal, Lognormal, Exponential, Gamma, Weibull, Beta, Chi-square, Student-t, F, Cauchy, Pareto, Gumbel, GEV, Rayleigh, and more...

**Discrete (15+):** Poisson, Binomial, Negative Binomial, Geometric, Hypergeometric, and more...

### Advanced Estimation Methods

- **Maximum Likelihood (MLE)** - efficient and asymptotically optimal
- **Method of Moments** - robust to outliers
- **Quantile Matching** - fits specific percentiles
- **Bayesian Estimation** - full posterior with uncertainty

### Model Selection Criteria

- **AIC/BIC** - penalized likelihood
- **WAIC** - Bayesian information criterion  
- **LOO-CV** - leave-one-out cross-validation
- **Bayesian Model Averaging** - weighted ensemble

---

## 🔬 Advanced Examples

### Reliability Engineering

```python
failure_times = np.array([120, 145, 167, 189, 201])
censored = np.array([0, 0, 0, 1, 0])  # 1=censored

fitter = DistributionFitter(
    data=failure_times,
    censoring=censored,
    censoring_type='right'
)

results = fitter.fit(['weibull', 'lognormal', 'gamma'])
reliability = results.best_model.reliability(t=200)
print(f"Reliability at t=200h: {reliability:.3f}")
```

### Financial Risk (VaR)

```python
returns = load_stock_returns('AAPL')
fitter = DistributionFitter(returns)
results = fitter.fit(['normal', 'student_t', 'gev'])

var_99 = results.best_model.ppf(0.01)
print(f"VaR(99%): {var_99:.2%}")
```

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

## 📞 Contact

**Ali Sadeghi Aghili**  
- Website: [zil.ink/thedatascientist](https://zil.ink/thedatascientist)  
- LinkTree: [linktr.ee/aliaghili](https://linktr.ee/aliaghili)
- GitHub: [@alisadeghiaghili](https://github.com/alisadeghiaghili)

---

**Made with ❤️ in Frankfurt & Tehran**
