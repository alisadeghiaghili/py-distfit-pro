<!--
LICENSE NOTICE: py-distfit-pro uses a dual-license model.

• Personal/non-commercial use (research, education, internal analytics, personal projects):
  → Apache License 2.0 — free from day one

• Commercial/hosting use (SaaS, competing products, commercial redistribution):
  → BUSL-1.1 restrictions apply until 2030-09-05; contact licensor for permission

• Change Date: 2030-09-05 — entire work becomes Apache-2.0 for all uses

See LICENSE for full terms.
-->

# Data Unicorn 🦄

[![Language](https://img.shields.io/badge/language-Python-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0%20%2F%20BUSL--1.1-orange)](LICENSE)
[![Release](https://img.shields.io/badge/release-1.0.0-blue)](https://github.com/alisadeghiaghili/py-distfit-pro/releases)
[![CHANGELOG](https://img.shields.io/badge/CHANGELOG-latest-purple)](CHANGELOG.md)

**Professional-grade distribution fitting with statistical rigor and enterprise reliability.**

---

## 📋 Quick Navigation

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Core Concepts](#-core-concepts)
- [Examples](#-examples)
- [Documentation](#-documentation)
- [License](#-license)

---

## 🎯 Overview

Data Unicorn is a Python package for fitting univariate probability distributions to data with:

- **Statistical correctness** — validated estimators, proper p-values, and calibrated tests
- **Production reliability** — comprehensive test coverage, error handling, and deterministic results
- **Scalability** — out-of-core execution for datasets larger than memory

Designed for data engineers, ML practitioners, and statisticians who need trustworthy distribution fitting without hidden assumptions or silent failures.

---

## ✨ Key Features

- **Comprehensive distribution support**: Gamma, Log-Normal, Weibull, Beta, and more
- **Multiple fitting methods**: MLE, Method of Moments, L-Moments, Quantile Matching
- **Goodness-of-fit testing**: Anderson-Darling, Kolmogorov-Smirnov, Chi-Square with proper p-values
- **Model selection**: AIC, AICc, BIC for principled comparison
- **Bootstrap confidence intervals**: For parameters and statistics
- **Out-of-core processing**: Fit distributions to data larger than RAM
- **Deterministic results**: Reproducible fits with explicit RNG control

---

## 📦 Installation

```bash
pip install py-distfit-pro
```

Or from source:

```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e .
```

---

## 🚀 Quick Start

```python
import numpy as np
from distfit_pro import fit_distribution

# Generate sample data
data = np.random.gamma(shape=2.0, scale=1.5, size=10000)

# Fit a Gamma distribution
result = fit_distribution(data, family="gamma")

print(f"Estimated shape: {result.params['shape']:.4f}")
print(f"Estimated scale: {result.params['scale']:.4f}")
print(f"AIC: {result.aic:.4f}")
print(f"Anderson-Darling p-value: {result.gof['anderson_darling']['p_value']:.4f}")
```

---

## 📚 Core Concepts

### Fitting Methods

- **MLE** (Maximum Likelihood Estimation) — default, asymptotically efficient
- **Method of Moments** — fast, closed-form for many families
- **L-Moments** — robust to outliers, available for select families
- **Quantile Matching** — useful for heavy-tailed distributions

### Goodness-of-Fit

- **Anderson-Darling** — sensitive to tail behavior, preferred for most applications
- **Kolmogorov-Smirnov** — general-purpose, less power in tails
- **Chi-Square** — requires binning, useful for visual validation

All p-values are computed using family-specific critical values or bootstrap calibration where analytical results are unavailable.

### Model Selection

- **AIC** — Akaike Information Criterion
- **AICc** — AIC with small-sample correction
- **BIC** — Bayesian Information Criterion (stronger penalty for complexity)

Lower values indicate better fit, penalizing over-parameterization.

---

## 📖 Examples

See the `examples/` directory for complete workflows:

- `examples/basic_fitting.py` — single distribution fit
- `examples/model_comparison.py` — comparing multiple families
- `examples/bootstrap_ci.py` — confidence intervals via bootstrap
- `examples/large_data.py` — out-of-core fitting with generators

---

## 📄 Documentation

- [API Reference](docs/api.md)
- [Statistical Methods](docs/methods.md)
- [Out-of-Core Guide](docs/streaming.md)
- [Contributing](CONTRIBUTING.md)
- [CHANGELOG](CHANGELOG.md)

---

## ⚖️ License

**py-distfit-pro** uses a **dual-license model**:

### ✅ Apache License 2.0 (from day one)

Free for **personal and non-commercial use**, including:

- 🎓 Academic research and education
- 💼 Internal business analytics (not offered as a service)
- 🧪 Personal projects and experimentation
- 🤝 Non-profit and open-source community work

### ⚠️ BUSL-1.1 restrictions (until 2030-09-05)

Applies if your use includes:

- ☁️ Offering py-distfit-pro as a hosted/cloud service (SaaS)
- 🏢 Embedding in a product that competes with the author's paid offering
- 💰 Commercial redistribution without modification

For these cases, **contact the licensor** for explicit permission or a commercial license.

### 📅 Change Date: 2030-09-05

On this date, **the entire work becomes Apache-2.0** for all uses, with no restrictions.

For full terms, see the [LICENSE](LICENSE) file.

---

## 📬 Contact

- **Author**: Ali Sadeghi Aghili
- **Email**: alisadeghiaghili@gmail.com
- **GitHub**: [@alisadeghiaghili](https://github.com/alisadeghiaghili)

---

**Built with statistical rigor for production environments.** 🦄
