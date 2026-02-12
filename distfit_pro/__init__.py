"""
DistFit Pro - Professional Distribution Fitting for Python
===========================================================

A comprehensive library for statistical distribution fitting that surpasses
EasyFit and R's fitdistrplus.

Features:
---------
- 30 statistical distributions (25 continuous + 5 discrete)
- Multiple estimation methods (MLE, Moments, Quantile)
- Goodness-of-fit tests (KS, AD, Chi-Square, Cramér-von Mises)
- Bootstrap confidence intervals (Parametric & Non-parametric)
- Enhanced diagnostics (Residuals, Influence, Outliers)
- Weighted data support
- Multilingual (English, فارسی, Deutsch)

Quick Start:
------------
>>> from distfit_pro import get_distribution
>>> import numpy as np
>>>
>>> data = np.random.normal(10, 2, 1000)
>>> dist = get_distribution('normal')
>>> dist.fit(data)
>>> print(dist.summary())

See Also:
---------
- Documentation: https://github.com/alisadeghiaghili/py-distfit-pro/docs
- Tutorials: docs/source/tutorial/
- Examples: docs/source/examples/

Author: Ali Sadeghi Aghili
License: MIT
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Ali Sadeghi Aghili"
__license__ = "MIT"
__all__ = [
    "get_distribution",
    "list_distributions",
    "list_continuous_distributions",
    "list_discrete_distributions",
    "set_language",
    "get_language",
]

from .core.distributions import (
    get_distribution,
    list_distributions,
    list_continuous_distributions,
    list_discrete_distributions,
)

from .locales import set_language, get_language

# Version info
version_info = (1, 0, 0)
