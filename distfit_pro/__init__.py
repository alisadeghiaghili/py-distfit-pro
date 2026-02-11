"""
DistFit Pro - Professional Distribution Fitting for Python
===========================================================

A comprehensive package for statistical distribution fitting with:
- 30 distributions (25 continuous, 5 discrete)
- Goodness-of-fit tests (KS, AD, Chi-Square, CvM)
- Bootstrap confidence intervals
- Enhanced diagnostics
- Weighted data support
- Multilingual support (EN/FA/DE)

Quick Start
-----------
>>> from distfit_pro import get_distribution
>>> import numpy as np
>>> 
>>> data = np.random.normal(10, 2, 1000)
>>> dist = get_distribution('normal')
>>> dist.fit(data)
>>> print(dist.summary())

Modules
-------
- core.distributions: Distribution classes
- core.gof_tests: Goodness-of-fit tests
- core.bootstrap: Bootstrap confidence intervals
- core.diagnostics: Enhanced diagnostics
- core.weighted: Weighted data support
- plotting: Visualization tools
- locales: Multilingual support
"""

__version__ = "1.0.0"
__author__ = "Ali Sadeghi Aghili"
__email__ = ""
__license__ = "MIT"
__url__ = "https://github.com/alisadeghiaghili/py-distfit-pro"

from .core.distributions import (
    get_distribution,
    list_distributions,
    list_continuous_distributions,
    list_discrete_distributions,
)

from .locales import set_language, get_language, t

__all__ = [
    # Version
    "__version__",
    
    # Distribution functions
    "get_distribution",
    "list_distributions",
    "list_continuous_distributions",
    "list_discrete_distributions",
    
    # Localization
    "set_language",
    "get_language",
    "t",
]
