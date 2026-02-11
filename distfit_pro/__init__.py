"""
DistFit Pro - Professional Distribution Fitting Package
========================================================

A comprehensive, production-ready Python package for statistical
distribution fitting with model selection, diagnostics, and explanations.

Author: Ali Sadeghi Aghili
Website: https://linktr.ee/aliaghili
GitHub: https://github.com/alisadeghiaghili/py-distfit-pro

Language Support:
-----------------
The package supports multilingual output in:
- English (en) - Default
- Persian/Farsi (fa)
- German (de)

Example:
--------
>>> from distfit_pro import set_language, DistributionFitter
>>> import numpy as np
>>> 
>>> # Set output language
>>> set_language('en')  # English
>>> # set_language('fa')  # Persian
>>> # set_language('de')  # German
>>> 
>>> # Fit distribution
>>> data = np.random.normal(10, 2, 1000)
>>> fitter = DistributionFitter()
>>> results = fitter.fit(data)
>>> print(results.best_distribution.summary())  # Output in selected language
"""

__version__ = "0.1.0"
__author__ = "Ali Sadeghi Aghili"
__email__ = "alisadeghiaghili@gmail.com"

# Configuration
from .core.config import set_language, get_language

# Core imports
from .fitting.fitter import DistributionFitter, FitResults
from .core.distributions import (
    BaseDistribution,
    NormalDistribution,
    LognormalDistribution,
    WeibullDistribution,
    GammaDistribution,
    ExponentialDistribution,
    get_distribution,
    list_distributions
)
from .core.model_selection import ModelSelection, ModelScore, DeltaComparison
from .visualization.plots import DistributionPlotter

# Public API
__all__ = [
    # Configuration
    'set_language',
    'get_language',
    
    # Main fitter
    'DistributionFitter',
    'FitResults',
    
    # Distributions
    'BaseDistribution',
    'NormalDistribution',
    'LognormalDistribution',
    'WeibullDistribution',
    'GammaDistribution',
    'ExponentialDistribution',
    'get_distribution',
    'list_distributions',
    
    # Model selection
    'ModelSelection',
    'ModelScore',
    'DeltaComparison',
    
    # Visualization
    'DistributionPlotter',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
]
