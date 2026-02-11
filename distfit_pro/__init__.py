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

Phase 2 Features:
----------------
>>> from distfit_pro.core.gof_tests import GoodnessOfFitTests
>>> from distfit_pro.core.bootstrap import Bootstrap
>>> from distfit_pro.core.diagnostics import Diagnostics
>>> from distfit_pro.core.weighted import WeightedDistributionFitter
>>> from distfit_pro.core.mixture import MixtureModel
>>> 
>>> # GOF tests
>>> gof = GoodnessOfFitTests()
>>> ks_result = gof.kolmogorov_smirnov(data, fitted_dist)
>>> ad_result = gof.anderson_darling(data, fitted_dist)
>>> 
>>> # Bootstrap confidence intervals
>>> bs = Bootstrap(n_bootstrap=1000, n_jobs=-1)
>>> ci = bs.parametric(data, fitted_dist)
>>> 
>>> # Diagnostics
>>> diag = Diagnostics()
>>> residuals = diag.compute_residuals(data, fitted_dist)
>>> outliers = diag.detect_outliers(data, fitted_dist)
>>> influence = diag.analyze_influence(data, fitted_dist)
>>> 
>>> # Weighted fitting
>>> weights = np.ones_like(data)
>>> fitter = WeightedDistributionFitter()
>>> params = fitter.fit(data, weights, fitted_dist)
>>> 
>>> # Mixture models
>>> mixture = MixtureModel(n_components=2, distribution_name='normal')
>>> mixture.fit(data)
"""

__version__ = "0.2.0"  # Phase 2 complete
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

# Phase 2 imports
from .core.gof_tests import GoodnessOfFitTests, format_gof_results
from .core.bootstrap import Bootstrap, BootstrapResult, format_bootstrap_results
from .core.diagnostics import (
    Diagnostics, 
    ResidualAnalysis, 
    OutlierDetection, 
    InfluenceAnalysis,
    format_diagnostics
)
from .core.weighted import WeightedDistributionFitter
from .core.mixture import MixtureModel, MixtureComponent

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
    
    # Phase 2: Goodness-of-Fit Tests
    'GoodnessOfFitTests',
    'format_gof_results',
    
    # Phase 2: Bootstrap
    'Bootstrap',
    'BootstrapResult',
    'format_bootstrap_results',
    
    # Phase 2: Diagnostics
    'Diagnostics',
    'ResidualAnalysis',
    'OutlierDetection',
    'InfluenceAnalysis',
    'format_diagnostics',
    
    # Phase 2: Weighted & Mixture
    'WeightedDistributionFitter',
    'MixtureModel',
    'MixtureComponent',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
]
