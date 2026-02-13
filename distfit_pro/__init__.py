"""
distfit-pro: Professional Distribution Fitting Library
=====================================================

A comprehensive library for fitting probability distributions to data,
with enhanced goodness-of-fit testing, bootstrap confidence intervals,
and beautiful visualizations.

Author: Ali Sadeghi Aghili
Email: alisadeghiaghili@gmail.com
Version: 0.1.0
License: MIT

Features:
---------
- 25 probability distributions (20 continuous + 5 discrete)
- Multiple fitting methods (MLE, Method of Moments)
- 4 goodness-of-fit tests (KS, AD, Chi-Square, CvM)
- Bootstrap confidence intervals (parametric & non-parametric)
- Professional visualizations (matplotlib & plotly)
- Multilingual support (English, Persian, German)
- High-level API for easy usage

Quick Start:
-----------
>>> from distfit_pro import fit
>>> import numpy as np
>>>
>>> # Generate sample data
>>> data = np.random.normal(10, 2, 1000)
>>>
>>> # Quick fit
>>> result = fit(data, 'normal')
>>> print(result.summary())

For more examples, see: https://github.com/alisadeghiaghili/py-distfit-pro
"""

__version__ = "0.1.0"
__author__ = "Ali Sadeghi Aghili"
__email__ = "alisadeghiaghili@gmail.com"
__license__ = "MIT"

# ============================================================================
# CORE DISTRIBUTIONS
# ============================================================================

from .core.distributions import (
    # Continuous distributions
    NormalDistribution,
    ExponentialDistribution,
    UniformDistribution,
    GammaDistribution,
    BetaDistribution,
    WeibullDistribution,
    LognormalDistribution,
    LogisticDistribution,
    GumbelDistribution,
    ParetoDistribution,
    CauchyDistribution,
    StudentTDistribution,
    ChiSquareDistribution,
    FDistribution,
    LaplaceDistribution,
    RayleighDistribution,
    WaldDistribution,
    TriangularDistribution,
    BurrDistribution,
    GenExtremeDistribution,
    
    # Discrete distributions
    PoissonDistribution,
    BinomialDistribution,
    NegativeBinomialDistribution,
    GeometricDistribution,
    HypergeometricDistribution,
    
    # Utilities
    get_distribution,
    list_distributions,
)

# Base classes for advanced users
from .core.base import (
    BaseDistribution,
    ContinuousDistribution,
    DiscreteDistribution,
    DistributionInfo,
    FittingMethod,
)

# ============================================================================
# GOODNESS-OF-FIT TESTS
# ============================================================================

from .gof import (
    GOFTest,
    GOFResult,
    KolmogorovSmirnovTest,
    AndersonDarlingTest,
    ChiSquareTest,
    CramerVonMisesTest,
)

# Alias for compatibility
GOFTestResult = GOFResult

# ============================================================================
# BOOTSTRAP METHODS
# ============================================================================

from .bootstrap import (
    bootstrap_ci,
    parametric_bootstrap,
    nonparametric_bootstrap,
    BootstrapResult,
    bootstrap_hypothesis_test,
    bootstrap_goodness_of_fit,
)

# ============================================================================
# HIGH-LEVEL API
# ============================================================================

from .fitting import (
    DistributionFitter,
    FitResults,
    FitResult,
    fit,
    find_best_distribution,
    compare_distributions,
    ComparisonResult,
)

# ============================================================================
# VISUALIZATION
# ============================================================================

from .visualization import (
    DistributionPlotter,
)

# ============================================================================
# CONFIG
# ============================================================================

from .core.config import config

# Expose config functions at top level
def set_language(lang: str):
    """Set display language: 'en', 'fa', 'de'"""
    config.set_language(lang)

def set_verbosity(level: str):
    """Set verbosity: 'silent', 'normal', 'verbose', 'debug'"""
    config.set_verbosity(level)

def get_language() -> str:
    """Get current language"""
    return config.language

def get_verbosity() -> str:
    """Get current verbosity level"""
    return config.verbosity

# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    
    # === CONTINUOUS DISTRIBUTIONS (20) ===
    "NormalDistribution",
    "ExponentialDistribution",
    "UniformDistribution",
    "GammaDistribution",
    "BetaDistribution",
    "WeibullDistribution",
    "LognormalDistribution",
    "LogisticDistribution",
    "GumbelDistribution",
    "ParetoDistribution",
    "CauchyDistribution",
    "StudentTDistribution",
    "ChiSquareDistribution",
    "FDistribution",
    "LaplaceDistribution",
    "RayleighDistribution",
    "WaldDistribution",
    "TriangularDistribution",
    "BurrDistribution",
    "GenExtremeDistribution",
    
    # === DISCRETE DISTRIBUTIONS (5) ===
    "PoissonDistribution",
    "BinomialDistribution",
    "NegativeBinomialDistribution",
    "GeometricDistribution",
    "HypergeometricDistribution",
    
    # === UTILITIES ===
    "get_distribution",
    "list_distributions",
    
    # === BASE CLASSES ===
    "BaseDistribution",
    "ContinuousDistribution",
    "DiscreteDistribution",
    "DistributionInfo",
    "FittingMethod",
    
    # === GOODNESS-OF-FIT TESTS (4) ===
    "GOFTest",
    "GOFResult",
    "GOFTestResult",  # Alias
    "KolmogorovSmirnovTest",
    "AndersonDarlingTest",
    "ChiSquareTest",
    "CramerVonMisesTest",
    
    # === BOOTSTRAP METHODS ===
    "bootstrap_ci",
    "parametric_bootstrap",
    "nonparametric_bootstrap",
    "BootstrapResult",
    "bootstrap_hypothesis_test",
    "bootstrap_goodness_of_fit",
    
    # === HIGH-LEVEL API ===
    "DistributionFitter",
    "FitResults",
    "FitResult",
    "fit",
    "find_best_distribution",
    "compare_distributions",
    "ComparisonResult",
    
    # === VISUALIZATION ===
    "DistributionPlotter",
    
    # === CONFIG ===
    "config",
    "set_language",
    "set_verbosity",
    "get_language",
    "get_verbosity",
]
