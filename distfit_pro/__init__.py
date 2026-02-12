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
>>> from distfit_pro import DistributionFitter
>>> import numpy as np
>>>
>>> # Generate sample data
>>> data = np.random.normal(10, 2, 1000)
>>>
>>> # Fit distributions
>>> fitter = DistributionFitter(data)
>>> results = fitter.fit()
>>> print(results.summary())
>>>
>>> # Plot results
>>> results.plot(kind='comparison')

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
)

# ============================================================================
# VISUALIZATION
# ============================================================================

from .visualization import (
    DistributionPlotter,
)

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
    
    # === VISUALIZATION ===
    "DistributionPlotter",
]


# ============================================================================
# PACKAGE METADATA
# ============================================================================

def get_info() -> dict:
    """
    Get package information.
    
    Returns
    -------
    info : dict
        Package metadata
    """
    return {
        'name': 'distfit-pro',
        'version': __version__,
        'author': __author__,
        'email': __email__,
        'license': __license__,
        'n_distributions': 25,
        'n_continuous': 20,
        'n_discrete': 5,
        'n_gof_tests': 4,
        'features': [
            'Multiple fitting methods',
            'Goodness-of-fit tests',
            'Bootstrap confidence intervals',
            'Professional visualizations',
            'Multilingual support',
        ]
    }


def list_all_distributions() -> dict:
    """
    List all available distributions grouped by type.
    
    Returns
    -------
    distributions : dict
        Dictionary with 'continuous' and 'discrete' lists
    """
    return {
        'continuous': list_distributions(continuous_only=True),
        'discrete': list_distributions(discrete_only=True),
        'total': len(list_distributions())
    }


def show_welcome_message():
    """
    Show welcome message with package info.
    """
    print("="*70)
    print("  distfit-pro v{}".format(__version__))
    print("  Professional Distribution Fitting Library")
    print("="*70)
    print("\nFeatures:")
    print("  • 25 probability distributions")
    print("  • 4 goodness-of-fit tests")
    print("  • Bootstrap confidence intervals")
    print("  • Professional visualizations")
    print("  • Multilingual support")
    print("\nQuick start:")
    print("  >>> from distfit_pro import DistributionFitter")
    print("  >>> fitter = DistributionFitter(data)")
    print("  >>> results = fitter.fit()")
    print("  >>> results.plot()")
    print("\nDocumentation: https://github.com/alisadeghiaghili/py-distfit-pro")
    print("="*70)
