"""
distfit-pro: Professional Distribution Fitting Library
=====================================================

A comprehensive library for fitting probability distributions to data,
with enhanced goodness-of-fit testing, bootstrap confidence intervals,
and beautiful visualizations.

Author: Ali Sadeghi Aghili
Version: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "Ali Sadeghi Aghili"
__email__ = "alisadeghiaghili@gmail.com"

# Core distributions
from .core.distributions import (
    NormalDistribution,
    ExponentialDistribution,
    get_distribution,
    list_distributions,
)

# Base classes (for advanced users)
from .core.base import (
    BaseDistribution,
    ContinuousDistribution,
    DiscreteDistribution,
    DistributionInfo,
    FittingMethod,
)


__all__ = [
    # Version info
    "__version__",
    "__author__",
    
    # Distributions
    "NormalDistribution",
    "ExponentialDistribution",
    "get_distribution",
    "list_distributions",
    
    # Base classes
    "BaseDistribution",
    "ContinuousDistribution",
    "DiscreteDistribution",
    "DistributionInfo",
    "FittingMethod",
]
