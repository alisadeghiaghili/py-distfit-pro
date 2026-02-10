"""
DistFit Pro - Professional Distribution Fitting Package
========================================================

A comprehensive, production-ready Python package for statistical
distribution fitting with model selection, diagnostics, and explanations.

Author: Ali Sadeghi Aghili
Website: https://linktr.ee/aliaghili
GitHub: https://github.com/alisadeghiaghili/py-distfit-pro
"""

__version__ = "0.1.0"
__author__ = "Ali Sadeghi Aghili"
__email__ = "alisadeghiaghili@gmail.com"

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
