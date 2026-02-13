"""
Fitting Module
==============

High-level API for distribution fitting.
"""

from .fitter import DistributionFitter, FitResults
from .api import (
    fit,
    find_best_distribution,
    compare_distributions,
    FitResult,
    ComparisonResult,
)

__all__ = [
    'DistributionFitter',
    'FitResults',
    'FitResult',
    'fit',
    'find_best_distribution',
    'compare_distributions',
    'ComparisonResult',
]
