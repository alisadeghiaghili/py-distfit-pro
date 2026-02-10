"""
DistFit Pro - Professional Distribution Fitting Package
======================================================

Author: Ali Aghili
Website: https://zil.ink/thedatascientist
"""

__version__ = "0.1.0"
__author__ = "Ali Aghili"

from .fitting.fitter import DistributionFitter
from .core.distributions import get_distribution, list_distributions

__all__ = [
    'DistributionFitter',
    'get_distribution', 
    'list_distributions'
]
