"""
Distribution Implementations
============================

Concrete implementations of all probability distributions.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional
from .base import (
    ContinuousDistribution,
    DiscreteDistribution,
    DistributionInfo
)


# ============================================================================
# CONTINUOUS DISTRIBUTIONS
# ============================================================================

class NormalDistribution(ContinuousDistribution):
    """
    Normal (Gaussian) Distribution.
    
    The most fundamental continuous probability distribution.
    
    Parameters
    ----------
    loc : float
        Mean (μ)
    scale : float
        Standard deviation (σ > 0)
    
    Support
    -------
    x ∈ (-∞, ∞)
    
    PDF
    ---
    f(x|μ,σ) = (1/(σ√(2π))) * exp(-(x-μ)²/(2σ²))
    
    Applications
    -----------
    - Natural phenomena (heights, weights, measurement errors)
    - Central Limit Theorem applications
    - Statistical inference
    - Financial returns (approximation)
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.norm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="normal",
            scipy_name="norm",
            display_name="Normal Distribution",
            description="Symmetric bell-shaped continuous distribution. "
                       "Fundamental in statistics due to Central Limit Theorem. "
                       "Characterized by mean (location) and standard deviation (scale).",
            parameters=["loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        """
        Maximum Likelihood Estimation for Normal distribution.
        
        MLE for Normal:
        - μ_hat = sample mean
        - σ_hat = sample standard deviation (MLE uses n, not n-1)
        """
        # scipy.stats.norm.fit uses MLE
        loc, scale = self._scipy_dist.fit(data)
        
        self._params = {
            'loc': loc,
            'scale': scale
        }
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """
        Method of Moments for Normal distribution.
        
        For Normal, MoM coincides with MLE:
        - μ = E[X] = sample mean
        - σ² = Var[X] = sample variance
        """
        loc = np.mean(data)
        scale = np.std(data, ddof=1)  # Use unbiased estimator
        
        self._params = {
            'loc': loc,
            'scale': scale
        }
    
    def mode(self) -> float:
        """
        Mode of Normal distribution.
        
        For Normal distribution, mode = mean = median
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        """Convert parameters to scipy format"""
        return {
            'loc': self._params['loc'],
            'scale': self._params['scale']
        }


class ExponentialDistribution(ContinuousDistribution):
    """
    Exponential Distribution.
    
    Memoryless distribution modeling time between events.
    
    Parameters
    ----------
    scale : float
        Scale parameter (1/λ), where λ is the rate parameter (> 0)
    
    Support
    -------
    x ∈ [0, ∞)
    
    PDF
    ---
    f(x|λ) = λ * exp(-λx) for x ≥ 0
    
    Applications
    -----------
    - Time between events in Poisson process
    - Lifetime/reliability analysis
    - Queueing theory
    - Radioactive decay
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.expon
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="exponential",
            scipy_name="expon",
            display_name="Exponential Distribution",
            description="Memoryless continuous distribution modeling waiting times. "
                       "Characterized by constant hazard rate. "
                       "Natural model for time between independent events.",
            parameters=["scale"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        """
        MLE for Exponential distribution.
        
        MLE: λ_hat = 1 / sample_mean
        So scale_hat = sample_mean
        """
        if np.any(data < 0):
            raise ValueError("Exponential distribution requires non-negative data")
        
        # scipy.expon.fit fixes loc=0 by default
        loc, scale = self._scipy_dist.fit(data, floc=0)
        
        self._params = {
            'scale': scale
        }
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """
        Method of Moments for Exponential.
        
        E[X] = 1/λ = scale
        So scale = sample_mean
        """
        if np.any(data < 0):
            raise ValueError("Exponential distribution requires non-negative data")
        
        scale = np.mean(data)
        
        self._params = {
            'scale': scale
        }
    
    def mode(self) -> float:
        """
        Mode of Exponential distribution.
        
        Mode is always 0 for exponential distribution.
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        """Convert parameters to scipy format"""
        return {
            'loc': 0,
            'scale': self._params['scale']
        }


# ============================================================================
# DISTRIBUTION REGISTRY
# ============================================================================

# Registry mapping names to distribution classes
_DISTRIBUTION_REGISTRY = {
    'normal': NormalDistribution,
    'exponential': ExponentialDistribution,
    # More distributions will be added here
}


def get_distribution(name: str) -> ContinuousDistribution:
    """
    Get distribution class by name.
    
    Parameters
    ----------
    name : str
        Distribution name (e.g., 'normal', 'exponential')
    
    Returns
    -------
    distribution : BaseDistribution
        Distribution instance
    
    Raises
    ------
    ValueError
        If distribution name is not recognized
    
    Examples
    --------
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    """
    name_lower = name.lower()
    
    if name_lower not in _DISTRIBUTION_REGISTRY:
        available = ", ".join(_DISTRIBUTION_REGISTRY.keys())
        raise ValueError(
            f"Unknown distribution: {name}. "
            f"Available distributions: {available}"
        )
    
    return _DISTRIBUTION_REGISTRY[name_lower]()


def list_distributions() -> list:
    """
    List all available distributions.
    
    Returns
    -------
    distributions : list of str
        Names of available distributions
    """
    return sorted(_DISTRIBUTION_REGISTRY.keys())
