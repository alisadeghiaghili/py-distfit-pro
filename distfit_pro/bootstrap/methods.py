"""
Bootstrap Methods
================

Implementation of bootstrap confidence intervals and resampling methods.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from typing import Callable, Optional, Tuple, Dict, Union
from dataclasses import dataclass
from scipy import stats
from ..core.base import BaseDistribution


@dataclass
class BootstrapResult:
    """
    Bootstrap result with confidence intervals.
    
    Attributes
    ----------
    statistic : float
        Original statistic value
    bootstrap_distribution : ndarray
        Bootstrap distribution of the statistic
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    confidence_level : float
        Confidence level (e.g., 0.95)
    method : str
        Bootstrap method used
    n_bootstrap : int
        Number of bootstrap samples
    std_error : float
        Bootstrap standard error
    bias : float
        Bootstrap bias estimate
    """
    statistic: float
    bootstrap_distribution: np.ndarray
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    n_bootstrap: int
    std_error: float
    bias: float
    
    def summary(self) -> str:
        """
        Generate summary of bootstrap results.
        
        Returns
        -------
        summary : str
            Formatted summary
        """
        lines = [
            "="*60,
            "Bootstrap Results",
            "="*60,
            f"Method: {self.method}",
            f"Bootstrap samples: {self.n_bootstrap}",
            f"Confidence level: {self.confidence_level*100:.1f}%",
            "",
            f"Original statistic: {self.statistic:.6f}",
            f"Bootstrap mean: {np.mean(self.bootstrap_distribution):.6f}",
            f"Bootstrap bias: {self.bias:.6f}",
            f"Bootstrap SE: {self.std_error:.6f}",
            "",
            f"Confidence Interval:",
            f"  [{self.ci_lower:.6f}, {self.ci_upper:.6f}]",
            "="*60
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"BootstrapResult(statistic={self.statistic:.4f}, "
            f"CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}], "
            f"method='{self.method}')"
        )


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable,
    n_bootstrap: int = 10000,
    confidence_level: float = 0.95,
    method: str = 'percentile',
    random_state: Optional[int] = None
) -> BootstrapResult:
    """
    Calculate bootstrap confidence interval for a statistic.
    
    Parameters
    ----------
    data : array-like
        Original data
    statistic : callable
        Function that computes statistic from data
        Should accept array and return scalar
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    confidence_level : float, default=0.95
        Confidence level (between 0 and 1)
    method : str, default='percentile'
        Bootstrap CI method:
        - 'percentile': Simple percentile method
        - 'bca': Bias-corrected and accelerated (BCa)
        - 'basic': Basic bootstrap
        - 'studentized': Studentized bootstrap
    random_state : int, optional
        Random seed for reproducibility
    
    Returns
    -------
    result : BootstrapResult
        Bootstrap results with confidence interval
    
    Examples
    --------
    >>> data = np.random.normal(10, 2, 100)
    >>> result = bootstrap_ci(data, np.mean, n_bootstrap=10000)
    >>> print(result.summary())
    """
    np.random.seed(random_state)
    
    # Calculate original statistic
    original_stat = statistic(data)
    
    # Generate bootstrap samples
    n = len(data)
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(bootstrap_sample)
    
    # Calculate confidence interval based on method
    if method == 'percentile':
        ci_lower, ci_upper = _percentile_ci(
            bootstrap_stats, confidence_level
        )
    elif method == 'bca':
        ci_lower, ci_upper = _bca_ci(
            data, statistic, original_stat, bootstrap_stats, confidence_level
        )
    elif method == 'basic':
        ci_lower, ci_upper = _basic_ci(
            original_stat, bootstrap_stats, confidence_level
        )
    elif method == 'studentized':
        ci_lower, ci_upper = _studentized_ci(
            data, statistic, original_stat, bootstrap_stats, confidence_level
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate bootstrap statistics
    bootstrap_mean = np.mean(bootstrap_stats)
    bias = bootstrap_mean - original_stat
    std_error = np.std(bootstrap_stats, ddof=1)
    
    return BootstrapResult(
        statistic=original_stat,
        bootstrap_distribution=bootstrap_stats,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        confidence_level=confidence_level,
        method=method,
        n_bootstrap=n_bootstrap,
        std_error=std_error,
        bias=bias
    )


def parametric_bootstrap(
    distribution: BaseDistribution,
    n_samples: int,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: Optional[int] = None
) -> Dict[str, BootstrapResult]:
    """
    Parametric bootstrap for distribution parameters.
    
    Generates bootstrap samples from fitted distribution and
    re-estimates parameters to get confidence intervals.
    
    Parameters
    ----------
    distribution : BaseDistribution
        Fitted distribution
    n_samples : int
        Sample size for each bootstrap sample
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level
    random_state : int, optional
        Random seed
    
    Returns
    -------
    results : dict
        Dictionary mapping parameter names to BootstrapResult objects
    
    Examples
    --------
    >>> from distfit_pro import NormalDistribution
    >>> dist = NormalDistribution()
    >>> dist.fit(data)
    >>> results = parametric_bootstrap(dist, n_samples=100, n_bootstrap=1000)
    >>> print(results['loc'].summary())
    """
    if not distribution.fitted:
        raise ValueError("Distribution must be fitted first")
    
    np.random.seed(random_state)
    
    # Get parameter names
    param_names = list(distribution.params.keys())
    n_params = len(param_names)
    
    # Store bootstrap parameter estimates
    bootstrap_params = np.zeros((n_bootstrap, n_params))
    
    # Bootstrap loop
    for i in range(n_bootstrap):
        # Generate sample from fitted distribution
        bootstrap_sample = distribution.rvs(size=n_samples)
        
        # Create new distribution instance and fit
        bootstrap_dist = distribution.__class__()
        try:
            bootstrap_dist.fit(bootstrap_sample, method='mle')
            
            # Store parameters
            for j, param_name in enumerate(param_names):
                bootstrap_params[i, j] = bootstrap_dist.params[param_name]
        except:
            # If fitting fails, use original parameters
            for j, param_name in enumerate(param_names):
                bootstrap_params[i, j] = distribution.params[param_name]
    
    # Calculate confidence intervals for each parameter
    results = {}
    alpha = 1 - confidence_level
    
    for j, param_name in enumerate(param_names):
        original_value = distribution.params[param_name]
        bootstrap_dist = bootstrap_params[:, j]
        
        ci_lower = np.percentile(bootstrap_dist, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2))
        
        bias = np.mean(bootstrap_dist) - original_value
        std_error = np.std(bootstrap_dist, ddof=1)
        
        results[param_name] = BootstrapResult(
            statistic=original_value,
            bootstrap_distribution=bootstrap_dist,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            method='parametric',
            n_bootstrap=n_bootstrap,
            std_error=std_error,
            bias=bias
        )
    
    return results


def nonparametric_bootstrap(
    data: np.ndarray,
    distribution_class: type,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    method: str = 'mle',
    random_state: Optional[int] = None
) -> Dict[str, BootstrapResult]:
    """
    Non-parametric bootstrap for distribution parameters.
    
    Resamples original data and re-estimates parameters.
    
    Parameters
    ----------
    data : array-like
        Original data
    distribution_class : type
        Distribution class (e.g., NormalDistribution)
    n_bootstrap : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level
    method : str, default='mle'
        Fitting method
    random_state : int, optional
        Random seed
    
    Returns
    -------
    results : dict
        Dictionary mapping parameter names to BootstrapResult objects
    
    Examples
    --------
    >>> from distfit_pro import NormalDistribution
    >>> data = np.random.normal(10, 2, 100)
    >>> results = nonparametric_bootstrap(data, NormalDistribution, n_bootstrap=1000)
    >>> print(results['loc'].summary())
    """
    np.random.seed(random_state)
    
    # Fit original data
    original_dist = distribution_class()
    original_dist.fit(data, method=method)
    
    param_names = list(original_dist.params.keys())
    n_params = len(param_names)
    n = len(data)
    
    # Bootstrap loop
    bootstrap_params = np.zeros((n_bootstrap, n_params))
    
    for i in range(n_bootstrap):
        # Resample data
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        
        # Fit distribution
        bootstrap_dist = distribution_class()
        try:
            bootstrap_dist.fit(bootstrap_sample, method=method)
            
            for j, param_name in enumerate(param_names):
                bootstrap_params[i, j] = bootstrap_dist.params[param_name]
        except:
            # If fitting fails, use original parameters
            for j, param_name in enumerate(param_names):
                bootstrap_params[i, j] = original_dist.params[param_name]
    
    # Calculate confidence intervals
    results = {}
    alpha = 1 - confidence_level
    
    for j, param_name in enumerate(param_names):
        original_value = original_dist.params[param_name]
        bootstrap_dist = bootstrap_params[:, j]
        
        ci_lower = np.percentile(bootstrap_dist, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_dist, 100 * (1 - alpha / 2))
        
        bias = np.mean(bootstrap_dist) - original_value
        std_error = np.std(bootstrap_dist, ddof=1)
        
        results[param_name] = BootstrapResult(
            statistic=original_value,
            bootstrap_distribution=bootstrap_dist,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence_level,
            method='nonparametric',
            n_bootstrap=n_bootstrap,
            std_error=std_error,
            bias=bias
        )
    
    return results


# Helper functions for different CI methods

def _percentile_ci(
    bootstrap_stats: np.ndarray,
    confidence_level: float
) -> Tuple[float, float]:
    """Simple percentile confidence interval."""
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    return ci_lower, ci_upper


def _bca_ci(
    data: np.ndarray,
    statistic: Callable,
    original_stat: float,
    bootstrap_stats: np.ndarray,
    confidence_level: float
) -> Tuple[float, float]:
    """
    Bias-corrected and accelerated (BCa) confidence interval.
    
    More accurate than percentile method, especially for skewed distributions.
    """
    # Calculate bias correction
    z0 = stats.norm.ppf(np.mean(bootstrap_stats < original_stat))
    
    # Calculate acceleration using jackknife
    n = len(data)
    jackknife_stats = np.zeros(n)
    
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats[i] = statistic(jackknife_sample)
    
    jackknife_mean = np.mean(jackknife_stats)
    numerator = np.sum((jackknife_mean - jackknife_stats)**3)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_stats)**2)**1.5)
    
    if denominator == 0:
        a = 0
    else:
        a = numerator / denominator
    
    # Calculate adjusted percentiles
    alpha = 1 - confidence_level
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
    
    alpha_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
    alpha_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
    
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha_lower)
    ci_upper = np.percentile(bootstrap_stats, 100 * alpha_upper)
    
    return ci_lower, ci_upper


def _basic_ci(
    original_stat: float,
    bootstrap_stats: np.ndarray,
    confidence_level: float
) -> Tuple[float, float]:
    """Basic bootstrap confidence interval."""
    alpha = 1 - confidence_level
    lower_percentile = np.percentile(bootstrap_stats, 100 * alpha / 2)
    upper_percentile = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    # Reflect around original statistic
    ci_lower = 2 * original_stat - upper_percentile
    ci_upper = 2 * original_stat - lower_percentile
    
    return ci_lower, ci_upper


def _studentized_ci(
    data: np.ndarray,
    statistic: Callable,
    original_stat: float,
    bootstrap_stats: np.ndarray,
    confidence_level: float
) -> Tuple[float, float]:
    """
    Studentized (bootstrap-t) confidence interval.
    
    More accurate but computationally intensive.
    """
    # Simplified version - full studentized bootstrap requires nested bootstrap
    # This is an approximation using bootstrap standard error
    
    std_error = np.std(bootstrap_stats, ddof=1)
    alpha = 1 - confidence_level
    
    # Use t-distribution quantiles
    t_critical = stats.t.ppf(1 - alpha / 2, len(data) - 1)
    
    ci_lower = original_stat - t_critical * std_error
    ci_upper = original_stat + t_critical * std_error
    
    return ci_lower, ci_upper
