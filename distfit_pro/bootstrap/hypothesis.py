"""
Bootstrap Hypothesis Testing
============================

Bootstrap methods for hypothesis testing and goodness-of-fit.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from typing import Optional, Callable
from dataclasses import dataclass
from ..core.base import BaseDistribution


@dataclass
class BootstrapHypothesisResult:
    """
    Result of bootstrap hypothesis test.
    
    Attributes
    ----------
    test_name : str
        Name of the test
    statistic : float
        Observed test statistic
    p_value : float
        Bootstrap p-value
    bootstrap_distribution : ndarray
        Bootstrap distribution of test statistic under null
    n_bootstrap : int
        Number of bootstrap samples
    reject_null : bool
        Whether to reject null hypothesis
    alpha : float
        Significance level
    """
    test_name: str
    statistic: float
    p_value: float
    bootstrap_distribution: np.ndarray
    n_bootstrap: int
    reject_null: bool
    alpha: float
    
    def summary(self) -> str:
        """
        Generate summary of test results.
        
        Returns
        -------
        summary : str
            Formatted summary
        """
        lines = [
            "="*60,
            f"{self.test_name} (Bootstrap)",
            "="*60,
            f"Bootstrap samples: {self.n_bootstrap}",
            f"Significance level: {self.alpha}",
            "",
            f"Test statistic: {self.statistic:.6f}",
            f"Bootstrap p-value: {self.p_value:.6f}",
            "",
            f"Decision: {'REJECT' if self.reject_null else 'FAIL TO REJECT'} null hypothesis",
            "="*60
        ]
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return (
            f"BootstrapHypothesisResult(test='{self.test_name}', "
            f"statistic={self.statistic:.4f}, p={self.p_value:.4f})"
        )


def bootstrap_hypothesis_test(
    data: np.ndarray,
    test_statistic: Callable,
    null_generator: Callable,
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    alternative: str = 'two-sided',
    random_state: Optional[int] = None
) -> BootstrapHypothesisResult:
    """
    General bootstrap hypothesis test.
    
    Parameters
    ----------
    data : array-like
        Observed data
    test_statistic : callable
        Function that computes test statistic from data
    null_generator : callable
        Function that generates data under null hypothesis
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level
    alternative : str, default='two-sided'
        Alternative hypothesis: 'two-sided', 'less', or 'greater'
    random_state : int, optional
        Random seed
    
    Returns
    -------
    result : BootstrapHypothesisResult
        Test result with bootstrap p-value
    
    Examples
    --------
    >>> # Test if mean equals 0
    >>> data = np.random.normal(0.5, 1, 100)
    >>> test_stat = lambda x: np.mean(x)
    >>> null_gen = lambda: np.random.normal(0, 1, 100)
    >>> result = bootstrap_hypothesis_test(data, test_stat, null_gen)
    """
    np.random.seed(random_state)
    
    # Calculate observed test statistic
    observed_stat = test_statistic(data)
    
    # Generate bootstrap distribution under null
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        null_data = null_generator()
        bootstrap_stats[i] = test_statistic(null_data)
    
    # Calculate p-value based on alternative
    if alternative == 'two-sided':
        p_value = np.mean(np.abs(bootstrap_stats) >= np.abs(observed_stat))
    elif alternative == 'greater':
        p_value = np.mean(bootstrap_stats >= observed_stat)
    elif alternative == 'less':
        p_value = np.mean(bootstrap_stats <= observed_stat)
    else:
        raise ValueError(f"Unknown alternative: {alternative}")
    
    reject_null = p_value < alpha
    
    return BootstrapHypothesisResult(
        test_name="Bootstrap Hypothesis Test",
        statistic=observed_stat,
        p_value=p_value,
        bootstrap_distribution=bootstrap_stats,
        n_bootstrap=n_bootstrap,
        reject_null=reject_null,
        alpha=alpha
    )


def bootstrap_goodness_of_fit(
    data: np.ndarray,
    distribution: BaseDistribution,
    test_statistic: str = 'ks',
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    random_state: Optional[int] = None
) -> BootstrapHypothesisResult:
    """
    Bootstrap goodness-of-fit test.
    
    More accurate than asymptotic tests when parameters are estimated from data.
    
    Parameters
    ----------
    data : array-like
        Observed data
    distribution : BaseDistribution
        Fitted distribution to test
    test_statistic : str, default='ks'
        Test statistic to use:
        - 'ks': Kolmogorov-Smirnov statistic
        - 'ad': Anderson-Darling statistic
        - 'cvm': Cramer-von Mises statistic
    n_bootstrap : int, default=10000
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level
    random_state : int, optional
        Random seed
    
    Returns
    -------
    result : BootstrapHypothesisResult
        GOF test result with bootstrap p-value
    
    Examples
    --------
    >>> from distfit_pro import NormalDistribution
    >>> data = np.random.normal(10, 2, 100)
    >>> dist = NormalDistribution()
    >>> dist.fit(data)
    >>> result = bootstrap_goodness_of_fit(data, dist, test_statistic='ks')
    >>> print(result.summary())
    """
    if not distribution.fitted:
        raise ValueError("Distribution must be fitted first")
    
    np.random.seed(random_state)
    
    # Select test statistic function
    if test_statistic == 'ks':
        stat_func = _ks_statistic
    elif test_statistic == 'ad':
        stat_func = _ad_statistic
    elif test_statistic == 'cvm':
        stat_func = _cvm_statistic
    else:
        raise ValueError(f"Unknown test statistic: {test_statistic}")
    
    # Calculate observed statistic
    observed_stat = stat_func(data, distribution)
    
    # Bootstrap loop
    bootstrap_stats = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        # Generate sample from fitted distribution
        bootstrap_sample = distribution.rvs(size=len(data))
        
        # Fit same distribution type to bootstrap sample
        bootstrap_dist = distribution.__class__()
        try:
            bootstrap_dist.fit(bootstrap_sample, method='mle')
            bootstrap_stats[i] = stat_func(bootstrap_sample, bootstrap_dist)
        except:
            # If fitting fails, use a large value (conservative)
            bootstrap_stats[i] = np.inf
    
    # Calculate p-value
    p_value = np.mean(bootstrap_stats >= observed_stat)
    reject_null = p_value < alpha
    
    test_names = {
        'ks': 'Kolmogorov-Smirnov',
        'ad': 'Anderson-Darling',
        'cvm': 'Cramer-von Mises'
    }
    
    return BootstrapHypothesisResult(
        test_name=f"{test_names[test_statistic]} Goodness-of-Fit Test",
        statistic=observed_stat,
        p_value=p_value,
        bootstrap_distribution=bootstrap_stats,
        n_bootstrap=n_bootstrap,
        reject_null=reject_null,
        alpha=alpha
    )


# Helper functions for test statistics

def _ks_statistic(data: np.ndarray, distribution: BaseDistribution) -> float:
    """Calculate Kolmogorov-Smirnov statistic."""
    from scipy import stats
    statistic, _ = stats.kstest(data, distribution.cdf)
    return statistic


def _ad_statistic(data: np.ndarray, distribution: BaseDistribution) -> float:
    """Calculate Anderson-Darling statistic."""
    n = len(data)
    data_sorted = np.sort(data)
    
    # Calculate CDF values
    cdf_vals = distribution.cdf(data_sorted)
    cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
    
    # Calculate AD statistic
    i = np.arange(1, n + 1)
    ad_statistic = -n - np.sum(
        (2*i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
    ) / n
    
    return ad_statistic


def _cvm_statistic(data: np.ndarray, distribution: BaseDistribution) -> float:
    """Calculate Cramer-von Mises statistic."""
    n = len(data)
    data_sorted = np.sort(data)
    
    # Calculate CDF values
    cdf_vals = distribution.cdf(data_sorted)
    
    # Calculate CvM statistic
    i = np.arange(1, n + 1)
    cvm_statistic = (1 / (12 * n)) + np.sum((cdf_vals - (2*i - 1) / (2*n))**2)
    
    return cvm_statistic
