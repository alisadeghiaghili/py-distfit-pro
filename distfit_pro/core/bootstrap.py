"""
Bootstrap Confidence Intervals
===============================

Provides parametric and non-parametric bootstrap methods for estimating
confidence intervals of distribution parameters.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Callable
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class BootstrapResult:
    """
    Bootstrap confidence interval result
    
    Attributes:
    -----------
    parameter : str
        Parameter name
    estimate : float
        Point estimate
    ci_lower : float
        Lower bound of CI
    ci_upper : float
        Upper bound of CI
    confidence_level : float
        Confidence level (e.g., 0.95)
    method : str
        Bootstrap method used
    n_bootstrap : int
        Number of bootstrap samples
    """
    parameter: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    n_bootstrap: int
    
    def __repr__(self) -> str:
        return f"""
Bootstrap CI for {self.parameter}
{'=' * 50}
Point Estimate: {self.estimate:.6f}
{int(self.confidence_level*100)}% CI: [{self.ci_lower:.6f}, {self.ci_upper:.6f}]
Method: {self.method}
Bootstrap Samples: {self.n_bootstrap}
"""


class Bootstrap:
    """
    Bootstrap methods for confidence interval estimation
    
    Example:
    --------
    >>> from distfit_pro import get_distribution
    >>> import numpy as np
    >>> 
    >>> data = np.random.normal(5, 2, 1000)
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> 
    >>> # Parametric bootstrap
    >>> ci_params = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    >>> for param, result in ci_params.items():
    ...     print(result)
    >>> 
    >>> # Non-parametric bootstrap
    >>> ci_nonparam = Bootstrap.nonparametric(data, dist, n_bootstrap=1000)
    """
    
    @staticmethod
    def parametric(data: np.ndarray,
                   distribution,
                   n_bootstrap: int = 1000,
                   confidence_level: float = 0.95,
                   n_jobs: int = -1,
                   random_state: Optional[int] = None) -> Dict[str, BootstrapResult]:
        """
        Parametric Bootstrap
        
        Generates bootstrap samples from the fitted distribution,
        refits parameters, and estimates confidence intervals.
        
        Parameters:
        -----------
        data : array-like
            Original observed data
        distribution : BaseDistribution
            Fitted distribution object
        n_bootstrap : int
            Number of bootstrap samples (default: 1000)
        confidence_level : float
            Confidence level (default: 0.95)
        n_jobs : int
            Number of parallel jobs (default: -1 = all cores)
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        results : dict
            Dictionary of BootstrapResult for each parameter
        """
        if not distribution.fitted:
            raise ValueError("Distribution must be fitted first")
        
        n = len(data)
        alpha = 1 - confidence_level
        
        def _bootstrap_iteration(seed):
            """Single bootstrap iteration"""
            # Generate bootstrap sample from fitted distribution
            boot_sample = distribution.rvs(size=n, random_state=seed)
            
            # Create new distribution instance and fit
            from .distributions import get_distribution
            boot_dist = get_distribution(distribution.info.name)
            try:
                boot_dist.fit(boot_sample, method='mle')
                return boot_dist.params
            except:
                return None
        
        # Generate seeds for reproducibility
        if random_state is not None:
            np.random.seed(random_state)
        seeds = np.random.randint(0, 2**31 - 1, size=n_bootstrap)
        
        # Parallel bootstrap
        boot_params_list = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_iteration)(seed) 
            for seed in tqdm(seeds, desc="Parametric Bootstrap")
        )
        
        # Filter out failed fits
        boot_params_list = [p for p in boot_params_list if p is not None]
        
        if len(boot_params_list) < n_bootstrap * 0.9:
            print(f"Warning: {n_bootstrap - len(boot_params_list)} bootstrap samples failed")
        
        # Convert to dict of arrays
        boot_params = {}
        for param_name in distribution.params.keys():
            boot_params[param_name] = np.array([p[param_name] for p in boot_params_list])
        
        # Calculate confidence intervals
        results = {}
        for param_name, param_value in distribution.params.items():
            boot_values = boot_params[param_name]
            ci_lower = np.percentile(boot_values, alpha/2 * 100)
            ci_upper = np.percentile(boot_values, (1 - alpha/2) * 100)
            
            results[param_name] = BootstrapResult(
                parameter=param_name,
                estimate=param_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=confidence_level,
                method="Parametric Bootstrap",
                n_bootstrap=len(boot_params_list)
            )
        
        return results
    
    @staticmethod
    def nonparametric(data: np.ndarray,
                     distribution,
                     n_bootstrap: int = 1000,
                     confidence_level: float = 0.95,
                     n_jobs: int = -1,
                     random_state: Optional[int] = None) -> Dict[str, BootstrapResult]:
        """
        Non-parametric Bootstrap
        
        Resamples from the original data with replacement,
        refits parameters, and estimates confidence intervals.
        
        Parameters:
        -----------
        data : array-like
            Original observed data
        distribution : BaseDistribution
            Fitted distribution object
        n_bootstrap : int
            Number of bootstrap samples (default: 1000)
        confidence_level : float
            Confidence level (default: 0.95)
        n_jobs : int
            Number of parallel jobs (default: -1 = all cores)
        random_state : int, optional
            Random seed for reproducibility
            
        Returns:
        --------
        results : dict
            Dictionary of BootstrapResult for each parameter
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        alpha = 1 - confidence_level
        
        def _bootstrap_iteration(seed):
            """Single bootstrap iteration"""
            np.random.seed(seed)
            # Resample with replacement
            boot_sample = np.random.choice(data, size=n, replace=True)
            
            # Create new distribution instance and fit
            from .distributions import get_distribution
            boot_dist = get_distribution(distribution.info.name)
            try:
                boot_dist.fit(boot_sample, method='mle')
                return boot_dist.params
            except:
                return None
        
        # Generate seeds
        if random_state is not None:
            np.random.seed(random_state)
        seeds = np.random.randint(0, 2**31 - 1, size=n_bootstrap)
        
        # Parallel bootstrap
        boot_params_list = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_iteration)(seed) 
            for seed in tqdm(seeds, desc="Non-parametric Bootstrap")
        )
        
        # Filter out failed fits
        boot_params_list = [p for p in boot_params_list if p is not None]
        
        if len(boot_params_list) < n_bootstrap * 0.9:
            print(f"Warning: {n_bootstrap - len(boot_params_list)} bootstrap samples failed")
        
        # Convert to dict of arrays
        boot_params = {}
        for param_name in distribution.params.keys():
            boot_params[param_name] = np.array([p[param_name] for p in boot_params_list])
        
        # Calculate confidence intervals
        results = {}
        for param_name, param_value in distribution.params.items():
            boot_values = boot_params[param_name]
            ci_lower = np.percentile(boot_values, alpha/2 * 100)
            ci_upper = np.percentile(boot_values, (1 - alpha/2) * 100)
            
            results[param_name] = BootstrapResult(
                parameter=param_name,
                estimate=param_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=confidence_level,
                method="Non-parametric Bootstrap",
                n_bootstrap=len(boot_params_list)
            )
        
        return results
    
    @staticmethod
    def percentile_ci(bootstrap_samples: np.ndarray,
                     confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate percentile-based confidence interval
        
        Parameters:
        -----------
        bootstrap_samples : array-like
            Bootstrap parameter estimates
        confidence_level : float
            Confidence level (default: 0.95)
            
        Returns:
        --------
        ci_lower, ci_upper : tuple
            Confidence interval bounds
        """
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_samples, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_samples, (1 - alpha/2) * 100)
        return ci_lower, ci_upper
    
    @staticmethod
    def bca_ci(bootstrap_samples: np.ndarray,
              original_estimate: float,
              data: np.ndarray,
              estimator_func: Callable,
              confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Bias-Corrected and Accelerated (BCa) Bootstrap CI
        
        More accurate than percentile method, especially for skewed distributions.
        
        Parameters:
        -----------
        bootstrap_samples : array-like
            Bootstrap parameter estimates
        original_estimate : float
            Original parameter estimate
        data : array-like
            Original data
        estimator_func : callable
            Function that estimates parameter from data
        confidence_level : float
            Confidence level (default: 0.95)
            
        Returns:
        --------
        ci_lower, ci_upper : tuple
            BCa confidence interval bounds
        """
        from scipy import stats
        
        n = len(data)
        alpha = 1 - confidence_level
        
        # Bias correction
        z0 = stats.norm.ppf(np.mean(bootstrap_samples < original_estimate))
        
        # Acceleration (jackknife)
        jackknife_estimates = []
        for i in range(n):
            jack_sample = np.delete(data, i)
            jack_estimate = estimator_func(jack_sample)
            jackknife_estimates.append(jack_estimate)
        
        jackknife_estimates = np.array(jackknife_estimates)
        jack_mean = np.mean(jackknife_estimates)
        
        numerator = np.sum((jack_mean - jackknife_estimates)**3)
        denominator = 6 * (np.sum((jack_mean - jackknife_estimates)**2)**(3/2))
        
        if denominator == 0:
            a = 0
        else:
            a = numerator / denominator
        
        # Adjusted percentiles
        z_alpha_lower = stats.norm.ppf(alpha/2)
        z_alpha_upper = stats.norm.ppf(1 - alpha/2)
        
        p_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower)/(1 - a*(z0 + z_alpha_lower)))
        p_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper)/(1 - a*(z0 + z_alpha_upper)))
        
        ci_lower = np.percentile(bootstrap_samples, p_lower * 100)
        ci_upper = np.percentile(bootstrap_samples, p_upper * 100)
        
        return ci_lower, ci_upper
