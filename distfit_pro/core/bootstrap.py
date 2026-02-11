"""
Bootstrap Confidence Intervals
==============================

Implements parametric and non-parametric bootstrap methods
for estimating confidence intervals of distribution parameters.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from joblib import Parallel, delayed
from dataclasses import dataclass


@dataclass
class BootstrapResult:
    """
    Result from bootstrap confidence interval estimation
    
    Attributes:
    -----------
    parameter : str
        Parameter name
    estimate : float
        Point estimate
    ci_lower : float
        Lower confidence bound
    ci_upper : float
        Upper confidence bound
    confidence_level : float
        Confidence level (e.g., 0.95)
    method : str
        Bootstrap method used
    """
    parameter: str
    estimate: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    method: str
    bootstrap_samples: Optional[np.ndarray] = None


class Bootstrap:
    """
    Bootstrap methods for confidence interval estimation
    
    Example:
    --------
    >>> from distfit_pro.core.distributions import get_distribution
    >>> from distfit_pro.core.bootstrap import Bootstrap
    >>> 
    >>> data = np.random.normal(10, 2, 1000)
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> 
    >>> bs = Bootstrap(n_bootstrap=1000, n_jobs=-1)
    >>> ci = bs.parametric(data, dist, alpha=0.05)
    >>> 
    >>> print(f"Mean: {ci['loc'].estimate:.2f}")
    >>> print(f"95% CI: [{ci['loc'].ci_lower:.2f}, {ci['loc'].ci_upper:.2f}]")
    """
    
    def __init__(self, 
                 n_bootstrap: int = 1000,
                 confidence_level: float = 0.95,
                 n_jobs: int = 1,
                 random_state: Optional[int] = None):
        """
        Initialize Bootstrap
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples
        confidence_level : float
            Confidence level (0-1), default 0.95 for 95% CI
        n_jobs : int
            Number of parallel jobs (-1 for all cores)
        random_state : int, optional
            Random seed for reproducibility
        """
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def parametric(self, 
                   data: np.ndarray,
                   distribution,
                   method: str = 'percentile') -> Dict[str, BootstrapResult]:
        """
        Parametric bootstrap
        
        Assumes the fitted distribution is correct and generates
        synthetic data from it.
        
        Parameters:
        -----------
        data : array-like
            Original data
        distribution : BaseDistribution
            Fitted distribution
        method : str
            CI method: 'percentile', 'basic', 'bca' (bias-corrected)
            
        Returns:
        --------
        ci : dict
            Dictionary mapping parameter names to BootstrapResult objects
        """
        if not distribution.fitted:
            raise ValueError("Distribution must be fitted before bootstrap")
        
        n = len(data)
        original_params = distribution.params.copy()
        
        # Generate bootstrap samples
        def _bootstrap_sample(seed):
            if seed is not None:
                np.random.seed(seed)
            synthetic_data = distribution.rvs(size=n)
            dist_copy = distribution.__class__()
            try:
                dist_copy.fit(synthetic_data, method='mle')
                return dist_copy.params
            except:
                return None
        
        # Parallel bootstrap
        seeds = [self.random_state + i if self.random_state is not None else None 
                for i in range(self.n_bootstrap)]
        
        bootstrap_params = Parallel(n_jobs=self.n_jobs)(
            delayed(_bootstrap_sample)(seed) for seed in seeds
        )
        
        # Filter out failed fits
        bootstrap_params = [p for p in bootstrap_params if p is not None]
        
        if len(bootstrap_params) < self.n_bootstrap * 0.9:
            print(f"Warning: Only {len(bootstrap_params)}/{self.n_bootstrap} bootstrap samples succeeded")
        
        # Convert to dict of arrays
        bootstrap_dict = {}
        for param_name in original_params.keys():
            bootstrap_dict[param_name] = np.array(
                [p[param_name] for p in bootstrap_params]
            )
        
        # Calculate confidence intervals
        results = {}
        for param_name, original_value in original_params.items():
            boot_samples = bootstrap_dict[param_name]
            
            if method == 'percentile':
                ci_lower = np.percentile(boot_samples, 100 * self.alpha / 2)
                ci_upper = np.percentile(boot_samples, 100 * (1 - self.alpha / 2))
            
            elif method == 'basic':
                # Basic bootstrap (pivot method)
                delta_lower = np.percentile(boot_samples - original_value, 100 * self.alpha / 2)
                delta_upper = np.percentile(boot_samples - original_value, 100 * (1 - self.alpha / 2))
                ci_lower = original_value - delta_upper
                ci_upper = original_value - delta_lower
            
            elif method == 'bca':
                # Bias-corrected and accelerated (BCa)
                ci_lower, ci_upper = self._bca_ci(boot_samples, original_value)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[param_name] = BootstrapResult(
                parameter=param_name,
                estimate=original_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=self.confidence_level,
                method=f'parametric-{method}',
                bootstrap_samples=boot_samples
            )
        
        return results
    
    def nonparametric(self,
                     data: np.ndarray,
                     distribution,
                     method: str = 'percentile') -> Dict[str, BootstrapResult]:
        """
        Non-parametric bootstrap
        
        Resamples from original data (with replacement).
        Makes no distributional assumptions.
        
        Parameters:
        -----------
        data : array-like
            Original data
        distribution : BaseDistribution
            Distribution class (not necessarily fitted)
        method : str
            CI method: 'percentile', 'basic', 'bca'
            
        Returns:
        --------
        ci : dict
            Dictionary mapping parameter names to BootstrapResult objects
        """
        data = np.asarray(data).flatten()
        n = len(data)
        
        # Fit to original data
        dist_orig = distribution.__class__()
        dist_orig.fit(data, method='mle')
        original_params = dist_orig.params.copy()
        
        # Generate bootstrap samples
        def _bootstrap_sample(seed):
            if seed is not None:
                np.random.seed(seed)
            resample_idx = np.random.choice(n, size=n, replace=True)
            resample_data = data[resample_idx]
            dist_copy = distribution.__class__()
            try:
                dist_copy.fit(resample_data, method='mle')
                return dist_copy.params
            except:
                return None
        
        # Parallel bootstrap
        seeds = [self.random_state + i if self.random_state is not None else None 
                for i in range(self.n_bootstrap)]
        
        bootstrap_params = Parallel(n_jobs=self.n_jobs)(
            delayed(_bootstrap_sample)(seed) for seed in seeds
        )
        
        # Filter out failed fits
        bootstrap_params = [p for p in bootstrap_params if p is not None]
        
        if len(bootstrap_params) < self.n_bootstrap * 0.9:
            print(f"Warning: Only {len(bootstrap_params)}/{self.n_bootstrap} bootstrap samples succeeded")
        
        # Convert to dict of arrays
        bootstrap_dict = {}
        for param_name in original_params.keys():
            bootstrap_dict[param_name] = np.array(
                [p[param_name] for p in bootstrap_params]
            )
        
        # Calculate confidence intervals
        results = {}
        for param_name, original_value in original_params.items():
            boot_samples = bootstrap_dict[param_name]
            
            if method == 'percentile':
                ci_lower = np.percentile(boot_samples, 100 * self.alpha / 2)
                ci_upper = np.percentile(boot_samples, 100 * (1 - self.alpha / 2))
            
            elif method == 'basic':
                delta_lower = np.percentile(boot_samples - original_value, 100 * self.alpha / 2)
                delta_upper = np.percentile(boot_samples - original_value, 100 * (1 - self.alpha / 2))
                ci_lower = original_value - delta_upper
                ci_upper = original_value - delta_lower
            
            elif method == 'bca':
                ci_lower, ci_upper = self._bca_ci(boot_samples, original_value)
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            results[param_name] = BootstrapResult(
                parameter=param_name,
                estimate=original_value,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                confidence_level=self.confidence_level,
                method=f'nonparametric-{method}',
                bootstrap_samples=boot_samples
            )
        
        return results
    
    def _bca_ci(self, boot_samples: np.ndarray, original_value: float) -> Tuple[float, float]:
        """
        Bias-Corrected and Accelerated (BCa) confidence interval
        
        More accurate but computationally intensive.
        """
        from scipy.stats import norm
        
        # Bias correction
        z0 = norm.ppf(np.mean(boot_samples < original_value))
        
        # Acceleration (jackknife)
        n = len(boot_samples)
        jack_samples = np.array([
            np.mean(np.delete(boot_samples, i)) for i in range(min(n, 100))
        ])
        jack_mean = np.mean(jack_samples)
        a = np.sum((jack_mean - jack_samples)**3) / (6 * np.sum((jack_mean - jack_samples)**2)**1.5)
        
        # Adjusted percentiles
        z_alpha_lower = norm.ppf(self.alpha / 2)
        z_alpha_upper = norm.ppf(1 - self.alpha / 2)
        
        p_lower = norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        p_upper = norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        ci_lower = np.percentile(boot_samples, 100 * p_lower)
        ci_upper = np.percentile(boot_samples, 100 * p_upper)
        
        return ci_lower, ci_upper


def format_bootstrap_results(results: Dict[str, BootstrapResult]) -> str:
    """
    Format bootstrap results for display
    
    Parameters:
    -----------
    results : dict
        Results from Bootstrap.parametric() or Bootstrap.nonparametric()
        
    Returns:
    --------
    formatted : str
        Formatted string
    """
    output = []
    output.append("\n" + "="*70)
    output.append("BOOTSTRAP CONFIDENCE INTERVALS")
    output.append("="*70)
    
    for param_name, result in results.items():
        output.append(f"\n{param_name}:")
        output.append(f"  Estimate:  {result.estimate:.6f}")
        output.append(f"  {int(result.confidence_level*100)}% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        output.append(f"  Width:     {result.ci_upper - result.ci_lower:.6f}")
        output.append(f"  Method:    {result.method}")
    
    output.append("\n" + "="*70)
    return "\n".join(output)
