"""
Weighted Distribution Fitting
=============================

Support for fitting distributions to weighted data.
"""

from typing import Dict, Optional
import numpy as np
from scipy.optimize import minimize


class WeightedDistributionFitter:
    """
    Fit distributions to weighted data
    
    Useful when:
    - Different observations have different importance
    - Data represents aggregated counts
    - Stratified sampling with known weights
    
    Example:
    --------
    >>> from distfit_pro.core.distributions import get_distribution
    >>> from distfit_pro.core.weighted import WeightedDistributionFitter
    >>> 
    >>> data = np.random.normal(10, 2, 100)
    >>> weights = np.random.uniform(0.5, 1.5, 100)  # Vary importance
    >>> 
    >>> dist = get_distribution('normal')
    >>> fitter = WeightedDistributionFitter()
    >>> params = fitter.fit(data, weights, dist, method='mle')
    >>> print(params)
    """
    
    def fit(self,
            data: np.ndarray,
            weights: np.ndarray,
            distribution,
            method: str = 'mle') -> Dict[str, float]:
        """
        Fit distribution to weighted data
        
        Parameters:
        -----------
        data : array-like
            Observed data
        weights : array-like
            Observation weights (must be positive)
        distribution : BaseDistribution
            Distribution to fit
        method : str
            'mle': weighted maximum likelihood
            'moments': weighted method of moments
            
        Returns:
        --------
        params : dict
            Fitted parameters
        """
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        if len(data) != len(weights):
            raise ValueError("data and weights must have same length")
        
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")
        
        # Remove zero-weight observations
        mask = weights > 0
        data = data[mask]
        weights = weights[mask]
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        if method == 'mle':
            return self._fit_weighted_mle(data, weights, distribution)
        elif method == 'moments':
            return self._fit_weighted_moments(data, weights, distribution)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _fit_weighted_mle(self,
                         data: np.ndarray,
                         weights: np.ndarray,
                         distribution) -> Dict[str, float]:
        """
        Weighted maximum likelihood estimation
        """
        # Negative weighted log-likelihood
        def neg_weighted_ll(params_array):
            try:
                dist_copy = distribution.__class__()
                dist_copy.params = dist_copy._array_to_params(params_array)
                log_probs = dist_copy.logpdf(data)
                weighted_ll = np.sum(weights * log_probs)
                return -weighted_ll
            except:
                return np.inf
        
        # Initial guess from weighted moments
        initial_params = self._fit_weighted_moments(data, weights, distribution)
        x0 = distribution._params_to_array(initial_params)
        
        # Optimize
        result = minimize(neg_weighted_ll, x0, method='Nelder-Mead')
        
        return distribution._array_to_params(result.x)
    
    def _fit_weighted_moments(self,
                             data: np.ndarray,
                             weights: np.ndarray,
                             distribution) -> Dict[str, float]:
        """
        Weighted method of moments
        """
        # Weighted mean and variance
        weighted_mean = np.sum(weights * data)
        weighted_var = np.sum(weights * (data - weighted_mean)**2)
        
        # Map to distribution parameters
        dist_name = distribution.info.name
        
        if dist_name == 'normal':
            return {'loc': weighted_mean, 'scale': np.sqrt(weighted_var)}
        
        elif dist_name == 'lognormal':
            log_data = np.log(data[data > 0])
            log_weights = weights[data > 0]
            log_weights = log_weights / np.sum(log_weights)
            weighted_log_mean = np.sum(log_weights * log_data)
            weighted_log_var = np.sum(log_weights * (log_data - weighted_log_mean)**2)
            return {'s': np.sqrt(weighted_log_var), 'scale': np.exp(weighted_log_mean)}
        
        elif dist_name == 'gamma':
            return {'a': weighted_mean**2 / weighted_var, 
                   'scale': weighted_var / weighted_mean}
        
        elif dist_name == 'exponential':
            return {'scale': weighted_mean}
        
        elif dist_name == 'beta':
            common = weighted_mean * (1 - weighted_mean) / weighted_var - 1
            return {'a': weighted_mean * common, 'b': (1 - weighted_mean) * common}
        
        else:
            # Fallback: use unweighted method
            return distribution.fit_moments(data)
