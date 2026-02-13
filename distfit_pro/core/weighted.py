"""
Weighted Distribution Fitting
==============================

Support for fitting distributions to weighted data.
"""

from typing import Dict, Optional
import numpy as np
from scipy.optimize import minimize
from scipy import stats


class WeightedFitting:
    """
    Weighted fitting methods for distributions
    
    Useful when:
    - Different observations have different reliabilities
    - Data comes from stratified sampling
    - Survey data with sampling weights
    - Aggregated data with frequency counts
    
    Example:
    --------
    >>> from distfit_pro import get_distribution
    >>> from distfit_pro.core.weighted import WeightedFitting
    >>> import numpy as np
    >>> 
    >>> data = np.random.normal(5, 2, 1000)
    >>> weights = np.random.uniform(0.5, 1.5, 1000)  # Different reliabilities
    >>> 
    >>> dist = get_distribution('normal')
    >>> params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    >>> dist.params = params
    >>> dist.fitted = True
    """
    
    @staticmethod
    def fit_weighted_mle(data: np.ndarray,
                        weights: np.ndarray,
                        distribution,
                        **kwargs) -> Dict[str, float]:
        """
        Weighted Maximum Likelihood Estimation
        
        Parameters:
        -----------
        data : array-like
            Observed data
        weights : array-like
            Weights for each observation (non-negative)
        distribution : BaseDistribution
            Distribution to fit
            
        Returns:
        --------
        params : dict
            Fitted parameters
        """
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        if len(data) != len(weights):
            raise ValueError("Data and weights must have same length")
        
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        
        # Remove NaN and zero-weight observations
        mask = ~(np.isnan(data) | np.isnan(weights) | (weights == 0))
        data = data[mask]
        weights = weights[mask]
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Use weighted moments as initial guess
        initial_params = WeightedFitting.fit_weighted_moments(data, weights, distribution)
        
        def weighted_neg_log_likelihood(params_array):
            """Negative weighted log-likelihood"""
            try:
                # Convert array to parameter dict
                param_dict = distribution.array_to_params(params_array)
                distribution.params = param_dict
                
                # Compute log-likelihood for each observation
                log_lik = distribution.logpdf(data)
                
                # Weighted sum
                weighted_log_lik = np.sum(weights * log_lik)
                
                return -weighted_log_lik
            except:
                return np.inf
        
        # Optimize
        x0 = distribution.params_to_array(initial_params)
        result = minimize(
            weighted_neg_log_likelihood,
            x0,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        
        return distribution.array_to_params(result.x)
    
    @staticmethod
    def fit_weighted_moments(data: np.ndarray,
                            weights: np.ndarray,
                            distribution) -> Dict[str, float]:
        """
        Weighted Method of Moments
        
        Parameters:
        -----------
        data : array-like
            Observed data
        weights : array-like
            Weights for each observation
        distribution : BaseDistribution
            Distribution to fit
            
        Returns:
        --------
        params : dict
            Fitted parameters
        """
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        # Remove NaN and zero-weight
        mask = ~(np.isnan(data) | np.isnan(weights) | (weights == 0))
        data = data[mask]
        weights = weights[mask]
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Weighted moments
        wmean = np.sum(weights * data)
        wvar = np.sum(weights * (data - wmean)**2)
        wstd = np.sqrt(wvar)
        
        # Distribution-specific fitting
        dist_name = distribution.info.name
        
        if dist_name == 'normal':
            return {'loc': wmean, 'scale': wstd}
        
        elif dist_name == 'lognormal':
            positive_data = data[data > 0]
            positive_weights = weights[data > 0]
            positive_weights = positive_weights / np.sum(positive_weights)
            
            log_data = np.log(positive_data)
            log_mean = np.sum(positive_weights * log_data)
            log_var = np.sum(positive_weights * (log_data - log_mean)**2)
            
            return {'s': np.sqrt(log_var), 'scale': np.exp(log_mean)}
        
        elif dist_name == 'exponential':
            return {'scale': wmean}
        
        elif dist_name == 'gamma':
            return {'a': wmean**2 / wvar, 'scale': wvar / wmean}
        
        elif dist_name == 'weibull':
            # Approximate using moments
            cv = wstd / wmean  # Coefficient of variation
            # Approximate shape parameter
            k = 1.0 / cv if cv > 0 else 1.0
            return {'c': k, 'scale': wmean}
        
        elif dist_name == 'beta':
            # Clip to (0, 1)
            data_clipped = np.clip(data, 1e-6, 1 - 1e-6)
            wmean_clipped = np.sum(weights * data_clipped)
            wvar_clipped = np.sum(weights * (data_clipped - wmean_clipped)**2)
            
            if wvar_clipped > 0:
                common = wmean_clipped * (1 - wmean_clipped) / wvar_clipped - 1
                return {'a': wmean_clipped * common, 'b': (1 - wmean_clipped) * common}
            else:
                return {'a': 1.0, 'b': 1.0}
        
        elif dist_name == 'uniform':
            return {'loc': np.min(data), 'scale': np.max(data) - np.min(data)}
        
        elif dist_name == 'logistic':
            return {'loc': wmean, 'scale': wstd * np.sqrt(3) / np.pi}
        
        elif dist_name == 'laplace':
            wmedian = WeightedFitting.weighted_quantile(data, weights, 0.5)
            wmad = np.sum(weights * np.abs(data - wmedian))
            return {'loc': wmedian, 'scale': wmad}
        
        elif dist_name == 'poisson':
            return {'mu': wmean}
        
        elif dist_name == 'binomial':
            n = int(np.max(data))
            p = wmean / n if n > 0 else 0.5
            return {'n': n, 'p': np.clip(p, 0.01, 0.99)}
        
        elif dist_name == 'nbinom':
            if wvar > wmean:
                p = wmean / wvar
                n = wmean * p / (1 - p) if p < 1 else 1
                return {'n': max(n, 1), 'p': np.clip(p, 0.01, 0.99)}
            else:
                return {'n': 1, 'p': 0.5}
        
        elif dist_name == 'geometric':
            p = 1.0 / wmean if wmean > 0 else 0.5
            return {'p': np.clip(p, 0.01, 0.99)}
        
        else:
            # Fallback: try unweighted moments
            return distribution.fit_moments(data)
    
    @staticmethod
    def weighted_quantile(data: np.ndarray,
                         weights: np.ndarray,
                         quantile: float) -> float:
        """
        Compute weighted quantile
        
        Parameters:
        -----------
        data : array-like
            Data values
        weights : array-like
            Weights
        quantile : float
            Quantile to compute (0 to 1)
            
        Returns:
        --------
        q : float
            Weighted quantile value
        """
        # Sort data and weights
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Cumulative weights
        cumsum = np.cumsum(sorted_weights)
        cumsum = cumsum / cumsum[-1]  # Normalize
        
        # Find quantile
        idx = np.searchsorted(cumsum, quantile)
        if idx >= len(sorted_data):
            return sorted_data[-1]
        return sorted_data[idx]
    
    @staticmethod
    def weighted_stats(data: np.ndarray,
                      weights: np.ndarray) -> Dict[str, float]:
        """
        Compute weighted statistics
        
        Parameters:
        -----------
        data : array-like
            Data values
        weights : array-like
            Weights
            
        Returns:
        --------
        stats : dict
            Dictionary with mean, var, std, median, quantiles
        """
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        # Basic stats
        wmean = np.sum(weights * data)
        wvar = np.sum(weights * (data - wmean)**2)
        wstd = np.sqrt(wvar)
        
        # Quantiles
        wmedian = WeightedFitting.weighted_quantile(data, weights, 0.5)
        wq25 = WeightedFitting.weighted_quantile(data, weights, 0.25)
        wq75 = WeightedFitting.weighted_quantile(data, weights, 0.75)
        
        return {
            'mean': wmean,
            'var': wvar,
            'std': wstd,
            'median': wmedian,
            'q25': wq25,
            'q75': wq75
        }
    
    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        """
        Calculate effective sample size for weighted data
        
        ESS = (sum(w))^2 / sum(w^2)
        
        Parameters:
        -----------
        weights : array-like
            Weights
            
        Returns:
        --------
        ess : float
            Effective sample size
        """
        weights = np.asarray(weights).flatten()
        weights = weights[weights > 0]
        
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights**2)
        
        return sum_w**2 / sum_w2 if sum_w2 > 0 else 0
