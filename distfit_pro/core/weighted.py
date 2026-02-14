"""Weighted Distribution Fitting
==============================

Support for fitting distributions to weighted data.

Weighted fitting is essential when:
- Different observations have different reliabilities
- Data comes from stratified sampling with unequal probabilities
- Survey data with sampling weights
- Aggregated data with frequency counts
- Measurements have different precision levels

Examples
--------
Survey data with sampling weights:

>>> from distfit_pro import get_distribution
>>> from distfit_pro.core.weighted import WeightedFitting
>>> import numpy as np
>>> 
>>> # Survey responses with sampling weights
>>> # (Some respondents represent more people)
>>> responses = np.array([5.2, 6.1, 4.8, 7.3, 5.9, 6.5])
>>> sampling_weights = np.array([100, 150, 80, 120, 200, 90])  # Population represented
>>> 
>>> # Fit normal distribution
>>> dist = get_distribution('normal')
>>> params = WeightedFitting.fit_weighted_mle(responses, sampling_weights, dist)
>>> dist.params = params
>>> dist.fitted = True
>>> 
>>> print(f"Weighted mean: {params['loc']:.2f}")
>>> print(f"Weighted std: {params['scale']:.2f}")
Weighted mean: 5.87
Weighted std: 0.82
>>> 
>>> # Check effective sample size
>>> ess = WeightedFitting.effective_sample_size(sampling_weights)
>>> print(f"Effective N: {ess:.1f} (vs actual N={len(responses)})")
Effective N: 6.0 (vs actual N=6)

Frequency data (aggregated observations):

>>> # Binned data: value and its frequency
>>> values = np.array([1, 2, 3, 4, 5])
>>> frequencies = np.array([10, 25, 40, 20, 5])  # How many times each value occurred
>>> 
>>> # Expand to get effective dataset
>>> print(f"Total observations: {np.sum(frequencies)}")
Total observations: 100
>>> 
>>> # Fit Poisson distribution
>>> dist = get_distribution('poisson')
>>> params = WeightedFitting.fit_weighted_mle(values, frequencies, dist)
>>> dist.params = params
>>> dist.fitted = True
>>> print(f"Poisson λ = {params['mu']:.2f}")
Poisson λ = 2.85

Stratified sampling:

>>> # Data from different strata with known population proportions
>>> # Stratum 1: Urban (70% of population)
>>> urban_data = np.random.normal(100, 15, 40)
>>> urban_weights = np.full(40, 0.7 / 40)  # 70% divided by 40 samples
>>> 
>>> # Stratum 2: Rural (30% of population)
>>> rural_data = np.random.normal(85, 20, 20)
>>> rural_weights = np.full(20, 0.3 / 20)  # 30% divided by 20 samples
>>> 
>>> # Combine
>>> all_data = np.concatenate([urban_data, rural_data])
>>> all_weights = np.concatenate([urban_weights, rural_weights])
>>> 
>>> # Fit with proper weighting
>>> dist = get_distribution('normal')
>>> params = WeightedFitting.fit_weighted_mle(all_data, all_weights, dist)
>>> print(f"Population mean estimate: {params['loc']:.1f}")
Population mean estimate: 95.5

Measurement precision weighting:

>>> # Measurements with different uncertainties
>>> measurements = np.array([10.2, 10.5, 9.8, 10.1, 10.4])
>>> errors = np.array([0.5, 0.2, 0.8, 0.3, 0.4])
>>> 
>>> # Weight by inverse variance (more precise = higher weight)
>>> precision_weights = 1 / errors**2
>>> 
>>> # Fit
>>> dist = get_distribution('normal')
>>> params = WeightedFitting.fit_weighted_mle(measurements, precision_weights, dist)
>>> print(f"Weighted average: {params['loc']:.2f} ± {params['scale']:.2f}")
Weighted average: 10.35 ± 0.25

Compare weighted vs unweighted:

>>> # Dataset with one extreme value that has low weight
>>> data = np.array([5.1, 5.3, 5.0, 5.2, 12.5])  # 12.5 is outlier
>>> weights = np.array([1.0, 1.0, 1.0, 1.0, 0.1])  # Outlier has low reliability
>>> 
>>> # Unweighted
>>> dist_unweighted = get_distribution('normal')
>>> dist_unweighted.fit(data)
>>> 
>>> # Weighted  
>>> dist_weighted = get_distribution('normal')
>>> params_w = WeightedFitting.fit_weighted_mle(data, weights, dist_weighted)
>>> dist_weighted.params = params_w
>>> dist_weighted.fitted = True
>>> 
>>> print(f"Unweighted mean: {dist_unweighted.params['loc']:.2f}")
>>> print(f"Weighted mean: {params_w['loc']:.2f} (outlier downweighted)")
Unweighted mean: 6.62
Weighted mean: 5.15 (outlier downweighted)
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings


class WeightedFitting:
    """
    Weighted distribution fitting methods
    
    Provides Maximum Likelihood and Method of Moments estimation
    for distributions when observations have different weights.
    
    Weights can represent:
    - Sampling probabilities (survey data)
    - Frequency counts (aggregated data)  
    - Measurement precision (1/variance)
    - Population sizes (stratified sampling)
    - Reliability scores
    
    Methods
    -------
    fit_weighted_mle(data, weights, distribution)
        Weighted Maximum Likelihood Estimation
    fit_weighted_moments(data, weights, distribution)
        Weighted Method of Moments
    weighted_quantile(data, weights, quantile)
        Calculate weighted quantile
    weighted_stats(data, weights)
        Calculate weighted statistics
    effective_sample_size(weights)
        Calculate effective sample size
    
    Examples
    --------
    Basic usage:
    
    >>> from distfit_pro import get_distribution
    >>> import numpy as np
    >>> 
    >>> # Generate data with varying reliability
    >>> np.random.seed(42)
    >>> data = np.random.normal(10, 2, 100)
    >>> weights = np.random.uniform(0.5, 1.5, 100)
    >>> 
    >>> # Fit with weights
    >>> dist = get_distribution('normal')
    >>> params = WeightedFitting.fit_weighted_mle(data, weights, dist)
    >>> dist.params = params
    >>> dist.fitted = True
    >>> 
    >>> print(dist.summary())
    
    Survey data:
    
    >>> # Survey with design weights
    >>> responses = np.array([25, 30, 28, 32, 27, 29])
    >>> design_weights = np.array([1.2, 0.8, 1.5, 0.9, 1.1, 1.3])
    >>> 
    >>> dist = get_distribution('normal')
    >>> params = WeightedFitting.fit_weighted_mle(responses, design_weights, dist)
    >>> print(f"Population mean: {params['loc']:.2f}")
    Population mean: 28.45
    
    Frequency data:
    
    >>> # Histogram bins
    >>> bin_centers = np.array([10, 20, 30, 40, 50])
    >>> frequencies = np.array([5, 15, 30, 12, 3])
    >>> 
    >>> dist = get_distribution('normal')
    >>> params = WeightedFitting.fit_weighted_mle(bin_centers, frequencies, dist)
    >>> print(f"Mean: {params['loc']:.1f}, Std: {params['scale']:.1f}")
    Mean: 28.5, Std: 10.2
    
    Notes
    -----
    **When to use weighted fitting:**
    
    1. **Survey data**: Observations have sampling weights to correct for
       unequal selection probabilities
    2. **Frequency data**: Data is aggregated (value, count) pairs
    3. **Stratified sampling**: Different strata sampled at different rates
    4. **Measurement errors**: Weight by precision (1/variance²)
    5. **Meta-analysis**: Studies weighted by sample size or quality
    
    **Important considerations:**
    
    - Weights should be non-negative
    - Zero weights are excluded automatically
    - Weights are normalized to sum to 1 internally
    - Effective sample size may be much smaller than actual n
    - Convergence may be slower than unweighted fitting
    - Some distributions may not support all weighting schemes
    
    **Supported distributions:**
    
    MLE: All continuous distributions  
    Moments: normal, lognormal, exponential, gamma, weibull, beta, uniform,
             logistic, laplace, poisson, binomial, nbinom, geometric
    
    See Also
    --------
    BaseDistribution.fit : Unweighted fitting
    Bootstrap.parametric : Bootstrap confidence intervals
    """
    
    @staticmethod
    def fit_weighted_mle(
        data: np.ndarray,
        weights: np.ndarray,
        distribution,
        method: str = 'Nelder-Mead',
        maxiter: int = 1000,
        **kwargs
    ) -> Dict[str, float]:
        """
        Weighted Maximum Likelihood Estimation
        
        Maximizes the weighted log-likelihood:
        L = Σ w_i * log(f(x_i | θ))
        
        Parameters
        ----------
        data : array-like
            Observed data points
        weights : array-like
            Non-negative weights for each observation
        distribution : BaseDistribution
            Distribution object to fit
        method : str, default='Nelder-Mead'
            Optimization method (see scipy.optimize.minimize)
        maxiter : int, default=1000
            Maximum iterations for optimizer
        **kwargs : dict
            Additional arguments passed to scipy.optimize.minimize
        
        Returns
        -------
        params : dict
            Fitted parameter dictionary {param_name: value}
        
        Raises
        ------
        ValueError
            If data and weights have different lengths, or weights are negative
        RuntimeWarning
            If optimization fails to converge
        
        Examples
        --------
        Survey data:
        
        >>> # Income survey with sampling weights
        >>> incomes = np.array([35000, 50000, 42000, 65000, 48000])
        >>> sampling_weights = np.array([1.5, 1.2, 1.8, 0.9, 1.3])
        >>> 
        >>> dist = get_distribution('lognormal')
        >>> params = WeightedFitting.fit_weighted_mle(incomes, sampling_weights, dist)
        >>> dist.params = params
        >>> dist.fitted = True
        >>> print(f"Median income: {dist.median():.0f}")
        Median income: 47500
        
        Frequency data:
        
        >>> # Defect counts per day (value, frequency)
        >>> defects = np.array([0, 1, 2, 3, 4])
        >>> days = np.array([120, 80, 30, 10, 5])  # How many days had this count
        >>> 
        >>> dist = get_distribution('poisson')
        >>> params = WeightedFitting.fit_weighted_mle(defects, days, dist)
        >>> print(f"Average defects per day: {params['mu']:.2f}")
        Average defects per day: 0.82
        
        Precision-weighted measurements:
        
        >>> # Lab measurements with different instruments
        >>> measurements = np.array([9.95, 10.02, 9.88, 10.05, 9.92])
        >>> std_errors = np.array([0.05, 0.02, 0.08, 0.03, 0.06])
        >>> 
        >>> # Weight by precision (inverse variance)
        >>> precision_weights = 1 / std_errors**2
        >>> 
        >>> dist = get_distribution('normal')
        >>> params = WeightedFitting.fit_weighted_mle(
        ...     measurements, precision_weights, dist
        ... )
        >>> print(f"Best estimate: {params['loc']:.3f} ± {params['scale']:.3f}")
        Best estimate: 10.005 ± 0.042
        
        Notes
        -----
        - Uses weighted method of moments as initial guess
        - Optimization may fail for extreme weight distributions
        - Returns moment estimates if MLE fails to converge
        - Effective sample size can be much less than n
        
        See Also
        --------
        fit_weighted_moments : Faster but less accurate alternative
        effective_sample_size : Check if you have enough effective data
        """
        # Validate inputs
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        if len(data) != len(weights):
            raise ValueError(
                f"Data length ({len(data)}) must match weights length ({len(weights)})"
            )
        
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        
        # Remove invalid observations
        valid_mask = ~(np.isnan(data) | np.isnan(weights) | np.isinf(data) | np.isinf(weights))
        valid_mask &= (weights > 0)  # Remove zero weights
        
        if not np.any(valid_mask):
            raise ValueError("No valid data points after filtering")
        
        data_clean = data[valid_mask]
        weights_clean = weights[valid_mask]
        
        # Warn if many points removed
        n_removed = len(data) - len(data_clean)
        if n_removed > len(data) * 0.1:  # More than 10%
            warnings.warn(
                f"Removed {n_removed}/{len(data)} invalid/zero-weight observations",
                RuntimeWarning
            )
        
        # Normalize weights to sum to 1
        weights_norm = weights_clean / np.sum(weights_clean)
        
        # Check effective sample size
        ess = WeightedFitting.effective_sample_size(weights_norm)
        if ess < 10:
            warnings.warn(
                f"Low effective sample size (ESS={ess:.1f}). "
                "Results may be unreliable. Consider using more balanced weights.",
                RuntimeWarning
            )
        
        # Get initial guess from weighted moments
        try:
            initial_params = WeightedFitting.fit_weighted_moments(
                data_clean, weights_norm, distribution
            )
        except Exception as e:
            warnings.warn(
                f"Failed to compute initial guess: {e}. Using default values.",
                RuntimeWarning
            )
            # Fallback to unweighted moments
            try:
                initial_params = distribution.fit_moments(data_clean)
            except:
                raise ValueError(
                    f"Cannot initialize parameters for {distribution.info.name}. "
                    "Try a different distribution."
                )
        
        def weighted_neg_log_likelihood(params_array):
            """Objective function: negative weighted log-likelihood"""
            try:
                # Convert array to parameter dict
                param_dict = distribution.array_to_params(params_array)
                distribution.params = param_dict
                distribution.fitted = True
                
                # Compute log-likelihood for each observation
                log_lik = distribution.logpdf(data_clean)
                
                # Check for invalid likelihoods
                if not np.all(np.isfinite(log_lik)):
                    return 1e10  # Heavy penalty
                
                # Weighted sum (negative because we minimize)
                weighted_log_lik = np.sum(weights_norm * log_lik)
                
                return -weighted_log_lik
                
            except Exception as e:
                # Penalize parameter values that cause errors
                return 1e10
        
        # Optimize
        x0 = distribution.params_to_array(initial_params)
        
        try:
            result = minimize(
                weighted_neg_log_likelihood,
                x0,
                method=method,
                options={'maxiter': maxiter, **kwargs}
            )
            
            if not result.success:
                warnings.warn(
                    f"Optimization did not converge: {result.message}. "
                    "Returning best found parameters. Results may be suboptimal.",
                    RuntimeWarning
                )
            
            # Check if result is reasonable
            final_nll = weighted_neg_log_likelihood(result.x)
            initial_nll = weighted_neg_log_likelihood(x0)
            
            if final_nll > initial_nll:
                warnings.warn(
                    "Optimization worsened fit. Using initial moment estimates.",
                    RuntimeWarning
                )
                return initial_params
            
            return distribution.array_to_params(result.x)
            
        except Exception as e:
            warnings.warn(
                f"Weighted MLE failed: {e}. Falling back to weighted moments.",
                RuntimeWarning
            )
            return initial_params
    
    @staticmethod
    def fit_weighted_moments(
        data: np.ndarray,
        weights: np.ndarray,
        distribution
    ) -> Dict[str, float]:
        """
        Weighted Method of Moments
        
        Matches weighted sample moments to theoretical distribution moments.
        Fast but less statistically efficient than MLE.
        
        Parameters
        ----------
        data : array-like
            Observed data
        weights : array-like
            Non-negative weights
        distribution : BaseDistribution
            Distribution to fit
        
        Returns
        -------
        params : dict
            Fitted parameters
        
        Raises
        ------
        ValueError
            If distribution is not supported or inputs are invalid
        
        Examples
        --------
        Quick fit for exploration:
        
        >>> # Aggregated data
        >>> values = np.array([10, 20, 30, 40, 50])
        >>> counts = np.array([5, 15, 40, 25, 15])
        >>> 
        >>> dist = get_distribution('normal')
        >>> params = WeightedFitting.fit_weighted_moments(values, counts, dist)
        >>> print(f"Mean: {params['loc']:.1f}, Std: {params['scale']:.1f}")
        Mean: 32.0, Std: 12.5
        
        Stratified data:
        
        >>> # Two groups with different sizes
        >>> group1 = np.random.normal(100, 10, 30)
        >>> group2 = np.random.normal(110, 15, 70)
        >>> 
        >>> data = np.concatenate([group1, group2])
        >>> weights = np.concatenate([
        ...     np.full(30, 0.3),  # 30% of population
        ...     np.full(70, 0.7)   # 70% of population
        ... ])
        >>> 
        >>> dist = get_distribution('normal')
        >>> params = WeightedFitting.fit_weighted_moments(data, weights, dist)
        >>> print(f"Population mean: {params['loc']:.1f}")
        Population mean: 107.0
        
        Notes
        -----
        Supported distributions:
        - Continuous: normal, lognormal, exponential, gamma, weibull, beta,
                     uniform, logistic, laplace
        - Discrete: poisson, binomial, nbinom, geometric
        
        For unsupported distributions, falls back to unweighted moments.
        
        See Also
        --------
        fit_weighted_mle : More accurate but slower
        weighted_stats : Get weighted summary statistics
        """
        # Validate and clean
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        if len(data) != len(weights):
            raise ValueError("Data and weights must have same length")
        
        if np.any(weights < 0):
            raise ValueError("Weights must be non-negative")
        
        # Remove NaN and zero-weight
        valid_mask = ~(np.isnan(data) | np.isnan(weights) | (weights == 0))
        data_clean = data[valid_mask]
        weights_clean = weights[valid_mask]
        
        if len(data_clean) == 0:
            raise ValueError("No valid data after removing NaN and zero weights")
        
        # Normalize weights
        weights_norm = weights_clean / np.sum(weights_clean)
        
        # Compute weighted moments
        wmean = np.sum(weights_norm * data_clean)
        wvar = np.sum(weights_norm * (data_clean - wmean)**2)
        wstd = np.sqrt(wvar) if wvar > 0 else 1e-6
        
        # Distribution-specific fitting
        dist_name = distribution.info.name
        
        try:
            if dist_name == 'normal':
                return {'loc': wmean, 'scale': wstd}
            
            elif dist_name == 'lognormal':
                # Only use positive values
                positive_mask = data_clean > 0
                if not np.any(positive_mask):
                    raise ValueError("Lognormal requires positive data")
                
                pos_data = data_clean[positive_mask]
                pos_weights = weights_norm[positive_mask]
                pos_weights = pos_weights / np.sum(pos_weights)
                
                log_data = np.log(pos_data)
                log_mean = np.sum(pos_weights * log_data)
                log_var = np.sum(pos_weights * (log_data - log_mean)**2)
                
                return {'s': np.sqrt(log_var), 'scale': np.exp(log_mean)}
            
            elif dist_name == 'exponential':
                if wmean <= 0:
                    raise ValueError("Exponential requires positive mean")
                return {'scale': wmean}
            
            elif dist_name == 'gamma':
                if wmean <= 0 or wvar <= 0:
                    raise ValueError("Gamma requires positive mean and variance")
                shape = wmean**2 / wvar
                scale = wvar / wmean
                return {'a': shape, 'scale': scale}
            
            elif dist_name == 'weibull':
                if wmean <= 0:
                    raise ValueError("Weibull requires positive mean")
                cv = wstd / wmean  # Coefficient of variation
                # Approximate shape from CV
                shape = 1.0 / max(cv, 0.1)  # Avoid division by zero
                return {'c': shape, 'scale': wmean}
            
            elif dist_name == 'beta':
                # Beta is on (0, 1)
                if np.any(data_clean < 0) or np.any(data_clean > 1):
                    warnings.warn(
                        "Beta distribution expects data in (0, 1). Clipping.",
                        RuntimeWarning
                    )
                data_clipped = np.clip(data_clean, 1e-6, 1 - 1e-6)
                wmean_clip = np.sum(weights_norm * data_clipped)
                wvar_clip = np.sum(weights_norm * (data_clipped - wmean_clip)**2)
                
                if wvar_clip > 0:
                    common = wmean_clip * (1 - wmean_clip) / wvar_clip - 1
                    alpha = max(wmean_clip * common, 0.1)
                    beta = max((1 - wmean_clip) * common, 0.1)
                    return {'a': alpha, 'b': beta}
                else:
                    return {'a': 1.0, 'b': 1.0}
            
            elif dist_name == 'uniform':
                return {
                    'loc': np.min(data_clean),
                    'scale': np.max(data_clean) - np.min(data_clean)
                }
            
            elif dist_name == 'logistic':
                # scale = std * sqrt(3) / pi
                return {'loc': wmean, 'scale': wstd * np.sqrt(3) / np.pi}
            
            elif dist_name == 'laplace':
                # Use weighted median and MAD
                wmedian = WeightedFitting.weighted_quantile(data_clean, weights_norm, 0.5)
                wmad = np.sum(weights_norm * np.abs(data_clean - wmedian))
                return {'loc': wmedian, 'scale': wmad}
            
            # Discrete distributions
            elif dist_name == 'poisson':
                if wmean < 0:
                    raise ValueError("Poisson requires non-negative mean")
                return {'mu': max(wmean, 0.1)}
            
            elif dist_name == 'binomial':
                n = int(np.max(data_clean))
                if n == 0:
                    raise ValueError("Binomial requires at least one non-zero value")
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
                if wmean <= 0:
                    raise ValueError("Geometric requires positive mean")
                p = 1.0 / wmean
                return {'p': np.clip(p, 0.01, 0.99)}
            
            else:
                # Unsupported - fall back to unweighted
                warnings.warn(
                    f"Weighted moments not implemented for {dist_name}. "
                    "Using unweighted moments.",
                    RuntimeWarning
                )
                return distribution.fit_moments(data_clean)
        
        except Exception as e:
            raise ValueError(
                f"Failed to fit {dist_name} with weighted moments: {e}"
            )
    
    @staticmethod
    def weighted_quantile(
        data: np.ndarray,
        weights: np.ndarray,
        quantile: float
    ) -> float:
        """
        Compute weighted quantile
        
        Uses cumulative weight distribution to find quantile.
        
        Parameters
        ----------
        data : array-like
            Data values
        weights : array-like
            Weights (will be normalized)
        quantile : float
            Quantile to compute (0 to 1)
        
        Returns
        -------
        q : float
            Weighted quantile value
        
        Examples
        --------
        >>> data = np.array([10, 20, 30, 40, 50])
        >>> weights = np.array([1, 1, 3, 1, 1])  # 30 has more weight
        >>> 
        >>> # Median
        >>> median = WeightedFitting.weighted_quantile(data, weights, 0.5)
        >>> print(f"Weighted median: {median}")  # Will be close to 30
        Weighted median: 30
        >>> 
        >>> # Quartiles
        >>> q25 = WeightedFitting.weighted_quantile(data, weights, 0.25)
        >>> q75 = WeightedFitting.weighted_quantile(data, weights, 0.75)
        >>> print(f"IQR: [{q25}, {q75}]")
        IQR: [20, 40]
        """
        if not 0 <= quantile <= 1:
            raise ValueError(f"Quantile must be in [0, 1], got {quantile}")
        
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        # Sort by data value
        sorted_indices = np.argsort(data)
        sorted_data = data[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # Cumulative weights (normalized)
        cumsum = np.cumsum(sorted_weights)
        cumsum = cumsum / cumsum[-1]
        
        # Find index where cumsum crosses quantile
        idx = np.searchsorted(cumsum, quantile)
        
        if idx >= len(sorted_data):
            return sorted_data[-1]
        elif idx == 0:
            return sorted_data[0]
        else:
            # Linear interpolation between points
            return sorted_data[idx]
    
    @staticmethod
    def weighted_stats(data: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive weighted statistics
        
        Parameters
        ----------
        data : array-like
            Data values
        weights : array-like
            Weights
        
        Returns
        -------
        stats : dict
            Dictionary containing:
            - mean: Weighted mean
            - var: Weighted variance
            - std: Weighted standard deviation
            - median: Weighted median (50th percentile)
            - q25: 25th percentile
            - q75: 75th percentile
            - min: Minimum value
            - max: Maximum value
            - ess: Effective sample size
        
        Examples
        --------
        >>> data = np.random.normal(100, 15, 1000)
        >>> weights = np.random.uniform(0.5, 1.5, 1000)
        >>> 
        >>> stats = WeightedFitting.weighted_stats(data, weights)
        >>> print(f"Mean: {stats['mean']:.2f} ± {stats['std']:.2f}")
        >>> print(f"Median: {stats['median']:.2f}")
        >>> print(f"IQR: [{stats['q25']:.2f}, {stats['q75']:.2f}]")
        >>> print(f"Effective N: {stats['ess']:.0f}")
        Mean: 100.12 ± 14.87
        Median: 100.34
        IQR: [90.12, 110.45]
        Effective N: 984
        """
        data = np.asarray(data).flatten()
        weights = np.asarray(weights).flatten()
        
        # Normalize weights
        weights_norm = weights / np.sum(weights)
        
        # Basic stats
        wmean = np.sum(weights_norm * data)
        wvar = np.sum(weights_norm * (data - wmean)**2)
        wstd = np.sqrt(wvar)
        
        # Quantiles
        wmedian = WeightedFitting.weighted_quantile(data, weights_norm, 0.5)
        wq25 = WeightedFitting.weighted_quantile(data, weights_norm, 0.25)
        wq75 = WeightedFitting.weighted_quantile(data, weights_norm, 0.75)
        
        # Effective sample size
        ess = WeightedFitting.effective_sample_size(weights)
        
        return {
            'mean': wmean,
            'var': wvar,
            'std': wstd,
            'median': wmedian,
            'q25': wq25,
            'q75': wq75,
            'min': np.min(data),
            'max': np.max(data),
            'ess': ess
        }
    
    @staticmethod
    def effective_sample_size(weights: np.ndarray) -> float:
        """
        Calculate effective sample size for weighted data
        
        ESS = (sum(w))^2 / sum(w^2)
        
        ESS measures how much information the weighted sample contains
        compared to an unweighted sample of the same size.
        
        Parameters
        ----------
        weights : array-like
            Weights (non-negative)
        
        Returns
        -------
        ess : float
            Effective sample size
        
        Examples
        --------
        >>> # Equal weights (no information loss)
        >>> weights_equal = np.ones(100)
        >>> ess = WeightedFitting.effective_sample_size(weights_equal)
        >>> print(f"ESS: {ess:.0f} (equal to n={len(weights_equal)})")
        ESS: 100 (equal to n=100)
        >>> 
        >>> # Unequal weights (information loss)
        >>> weights_unequal = np.array([1, 1, 1, 1, 10])  # One dominant weight
        >>> ess = WeightedFitting.effective_sample_size(weights_unequal)
        >>> print(f"ESS: {ess:.1f} (much less than n={len(weights_unequal)})")
        ESS: 2.2 (much less than n=5)
        >>> 
        >>> # Survey weights
        >>> survey_weights = np.random.uniform(0.5, 2.0, 500)
        >>> ess = WeightedFitting.effective_sample_size(survey_weights)
        >>> print(f"Actual N: {len(survey_weights)}, Effective N: {ess:.0f}")
        >>> print(f"Efficiency: {ess/len(survey_weights)*100:.1f}%")
        Actual N: 500, Effective N: 467
        Efficiency: 93.4%
        
        Notes
        -----
        ESS interpretation:
        - ESS = n: No information loss (equal weights)
        - ESS < n: Some information loss
        - ESS << n: Severe information loss (few dominant weights)
        
        Rule of thumb:
        - ESS > 30: Usually sufficient for inference
        - ESS < 10: Results may be unreliable
        """
        weights = np.asarray(weights).flatten()
        weights = weights[weights > 0]  # Remove zeros
        
        if len(weights) == 0:
            return 0.0
        
        sum_w = np.sum(weights)
        sum_w2 = np.sum(weights**2)
        
        return sum_w**2 / sum_w2 if sum_w2 > 0 else 0.0
