"""
Diagnostics for Distribution Fitting
=====================================

Provides residual analysis, influence diagnostics, and outlier detection.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class ResidualAnalysis:
    """
    Residual analysis results
    
    Attributes:
    -----------
    quantile_residuals : np.ndarray
        Quantile (randomized) residuals
    pearson_residuals : np.ndarray
        Pearson residuals
    deviance_residuals : np.ndarray
        Deviance residuals
    standardized_residuals : np.ndarray
        Standardized residuals
    """
    quantile_residuals: np.ndarray
    pearson_residuals: np.ndarray
    deviance_residuals: np.ndarray
    standardized_residuals: np.ndarray
    
    def summary(self) -> str:
        """Summary of residual diagnostics"""
        return f"""
Residual Analysis Summary
{'=' * 50}
Quantile Residuals:
  Mean: {np.mean(self.quantile_residuals):.6f}
  Std: {np.std(self.quantile_residuals):.6f}
  Range: [{np.min(self.quantile_residuals):.3f}, {np.max(self.quantile_residuals):.3f}]

Pearson Residuals:
  Mean: {np.mean(self.pearson_residuals):.6f}
  Std: {np.std(self.pearson_residuals):.6f}
  
Deviance Residuals:
  Mean: {np.mean(self.deviance_residuals):.6f}
  Std: {np.std(self.deviance_residuals):.6f}
"""


@dataclass
class InfluenceDiagnostics:
    """
    Influence diagnostics results
    
    Attributes:
    -----------
    cooks_distance : np.ndarray
        Cook's distance for each observation
    leverage : np.ndarray
        Leverage values
    dffits : np.ndarray
        DFFITS values
    influential_indices : np.ndarray
        Indices of influential observations
    """
    cooks_distance: np.ndarray
    leverage: np.ndarray
    dffits: np.ndarray
    influential_indices: np.ndarray
    
    def summary(self) -> str:
        """Summary of influence diagnostics"""
        return f"""
Influence Diagnostics Summary
{'=' * 50}
Cook's Distance:
  Max: {np.max(self.cooks_distance):.6f}
  Threshold: {4/len(self.cooks_distance):.6f}
  Influential: {len(self.influential_indices)} observations

Influential Indices: {self.influential_indices.tolist()[:10]}
{f'... and {len(self.influential_indices)-10} more' if len(self.influential_indices) > 10 else ''}
"""


@dataclass
class OutlierDetection:
    """
    Outlier detection results
    
    Attributes:
    -----------
    outlier_indices : np.ndarray
        Indices of detected outliers
    outlier_scores : np.ndarray
        Outlier scores for all observations
    method : str
        Detection method used
    threshold : float
        Threshold used for detection
    """
    outlier_indices: np.ndarray
    outlier_scores: np.ndarray
    method: str
    threshold: float
    
    def summary(self) -> str:
        """Summary of outlier detection"""
        return f"""
Outlier Detection Summary
{'=' * 50}
Method: {self.method}
Threshold: {self.threshold:.6f}
Outliers Detected: {len(self.outlier_indices)}

Outlier Indices: {self.outlier_indices.tolist()[:10]}
{f'... and {len(self.outlier_indices)-10} more' if len(self.outlier_indices) > 10 else ''}

Score Range: [{np.min(self.outlier_scores):.3f}, {np.max(self.outlier_scores):.3f}]
"""


class Diagnostics:
    """
    Diagnostic tools for fitted distributions
    
    Example:
    --------
    >>> from distfit_pro import get_distribution
    >>> from distfit_pro.core.diagnostics import Diagnostics
    >>> import numpy as np
    >>> 
    >>> data = np.random.normal(0, 1, 1000)
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> 
    >>> # Residual analysis
    >>> residuals = Diagnostics.residual_analysis(data, dist)
    >>> print(residuals.summary())
    >>> 
    >>> # Outlier detection
    >>> outliers = Diagnostics.detect_outliers(data, dist, method='zscore')
    >>> print(outliers.summary())
    """
    
    @staticmethod
    def residual_analysis(data: np.ndarray, distribution) -> ResidualAnalysis:
        """
        Compute various types of residuals
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        analysis : ResidualAnalysis
            Residual analysis results
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        # Quantile residuals (randomized quantile residuals)
        u = distribution.cdf(data)
        u = np.clip(u, 1e-10, 1 - 1e-10)  # Avoid extremes
        quantile_residuals = stats.norm.ppf(u)
        
        # Pearson residuals
        expected = distribution.mean()
        std = distribution.std()
        pearson_residuals = (data - expected) / std
        
        # Deviance residuals
        log_likelihood = distribution.logpdf(data)
        deviance_residuals = np.sign(data - expected) * np.sqrt(-2 * log_likelihood)
        
        # Standardized residuals
        residuals = data - expected
        standardized_residuals = residuals / np.std(residuals)
        
        return ResidualAnalysis(
            quantile_residuals=quantile_residuals,
            pearson_residuals=pearson_residuals,
            deviance_residuals=deviance_residuals,
            standardized_residuals=standardized_residuals
        )
    
    @staticmethod
    def influence_diagnostics(data: np.ndarray, distribution) -> InfluenceDiagnostics:
        """
        Compute influence diagnostics
        
        Identifies observations that have large influence on parameter estimates.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        diagnostics : InfluenceDiagnostics
            Influence diagnostics results
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Cook's distance (approximation)
        residuals = Diagnostics.residual_analysis(data, distribution)
        standardized_res = residuals.standardized_residuals
        
        # Leverage (approximation using PDF)
        pdf_values = distribution.pdf(data)
        leverage = pdf_values / np.max(pdf_values)
        
        # Cook's distance
        cooks_distance = (standardized_res**2 / 2) * (leverage / (1 - leverage + 1e-10))
        
        # DFFITS
        dffits = standardized_res * np.sqrt(leverage / (1 - leverage + 1e-10))
        
        # Identify influential observations (Cook's D > 4/n)
        threshold = 4 / n
        influential_indices = np.where(cooks_distance > threshold)[0]
        
        return InfluenceDiagnostics(
            cooks_distance=cooks_distance,
            leverage=leverage,
            dffits=dffits,
            influential_indices=influential_indices
        )
    
    @staticmethod
    def detect_outliers(data: np.ndarray,
                       distribution,
                       method: str = 'zscore',
                       threshold: Optional[float] = None) -> OutlierDetection:
        """
        Detect outliers using various methods
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
        method : str
            Detection method: 'zscore', 'iqr', 'likelihood'
        threshold : float, optional
            Custom threshold (default depends on method)
            
        Returns:
        --------
        detection : OutlierDetection
            Outlier detection results
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        if method == 'zscore':
            # Z-score method
            mean = distribution.mean()
            std = distribution.std()
            z_scores = np.abs((data - mean) / std)
            
            if threshold is None:
                threshold = 3.0
            
            outlier_indices = np.where(z_scores > threshold)[0]
            outlier_scores = z_scores
            
        elif method == 'iqr':
            # Interquartile Range method
            q1 = distribution.ppf(0.25)
            q3 = distribution.ppf(0.75)
            iqr = q3 - q1
            
            if threshold is None:
                threshold = 1.5
            
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_indices = np.where((data < lower_bound) | (data > upper_bound))[0]
            outlier_scores = np.minimum(
                np.abs(data - lower_bound) / iqr,
                np.abs(data - upper_bound) / iqr
            )
            
        elif method == 'likelihood':
            # Likelihood-based method
            log_likelihood = distribution.logpdf(data)
            
            if threshold is None:
                # Use 1st percentile of log-likelihood
                threshold = np.percentile(log_likelihood, 1)
            
            outlier_indices = np.where(log_likelihood < threshold)[0]
            outlier_scores = -log_likelihood
            
        elif method == 'mahalanobis':
            # Mahalanobis distance (univariate case)
            mean = distribution.mean()
            std = distribution.std()
            mahal_dist = np.abs((data - mean) / std)
            
            if threshold is None:
                threshold = stats.chi2.ppf(0.99, df=1)
            
            outlier_indices = np.where(mahal_dist**2 > threshold)[0]
            outlier_scores = mahal_dist**2
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'zscore', 'iqr', 'likelihood', or 'mahalanobis'")
        
        return OutlierDetection(
            outlier_indices=outlier_indices,
            outlier_scores=outlier_scores,
            method=method,
            threshold=threshold
        )
    
    @staticmethod
    def qq_diagnostics(data: np.ndarray, distribution) -> Dict[str, np.ndarray]:
        """
        Compute Q-Q plot data for diagnostics
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        qq_data : dict
            Dictionary with 'theoretical', 'sample', and 'residuals'
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        data_sorted = np.sort(data)
        n = len(data)
        
        # Theoretical quantiles
        p = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quantiles = distribution.ppf(p)
        
        # Sample quantiles
        sample_quantiles = data_sorted
        
        # Q-Q residuals
        qq_residuals = sample_quantiles - theoretical_quantiles
        
        return {
            'theoretical': theoretical_quantiles,
            'sample': sample_quantiles,
            'residuals': qq_residuals,
            'correlation': np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
        }
    
    @staticmethod
    def pp_diagnostics(data: np.ndarray, distribution) -> Dict[str, np.ndarray]:
        """
        Compute P-P plot data for diagnostics
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        pp_data : dict
            Dictionary with 'theoretical', 'empirical', and 'residuals'
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        data_sorted = np.sort(data)
        n = len(data)
        
        # Empirical probabilities
        empirical_prob = (np.arange(1, n + 1)) / n
        
        # Theoretical probabilities
        theoretical_prob = distribution.cdf(data_sorted)
        
        # P-P residuals
        pp_residuals = empirical_prob - theoretical_prob
        
        return {
            'theoretical': theoretical_prob,
            'empirical': empirical_prob,
            'residuals': pp_residuals,
            'max_deviation': np.max(np.abs(pp_residuals))
        }
    
    @staticmethod
    def worm_plot_data(data: np.ndarray, distribution) -> Dict[str, np.ndarray]:
        """
        Compute worm plot data (detrended Q-Q plot)
        
        Useful for detecting systematic deviations from the fitted distribution.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        worm_data : dict
            Dictionary with worm plot data
        """
        qq_data = Diagnostics.qq_diagnostics(data, distribution)
        
        theoretical = qq_data['theoretical']
        sample = qq_data['sample']
        
        # Fit linear trend
        slope, intercept = np.polyfit(theoretical, sample, 1)
        
        # Detrended residuals
        expected = slope * theoretical + intercept
        worm_residuals = sample - expected
        
        # Standardize
        worm_residuals = worm_residuals / np.std(worm_residuals)
        
        return {
            'theoretical': theoretical,
            'worm_residuals': worm_residuals,
            'slope': slope,
            'intercept': intercept
        }
