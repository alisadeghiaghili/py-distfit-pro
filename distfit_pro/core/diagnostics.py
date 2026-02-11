"""
Advanced Diagnostics
===================

Tools for assessing distribution fit quality:
- Residual analysis
- Outlier detection
- Influence analysis
- Tail behavior assessment
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from scipy import stats


@dataclass
class ResidualAnalysis:
    """
    Results from residual analysis
    
    Attributes:
    -----------
    raw_residuals : np.ndarray
        Observed - Expected
    standardized_residuals : np.ndarray
        Raw residuals / std(residuals)
    quantile_residuals : np.ndarray
        Based on probability integral transform
    """
    raw_residuals: np.ndarray
    standardized_residuals: np.ndarray
    quantile_residuals: np.ndarray
    
    @property
    def mean(self) -> float:
        return np.mean(self.raw_residuals)
    
    @property
    def std(self) -> float:
        return np.std(self.raw_residuals, ddof=1)
    
    @property
    def skewness(self) -> float:
        return stats.skew(self.raw_residuals)
    
    @property
    def kurtosis(self) -> float:
        return stats.kurtosis(self.raw_residuals)


@dataclass
class OutlierDetection:
    """
    Results from outlier detection
    
    Attributes:
    -----------
    outlier_indices : np.ndarray
        Indices of detected outliers
    outlier_values : np.ndarray
        Values of detected outliers
    method : str
        Detection method used
    threshold : float
        Threshold value used
    """
    outlier_indices: np.ndarray
    outlier_values: np.ndarray
    method: str
    threshold: float
    
    @property
    def n_outliers(self) -> int:
        return len(self.outlier_indices)
    
    @property
    def proportion(self) -> float:
        """Proportion of outliers"""
        return len(self.outlier_indices) / len(self.outlier_values) if len(self.outlier_values) > 0 else 0


@dataclass
class InfluenceAnalysis:
    """
    Results from influence analysis
    
    Attributes:
    -----------
    cook_distance : np.ndarray
        Cook's distance for each observation
    leverage : np.ndarray
        Leverage values
    influential_indices : np.ndarray
        Indices of influential points
    """
    cook_distance: np.ndarray
    leverage: np.ndarray
    influential_indices: np.ndarray
    
    @property
    def n_influential(self) -> int:
        return len(self.influential_indices)


class Diagnostics:
    """
    Advanced diagnostic tools for distribution fitting
    
    Example:
    --------
    >>> from distfit_pro.core.distributions import get_distribution
    >>> from distfit_pro.core.diagnostics import Diagnostics
    >>> 
    >>> data = np.random.normal(0, 1, 1000)
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> 
    >>> diag = Diagnostics()
    >>> residuals = diag.compute_residuals(data, dist)
    >>> outliers = diag.detect_outliers(data, dist)
    >>> influence = diag.analyze_influence(data, dist)
    """
    
    def compute_residuals(self,
                         data: np.ndarray,
                         distribution) -> ResidualAnalysis:
        """
        Compute different types of residuals
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        residuals : ResidualAnalysis
            Residual analysis results
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Sort data
        data_sorted = np.sort(data)
        
        # Expected values (theoretical quantiles)
        empirical_probs = (np.arange(1, n + 1) - 0.5) / n
        expected_values = distribution.ppf(empirical_probs)
        
        # Raw residuals
        raw_residuals = data_sorted - expected_values
        
        # Standardized residuals
        std_residuals = raw_residuals / np.std(raw_residuals, ddof=1)
        
        # Quantile residuals (probability integral transform)
        # More sophisticated: transforms data to N(0,1) if fit is good
        cdf_values = distribution.cdf(data_sorted)
        cdf_values = np.clip(cdf_values, 1e-10, 1 - 1e-10)
        quantile_residuals = stats.norm.ppf(cdf_values)
        
        return ResidualAnalysis(
            raw_residuals=raw_residuals,
            standardized_residuals=std_residuals,
            quantile_residuals=quantile_residuals
        )
    
    def detect_outliers(self,
                       data: np.ndarray,
                       distribution,
                       method: str = 'iqr',
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
            'iqr': Interquartile range
            'zscore': Z-score
            'modified_zscore': Modified Z-score (MAD-based)
            'quantile': Based on theoretical quantiles
        threshold : float, optional
            Custom threshold (method-specific)
            
        Returns:
        --------
        outliers : OutlierDetection
            Outlier detection results
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        if method == 'iqr':
            # Interquartile range method
            Q1, Q3 = np.percentile(data, [25, 75])
            IQR = Q3 - Q1
            threshold = threshold or 1.5
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (data < lower_bound) | (data > upper_bound)
        
        elif method == 'zscore':
            # Z-score method
            threshold = threshold or 3.0
            z_scores = np.abs((data - np.mean(data)) / np.std(data, ddof=1))
            outlier_mask = z_scores > threshold
        
        elif method == 'modified_zscore':
            # Modified Z-score using MAD
            threshold = threshold or 3.5
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / (mad + 1e-10)
            outlier_mask = np.abs(modified_z_scores) > threshold
        
        elif method == 'quantile':
            # Based on theoretical distribution
            threshold = threshold or 0.001  # Probability threshold
            cdf_values = distribution.cdf(data)
            outlier_mask = (cdf_values < threshold) | (cdf_values > 1 - threshold)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        outlier_indices = np.where(outlier_mask)[0]
        outlier_values = data[outlier_mask]
        
        return OutlierDetection(
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            method=method,
            threshold=threshold
        )
    
    def analyze_influence(self,
                         data: np.ndarray,
                         distribution,
                         threshold: Optional[float] = None) -> InfluenceAnalysis:
        """
        Analyze influence of individual observations
        
        Identifies points that have large impact on parameter estimates.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
        threshold : float, optional
            Threshold for Cook's distance (default: 4/n)
            
        Returns:
        --------
        influence : InfluenceAnalysis
            Influence analysis results
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Original parameters
        original_params = distribution.params.copy()
        
        # Compute Cook's distance by leave-one-out
        cook_distances = np.zeros(n)
        
        for i in range(n):
            # Leave out observation i
            data_loo = np.delete(data, i)
            
            # Refit
            try:
                dist_loo = distribution.__class__()
                dist_loo.fit(data_loo, method='mle')
                
                # Parameter change
                param_diff = np.array([
                    (dist_loo.params[k] - original_params[k]) / (original_params[k] + 1e-10)
                    for k in original_params.keys()
                ])
                
                # Cook's distance (simplified)
                cook_distances[i] = np.sum(param_diff**2)
            except:
                cook_distances[i] = 0
        
        # Leverage (simplified: based on distance from mean)
        mean_data = np.mean(data)
        std_data = np.std(data, ddof=1)
        leverage = np.abs(data - mean_data) / (std_data + 1e-10)
        leverage = leverage / np.max(leverage)  # Normalize to [0, 1]
        
        # Threshold for influential points
        if threshold is None:
            threshold = 4 / n
        
        influential_indices = np.where(cook_distances > threshold)[0]
        
        return InfluenceAnalysis(
            cook_distance=cook_distances,
            leverage=leverage,
            influential_indices=influential_indices
        )
    
    def assess_tail_behavior(self,
                            data: np.ndarray,
                            distribution) -> Dict:
        """
        Assess tail behavior of the fit
        
        Checks if the distribution adequately captures extreme values.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        tail_assessment : dict
            - left_tail: dict with p-value and verdict
            - right_tail: dict with p-value and verdict
            - overall: overall tail fit quality
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        # Left tail: compare 5th percentile region
        left_cutoff = np.percentile(data, 5)
        left_data = data[data <= left_cutoff]
        left_expected = np.sum(distribution.cdf(data) <= 0.05)
        left_observed = len(left_data)
        
        # Binomial test
        left_pvalue = stats.binom_test(left_observed, len(data), 0.05, alternative='two-sided')
        
        # Right tail: compare 95th percentile region
        right_cutoff = np.percentile(data, 95)
        right_data = data[data >= right_cutoff]
        right_expected = np.sum(distribution.cdf(data) >= 0.95)
        right_observed = len(right_data)
        
        right_pvalue = stats.binom_test(right_observed, len(data), 0.05, alternative='two-sided')
        
        # Verdict
        def verdict(pvalue):
            if pvalue < 0.01:
                return "poor"
            elif pvalue < 0.05:
                return "questionable"
            else:
                return "acceptable"
        
        return {
            'left_tail': {
                'observed': int(left_observed),
                'expected': float(left_expected),
                'p_value': float(left_pvalue),
                'verdict': verdict(left_pvalue)
            },
            'right_tail': {
                'observed': int(right_observed),
                'expected': float(right_expected),
                'p_value': float(right_pvalue),
                'verdict': verdict(right_pvalue)
            },
            'overall': 'acceptable' if left_pvalue >= 0.05 and right_pvalue >= 0.05 else 'poor'
        }
    
    def run_full_diagnostics(self,
                           data: np.ndarray,
                           distribution) -> Dict:
        """
        Run all diagnostic tests
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        diagnostics : dict
            Complete diagnostic results
        """
        return {
            'residuals': self.compute_residuals(data, distribution),
            'outliers_iqr': self.detect_outliers(data, distribution, method='iqr'),
            'outliers_zscore': self.detect_outliers(data, distribution, method='zscore'),
            'influence': self.analyze_influence(data, distribution),
            'tail_behavior': self.assess_tail_behavior(data, distribution)
        }


def format_diagnostics(diagnostics: Dict) -> str:
    """
    Format diagnostic results for display
    
    Parameters:
    -----------
    diagnostics : dict
        Results from Diagnostics.run_full_diagnostics()
        
    Returns:
    --------
    formatted : str
        Formatted string
    """
    output = []
    output.append("\n" + "="*70)
    output.append("DIAGNOSTIC ANALYSIS")
    output.append("="*70)
    
    # Residuals
    if 'residuals' in diagnostics:
        res = diagnostics['residuals']
        output.append("\nResidual Analysis:")
        output.append(f"  Mean:     {res.mean:.6f}")
        output.append(f"  Std:      {res.std:.6f}")
        output.append(f"  Skewness: {res.skewness:.6f}")
        output.append(f"  Kurtosis: {res.kurtosis:.6f}")
    
    # Outliers
    if 'outliers_iqr' in diagnostics:
        out = diagnostics['outliers_iqr']
        output.append(f"\nOutliers (IQR method):")
        output.append(f"  Count:      {out.n_outliers}")
        output.append(f"  Proportion: {out.proportion:.4f}")
    
    # Influence
    if 'influence' in diagnostics:
        inf = diagnostics['influence']
        output.append(f"\nInfluential Points:")
        output.append(f"  Count: {inf.n_influential}")
        if inf.n_influential > 0:
            output.append(f"  Indices: {inf.influential_indices[:10]}..." if len(inf.influential_indices) > 10 
                        else f"  Indices: {inf.influential_indices}")
    
    # Tail behavior
    if 'tail_behavior' in diagnostics:
        tail = diagnostics['tail_behavior']
        output.append(f"\nTail Behavior:")
        output.append(f"  Left tail:  {tail['left_tail']['verdict']} (p={tail['left_tail']['p_value']:.4f})")
        output.append(f"  Right tail: {tail['right_tail']['verdict']} (p={tail['right_tail']['p_value']:.4f})")
        output.append(f"  Overall:    {tail['overall']}")
    
    output.append("\n" + "="*70)
    return "\n".join(output)
