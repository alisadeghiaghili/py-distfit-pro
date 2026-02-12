"""
Chi-Square Goodness-of-Fit Test
==============================

Implementation of the Chi-Square test for goodness-of-fit.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from scipy import stats
from typing import Optional, Union
from .base import GOFTest, GOFResult
from ..core.base import BaseDistribution


class ChiSquareTest(GOFTest):
    """
    Chi-Square goodness-of-fit test.
    
    The Chi-Square test divides the data into bins and compares
    observed frequencies with expected frequencies.
    
    Test Statistic:
        χ² = sum[(O_i - E_i)² / E_i]
    
    where O_i is observed frequency and E_i is expected frequency.
    
    Advantages:
    -----------
    - Works for discrete and continuous distributions
    - Easy to understand and interpret
    - Flexible binning strategies
    
    Limitations:
    -----------
    - Requires adequate sample size (n > 30)
    - Sensitive to choice of bins
    - Each bin should have at least 5 expected observations
    - Less powerful than KS or AD for continuous distributions
    
    References
    ----------
    - Pearson, K. (1900). "On the criterion that a given system of deviations"
    """
    
    def __init__(self, n_bins: Optional[int] = None, alpha: float = 0.05):
        """
        Initialize Chi-Square test.
        
        Parameters
        ----------
        n_bins : int, optional
            Number of bins. If None, determined automatically.
        alpha : float, default=0.05
            Significance level
        """
        super().__init__(alpha=alpha)
        self.n_bins = n_bins
    
    @property
    def name(self) -> str:
        return "Chi-Square Test"
    
    @property
    def description(self) -> str:
        return (
            "Tests goodness-of-fit by comparing observed and expected frequencies "
            "in bins. Works for both discrete and continuous distributions."
        )
    
    def test(
        self,
        data: np.ndarray,
        distribution: BaseDistribution,
        bins: Optional[Union[int, np.ndarray]] = None
    ) -> GOFResult:
        """
        Perform Chi-Square test.
        
        Parameters
        ----------
        data : array-like
            Sample data
        distribution : BaseDistribution
            Fitted distribution to test against
        bins : int or array-like, optional
            Number of bins or bin edges. If None, determined automatically.
        
        Returns
        -------
        result : GOFResult
            Test result with chi-square statistic and p-value
        """
        data = self._validate_inputs(data, distribution)
        n = len(data)
        
        # Determine bins
        if bins is None:
            bins = self._determine_bins(n)
        
        # Create histogram
        observed, bin_edges = np.histogram(data, bins=bins)
        
        # Calculate expected frequencies
        expected = self._calculate_expected(n, bin_edges, distribution)
        
        # Combine bins with low expected frequency
        observed, expected = self._combine_low_frequency_bins(observed, expected)
        
        # Calculate chi-square statistic
        chi2_statistic = np.sum((observed - expected)**2 / expected)
        
        # Degrees of freedom
        # df = number of bins - 1 - number of estimated parameters
        n_params = len(distribution.params)
        df = len(observed) - 1 - n_params
        df = max(df, 1)  # At least 1 degree of freedom
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi2_statistic, df)
        
        # Critical values
        critical_values = {
            0.10: stats.chi2.ppf(0.90, df),
            0.05: stats.chi2.ppf(0.95, df),
            0.01: stats.chi2.ppf(0.99, df),
        }
        
        extra_info = {
            'degrees_of_freedom': df,
            'n_bins': len(observed),
            'n_parameters': n_params,
            'observed_frequencies': observed.tolist(),
            'expected_frequencies': expected.tolist(),
        }
        
        return GOFResult(
            test_name=self.name,
            statistic=chi2_statistic,
            p_value=p_value,
            critical_values=critical_values,
            alpha=self.alpha,
            sample_size=n,
            distribution_name=distribution.info.display_name,
            extra_info=extra_info
        )
    
    def _determine_bins(self, n: int) -> int:
        """
        Determine optimal number of bins.
        
        Uses Sturges' rule as default: k = 1 + log2(n)
        
        Parameters
        ----------
        n : int
            Sample size
        
        Returns
        -------
        n_bins : int
            Number of bins
        """
        if self.n_bins is not None:
            return self.n_bins
        
        # Sturges' rule
        n_bins = int(np.ceil(1 + np.log2(n)))
        
        # Ensure reasonable range
        n_bins = max(5, min(n_bins, 20))
        
        return n_bins
    
    def _calculate_expected(self, n: int, bin_edges: np.ndarray, distribution: BaseDistribution) -> np.ndarray:
        """
        Calculate expected frequencies in bins.
        
        Parameters
        ----------
        n : int
            Sample size
        bin_edges : ndarray
            Bin edges
        distribution : BaseDistribution
            Theoretical distribution
        
        Returns
        -------
        expected : ndarray
            Expected frequencies
        """
        # Calculate probabilities for each bin
        probabilities = np.diff(distribution.cdf(bin_edges))
        
        # Expected frequencies
        expected = n * probabilities
        
        return expected
    
    def _combine_low_frequency_bins(self, observed: np.ndarray, expected: np.ndarray, min_expected: float = 5.0) -> tuple:
        """
        Combine bins with low expected frequency.
        
        Chi-square test requires each bin to have at least 5 expected observations.
        
        Parameters
        ----------
        observed : ndarray
            Observed frequencies
        expected : ndarray
            Expected frequencies
        min_expected : float, default=5.0
            Minimum expected frequency per bin
        
        Returns
        -------
        observed_combined : ndarray
            Combined observed frequencies
        expected_combined : ndarray
            Combined expected frequencies
        """
        observed_combined = []
        expected_combined = []
        
        current_obs = 0
        current_exp = 0
        
        for obs, exp in zip(observed, expected):
            current_obs += obs
            current_exp += exp
            
            if current_exp >= min_expected:
                observed_combined.append(current_obs)
                expected_combined.append(current_exp)
                current_obs = 0
                current_exp = 0
        
        # Add remaining to last bin
        if current_obs > 0 or current_exp > 0:
            if len(observed_combined) > 0:
                observed_combined[-1] += current_obs
                expected_combined[-1] += current_exp
            else:
                observed_combined.append(current_obs)
                expected_combined.append(current_exp)
        
        return np.array(observed_combined), np.array(expected_combined)

