"""
Anderson-Darling Goodness-of-Fit Test
====================================

Implementation of the Anderson-Darling test for goodness-of-fit.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from scipy import stats
from typing import Optional, Dict
from .base import GOFTest, GOFResult
from ..core.base import BaseDistribution


class AndersonDarlingTest(GOFTest):
    """
    Anderson-Darling goodness-of-fit test.
    
    The Anderson-Darling test is a modification of the Kolmogorov-Smirnov test
    that gives more weight to the tails of the distribution.
    
    Test Statistic:
        A² = -n - (1/n) * sum[(2i-1) * (ln(F(x_i)) + ln(1-F(x_{n+1-i})))]
    
    Advantages:
    -----------
    - More powerful than KS test, especially for tail differences
    - Gives more weight to extreme values
    - Distribution-specific critical values available
    
    Limitations:
    -----------
    - Critical values depend on the distribution
    - More complex than KS test
    - Sensitive to parameter estimation
    
    References
    ----------
    - Anderson, T.W. and Darling, D.A. (1952). "Asymptotic theory of certain
      'goodness of fit' criteria based on stochastic processes"
    """
    
    @property
    def name(self) -> str:
        return "Anderson-Darling Test"
    
    @property
    def description(self) -> str:
        return (
            "Tests goodness-of-fit with emphasis on tail behavior. "
            "More powerful than KS test for detecting distribution differences."
        )
    
    def test(
        self,
        data: np.ndarray,
        distribution: BaseDistribution
    ) -> GOFResult:
        """
        Perform Anderson-Darling test.
        
        Parameters
        ----------
        data : array-like
            Sample data
        distribution : BaseDistribution
            Fitted distribution to test against
        
        Returns
        -------
        result : GOFResult
            Test result with AD statistic and p-value
        """
        data = self._validate_inputs(data, distribution)
        n = len(data)
        
        # Calculate AD statistic
        ad_statistic = self._calculate_ad_statistic(data, distribution)
        
        # Adjust for sample size (modified AD statistic)
        ad_modified = ad_statistic * (1 + 0.75/n + 2.25/n**2)
        
        # Get p-value (approximate)
        p_value = self._estimate_p_value(ad_modified)
        
        # Get critical values
        critical_values = self._get_critical_values()
        
        extra_info = {
            'ad_statistic_unadjusted': ad_statistic,
            'ad_statistic_modified': ad_modified,
            'note': 'Modified AD statistic accounts for sample size'
        }
        
        return GOFResult(
            test_name=self.name,
            statistic=ad_statistic,
            p_value=p_value,
            critical_values=critical_values,
            alpha=self.alpha,
            sample_size=n,
            distribution_name=distribution.info.display_name,
            extra_info=extra_info
        )
    
    def _calculate_ad_statistic(self, data: np.ndarray, distribution: BaseDistribution) -> float:
        """
        Calculate Anderson-Darling statistic.
        
        Parameters
        ----------
        data : ndarray
            Sample data
        distribution : BaseDistribution
            Theoretical distribution
        
        Returns
        -------
        ad_statistic : float
            Anderson-Darling test statistic
        """
        n = len(data)
        data_sorted = np.sort(data)
        
        # Calculate CDF values
        cdf_vals = distribution.cdf(data_sorted)
        
        # Avoid log(0) and log(1)
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)
        
        # Calculate AD statistic
        i = np.arange(1, n + 1)
        ad_statistic = -n - np.sum(
            (2*i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
        ) / n
        
        return ad_statistic
    
    def _estimate_p_value(self, ad_modified: float) -> float:
        """
        Estimate p-value using empirical formula.
        
        Based on D'Agostino & Stephens (1986) approximation.
        
        Parameters
        ----------
        ad_modified : float
            Modified AD statistic
        
        Returns
        -------
        p_value : float
            Approximate p-value
        """
        # Empirical approximation for normal distribution
        # This is an approximation and works reasonably well for most distributions
        if ad_modified < 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14*ad_modified - 223.73*ad_modified**2)
        elif ad_modified < 0.34:
            p_value = 1 - np.exp(-8.318 + 42.796*ad_modified - 59.938*ad_modified**2)
        elif ad_modified < 0.6:
            p_value = np.exp(0.9177 - 4.279*ad_modified - 1.38*ad_modified**2)
        else:
            p_value = np.exp(1.2937 - 5.709*ad_modified + 0.0186*ad_modified**2)
        
        return np.clip(p_value, 0, 1)
    
    def _get_critical_values(self) -> Dict[float, float]:
        """
        Get critical values for common significance levels.
        
        These are approximate values for the normal distribution.
        Values may differ for other distributions.
        
        Returns
        -------
        critical_values : dict
            Critical values at common alpha levels
        """
        return {
            0.10: 0.631,  # 10% significance
            0.05: 0.752,  # 5% significance
            0.025: 0.873,  # 2.5% significance
            0.01: 1.035,  # 1% significance
        }

