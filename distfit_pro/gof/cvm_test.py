"""
Cramer-von Mises Goodness-of-Fit Test
====================================

Implementation of the Cramer-von Mises test for goodness-of-fit.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from scipy import stats
from typing import Dict
from .base import GOFTest, GOFResult
from ..core.base import BaseDistribution


class CramerVonMisesTest(GOFTest):
    """
    Cramer-von Mises goodness-of-fit test.
    
    The CvM test is based on the integrated squared difference between
    the empirical and theoretical CDFs.
    
    Test Statistic:
        W² = (1/12n) + sum[(F(x_i) - (2i-1)/(2n))²]
    
    Advantages:
    -----------
    - More powerful than KS for certain alternatives
    - Gives equal weight to all parts of distribution
    - Distribution-free (asymptotically)
    
    Limitations:
    -----------
    - Less intuitive than KS test
    - Critical values depend on distribution
    - Less well-known than KS or AD
    
    References
    ----------
    - Cramér, H. (1928). "On the composition of elementary errors"
    - von Mises, R. (1928). "Wahrscheinlichkeit, Statistik und Wahrheit"
    """
    
    @property
    def name(self) -> str:
        return "Cramér-von Mises Test"
    
    @property
    def description(self) -> str:
        return (
            "Tests goodness-of-fit using integrated squared difference between CDFs. "
            "Provides good overall power across the distribution."
        )
    
    def test(
        self,
        data: np.ndarray,
        distribution: BaseDistribution
    ) -> GOFResult:
        """
        Perform Cramér-von Mises test.
        
        Parameters
        ----------
        data : array-like
            Sample data
        distribution : BaseDistribution
            Fitted distribution to test against
        
        Returns
        -------
        result : GOFResult
            Test result with CvM statistic and p-value
        """
        data = self._validate_inputs(data, distribution)
        n = len(data)
        
        # Calculate CvM statistic
        cvm_statistic = self._calculate_cvm_statistic(data, distribution)
        
        # Adjust for sample size
        cvm_modified = cvm_statistic * (1 + 0.5/n)
        
        # Estimate p-value
        p_value = self._estimate_p_value(cvm_modified)
        
        # Get critical values
        critical_values = self._get_critical_values()
        
        extra_info = {
            'cvm_statistic_unadjusted': cvm_statistic,
            'cvm_statistic_modified': cvm_modified,
            'note': 'Modified CvM statistic accounts for sample size'
        }
        
        return GOFResult(
            test_name=self.name,
            statistic=cvm_statistic,
            p_value=p_value,
            critical_values=critical_values,
            alpha=self.alpha,
            sample_size=n,
            distribution_name=distribution.info.display_name,
            extra_info=extra_info
        )
    
    def _calculate_cvm_statistic(self, data: np.ndarray, distribution: BaseDistribution) -> float:
        """
        Calculate Cramér-von Mises statistic.
        
        Parameters
        ----------
        data : ndarray
            Sample data
        distribution : BaseDistribution
            Theoretical distribution
        
        Returns
        -------
        cvm_statistic : float
            Cramér-von Mises test statistic
        """
        n = len(data)
        data_sorted = np.sort(data)
        
        # Calculate CDF values
        cdf_vals = distribution.cdf(data_sorted)
        
        # Calculate CvM statistic
        i = np.arange(1, n + 1)
        cvm_statistic = (1 / (12 * n)) + np.sum((cdf_vals - (2*i - 1) / (2*n))**2)
        
        return cvm_statistic
    
    def _estimate_p_value(self, cvm_modified: float) -> float:
        """
        Estimate p-value using empirical formula.
        
        Based on asymptotic distribution approximations.
        
        Parameters
        ----------
        cvm_modified : float
            Modified CvM statistic
        
        Returns
        -------
        p_value : float
            Approximate p-value
        """
        # Empirical approximation (similar to AD test)
        # These formulas are approximations for the normal distribution
        if cvm_modified < 0.0275:
            p_value = 1 - np.exp(-13.953 + 775.5*cvm_modified - 12542.6*cvm_modified**2)
        elif cvm_modified < 0.051:
            p_value = 1 - np.exp(-5.903 + 179.546*cvm_modified - 1515.29*cvm_modified**2)
        elif cvm_modified < 0.092:
            p_value = np.exp(0.886 - 31.62*cvm_modified + 10.897*cvm_modified**2)
        else:
            p_value = np.exp(1.111 - 34.242*cvm_modified + 12.832*cvm_modified**2)
        
        return np.clip(p_value, 0, 1)
    
    def _get_critical_values(self) -> Dict[float, float]:
        """
        Get critical values for common significance levels.
        
        These are approximate values for the normal distribution.
        
        Returns
        -------
        critical_values : dict
            Critical values at common alpha levels
        """
        return {
            0.10: 0.104,  # 10% significance
            0.05: 0.126,  # 5% significance
            0.025: 0.148,  # 2.5% significance
            0.01: 0.178,  # 1% significance
        }
