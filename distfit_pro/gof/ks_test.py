"""
Kolmogorov-Smirnov Goodness-of-Fit Test
=======================================

Implementation of the Kolmogorov-Smirnov test for goodness-of-fit.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from scipy import stats
from typing import Optional
from .base import GOFTest, GOFResult
from ..core.base import BaseDistribution


class KolmogorovSmirnovTest(GOFTest):
    """
    Kolmogorov-Smirnov goodness-of-fit test.
    
    The KS test compares the empirical cumulative distribution function
    with the theoretical CDF. The test statistic is the maximum absolute
    difference between the two CDFs.
    
    Test Statistic:
        D = sup|F_n(x) - F(x)|
    
    where F_n is the empirical CDF and F is the theoretical CDF.
    
    Advantages:
    -----------
    - Distribution-free (non-parametric)
    - Sensitive to differences in location and shape
    - Works for continuous distributions
    - Easy to interpret
    
    Limitations:
    -----------
    - Less powerful than Anderson-Darling for detecting tail differences
    - Requires continuous distributions
    - Conservative for discrete distributions
    
    References
    ----------
    - Kolmogorov, A. (1933). "Sulla determinazione empirica di una legge di distribuzione"
    - Smirnov, N. (1948). "Table for estimating the goodness of fit of empirical distributions"
    """
    
    @property
    def name(self) -> str:
        return "Kolmogorov-Smirnov Test"
    
    @property
    def description(self) -> str:
        return (
            "Tests goodness-of-fit by comparing empirical and theoretical CDFs. "
            "Measures maximum vertical distance between the two CDFs."
        )
    
    def test(
        self,
        data: np.ndarray,
        distribution: BaseDistribution,
        alternative: str = 'two-sided'
    ) -> GOFResult:
        """
        Perform Kolmogorov-Smirnov test.
        
        Parameters
        ----------
        data : array-like
            Sample data
        distribution : BaseDistribution
            Fitted distribution to test against
        alternative : str, default='two-sided'
            Alternative hypothesis: 'two-sided', 'less', or 'greater'
        
        Returns
        -------
        result : GOFResult
            Test result with KS statistic and p-value
        """
        # Validate inputs
        data = self._validate_inputs(data, distribution)
        n = len(data)
        
        # Perform KS test using scipy
        # We use the CDF from our fitted distribution
        ks_statistic, p_value = stats.kstest(
            data,
            distribution.cdf,
            alternative=alternative
        )
        
        # Get critical values
        critical_values = self._get_critical_values(n)
        
        # Additional information
        x_sorted, ecdf = self._calculate_ecdf(data)
        tcdf = distribution.cdf(x_sorted)
        
        # Find location of maximum difference
        diff = np.abs(ecdf - tcdf)
        max_diff_idx = np.argmax(diff)
        max_diff_location = x_sorted[max_diff_idx]
        
        extra_info = {
            'alternative': alternative,
            'max_diff_location': max_diff_location,
            'ecdf_at_max': ecdf[max_diff_idx],
            'tcdf_at_max': tcdf[max_diff_idx],
        }
        
        return GOFResult(
            test_name=self.name,
            statistic=ks_statistic,
            p_value=p_value,
            critical_values=critical_values,
            alpha=self.alpha,
            sample_size=n,
            distribution_name=distribution.info.display_name,
            extra_info=extra_info
        )
    
    def _get_critical_values(self, n: int) -> dict:
        """
        Get critical values for common significance levels.
        
        For large n, uses asymptotic approximation:
        D_critical ≈ c(alpha) / sqrt(n)
        
        Parameters
        ----------
        n : int
            Sample size
        
        Returns
        -------
        critical_values : dict
            Critical values at alpha = 0.10, 0.05, 0.01
        """
        # Asymptotic critical values for two-sided test
        # c(alpha) values from KS distribution
        c_values = {
            0.10: 1.22,  # 90% confidence
            0.05: 1.36,  # 95% confidence
            0.01: 1.63,  # 99% confidence
        }
        
        critical_values = {}
        for alpha, c in c_values.items():
            if n > 35:
                # Asymptotic formula
                critical_values[alpha] = c / np.sqrt(n)
            else:
                # Use scipy's ppf for exact values
                critical_values[alpha] = stats.ksone.ppf(1 - alpha, n)
        
        return critical_values
    
    def test_with_bootstrap(
        self,
        data: np.ndarray,
        distribution: BaseDistribution,
        n_bootstrap: int = 1000,
        random_state: Optional[int] = None
    ) -> GOFResult:
        """
        Perform KS test with bootstrap p-value estimation.
        
        This is useful when the theoretical p-value may not be accurate
        (e.g., when distribution parameters are estimated from data).
        
        Parameters
        ----------
        data : array-like
            Sample data
        distribution : BaseDistribution
            Fitted distribution
        n_bootstrap : int, default=1000
            Number of bootstrap samples
        random_state : int, optional
            Random seed for reproducibility
        
        Returns
        -------
        result : GOFResult
            Test result with bootstrap p-value
        """
        data = self._validate_inputs(data, distribution)
        n = len(data)
        
        # Calculate observed KS statistic
        ks_obs, _ = stats.kstest(data, distribution.cdf)
        
        # Bootstrap
        np.random.seed(random_state)
        ks_bootstrap = np.zeros(n_bootstrap)
        
        for i in range(n_bootstrap):
            # Generate sample from fitted distribution
            boot_sample = distribution.rvs(size=n)
            
            # Fit same distribution type to bootstrap sample
            boot_dist = distribution.__class__()
            boot_dist.fit(boot_sample, method='mle')
            
            # Calculate KS statistic
            ks_bootstrap[i], _ = stats.kstest(boot_sample, boot_dist.cdf)
        
        # Calculate bootstrap p-value
        p_value = np.mean(ks_bootstrap >= ks_obs)
        
        extra_info = {
            'bootstrap_samples': n_bootstrap,
            'bootstrap_mean_statistic': np.mean(ks_bootstrap),
            'bootstrap_std_statistic': np.std(ks_bootstrap),
        }
        
        return GOFResult(
            test_name=f"{self.name} (Bootstrap)",
            statistic=ks_obs,
            p_value=p_value,
            alpha=self.alpha,
            sample_size=n,
            distribution_name=distribution.info.display_name,
            extra_info=extra_info
        )
