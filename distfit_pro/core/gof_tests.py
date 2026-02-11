"""
Goodness-of-Fit Tests
=====================

Statistical tests to assess how well a fitted distribution matches the data.

Implemented tests:
- Kolmogorov-Smirnov (KS)
- Anderson-Darling (AD)
- Chi-Square (χ²)
- Cramér-von Mises (CvM)
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from scipy import stats
from scipy.special import kolmogorov


@dataclass
class GOFResult:
    """
    Result of a goodness-of-fit test
    
    Attributes:
    -----------
    test_name : str
        Name of the test
    statistic : float
        Test statistic value
    p_value : float
        P-value of the test
    critical_value : float, optional
        Critical value at alpha=0.05
    reject_null : bool
        Whether to reject null hypothesis at alpha=0.05
    interpretation : str
        Human-readable interpretation
    """
    test_name: str
    statistic: float
    p_value: float
    critical_value: Optional[float] = None
    reject_null: bool = False
    interpretation: str = ""
    
    def __repr__(self) -> str:
        return f"""
{self.test_name} Test Results
{'=' * 50}
Statistic: {self.statistic:.6f}
P-value: {self.p_value:.6f}
Reject H0 (α=0.05): {self.reject_null}

{self.interpretation}
"""


class GOFTests:
    """
    Goodness-of-Fit Tests for Distribution Fitting
    
    Example:
    --------
    >>> from distfit_pro import get_distribution
    >>> import numpy as np
    >>> 
    >>> data = np.random.normal(0, 1, 1000)
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> 
    >>> # Perform GOF tests
    >>> ks_result = GOFTests.kolmogorov_smirnov(data, dist)
    >>> ad_result = GOFTests.anderson_darling(data, dist)
    >>> chi_result = GOFTests.chi_square(data, dist)
    >>> 
    >>> print(ks_result)
    >>> print(ad_result)
    >>> print(chi_result)
    """
    
    @staticmethod
    def kolmogorov_smirnov(data: np.ndarray, 
                          distribution, 
                          alpha: float = 0.05) -> GOFResult:
        """
        Kolmogorov-Smirnov Test
        
        Tests if data follows the specified distribution by measuring
        the maximum distance between empirical and theoretical CDFs.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution object
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        result : GOFResult
            Test result with statistic and p-value
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Compute empirical CDF
        data_sorted = np.sort(data)
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Compute theoretical CDF
        theoretical_cdf = distribution.cdf(data_sorted)
        
        # KS statistic: maximum absolute difference
        d_plus = np.max(empirical_cdf - theoretical_cdf)
        d_minus = np.max(theoretical_cdf - (empirical_cdf - 1/n))
        ks_stat = max(d_plus, d_minus)
        
        # P-value using Kolmogorov distribution
        p_value = 1 - kolmogorov(ks_stat * np.sqrt(n))
        
        # Critical value
        critical_value = 1.36 / np.sqrt(n)  # For alpha=0.05
        
        reject_null = p_value < alpha
        
        if reject_null:
            interpretation = (
                f"The data does NOT follow the {distribution.info.display_name}.\n"
                f"Evidence: D = {ks_stat:.6f} > {critical_value:.6f} (critical value)\n"
                f"The fitted distribution is likely inappropriate for this data."
            )
        else:
            interpretation = (
                f"The data is consistent with the {distribution.info.display_name}.\n"
                f"Evidence: D = {ks_stat:.6f} ≤ {critical_value:.6f} (critical value)\n"
                f"No significant evidence against the fitted distribution."
            )
        
        return GOFResult(
            test_name="Kolmogorov-Smirnov",
            statistic=ks_stat,
            p_value=p_value,
            critical_value=critical_value,
            reject_null=reject_null,
            interpretation=interpretation
        )
    
    @staticmethod
    def anderson_darling(data: np.ndarray, 
                        distribution, 
                        alpha: float = 0.05) -> GOFResult:
        """
        Anderson-Darling Test
        
        More sensitive to tail differences than KS test.
        Gives more weight to deviations in the distribution tails.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution object
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        result : GOFResult
            Test result with statistic and p-value
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Sort data
        data_sorted = np.sort(data)
        
        # Compute theoretical CDF
        F = distribution.cdf(data_sorted)
        
        # Avoid log(0)
        F = np.clip(F, 1e-300, 1 - 1e-300)
        
        # Anderson-Darling statistic
        i = np.arange(1, n + 1)
        ad_stat = -n - np.sum((2*i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n
        
        # Critical values (approximate for general distributions)
        critical_values = {
            0.10: 1.933,
            0.05: 2.492,
            0.025: 3.070,
            0.01: 3.857
        }
        
        critical_value = critical_values.get(alpha, 2.492)
        
        # Approximate p-value (rough estimate)
        if ad_stat < 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14*ad_stat - 223.73*ad_stat**2)
        elif ad_stat < 0.34:
            p_value = 1 - np.exp(-8.318 + 42.796*ad_stat - 59.938*ad_stat**2)
        elif ad_stat < 0.6:
            p_value = np.exp(0.9177 - 4.279*ad_stat - 1.38*ad_stat**2)
        elif ad_stat < 10:
            p_value = np.exp(1.2937 - 5.709*ad_stat + 0.0186*ad_stat**2)
        else:
            p_value = 3.7e-24
        
        p_value = np.clip(p_value, 0, 1)
        reject_null = ad_stat > critical_value
        
        if reject_null:
            interpretation = (
                f"The data does NOT follow the {distribution.info.display_name}.\n"
                f"Evidence: A² = {ad_stat:.6f} > {critical_value:.6f} (critical value)\n"
                f"Particularly poor fit in the distribution tails."
            )
        else:
            interpretation = (
                f"The data is consistent with the {distribution.info.display_name}.\n"
                f"Evidence: A² = {ad_stat:.6f} ≤ {critical_value:.6f} (critical value)\n"
                f"Good fit, including in the tails."
            )
        
        return GOFResult(
            test_name="Anderson-Darling",
            statistic=ad_stat,
            p_value=p_value,
            critical_value=critical_value,
            reject_null=reject_null,
            interpretation=interpretation
        )
    
    @staticmethod
    def chi_square(data: np.ndarray, 
                   distribution, 
                   n_bins: Optional[int] = None,
                   alpha: float = 0.05) -> GOFResult:
        """
        Chi-Square Goodness-of-Fit Test
        
        Tests if observed frequencies match expected frequencies
        from the fitted distribution.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution object
        n_bins : int, optional
            Number of bins (default: auto-calculated)
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        result : GOFResult
            Test result with statistic and p-value
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Auto-calculate bins if not provided
        if n_bins is None:
            # Sturges' rule with minimum
            n_bins = max(int(np.ceil(np.log2(n) + 1)), 10)
        
        # For discrete distributions, use actual values as bins
        if hasattr(distribution, '_is_discrete') and distribution._is_discrete:
            unique_vals = np.unique(data)
            if len(unique_vals) <= 30:
                observed, bin_edges = np.histogram(data, bins=len(unique_vals))
                expected = np.array([n * distribution.pdf(np.array([v]))[0] 
                                   for v in unique_vals])
            else:
                observed, bin_edges = np.histogram(data, bins=n_bins)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                bin_width = bin_edges[1] - bin_edges[0]
                expected = np.array([n * distribution.pdf(np.array([c]))[0] * bin_width 
                                   for c in bin_centers])
        else:
            # Continuous distribution
            observed, bin_edges = np.histogram(data, bins=n_bins)
            
            # Calculate expected frequencies
            expected_probs = np.diff(distribution.cdf(bin_edges))
            expected = n * expected_probs
        
        # Merge bins with expected < 5
        mask = expected >= 5
        if np.sum(mask) < 3:
            # Too few bins, reduce n_bins
            return GOFTests.chi_square(data, distribution, 
                                      n_bins=max(n_bins//2, 3), alpha=alpha)
        
        observed = observed[mask]
        expected = expected[mask]
        
        # Chi-square statistic
        chi2_stat = np.sum((observed - expected)**2 / expected)
        
        # Degrees of freedom
        k = len(observed)
        n_params = len(distribution.params) if distribution.params else 0
        df = k - 1 - n_params
        
        if df <= 0:
            return GOFResult(
                test_name="Chi-Square",
                statistic=chi2_stat,
                p_value=np.nan,
                reject_null=False,
                interpretation="Not enough degrees of freedom for test."
            )
        
        # P-value
        p_value = 1 - stats.chi2.cdf(chi2_stat, df)
        
        # Critical value
        critical_value = stats.chi2.ppf(1 - alpha, df)
        
        reject_null = p_value < alpha
        
        if reject_null:
            interpretation = (
                f"The data does NOT follow the {distribution.info.display_name}.\n"
                f"Evidence: χ² = {chi2_stat:.6f} > {critical_value:.6f} (critical value)\n"
                f"Observed frequencies differ significantly from expected."
            )
        else:
            interpretation = (
                f"The data is consistent with the {distribution.info.display_name}.\n"
                f"Evidence: χ² = {chi2_stat:.6f} ≤ {critical_value:.6f} (critical value)\n"
                f"Observed frequencies match expected frequencies well."
            )
        
        return GOFResult(
            test_name="Chi-Square",
            statistic=chi2_stat,
            p_value=p_value,
            critical_value=critical_value,
            reject_null=reject_null,
            interpretation=interpretation
        )
    
    @staticmethod
    def cramer_von_mises(data: np.ndarray, 
                        distribution, 
                        alpha: float = 0.05) -> GOFResult:
        """
        Cramér-von Mises Test
        
        Similar to KS but uses squared differences integrated over
        the entire distribution (more sensitive to middle deviations).
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution object
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        result : GOFResult
            Test result with statistic and p-value
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Sort data
        data_sorted = np.sort(data)
        
        # Compute theoretical CDF
        F = distribution.cdf(data_sorted)
        
        # Cramér-von Mises statistic
        i = np.arange(1, n + 1)
        cvm_stat = np.sum((F - (2*i - 1)/(2*n))**2) + 1/(12*n)
        
        # Critical values (approximate)
        critical_values = {
            0.10: 0.347,
            0.05: 0.461,
            0.025: 0.581,
            0.01: 0.743
        }
        
        critical_value = critical_values.get(alpha, 0.461)
        
        # Approximate p-value
        if cvm_stat < 0.0275:
            p_value = 1.0
        elif cvm_stat < 0.051:
            p_value = 1 - np.exp(-13.953 + 775.5*cvm_stat - 12542.61*cvm_stat**2)
        elif cvm_stat < 0.092:
            p_value = 1 - np.exp(-5.903 + 179.546*cvm_stat - 1515.29*cvm_stat**2)
        elif cvm_stat < 0.1:
            p_value = np.exp(0.886 - 31.62*cvm_stat + 10.897*cvm_stat**2)
        elif cvm_stat < 1.1:
            p_value = np.exp(1.111 - 34.242*cvm_stat + 12.832*cvm_stat**2)
        else:
            p_value = 7.37e-10
        
        p_value = np.clip(p_value, 0, 1)
        reject_null = cvm_stat > critical_value
        
        if reject_null:
            interpretation = (
                f"The data does NOT follow the {distribution.info.display_name}.\n"
                f"Evidence: W² = {cvm_stat:.6f} > {critical_value:.6f} (critical value)\n"
                f"Systematic deviation from expected distribution."
            )
        else:
            interpretation = (
                f"The data is consistent with the {distribution.info.display_name}.\n"
                f"Evidence: W² = {cvm_stat:.6f} ≤ {critical_value:.6f} (critical value)\n"
                f"No significant systematic deviation."
            )
        
        return GOFResult(
            test_name="Cramér-von Mises",
            statistic=cvm_stat,
            p_value=p_value,
            critical_value=critical_value,
            reject_null=reject_null,
            interpretation=interpretation
        )
    
    @staticmethod
    def run_all_tests(data: np.ndarray, 
                     distribution,
                     alpha: float = 0.05) -> Dict[str, GOFResult]:
        """
        Run all available GOF tests
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution object
        alpha : float
            Significance level (default: 0.05)
            
        Returns:
        --------
        results : dict
            Dictionary of test results
            
        Example:
        --------
        >>> results = GOFTests.run_all_tests(data, dist)
        >>> for name, result in results.items():
        ...     print(result)
        """
        results = {}
        
        try:
            results['ks'] = GOFTests.kolmogorov_smirnov(data, distribution, alpha)
        except Exception as e:
            print(f"KS test failed: {e}")
        
        try:
            results['ad'] = GOFTests.anderson_darling(data, distribution, alpha)
        except Exception as e:
            print(f"AD test failed: {e}")
        
        try:
            results['chi2'] = GOFTests.chi_square(data, distribution, alpha=alpha)
        except Exception as e:
            print(f"Chi-square test failed: {e}")
        
        try:
            results['cvm'] = GOFTests.cramer_von_mises(data, distribution, alpha)
        except Exception as e:
            print(f"CvM test failed: {e}")
        
        return results
    
    @staticmethod
    def summary_table(results: Dict[str, GOFResult]) -> str:
        """
        Create summary table of all test results
        
        Parameters:
        -----------
        results : dict
            Dictionary of GOF test results
            
        Returns:
        --------
        summary : str
            Formatted table
        """
        summary = """
╔═══════════════════════════════════════════════════════════════╗
║                 Goodness-of-Fit Test Summary                  ║
╠═══════════════════════════════════════════════════════════════╣
║  Test                  Statistic    P-value    Reject H0      ║
╠═══════════════════════════════════════════════════════════════╣
"""
        
        for name, result in results.items():
            test_abbr = result.test_name[:20].ljust(20)
            reject_str = "Yes" if result.reject_null else "No"
            summary += f"║  {test_abbr}  {result.statistic:>10.6f}  {result.p_value:>9.6f}  {reject_str:^10}  ║\n"
        
        summary += "╚═══════════════════════════════════════════════════════════════╝\n"
        
        # Overall assessment
        n_reject = sum(1 for r in results.values() if r.reject_null)
        n_total = len(results)
        
        if n_reject == 0:
            summary += "\n✅ All tests passed: Distribution fits well!\n"
        elif n_reject == n_total:
            summary += "\n❌ All tests failed: Distribution does NOT fit!\n"
        else:
            summary += f"\n⚠️  Mixed results: {n_reject}/{n_total} tests failed.\n"
        
        return summary
