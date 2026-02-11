"""
Goodness-of-Fit Tests
====================

Implements statistical tests to assess distribution fit quality:
- Kolmogorov-Smirnov (KS)
- Anderson-Darling (AD)
- Cramér-von Mises (CVM)
- Chi-Square (χ²)
- Likelihood Ratio Test
"""

from typing import Dict, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.special import kolmogorov


class GoodnessOfFitTests:
    """
    Collection of goodness-of-fit tests for distribution fitting
    
    Example:
    --------
    >>> from distfit_pro.core.distributions import get_distribution
    >>> from distfit_pro.core.gof_tests import GoodnessOfFitTests
    >>> 
    >>> data = np.random.normal(0, 1, 1000)
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> 
    >>> gof = GoodnessOfFitTests()
    >>> ks_result = gof.kolmogorov_smirnov(data, dist)
    >>> print(f"KS statistic: {ks_result['statistic']:.4f}")
    >>> print(f"p-value: {ks_result['p_value']:.4f}")
    """
    
    @staticmethod
    def kolmogorov_smirnov(data: np.ndarray, 
                          distribution,
                          alternative: str = 'two-sided') -> Dict:
        """
        Kolmogorov-Smirnov test
        
        Tests the hypothesis that data follows the fitted distribution by
        comparing empirical CDF with theoretical CDF.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
        alternative : str
            'two-sided' (default), 'less', or 'greater'
            
        Returns:
        --------
        result : dict
            - statistic: KS test statistic (D)
            - p_value: p-value
            - critical_value: critical value at α=0.05
            - reject_null: whether to reject H0 at α=0.05
            
        Interpretation:
        ---------------
        - Small p-value (< 0.05): reject null hypothesis → poor fit
        - Large p-value (≥ 0.05): fail to reject → acceptable fit
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Empirical CDF
        data_sorted = np.sort(data)
        ecdf = np.arange(1, n + 1) / n
        
        # Theoretical CDF
        tcdf = distribution.cdf(data_sorted)
        
        # KS statistic
        if alternative == 'two-sided':
            d_plus = np.max(ecdf - tcdf)
            d_minus = np.max(tcdf - (ecdf - 1/n))
            statistic = max(d_plus, d_minus)
        elif alternative == 'less':
            statistic = np.max(tcdf - (ecdf - 1/n))
        elif alternative == 'greater':
            statistic = np.max(ecdf - tcdf)
        else:
            raise ValueError(f"Invalid alternative: {alternative}")
        
        # p-value using Kolmogorov distribution
        if alternative == 'two-sided':
            p_value = 2 * (1 - kolmogorov(np.sqrt(n) * statistic))
            p_value = np.clip(p_value, 0, 1)
        else:
            p_value = np.exp(-2 * n * statistic**2)
        
        # Critical value at α=0.05
        if alternative == 'two-sided':
            critical_value = 1.36 / np.sqrt(n)  # Approximation
        else:
            critical_value = 1.22 / np.sqrt(n)
        
        return {
            'test': 'Kolmogorov-Smirnov',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'critical_value': float(critical_value),
            'reject_null': p_value < 0.05,
            'n': n,
            'alternative': alternative
        }
    
    @staticmethod
    def anderson_darling(data: np.ndarray, distribution) -> Dict:
        """
        Anderson-Darling test
        
        More sensitive to tail differences than KS test.
        Gives more weight to deviations in the tails.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        result : dict
            - statistic: A² test statistic
            - p_value: approximate p-value
            - critical_values: dict of critical values for different α
            - reject_null: whether to reject H0 at α=0.05
            
        Interpretation:
        ---------------
        - Larger A² → worse fit
        - A² < critical value → acceptable fit
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Sort data
        data_sorted = np.sort(data)
        
        # Theoretical CDF
        F = distribution.cdf(data_sorted)
        F = np.clip(F, 1e-10, 1 - 1e-10)  # Avoid log(0)
        
        # Anderson-Darling statistic
        i = np.arange(1, n + 1)
        A2 = -n - np.sum((2*i - 1) * (np.log(F) + np.log(1 - F[::-1]))) / n
        
        # Adjusted statistic
        A2_adj = A2 * (1 + 0.75/n + 2.25/n**2)
        
        # Critical values (approximate, for normal distribution)
        critical_values = {
            0.10: 0.631,
            0.05: 0.752,
            0.025: 0.873,
            0.01: 1.035
        }
        
        # Approximate p-value (for normal distribution)
        if A2_adj < 0.2:
            p_value = 1 - np.exp(-13.436 + 101.14*A2_adj - 223.73*A2_adj**2)
        elif A2_adj < 0.34:
            p_value = 1 - np.exp(-8.318 + 42.796*A2_adj - 59.938*A2_adj**2)
        elif A2_adj < 0.6:
            p_value = np.exp(0.9177 - 4.279*A2_adj - 1.38*A2_adj**2)
        else:
            p_value = np.exp(1.2937 - 5.709*A2_adj + 0.0186*A2_adj**2)
        
        p_value = np.clip(p_value, 0, 1)
        
        return {
            'test': 'Anderson-Darling',
            'statistic': float(A2),
            'statistic_adjusted': float(A2_adj),
            'p_value': float(p_value),
            'critical_values': critical_values,
            'reject_null': A2_adj > critical_values[0.05],
            'n': n
        }
    
    @staticmethod
    def cramer_von_mises(data: np.ndarray, distribution) -> Dict:
        """
        Cramér-von Mises test
        
        Similar to KS but uses squared differences.
        More sensitive to differences in the middle of the distribution.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        result : dict
            - statistic: W² test statistic
            - p_value: approximate p-value
            - critical_value: critical value at α=0.05
            - reject_null: whether to reject H0 at α=0.05
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Sort data
        data_sorted = np.sort(data)
        
        # Theoretical CDF
        F = distribution.cdf(data_sorted)
        
        # Cramér-von Mises statistic
        i = np.arange(1, n + 1)
        W2 = np.sum((F - (2*i - 1)/(2*n))**2) + 1/(12*n)
        
        # Adjusted statistic
        W2_adj = W2 * (1 + 0.5/n)
        
        # Critical value at α=0.05 (approximate)
        critical_value = 0.461 / (1 + 1/n)
        
        # Approximate p-value
        if W2_adj < 0.0275:
            p_value = 1 - 0.0
        elif W2_adj < 0.051:
            p_value = 1 - np.exp(-13.953 + 775.5*W2_adj - 12542.6*W2_adj**2)
        elif W2_adj < 0.092:
            p_value = 1 - np.exp(-5.903 + 179.546*W2_adj - 1515.29*W2_adj**2)
        else:
            p_value = np.exp(-0.886 - 31.62*W2_adj + 10.897*W2_adj**2)
        
        p_value = np.clip(p_value, 0, 1)
        
        return {
            'test': 'Cramér-von Mises',
            'statistic': float(W2),
            'statistic_adjusted': float(W2_adj),
            'p_value': float(p_value),
            'critical_value': float(critical_value),
            'reject_null': W2_adj > critical_value,
            'n': n
        }
    
    @staticmethod
    def chi_square(data: np.ndarray, 
                   distribution,
                   bins: Optional[int] = None,
                   min_expected: float = 5.0) -> Dict:
        """
        Chi-Square goodness-of-fit test
        
        Tests by binning data and comparing observed vs expected frequencies.
        Best for discrete distributions or when data is naturally binned.
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
        bins : int, optional
            Number of bins (default: Sturges' rule)
        min_expected : float
            Minimum expected frequency per bin (bins may be merged)
            
        Returns:
        --------
        result : dict
            - statistic: χ² test statistic
            - p_value: p-value from chi-square distribution
            - df: degrees of freedom
            - bins: number of bins used
            - reject_null: whether to reject H0 at α=0.05
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        n = len(data)
        
        # Determine number of bins
        if bins is None:
            # Sturges' rule
            bins = int(np.ceil(np.log2(n) + 1))
        
        # Create bins
        if distribution._is_discrete:
            # For discrete: use integer bins
            min_val, max_val = int(np.floor(data.min())), int(np.ceil(data.max()))
            bin_edges = np.arange(min_val, max_val + 2) - 0.5
        else:
            # For continuous: equal-width bins
            bin_edges = np.linspace(data.min(), data.max(), bins + 1)
        
        # Observed frequencies
        observed, _ = np.histogram(data, bins=bin_edges)
        
        # Expected frequencies
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        if distribution._is_discrete:
            # For discrete: use PMF
            expected = n * distribution.pdf(np.round(bin_centers))
        else:
            # For continuous: integrate PDF over bins
            expected = n * (distribution.cdf(bin_edges[1:]) - 
                          distribution.cdf(bin_edges[:-1]))
        
        # Merge bins with low expected frequency
        while len(expected) > 2 and np.min(expected) < min_expected:
            min_idx = np.argmin(expected)
            if min_idx == 0:
                observed[1] += observed[0]
                expected[1] += expected[0]
                observed = observed[1:]
                expected = expected[1:]
            elif min_idx == len(expected) - 1:
                observed[-2] += observed[-1]
                expected[-2] += expected[-1]
                observed = observed[:-1]
                expected = expected[:-1]
            else:
                # Merge with neighbor with smaller expected frequency
                if expected[min_idx-1] < expected[min_idx+1]:
                    observed[min_idx-1] += observed[min_idx]
                    expected[min_idx-1] += expected[min_idx]
                    observed = np.delete(observed, min_idx)
                    expected = np.delete(expected, min_idx)
                else:
                    observed[min_idx+1] += observed[min_idx]
                    expected[min_idx+1] += expected[min_idx]
                    observed = np.delete(observed, min_idx)
                    expected = np.delete(expected, min_idx)
        
        # Chi-square statistic
        chi2 = np.sum((observed - expected)**2 / (expected + 1e-10))
        
        # Degrees of freedom: bins - 1 - number of estimated parameters
        k = len(observed)
        num_params = len(distribution.params)
        df = k - 1 - num_params
        
        if df <= 0:
            return {
                'test': 'Chi-Square',
                'statistic': np.nan,
                'p_value': np.nan,
                'df': df,
                'bins': k,
                'reject_null': None,
                'warning': 'Not enough bins for test (df ≤ 0)'
            }
        
        # p-value
        p_value = 1 - stats.chi2.cdf(chi2, df)
        
        return {
            'test': 'Chi-Square',
            'statistic': float(chi2),
            'p_value': float(p_value),
            'df': int(df),
            'bins': int(k),
            'reject_null': p_value < 0.05,
            'n': n
        }
    
    @staticmethod
    def likelihood_ratio(data: np.ndarray,
                        dist1,
                        dist2) -> Dict:
        """
        Likelihood Ratio Test
        
        Compare two nested models (dist2 must be more general than dist1).
        
        Parameters:
        -----------
        data : array-like
            Observed data
        dist1 : BaseDistribution
            Simpler (nested) model
        dist2 : BaseDistribution
            More complex model
            
        Returns:
        --------
        result : dict
            - statistic: -2 * log(L1/L2) = 2 * (logL2 - logL1)
            - p_value: p-value from chi-square distribution
            - df: difference in number of parameters
            - reject_null: whether dist2 is significantly better
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        # Log-likelihoods
        logL1 = np.sum(dist1.logpdf(data))
        logL2 = np.sum(dist2.logpdf(data))
        
        # LR statistic
        lr_stat = 2 * (logL2 - logL1)
        
        # Degrees of freedom
        df = len(dist2.params) - len(dist1.params)
        
        if df <= 0:
            return {
                'test': 'Likelihood Ratio',
                'statistic': np.nan,
                'p_value': np.nan,
                'df': df,
                'reject_null': None,
                'warning': 'dist2 must have more parameters than dist1'
            }
        
        # p-value
        p_value = 1 - stats.chi2.cdf(lr_stat, df)
        
        return {
            'test': 'Likelihood Ratio',
            'statistic': float(lr_stat),
            'p_value': float(p_value),
            'df': int(df),
            'logL1': float(logL1),
            'logL2': float(logL2),
            'reject_null': p_value < 0.05,
            'interpretation': 'dist2 significantly better' if p_value < 0.05 else 'no significant difference'
        }
    
    def run_all_tests(self, data: np.ndarray, distribution) -> Dict:
        """
        Run all applicable goodness-of-fit tests
        
        Parameters:
        -----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution
            
        Returns:
        --------
        results : dict
            Dictionary with all test results
        """
        results = {
            'ks': self.kolmogorov_smirnov(data, distribution),
            'ad': self.anderson_darling(data, distribution),
            'cvm': self.cramer_von_mises(data, distribution),
            'chi2': self.chi_square(data, distribution)
        }
        
        # Summary
        results['summary'] = {
            'all_pass': all(not r.get('reject_null', False) 
                          for r in results.values() 
                          if isinstance(r, dict) and 'reject_null' in r),
            'any_fail': any(r.get('reject_null', False) 
                          for r in results.values() 
                          if isinstance(r, dict) and 'reject_null' in r)
        }
        
        return results


def format_gof_results(results: Dict) -> str:
    """
    Format GOF test results for display
    
    Parameters:
    -----------
    results : dict
        Results from GoodnessOfFitTests.run_all_tests()
        
    Returns:
    --------
    formatted : str
        Formatted string
    """
    output = []
    output.append("\n" + "="*70)
    output.append("GOODNESS-OF-FIT TEST RESULTS")
    output.append("="*70)
    
    for test_key, result in results.items():
        if test_key == 'summary':
            continue
        if not isinstance(result, dict):
            continue
        
        output.append(f"\n{result['test']}:")
        output.append(f"  Statistic: {result['statistic']:.6f}")
        output.append(f"  p-value:   {result['p_value']:.6f}")
        
        if result.get('reject_null') is not None:
            status = "❌ REJECT H0 (poor fit)" if result['reject_null'] else "✅ FAIL TO REJECT (acceptable fit)"
            output.append(f"  Result:    {status}")
    
    if 'summary' in results:
        output.append("\n" + "-"*70)
        if results['summary']['all_pass']:
            output.append("✅ ALL TESTS PASS - Good fit")
        elif results['summary']['any_fail']:
            output.append("❌ SOME TESTS FAIL - Consider alternative distribution")
    
    output.append("="*70)
    return "\n".join(output)
