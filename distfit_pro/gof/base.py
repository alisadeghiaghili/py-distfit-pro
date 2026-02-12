"""
Base Classes for Goodness-of-Fit Tests
======================================

Abstract base classes and utilities for GOF testing.

Author: Ali Sadeghi Aghili
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from ..core.base import BaseDistribution


@dataclass
class GOFResult:
    """
    Result of a goodness-of-fit test.
    
    Attributes
    ----------
    test_name : str
        Name of the test performed
    statistic : float
        Test statistic value
    p_value : float
        P-value of the test
    critical_values : dict, optional
        Critical values at different significance levels
    reject_null : bool
        Whether to reject null hypothesis at alpha=0.05
    alpha : float
        Significance level used
    sample_size : int
        Size of the sample
    distribution_name : str
        Name of the tested distribution
    extra_info : dict, optional
        Additional test-specific information
    """
    test_name: str
    statistic: float
    p_value: float
    critical_values: Optional[Dict[float, float]] = None
    reject_null: bool = False
    alpha: float = 0.05
    sample_size: int = 0
    distribution_name: str = ""
    extra_info: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Calculate reject_null after initialization"""
        if self.p_value is not None:
            self.reject_null = self.p_value < self.alpha
    
    def summary(self) -> str:
        """
        Generate human-readable summary.
        
        Returns
        -------
        summary : str
            Formatted summary of test results
        """
        lines = [
            f"{'='*60}",
            f"{self.test_name}",
            f"{'='*60}",
            f"Distribution: {self.distribution_name}",
            f"Sample size: {self.sample_size}",
            f"Significance level: {self.alpha}",
            "",
            f"Test statistic: {self.statistic:.6f}",
            f"P-value: {self.p_value:.6f}",
            "",
        ]
        
        if self.critical_values:
            lines.append("Critical values:")
            for alpha, cv in sorted(self.critical_values.items()):
                lines.append(f"  α = {alpha}: {cv:.6f}")
            lines.append("")
        
        # Decision
        if self.reject_null:
            decision = f"REJECT null hypothesis (p={self.p_value:.4f} < {self.alpha})"
            interpretation = "The data does NOT fit the specified distribution well."
        else:
            decision = f"FAIL TO REJECT null hypothesis (p={self.p_value:.4f} >= {self.alpha})"
            interpretation = "The data fits the specified distribution reasonably well."
        
        lines.extend([
            f"Decision: {decision}",
            f"Interpretation: {interpretation}",
        ])
        
        if self.extra_info:
            lines.append("\nAdditional Information:")
            for key, value in self.extra_info.items():
                lines.append(f"  {key}: {value}")
        
        lines.append("="*60)
        return "\n".join(lines)
    
    def interpret_strength(self) -> str:
        """
        Interpret the strength of evidence.
        
        Returns
        -------
        interpretation : str
            Qualitative interpretation of p-value
        """
        if self.p_value < 0.01:
            return "Very strong evidence against the fit"
        elif self.p_value < 0.05:
            return "Strong evidence against the fit"
        elif self.p_value < 0.10:
            return "Moderate evidence against the fit"
        elif self.p_value < 0.20:
            return "Weak evidence against the fit"
        else:
            return "Little or no evidence against the fit"
    
    def __str__(self) -> str:
        return self.summary()
    
    def __repr__(self) -> str:
        return (
            f"GOFResult(test='{self.test_name}', "
            f"statistic={self.statistic:.4f}, "
            f"p_value={self.p_value:.4f}, "
            f"reject={self.reject_null})"
        )


class GOFTest(ABC):
    """
    Abstract base class for goodness-of-fit tests.
    
    All GOF tests should inherit from this class and implement
    the test() method.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize GOF test.
        
        Parameters
        ----------
        alpha : float, default=0.05
            Significance level for hypothesis testing
        """
        self.alpha = alpha
    
    @abstractmethod
    def test(
        self,
        data: np.ndarray,
        distribution: BaseDistribution
    ) -> GOFResult:
        """
        Perform goodness-of-fit test.
        
        Parameters
        ----------
        data : array-like
            Observed data
        distribution : BaseDistribution
            Fitted distribution to test against
        
        Returns
        -------
        result : GOFResult
            Test result with statistic and p-value
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the test"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Brief description of the test"""
        pass
    
    def _validate_inputs(
        self,
        data: np.ndarray,
        distribution: BaseDistribution
    ) -> np.ndarray:
        """
        Validate inputs for GOF test.
        
        Parameters
        ----------
        data : array-like
            Data to validate
        distribution : BaseDistribution
            Distribution to validate
        
        Returns
        -------
        data : ndarray
            Validated and cleaned data
        
        Raises
        ------
        ValueError
            If inputs are invalid
        """
        # Convert to numpy array
        data = np.asarray(data, dtype=float)
        
        # Check for empty data
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        
        # Check for NaN/Inf
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValueError("Data contains infinite values")
        
        # Check distribution is fitted
        if not distribution.fitted:
            raise ValueError("Distribution must be fitted before testing")
        
        # Minimum sample size
        if data.size < 3:
            raise ValueError("Sample size must be at least 3")
        
        return data
    
    def _calculate_ecdf(self, data: np.ndarray) -> tuple:
        """
        Calculate empirical cumulative distribution function.
        
        Parameters
        ----------
        data : ndarray
            Data points
        
        Returns
        -------
        x : ndarray
            Sorted data points
        ecdf : ndarray
            ECDF values at each point
        """
        x = np.sort(data)
        n = len(x)
        ecdf = np.arange(1, n + 1) / n
        return x, ecdf
    
    def compare_distributions(
        self,
        data: np.ndarray,
        distributions: list,
        return_best: bool = True
    ) -> Dict[str, GOFResult]:
        """
        Compare multiple distributions using this GOF test.
        
        Parameters
        ----------
        data : array-like
            Data to test
        distributions : list of BaseDistribution
            Fitted distributions to compare
        return_best : bool, default=True
            If True, also return the best-fitting distribution
        
        Returns
        -------
        results : dict
            Dictionary mapping distribution names to GOFResult objects
        """
        results = {}
        
        for dist in distributions:
            result = self.test(data, dist)
            results[dist.info.name] = result
        
        if return_best:
            # Best is the one with highest p-value
            best_name = max(results.keys(), key=lambda k: results[k].p_value)
            results['_best'] = best_name
        
        return results
    
    def __repr__(self) -> str:
        return f"<{self.name} (α={self.alpha})>"
    
    def __str__(self) -> str:
        return f"{self.name}\n{self.description}"


def interpret_p_value(p_value: float, alpha: float = 0.05) -> str:
    """
    Provide interpretation of p-value.
    
    Parameters
    ----------
    p_value : float
        P-value from statistical test
    alpha : float, default=0.05
        Significance level
    
    Returns
    -------
    interpretation : str
        Plain English interpretation
    """
    if p_value < 0.001:
        strength = "extremely strong"
    elif p_value < 0.01:
        strength = "very strong"
    elif p_value < 0.05:
        strength = "strong"
    elif p_value < 0.10:
        strength = "moderate"
    else:
        strength = "weak or no"
    
    if p_value < alpha:
        return (
            f"P-value = {p_value:.4f} provides {strength} evidence "
            f"against the null hypothesis. The data does NOT fit the "
            f"specified distribution well."
        )
    else:
        return (
            f"P-value = {p_value:.4f} provides {strength} evidence "
            f"against the null hypothesis. The data fits the specified "
            f"distribution reasonably well."
        )
