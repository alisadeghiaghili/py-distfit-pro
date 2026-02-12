"""
Base Classes for Distribution Framework
========================================

Abstract base classes and common utilities for all distributions.

Author: Ali Sadeghi Aghili
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
import numpy as np
from scipy import stats


class FittingMethod(Enum):
    """Enumeration of supported fitting methods"""
    MLE = "mle"
    MOM = "mom"  # Method of Moments


@dataclass
class DistributionInfo:
    """Metadata about a distribution"""
    name: str
    scipy_name: str  # Name in scipy.stats
    display_name: str
    description: str
    parameters: List[str]
    support: str  # e.g., "(0, inf)", "(-inf, inf)", "[0, 1]"
    is_discrete: bool = False
    has_shape_params: bool = False
    
    def __str__(self) -> str:
        return f"{self.display_name}: {self.description}"


class BaseDistribution(ABC):
    """
    Abstract base class for all probability distributions.
    
    This class provides a unified interface for working with distributions,
    wrapping scipy.stats with additional functionality.
    """
    
    def __init__(self):
        self._scipy_dist: Optional[stats.rv_continuous] = None
        self._params: Dict[str, float] = {}
        self._fitted: bool = False
        self._is_discrete: bool = False
        self._data: Optional[np.ndarray] = None
        
    @property
    @abstractmethod
    def info(self) -> DistributionInfo:
        """Return distribution metadata"""
        pass
    
    @property
    def fitted(self) -> bool:
        """Check if distribution has been fitted"""
        return self._fitted
    
    @property
    def params(self) -> Dict[str, float]:
        """Get fitted parameters"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._params.copy()
    
    @params.setter
    def params(self, value: Dict[str, float]):
        """Set parameters directly"""
        self._params = value
        self._fitted = True
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Get original data used for fitting"""
        return self._data
    
    # ========================================================================
    # FITTING METHODS
    # ========================================================================
    
    def fit(self, data: np.ndarray, method: str = 'mle', **kwargs) -> 'BaseDistribution':
        """
        Fit distribution to data.
        
        Parameters
        ----------
        data : array-like
            Data to fit
        method : str, default='mle'
            Fitting method: 'mle' or 'mom'
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        self : BaseDistribution
            Fitted distribution instance
        """
        data = self._validate_data(data)
        self._data = data
        
        if method.lower() == 'mle':
            self._fit_mle(data, **kwargs)
        elif method.lower() == 'mom':
            self._fit_mom(data, **kwargs)
        else:
            raise ValueError(f"Unknown fitting method: {method}")
        
        self._fitted = True
        return self
    
    @abstractmethod
    def _fit_mle(self, data: np.ndarray, **kwargs):
        """Maximum Likelihood Estimation (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """Method of Moments (to be implemented by subclasses)"""
        pass
    
    # ========================================================================
    # PROBABILITY FUNCTIONS
    # ========================================================================
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function (or PMF for discrete).
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate PDF
            
        Returns
        -------
        pdf : ndarray
            PDF values
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        x = np.asarray(x)
        return self._scipy_dist.pdf(x, **self._get_scipy_params())
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of probability density function"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        x = np.asarray(x)
        return self._scipy_dist.logpdf(x, **self._get_scipy_params())
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function.
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate CDF
            
        Returns
        -------
        cdf : ndarray
            CDF values
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        x = np.asarray(x)
        return self._scipy_dist.cdf(x, **self._get_scipy_params())
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent point function (inverse of CDF).
        
        Parameters
        ----------
        q : array-like
            Probabilities
            
        Returns
        -------
        ppf : ndarray
            Quantiles
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        q = np.asarray(q)
        return self._scipy_dist.ppf(q, **self._get_scipy_params())
    
    def sf(self, x: np.ndarray) -> np.ndarray:
        """
        Survival function (1 - CDF).
        
        Parameters
        ----------
        x : array-like
            Points at which to evaluate survival function
            
        Returns
        -------
        sf : ndarray
            Survival function values
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        x = np.asarray(x)
        return self._scipy_dist.sf(x, **self._get_scipy_params())
    
    def isf(self, q: np.ndarray) -> np.ndarray:
        """Inverse survival function"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        q = np.asarray(q)
        return self._scipy_dist.isf(q, **self._get_scipy_params())
    
    # ========================================================================
    # MOMENTS AND STATISTICS
    # ========================================================================
    
    def mean(self) -> float:
        """Expected value"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.mean(**self._get_scipy_params())
    
    def var(self) -> float:
        """Variance"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.var(**self._get_scipy_params())
    
    def std(self) -> float:
        """Standard deviation"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.std(**self._get_scipy_params())
    
    def median(self) -> float:
        """Median (50th percentile)"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.median(**self._get_scipy_params())
    
    def mode(self) -> float:
        """Mode (most likely value)"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        # Most distributions don't have mode method in scipy
        # Implement in subclasses if needed
        raise NotImplementedError(f"Mode not implemented for {self.info.name}")
    
    def skewness(self) -> float:
        """Skewness"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.stats(**self._get_scipy_params(), moments='s')
    
    def kurtosis(self) -> float:
        """Excess kurtosis"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.stats(**self._get_scipy_params(), moments='k')
    
    def entropy(self) -> float:
        """Differential entropy"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.entropy(**self._get_scipy_params())
    
    # ========================================================================
    # RANDOM SAMPLING
    # ========================================================================
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples.
        
        Parameters
        ----------
        size : int, default=1
            Number of samples to generate
        random_state : int, optional
            Random seed for reproducibility
            
        Returns
        -------
        samples : ndarray
            Random samples
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._scipy_dist.rvs(
            **self._get_scipy_params(),
            size=size,
            random_state=random_state
        )
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def log_likelihood(self, data: Optional[np.ndarray] = None) -> float:
        """
        Calculate log-likelihood of data.
        
        Parameters
        ----------
        data : array-like, optional
            Data to evaluate. If None, uses fitted data.
            
        Returns
        -------
        ll : float
            Log-likelihood
        """
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        
        if data is None:
            if self._data is None:
                raise ValueError("No data available")
            data = self._data
        else:
            data = np.asarray(data)
        
        return np.sum(self.logpdf(data))
    
    def aic(self, data: Optional[np.ndarray] = None) -> float:
        """
        Akaike Information Criterion.
        
        AIC = 2k - 2ln(L)
        where k is number of parameters and L is likelihood
        """
        k = len(self._params)
        ll = self.log_likelihood(data)
        return 2 * k - 2 * ll
    
    def bic(self, data: Optional[np.ndarray] = None) -> float:
        """
        Bayesian Information Criterion.
        
        BIC = k*ln(n) - 2ln(L)
        where k is number of parameters, n is sample size, L is likelihood
        """
        if data is None:
            if self._data is None:
                raise ValueError("No data available")
            data = self._data
        else:
            data = np.asarray(data)
        
        k = len(self._params)
        n = len(data)
        ll = self.log_likelihood(data)
        return k * np.log(n) - 2 * ll
    
    def summary(self) -> str:
        """
        Generate summary string of fitted distribution.
        
        Returns
        -------
        summary : str
            Summary text
        """
        if not self._fitted:
            return f"{self.info.display_name} (not fitted)"
        
        lines = [
            f"Distribution: {self.info.display_name}",
            f"Description: {self.info.description}",
            f"Support: {self.info.support}",
            "",
            "Parameters:",
        ]
        
        for param, value in self._params.items():
            lines.append(f"  {param}: {value:.6f}")
        
        lines.extend([
            "",
            "Statistics:",
            f"  Mean: {self.mean():.6f}",
            f"  Variance: {self.var():.6f}",
            f"  Std Dev: {self.std():.6f}",
            f"  Median: {self.median():.6f}",
        ])
        
        try:
            lines.append(f"  Skewness: {self.skewness():.6f}")
            lines.append(f"  Kurtosis: {self.kurtosis():.6f}")
        except:
            pass
        
        if self._data is not None:
            lines.extend([
                "",
                "Goodness of Fit:",
                f"  Log-Likelihood: {self.log_likelihood():.2f}",
                f"  AIC: {self.aic():.2f}",
                f"  BIC: {self.bic():.2f}",
            ])
        
        return "\n".join(lines)
    
    def explain(self) -> str:
        """
        Return educational explanation of the distribution.
        
        Returns
        -------
        explanation : str
            Explanation text
        """
        return self.info.description
    
    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================
    
    def _validate_data(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """Validate and convert input data"""
        data = np.asarray(data, dtype=float)
        
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        
        if np.any(np.isinf(data)):
            raise ValueError("Data contains infinite values")
        
        return data
    
    @abstractmethod
    def _get_scipy_params(self) -> Dict[str, float]:
        """
        Convert internal parameters to scipy format.
        
        Must be implemented by subclasses to handle parameter naming
        differences between our API and scipy.
        """
        pass
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"<{self.info.display_name} distribution ({status})>"
    
    def __str__(self) -> str:
        return self.summary()


class ContinuousDistribution(BaseDistribution):
    """Base class for continuous distributions"""
    
    def __init__(self):
        super().__init__()
        self._is_discrete = False


class DiscreteDistribution(BaseDistribution):
    """Base class for discrete distributions"""
    
    def __init__(self):
        super().__init__()
        self._is_discrete = True
    
    def pmf(self, k: np.ndarray) -> np.ndarray:
        """Probability mass function (alias for pdf)"""
        return self.pdf(k)
    
    def logpmf(self, k: np.ndarray) -> np.ndarray:
        """Log of probability mass function"""
        return self.logpdf(k)
