"""
Base Classes for Distribution Framework
========================================

Abstract base classes and utilities for all distributions.

Author: Ali Sadeghi Aghili
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Union, Callable
import numpy as np
from scipy import stats
import warnings

try:
    from ..utils.verbose import logger
    from ..core.config import config
    from ..locales import t
    VERBOSE_AVAILABLE = True
except ImportError:
    VERBOSE_AVAILABLE = False


class FittingMethod(Enum):
    """Supported parameter estimation methods"""
    MLE = "mle"
    MOM = "mom"
    MOMENTS = "moments"


@dataclass
class DistributionInfo:
    """Metadata describing a probability distribution."""
    name: str
    scipy_name: str
    display_name: str
    description: str
    parameters: List[str]
    support: str
    is_discrete: bool = False
    has_shape_params: bool = False
    
    def __str__(self) -> str:
        return f"{self.display_name}: {self.description}"


class BaseDistribution(ABC):
    """Base class for all probability distributions."""
    
    # Parameter name aliases (for backward compatibility with scipy/tests)
    PARAM_ALIASES = {
        'a': 'alpha',  # Beta, Gamma
        'b': 'beta',   # Beta
        'c': 'shape',  # Weibull, Burr
        's': 'shape',  # Lognormal
        'mu': 'mean',  # Poisson, Wald
    }
    
    def __init__(self):
        self._scipy_dist: Optional[stats.rv_continuous] = None
        self._params: Dict[str, float] = {}
        self._fitted: bool = False
        self._is_discrete: bool = False
        self._data: Optional[np.ndarray] = None
        
    @property
    @abstractmethod
    def info(self) -> DistributionInfo:
        """Distribution metadata"""
        pass
    
    @property
    def fitted(self) -> bool:
        """True if fit() has been called"""
        return self._fitted
    
    @fitted.setter
    def fitted(self, value: bool):
        """Allow manual setting for testing"""
        self._fitted = value
    
    @property
    def params(self) -> Optional[Dict[str, float]]:
        """Fitted parameter values with alias support"""
        if not self._fitted:
            return None
        
        # Return params with both real names and aliases
        result = self._params.copy()
        
        # Add aliases
        for alias, real_name in self.PARAM_ALIASES.items():
            if real_name in result:
                result[alias] = result[real_name]
        
        return result
    
    @params.setter
    def params(self, value: Dict[str, float]):
        """Set parameters manually"""
        self._params = value
        self._fitted = True
    
    def get_param(self, name: str) -> Optional[float]:
        """Get parameter by name (supports aliases)"""
        if not self._fitted:
            return None
        
        # Check real name first
        if name in self._params:
            return self._params[name]
        
        # Check alias
        real_name = self.PARAM_ALIASES.get(name)
        if real_name and real_name in self._params:
            return self._params[real_name]
        
        return None
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Original data used for fitting"""
        return self._data
    
    def params_to_array(self) -> np.ndarray:
        """Convert parameters to array (for weighted fitting)"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return np.array([self._params[k] for k in sorted(self._params.keys())])
    
    def array_to_params(self, arr: np.ndarray) -> Dict[str, float]:
        """Convert array back to parameters dict"""
        keys = sorted(self._params.keys())
        return {k: v for k, v in zip(keys, arr)}
    
    # ===== FITTING =====
    
    def fit(self, data: np.ndarray, method: str = 'mle', verbose: Optional[bool] = None, **kwargs) -> 'BaseDistribution':
        """Fit distribution to data"""
        method = method.lower()
        if method == 'moments':
            method = 'mom'
        
        handle_nan = kwargs.pop('handle_nan', None)
        if handle_nan == 'remove':
            data = data[~np.isnan(data)]
        
        data = self._validate_data(data, strict=handle_nan != 'remove')
        self._data = data
        
        if VERBOSE_AVAILABLE:
            use_verbose = verbose if verbose is not None else config.is_verbose()
        else:
            use_verbose = False
        
        try:
            if method == 'mle':
                self._fit_mle(data, **kwargs)
            elif method == 'mom':
                self._fit_mom(data, **kwargs)
            else:
                raise ValueError(f"Unknown method: {method}. Use 'mle' or 'mom'.")
            
            self._fitted = True
            return self
            
        except Exception as e:
            if VERBOSE_AVAILABLE and not config.is_silent():
                logger.error(f"Fitting failed: {str(e)}")
            raise
    
    @abstractmethod
    def _fit_mle(self, data: np.ndarray, **kwargs):
        pass
    
    @abstractmethod
    def _fit_mom(self, data: np.ndarray, **kwargs):
        pass
    
    # ===== PROBABILITY FUNCTIONS =====
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """PDF or PMF"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        if self._is_discrete:
            return self._scipy_dist.pmf(x, **self._get_scipy_params())
        return self._scipy_dist.pdf(x, **self._get_scipy_params())
    
    def pmf(self, k: np.ndarray) -> np.ndarray:
        """PMF for discrete distributions"""
        return self.pdf(k)
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log PDF"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        if self._is_discrete:
            return self._scipy_dist.logpmf(x, **self._get_scipy_params())
        return self._scipy_dist.logpdf(x, **self._get_scipy_params())
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """CDF"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        return self._scipy_dist.cdf(x, **self._get_scipy_params())
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Inverse CDF"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        q = np.asarray(q)
        return self._scipy_dist.ppf(q, **self._get_scipy_params())
    
    def sf(self, x: np.ndarray) -> np.ndarray:
        """Survival function (1 - CDF)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        return self._scipy_dist.sf(x, **self._get_scipy_params())
    
    def reliability(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Reliability function R(t) = P(X > t) = 1 - F(t)"""
        return self.sf(t)
    
    def hazard_rate(self, t: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Hazard rate function h(t) = f(t) / R(t)"""
        t = np.asarray(t)
        pdf_val = self.pdf(t)
        sf_val = self.sf(t)
        # Avoid division by zero
        return np.where(sf_val > 0, pdf_val / sf_val, np.inf)
    
    # ===== MOMENTS =====
    
    def mean(self) -> float:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return float(self._scipy_dist.mean(**self._get_scipy_params()))
    
    def var(self) -> float:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return float(self._scipy_dist.var(**self._get_scipy_params()))
    
    def std(self) -> float:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return float(self._scipy_dist.std(**self._get_scipy_params()))
    
    def median(self) -> float:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return float(self._scipy_dist.median(**self._get_scipy_params()))
    
    def skewness(self) -> float:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return float(self._scipy_dist.stats(**self._get_scipy_params(), moments='s'))
    
    def kurtosis(self) -> float:
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return float(self._scipy_dist.stats(**self._get_scipy_params(), moments='k'))
    
    # ===== SAMPLING =====
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """Generate random samples"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.rvs(
            **self._get_scipy_params(),
            size=size,
            random_state=random_state
        )
    
    # ===== MODEL SELECTION =====
    
    def log_likelihood(self, data: Optional[np.ndarray] = None) -> float:
        """Log-likelihood"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        
        if data is None:
            if self._data is None:
                raise ValueError("No data available")
            data = self._data
        
        return float(np.sum(self.logpdf(data)))
    
    def aic(self, data: Optional[np.ndarray] = None) -> float:
        """AIC"""
        k = len(self._params)
        ll = self.log_likelihood(data)
        return 2 * k - 2 * ll
    
    def bic(self, data: Optional[np.ndarray] = None) -> float:
        """BIC"""
        if data is None:
            data = self._data
        k = len(self._params)
        n = len(data)
        ll = self.log_likelihood(data)
        return k * np.log(n) - 2 * ll
    
    # ===== OUTPUT =====
    
    def summary(self) -> str:
        """Text summary"""
        if not self._fitted:
            return f"{self.info.display_name} (not fitted)"
        
        lines = [
            f"{self.info.display_name}",
            f"Parameters: {self._params}",
            f"Mean: {self.mean():.4f}",
            f"Std: {self.std():.4f}",
        ]
        return " | ".join(lines)
    
    def explain(self) -> str:
        """Explanation (just description, no name)"""
        return self.info.description
    
    def _validate_data(self, data: Union[List, np.ndarray], strict: bool = True) -> np.ndarray:
        """Validate data"""
        data = np.asarray(data, dtype=float)
        
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        
        if strict:
            if np.any(np.isnan(data)):
                raise ValueError("Data contains NaN values")
            if np.any(np.isinf(data)):
                raise ValueError("Data contains infinite values")
        
        if data.size < 5:
            warnings.warn("Very small sample size (< 5)")
        
        return data
    
    @abstractmethod
    def _get_scipy_params(self) -> Dict[str, float]:
        pass
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"<{self.info.display_name} ({status})>"


class ContinuousDistribution(BaseDistribution):
    """Continuous distributions"""
    pass


class DiscreteDistribution(BaseDistribution):
    """Discrete distributions"""
    
    def __init__(self):
        super().__init__()
        self._is_discrete = True
