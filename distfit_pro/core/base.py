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

# Import verbose logger
try:
    from ..utils.verbose import logger
    from ..core.config import config
    VERBOSE_AVAILABLE = True
except ImportError:
    VERBOSE_AVAILABLE = False


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
    wrapping scipy.stats with additional functionality and verbose explanations.
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
    # FITTING METHODS (WITH VERBOSE SUPPORT)
    # ========================================================================
    
    def fit(self, data: np.ndarray, method: str = 'mle', verbose: Optional[bool] = None, **kwargs) -> 'BaseDistribution':
        """
        Fit distribution to data with optional verbose explanations.
        
        Parameters
        ----------
        data : array-like
            Data to fit
        method : str, default='mle'
            Fitting method: 'mle' or 'mom'
        verbose : bool, optional
            Override global verbosity setting
        **kwargs : dict
            Additional parameters for fitting
            
        Returns
        -------
        self : BaseDistribution
            Fitted distribution instance
            
        Examples
        --------
        >>> dist = NormalDistribution()
        >>> dist.fit(data, method='mle', verbose=True)
        >>> print(dist.summary())
        """
        # Validate data
        data = self._validate_data(data)
        self._data = data
        
        # Determine if we should be verbose
        if VERBOSE_AVAILABLE:
            use_verbose = verbose if verbose is not None else config.is_verbose()
            
            if use_verbose:
                logger.subsection(f"Fitting {self.info.display_name}")
                logger.explain_data_characteristics(data)
                logger.explain_fitting_process(self.info.display_name, method, len(data))
        else:
            use_verbose = False
        
        # Perform fitting
        try:
            if method.lower() == 'mle':
                self._fit_mle(data, **kwargs)
                if use_verbose:
                    logger.success(f"MLE fitting completed successfully")
            elif method.lower() == 'mom':
                self._fit_mom(data, **kwargs)
                if use_verbose:
                    logger.success(f"Method of Moments fitting completed successfully")
            else:
                raise ValueError(f"Unknown fitting method: {method}")
            
            self._fitted = True
            
            # Verbose parameter explanation
            if use_verbose:
                self._explain_fitted_parameters()
                self._explain_statistics()
            
            return self
            
        except Exception as e:
            if VERBOSE_AVAILABLE and not config.is_silent():
                logger.error(f"Fitting failed: {str(e)}")
            raise
    
    @abstractmethod
    def _fit_mle(self, data: np.ndarray, **kwargs):
        """Maximum Likelihood Estimation (to be implemented by subclasses)"""
        pass
    
    @abstractmethod
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """Method of Moments (to be implemented by subclasses)"""
        pass
    
    def _explain_fitted_parameters(self):
        """Explain fitted parameters in plain language (verbose mode)."""
        if not VERBOSE_AVAILABLE or not config.is_verbose():
            return
        
        logger.subsection("Fitted Parameters")
        
        for param_name, param_value in self._params.items():
            # Get parameter meaning
            meaning = self._get_parameter_meaning(param_name)
            impact = self._get_parameter_impact(param_name, param_value)
            
            logger.explain_parameter(param_name, param_value, meaning, impact)
    
    def _get_parameter_meaning(self, param_name: str) -> str:
        """Get meaning of parameter (can be overridden by subclasses)."""
        meanings = {
            'loc': 'Location parameter (center of distribution)',
            'scale': 'Scale parameter (spread of distribution)',
            'shape': 'Shape parameter (controls distribution shape)',
            'mu': 'Mean parameter',
            'sigma': 'Standard deviation parameter',
            'alpha': 'First shape parameter',
            'beta': 'Second shape parameter',
            'df': 'Degrees of freedom',
            'p': 'Probability parameter',
            'n': 'Number of trials',
        }
        return meanings.get(param_name, f'{param_name} parameter')
    
    def _get_parameter_impact(self, param_name: str, value: float) -> str:
        """Get practical interpretation of parameter value."""
        if param_name in ['scale', 'sigma']:
            if value < 1:
                return "Low variability - data points clustered tightly"
            elif value > 5:
                return "High variability - data points spread widely"
            else:
                return "Moderate variability"
        
        if param_name == 'shape':
            if value < 1:
                return "Distribution has decreasing hazard rate"
            elif value > 1:
                return "Distribution has increasing hazard rate"
            else:
                return "Constant hazard rate (exponential)"
        
        return "See documentation for interpretation"
    
    def _explain_statistics(self):
        """Explain key statistics (verbose mode)."""
        if not VERBOSE_AVAILABLE or not config.is_verbose():
            return
        
        logger.subsection("Distribution Statistics")
        
        try:
            mean_val = self.mean()
            logger.explain_statistic("Mean", mean_val, f"Expected value: {mean_val:.4f}")
            
            std_val = self.std()
            logger.explain_statistic("Std Dev", std_val, f"Typical deviation from mean: {std_val:.4f}")
            
            median_val = self.median()
            logger.explain_statistic("Median", median_val, "50% of data below this value")
            
            try:
                skew_val = self.skewness()
                if abs(skew_val) < 0.5:
                    skew_interp = "Approximately symmetric"
                elif skew_val > 0:
                    skew_interp = "Right-skewed (long tail to the right)"
                else:
                    skew_interp = "Left-skewed (long tail to the left)"
                logger.explain_statistic("Skewness", skew_val, skew_interp)
            except:
                pass
            
            try:
                kurt_val = self.kurtosis()
                if abs(kurt_val) < 0.5:
                    kurt_interp = "Normal tail behavior"
                elif kurt_val > 0:
                    kurt_interp = "Heavy tails (more extreme values)"
                else:
                    kurt_interp = "Light tails (fewer extreme values)"
                logger.explain_statistic("Kurtosis", kurt_val, kurt_interp)
            except:
                pass
                
        except Exception as e:
            if VERBOSE_AVAILABLE and config.is_debug():
                logger.debug(f"Could not calculate statistics: {e}")
    
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
        q = np.asarray(x)
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
