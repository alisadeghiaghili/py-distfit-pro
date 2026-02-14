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

# Import verbose logger if available
try:
    from ..utils.verbose import logger
    from ..core.config import config
    from ..locales import t
    VERBOSE_AVAILABLE = True
except ImportError:
    VERBOSE_AVAILABLE = False


class FittingMethod(Enum):
    """Supported parameter estimation methods"""
    MLE = "mle"  # Maximum Likelihood
    MOM = "mom"  # Method of Moments


@dataclass
class DistributionInfo:
    """
    Metadata describing a probability distribution.
    
    This holds all the static information about a distribution
    that doesn't change when we fit it to data.
    """
    name: str  # Short name like 'normal'
    scipy_name: str  # Name scipy.stats uses
    display_name: str  # Pretty name for output
    description: str  # What it's used for
    parameters: List[str]  # Parameter names
    support: str  # Valid range: "(0, inf)", "(-inf, inf)", etc
    is_discrete: bool = False
    has_shape_params: bool = False
    
    def __str__(self) -> str:
        return f"{self.display_name}: {self.description}"


class BaseDistribution(ABC):
    """
    Base class for all probability distributions.
    
    Provides:
    - Unified fitting interface (MLE, moments)
    - Scipy wrapper for pdf/cdf/ppf/etc
    - Verbose mode with explanations
    - AIC/BIC calculation
    - Random sampling
    
    Subclasses implement:
    - _fit_mle(): MLE estimation logic
    - _fit_mom(): Method of moments
    - _get_scipy_params(): Convert params to scipy format
    - info property: Distribution metadata
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
        """Distribution metadata (name, description, params, etc)"""
        pass
    
    @property
    def fitted(self) -> bool:
        """True if fit() has been called"""
        return self._fitted
    
    @property
    def params(self) -> Dict[str, float]:
        """Fitted parameter values {param_name: value}"""
        if not self._fitted:
            raise ValueError("Distribution not fitted yet")
        return self._params.copy()
    
    @params.setter
    def params(self, value: Dict[str, float]):
        """Set parameters manually (e.g., from saved model)"""
        self._params = value
        self._fitted = True
    
    @property
    def data(self) -> Optional[np.ndarray]:
        """Original data used for fitting (if available)"""
        return self._data
    
    # ========================================================================
    # FITTING
    # ========================================================================
    
    def fit(self, data: np.ndarray, method: str = 'mle', verbose: Optional[bool] = None, **kwargs) -> 'BaseDistribution':
        """
        Fit distribution parameters to observed data.
        
        Parameters
        ----------
        data : array-like
            Observed data points. Must be 1D numeric.
            NaN/Inf values will raise error.
        method : {'mle', 'mom'}, default='mle'
            Parameter estimation method:
            - 'mle': Maximum likelihood (most accurate, slower)
            - 'mom': Method of moments (fast, less accurate)
        verbose : bool, optional
            Override global verbosity. If None, uses config.verbosity.
        **kwargs : dict
            Additional fitting options passed to estimation method.
            
        Returns
        -------
        self : BaseDistribution
            Returns self for method chaining.
            
        Raises
        ------
        ValueError
            If data is empty, contains NaN/Inf, or is invalid for this distribution.
        RuntimeError
            If optimization fails to converge.
            
        Examples
        --------
        >>> import numpy as np
        >>> from distfit_pro import get_distribution
        >>> 
        >>> # Generate normal data
        >>> data = np.random.normal(10, 2, 1000)
        >>> 
        >>> # Fit with MLE (default)
        >>> dist = get_distribution('normal')
        >>> dist.fit(data)
        >>> print(dist.params)
        {'loc': 10.05, 'scale': 1.98}
        >>> 
        >>> # Fit with method of moments
        >>> dist.fit(data, method='mom')
        >>> 
        >>> # Chain methods
        >>> dist.fit(data).summary()
        
        Notes
        -----
        MLE is generally more accurate but can fail on:
        - Small samples (< 30 points)
        - Extreme outliers
        - Data near distribution bounds
        
        Method of moments is more robust but less efficient.
        """
        # Clean and validate data
        data = self._validate_data(data)
        self._data = data
        
        # Check if we should show verbose output
        if VERBOSE_AVAILABLE:
            use_verbose = verbose if verbose is not None else config.is_verbose()
            
            if use_verbose:
                # Show what we're doing
                logger.subsection(t('section_fitting') + f" {self.info.display_name}")
                logger.explain_data_characteristics(data)
                logger.explain_fitting_process(self.info.display_name, method, len(data))
        else:
            use_verbose = False
        
        # Do the actual fitting
        try:
            if method.lower() == 'mle':
                self._fit_mle(data, **kwargs)
                if use_verbose:
                    logger.success(f"MLE {t('fitting_completed_successfully')}")
            elif method.lower() == 'mom':
                self._fit_mom(data, **kwargs)
                if use_verbose:
                    logger.success(f"MoM {t('fitting_completed_successfully')}")
            else:
                raise ValueError(f"Unknown method: {method}. Use 'mle' or 'mom'.")
            
            self._fitted = True
            
            # Explain what we found (verbose mode)
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
        """MLE implementation - must override in subclass"""
        pass
    
    @abstractmethod
    def _fit_mom(self, data: np.ndarray, **kwargs):
        """Method of moments implementation - must override in subclass"""
        pass
    
    def _explain_fitted_parameters(self):
        """
        Explain fitted parameters in plain language (verbose only).
        
        Uses translations to explain what each parameter means
        and its practical impact on the distribution.
        """
        if not VERBOSE_AVAILABLE or not config.is_verbose():
            return
        
        logger.subsection(t('section_fitted_parameters'))
        
        for param_name, param_value in self._params.items():
            # Get translated meaning and impact
            meaning = self._get_parameter_meaning(param_name)
            impact = self._get_parameter_impact(param_name, param_value)
            
            logger.explain_parameter(param_name, param_value, meaning, impact)
    
    def _get_parameter_meaning(self, param_name: str) -> str:
        """
        Get translated explanation of what a parameter means.
        
        Subclasses can override for distribution-specific meanings.
        """
        # Try to get translation first
        key = f'param_{param_name}_meaning'
        meaning = t(key)
        
        # If translation not found (key returned as-is), use default
        if meaning == key:
            defaults = {
                'loc': 'Location parameter (center of distribution)',
                'scale': 'Scale parameter (spread of distribution)',
                'shape': 'Shape parameter (affects distribution shape)',
                'mu': 'Mean parameter',
                'sigma': 'Standard deviation parameter',
                'alpha': 'First shape parameter',
                'beta': 'Second shape parameter',
                'df': 'Degrees of freedom',
                'p': 'Probability parameter',
                'n': 'Number of trials',
            }
            meaning = defaults.get(param_name, f'{param_name} parameter')
        
        return meaning
    
    def _get_parameter_impact(self, param_name: str, value: float) -> str:
        """
        Explain practical impact of parameter value.
        
        Interprets the parameter value in plain language
        (e.g., "high variability" vs "low variability").
        """
        # Check for scale/spread parameters
        if param_name in ['scale', 'sigma']:
            if value < 1:
                return t('impact_low_variability')
            elif value > 5:
                return t('impact_high_variability')
            else:
                return t('impact_moderate_variability')
        
        # Check for shape parameters (Weibull, Gamma, etc)
        if param_name in ['shape', 'k', 'c']:
            if value < 1:
                return "Decreasing hazard rate"
            elif value > 1:
                return "Increasing hazard rate (wear-out)"
            else:
                return "Constant hazard (exponential)"
        
        return t('impact_see_docs')
    
    def _explain_statistics(self):
        """
        Show key statistics with interpretations (verbose mode).
        
        Calculates mean, std, median, skewness, kurtosis
        and explains what they mean in plain language.
        """
        if not VERBOSE_AVAILABLE or not config.is_verbose():
            return
        
        logger.subsection(t('section_distribution_statistics'))
        
        try:
            # Mean
            mean_val = self.mean()
            logger.explain_statistic(
                t('mean'), mean_val,
                f"→ {t('stat_expected_value')}: {mean_val:.4f}"
            )
            
            # Standard deviation
            std_val = self.std()
            logger.explain_statistic(
                t('std_dev'), std_val,
                f"→ {t('stat_typical_deviation')}: {std_val:.4f}"
            )
            
            # Median
            median_val = self.median()
            logger.explain_statistic(
                t('median'), median_val,
                f"→ {t('stat_50_percent_below')}"
            )
            
            # Skewness (if available)
            try:
                skew_val = self.skewness()
                if abs(skew_val) < 0.5:
                    skew_interp = t('stat_approximately_symmetric')
                elif skew_val > 0:
                    skew_interp = t('data_right_skewed')
                else:
                    skew_interp = t('data_left_skewed')
                logger.explain_statistic(t('skewness'), skew_val, f"→ {skew_interp}")
            except:
                pass
            
            # Kurtosis (if available)
            try:
                kurt_val = self.kurtosis()
                if abs(kurt_val) < 0.5:
                    kurt_interp = t('stat_normal_tail_behavior')
                elif kurt_val > 0:
                    kurt_interp = t('heavy_tails')
                else:
                    kurt_interp = t('light_tails')
                logger.explain_statistic(t('kurtosis'), kurt_val, f"→ {kurt_interp}")
            except:
                pass
                
        except Exception as e:
            # Only show error in debug mode
            if VERBOSE_AVAILABLE and config.is_debug():
                logger.debug(f"Could not calculate some statistics: {e}")
    
    # ========================================================================
    # PROBABILITY FUNCTIONS
    # ========================================================================
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """
        Probability density function (PDF) or probability mass function (PMF).
        
        For continuous distributions: probability density at x.
        For discrete distributions: probability mass at x.
        
        Parameters
        ----------
        x : array-like
            Points to evaluate
            
        Returns
        -------
        pdf : ndarray
            Density/mass values
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        return self._scipy_dist.pdf(x, **self._get_scipy_params())
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log of PDF (more numerically stable for likelihood calculations)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        return self._scipy_dist.logpdf(x, **self._get_scipy_params())
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Cumulative distribution function.
        
        Returns P(X <= x) for each x.
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        return self._scipy_dist.cdf(x, **self._get_scipy_params())
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """
        Percent point function (inverse CDF / quantile function).
        
        Returns x such that P(X <= x) = q.
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        q = np.asarray(q)
        return self._scipy_dist.ppf(q, **self._get_scipy_params())
    
    def sf(self, x: np.ndarray) -> np.ndarray:
        """
        Survival function: P(X > x) = 1 - CDF(x).
        
        Useful for reliability analysis.
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        x = np.asarray(x)
        return self._scipy_dist.sf(x, **self._get_scipy_params())
    
    def isf(self, q: np.ndarray) -> np.ndarray:
        """Inverse survival function: x such that P(X > x) = q"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        q = np.asarray(q)
        return self._scipy_dist.isf(q, **self._get_scipy_params())
    
    # ========================================================================
    # MOMENTS AND SUMMARY STATS
    # ========================================================================
    
    def mean(self) -> float:
        """Expected value E[X]"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.mean(**self._get_scipy_params())
    
    def var(self) -> float:
        """Variance Var(X)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.var(**self._get_scipy_params())
    
    def std(self) -> float:
        """Standard deviation sqrt(Var(X))"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.std(**self._get_scipy_params())
    
    def median(self) -> float:
        """Median (50th percentile)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.median(**self._get_scipy_params())
    
    def mode(self) -> float:
        """
        Mode (most probable value).
        
        Not all distributions have a mode.
        Override in subclass if analytical form exists.
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        raise NotImplementedError(f"Mode not available for {self.info.name}")
    
    def skewness(self) -> float:
        """Skewness (3rd standardized moment)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.stats(**self._get_scipy_params(), moments='s')
    
    def kurtosis(self) -> float:
        """Excess kurtosis (4th standardized moment - 3)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.stats(**self._get_scipy_params(), moments='k')
    
    def entropy(self) -> float:
        """Differential entropy (nats)"""
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.entropy(**self._get_scipy_params())
    
    # ========================================================================
    # RANDOM SAMPLING
    # ========================================================================
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from fitted distribution.
        
        Parameters
        ----------
        size : int, default=1
            Number of samples
        random_state : int, optional
            Seed for reproducibility
            
        Returns
        -------
        samples : ndarray
            Random samples
            
        Examples
        --------
        >>> dist.fit(data)
        >>> samples = dist.rvs(1000, random_state=42)
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        return self._scipy_dist.rvs(
            **self._get_scipy_params(),
            size=size,
            random_state=random_state
        )
    
    # ========================================================================
    # MODEL SELECTION CRITERIA
    # ========================================================================
    
    def log_likelihood(self, data: Optional[np.ndarray] = None) -> float:
        """
        Log-likelihood of data given fitted parameters.
        
        Parameters
        ----------
        data : array-like, optional
            Data to evaluate. If None, uses original fitting data.
            
        Returns
        -------
        ll : float
            Sum of log(pdf(x)) over all data points
        """
        if not self._fitted:
            raise ValueError("Must call fit() first")
        
        if data is None:
            if self._data is None:
                raise ValueError("No data available. Pass data explicitly.")
            data = self._data
        else:
            data = np.asarray(data)
        
        return np.sum(self.logpdf(data))
    
    def aic(self, data: Optional[np.ndarray] = None) -> float:
        """
        Akaike Information Criterion.
        
        AIC = 2k - 2*log(L)
        where k = number of parameters, L = likelihood
        
        Lower is better.
        """
        k = len(self._params)
        ll = self.log_likelihood(data)
        return 2 * k - 2 * ll
    
    def bic(self, data: Optional[np.ndarray] = None) -> float:
        """
        Bayesian Information Criterion.
        
        BIC = k*log(n) - 2*log(L)
        where k = num params, n = sample size, L = likelihood
        
        Lower is better. Penalizes complexity more than AIC.
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
    
    # ========================================================================
    # OUTPUT
    # ========================================================================
    
    def summary(self) -> str:
        """
        Generate beautifully formatted summary with i18n support.
        
        Returns
        -------
        summary : str
            Multi-line summary with box characters and translations
        """
        if not self._fitted:
            # Simple unfitted message (no extra text)
            return f"{self.info.display_name} (not fitted)"
        
        # Import translation function
        if VERBOSE_AVAILABLE:
            def t_local(key):
                return t(key)
        else:
            # Fallback if imports fail
            def t_local(key):
                fallback = {
                    'estimated_parameters': 'ESTIMATED PARAMETERS',
                    'location_statistics': 'LOCATION STATISTICS',
                    'spread_statistics': 'SPREAD STATISTICS',
                    'shape_statistics': 'SHAPE STATISTICS',
                    'goodness_of_fit': 'GOODNESS OF FIT',
                    'log_likelihood': 'Log-Likelihood',
                    'aic': 'AIC',
                    'bic': 'BIC',
                    'mean': 'Mean',
                    'median': 'Median',
                    'mode': 'Mode',
                    'variance': 'Variance',
                    'std_deviation': 'Std Deviation',
                    'skewness': 'Skewness',
                    'kurtosis': 'Kurtosis',
                }
                return fallback.get(key, key)
        
        lines = []
        
        # Header box
        title = self.info.display_name
        lines.append("╔" + "═" * 62 + "╗")
        lines.append(f"║ {title:^60} ║")
        lines.append("╠" + "═" * 62 + "╣")
        param_header = f"║  📊 {t_local('estimated_parameters')}"
        lines.append(param_header + " " * (64 - len(param_header)) + "║")
        lines.append("╚" + "═" * 62 + "╝")
        
        # Parameters with display names
        for param, value in self._params.items():
            # Get translated display name
            if param == 'loc':
                if self.info.name == 'normal':
                    mean_trans = t_local('mean')
                    param_display = f'μ ({mean_trans})' if mean_trans != 'Mean' else 'μ (mean)'
                else:
                    param_display = 'location'
            elif param == 'scale':
                if self.info.name == 'normal':
                    std_trans = t_local('std_deviation')
                    # Extract just first word for brevity
                    std_short = std_trans.split()[0] if ' ' in std_trans else std_trans
                    param_display = f'σ ({std_short})' if std_trans != 'Std Deviation' else 'σ (std)'
                else:
                    param_display = 'scale'
            elif param == 'alpha':
                param_display = 'α (shape)'
            elif param == 'beta':
                param_display = 'β (scale/rate)'
            elif param == 'c':
                param_display = 'c (shape)'
            elif param == 's':
                param_display = 's (shape)'
            elif param == 'df':
                param_display = 'df (degrees of freedom)'
            else:
                param_display = param
            
            lines.append(f"   {param_display:<30} = {value:>15.6f}")
        
        # Location Statistics box
        lines.append("")
        lines.append("╔" + "═" * 62 + "╗")
        loc_header = f"║  📍 {t_local('location_statistics')}"
        lines.append(loc_header + " " * (64 - len(loc_header)) + "║")
        lines.append("╚" + "═" * 62 + "╝")
        
        try:
            mean_val = self.mean()
            lines.append(f"   {t_local('mean'):<30} = {mean_val:>15.6f}")
        except:
            pass
        
        try:
            median_val = self.median()
            lines.append(f"   {t_local('median'):<30} = {median_val:>15.6f}")
        except:
            pass
        
        try:
            mode_val = self.mode()
            lines.append(f"   {t_local('mode'):<30} = {mode_val:>15.6f}")
        except:
            pass
        
        # Spread Statistics box
        lines.append("")
        lines.append("╔" + "═" * 62 + "╗")
        spread_header = f"║  📏 {t_local('spread_statistics')}"
        lines.append(spread_header + " " * (64 - len(spread_header)) + "║")
        lines.append("╚" + "═" * 62 + "╝")
        
        try:
            var_val = self.var()
            lines.append(f"   {t_local('variance'):<30} = {var_val:>15.6f}")
        except:
            pass
        
        try:
            std_val = self.std()
            lines.append(f"   {t_local('std_deviation'):<30} = {std_val:>15.6f}")
        except:
            pass
        
        # Shape Statistics box
        try:
            skew_val = self.skewness()
            kurt_val = self.kurtosis()
            
            lines.append("")
            lines.append("╔" + "═" * 62 + "╗")
            shape_header = f"║  📐 {t_local('shape_statistics')}"
            lines.append(shape_header + " " * (64 - len(shape_header)) + "║")
            lines.append("╚" + "═" * 62 + "╝")
            lines.append(f"   {t_local('skewness'):<30} = {skew_val:>15.6f}")
            
            kurt_label = t_local('kurtosis')
            if kurt_label == 'Kurtosis':
                kurt_label = 'Kurtosis (excess)'
            lines.append(f"   {kurt_label:<30} = {kurt_val:>15.6f}")
        except:
            pass
        
        # Goodness of Fit box (FULLY i18n translated)
        if self._data is not None:
            lines.append("")
            lines.append("╔" + "═" * 62 + "╗")
            gof_header = f"║  ✅ {t_local('goodness_of_fit')}"
            lines.append(gof_header + " " * (64 - len(gof_header)) + "║")
            lines.append("╚" + "═" * 62 + "╝")
            
            try:
                ll = self.log_likelihood()
                lines.append(f"   {t_local('log_likelihood'):<30} = {ll:>15.2f}")
            except:
                pass
            
            try:
                aic_val = self.aic()
                lines.append(f"   {t_local('aic'):<30} = {aic_val:>15.2f}")
            except:
                pass
            
            try:
                bic_val = self.bic()
                lines.append(f"   {t_local('bic'):<30} = {bic_val:>15.2f}")
            except:
                pass
        
        return "\n".join(lines)
    
    def explain(self) -> str:
        """
        Educational explanation of the distribution.
        
        Returns brief description of what this distribution
        represents and when to use it.
        """
        # Include distribution name for better context
        return f"{self.info.display_name}: {self.info.description}"
    
    # ========================================================================
    # INTERNAL
    # ========================================================================
    
    def _validate_data(self, data: Union[List, np.ndarray]) -> np.ndarray:
        """
        Check data is valid before fitting.
        
        Converts to numpy array and checks for:
        - Empty data
        - NaN values
        - Infinite values
        """
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
        Convert our parameter format to scipy's format.
        
        Must be implemented by subclasses because scipy uses
        different naming (e.g., 's' vs 'shape', 'loc' vs 'mu').
        
        Returns
        -------
        scipy_params : dict
            Parameters in scipy format
        """
        pass
    
    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"<{self.info.display_name} distribution ({status})>"
    
    def __str__(self) -> str:
        return self.summary()


class ContinuousDistribution(BaseDistribution):
    """Base for continuous distributions (Normal, Exponential, etc)"""
    
    def __init__(self):
        super().__init__()
        self._is_discrete = False


class DiscreteDistribution(BaseDistribution):
    """Base for discrete distributions (Poisson, Binomial, etc)"""
    
    def __init__(self):
        super().__init__()
        self._is_discrete = True
    
    def pmf(self, k: np.ndarray) -> np.ndarray:
        """Probability mass function (alias for pdf for discrete)"""
        return self.pdf(k)
    
    def logpmf(self, k: np.ndarray) -> np.ndarray:
        """Log of PMF"""
        return self.logpdf(k)
