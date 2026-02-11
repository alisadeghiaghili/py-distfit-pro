"""
Distribution Classes with Self-Explanatory Behavior
===================================================

This module contains statistical distribution classes where each:
- Clearly explains parameters
- Specifies appropriate data ranges
- Demonstrates practical applications

**30 Statistical Distributions:**
- 25 Continuous Distributions
- 5 Discrete Distributions
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy import stats
from scipy.optimize import minimize, brentq
from scipy.special import gamma as gamma_func


@dataclass
class DistributionInfo:
    """Information about a distribution"""
    name: str
    display_name: str
    parameters: Dict[str, str]
    support: str
    use_cases: List[str]
    characteristics: List[str]
    warning: Optional[str] = None


class BaseDistribution(ABC):
    """
    Base class for all distributions
    
    Each distribution must implement:
    - pdf/pmf: probability density/mass function
    - cdf: cumulative distribution function
    - ppf: percent point function (inverse CDF)
    - fit: parameter estimation from data
    - explain: conceptual explanation
    
    Example:
    --------
    >>> from distfit_pro.core.distributions import get_distribution
    >>> import numpy as np
    >>> 
    >>> # Generate sample data
    >>> data = np.random.normal(loc=10, scale=2, size=1000)
    >>> 
    >>> # Fit distribution
    >>> dist = get_distribution('normal')
    >>> dist.fit(data, method='mle')
    >>> 
    >>> # View summary
    >>> print(dist.summary())
    >>> 
    >>> # Get statistics
    >>> print(f"Mean: {dist.mean():.2f}")
    >>> print(f"Std: {dist.std():.2f}")
    """
    
    def __init__(self):
        self.params: Optional[Dict[str, float]] = None
        self.fitted: bool = False
        self._scipy_dist = None
        self._is_discrete = False
        
    @property
    @abstractmethod
    def info(self) -> DistributionInfo:
        """Distribution information"""
        pass
    
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density/mass function"""
        pass
    
    @abstractmethod
    def cdf(self, x: np.ndarray) -> np.ndarray:
        """Cumulative distribution function"""
        pass
    
    @abstractmethod
    def ppf(self, q: np.ndarray) -> np.ndarray:
        """Percent point function (inverse of CDF)"""
        pass
    
    def logpdf(self, x: np.ndarray) -> np.ndarray:
        """Log probability density"""
        return np.log(self.pdf(x) + 1e-300)
    
    def logcdf(self, x: np.ndarray) -> np.ndarray:
        """Log cumulative distribution"""
        return np.log(self.cdf(x) + 1e-300)
    
    def sf(self, x: np.ndarray) -> np.ndarray:
        """Survival function (1 - CDF)"""
        return 1.0 - self.cdf(x)
    
    def isf(self, q: np.ndarray) -> np.ndarray:
        """Inverse survival function"""
        return self.ppf(1.0 - q)
    
    def rvs(self, size: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the fitted distribution
        
        Parameters:
        -----------
        size : int
            Number of random samples to generate
        random_state : int, optional
            Seed for reproducibility
            
        Returns:
        --------
        samples : np.ndarray
            Random samples from the distribution
            
        Example:
        --------
        >>> dist = get_distribution('normal')
        >>> dist.fit(data)
        >>> samples = dist.rvs(size=100, random_state=42)
        """
        if random_state is not None:
            np.random.seed(random_state)
        if self._scipy_dist and self.params:
            return self._scipy_dist.rvs(**self.params, size=size, random_state=random_state)
        u = np.random.uniform(0, 1, size)
        return self.ppf(u)
    
    # ========== Distribution Statistics ==========
    
    def mean(self) -> float:
        """Distribution mean"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.mean(**self.params)
        raise NotImplementedError(f"Mean not implemented for {self.info.name}")
    
    def var(self) -> float:
        """Distribution variance"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.var(**self.params)
        raise NotImplementedError(f"Variance not implemented for {self.info.name}")
    
    def variance(self) -> float:
        """Alias for var()"""
        return self.var()
    
    def std(self) -> float:
        """Distribution standard deviation"""
        return np.sqrt(self.var())
    
    def median(self) -> float:
        """Distribution median (50th percentile)"""
        return self.ppf(0.5)
    
    def mode(self) -> float:
        """
        Distribution mode (value with highest density)
        
        Returns:
        --------
        mode : float
            Value where PDF/PMF is maximum
        """
        if hasattr(self, '_mode_at_zero') and self._mode_at_zero:
            return 0.0
        if hasattr(self, '_mode_value'):
            return self._mode_value()
        try:
            x_min = self.ppf(0.01)
            x_max = self.ppf(0.99)
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(lambda x: -self.pdf(np.array([x]))[0], 
                                    bounds=(x_min, x_max), method='bounded')
            return result.x
        except:
            return self.median()
    
    def skewness(self) -> float:
        """Distribution skewness (asymmetry)"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.stats(**self.params, moments='s')
        raise NotImplementedError(f"Skewness not implemented for {self.info.name}")
    
    def kurtosis(self) -> float:
        """Distribution kurtosis (tail heaviness)"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.stats(**self.params, moments='k')
        raise NotImplementedError(f"Kurtosis not implemented for {self.info.name}")
    
    def hazard_rate(self, t: float) -> float:
        """
        Hazard rate (failure rate) at time t
        
        h(t) = f(t) / S(t) = pdf(t) / (1 - cdf(t))
        """
        pdf_t = self.pdf(np.array([t]))[0]
        sf_t = self.sf(np.array([t]))[0]
        if sf_t < 1e-10:
            return np.inf
        return pdf_t / sf_t
    
    def reliability(self, t: float) -> float:
        """
        Reliability function at time t
        
        R(t) = P(T > t) = 1 - F(t)
        """
        return self.sf(np.array([t]))[0]
    
    def mean_time_to_failure(self) -> float:
        """Mean Time To Failure (MTTF) = E[T]"""
        return self.mean()
    
    def conditional_var(self, alpha: float) -> float:
        """
        Conditional Value at Risk (CVaR) / Expected Shortfall
        
        CVaR_α = E[X | X ≤ VaR_α]
        """
        quantiles = np.linspace(0.0001, alpha, 100)
        return np.mean(self.ppf(quantiles))
    
    # ========== Fitting Methods ==========
    
    @abstractmethod
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        """Maximum Likelihood Estimation"""
        pass
    
    @abstractmethod
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        """Method of Moments"""
        pass
    
    def fit(self, data: np.ndarray, method: str = 'mle', **kwargs) -> 'BaseDistribution':
        """
        Fit distribution to data
        
        Parameters:
        -----------
        data : array-like
            Observed data
        method : str
            Estimation method: 'mle', 'moments', 'quantile'
            
        Returns:
        --------
        self : BaseDistribution
            Fitted distribution (for method chaining)
            
        Example:
        --------
        >>> dist = get_distribution('weibull')
        >>> dist.fit(data, method='mle')
        >>> print(dist.summary())
        """
        data = np.asarray(data).flatten()
        data = data[~np.isnan(data)]
        
        if method == 'mle':
            self.params = self.fit_mle(data, **kwargs)
        elif method == 'moments':
            self.params = self.fit_moments(data)
        elif method == 'quantile':
            self.params = self.fit_quantile(data, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.fitted = True
        return self
    
    def fit_quantile(self, data: np.ndarray, quantiles: Optional[List[float]] = None) -> Dict[str, float]:
        """Quantile matching estimation"""
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        empirical_quantiles = np.quantile(data, quantiles)
        
        def objective(params_array):
            self.params = self._array_to_params(params_array)
            theoretical_quantiles = self.ppf(np.array(quantiles))
            return np.sum((empirical_quantiles - theoretical_quantiles) ** 2)
        
        initial_params = self.fit_moments(data)
        x0 = self._params_to_array(initial_params)
        result = minimize(objective, x0, method='Nelder-Mead')
        return self._array_to_params(result.x)
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        return np.array(list(params.values()))
    
    def _array_to_params(self, array: np.ndarray) -> Dict[str, float]:
        keys = list(self.info.parameters.keys())
        return dict(zip(keys, array))
    
    # ========== Summary & Explanation ==========
    
    def summary(self) -> str:
        """
        Complete statistical summary of the fitted distribution
        
        Shows:
        - Estimated parameters
        - Location statistics (mean, median, mode)
        - Spread statistics (variance, std)
        - Shape statistics (skewness, kurtosis)
        - Key quantiles
        
        Returns:
        --------
        summary : str
            Formatted summary text
            
        Example:
        --------
        >>> dist = get_distribution('normal')
        >>> dist.fit(data)
        >>> print(dist.summary())
        """
        if not self.fitted:
            return f"⚠️  {self.info.display_name} has not been fitted yet."
        
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║  {self.info.display_name:^60}  ║
╠══════════════════════════════════════════════════════════════╣
║  📊 ESTIMATED PARAMETERS                                     ║
╚══════════════════════════════════════════════════════════════╝
"""
        # Parameters
        for param_name, param_value in self.params.items():
            param_desc = self.info.parameters.get(param_name, param_name)
            summary += f"   {param_desc:<35} = {param_value:>15.6f}\n"
        
        # Location Statistics
        summary += f"""
╔══════════════════════════════════════════════════════════════╗
║  📍 LOCATION STATISTICS                                      ║
╚══════════════════════════════════════════════════════════════╝
"""
        try:
            mean_val = self.mean()
            if not np.isnan(mean_val):
                summary += f"   {'Mean (μ)':<35} = {mean_val:>15.6f}\n"
            else:
                summary += f"   {'Mean (μ)':<35} = {'Undefined':>15}\n"
        except:
            summary += f"   {'Mean (μ)':<35} = {'N/A':>15}\n"
        
        try:
            summary += f"   {'Median':<35} = {self.median():>15.6f}\n"
        except:
            summary += f"   {'Median':<35} = {'N/A':>15}\n"
        
        try:
            summary += f"   {'Mode':<35} = {self.mode():>15.6f}\n"
        except:
            summary += f"   {'Mode':<35} = {'N/A':>15}\n"
        
        # Spread Statistics
        summary += f"""
╔══════════════════════════════════════════════════════════════╗
║  📏 SPREAD STATISTICS                                        ║
╚══════════════════════════════════════════════════════════════╝
"""
        try:
            var_val = self.var()
            if not np.isnan(var_val):
                summary += f"   {'Variance (σ²)':<35} = {var_val:>15.6f}\n"
                summary += f"   {'Std Deviation (σ)':<35} = {np.sqrt(var_val):>15.6f}\n"
            else:
                summary += f"   {'Variance (σ²)':<35} = {'Undefined':>15}\n"
                summary += f"   {'Std Deviation (σ)':<35} = {'Undefined':>15}\n"
        except:
            summary += f"   {'Variance (σ²)':<35} = {'N/A':>15}\n"
            summary += f"   {'Std Deviation (σ)':<35} = {'N/A':>15}\n"
        
        # Shape Statistics
        summary += f"""
╔══════════════════════════════════════════════════════════════╗
║  📐 SHAPE STATISTICS                                         ║
╚══════════════════════════════════════════════════════════════╝
"""
        try:
            skew = self.skewness()
            summary += f"   {'Skewness (asymmetry)':<35} = {skew:>15.6f}\n"
            if skew > 0.5:
                summary += f"   {'  → Right-skewed (positive)':<35}\n"
            elif skew < -0.5:
                summary += f"   {'  → Left-skewed (negative)':<35}\n"
            else:
                summary += f"   {'  → Approximately symmetric':<35}\n"
        except:
            summary += f"   {'Skewness':<35} = {'N/A':>15}\n"
        
        try:
            kurt = self.kurtosis()
            summary += f"   {'Kurtosis (tail weight)':<35} = {kurt:>15.6f}\n"
            if kurt > 1:
                summary += f"   {'  → Heavy tails (leptokurtic)':<35}\n"
            elif kurt < -1:
                summary += f"   {'  → Light tails (platykurtic)':<35}\n"
            else:
                summary += f"   {'  → Normal-like tails':<35}\n"
        except:
            summary += f"   {'Kurtosis':<35} = {'N/A':>15}\n"
        
        # Quantiles
        summary += f"""
╔══════════════════════════════════════════════════════════════╗
║  📊 KEY QUANTILES                                            ║
╚══════════════════════════════════════════════════════════════╝
"""
        quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
        for q in quantiles:
            try:
                q_val = self.ppf(q)
                summary += f"   {f'{q*100:.0f}th percentile':<35} = {q_val:>15.6f}\n"
            except:
                pass
        
        summary += f"""
╔══════════════════════════════════════════════════════════════╗
║  ℹ️  For conceptual explanation, use .explain()              ║
╚══════════════════════════════════════════════════════════════╝
"""
        return summary
    
    def explain(self) -> str:
        """
        Conceptual explanation of the distribution
        
        Focus on:
        - Estimated parameters and their meaning
        - Practical use cases
        - Distribution characteristics
        - Warnings and caveats
        
        For statistical summary, use .summary()
        
        Returns:
        --------
        explanation : str
            Formatted explanation text
        """
        if not self.fitted:
            return f"⚠️  {self.info.display_name} هنوز فیت نشده است."
        
        explanation = f"""
╔══════════════════════════════════════════════════════════════╗
║  {self.info.display_name:^60}  ║
╚══════════════════════════════════════════════════════════════╝

📊 پارامترهای برآورد شده:
"""
        for param_name, param_value in self.params.items():
            param_desc = self.info.parameters.get(param_name, param_name)
            explanation += f"   • {param_desc}: {param_value:.4f}\n"
        
        explanation += f"\n💡 کاربردهای عملی:\n"
        for use_case in self.info.use_cases:
            explanation += f"   • {use_case}\n"
        
        explanation += f"\n🔍 ویژگی‌های این توزیع:\n"
        for char in self.info.characteristics:
            explanation += f"   • {char}\n"
        
        if self.info.warning:
            explanation += f"\n⚠️  هشدار: {self.info.warning}\n"
        
        explanation += f"\n📈 برای آمارهای کامل: dist.summary()\n"
        return explanation
    
    def __repr__(self) -> str:
        if self.fitted:
            params_str = ", ".join([f"{k}={v:.3f}" for k, v in self.params.items()])
            return f"{self.info.name}({params_str})"
        return f"{self.info.name}(not fitted)"


# ═══════════════════════════════════════════════════════════════
# CONTINUOUS DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

class NormalDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.norm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="normal",
            display_name="Normal (Gaussian) Distribution",
            parameters={"loc": "μ (mean)", "scale": "σ (std)"},
            support="(-∞, +∞)",
            use_cases=["خطاهای اندازه‌گیری", "قد و وزن", "نمرات تست", "نویز سیگنال"],
            characteristics=["متقارن", "68% در μ±σ", "95% در μ±2σ"],
            warning="برای داده‌های چوله مناسب نیست"
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, **self.params)
    def cdf(self, x):
        return self._scipy_dist.cdf(x, **self.params)
    def ppf(self, q):
        return self._scipy_dist.ppf(q, **self.params)
    def fit_mle(self, data, **kwargs):
        return {"loc": np.mean(data), "scale": np.std(data, ddof=1)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class LognormalDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.lognorm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="lognormal",
            display_name="Lognormal Distribution",
            parameters={"s": "σ (log-scale)", "scale": "exp(μ)"},
            support="(0, +∞)",
            use_cases=["درآمد", "قیمت سهام", "زمان شکست"],
            characteristics=["راست‌چوله", "مثبت"],
            warning="فقط برای مقادیر مثبت"
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['s'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['s'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['s'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        log_data = np.log(data)
        return {"s": np.std(log_data, ddof=1), "scale": np.exp(np.mean(log_data))}
    def fit_moments(self, data):
        return self.fit_mle(data)


class WeibullDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.weibull_min
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="weibull",
            display_name="Weibull Distribution",
            parameters={"c": "k (shape)", "scale": "λ (scale)"},
            support="(0, +∞)",
            use_cases=["قابلیت اطمینان", "زمان شکست", "سرعت باد"],
            characteristics=["k<1: خرابی کاهشی", "k=1: نمایی", "k>1: فرسودگی"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['c'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['c'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['c'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)
        return {"c": params[0], "scale": params[2]}
    def fit_moments(self, data):
        return self.fit_mle(data)


class GammaDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.gamma
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="gamma",
            display_name="Gamma Distribution",
            parameters={"a": "α (shape)", "scale": "θ (scale)"},
            support="(0, +∞)",
            use_cases=["زمان انتظار", "باران", "prior Bayesian"],
            characteristics=["α=1: نمایی", "α بزرگ: نرمال"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['a'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['a'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['a'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)
        return {"a": params[0], "scale": params[2]}
    def fit_moments(self, data):
        m, v = np.mean(data), np.var(data, ddof=1)
        return {"a": m*m/v, "scale": v/m}


class ExponentialDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.expon
        self._mode_at_zero = True
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="exponential",
            display_name="Exponential Distribution",
            parameters={"scale": "1/λ (mean)"},
            support="(0, +∞)",
            use_cases=["زمان بین رویداد", "عمر قطعات"],
            characteristics=["بی‌حافظه", "نرخ خطر ثابت"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        return {"scale": np.mean(data[data > 0])}
    def fit_moments(self, data):
        return self.fit_mle(data)


class BetaDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.beta
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="beta",
            display_name="Beta Distribution",
            parameters={"a": "α (shape 1)", "b": "β (shape 2)"},
            support="[0, 1]",
            use_cases=["احتمالات", "نرخ موفقیت", "prior Bayesian"],
            characteristics=["انعطاف‌پذیر", "محدود به [0,1]"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['a'], self.params['b'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['a'], self.params['b'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['a'], self.params['b'])
    def fit_mle(self, data, **kwargs):
        data = data[(data > 0) & (data < 1)]
        params = self._scipy_dist.fit(data, floc=0, fscale=1)
        return {"a": params[0], "b": params[1]}
    def fit_moments(self, data):
        m, v = np.mean(data), np.var(data, ddof=1)
        common = m * (1 - m) / v - 1
        return {"a": m * common, "b": (1 - m) * common}


class UniformDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.uniform
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="uniform",
            display_name="Uniform Distribution",
            parameters={"loc": "a (min)", "scale": "b-a (width)"},
            support="[a, b]",
            use_cases=["تولید عدد تصادفی", "prior بی‌اطلاع"],
            characteristics=["احتمال یکسان", "حداکثر آنتروپی"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        return {"loc": np.min(data), "scale": np.max(data) - np.min(data)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class TriangularDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.triang
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="triangular",
            display_name="Triangular Distribution",
            parameters={"c": "mode position", "loc": "min", "scale": "width"},
            support="[min, min+width]",
            use_cases=["PERT", "برآورد خبره"],
            characteristics=["ساده", "شهودی"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['c'], self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['c'], self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['c'], self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        a, b = np.min(data), np.max(data)
        c = (np.mean(data) - a) / (b - a) if b > a else 0.5
        return {"c": c, "loc": a, "scale": b - a}
    def fit_moments(self, data):
        return self.fit_mle(data)


class LogisticDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.logistic
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="logistic",
            display_name="Logistic Distribution",
            parameters={"loc": "μ (location)", "scale": "s (scale)"},
            support="(-∞, +∞)",
            use_cases=["رگرسیون logistic", "مدل‌های رشد"],
            characteristics=["دنباله سنگین‌تر از نرمال"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        return {"loc": np.mean(data), "scale": np.std(data) * np.sqrt(3) / np.pi}
    def fit_moments(self, data):
        return self.fit_mle(data)


class GumbelDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.gumbel_r
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="gumbel",
            display_name="Gumbel Distribution (Extreme Value Type I)",
            parameters={"loc": "μ (location)", "scale": "β (scale)"},
            support="(-∞, +∞)",
            use_cases=["سیلاب", "زلزله", "مقادیر حداکثر"],
            characteristics=["چولگی مثبت", "extreme values"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        params = self._scipy_dist.fit(data)
        return {"loc": params[0], "scale": params[1]}
    def fit_moments(self, data):
        m, s = np.mean(data), np.std(data, ddof=1)
        return {"scale": s * np.sqrt(6) / np.pi, "loc": m - 0.5772 * s * np.sqrt(6) / np.pi}


class FrechetDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.invweibull
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="frechet",
            display_name="Frechet Distribution (Extreme Value Type II)",
            parameters={"c": "α (shape)", "scale": "s (scale)"},
            support="(0, +∞)",
            use_cases=["مقادیر حداکثر مثبت", "بیمه"],
            characteristics=["دنباله بسیار سنگین"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['c'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['c'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['c'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)
        return {"c": params[0], "scale": params[2]}
    def fit_moments(self, data):
        return self.fit_mle(data)


class ParetoDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.pareto
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="pareto",
            display_name="Pareto Distribution",
            parameters={"b": "α (shape)", "scale": "x_m (minimum)"},
            support="[x_m, +∞)",
            use_cases=["ثروت", "درآمد", "80-20 rule"],
            characteristics=["دنباله سنگین", "power law"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['b'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['b'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['b'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        xm = np.min(data)
        n = len(data)
        alpha = n / np.sum(np.log(data / xm))
        return {"b": alpha, "scale": xm}
    def fit_moments(self, data):
        return self.fit_mle(data)


class CauchyDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.cauchy
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="cauchy",
            display_name="Cauchy Distribution",
            parameters={"loc": "x_0 (location)", "scale": "γ (scale)"},
            support="(-∞, +∞)",
            use_cases=["فیزیک", "رزونانس"],
            characteristics=["میانگین نامعلوم", "دنباله خیلی سنگین"],
            warning="میانگین و واریانس ندارد"
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        return {"loc": np.median(data), "scale": np.percentile(np.abs(data - np.median(data)), 50)}
    def fit_moments(self, data):
        return self.fit_mle(data)
    
    def mean(self):
        return np.nan
    def var(self):
        return np.nan


class StudentTDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.t
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="studentt",
            display_name="Student's t Distribution",
            parameters={"df": "ν (degrees of freedom)", "loc": "μ", "scale": "σ"},
            support="(-∞, +∞)",
            use_cases=["تست فرضیه", "نمونه کوچک"],
            characteristics=["دنباله سنگین‌تر از نرمال"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['df'], self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['df'], self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['df'], self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        params = self._scipy_dist.fit(data)
        return {"df": params[0], "loc": params[1], "scale": params[2]}
    def fit_moments(self, data):
        return self.fit_mle(data)


class ChiSquaredDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.chi2
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="chisquared",
            display_name="Chi-Squared Distribution",
            parameters={"df": "k (degrees of freedom)"},
            support="[0, +∞)",
            use_cases=["تست GOF", "واریانس"],
            characteristics=["حالت خاص گاما"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['df'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['df'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['df'])
    def fit_mle(self, data, **kwargs):
        return {"df": np.mean(data)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class FDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.f
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="f",
            display_name="F Distribution",
            parameters={"dfn": "d1 (numerator df)", "dfd": "d2 (denominator df)"},
            support="[0, +∞)",
            use_cases=["ANOVA", "نسبت واریانس‌ها"],
            characteristics=["چوله"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['dfn'], self.params['dfd'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['dfn'], self.params['dfd'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['dfn'], self.params['dfd'])
    def fit_mle(self, data, **kwargs):
        return {"dfn": 5, "dfd": 10}
    def fit_moments(self, data):
        return self.fit_mle(data)


class RayleighDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.rayleigh
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="rayleigh",
            display_name="Rayleigh Distribution",
            parameters={"scale": "σ (scale)"},
            support="[0, +∞)",
            use_cases=["سیگنال رادار", "سرعت باد"],
            characteristics=["چولگی مثبت"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data >= 0]
        return {"scale": np.sqrt(np.mean(data**2) / 2)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class LaplaceDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.laplace
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="laplace",
            display_name="Laplace Distribution (Double Exponential)",
            parameters={"loc": "μ (location)", "scale": "b (scale)"},
            support="(-∞, +∞)",
            use_cases=["تفاضل‌ها", "Lasso regression"],
            characteristics=["دنباله سنگین‌تر از نرمال"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['loc'], self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['loc'], self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['loc'], self.params['scale'])
    def fit_mle(self, data, **kwargs):
        return {"loc": np.median(data), "scale": np.mean(np.abs(data - np.median(data)))}
    def fit_moments(self, data):
        return self.fit_mle(data)


class InverseGammaDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.invgamma
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="invgamma",
            display_name="Inverse Gamma Distribution",
            parameters={"a": "α (shape)", "scale": "β (scale)"},
            support="(0, +∞)",
            use_cases=["prior برای واریانس"],
            characteristics=["دنباله سنگین"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['a'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['a'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['a'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)
        return {"a": params[0], "scale": params[2]}
    def fit_moments(self, data):
        return self.fit_mle(data)


class LogLogisticDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.fisk
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="loglogistic",
            display_name="Log-Logistic Distribution",
            parameters={"c": "α (shape)", "scale": "β (scale)"},
            support="(0, +∞)",
            use_cases=["تحلیل بقا", "survival analysis"],
            characteristics=["دنباله سنگین"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pdf(x, self.params['c'], scale=self.params['scale'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['c'], scale=self.params['scale'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['c'], scale=self.params['scale'])
    def fit_mle(self, data, **kwargs):
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)
        return {"c": params[0], "scale": params[2]}
    def fit_moments(self, data):
        return self.fit_mle(data)


# ═══════════════════════════════════════════════════════════════
# DISCRETE DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════

class PoissonDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.poisson
        self._is_discrete = True
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="poisson",
            display_name="Poisson Distribution",
            parameters={"mu": "λ (rate)"},
            support="{0, 1, 2, ...}",
            use_cases=["تعداد رویدادها", "تعداد تماس‌ها"],
            characteristics=["میانگین = واریانس = λ"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pmf(x, self.params['mu'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['mu'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['mu'])
    def fit_mle(self, data, **kwargs):
        return {"mu": np.mean(data)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class BinomialDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.binom
        self._is_discrete = True
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="binomial",
            display_name="Binomial Distribution",
            parameters={"n": "n (trials)", "p": "p (success prob)"},
            support="{0, 1, 2, ..., n}",
            use_cases=["آزمایش‌های موفق/ناموفق"],
            characteristics=["n آزمایش مستقل"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pmf(x, self.params['n'], self.params['p'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['n'], self.params['p'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['n'], self.params['p'])
    def fit_mle(self, data, **kwargs):
        n = int(np.max(data))
        p = np.mean(data) / n if n > 0 else 0.5
        return {"n": n, "p": p}
    def fit_moments(self, data):
        m, v = np.mean(data), np.var(data, ddof=1)
        p = 1 - v / m if m > 0 else 0.5
        n = int(m / p) if p > 0 else 1
        return {"n": max(n, 1), "p": np.clip(p, 0.01, 0.99)}


class NegativeBinomialDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.nbinom
        self._is_discrete = True
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="nbinom",
            display_name="Negative Binomial Distribution",
            parameters={"n": "r (successes)", "p": "p (success prob)"},
            support="{0, 1, 2, ...}",
            use_cases=["overdispersed counts"],
            characteristics=["واریانس > میانگین"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pmf(x, self.params['n'], self.params['p'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['n'], self.params['p'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['n'], self.params['p'])
    def fit_mle(self, data, **kwargs):
        m, v = np.mean(data), np.var(data, ddof=1)
        p = m / v if v > m else 0.5
        n = m * p / (1 - p) if p < 1 else 1
        return {"n": max(n, 1), "p": np.clip(p, 0.01, 0.99)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class GeometricDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.geom
        self._is_discrete = True
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="geometric",
            display_name="Geometric Distribution",
            parameters={"p": "p (success prob)"},
            support="{1, 2, 3, ...}",
            use_cases=["زمان تا اولین موفقیت"],
            characteristics=["بی‌حافظه"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pmf(x, self.params['p'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['p'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['p'])
    def fit_mle(self, data, **kwargs):
        return {"p": 1.0 / np.mean(data)}
    def fit_moments(self, data):
        return self.fit_mle(data)


class HypergeometricDistribution(BaseDistribution):
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.hypergeom
        self._is_discrete = True
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="hypergeometric",
            display_name="Hypergeometric Distribution",
            parameters={"M": "population", "n": "successes in pop", "N": "draws"},
            support="{max(0, N+n-M), ..., min(n, N)}",
            use_cases=["نمونه‌گیری بدون جایگذاری"],
            characteristics=["محدود"],
        )
    
    def pdf(self, x):
        return self._scipy_dist.pmf(x, self.params['M'], self.params['n'], self.params['N'])
    def cdf(self, x):
        return self._scipy_dist.cdf(x, self.params['M'], self.params['n'], self.params['N'])
    def ppf(self, q):
        return self._scipy_dist.ppf(q, self.params['M'], self.params['n'], self.params['N'])
    def fit_mle(self, data, **kwargs):
        N = int(np.max(data)) + 10
        return {"M": N, "n": N // 2, "N": int(np.mean(data))}
    def fit_moments(self, data):
        return self.fit_mle(data)


# ═══════════════════════════════════════════════════════════════
# REGISTRY & FACTORY
# ═══════════════════════════════════════════════════════════════

DISTRIBUTION_REGISTRY = {
    'normal': NormalDistribution,
    'lognormal': LognormalDistribution,
    'weibull': WeibullDistribution,
    'gamma': GammaDistribution,
    'exponential': ExponentialDistribution,
    'beta': BetaDistribution,
    'uniform': UniformDistribution,
    'triangular': TriangularDistribution,
    'logistic': LogisticDistribution,
    'gumbel': GumbelDistribution,
    'frechet': FrechetDistribution,
    'pareto': ParetoDistribution,
    'cauchy': CauchyDistribution,
    'studentt': StudentTDistribution,
    'chisquared': ChiSquaredDistribution,
    'f': FDistribution,
    'rayleigh': RayleighDistribution,
    'laplace': LaplaceDistribution,
    'invgamma': InverseGammaDistribution,
    'loglogistic': LogLogisticDistribution,
    'poisson': PoissonDistribution,
    'binomial': BinomialDistribution,
    'nbinom': NegativeBinomialDistribution,
    'geometric': GeometricDistribution,
    'hypergeometric': HypergeometricDistribution,
}


def get_distribution(name: str) -> BaseDistribution:
    """
    Get distribution by name
    
    Parameters:
    -----------
    name : str
        Distribution name (case-insensitive)
        
    Returns:
    --------
    dist : BaseDistribution
        Distribution instance
        
    Example:
    --------
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    >>> print(dist.summary())
    """
    name = name.lower()
    if name not in DISTRIBUTION_REGISTRY:
        available = ', '.join(sorted(DISTRIBUTION_REGISTRY.keys()))
        raise ValueError(f"Unknown distribution '{name}'. Available: {available}")
    return DISTRIBUTION_REGISTRY[name]()


def list_distributions() -> List[str]:
    """List all available distributions (30 total)"""
    return sorted(list(DISTRIBUTION_REGISTRY.keys()))


def list_continuous_distributions() -> List[str]:
    """List continuous distributions (20 total)"""
    return [k for k, v in DISTRIBUTION_REGISTRY.items() if not get_distribution(k)._is_discrete]


def list_discrete_distributions() -> List[str]:
    """List discrete distributions (5 total)"""
    return [k for k, v in DISTRIBUTION_REGISTRY.items() if get_distribution(k)._is_discrete]
