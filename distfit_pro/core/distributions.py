"""
Distribution Classes with Self-Explanatory Behavior
===================================================

این ماژول شامل کلاس‌های توزیع‌های آماری است که هر کدام:
- پارامترها را واضح توضیح می‌دهند
- محدوده‌ی مناسب برای داده را می‌گویند
- کاربردهای عملی را نشان می‌دهند
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from scipy import stats
from scipy.optimize import minimize


@dataclass
class DistributionInfo:
    """اطلاعات توضیحی هر توزیع"""
    name: str
    display_name: str
    parameters: Dict[str, str]
    support: str
    use_cases: List[str]
    characteristics: List[str]
    warning: Optional[str] = None


class BaseDistribution(ABC):
    """
    کلاس پایه برای همه‌ی توزیع‌ها
    
    هر توزیع باید:
    - pdf/pmf: چگالی احتمال
    - cdf: تابع توزیع تجمعی
    - ppf: تابع معکوس CDF (کوانتایل)
    - fit: برآورد پارامترها از داده
    - explain: توضیح نتایج
    را پیاده کند
    """
    
    def __init__(self):
        self.params: Optional[Dict[str, float]] = None
        self.fitted: bool = False
        self._scipy_dist = None
        
    @property
    @abstractmethod
    def info(self) -> DistributionInfo:
        """اطلاعات توضیحی توزیع"""
        pass
    
    @abstractmethod
    def pdf(self, x: np.ndarray) -> np.ndarray:
        """Probability density function"""
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
        """Generate random samples"""
        if random_state is not None:
            np.random.seed(random_state)
        u = np.random.uniform(0, 1, size)
        return self.ppf(u)
    
    def mean(self) -> float:
        """Distribution mean"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.mean(**self.params)
        raise NotImplementedError
    
    def var(self) -> float:
        """Distribution variance"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.var(**self.params)
        raise NotImplementedError
    
    def std(self) -> float:
        """Distribution standard deviation"""
        return np.sqrt(self.var())
    
    def skewness(self) -> float:
        """Distribution skewness"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.stats(**self.params, moments='s')
        raise NotImplementedError
    
    def kurtosis(self) -> float:
        """Distribution kurtosis (excess)"""
        if self._scipy_dist and self.params:
            return self._scipy_dist.stats(**self.params, moments='k')
        raise NotImplementedError
    
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
        فیت توزیع به داده
        
        Parameters:
        -----------
        data : array-like
            داده‌ی مشاهده‌شده
        method : str
            روش تخمین: 'mle', 'moments', 'quantile'
        
        Returns:
        --------
        self : fitted distribution
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
        """
        Quantile matching estimation
        
        برآورد با تطبیق کوانتایل‌ها
        """
        if quantiles is None:
            quantiles = [0.25, 0.5, 0.75]
        
        empirical_quantiles = np.quantile(data, quantiles)
        
        def objective(params_array):
            self.params = self._array_to_params(params_array)
            theoretical_quantiles = self.ppf(np.array(quantiles))
            return np.sum((empirical_quantiles - theoretical_quantiles) ** 2)
        
        # Initial guess from moments
        initial_params = self.fit_moments(data)
        x0 = self._params_to_array(initial_params)
        
        result = minimize(objective, x0, method='Nelder-Mead')
        return self._array_to_params(result.x)
    
    def _params_to_array(self, params: Dict[str, float]) -> np.ndarray:
        """Convert params dict to array"""
        return np.array(list(params.values()))
    
    def _array_to_params(self, array: np.ndarray) -> Dict[str, float]:
        """Convert array to params dict"""
        keys = list(self.info.parameters.keys())
        return dict(zip(keys, array))
    
    def explain(self) -> str:
        """
        توضیح کامل درباره‌ی توزیع و پارامترهای برآورد شده
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
        
        explanation += f"\n📈 ویژگی‌های توزیع:\n"
        try:
            explanation += f"   • میانگین: {self.mean():.4f}\n"
            explanation += f"   • انحراف معیار: {self.std():.4f}\n"
            explanation += f"   • چولگی: {self.skewness():.4f}\n"
            explanation += f"   • کشیدگی: {self.kurtosis():.4f}\n"
        except:
            pass
        
        explanation += f"\n💡 کاربردهای عملی:\n"
        for use_case in self.info.use_cases:
            explanation += f"   • {use_case}\n"
        
        explanation += f"\n🔍 ویژگی‌های این توزیع:\n"
        for char in self.info.characteristics:
            explanation += f"   • {char}\n"
        
        if self.info.warning:
            explanation += f"\n⚠️  هشدار: {self.info.warning}\n"
        
        return explanation
    
    def __repr__(self) -> str:
        if self.fitted:
            params_str = ", ".join([f"{k}={v:.3f}" for k, v in self.params.items()])
            return f"{self.info.name}({params_str})"
        return f"{self.info.name}(not fitted)"


# ═══════════════════════════════════════════════════════════════
# توزیع‌های پیوسته
# ═══════════════════════════════════════════════════════════════

class NormalDistribution(BaseDistribution):
    """
    توزیع نرمال (گوسی)
    
    توزیع پایه‌ای که برای متغیرهایی که حاصل جمع تعداد زیادی 
    اثر مستقل و کوچک هستند مناسب است (قضیه حد مرکزی).
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.norm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="normal",
            display_name="Normal (Gaussian) Distribution",
            parameters={
                "loc": "μ (mean/location)",
                "scale": "σ (standard deviation/scale)"
            },
            support="(-∞, +∞)",
            use_cases=[
                "خطاهای اندازه‌گیری",
                "قد و وزن افراد",
                "نمرات تست‌های استاندارد",
                "نویز در سیگنال‌ها",
                "بازده‌های مالی (تقریب)"
            ],
            characteristics=[
                "متقارن حول میانگین",
                "به شکل زنگ",
                "68% داده در μ±σ",
                "95% داده در μ±2σ",
                "99.7% داده در μ±3σ"
            ],
            warning="برای داده‌های چوله یا دارای دنباله سنگین مناسب نیست"
        )
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.pdf(x, **self.params)
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.cdf(x, **self.params)
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self._scipy_dist.ppf(q, **self.params)
    
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        """MLE for normal is simply sample mean and std"""
        return {
            "loc": np.mean(data),
            "scale": np.std(data, ddof=1)
        }
    
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        """Same as MLE for normal"""
        return self.fit_mle(data)


class LognormalDistribution(BaseDistribution):
    """
    توزیع لوگ‌نرمال
    
    برای متغیرهایی که لگاریتم آن‌ها نرمال است.
    مناسب برای داده‌های مثبت و راست‌چوله.
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.lognorm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="lognormal",
            display_name="Lognormal Distribution",
            parameters={
                "s": "σ (shape - log-scale)",
                "scale": "exp(μ) (scale - log-location)"
            },
            support="(0, +∞)",
            use_cases=[
                "درآمد و ثروت افراد",
                "اندازه‌ی ذرات و سلول‌ها",
                "زمان رخداد شکست (reliability)",
                "قیمت سهام و دارایی‌ها",
                "طول عمر باتری و قطعات الکترونیکی"
            ],
            characteristics=[
                "فقط برای مقادیر مثبت",
                "راست‌چوله (دنباله سمت راست بلند)",
                "حاصل‌ضرب متغیرهای تصادفی مثبت",
                "میانه < میانگین (به دلیل چولگی)"
            ],
            warning="برای داده‌های با مقادیر منفی یا صفر قابل استفاده نیست"
        )
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.pdf(x, self.params['s'], scale=self.params['scale'])
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.cdf(x, self.params['s'], scale=self.params['scale'])
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self._scipy_dist.ppf(q, self.params['s'], scale=self.params['scale'])
    
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        data = data[data > 0]  # فقط مقادیر مثبت
        log_data = np.log(data)
        mu = np.mean(log_data)
        sigma = np.std(log_data, ddof=1)
        return {"s": sigma, "scale": np.exp(mu)}
    
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        return self.fit_mle(data)  # برای لوگ‌نرمال یکسان است


class WeibullDistribution(BaseDistribution):
    """
    توزیع وایبل
    
    بسیار در تحلیل قابلیت اطمینان و زمان شکست استفاده می‌شود.
    انعطاف بالا برای مدل‌سازی نرخ خرابی متغیر.
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.weibull_min
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="weibull",
            display_name="Weibull Distribution",
            parameters={
                "c": "k (shape - تعیین‌کننده‌ی شکل نرخ خرابی)",
                "scale": "λ (scale - مشخص‌کننده‌ی مقیاس)"
            },
            support="(0, +∞)",
            use_cases=[
                "تحلیل قابلیت اطمینان (reliability)",
                "زمان تا شکست (failure time)",
                "تحلیل عمر (lifetime analysis)",
                "سرعت باد (meteorology)",
                "زمان بازگشت در هیدرولوژی"
            ],
            characteristics=[
                "k < 1: نرخ خرابی کاهشی (infant mortality)",
                "k = 1: نرخ خرابی ثابت = توزیع نمایی",
                "k > 1: نرخ خرابی افزایشی (wear-out)",
                "k ≈ 3.5: تقریباً نرمال",
                "انعطاف بالا در مدل‌سازی"
            ],
            warning="حساس به داده‌های پرت - از روش‌های robust استفاده کنید"
        )
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.pdf(x, self.params['c'], scale=self.params['scale'])
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.cdf(x, self.params['c'], scale=self.params['scale'])
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self._scipy_dist.ppf(q, self.params['c'], scale=self.params['scale'])
    
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)  # location=0
        return {"c": params[0], "scale": params[2]}
    
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        # برای وایبل MOM پیچیده است، از MLE استفاده می‌کنیم
        return self.fit_mle(data)


class GammaDistribution(BaseDistribution):
    """
    توزیع گاما
    
    برای مدل‌سازی زمان‌های انتظار و فرآیندهای شمارش.
    تعمیم توزیع نمایی.
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.gamma
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="gamma",
            display_name="Gamma Distribution",
            parameters={
                "a": "α (shape - شکل)",
                "scale": "θ (scale - مقیاس)"
            },
            support="(0, +∞)",
            use_cases=[
                "زمان انتظار برای k رویداد (فرآیند پواسون)",
                "مدل‌سازی باران و جریان رودخانه",
                "مدل‌سازی بار (load) در network",
                "توزیع prior در Bayesian",
                "زمان سرویس در صف‌ها"
            ],
            characteristics=[
                "α = 1: می‌شود توزیع نمایی",
                "α بزرگ: نزدیک به نرمال",
                "راست‌چوله برای α کوچک",
                "انعطاف‌پذیر در shape"
            ]
        )
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.pdf(x, self.params['a'], scale=self.params['scale'])
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.cdf(x, self.params['a'], scale=self.params['scale'])
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self._scipy_dist.ppf(q, self.params['a'], scale=self.params['scale'])
    
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        data = data[data > 0]
        params = self._scipy_dist.fit(data, floc=0)
        return {"a": params[0], "scale": params[2]}
    
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        m = np.mean(data)
        v = np.var(data, ddof=1)
        scale = v / m
        shape = m / scale
        return {"a": shape, "scale": scale}


class ExponentialDistribution(BaseDistribution):
    """
    توزیع نمایی
    
    برای زمان بین رویدادهای مستقل با نرخ ثابت.
    """
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.expon
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="exponential",
            display_name="Exponential Distribution",
            parameters={
                "scale": "1/λ (mean - میانگین)"
            },
            support="(0, +∞)",
            use_cases=[
                "زمان بین ورود مشتری (قانون پواسون)",
                "عمر قطعات با نرخ خرابی ثابت",
                "زمان‌های فاصله در radioactive decay",
                "مدت زمان تماس تلفنی"
            ],
            characteristics=[
                "بی‌حافظه (memoryless)",
                "نرخ خطر ثابت",
                "حداکثر آنتروپی برای میانگین مشخص",
                "ساده‌ترین توزیع برای lifetime"
            ]
        )
    
    def pdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.pdf(x, scale=self.params['scale'])
    
    def cdf(self, x: np.ndarray) -> np.ndarray:
        return self._scipy_dist.cdf(x, scale=self.params['scale'])
    
    def ppf(self, q: np.ndarray) -> np.ndarray:
        return self._scipy_dist.ppf(q, scale=self.params['scale'])
    
    def fit_mle(self, data: np.ndarray, **kwargs) -> Dict[str, float]:
        return {"scale": np.mean(data[data > 0])}
    
    def fit_moments(self, data: np.ndarray) -> Dict[str, float]:
        return self.fit_mle(data)


# Factory برای ساخت راحت توزیع‌ها
DISTRIBUTION_REGISTRY = {
    'normal': NormalDistribution,
    'lognormal': LognormalDistribution,
    'weibull': WeibullDistribution,
    'gamma': GammaDistribution,
    'exponential': ExponentialDistribution,
}


def get_distribution(name: str) -> BaseDistribution:
    """
    دریافت یک توزیع بر اساس نام
    
    Example:
    --------
    >>> dist = get_distribution('normal')
    >>> dist.fit(data)
    """
    name = name.lower()
    if name not in DISTRIBUTION_REGISTRY:
        available = ', '.join(DISTRIBUTION_REGISTRY.keys())
        raise ValueError(f"Unknown distribution '{name}'. Available: {available}")
    return DISTRIBUTION_REGISTRY[name]()


def list_distributions() -> List[str]:
    """لیست تمام توزیع‌های موجود"""
    return list(DISTRIBUTION_REGISTRY.keys())
