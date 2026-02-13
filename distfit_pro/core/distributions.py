"""
Distribution Implementations
============================

Concrete implementations of all probability distributions.
25 distributions total: 20 continuous + 5 discrete

Author: Ali Sadeghi Aghili
"""

import numpy as np
from scipy import stats
from typing import Dict, Optional
from .base import (
    ContinuousDistribution,
    DiscreteDistribution,
    DistributionInfo
)


# ============================================================================
# CONTINUOUS DISTRIBUTIONS (20)
# ============================================================================

class NormalDistribution(ContinuousDistribution):
    """Normal (Gaussian) Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.norm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="normal",
            scipy_name="norm",
            display_name="Normal Distribution",
            description="Symmetric bell-shaped continuous distribution. "
                       "Fundamental in statistics due to Central Limit Theorem.",
            parameters=["loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        loc, scale = self._scipy_dist.fit(data)
        self._params = {'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._params = {
            'loc': np.mean(data),
            'scale': np.std(data, ddof=1)
        }
    
    def mode(self) -> float:
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': self._params['loc'], 'scale': self._params['scale']}


class ExponentialDistribution(ContinuousDistribution):
    """Exponential Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.expon
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="exponential",
            scipy_name="expon",
            display_name="Exponential Distribution",
            description="Memoryless continuous distribution modeling waiting times.",
            parameters=["scale"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Exponential distribution requires non-negative data")
        loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Exponential distribution requires non-negative data")
        self._params = {'scale': np.mean(data)}
    
    def mode(self) -> float:
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': 0, 'scale': self._params['scale']}


class UniformDistribution(ContinuousDistribution):
    """Uniform Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.uniform
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="uniform",
            scipy_name="uniform",
            display_name="Uniform Distribution",
            description="Constant probability over a finite interval.",
            parameters=["loc", "scale"],
            support="[a, b]",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        loc, scale = self._scipy_dist.fit(data)
        self._params = {'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        a = np.min(data)
        b = np.max(data)
        self._params = {'loc': a, 'scale': b - a}
    
    def mode(self) -> float:
        raise NotImplementedError("Uniform distribution has no unique mode")
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': self._params['loc'], 'scale': self._params['scale']}


class GammaDistribution(ContinuousDistribution):
    """Gamma Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.gamma
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="gamma",
            scipy_name="gamma",
            display_name="Gamma Distribution",
            description="Flexible distribution for positive continuous data. Sum of exponentials.",
            parameters=["shape", "scale"],
            support="(0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Gamma distribution requires positive data")
        shape, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'shape': shape, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Gamma distribution requires positive data")
        mean_val = np.mean(data)
        var_val = np.var(data, ddof=1)
        scale = var_val / mean_val
        shape = mean_val / scale
        self._params = {'shape': shape, 'scale': scale}
    
    def mode(self) -> float:
        if self._params['shape'] >= 1:
            return (self._params['shape'] - 1) * self._params['scale']
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'a': self._params['shape'], 'loc': 0, 'scale': self._params['scale']}


class BetaDistribution(ContinuousDistribution):
    """Beta Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.beta
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="beta",
            scipy_name="beta",
            display_name="Beta Distribution",
            description="Flexible distribution for data bounded between 0 and 1.",
            parameters=["alpha", "beta"],
            support="[0, 1]",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any((data < 0) | (data > 1)):
            raise ValueError("Beta distribution requires data in [0, 1]")
        alpha, beta, loc, scale = self._scipy_dist.fit(data, floc=0, fscale=1)
        self._params = {'alpha': alpha, 'beta': beta}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any((data < 0) | (data > 1)):
            raise ValueError("Beta distribution requires data in [0, 1]")
        mean_val = np.mean(data)
        var_val = np.var(data, ddof=1)
        common = mean_val * (1 - mean_val) / var_val - 1
        alpha = mean_val * common
        beta = (1 - mean_val) * common
        self._params = {'alpha': max(alpha, 0.1), 'beta': max(beta, 0.1)}
    
    def mode(self) -> float:
        alpha, beta = self._params['alpha'], self._params['beta']
        if alpha > 1 and beta > 1:
            return (alpha - 1) / (alpha + beta - 2)
        raise NotImplementedError("Mode undefined for alpha<=1 or beta<=1")
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'a': self._params['alpha'], 'b': self._params['beta'], 'loc': 0, 'scale': 1}


class WeibullDistribution(ContinuousDistribution):
    """Weibull Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.weibull_min
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="weibull",
            scipy_name="weibull_min",
            display_name="Weibull Distribution",
            description="Widely used in reliability analysis and lifetime modeling.",
            parameters=["shape", "scale"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Weibull distribution requires non-negative data")
        shape, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'shape': shape, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        shape, scale = self._params['shape'], self._params['scale']
        if shape > 1:
            return scale * ((shape - 1) / shape) ** (1 / shape)
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'c': self._params['shape'], 'loc': 0, 'scale': self._params['scale']}


class LognormalDistribution(ContinuousDistribution):
    """Lognormal Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.lognorm
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="lognormal",
            scipy_name="lognorm",
            display_name="Lognormal Distribution",
            description="Distribution of variable whose logarithm is normally distributed.",
            parameters=["shape", "scale"],
            support="(0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Lognormal distribution requires positive data")
        shape, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'shape': shape, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Lognormal distribution requires positive data")
        log_data = np.log(data)
        shape = np.std(log_data, ddof=1)
        scale = np.exp(np.mean(log_data))
        self._params = {'shape': shape, 'scale': scale}
    
    def mode(self) -> float:
        shape, scale = self._params['shape'], self._params['scale']
        return scale * np.exp(-shape**2)
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'s': self._params['shape'], 'loc': 0, 'scale': self._params['scale']}


class LogisticDistribution(ContinuousDistribution):
    """Logistic Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.logistic
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="logistic",
            scipy_name="logistic",
            display_name="Logistic Distribution",
            description="Similar to normal but with heavier tails.",
            parameters=["loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        loc, scale = self._scipy_dist.fit(data)
        self._params = {'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        loc = np.mean(data)
        scale = np.std(data, ddof=1) * np.sqrt(3) / np.pi
        self._params = {'loc': loc, 'scale': scale}
    
    def mode(self) -> float:
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': self._params['loc'], 'scale': self._params['scale']}


class GumbelDistribution(ContinuousDistribution):
    """Gumbel Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.gumbel_r
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="gumbel",
            scipy_name="gumbel_r",
            display_name="Gumbel Distribution",
            description="Type I extreme value distribution.",
            parameters=["loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        loc, scale = self._scipy_dist.fit(data)
        self._params = {'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        mean_val = np.mean(data)
        std_val = np.std(data, ddof=1)
        scale = std_val * np.sqrt(6) / np.pi
        loc = mean_val - 0.5772 * scale
        self._params = {'loc': loc, 'scale': scale}
    
    def mode(self) -> float:
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': self._params['loc'], 'scale': self._params['scale']}


class ParetoDistribution(ContinuousDistribution):
    """Pareto Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.pareto
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="pareto",
            scipy_name="pareto",
            display_name="Pareto Distribution",
            description="Power law distribution.",
            parameters=["shape", "scale"],
            support="[scale, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Pareto distribution requires positive data")
        shape, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'shape': shape, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Pareto distribution requires positive data")
        scale = np.min(data)
        mean_val = np.mean(data)
        shape = mean_val / (mean_val - scale)
        self._params = {'shape': max(shape, 1.01), 'scale': scale}
    
    def mode(self) -> float:
        return self._params['scale']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'b': self._params['shape'], 'loc': 0, 'scale': self._params['scale']}


class CauchyDistribution(ContinuousDistribution):
    """Cauchy Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.cauchy
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="cauchy",
            scipy_name="cauchy",
            display_name="Cauchy Distribution",
            description="Heavy-tailed distribution with undefined mean and variance.",
            parameters=["loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        loc, scale = self._scipy_dist.fit(data)
        self._params = {'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        loc = np.median(data)
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        scale = iqr / 2
        self._params = {'loc': loc, 'scale': scale}
    
    def mode(self) -> float:
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': self._params['loc'], 'scale': self._params['scale']}


class StudentTDistribution(ContinuousDistribution):
    """Student's t Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.t
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="studentt",
            scipy_name="t",
            display_name="Student's t Distribution",
            description="Heavy-tailed distribution used in statistical inference.",
            parameters=["df", "loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        df, loc, scale = self._scipy_dist.fit(data)
        self._params = {'df': df, 'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'df': self._params['df'], 'loc': self._params['loc'], 'scale': self._params['scale']}


class ChiSquareDistribution(ContinuousDistribution):
    """Chi-Square Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.chi2
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="chisquare",
            scipy_name="chi2",
            display_name="Chi-Square Distribution",
            description="Distribution of sum of squared standard normal variables.",
            parameters=["df"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Chi-square distribution requires non-negative data")
        df, loc, scale = self._scipy_dist.fit(data, floc=0, fscale=1)
        self._params = {'df': df}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Chi-square distribution requires non-negative data")
        df = np.mean(data)
        self._params = {'df': max(df, 1)}
    
    def mode(self) -> float:
        df = self._params['df']
        return max(df - 2, 0)
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'df': self._params['df'], 'loc': 0, 'scale': 1}


class FDistribution(ContinuousDistribution):
    """F Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.f
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="f",
            scipy_name="f",
            display_name="F Distribution",
            description="Ratio of two chi-square distributions.",
            parameters=["dfn", "dfd"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("F distribution requires non-negative data")
        dfn, dfd, loc, scale = self._scipy_dist.fit(data, floc=0, fscale=1)
        self._params = {'dfn': dfn, 'dfd': dfd}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        dfn, dfd = self._params['dfn'], self._params['dfd']
        if dfn > 2:
            return (dfn - 2) / dfn * dfd / (dfd + 2)
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'dfn': self._params['dfn'], 'dfd': self._params['dfd'], 'loc': 0, 'scale': 1}


class LaplaceDistribution(ContinuousDistribution):
    """Laplace Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.laplace
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="laplace",
            scipy_name="laplace",
            display_name="Laplace Distribution",
            description="Double exponential distribution.",
            parameters=["loc", "scale"],
            support="(-inf, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        loc, scale = self._scipy_dist.fit(data)
        self._params = {'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        loc = np.median(data)
        scale = np.mean(np.abs(data - loc))
        self._params = {'loc': loc, 'scale': scale}
    
    def mode(self) -> float:
        return self._params['loc']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': self._params['loc'], 'scale': self._params['scale']}


class RayleighDistribution(ContinuousDistribution):
    """Rayleigh Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.rayleigh
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="rayleigh",
            scipy_name="rayleigh",
            display_name="Rayleigh Distribution",
            description="Models magnitude of 2D vector with normally distributed components.",
            parameters=["scale"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Rayleigh distribution requires non-negative data")
        loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Rayleigh distribution requires non-negative data")
        scale = np.sqrt(np.mean(data**2) / 2)
        self._params = {'scale': scale}
    
    def mode(self) -> float:
        return self._params['scale']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'loc': 0, 'scale': self._params['scale']}


class WaldDistribution(ContinuousDistribution):
    """Wald (Inverse Gaussian) Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.invgauss
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="wald",
            scipy_name="invgauss",
            display_name="Wald Distribution",
            description="Inverse Gaussian distribution.",
            parameters=["mean", "scale"],
            support="(0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Wald distribution requires positive data")
        mu, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'mean': mu * scale, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        if np.any(data <= 0):
            raise ValueError("Wald distribution requires positive data")
        mean_val = np.mean(data)
        self._params = {'mean': mean_val, 'scale': mean_val}
    
    def mode(self) -> float:
        mean = self._params['mean']
        scale = self._params['scale']
        return mean * (np.sqrt(1 + (1.5 * mean / scale)**2) - 1.5 * mean / scale)
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'mu': self._params['mean'] / self._params['scale'], 'loc': 0, 'scale': self._params['scale']}


class TriangularDistribution(ContinuousDistribution):
    """Triangular Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.triang
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="triangular",
            scipy_name="triang",
            display_name="Triangular Distribution",
            description="Simple distribution defined by minimum, maximum, and mode.",
            parameters=["c", "loc", "scale"],
            support="[a, b]",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        c, loc, scale = self._scipy_dist.fit(data)
        self._params = {'c': c, 'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        a = np.min(data)
        b = np.max(data)
        mean_val = np.mean(data)
        c = (mean_val - a) / (b - a)
        self._params = {'c': c, 'loc': a, 'scale': b - a}
    
    def mode(self) -> float:
        return self._params['loc'] + self._params['c'] * self._params['scale']
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'c': self._params['c'], 'loc': self._params['loc'], 'scale': self._params['scale']}


class BurrDistribution(ContinuousDistribution):
    """Burr Type XII Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.burr12
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="burr",
            scipy_name="burr12",
            display_name="Burr Distribution",
            description="Flexible distribution for modeling heavy-tailed data.",
            parameters=["c", "d", "scale"],
            support="[0, inf)",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if np.any(data < 0):
            raise ValueError("Burr distribution requires non-negative data")
        c, d, loc, scale = self._scipy_dist.fit(data, floc=0)
        self._params = {'c': c, 'd': d, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        c, d = self._params['c'], self._params['d']
        scale = self._params['scale']
        if c > 1:
            return scale * ((c - 1) / (c * d + 1)) ** (1 / c)
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'c': self._params['c'], 'd': self._params['d'], 'loc': 0, 'scale': self._params['scale']}


class GenExtremeDistribution(ContinuousDistribution):
    """Generalized Extreme Value Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.genextreme
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="genextreme",
            scipy_name="genextreme",
            display_name="Generalized Extreme Value Distribution",
            description="Family of distributions for modeling extreme values.",
            parameters=["shape", "loc", "scale"],
            support="varies",
            is_discrete=False,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        shape, loc, scale = self._scipy_dist.fit(data)
        self._params = {'shape': shape, 'loc': loc, 'scale': scale}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        shape, loc, scale = self._params['shape'], self._params['loc'], self._params['scale']
        if shape != 0:
            return loc + scale * ((1 + shape) ** (-shape) - 1) / shape
        return loc
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'c': self._params['shape'], 'loc': self._params['loc'], 'scale': self._params['scale']}


# ============================================================================
# DISCRETE DISTRIBUTIONS (5)
# ============================================================================

class PoissonDistribution(DiscreteDistribution):
    """Poisson Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.poisson
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="poisson",
            scipy_name="poisson",
            display_name="Poisson Distribution",
            description="Models count of events in fixed interval with constant rate.",
            parameters=["mu"],
            support="{0, 1, 2, ...}",
            is_discrete=True,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        if not np.all(data >= 0):
            raise ValueError("Poisson distribution requires non-negative integer data")
        mu = np.mean(data)
        self._params = {'mu': mu}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        return np.floor(self._params['mu'])
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'mu': self._params['mu']}


class BinomialDistribution(DiscreteDistribution):
    """Binomial Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.binom
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="binomial",
            scipy_name="binom",
            display_name="Binomial Distribution",
            description="Number of successes in n independent Bernoulli trials.",
            parameters=["n", "p"],
            support="{0, 1, ..., n}",
            is_discrete=True,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, n: int = None, **kwargs):
        if n is None:
            n = int(np.max(data))
        p = np.mean(data) / n
        self._params = {'n': n, 'p': np.clip(p, 0.001, 0.999)}
    
    def _fit_mom(self, data: np.ndarray, n: int = None, **kwargs):
        self._fit_mle(data, n=n, **kwargs)
    
    def mode(self) -> float:
        return np.floor((self._params['n'] + 1) * self._params['p'])
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'n': self._params['n'], 'p': self._params['p']}


class NegativeBinomialDistribution(DiscreteDistribution):
    """Negative Binomial Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.nbinom
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="negativebinomial",
            scipy_name="nbinom",
            display_name="Negative Binomial Distribution",
            description="Number of failures before r-th success.",
            parameters=["n", "p"],
            support="{0, 1, 2, ...}",
            is_discrete=True,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        mean_val = np.mean(data)
        var_val = np.var(data, ddof=1)
        if var_val <= mean_val:
            var_val = mean_val * 1.1
        p = mean_val / var_val
        n = mean_val * p / (1 - p)
        self._params = {'n': max(n, 0.1), 'p': np.clip(p, 0.001, 0.999)}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        n, p = self._params['n'], self._params['p']
        if n > 1:
            return np.floor((n - 1) * (1 - p) / p)
        return 0.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'n': self._params['n'], 'p': self._params['p']}


class GeometricDistribution(DiscreteDistribution):
    """Geometric Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.geom
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="geometric",
            scipy_name="geom",
            display_name="Geometric Distribution",
            description="Number of trials until first success.",
            parameters=["p"],
            support="{1, 2, 3, ...}",
            is_discrete=True,
            has_shape_params=False
        )
    
    def _fit_mle(self, data: np.ndarray, **kwargs):
        p = 1 / np.mean(data)
        self._params = {'p': np.clip(p, 0.001, 0.999)}
    
    def _fit_mom(self, data: np.ndarray, **kwargs):
        self._fit_mle(data, **kwargs)
    
    def mode(self) -> float:
        return 1.0
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'p': self._params['p']}


class HypergeometricDistribution(DiscreteDistribution):
    """Hypergeometric Distribution"""
    
    def __init__(self):
        super().__init__()
        self._scipy_dist = stats.hypergeom
    
    @property
    def info(self) -> DistributionInfo:
        return DistributionInfo(
            name="hypergeometric",
            scipy_name="hypergeom",
            display_name="Hypergeometric Distribution",
            description="Sampling without replacement from finite population.",
            parameters=["M", "n", "N"],
            support="{max(0, n+N-M), ..., min(n, N)}",
            is_discrete=True,
            has_shape_params=True
        )
    
    def _fit_mle(self, data: np.ndarray, M: int = None, n: int = None, **kwargs):
        if M is None or n is None:
            raise ValueError("Hypergeometric requires M (population) and n (draws) parameters")
        N = int(np.round(np.mean(data) * M / n))
        self._params = {'M': M, 'n': n, 'N': N}
    
    def _fit_mom(self, data: np.ndarray, M: int = None, n: int = None, **kwargs):
        self._fit_mle(data, M=M, n=n, **kwargs)
    
    def mode(self) -> float:
        M, n, N = self._params['M'], self._params['n'], self._params['N']
        return np.floor((n + 1) * (N + 1) / (M + 2))
    
    def _get_scipy_params(self) -> Dict[str, float]:
        return {'M': self._params['M'], 'n': self._params['n'], 'N': self._params['N']}


# ============================================================================
# DISTRIBUTION REGISTRY - ALL 25 DISTRIBUTIONS
# ============================================================================

_DISTRIBUTION_REGISTRY = {
    # Continuous (20)
    'normal': NormalDistribution,
    'exponential': ExponentialDistribution,
    'uniform': UniformDistribution,
    'gamma': GammaDistribution,
    'beta': BetaDistribution,
    'weibull': WeibullDistribution,
    'lognormal': LognormalDistribution,
    'logistic': LogisticDistribution,
    'gumbel': GumbelDistribution,
    'pareto': ParetoDistribution,
    'cauchy': CauchyDistribution,
    'studentt': StudentTDistribution,
    'chisquare': ChiSquareDistribution,
    'f': FDistribution,
    'laplace': LaplaceDistribution,
    'rayleigh': RayleighDistribution,
    'wald': WaldDistribution,
    'triangular': TriangularDistribution,
    'burr': BurrDistribution,
    'genextreme': GenExtremeDistribution,
    # Discrete (5)
    'poisson': PoissonDistribution,
    'binomial': BinomialDistribution,
    'negativebinomial': NegativeBinomialDistribution,
    'geometric': GeometricDistribution,
    'hypergeometric': HypergeometricDistribution,
}


def get_distribution(name: str):
    """Get distribution by name (supports aliases and underscores)"""
    # Normalize: lowercase, remove spaces/dashes/underscores
    name_normalized = name.lower().replace(' ', '').replace('-', '').replace('_', '')
    
    # Try to find in registry (also strip underscores from keys)
    for key, dist_class in _DISTRIBUTION_REGISTRY.items():
        if key.replace('_', '') == name_normalized:
            return dist_class()
    
    available = ", ".join(sorted(_DISTRIBUTION_REGISTRY.keys()))
    raise ValueError(f"Unknown distribution: {name}. Available: {available}")


def list_distributions(discrete_only: bool = False, continuous_only: bool = False) -> list:
    """List available distributions"""
    if discrete_only:
        return sorted([k for k, v in _DISTRIBUTION_REGISTRY.items() 
                      if issubclass(v, DiscreteDistribution)])
    elif continuous_only:
        return sorted([k for k, v in _DISTRIBUTION_REGISTRY.items() 
                      if issubclass(v, ContinuousDistribution)])
    return sorted(_DISTRIBUTION_REGISTRY.keys())
