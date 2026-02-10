"""
Distribution Fitter - موتور اصلی فیت
====================================

این کلاس همه‌چیز را کنار هم می‌آورد:
- فیت چند توزیع
- مقایسه با معیارهای مختلف
- تشخیص خودکار
- توضیحات جامع
"""

import warnings
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm

from ..core.distributions import get_distribution, list_distributions, BaseDistribution
from ..core.model_selection import ModelSelection, DeltaComparison, ModelScore
from ..visualization.plots import DistributionPlotter


@dataclass
class FitResults:
    """
    نتایج فیت با توضیحات کامل
    """
    data_summary: Dict
    fitted_models: List[BaseDistribution]
    model_scores: List[ModelScore]
    best_model: BaseDistribution
    diagnostics: Dict
    recommendations: List[str]
    _data: np.ndarray = None  # داده اصلی برای plot
    
    def summary(self) -> str:
        """
        خلاصه کامل و self-explanatory از نتایج
        """
        output = []
        output.append("\n" + "="*70)
        output.append("🔍 نتایج فیت توزیع‌های آماری")
        output.append("="*70)
        
        # خلاصه داده
        output.append("\n📊 خلاصه داده:")
        ds = self.data_summary
        output.append(f"   • تعداد: {ds['n']}")
        output.append(f"   • میانگین: {ds['mean']:.4f} (CI 95%: [{ds['mean_ci'][0]:.4f}, {ds['mean_ci'][1]:.4f}])")
        output.append(f"   • انحراف معیار: {ds['std']:.4f}")
        output.append(f"   • چولگی: {ds['skewness']:.4f} → {ds['skewness_interp']}")
        output.append(f"   • کشیدگی: {ds['kurtosis']:.4f} → {ds['kurtosis_interp']}")
        
        if ds['n_outliers'] > 0:
            output.append(f"   • ⚠️  Outliers: {ds['n_outliers']} ({ds['outlier_pct']:.1f}%) detected")
        
        # رتبه‌بندی مدل‌ها
        output.append("\n🏆 رتبه‌بندی مدل‌ها:")
        output.append("\n" + self._create_ranking_table())
        
        # بهترین مدل
        output.append(f"\n✨ مدل برتر: {self.best_model.info.display_name}")
        output.append(self.best_model.explain())
        
        # تشخیص‌ها
        if self.diagnostics.get('notes'):
            output.append("\n⚠️  یادداشت‌های تشخیصی:")
            for note in self.diagnostics['notes']:
                output.append(f"   • {note}")
        
        # پیشنهادات
        if self.recommendations:
            output.append("\n💡 پیشنهادات:")
            for rec in self.recommendations:
                output.append(f"   • {rec}")
        
        return "\n".join(output)
    
    def _create_ranking_table(self) -> str:
        """ساخت جدول رتبه‌بندی"""
        rows = []
        header = f"{'رتبه':<6} {'توزیع':<15} {self.model_scores[0].criterion:<10} {'Δ':<10} {'وضعیت'}"
        rows.append(header)
        rows.append("-" * 70)
        
        best_score = self.model_scores[0].score
        
        for i, score in enumerate(self.model_scores, 1):
            delta = score.score - best_score
            if delta < 2:
                status = "✅"
            elif delta < 7:
                status = "⚠️"
            else:
                status = "❌"
            
            row = f"{i:<6} {score.distribution_name:<15} {score.score:<10.2f} {delta:<10.2f} {status}"
            rows.append(row)
        
        return "\n".join(rows)
    
    def get_best(self, criterion: Optional[str] = None) -> BaseDistribution:
        """دریافت بهترین مدل"""
        return self.best_model
    
    def plot(
        self,
        kind: str = 'comparison',
        figsize: Optional[Tuple[int, int]] = None,
        show_top_n: int = 3,
        show: bool = True
    ):
        """
        رسم نمودارهای تشخیصی و مقایسه‌ای
        
        Parameters:
        -----------
        kind : str
            نوع نمودار:
            - 'comparison': PDF, CDF, P-P, Q-Q plots
            - 'diagnostics': Residual analysis, tail behavior
            - 'interactive': Interactive Plotly dashboard
        figsize : tuple, optional
            اندازه figure (فقط برای matplotlib)
        show_top_n : int
            تعداد بهترین مدل‌ها برای نمایش
        show : bool
            نمایش نمودار (برای matplotlib)
        
        Returns:
        --------
        fig : matplotlib Figure یا plotly Figure
        
        Examples:
        ---------
        >>> results.plot(kind='comparison')  # P-P, Q-Q, PDF, CDF
        >>> results.plot(kind='diagnostics')  # Residuals, tail behavior
        >>> results.plot(kind='interactive')  # Interactive Plotly
        """
        if self._data is None:
            raise ValueError(
                "داده اصلی در دسترس نیست. این اتفاق نباید بیفتد - لطفاً bug report کنید!"
            )
        
        plotter = DistributionPlotter(
            data=self._data,
            fitted_models=self.fitted_models,
            best_model=self.best_model
        )
        
        if kind == 'comparison':
            figsize = figsize or (14, 10)
            fig = plotter.plot_comparison(figsize=figsize, show_top_n=show_top_n)
            if show:
                import matplotlib.pyplot as plt
                plt.show()
            return fig
        
        elif kind == 'diagnostics':
            figsize = figsize or (14, 10)
            fig = plotter.plot_diagnostics(figsize=figsize)
            if show:
                import matplotlib.pyplot as plt
                plt.show()
            return fig
        
        elif kind == 'interactive':
            fig = plotter.plot_interactive(show_top_n=show_top_n)
            if show:
                fig.show()
            return fig
        
        else:
            raise ValueError(
                f"نوع نمودار '{kind}' نامعتبر است. "
                "گزینه‌های معتبر: 'comparison', 'diagnostics', 'interactive'"
            )


class DistributionFitter:
    """
    کلاس اصلی برای فیت توزیع‌ها
    
    این کلاس:
    1. داده را آنالیز می‌کند
    2. توزیع‌های مناسب را انتخاب می‌کند (یا شما انتخاب کنید)
    3. هر توزیع را فیت می‌کند
    4. مدل‌ها را مقایسه می‌کند
    5. نتایج را به صورت self-explanatory ارائه می‌دهد
    
    Example:
    --------
    >>> fitter = DistributionFitter(data)
    >>> results = fitter.fit(distributions=['normal', 'lognormal', 'weibull'])
    >>> print(results.summary())
    >>> results.plot(kind='comparison')
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, list],
        censoring: Optional[np.ndarray] = None,
        censoring_type: Optional[str] = None,
        weights: Optional[np.ndarray] = None
    ):
        """
        ساخت fitter
        
        Parameters:
        -----------
        data : array-like
            داده‌ی مشاهده‌شده
        censoring : array-like, optional
            اندیکاتور censoring (1=مشاهده شده, 0=سانسور)
        censoring_type : str, optional
            'left', 'right', 'interval'
        weights : array-like, optional
            وزن هر مشاهده
        """
        self.data = np.asarray(data).flatten()
        self.data = self.data[~np.isnan(self.data)]
        
        self.censoring = censoring
        self.censoring_type = censoring_type
        self.weights = weights
        
        # آنالیز اولیه داده
        self.data_summary = self._analyze_data()
        
        # نتایج فیت
        self.fitted_models = []
        self.results: Optional[FitResults] = None
    
    def _analyze_data(self) -> Dict:
        """
        آنالیز اولیه داده
        
        این تابع:
        - آمارها را محاسبه می‌کند
        - outliers را شناسایی می‌کند
        - شکل توزیع را تشخیص می‌دهد
        - توضیح می‌دهد داده چه ویژگی‌هایی دارد
        """
        n = len(self.data)
        mean = np.mean(self.data)
        std = np.std(self.data, ddof=1)
        
        # Confidence interval for mean (t-distribution)
        se = std / np.sqrt(n)
        t_crit = stats.t.ppf(0.975, n-1)
        mean_ci = (mean - t_crit * se, mean + t_crit * se)
        
        # Shape statistics
        skewness = stats.skew(self.data)
        kurtosis = stats.kurtosis(self.data)  # excess kurtosis
        
        # Outlier detection (IQR method)
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (self.data < lower_bound) | (self.data > upper_bound)
        n_outliers = np.sum(outliers)
        
        # تفسیر چولگی
        if abs(skewness) < 0.5:
            skew_interp = "تقریباً متقارن"
        elif skewness > 0:
            skew_interp = f"راست‌چوله (دنباله سمت راست بلند)"
        else:
            skew_interp = f"چپ‌چوله (دنباله سمت چپ بلند)"
        
        # تفسیر کشیدگی
        if abs(kurtosis) < 0.5:
            kurt_interp = "نزدیک به نرمال"
        elif kurtosis > 0:
            kurt_interp = "دنباله‌های سنگین‌تر از نرمال (heavy-tailed)"
        else:
            kurt_interp = "دنباله‌های سبک‌تر از نرمال (light-tailed)"
        
        summary = {
            'n': n,
            'mean': mean,
            'mean_ci': mean_ci,
            'std': std,
            'median': np.median(self.data),
            'min': np.min(self.data),
            'max': np.max(self.data),
            'skewness': skewness,
            'skewness_interp': skew_interp,
            'kurtosis': kurtosis,
            'kurtosis_interp': kurt_interp,
            'n_outliers': n_outliers,
            'outlier_pct': 100 * n_outliers / n,
            'q1': q1,
            'q3': q3
        }
        
        return summary
    
    def suggest_distributions(self) -> List[str]:
        """
        پیشنهاد توزیع‌های مناسب بر اساس ویژگی‌های داده
        
        این تابع توضیح می‌دهد چرا این توزیع‌ها پیشنهاد می‌شوند.
        """
        suggestions = []
        reasons = []
        
        ds = self.data_summary
        all_positive = ds['min'] > 0
        is_skewed = abs(ds['skewness']) > 0.5
        is_heavy_tailed = ds['kurtosis'] > 1
        
        # همیشه Normal را امتحان کن (baseline)
        suggestions.append('normal')
        reasons.append("Normal: به عنوان baseline همیشه امتحان می‌شود")
        
        # اگر همه مثبت
        if all_positive:
            suggestions.append('lognormal')
            reasons.append("Lognormal: داده فقط مثبت است")
            
            suggestions.append('gamma')
            reasons.append("Gamma: مناسب برای داده‌های مثبت و راست‌چوله")
            
            suggestions.append('weibull')
            reasons.append("Weibull: مناسب برای lifetime و reliability")
            
            suggestions.append('exponential')
            reasons.append("Exponential: ساده‌ترین توزیع برای داده‌های مثبت")
        
        # اگر چوله
        if is_skewed and ds['skewness'] > 0 and all_positive:
            if 'lognormal' not in suggestions:
                suggestions.append('lognormal')
                reasons.append("Lognormal: داده راست‌چوله است")
        
        # اگر دنباله سنگین
        if is_heavy_tailed:
            reasons.append("⚠️  داده دنباله سنگین دارد - توزیع‌های Student-t یا Cauchy را در نظر بگیرید")
        
        print("\n💡 توزیع‌های پیشنهادی:")
        for reason in reasons:
            print(f"   • {reason}")
        
        return suggestions
    
    def fit(
        self,
        distributions: Optional[List[str]] = None,
        method: str = 'mle',
        criterion: str = 'aic',
        n_jobs: int = 1,
        verbose: bool = True
    ) -> FitResults:
        """
        فیت توزیع‌ها به داده
        
        Parameters:
        -----------
        distributions : list of str, optional
            نام توزیع‌ها. اگر None باشد، خودکار پیشنهاد می‌شود
        method : str
            روش تخمین: 'mle', 'moments', 'quantile'
        criterion : str
            معیار انتخاب مدل: 'aic', 'bic', 'loo_cv'
        n_jobs : int
            تعداد کورهای موازی (-1 = همه)
        verbose : bool
            نمایش پیشرفت
        
        Returns:
        --------
        results : FitResults
            نتایج کامل با توضیحات
        """
        # اگر توزیع مشخص نشده، پیشنهاد بده
        if distributions is None:
            distributions = self.suggest_distributions()
        
        if verbose:
            print(f"\n🚀 شروع فیت {len(distributions)} توزیع...")
            print(f"   • روش تخمین: {method.upper()}")
            print(f"   • معیار انتخاب: {criterion.upper()}")
            print(f"   • تعداد کور: {n_jobs if n_jobs > 0 else 'همه'}\n")
        
        # فیت موازی
        if n_jobs != 1:
            fitted = Parallel(n_jobs=n_jobs)(
                delayed(self._fit_single)(dist_name, method, verbose)
                for dist_name in distributions
            )
        else:
            fitted = []
            iterator = tqdm(distributions, desc="فیت توزیع‌ها") if verbose else distributions
            for dist_name in iterator:
                if verbose and not isinstance(iterator, tqdm):
                    print(f"   فیت {dist_name}...", end=" ")
                result = self._fit_single(dist_name, method, verbose=False)
                fitted.append(result)
                if verbose and not isinstance(iterator, tqdm):
                    print("✓")
        
        # حذف مدل‌هایی که فیت نشدند
        self.fitted_models = [f for f in fitted if f is not None]
        
        if len(self.fitted_models) == 0:
            raise RuntimeError("هیچ مدلی فیت نشد. لطفاً داده و توزیع‌ها را بررسی کنید.")
        
        # مقایسه مدل‌ها
        if verbose:
            print("\n📊 مقایسه مدل‌ها...")
        
        model_scores = ModelSelection.compare_models(
            self.data,
            self.fitted_models,
            criterion=criterion
        )
        
        best_model = self.fitted_models[
            [m.info.name for m in self.fitted_models].index(model_scores[0].distribution_name)
        ]
        
        # تشخیص‌ها
        diagnostics = self._run_diagnostics(best_model)
        
        # پیشنهادات
        recommendations = self._generate_recommendations(model_scores, diagnostics)
        
        # ساخت نتایج
        self.results = FitResults(
            data_summary=self.data_summary,
            fitted_models=self.fitted_models,
            model_scores=model_scores,
            best_model=best_model,
            diagnostics=diagnostics,
            recommendations=recommendations,
            _data=self.data.copy()  # ذخیره داده برای plot
        )
        
        if verbose:
            print("\n✅ فیت کامل شد!\n")
        
        return self.results
    
    def _fit_single(
        self,
        dist_name: str,
        method: str,
        verbose: bool = False
    ) -> Optional[BaseDistribution]:
        """
        فیت یک توزیع
        """
        try:
            dist = get_distribution(dist_name)
            dist.fit(self.data, method=method)
            return dist
        except Exception as e:
            if verbose:
                warnings.warn(f"فیت {dist_name} ناموفق بود: {str(e)}")
            return None
    
    def _run_diagnostics(self, model: BaseDistribution) -> Dict:
        """
        تشخیص‌های پایه
        """
        notes = []
        
        # KS test
        ks_stat, ks_pval = stats.kstest(self.data, model.cdf)
        if ks_pval < 0.05:
            notes.append(f"⚠️  آزمون KS معنی‌دار است (p={ks_pval:.4f}) - مدل شاید کاملاً مناسب نباشد")
        
        # Residual analysis (ساده)
        theoretical_quantiles = model.ppf(np.linspace(0.01, 0.99, len(self.data)))
        empirical_quantiles = np.sort(self.data)
        residuals = empirical_quantiles - theoretical_quantiles
        
        if np.max(np.abs(residuals)) > 2 * self.data_summary['std']:
            notes.append("⚠️  انحراف زیاد در دم‌های توزیع - بررسی Q-Q plot توصیه می‌شود")
        
        return {
            'ks_stat': ks_stat,
            'ks_pval': ks_pval,
            'notes': notes
        }
    
    def _generate_recommendations(
        self,
        scores: List[ModelScore],
        diagnostics: Dict
    ) -> List[str]:
        """
        تولید پیشنهادات
        """
        recs = []
        
        # اگر چند مدل نزدیک به هم
        if len(scores) > 1 and scores[1].score - scores[0].score < 2:
            recs.append(
                f"مدل‌های {scores[0].distribution_name} و {scores[1].distribution_name} "
                "تقریباً یکسان هستند. برای حساسیت‌سنجی هر دو را امتحان کنید."
            )
        
        # اگر KS معنی‌دار
        if diagnostics['ks_pval'] < 0.05:
            recs.append(
                "آزمون KS نشان می‌دهد fit کامل نیست. "
                "نمودارهای تشخیصی (Q-Q plot) را بررسی کنید."
            )
        
        # پیشنهاد bootstrap
        recs.append(
            "برای اطمینان از پارامترها، bootstrap CI را محاسبه کنید: "
            "results.best_model.bootstrap_ci()"
        )
        
        return recs
