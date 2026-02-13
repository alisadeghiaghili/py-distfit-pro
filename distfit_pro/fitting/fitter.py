"""
Distribution Fitter - Main Fitting Engine
==========================================

This class brings everything together:
- Fit multiple distributions
- Compare with different criteria
- Automatic detection
- Comprehensive explanations
"""

import warnings
from typing import List, Optional, Union, Dict, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from tqdm import tqdm

from ..core.base import BaseDistribution
from ..core.distributions import get_distribution, list_distributions
from ..core.model_selection import ModelSelection, DeltaComparison, ModelScore
from ..visualization.plots import DistributionPlotter
from ..locales import t


@dataclass
class FitResults:
    """
    Fitting results with complete explanations
    """
    data_summary: Dict
    fitted_models: List[BaseDistribution]
    model_scores: List[ModelScore]
    best_model: BaseDistribution
    diagnostics: Dict
    recommendations: List[str]
    _data: np.ndarray = None
    
    def summary(self) -> str:
        """
        Complete self-explanatory summary of results
        """
        output = []
        output.append("\n" + "="*70)
        output.append(f"🔍 {t('fit_results')}")
        output.append("="*70)
        
        # Data summary
        output.append(f"\n📊 {t('data_summary')}:")
        ds = self.data_summary
        output.append(f"   • {t('sample_size')}: {ds['n']}")
        output.append(f"   • {t('mean')}: {ds['mean']:.4f} ({t('ci_95')}: [{ds['mean_ci'][0]:.4f}, {ds['mean_ci'][1]:.4f}])")
        output.append(f"   • {t('std_dev')}: {ds['std']:.4f}")
        output.append(f"   • {t('skewness')}: {ds['skewness']:.4f} → {ds['skewness_interp']}")
        output.append(f"   • {t('kurtosis')}: {ds['kurtosis']:.4f} → {ds['kurtosis_interp']}")
        
        if ds['n_outliers'] > 0:
            output.append(f"   • ⚠️  {t('outliers')}: {ds['n_outliers']} ({ds['outlier_pct']:.1f}%) detected")
        
        # Model ranking
        output.append(f"\n🏆 {t('model_ranking')}:")
        output.append("\n" + self._create_ranking_table())
        
        # Best model
        output.append(f"\n✨ {t('best_model')}: {self.best_model.info.display_name}")
        output.append(self.best_model.explain())
        
        # Diagnostics
        if self.diagnostics.get('notes'):
            output.append(f"\n⚠️  {t('diagnostic_notes')}:")
            for note in self.diagnostics['notes']:
                output.append(f"   • {note}")
        
        # Recommendations
        if self.recommendations:
            output.append(f"\n💡 {t('recommendations')}:")
            for rec in self.recommendations:
                output.append(f"   • {rec}")
        
        return "\n".join(output)
    
    def _create_ranking_table(self) -> str:
        """Create ranking table"""
        rows = []
        header = f"{t('rank'):<6} {t('distribution'):<15} {self.model_scores[0].criterion:<10} {t('delta'):<10} {t('status')}"
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
        """Get best model"""
        return self.best_model
    
    def plot(
        self,
        kind: str = 'comparison',
        figsize: Optional[Tuple[int, int]] = None,
        show_top_n: int = 3,
        show: bool = True
    ):
        """
        Plot diagnostic and comparison plots
        
        Parameters:
        -----------
        kind : str
            Plot type:
            - 'comparison': PDF, CDF, P-P, Q-Q plots
            - 'diagnostics': Residual analysis, tail behavior
            - 'interactive': Interactive Plotly dashboard
        figsize : tuple, optional
            Figure size (matplotlib only)
        show_top_n : int
            Number of best models to show
        show : bool
            Show plot (for matplotlib)
        
        Returns:
        --------
        fig : matplotlib Figure or plotly Figure
        """
        if self._data is None:
            raise ValueError(t('error_no_data'))
        
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
            raise ValueError(t('error_invalid_plot_kind', kind=kind))


class DistributionFitter:
    """
    Main class for distribution fitting
    
    This class:
    1. Analyzes the data
    2. Selects appropriate distributions (or you choose)
    3. Fits each distribution
    4. Compares models
    5. Presents results in a self-explanatory manner
    """
    
    def __init__(
        self,
        data: Union[np.ndarray, list],
        censoring: Optional[np.ndarray] = None,
        censoring_type: Optional[str] = None,
        weights: Optional[np.ndarray] = None
    ):
        """
        Create fitter
        
        Parameters:
        -----------
        data : array-like
            Observed data
        censoring : array-like, optional
            Censoring indicator (1=observed, 0=censored)
        censoring_type : str, optional
            'left', 'right', 'interval'
        weights : array-like, optional
            Weight of each observation
        """
        self.data = np.asarray(data).flatten()
        self.data = self.data[~np.isnan(self.data)]
        
        self.censoring = censoring
        self.censoring_type = censoring_type
        self.weights = weights
        
        # Initial data analysis
        self.data_summary = self._analyze_data()
        
        # Fitting results
        self.fitted_models = []
        self.results: Optional[FitResults] = None
    
    def _analyze_data(self) -> Dict:
        """
        Initial data analysis
        
        This function:
        - Calculates statistics
        - Identifies outliers
        - Detects distribution shape
        - Explains data characteristics
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
        kurtosis = stats.kurtosis(self.data)
        
        # Outlier detection (IQR method)
        q1 = np.percentile(self.data, 25)
        q3 = np.percentile(self.data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = (self.data < lower_bound) | (self.data > upper_bound)
        n_outliers = np.sum(outliers)
        
        # Skewness interpretation
        if abs(skewness) < 0.5:
            skew_interp = t('data_approximately_symmetric')
        elif skewness > 0:
            skew_interp = t('data_right_skewed')
        else:
            skew_interp = t('data_left_skewed')
        
        # Kurtosis interpretation
        if abs(kurtosis) < 0.5:
            kurt_interp = t('data_near_normal')
        elif kurtosis > 0:
            kurt_interp = t('data_heavy_tailed')
        else:
            kurt_interp = t('data_light_tailed')
        
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
        Suggest appropriate distributions based on data characteristics
        """
        suggestions = []
        reasons = []
        
        ds = self.data_summary
        all_positive = ds['min'] > 0
        is_skewed = abs(ds['skewness']) > 0.5
        is_heavy_tailed = ds['kurtosis'] > 1
        
        # Always try Normal (baseline)
        suggestions.append('normal')
        reasons.append(t('suggest_reason_normal'))
        
        # If all positive
        if all_positive:
            suggestions.append('lognormal')
            reasons.append(t('suggest_reason_lognormal'))
            
            suggestions.append('gamma')
            reasons.append(t('suggest_reason_gamma'))
            
            suggestions.append('weibull')
            reasons.append(t('suggest_reason_weibull'))
            
            suggestions.append('exponential')
            reasons.append(t('suggest_reason_exponential'))
        
        # If skewed
        if is_skewed and ds['skewness'] > 0 and all_positive:
            if 'lognormal' not in suggestions:
                suggestions.append('lognormal')
                reasons.append(t('suggest_reason_skewed'))
        
        # If heavy-tailed
        if is_heavy_tailed:
            reasons.append(t('suggest_reason_heavy_tailed'))
        
        print(f"\n💡 {t('suggested_distributions')}:")
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
        Fit distributions to data
        
        Parameters:
        -----------
        distributions : list of str, optional
            Distribution names. If None, automatically suggested
        method : str
            Estimation method: 'mle', 'moments', 'quantile'
        criterion : str
            Model selection criterion: 'aic', 'bic', 'loo_cv'
        n_jobs : int
            Number of parallel cores (-1 = all)
        verbose : bool
            Show progress
        
        Returns:
        --------
        results : FitResults
            Complete results with explanations
        """
        # If no distributions specified, suggest
        if distributions is None:
            distributions = self.suggest_distributions()
        
        if verbose:
            print(f"\n🚀 {t('fitting', n=len(distributions))} {len(distributions)} {t('distributions')}...")
            print(f"   • {t('estimation_method')}: {method.upper()}")
            print(f"   • {t('selection_criterion')}: {criterion.upper()}")
            print(f"   • {t('num_cores')}: {n_jobs if n_jobs > 0 else t('all')}\n")
        
        # Parallel fitting
        if n_jobs != 1:
            fitted = Parallel(n_jobs=n_jobs)(
                delayed(self._fit_single)(dist_name, method, verbose)
                for dist_name in distributions
            )
        else:
            fitted = []
            iterator = tqdm(distributions, desc=t('fitting', n=0)) if verbose else distributions
            for dist_name in iterator:
                if verbose and not isinstance(iterator, tqdm):
                    print(f"   {t('fitting', n=0)} {dist_name}...", end=" ")
                result = self._fit_single(dist_name, method, verbose=False)
                fitted.append(result)
                if verbose and not isinstance(iterator, tqdm):
                    print("✓")
        
        # Remove models that failed to fit
        self.fitted_models = [f for f in fitted if f is not None]
        
        if len(self.fitted_models) == 0:
            raise RuntimeError(t('error_no_models_fitted'))
        
        # Compare models
        if verbose:
            print(f"\n📊 {t('comparing_models')}...")
        
        model_scores = ModelSelection.compare_models(
            self.data,
            self.fitted_models,
            criterion=criterion
        )
        
        best_model = self.fitted_models[
            [m.info.name for m in self.fitted_models].index(model_scores[0].distribution_name)
        ]
        
        # Diagnostics
        diagnostics = self._run_diagnostics(best_model)
        
        # Recommendations
        recommendations = self._generate_recommendations(model_scores, diagnostics)
        
        # Build results
        self.results = FitResults(
            data_summary=self.data_summary,
            fitted_models=self.fitted_models,
            model_scores=model_scores,
            best_model=best_model,
            diagnostics=diagnostics,
            recommendations=recommendations,
            _data=self.data.copy()
        )
        
        if verbose:
            print(f"\n✅ {t('fit_complete')}!\n")
        
        return self.results
    
    def _fit_single(
        self,
        dist_name: str,
        method: str,
        verbose: bool = False
    ) -> Optional[BaseDistribution]:
        """Fit a single distribution"""
        try:
            dist = get_distribution(dist_name)
            dist.fit(self.data, method=method)
            return dist
        except Exception as e:
            if verbose:
                warnings.warn(f"Fitting {dist_name} failed: {str(e)}")
            return None
    
    def _run_diagnostics(self, model: BaseDistribution) -> Dict:
        """Run basic diagnostics"""
        notes = []
        
        # KS test
        ks_stat, ks_pval = stats.kstest(self.data, model.cdf)
        if ks_pval < 0.05:
            notes.append(t('diagnostics_warning_ks', p=ks_pval))
        
        # Residual analysis
        theoretical_quantiles = model.ppf(np.linspace(0.01, 0.99, len(self.data)))
        empirical_quantiles = np.sort(self.data)
        residuals = empirical_quantiles - theoretical_quantiles
        
        if np.max(np.abs(residuals)) > 2 * self.data_summary['std']:
            notes.append(t('diagnostics_warning_residuals'))
        
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
        """Generate recommendations"""
        recs = []
        
        # If multiple models are close
        if len(scores) > 1 and scores[1].score - scores[0].score < 2:
            recs.append(
                t('recommendation_close_models', 
                  model1=scores[0].distribution_name, 
                  model2=scores[1].distribution_name)
            )
        
        # If KS is significant
        if diagnostics['ks_pval'] < 0.05:
            recs.append(t('recommendation_ks_significant'))
        
        # Suggest bootstrap
        recs.append(t('recommendation_bootstrap'))
        
        return recs
