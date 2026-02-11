"""
Visualization Module with i18n Support
=======================================

نمودارهای تشخیصی و مقایسه‌ای برای توزیع‌ها با پشتیبانی چندزبانه

This module includes:
- P-P and Q-Q plots
- PDF and CDF comparison
- Residual analysis
- Interactive plots with Plotly
- Full multilingual support (en/fa/de)
"""

import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from ..locales import t

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive plots will not be available.")


class DistributionPlotter:
    """
    Professional plotting class with multilingual support
    
    Supports all standard diagnostic plots:
    - Comparison plots (PDF, CDF, P-P, Q-Q)
    - Diagnostic plots (residuals, tail behavior)
    - Interactive dashboards (Plotly)
    
    All plot labels and titles are automatically translated based on
    the current language setting (set via set_language()).
    """
    
    def __init__(self, data: np.ndarray, fitted_models: List, best_model=None):
        """
        Parameters:
        -----------
        data : array-like
            Original data
        fitted_models : list
            List of fitted distribution models
        best_model : BaseDistribution, optional
            Best fitting model (defaults to first in list)
        """
        self.data = np.asarray(data)
        self.fitted_models = fitted_models
        self.best_model = best_model if best_model else fitted_models[0]
        
        # Pre-compute useful values
        self.sorted_data = np.sort(self.data)
        self.n = len(self.data)
        self.empirical_cdf = np.arange(1, self.n + 1) / self.n
        
    def plot_comparison(
        self,
        figsize: Tuple[int, int] = (14, 10),
        show_top_n: int = 3
    ) -> Figure:
        """
        Plot comparison charts (PDF, CDF, P-P, Q-Q) with i18n
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        show_top_n : int
            Number of best models to show
        
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # X range for plots
        x_min, x_max = self.data.min(), self.data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        
        # Select models to plot
        models_to_plot = self.fitted_models[:show_top_n]
        colors = plt.cm.Set2(np.linspace(0, 1, len(models_to_plot)))
        
        # 1. Histogram + PDF
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.data, bins=30, density=True, alpha=0.6, 
                 color='skyblue', edgecolor='black', label=t('data'))
        
        for i, (model, color) in enumerate(zip(models_to_plot, colors)):
            label = f"{model.info.display_name} (#{i+1})"
            linewidth = 3 if model == self.best_model else 1.5
            linestyle = '-' if model == self.best_model else '--'
            ax1.plot(x, model.pdf(x), color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label)
        
        ax1.set_xlabel(t('value'), fontsize=11)
        ax1.set_ylabel(t('density'), fontsize=11)
        ax1.set_title(f"📊 {t('pdf_comparison')}", fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. CDF Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.sorted_data, self.empirical_cdf, 'o', 
                 color='steelblue', alpha=0.4, markersize=4, label=t('empirical_cdf'))
        
        for i, (model, color) in enumerate(zip(models_to_plot, colors)):
            label = f"{model.info.display_name} (#{i+1})"
            linewidth = 3 if model == self.best_model else 1.5
            linestyle = '-' if model == self.best_model else '--'
            ax2.plot(self.sorted_data, model.cdf(self.sorted_data), 
                    color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        
        ax2.set_xlabel(t('value'), fontsize=11)
        ax2.set_ylabel(t('cumulative_probability'), fontsize=11)
        ax2.set_title(f"📈 {t('cdf_comparison')}", fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Q-Q Plot (best model only)
        ax3 = fig.add_subplot(gs[1, 0])
        theoretical_quantiles = self.best_model.ppf(
            np.linspace(0.01, 0.99, self.n)
        )
        
        ax3.scatter(theoretical_quantiles, self.sorted_data, 
                   alpha=0.6, s=30, color='darkorange', edgecolors='black', linewidth=0.5)
        
        # 45-degree line
        min_val = min(theoretical_quantiles.min(), self.sorted_data.min())
        max_val = max(theoretical_quantiles.max(), self.sorted_data.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label=t('perfect_fit'))
        
        ax3.set_xlabel(t('theoretical_quantiles'), fontsize=11)
        ax3.set_ylabel(t('empirical_quantiles'), fontsize=11)
        ax3.set_title(f"📐 {t('qq_plot')} ({self.best_model.info.display_name})", 
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. P-P Plot (best model only)
        ax4 = fig.add_subplot(gs[1, 1])
        theoretical_probs = self.best_model.cdf(self.sorted_data)
        
        ax4.scatter(theoretical_probs, self.empirical_cdf,
                   alpha=0.6, s=30, color='mediumseagreen', edgecolors='black', linewidth=0.5)
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label=t('perfect_fit'))
        
        ax4.set_xlabel(t('theoretical_probabilities'), fontsize=11)
        ax4.set_ylabel(t('empirical_probabilities'), fontsize=11)
        ax4.set_title(f"📉 {t('pp_plot')} ({self.best_model.info.display_name})", 
                     fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        fig.suptitle(t('comparison_plots'), 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_diagnostics(
        self,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Figure:
        """
        Plot diagnostic charts with i18n support
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
        
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        model = self.best_model
        
        # Calculate residuals
        theoretical_quantiles = model.ppf(np.linspace(0.01, 0.99, self.n))
        residuals = self.sorted_data - theoretical_quantiles
        standardized_residuals = residuals / np.std(residuals)
        
        # 1. Residual Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(theoretical_quantiles, residuals, alpha=0.6, s=30,
                   color='steelblue', edgecolors='black', linewidth=0.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, label=t('zero_line'))
        
        # ±2σ lines
        std_res = np.std(residuals)
        ax1.axhline(2*std_res, color='orange', linestyle=':', linewidth=1.5, label='±2σ')
        ax1.axhline(-2*std_res, color='orange', linestyle=':', linewidth=1.5)
        
        ax1.set_xlabel(t('theoretical_quantiles'), fontsize=11)
        ax1.set_ylabel(t('residuals'), fontsize=11)
        ax1.set_title(f"🔍 {t('residual_plot')}", fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Standardized Residuals
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(standardized_residuals, bins=20, density=True, 
                alpha=0.6, color='lightcoral', edgecolor='black')
        
        # Standard normal for comparison
        x_norm = np.linspace(-3, 3, 100)
        from scipy.stats import norm
        ax2.plot(x_norm, norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
        
        ax2.set_xlabel(t('standardized_residuals'), fontsize=11)
        ax2.set_ylabel(t('density'), fontsize=11)
        ax2.set_title(f"📊 {t('residual_distribution')}", fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Tail Behavior (log scale)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Survival function (1 - CDF)
        empirical_survival = 1 - self.empirical_cdf
        theoretical_survival = 1 - model.cdf(self.sorted_data)
        
        # Remove zeros for log
        mask = (empirical_survival > 1e-10) & (theoretical_survival > 1e-10)
        
        ax3.semilogy(self.sorted_data[mask], empirical_survival[mask], 
                    'o', alpha=0.5, markersize=5, label=t('empirical'), color='navy')
        ax3.semilogy(self.sorted_data[mask], theoretical_survival[mask],
                    '-', linewidth=2, label=t('fitted'), color='red')
        
        ax3.set_xlabel(t('value'), fontsize=11)
        ax3.set_ylabel('P(X > x) [log scale]', fontsize=11)
        ax3.set_title(f"📉 {t('tail_behavior')}", fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, which='both')
        
        # 4. Leverage plot
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Simple "influence": residual × probability density
        pdf_values = model.pdf(self.sorted_data)
        influence = np.abs(residuals) * pdf_values
        
        ax4.scatter(theoretical_quantiles, influence, alpha=0.6, s=30,
                   color='purple', edgecolors='black', linewidth=0.5)
        
        # Identify high-influence points
        threshold = np.percentile(influence, 95)
        high_influence = influence > threshold
        ax4.scatter(theoretical_quantiles[high_influence], 
                   influence[high_influence],
                   s=100, color='red', alpha=0.7, marker='x', linewidth=2,
                   label=t('high_influence', threshold=threshold))
        
        ax4.axhline(threshold, color='red', linestyle='--', linewidth=1.5)
        
        ax4.set_xlabel(t('theoretical_quantiles'), fontsize=11)
        ax4.set_ylabel(t('influence'), fontsize=11)
        ax4.set_title(f"⚠️ {t('influence_plot')}", fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        fig.suptitle(f"{t('diagnostic_plots')} - {model.info.display_name}", 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_interactive(
        self,
        show_top_n: int = 3
    ):
        """
        Create interactive plot with Plotly and i18n
        
        Parameters:
        -----------
        show_top_n : int
            Number of best models
        
        Returns:
        --------
        fig : plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is not installed. Install with: pip install plotly"
            )
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(t('pdf_comparison'), t('cdf_comparison'), 
                          t('qq_plot'), t('pp_plot')),
            specs=[[{}, {}], [{}, {}]]
        )
        
        x_min, x_max = self.data.min(), self.data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        
        models_to_plot = self.fitted_models[:show_top_n]
        
        # 1. Histogram + PDF
        fig.add_trace(
            go.Histogram(x=self.data, histnorm='probability density',
                        name=t('data'), opacity=0.6, marker_color='lightblue'),
            row=1, col=1
        )
        
        for i, model in enumerate(models_to_plot):
            fig.add_trace(
                go.Scatter(x=x, y=model.pdf(x), mode='lines',
                          name=f"{model.info.display_name} (#{i+1})",
                          line=dict(width=3 if model == self.best_model else 2)),
                row=1, col=1
            )
        
        # 2. CDF
        fig.add_trace(
            go.Scatter(x=self.sorted_data, y=self.empirical_cdf,
                      mode='markers', name=t('empirical_cdf'),
                      marker=dict(size=5, opacity=0.5)),
            row=1, col=2
        )
        
        for i, model in enumerate(models_to_plot):
            fig.add_trace(
                go.Scatter(x=self.sorted_data, y=model.cdf(self.sorted_data),
                          mode='lines', name=f"{model.info.display_name} (#{i+1})",
                          line=dict(width=3 if model == self.best_model else 2)),
                row=1, col=2
            )
        
        # 3. Q-Q Plot
        theoretical_q = self.best_model.ppf(np.linspace(0.01, 0.99, self.n))
        
        fig.add_trace(
            go.Scatter(x=theoretical_q, y=self.sorted_data,
                      mode='markers', name=t('data_points'),
                      marker=dict(size=6, color='orange')),
            row=2, col=1
        )
        
        min_val = min(theoretical_q.min(), self.sorted_data.min())
        max_val = max(theoretical_q.max(), self.sorted_data.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name=t('perfect_fit'),
                      line=dict(dash='dash', color='red', width=2)),
            row=2, col=1
        )
        
        # 4. P-P Plot
        theoretical_p = self.best_model.cdf(self.sorted_data)
        
        fig.add_trace(
            go.Scatter(x=theoretical_p, y=self.empirical_cdf,
                      mode='markers', name=t('data_points'),
                      marker=dict(size=6, color='green')),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1],
                      mode='lines', name=t('perfect_fit'),
                      line=dict(dash='dash', color='red', width=2)),
            row=2, col=2
        )
        
        # Layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=t('interactive_dashboard', model=self.best_model.info.display_name),
            title_font_size=16
        )
        
        fig.update_xaxes(title_text=t('value'), row=1, col=1)
        fig.update_xaxes(title_text=t('value'), row=1, col=2)
        fig.update_xaxes(title_text=t('theoretical_quantiles'), row=2, col=1)
        fig.update_xaxes(title_text=t('theoretical_probabilities'), row=2, col=2)
        
        fig.update_yaxes(title_text=t('density'), row=1, col=1)
        fig.update_yaxes(title_text='CDF', row=1, col=2)
        fig.update_yaxes(title_text=t('empirical_quantiles'), row=2, col=1)
        fig.update_yaxes(title_text=t('empirical_probabilities'), row=2, col=2)
        
        return fig
