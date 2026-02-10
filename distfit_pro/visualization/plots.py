"""
Visualization Module
====================

نمودارهای تشخیصی و مقایسه‌ای برای توزیع‌ها

این ماژول شامل:
- P-P و Q-Q plots
- مقایسه PDF و CDF
- Residual analysis
- نمودارهای interactive با Plotly
"""

import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive plots will not be available.")


class DistributionPlotter:
    """
    کلاس اصلی برای رسم نمودارها
    
    این کلاس همه انواع نمودارهای تشخیصی را رسم می‌کند:
    - Comparison plots (PDF, CDF, P-P, Q-Q)
    - Diagnostic plots (residuals, tail behavior)
    - Interactive dashboards
    """
    
    def __init__(self, data: np.ndarray, fitted_models: List, best_model=None):
        """
        Parameters:
        -----------
        data : array-like
            داده اصلی
        fitted_models : list
            لیست مدل‌های فیت‌شده
        best_model : BaseDistribution, optional
            بهترین مدل (اگر مشخص باشد)
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
        رسم نمودار مقایسه‌ای شامل PDF, CDF, P-P, Q-Q
        
        Parameters:
        -----------
        figsize : tuple
            اندازه figure
        show_top_n : int
            تعداد بهترین مدل‌ها برای نمایش
        
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # محدوده x برای نمودارها
        x_min, x_max = self.data.min(), self.data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        
        # انتخاب مدل‌ها برای نمایش
        models_to_plot = self.fitted_models[:show_top_n]
        colors = plt.cm.Set2(np.linspace(0, 1, len(models_to_plot)))
        
        # 1. Histogram + PDF
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.data, bins=30, density=True, alpha=0.6, 
                 color='skyblue', edgecolor='black', label='Data')
        
        for i, (model, color) in enumerate(zip(models_to_plot, colors)):
            label = f"{model.info.display_name} (#{i+1})"
            linewidth = 3 if model == self.best_model else 1.5
            linestyle = '-' if model == self.best_model else '--'
            ax1.plot(x, model.pdf(x), color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label)
        
        ax1.set_xlabel('Value', fontsize=11)
        ax1.set_ylabel('Density', fontsize=11)
        ax1.set_title('📊 PDF Comparison', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. CDF Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.sorted_data, self.empirical_cdf, 'o', 
                 color='steelblue', alpha=0.4, markersize=4, label='Empirical CDF')
        
        for i, (model, color) in enumerate(zip(models_to_plot, colors)):
            label = f"{model.info.display_name} (#{i+1})"
            linewidth = 3 if model == self.best_model else 1.5
            linestyle = '-' if model == self.best_model else '--'
            ax2.plot(self.sorted_data, model.cdf(self.sorted_data), 
                    color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        
        ax2.set_xlabel('Value', fontsize=11)
        ax2.set_ylabel('Cumulative Probability', fontsize=11)
        ax2.set_title('📈 CDF Comparison', fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Q-Q Plot (فقط بهترین مدل)
        ax3 = fig.add_subplot(gs[1, 0])
        theoretical_quantiles = self.best_model.ppf(
            np.linspace(0.01, 0.99, self.n)
        )
        
        ax3.scatter(theoretical_quantiles, self.sorted_data, 
                   alpha=0.6, s=30, color='darkorange', edgecolors='black', linewidth=0.5)
        
        # خط 45 درجه
        min_val = min(theoretical_quantiles.min(), self.sorted_data.min())
        max_val = max(theoretical_quantiles.max(), self.sorted_data.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label='Perfect fit')
        
        ax3.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax3.set_ylabel('Empirical Quantiles', fontsize=11)
        ax3.set_title(f'📐 Q-Q Plot ({self.best_model.info.display_name})', 
                     fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. P-P Plot (فقط بهترین مدل)
        ax4 = fig.add_subplot(gs[1, 1])
        theoretical_probs = self.best_model.cdf(self.sorted_data)
        
        ax4.scatter(theoretical_probs, self.empirical_cdf,
                   alpha=0.6, s=30, color='mediumseagreen', edgecolors='black', linewidth=0.5)
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')
        
        ax4.set_xlabel('Theoretical Probabilities', fontsize=11)
        ax4.set_ylabel('Empirical Probabilities', fontsize=11)
        ax4.set_title(f'📉 P-P Plot ({self.best_model.info.display_name})', 
                     fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        fig.suptitle('Distribution Fitting - Comparison Plots', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_diagnostics(
        self,
        figsize: Tuple[int, int] = (14, 10)
    ) -> Figure:
        """
        رسم نمودارهای تشخیصی: residuals, tail behavior, etc.
        
        Parameters:
        -----------
        figsize : tuple
            اندازه figure
        
        Returns:
        --------
        fig : matplotlib Figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        model = self.best_model
        
        # محاسبه residuals
        theoretical_quantiles = model.ppf(np.linspace(0.01, 0.99, self.n))
        residuals = self.sorted_data - theoretical_quantiles
        standardized_residuals = residuals / np.std(residuals)
        
        # 1. Residual Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(theoretical_quantiles, residuals, alpha=0.6, s=30,
                   color='steelblue', edgecolors='black', linewidth=0.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, label='Zero line')
        
        # خطوط ±2σ
        std_res = np.std(residuals)
        ax1.axhline(2*std_res, color='orange', linestyle=':', linewidth=1.5, label='±2σ')
        ax1.axhline(-2*std_res, color='orange', linestyle=':', linewidth=1.5)
        
        ax1.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax1.set_ylabel('Residuals', fontsize=11)
        ax1.set_title('🔍 Residual Plot', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Standardized Residuals
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(standardized_residuals, bins=20, density=True, 
                alpha=0.6, color='lightcoral', edgecolor='black')
        
        # نرمال استاندارد برای مقایسه
        x_norm = np.linspace(-3, 3, 100)
        from scipy.stats import norm
        ax2.plot(x_norm, norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
        
        ax2.set_xlabel('Standardized Residuals', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('📊 Residual Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Tail Behavior (log scale)
        ax3 = fig.add_subplot(gs[1, 0])
        
        # Survival function (1 - CDF)
        empirical_survival = 1 - self.empirical_cdf
        theoretical_survival = 1 - model.cdf(self.sorted_data)
        
        # حذف صفرها برای log
        mask = (empirical_survival > 1e-10) & (theoretical_survival > 1e-10)
        
        ax3.semilogy(self.sorted_data[mask], empirical_survival[mask], 
                    'o', alpha=0.5, markersize=5, label='Empirical', color='navy')
        ax3.semilogy(self.sorted_data[mask], theoretical_survival[mask],
                    '-', linewidth=2, label='Fitted', color='red')
        
        ax3.set_xlabel('Value', fontsize=11)
        ax3.set_ylabel('P(X > x) [log scale]', fontsize=11)
        ax3.set_title('📉 Tail Behavior (Survival Function)', fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, which='both')
        
        # 4. Leverage plot (Cook's distance style)
        ax4 = fig.add_subplot(gs[1, 1])
        
        # محاسبه "influence" ساده: residual × probability density
        pdf_values = model.pdf(self.sorted_data)
        influence = np.abs(residuals) * pdf_values
        
        ax4.scatter(theoretical_quantiles, influence, alpha=0.6, s=30,
                   color='purple', edgecolors='black', linewidth=0.5)
        
        # شناسایی نقاط پرنفوذ
        threshold = np.percentile(influence, 95)
        high_influence = influence > threshold
        ax4.scatter(theoretical_quantiles[high_influence], 
                   influence[high_influence],
                   s=100, color='red', alpha=0.7, marker='x', linewidth=2,
                   label=f'High influence (>{threshold:.3f})')
        
        ax4.axhline(threshold, color='red', linestyle='--', linewidth=1.5)
        
        ax4.set_xlabel('Theoretical Quantiles', fontsize=11)
        ax4.set_ylabel('Influence', fontsize=11)
        ax4.set_title('⚠️ Influence Plot', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        fig.suptitle(f'Diagnostic Plots - {model.info.display_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        return fig
    
    def plot_interactive(
        self,
        show_top_n: int = 3
    ):
        """
        رسم نمودار interactive با Plotly
        
        Parameters:
        -----------
        show_top_n : int
            تعداد بهترین مدل‌ها
        
        Returns:
        --------
        fig : plotly Figure
        """
        if not PLOTLY_AVAILABLE:
            raise ImportError(
                "Plotly is not installed. Install with: pip install plotly"
            )
        
        # ساخت subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PDF Comparison', 'CDF Comparison', 
                          'Q-Q Plot', 'P-P Plot'),
            specs=[[{}, {}], [{}, {}]]
        )
        
        x_min, x_max = self.data.min(), self.data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        
        models_to_plot = self.fitted_models[:show_top_n]
        
        # 1. Histogram + PDF
        fig.add_trace(
            go.Histogram(x=self.data, histnorm='probability density',
                        name='Data', opacity=0.6, marker_color='lightblue'),
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
                      mode='markers', name='Empirical CDF',
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
                      mode='markers', name='Data points',
                      marker=dict(size=6, color='orange')),
            row=2, col=1
        )
        
        min_val = min(theoretical_q.min(), self.sorted_data.min())
        max_val = max(theoretical_q.max(), self.sorted_data.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                      mode='lines', name='Perfect fit',
                      line=dict(dash='dash', color='red', width=2)),
            row=2, col=1
        )
        
        # 4. P-P Plot
        theoretical_p = self.best_model.cdf(self.sorted_data)
        
        fig.add_trace(
            go.Scatter(x=theoretical_p, y=self.empirical_cdf,
                      mode='markers', name='Data points',
                      marker=dict(size=6, color='green')),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1],
                      mode='lines', name='Perfect fit',
                      line=dict(dash='dash', color='red', width=2)),
            row=2, col=2
        )
        
        # Layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=f"Interactive Distribution Fitting Dashboard - Best: {self.best_model.info.display_name}",
            title_font_size=16
        )
        
        fig.update_xaxes(title_text="Value", row=1, col=1)
        fig.update_xaxes(title_text="Value", row=1, col=2)
        fig.update_xaxes(title_text="Theoretical Quantiles", row=2, col=1)
        fig.update_xaxes(title_text="Theoretical Probabilities", row=2, col=2)
        
        fig.update_yaxes(title_text="Density", row=1, col=1)
        fig.update_yaxes(title_text="CDF", row=1, col=2)
        fig.update_yaxes(title_text="Empirical Quantiles", row=2, col=1)
        fig.update_yaxes(title_text="Empirical Probabilities", row=2, col=2)
        
        return fig
