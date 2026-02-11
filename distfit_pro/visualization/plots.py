"""
Visualization Module with i18n Support and RTL Fix
===================================================

نمودارهای تشخیصی و مقایسه‌ای برای توزیع‌ها با پشتیبانی چندزبانه و RTL

This module includes:
- P-P and Q-Q plots
- PDF and CDF comparison  
- Residual analysis
- Interactive plots with Plotly
- Full multilingual support (en/fa/de) with proper RTL rendering
"""

import warnings
from typing import List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from ..locales import t
from ..core.config import get_language

# RTL support for Persian/Arabic
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    RTL_AVAILABLE = True
except ImportError:
    RTL_AVAILABLE = False
    warnings.warn(
        "arabic_reshaper and python-bidi not installed. "
        "Persian/Arabic text in plots may not render correctly.\n"
        "Install with: pip install arabic-reshaper python-bidi"
    )

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    warnings.warn("Plotly not installed. Interactive plots will not be available.")


def fix_rtl_text(text: str, prefix: str = "") -> str:
    """
    Fix RTL (Right-to-Left) text rendering for Persian/Arabic in matplotlib
    
    Parameters:
    -----------
    text : str
        Input text (may contain Persian/Arabic)
    prefix : str
        Prefix to add AFTER RTL processing (e.g., emoji)
        This prevents emoji from interfering with RTL algorithm
        
    Returns:
    --------
    fixed_text : str
        Text ready for matplotlib rendering
        
    Example:
    --------
    >>> text = "نمودار مقایسه"
    >>> fixed = fix_rtl_text(text, prefix="📊 ")
    >>> plt.title(fixed)  # Now renders correctly!
    """
    # Only apply RTL fix for Persian language
    if get_language() != 'fa' or not RTL_AVAILABLE:
        return prefix + text
    
    try:
        # Reshape Arabic/Persian characters
        reshaped = arabic_reshaper.reshape(text)
        # Apply bidi algorithm
        bidi_text = get_display(reshaped)
        # Add prefix AFTER RTL processing
        return prefix + bidi_text
    except Exception as e:
        warnings.warn(f"RTL text processing failed: {e}. Using original text.")
        return prefix + text


class DistributionPlotter:
    """
    Professional plotting class with multilingual support and RTL fix
    
    Supports all standard diagnostic plots:
    - Comparison plots (PDF, CDF, P-P, Q-Q)
    - Diagnostic plots (residuals, tail behavior)
    - Interactive dashboards (Plotly)
    
    All plot labels and titles are automatically translated based on
    the current language setting (set via set_language()).
    
    Persian/Arabic text is automatically fixed for proper RTL rendering.
    """
    
    def __init__(self, data: np.ndarray, fitted_models: List, best_model=None):
        self.data = np.asarray(data)
        self.fitted_models = fitted_models
        self.best_model = best_model if best_model else fitted_models[0]
        self.sorted_data = np.sort(self.data)
        self.n = len(self.data)
        self.empirical_cdf = np.arange(1, self.n + 1) / self.n
        
    def plot_comparison(self, figsize: Tuple[int, int] = (14, 10), show_top_n: int = 3) -> Figure:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        x_min, x_max = self.data.min(), self.data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        
        models_to_plot = self.fitted_models[:show_top_n]
        colors = plt.cm.Set2(np.linspace(0, 1, len(models_to_plot)))
        
        # 1. Histogram + PDF
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.hist(self.data, bins=30, density=True, alpha=0.6, 
                 color='skyblue', edgecolor='black', label=fix_rtl_text(t('data')))
        
        for i, (model, color) in enumerate(zip(models_to_plot, colors)):
            label = f"{model.info.display_name} (#{i+1})"
            linewidth = 3 if model == self.best_model else 1.5
            linestyle = '-' if model == self.best_model else '--'
            ax1.plot(x, model.pdf(x), color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label)
        
        ax1.set_xlabel(fix_rtl_text(t('value')), fontsize=11)
        ax1.set_ylabel(fix_rtl_text(t('density')), fontsize=11)
        ax1.set_title(fix_rtl_text(t('pdf_comparison'), prefix="📊 "), fontsize=13, fontweight='bold')
        ax1.legend(loc='best', fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. CDF
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.sorted_data, self.empirical_cdf, 'o', 
                 color='steelblue', alpha=0.4, markersize=4, label=fix_rtl_text(t('empirical_cdf')))
        
        for i, (model, color) in enumerate(zip(models_to_plot, colors)):
            label = f"{model.info.display_name} (#{i+1})"
            linewidth = 3 if model == self.best_model else 1.5
            linestyle = '-' if model == self.best_model else '--'
            ax2.plot(self.sorted_data, model.cdf(self.sorted_data), 
                    color=color, linewidth=linewidth, linestyle=linestyle, label=label)
        
        ax2.set_xlabel(fix_rtl_text(t('value')), fontsize=11)
        ax2.set_ylabel(fix_rtl_text(t('cumulative_probability')), fontsize=11)
        ax2.set_title(fix_rtl_text(t('cdf_comparison'), prefix="📈 "), fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Q-Q Plot
        ax3 = fig.add_subplot(gs[1, 0])
        theoretical_quantiles = self.best_model.ppf(np.linspace(0.01, 0.99, self.n))
        
        ax3.scatter(theoretical_quantiles, self.sorted_data, 
                   alpha=0.6, s=30, color='darkorange', edgecolors='black', linewidth=0.5)
        
        min_val = min(theoretical_quantiles.min(), self.sorted_data.min())
        max_val = max(theoretical_quantiles.max(), self.sorted_data.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 
                'r--', linewidth=2, label=fix_rtl_text(t('perfect_fit')))
        
        ax3.set_xlabel(fix_rtl_text(t('theoretical_quantiles')), fontsize=11)
        ax3.set_ylabel(fix_rtl_text(t('empirical_quantiles')), fontsize=11)
        # Q-Q plot title with model name
        title_text = f"{t('qq_plot')} ({self.best_model.info.display_name})"
        ax3.set_title(fix_rtl_text(title_text, prefix="📐 "), fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)
        
        # 4. P-P Plot
        ax4 = fig.add_subplot(gs[1, 1])
        theoretical_probs = self.best_model.cdf(self.sorted_data)
        
        ax4.scatter(theoretical_probs, self.empirical_cdf,
                   alpha=0.6, s=30, color='mediumseagreen', edgecolors='black', linewidth=0.5)
        ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label=fix_rtl_text(t('perfect_fit')))
        
        ax4.set_xlabel(fix_rtl_text(t('theoretical_probabilities')), fontsize=11)
        ax4.set_ylabel(fix_rtl_text(t('empirical_probabilities')), fontsize=11)
        # P-P plot title with model name
        title_text = f"{t('pp_plot')} ({self.best_model.info.display_name})"
        ax4.set_title(fix_rtl_text(title_text, prefix="📉 "), fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        fig.suptitle(fix_rtl_text(t('comparison_plots')), fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def plot_diagnostics(self, figsize: Tuple[int, int] = (14, 10)) -> Figure:
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        model = self.best_model
        theoretical_quantiles = model.ppf(np.linspace(0.01, 0.99, self.n))
        residuals = self.sorted_data - theoretical_quantiles
        standardized_residuals = residuals / np.std(residuals)
        
        # 1. Residual Plot
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(theoretical_quantiles, residuals, alpha=0.6, s=30,
                   color='steelblue', edgecolors='black', linewidth=0.5)
        ax1.axhline(0, color='red', linestyle='--', linewidth=2, label=fix_rtl_text(t('zero_line')))
        
        std_res = np.std(residuals)
        ax1.axhline(2*std_res, color='orange', linestyle=':', linewidth=1.5, label='±2σ')
        ax1.axhline(-2*std_res, color='orange', linestyle=':', linewidth=1.5)
        
        ax1.set_xlabel(fix_rtl_text(t('theoretical_quantiles')), fontsize=11)
        ax1.set_ylabel(fix_rtl_text(t('residuals')), fontsize=11)
        ax1.set_title(fix_rtl_text(t('residual_plot'), prefix="🔍 "), fontsize=13, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Standardized Residuals
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(standardized_residuals, bins=20, density=True, 
                alpha=0.6, color='lightcoral', edgecolor='black')
        
        x_norm = np.linspace(-3, 3, 100)
        from scipy.stats import norm
        ax2.plot(x_norm, norm.pdf(x_norm), 'r-', linewidth=2, label='N(0,1)')
        
        ax2.set_xlabel(fix_rtl_text(t('standardized_residuals')), fontsize=11)
        ax2.set_ylabel(fix_rtl_text(t('density')), fontsize=11)
        ax2.set_title(fix_rtl_text(t('residual_distribution'), prefix="📊 "), fontsize=13, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)
        
        # 3. Tail Behavior
        ax3 = fig.add_subplot(gs[1, 0])
        empirical_survival = 1 - self.empirical_cdf
        theoretical_survival = 1 - model.cdf(self.sorted_data)
        mask = (empirical_survival > 1e-10) & (theoretical_survival > 1e-10)
        
        ax3.semilogy(self.sorted_data[mask], empirical_survival[mask], 
                    'o', alpha=0.5, markersize=5, label=fix_rtl_text(t('empirical')), color='navy')
        ax3.semilogy(self.sorted_data[mask], theoretical_survival[mask],
                    '-', linewidth=2, label=fix_rtl_text(t('fitted')), color='red')
        
        ax3.set_xlabel(fix_rtl_text(t('value')), fontsize=11)
        ax3.set_ylabel('P(X > x) [log scale]', fontsize=11)
        ax3.set_title(fix_rtl_text(t('tail_behavior'), prefix="📉 "), fontsize=13, fontweight='bold')
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3, which='both')
        
        # 4. Influence Plot
        ax4 = fig.add_subplot(gs[1, 1])
        pdf_values = model.pdf(self.sorted_data)
        influence = np.abs(residuals) * pdf_values
        
        ax4.scatter(theoretical_quantiles, influence, alpha=0.6, s=30,
                   color='purple', edgecolors='black', linewidth=0.5)
        
        threshold = np.percentile(influence, 95)
        high_influence = influence > threshold
        ax4.scatter(theoretical_quantiles[high_influence], influence[high_influence],
                   s=100, color='red', alpha=0.7, marker='x', linewidth=2,
                   label=fix_rtl_text(t('high_influence', threshold=threshold)))
        ax4.axhline(threshold, color='red', linestyle='--', linewidth=1.5)
        
        ax4.set_xlabel(fix_rtl_text(t('theoretical_quantiles')), fontsize=11)
        ax4.set_ylabel(fix_rtl_text(t('influence')), fontsize=11)
        ax4.set_title(fix_rtl_text(t('influence_plot'), prefix="⚠️ "), fontsize=13, fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)
        
        # Main title with model name
        title_text = f"{t('diagnostic_plots')} - {model.info.display_name}"
        fig.suptitle(fix_rtl_text(title_text), fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        return fig
    
    def plot_interactive(self, show_top_n: int = 3):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not installed. Install with: pip install plotly")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(t('pdf_comparison'), t('cdf_comparison'), t('qq_plot'), t('pp_plot')),
            specs=[[{}, {}], [{}, {}]]
        )
        
        x_min, x_max = self.data.min(), self.data.max()
        x_range = x_max - x_min
        x = np.linspace(x_min - 0.1*x_range, x_max + 0.1*x_range, 200)
        models_to_plot = self.fitted_models[:show_top_n]
        
        # 1. PDF
        fig.add_trace(go.Histogram(x=self.data, histnorm='probability density',
                                   name=t('data'), opacity=0.6, marker_color='lightblue'), row=1, col=1)
        for i, model in enumerate(models_to_plot):
            fig.add_trace(go.Scatter(x=x, y=model.pdf(x), mode='lines',
                                     name=f"{model.info.display_name} (#{i+1})",
                                     line=dict(width=3 if model == self.best_model else 2)), row=1, col=1)
        
        # 2. CDF
        fig.add_trace(go.Scatter(x=self.sorted_data, y=self.empirical_cdf, mode='markers', 
                                 name=t('empirical_cdf'), marker=dict(size=5, opacity=0.5)), row=1, col=2)
        for i, model in enumerate(models_to_plot):
            fig.add_trace(go.Scatter(x=self.sorted_data, y=model.cdf(self.sorted_data), mode='lines',
                                     name=f"{model.info.display_name} (#{i+1})",
                                     line=dict(width=3 if model == self.best_model else 2)), row=1, col=2)
        
        # 3. Q-Q
        theoretical_q = self.best_model.ppf(np.linspace(0.01, 0.99, self.n))
        fig.add_trace(go.Scatter(x=theoretical_q, y=self.sorted_data, mode='markers',
                                 name=t('data_points'), marker=dict(size=6, color='orange')), row=2, col=1)
        min_val = min(theoretical_q.min(), self.sorted_data.min())
        max_val = max(theoretical_q.max(), self.sorted_data.max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                                 name=t('perfect_fit'), line=dict(dash='dash', color='red', width=2)), row=2, col=1)
        
        # 4. P-P
        theoretical_p = self.best_model.cdf(self.sorted_data)
        fig.add_trace(go.Scatter(x=theoretical_p, y=self.empirical_cdf, mode='markers',
                                 name=t('data_points'), marker=dict(size=6, color='green')), row=2, col=2)
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name=t('perfect_fit'),
                                 line=dict(dash='dash', color='red', width=2)), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True,
                         title_text=t('interactive_dashboard', model=self.best_model.info.display_name),
                         title_font_size=16)
        
        fig.update_xaxes(title_text=t('value'), row=1, col=1)
        fig.update_xaxes(title_text=t('value'), row=1, col=2)
        fig.update_xaxes(title_text=t('theoretical_quantiles'), row=2, col=1)
        fig.update_xaxes(title_text=t('theoretical_probabilities'), row=2, col=2)
        fig.update_yaxes(title_text=t('density'), row=1, col=1)
        fig.update_yaxes(title_text='CDF', row=1, col=2)
        fig.update_yaxes(title_text=t('empirical_quantiles'), row=2, col=1)
        fig.update_yaxes(title_text=t('empirical_probabilities'), row=2, col=2)
        
        return fig
