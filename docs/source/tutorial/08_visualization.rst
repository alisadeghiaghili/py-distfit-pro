Tutorial 8: Visualization
=========================

Learn to create publication-quality plots.

Basic Plotting
--------------

**Quick plot of fitted distribution:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.plotting import plot_fit
    import numpy as np
    
    # Generate and fit data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Plot
    plot_fit(data, dist, title="Normal Distribution Fit")

This creates a figure with:
- Histogram of data
- Fitted PDF curve
- Parameter estimates

Custom Histogram + PDF
----------------------

**Full control over plotting:**

.. code-block:: python

    import matplotlib.pyplot as plt
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Histogram
    plt.hist(data, bins=50, density=True, alpha=0.7, 
             color='skyblue', edgecolor='black', label='Data')
    
    # Fitted PDF
    x = np.linspace(data.min(), data.max(), 1000)
    pdf = dist.pdf(x)
    plt.plot(x, pdf, 'r-', lw=2, label='Fitted Normal')
    
    # Add mean line
    plt.axvline(dist.mean(), color='green', linestyle='--', 
                lw=2, label=f'Mean = {dist.mean():.2f}')
    
    # Labels and legend
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution Fit', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fit_plot.png', dpi=300)
    plt.show()

CDF Plot
--------

**Empirical vs theoretical CDF:**

.. code-block:: python

    # Sort data for empirical CDF
    data_sorted = np.sort(data)
    empirical_cdf = np.arange(1, len(data) + 1) / len(data)
    
    # Theoretical CDF
    theoretical_cdf = dist.cdf(data_sorted)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(data_sorted, empirical_cdf, 
             'o', markersize=3, alpha=0.5, label='Empirical CDF')
    plt.plot(data_sorted, theoretical_cdf, 
             'r-', lw=2, label='Fitted CDF')
    
    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title('CDF Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('cdf_plot.png', dpi=300)
    plt.show()

Q-Q Plot
--------

**Quantile-Quantile diagnostic plot:**

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    # Get Q-Q data
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    # Plot
    plt.figure(figsize=(8, 8))
    plt.scatter(qq_data['theoretical'], qq_data['sample'],
                alpha=0.6, s=30, edgecolor='black', linewidth=0.5)
    
    # 45-degree reference line
    min_val = min(qq_data['theoretical'].min(), qq_data['sample'].min())
    max_val = max(qq_data['theoretical'].max(), qq_data['sample'].max())
    plt.plot([min_val, max_val], [min_val, max_val],
             'r--', lw=2, label='Perfect Fit')
    
    # Confidence bands (optional)
    n = len(data)
    se = dist.std() / np.sqrt(n)
    plt.fill_between([min_val, max_val],
                      [min_val - 1.96*se, max_val - 1.96*se],
                      [min_val + 1.96*se, max_val + 1.96*se],
                      alpha=0.2, color='gray', label='95% CI')
    
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles', fontsize=12)
    plt.title(f'Q-Q Plot (r = {qq_data["correlation"]:.4f})', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.savefig('qq_plot.png', dpi=300)
    plt.show()

Multi-Panel Diagnostic Plot
---------------------------

**Comprehensive 4-panel diagnostic:**

.. code-block:: python

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram + PDF
    axes[0, 0].hist(data, bins=50, density=True, alpha=0.7,
                    color='skyblue', edgecolor='black')
    x = np.linspace(data.min(), data.max(), 1000)
    axes[0, 0].plot(x, dist.pdf(x), 'r-', lw=2)
    axes[0, 0].set_xlabel('Value')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Histogram + Fitted PDF')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. CDF comparison
    data_sorted = np.sort(data)
    empirical = np.arange(1, len(data) + 1) / len(data)
    axes[0, 1].plot(data_sorted, empirical, 'o', 
                    markersize=2, alpha=0.5, label='Empirical')
    axes[0, 1].plot(data_sorted, dist.cdf(data_sorted), 
                    'r-', lw=2, label='Fitted')
    axes[0, 1].set_xlabel('Value')
    axes[0, 1].set_ylabel('Cumulative Probability')
    axes[0, 1].set_title('CDF Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Q-Q Plot
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    axes[1, 0].scatter(qq_data['theoretical'], qq_data['sample'],
                       alpha=0.5, s=20)
    lims = [min(qq_data['theoretical'].min(), qq_data['sample'].min()),
            max(qq_data['theoretical'].max(), qq_data['sample'].max())]
    axes[1, 0].plot(lims, lims, 'r--', lw=2)
    axes[1, 0].set_xlabel('Theoretical Quantiles')
    axes[1, 0].set_ylabel('Sample Quantiles')
    axes[1, 0].set_title(f'Q-Q Plot (r={qq_data["correlation"]:.3f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    axes[1, 1].hist(residuals.quantile_residuals, bins=50,
                    alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].axvline(0, color='red', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Quantile Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Residual Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(f'{dist.info.display_name} Fit Diagnostics',
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('diagnostic_panel.png', dpi=300)
    plt.show()

Comparing Multiple Distributions
---------------------------------

**Visual comparison of different fits:**

.. code-block:: python

    # Try multiple distributions
    distributions = ['normal', 'lognormal', 'gamma', 'weibull']
    
    plt.figure(figsize=(12, 8))
    
    # Histogram
    plt.hist(data, bins=50, density=True, alpha=0.5,
             color='gray', edgecolor='black', label='Data')
    
    # Fit and plot each
    colors = ['red', 'blue', 'green', 'orange']
    x = np.linspace(data.min(), data.max(), 1000)
    
    for dist_name, color in zip(distributions, colors):
        dist_temp = get_distribution(dist_name)
        try:
            dist_temp.fit(data)
            pdf = dist_temp.pdf(x)
            plt.plot(x, pdf, color=color, lw=2, 
                    label=f'{dist_temp.info.display_name}')
        except:
            pass
    
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Comparing Multiple Distributions', 
              fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('comparison_plot.png', dpi=300)
    plt.show()

Interactive Plotly Visualization
---------------------------------

**Create interactive plots with Plotly:**

.. code-block:: python

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histogram + PDF', 'CDF Comparison',
                       'Q-Q Plot', 'Residuals')
    )
    
    # 1. Histogram + PDF
    hist_counts, bin_edges = np.histogram(data, bins=50, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig.add_trace(
        go.Bar(x=bin_centers, y=hist_counts, name='Data',
               opacity=0.7, marker_color='skyblue'),
        row=1, col=1
    )
    
    x_plot = np.linspace(data.min(), data.max(), 500)
    fig.add_trace(
        go.Scatter(x=x_plot, y=dist.pdf(x_plot), 
                  name='Fitted PDF', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # 2. CDF
    data_sorted = np.sort(data)
    empirical = np.arange(1, len(data) + 1) / len(data)
    
    fig.add_trace(
        go.Scatter(x=data_sorted, y=empirical, 
                  mode='markers', name='Empirical',
                  marker=dict(size=3, opacity=0.5)),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=data_sorted, y=dist.cdf(data_sorted),
                  name='Fitted', line=dict(color='red', width=2)),
        row=1, col=2
    )
    
    # 3. Q-Q Plot
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    fig.add_trace(
        go.Scatter(x=qq_data['theoretical'], y=qq_data['sample'],
                  mode='markers', name='Q-Q',
                  marker=dict(size=5, opacity=0.6)),
        row=2, col=1
    )
    
    # 45-degree line
    lims = [qq_data['theoretical'].min(), qq_data['theoretical'].max()]
    fig.add_trace(
        go.Scatter(x=lims, y=lims, 
                  name='Perfect Fit',
                  line=dict(color='red', dash='dash', width=2)),
        row=2, col=1
    )
    
    # 4. Residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    
    fig.add_trace(
        go.Histogram(x=residuals.quantile_residuals, 
                    name='Residuals',
                    marker_color='lightgreen', opacity=0.7,
                    nbinsx=50),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        title_text=f'{dist.info.display_name} Fit Diagnostics',
        showlegend=True,
        height=800
    )
    
    fig.write_html('interactive_diagnostics.html')
    fig.show()

Publication-Quality Figure
--------------------------

**For papers and reports:**

.. code-block:: python

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.size'] = 10
    mpl.rcParams['axes.labelsize'] = 12
    mpl.rcParams['axes.titlesize'] = 12
    mpl.rcParams['xtick.labelsize'] = 10
    mpl.rcParams['ytick.labelsize'] = 10
    mpl.rcParams['legend.fontsize'] = 10
    
    # Create figure (single column width = 3.5 inches)
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    
    # Plot data
    ax.hist(data, bins=40, density=True, alpha=0.6,
            color='gray', edgecolor='black', linewidth=0.5,
            label='Observed data')
    
    # Plot fit
    x = np.linspace(data.min(), data.max(), 500)
    ax.plot(x, dist.pdf(x), 'k-', lw=1.5,
            label=f'Normal fit (\u03bc={dist.params["loc"]:.2f}, \u03c3={dist.params["scale"]:.2f})')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Probability density')
    ax.legend(frameon=True, loc='best')
    ax.grid(True, alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('publication_figure.pdf', dpi=600, bbox_inches='tight')
    plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
    plt.show()

Next Steps
----------

- :doc:`09_advanced` - Advanced visualization techniques
- :doc:`../examples/real_world` - Real-world visualization examples
