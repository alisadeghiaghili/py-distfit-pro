Tutorial 8: Visualization
=========================

Learn how to create publication-quality plots.

Basic Plotting
--------------

**Plot fitted distribution:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.plotting import plot_distribution
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate and fit
    data = np.random.normal(10, 2, 1000)
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Plot
    fig = plot_distribution(data, dist)
    plt.show()

**Components:**

- Histogram of data
- Fitted PDF/PMF curve
- Distribution name and parameters
- GOF statistics

Customizing Plots
-----------------

**Matplotlib backend:**

.. code-block:: python

    from distfit_pro.plotting import plot_distribution
    
    fig = plot_distribution(
        data=data,
        distribution=dist,
        bins=30,  # Number of bins
        title="Custom Title",
        xlabel="Custom X-axis",
        ylabel="Density"
    )
    
    plt.savefig('distribution_fit.png', dpi=300)

**Plotly backend (interactive):**

.. code-block:: python

    from distfit_pro.plotting import plot_distribution_interactive
    
    fig = plot_distribution_interactive(
        data=data,
        distribution=dist
    )
    
    fig.show()
    # Or save
    fig.write_html('distribution_fit.html')

Diagnostic Plots
----------------

Q-Q Plot
^^^^^^^^

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    plt.figure(figsize=(8, 8))
    plt.scatter(qq_data['theoretical'], qq_data['sample'], 
                alpha=0.6, s=20)
    
    # Reference line
    min_val = min(qq_data['theoretical'].min(), qq_data['sample'].min())
    max_val = max(qq_data['theoretical'].max(), qq_data['sample'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 
             'r--', lw=2, label='Perfect fit')
    
    plt.xlabel('Theoretical Quantiles', fontsize=12)
    plt.ylabel('Sample Quantiles', fontsize=12)
    plt.title(f'Q-Q Plot (r = {qq_data["correlation"]:.4f})', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Residual Plot
^^^^^^^^^^^^^

.. code-block:: python

    residuals = Diagnostics.residual_analysis(data, dist)
    qres = residuals.quantile_residuals
    
    # Residual histogram
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(qres, bins=30, density=True, alpha=0.7, 
                 edgecolor='black')
    
    # Overlay N(0,1)
    x = np.linspace(qres.min(), qres.max(), 100)
    from scipy import stats
    axes[0].plot(x, stats.norm.pdf(x), 'r-', lw=2, 
                 label='N(0,1)')
    
    axes[0].set_xlabel('Quantile Residuals')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Residual Histogram')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Residual scatter
    axes[1].scatter(range(len(qres)), qres, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].axhline(y=2, color='gray', linestyle=':')
    axes[1].axhline(y=-2, color='gray', linestyle=':')
    axes[1].set_xlabel('Observation Index')
    axes[1].set_ylabel('Quantile Residuals')
    axes[1].set_title('Residual Scatter')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

Multi-Panel Diagnostic
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    def diagnostic_panel(data, dist):
        """Create 4-panel diagnostic plot"""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Histogram + PDF
        axes[0, 0].hist(data, bins=30, density=True, alpha=0.7,
                       edgecolor='black', label='Data')
        x = np.linspace(data.min(), data.max(), 200)
        axes[0, 0].plot(x, dist.pdf(x), 'r-', lw=2, label='Fitted')
        axes[0, 0].set_xlabel('Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].set_title('Histogram + Fitted PDF')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Q-Q Plot
        qq_data = Diagnostics.qq_diagnostics(data, dist)
        axes[0, 1].scatter(qq_data['theoretical'], qq_data['sample'],
                          alpha=0.5)
        lims = [min(qq_data['theoretical'].min(), qq_data['sample'].min()),
                max(qq_data['theoretical'].max(), qq_data['sample'].max())]
        axes[0, 1].plot(lims, lims, 'r--', lw=2)
        axes[0, 1].set_xlabel('Theoretical Quantiles')
        axes[0, 1].set_ylabel('Sample Quantiles')
        axes[0, 1].set_title(f'Q-Q Plot (r={qq_data["correlation"]:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residuals
        residuals = Diagnostics.residual_analysis(data, dist)
        qres = residuals.quantile_residuals
        axes[1, 0].hist(qres, bins=30, density=True, alpha=0.7,
                       edgecolor='black')
        x = np.linspace(qres.min(), qres.max(), 100)
        axes[1, 0].plot(x, stats.norm.pdf(x), 'r-', lw=2)
        axes[1, 0].set_xlabel('Quantile Residuals')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. P-P Plot
        pp_data = Diagnostics.pp_diagnostics(data, dist)
        axes[1, 1].scatter(pp_data['theoretical'], pp_data['empirical'],
                          alpha=0.5)
        axes[1, 1].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[1, 1].set_xlabel('Theoretical Probability')
        axes[1, 1].set_ylabel('Empirical Probability')
        axes[1, 1].set_title(f'P-P Plot (dev={pp_data["max_deviation"]:.3f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'{dist.info.display_name}', fontsize=16, y=1.00)
        plt.tight_layout()
        return fig
    
    # Use it
    fig = diagnostic_panel(data, dist)
    plt.show()

Comparing Distributions
-----------------------

**Overlay multiple fits:**

.. code-block:: python

    # Fit multiple distributions
    dists = {}
    for name in ['normal', 'lognormal', 'gamma']:
        d = get_distribution(name)
        try:
            d.fit(data)
            dists[name] = d
        except:
            pass
    
    # Plot all
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, density=True, alpha=0.5, 
             label='Data', edgecolor='black')
    
    colors = ['red', 'blue', 'green']
    x = np.linspace(data.min(), data.max(), 200)
    
    for (name, dist), color in zip(dists.items(), colors):
        plt.plot(x, dist.pdf(x), color=color, lw=2, 
                label=name.capitalize())
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Weighted Data Visualization
----------------------------

.. code-block:: python

    # Weighted histogram
    plt.figure(figsize=(10, 6))
    
    plt.hist(data, weights=weights, bins=30, density=True,
             alpha=0.7, label='Weighted data', edgecolor='black')
    
    x = np.linspace(data.min(), data.max(), 200)
    plt.plot(x, dist.pdf(x), 'r-', lw=2, label='Fitted (weighted)')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Weighted Distribution Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

Bootstrap Visualization
-----------------------

**Plot parameter uncertainty:**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    # Bootstrap
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    # Collect bootstrap samples manually for visualization
    boot_locs = []
    boot_scales = []
    
    for i in range(1000):
        boot_data = dist.rvs(size=len(data), random_state=i)
        boot_dist = get_distribution('normal')
        boot_dist.fit(boot_data)
        boot_locs.append(boot_dist.params['loc'])
        boot_scales.append(boot_dist.params['scale'])
    
    # Plot bootstrap distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean (loc)
    axes[0].hist(boot_locs, bins=30, density=True, alpha=0.7,
                edgecolor='black')
    axes[0].axvline(dist.params['loc'], color='red', 
                   linestyle='--', lw=2, label='Estimate')
    axes[0].axvline(ci_results['loc'].ci_lower, color='green',
                   linestyle=':', lw=2, label='95% CI')
    axes[0].axvline(ci_results['loc'].ci_upper, color='green',
                   linestyle=':', lw=2)
    axes[0].set_xlabel('Mean')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Bootstrap Distribution of Mean')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Std (scale)
    axes[1].hist(boot_scales, bins=30, density=True, alpha=0.7,
                edgecolor='black')
    axes[1].axvline(dist.params['scale'], color='red',
                   linestyle='--', lw=2, label='Estimate')
    axes[1].axvline(ci_results['scale'].ci_lower, color='green',
                   linestyle=':', lw=2, label='95% CI')
    axes[1].axvline(ci_results['scale'].ci_upper, color='green',
                   linestyle=':', lw=2)
    axes[1].set_xlabel('Std Deviation')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Bootstrap Distribution of Std')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

Publication-Quality Styling
---------------------------

**Use seaborn or custom styles:**

.. code-block:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Seaborn style
    sns.set_style('whitegrid')
    sns.set_context('paper', font_scale=1.5)
    
    # Or matplotlib style
    plt.style.use('seaborn-v0_8-paper')
    
    # High-res figure
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    ax.hist(data, bins=30, density=True, alpha=0.7,
            color='steelblue', edgecolor='black', linewidth=1.5)
    
    x = np.linspace(data.min(), data.max(), 200)
    ax.plot(x, dist.pdf(x), 'r-', lw=3, label='Fitted PDF')
    
    ax.set_xlabel('Value', fontsize=14, fontweight='bold')
    ax.set_ylabel('Density', fontsize=14, fontweight='bold')
    ax.set_title('Distribution Fit', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('publication_plot.png', dpi=300, bbox_inches='tight')
    plt.show()

Interactive Plotly
------------------

**Create interactive plots:**

.. code-block:: python

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Histogram + PDF', 'Q-Q Plot', 
                       'Residuals', 'CDF')
    )
    
    # 1. Histogram + PDF
    fig.add_trace(
        go.Histogram(x=data, histnorm='probability density',
                    name='Data', opacity=0.7),
        row=1, col=1
    )
    
    x = np.linspace(data.min(), data.max(), 200)
    fig.add_trace(
        go.Scatter(x=x, y=dist.pdf(x), mode='lines',
                  name='Fitted PDF', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # 2. Q-Q Plot
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    fig.add_trace(
        go.Scatter(x=qq_data['theoretical'], y=qq_data['sample'],
                  mode='markers', name='Q-Q', opacity=0.6),
        row=1, col=2
    )
    
    # Reference line
    lims = [min(qq_data['theoretical'].min(), qq_data['sample'].min()),
            max(qq_data['theoretical'].max(), qq_data['sample'].max())]
    fig.add_trace(
        go.Scatter(x=lims, y=lims, mode='lines',
                  name='Reference', line=dict(color='red', dash='dash')),
        row=1, col=2
    )
    
    # 3. Residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    fig.add_trace(
        go.Histogram(x=residuals.quantile_residuals, 
                    histnorm='probability density',
                    name='Residuals', opacity=0.7),
        row=2, col=1
    )
    
    # 4. CDF
    x_sorted = np.sort(data)
    empirical_cdf = np.arange(1, len(data)+1) / len(data)
    
    fig.add_trace(
        go.Scatter(x=x_sorted, y=empirical_cdf, mode='markers',
                  name='Empirical CDF', opacity=0.5),
        row=2, col=2
    )
    
    x = np.linspace(data.min(), data.max(), 200)
    fig.add_trace(
        go.Scatter(x=x, y=dist.cdf(x), mode='lines',
                  name='Fitted CDF', line=dict(color='red', width=2)),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"{dist.info.display_name} - Diagnostic Plots",
        showlegend=True
    )
    
    fig.show()

Next Steps
----------

- :doc:`09_advanced` - Advanced techniques
- :doc:`../examples/basic` - More plot examples
- :doc:`../api/plotting` - Plotting API reference
