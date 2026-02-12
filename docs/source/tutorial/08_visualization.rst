Tutorial 8: Visualization
=========================

Learn to visualize distributions and diagnostics.

.. note::
   This tutorial covers the visualization API. For actual plotting,
   see :doc:`../user_guide/visualization` for complete examples.

Basic Plotting
--------------

**Plot PDF and data histogram:**

.. code-block:: python

    from distfit_pro import get_distribution
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Generate data
    np.random.seed(42)
    data = np.random.normal(10, 2, 1000)
    
    # Fit distribution
    dist = get_distribution('normal')
    dist.fit(data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(data, bins=30, density=True, alpha=0.7, 
            color='skyblue', edgecolor='black', label='Data')
    
    # Fitted PDF
    x = np.linspace(data.min(), data.max(), 200)
    pdf = dist.pdf(x)
    ax.plot(x, pdf, 'r-', linewidth=2, label='Fitted PDF')
    
    # Labels
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Normal Distribution Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

Q-Q Plot
--------

**Quantile-Quantile plot for GOF assessment:**

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    # Get Q-Q data
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Q-Q plot
    ax.scatter(qq_data['theoretical'], qq_data['sample'], 
               alpha=0.6, s=20, label='Data')
    
    # Reference line (perfect fit)
    lims = [
        min(qq_data['theoretical'].min(), qq_data['sample'].min()),
        max(qq_data['theoretical'].max(), qq_data['sample'].max())
    ]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Perfect Fit')
    
    # Labels
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f'Q-Q Plot (Correlation: {qq_data["correlation"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

P-P Plot
--------

**Probability-Probability plot:**

.. code-block:: python

    # Get P-P data
    pp_data = Diagnostics.pp_diagnostics(data, dist)
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.scatter(pp_data['theoretical'], pp_data['empirical'],
               alpha=0.6, s=20)
    
    # Reference line
    ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Fit')
    
    ax.set_xlabel('Theoretical Probability')
    ax.set_ylabel('Empirical Probability')
    ax.set_title(f'P-P Plot (Max Dev: {pp_data["max_deviation"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

Residual Plots
--------------

**Visualize residuals:**

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    # Get residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Quantile residuals histogram
    axes[0, 0].hist(residuals.quantile_residuals, bins=30, 
                    density=True, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Quantile Residuals')
    axes[0, 0].set_xlabel('Residual')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add normal curve
    x = np.linspace(-4, 4, 100)
    from scipy import stats
    axes[0, 0].plot(x, stats.norm.pdf(x), 'r-', linewidth=2, label='N(0,1)')
    axes[0, 0].legend()
    
    # 2. Residuals vs fitted
    fitted_values = dist.mean() * np.ones_like(data)
    axes[0, 1].scatter(fitted_values, residuals.standardized_residuals,
                       alpha=0.5, s=20)
    axes[0, 1].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_title('Residuals vs Fitted')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Standardized Residuals')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residuals vs index
    axes[1, 0].scatter(range(len(data)), residuals.standardized_residuals,
                       alpha=0.5, s=20)
    axes[1, 0].axhline(0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_title('Residuals vs Index')
    axes[1, 0].set_xlabel('Index')
    axes[1, 0].set_ylabel('Standardized Residuals')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Normal Q-Q of residuals
    stats.probplot(residuals.quantile_residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Normal Q-Q Plot of Residuals')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

CDF Comparison
--------------

**Compare empirical and theoretical CDFs:**

.. code-block:: python

    # Empirical CDF
    data_sorted = np.sort(data)
    empirical_cdf = np.arange(1, len(data)+1) / len(data)
    
    # Theoretical CDF
    theoretical_cdf = dist.cdf(data_sorted)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(data_sorted, empirical_cdf, 
            'b-', linewidth=2, label='Empirical CDF', alpha=0.7)
    ax.plot(data_sorted, theoretical_cdf, 
            'r--', linewidth=2, label='Fitted CDF')
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

Multiple Distributions
----------------------

**Compare several fitted distributions:**

.. code-block:: python

    # Fit multiple distributions
    dist_names = ['normal', 'lognormal', 'gamma', 'weibull']
    distributions = {}
    
    for name in dist_names:
        try:
            dist = get_distribution(name)
            dist.fit(data)
            distributions[name] = dist
        except:
            pass
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Data histogram
    ax.hist(data, bins=30, density=True, alpha=0.5, 
            color='gray', edgecolor='black', label='Data')
    
    # Fitted PDFs
    x = np.linspace(data.min(), data.max(), 200)
    colors = ['red', 'blue', 'green', 'orange']
    
    for (name, dist), color in zip(distributions.items(), colors):
        pdf = dist.pdf(x)
        ax.plot(x, pdf, linewidth=2, label=name.capitalize(), color=color)
    
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

Influence Diagnostics Plot
--------------------------

**Visualize influential observations:**

.. code-block:: python

    from distfit_pro.core.diagnostics import Diagnostics
    
    # Get influence diagnostics
    influence = Diagnostics.influence_diagnostics(data, dist)
    
    # Plot Cook's distance
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cook's D
    threshold = 4 / len(data)
    axes[0].stem(range(len(data)), influence.cooks_distance, 
                 basefmt=" ", use_line_collection=True)
    axes[0].axhline(threshold, color='r', linestyle='--', 
                    linewidth=2, label=f'Threshold = {threshold:.4f}')
    axes[0].set_xlabel('Observation Index')
    axes[0].set_ylabel("Cook's Distance")
    axes[0].set_title("Cook's Distance Plot")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Leverage vs Residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    axes[1].scatter(influence.leverage, 
                    residuals.standardized_residuals,
                    alpha=0.6, s=50)
    axes[1].set_xlabel('Leverage')
    axes[1].set_ylabel('Standardized Residuals')
    axes[1].set_title('Leverage vs Residuals')
    axes[1].grid(True, alpha=0.3)
    
    # Mark influential points
    for idx in influence.influential_indices[:5]:  # Top 5
        axes[1].scatter(influence.leverage[idx],
                       residuals.standardized_residuals[idx],
                       color='red', s=100, marker='x')
    
    plt.tight_layout()
    plt.show()

Bootstrap Distribution
----------------------

**Visualize bootstrap parameter distribution:**

.. code-block:: python

    from distfit_pro.core.bootstrap import Bootstrap
    
    # Run bootstrap
    ci_results = Bootstrap.parametric(data, dist, n_bootstrap=1000)
    
    # Get bootstrap samples (need to rerun to collect)
    boot_samples_loc = []
    for i in range(1000):
        boot_data = dist.rvs(size=len(data), random_state=i)
        boot_dist = get_distribution('normal')
        boot_dist.fit(boot_data)
        boot_samples_loc.append(boot_dist.params['loc'])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram of bootstrap estimates
    ax.hist(boot_samples_loc, bins=30, density=True, 
            alpha=0.7, edgecolor='black', label='Bootstrap Distribution')
    
    # Original estimate
    ax.axvline(dist.params['loc'], color='red', 
               linestyle='--', linewidth=2, label='Original Estimate')
    
    # Confidence interval
    result = ci_results['loc']
    ax.axvline(result.ci_lower, color='green', 
               linestyle=':', linewidth=2, label='95% CI')
    ax.axvline(result.ci_upper, color='green', 
               linestyle=':', linewidth=2)
    
    ax.set_xlabel('Parameter Value')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap Distribution of Location Parameter')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

Interactive Plotly
------------------

**Create interactive plots with Plotly:**

.. code-block:: python

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('PDF Fit', 'CDF Comparison', 
                       'Q-Q Plot', 'Residuals')
    )
    
    # 1. PDF fit
    hist_data = go.Histogram(
        x=data, 
        histnorm='probability density',
        name='Data',
        opacity=0.7
    )
    
    x_range = np.linspace(data.min(), data.max(), 200)
    pdf_line = go.Scatter(
        x=x_range,
        y=dist.pdf(x_range),
        mode='lines',
        name='Fitted PDF',
        line=dict(color='red', width=2)
    )
    
    fig.add_trace(hist_data, row=1, col=1)
    fig.add_trace(pdf_line, row=1, col=1)
    
    # 2. CDF comparison
    data_sorted = np.sort(data)
    empirical = np.arange(1, len(data)+1) / len(data)
    
    fig.add_trace(go.Scatter(
        x=data_sorted, y=empirical,
        mode='lines', name='Empirical CDF'
    ), row=1, col=2)
    
    fig.add_trace(go.Scatter(
        x=data_sorted, y=dist.cdf(data_sorted),
        mode='lines', name='Fitted CDF',
        line=dict(dash='dash')
    ), row=1, col=2)
    
    # 3. Q-Q plot
    qq_data = Diagnostics.qq_diagnostics(data, dist)
    
    fig.add_trace(go.Scatter(
        x=qq_data['theoretical'],
        y=qq_data['sample'],
        mode='markers',
        name='Q-Q'
    ), row=2, col=1)
    
    # Reference line
    lims = [qq_data['theoretical'].min(), qq_data['theoretical'].max()]
    fig.add_trace(go.Scatter(
        x=lims, y=lims,
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Reference'
    ), row=2, col=1)
    
    # 4. Residuals
    residuals = Diagnostics.residual_analysis(data, dist)
    
    fig.add_trace(go.Scatter(
        x=list(range(len(data))),
        y=residuals.standardized_residuals,
        mode='markers',
        name='Residuals'
    ), row=2, col=2)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="Comprehensive Distribution Fit Diagnostics"
    )
    
    fig.show()

Best Practices
--------------

1. **Always visualize**
   
   Don't rely on statistics alone.

2. **Use multiple plots**
   
   Different plots reveal different issues.

3. **Check residuals**
   
   Should be random, no patterns.

4. **Save high-quality figures**
   
   .. code-block:: python
   
       plt.savefig('distribution_fit.png', dpi=300, bbox_inches='tight')

5. **Label everything**
   
   Titles, axes, legends - make it self-explanatory.

Next Steps
----------

- :doc:`09_advanced` - Advanced topics
- :doc:`../user_guide/visualization` - Complete visualization guide
- :doc:`../examples/real_world` - Real-world plotting examples
