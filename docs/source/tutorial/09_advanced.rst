Tutorial 9: Advanced Topics
===========================

Advanced techniques for power users.

Automatic Distribution Selection
---------------------------------

**Find the best distribution automatically:**

.. code-block:: python

    from distfit_pro import get_distribution, list_distributions
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    def auto_select_distribution(data, test='ks', alpha=0.05):
        """Automatically select best distribution"""
        
        candidates = list_distributions()
        results = []
        
        for dist_name in candidates:
            try:
                dist = get_distribution(dist_name)
                dist.fit(data, method='mle')
                
                # GOF test
                if test == 'ks':
                    gof = GOFTests.kolmogorov_smirnov(data, dist, alpha=alpha)
                elif test == 'ad':
                    gof = GOFTests.anderson_darling(data, dist, alpha=alpha)
                else:
                    raise ValueError(f"Unknown test: {test}")
                
                # Calculate AIC
                n = len(data)
                k = len(dist.params)
                log_lik = np.sum(dist.logpdf(data))
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(n) - 2 * log_lik
                
                results.append({
                    'distribution': dist_name,
                    'p_value': gof.p_value,
                    'reject': gof.reject_null,
                    'aic': aic,
                    'bic': bic,
                    'dist_obj': dist
                })
            except:
                pass
        
        # Filter: only non-rejected
        valid = [r for r in results if not r['reject']]
        
        if not valid:
            print("⚠️  No distribution passed GOF test!")
            # Return best AIC even if rejected
            return min(results, key=lambda x: x['aic'])
        
        # Return best AIC among valid
        best = min(valid, key=lambda x: x['aic'])
        return best
    
    # Example
    data = np.random.gamma(2, 3, 1000)
    best = auto_select_distribution(data)
    
    print(f"Best distribution: {best['distribution']}")
    print(f"AIC: {best['aic']:.2f}")
    print(f"P-value: {best['p_value']:.4f}")

Mixture Distributions
---------------------

**Fit mixture of distributions manually:**

.. code-block:: python

    # Generate mixture data
    np.random.seed(42)
    data1 = np.random.normal(5, 1, 500)
    data2 = np.random.normal(12, 1.5, 500)
    data_mixed = np.concatenate([data1, data2])
    
    # Fit two separate normals (EM algorithm would be better)
    # This is a simplified approach
    
    from scipy.optimize import minimize
    
    def mixture_pdf(x, params):
        """PDF of mixture of two normals"""
        w, mu1, sigma1, mu2, sigma2 = params
        
        from scipy.stats import norm
        pdf1 = norm.pdf(x, mu1, sigma1)
        pdf2 = norm.pdf(x, mu2, sigma2)
        
        return w * pdf1 + (1 - w) * pdf2
    
    def neg_log_likelihood(params, data):
        """Negative log-likelihood for mixture"""
        pdf_vals = mixture_pdf(data, params)
        pdf_vals = np.maximum(pdf_vals, 1e-300)  # Avoid log(0)
        return -np.sum(np.log(pdf_vals))
    
    # Initial guess
    w0 = 0.5
    mu1_0 = np.percentile(data_mixed, 25)
    mu2_0 = np.percentile(data_mixed, 75)
    sigma1_0 = sigma2_0 = np.std(data_mixed) / 2
    
    x0 = [w0, mu1_0, sigma1_0, mu2_0, sigma2_0]
    
    # Optimize
    bounds = [(0.01, 0.99),  # weight
              (None, None),   # mu1
              (0.01, None),   # sigma1
              (None, None),   # mu2
              (0.01, None)]   # sigma2
    
    result = minimize(neg_log_likelihood, x0, args=(data_mixed,),
                     bounds=bounds, method='L-BFGS-B')
    
    params_opt = result.x
    print("Mixture parameters:")
    print(f"  Weight: {params_opt[0]:.3f}")
    print(f"  Component 1: N({params_opt[1]:.2f}, {params_opt[2]:.2f})")
    print(f"  Component 2: N({params_opt[3]:.2f}, {params_opt[4]:.2f})")
    
    # Plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(data_mixed, bins=60, density=True, alpha=0.6,
             color='gray', edgecolor='black', label='Data')
    
    x = np.linspace(data_mixed.min(), data_mixed.max(), 1000)
    plt.plot(x, mixture_pdf(x, params_opt), 'r-', lw=2,
            label='Fitted Mixture')
    
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Gaussian Mixture Model')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

Truncated Distributions
-----------------------

**Fit truncated distributions:**

.. code-block:: python

    from scipy import stats
    
    # Generate truncated normal data (only positive values)
    lower_bound = 0
    upper_bound = np.inf
    
    # True distribution: Normal(5, 2) truncated at 0
    from scipy.stats import truncnorm
    a = (lower_bound - 5) / 2  # Standardized lower
    b = (upper_bound - 5) / 2  # Standardized upper
    data_trunc = truncnorm.rvs(a, b, loc=5, scale=2, size=1000)
    
    print(f"Data range: [{data_trunc.min():.2f}, {data_trunc.max():.2f}]")
    
    # Fit truncated normal manually
    def truncnorm_mle(data, lower, upper):
        """MLE for truncated normal"""
        
        def neg_log_lik(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            
            a_std = (lower - mu) / sigma
            b_std = (upper - mu) / sigma
            
            try:
                tn = truncnorm(a_std, b_std, loc=mu, scale=sigma)
                log_lik = np.sum(tn.logpdf(data))
                return -log_lik
            except:
                return np.inf
        
        # Initial guess
        mu0 = np.mean(data)
        sigma0 = np.std(data)
        
        result = minimize(neg_log_lik, [mu0, sigma0],
                         method='Nelder-Mead')
        
        return {'mu': result.x[0], 'sigma': result.x[1]}
    
    params_trunc = truncnorm_mle(data_trunc, lower_bound, upper_bound)
    print(f"\nTruncated Normal fit:")
    print(f"  μ = {params_trunc['mu']:.2f}")
    print(f"  σ = {params_trunc['sigma']:.2f}")

Time Series of Distributions
----------------------------

**Track how distribution parameters change over time:**

.. code-block:: python

    # Simulate data where mean increases over time
    np.random.seed(42)
    n_periods = 20
    samples_per_period = 100
    
    results = []
    
    for t in range(n_periods):
        # Mean increases linearly
        mean_t = 10 + 0.5 * t
        data_t = np.random.normal(mean_t, 2, samples_per_period)
        
        # Fit
        dist = get_distribution('normal')
        dist.fit(data_t)
        
        # Store results
        results.append({
            'time': t,
            'mean': dist.params['loc'],
            'std': dist.params['scale']
        })
    
    # Plot evolution
    import pandas as pd
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Mean over time
    axes[0].plot(df['time'], df['mean'], 'o-', lw=2)
    axes[0].set_xlabel('Time Period')
    axes[0].set_ylabel('Mean')
    axes[0].set_title('Mean Parameter Over Time')
    axes[0].grid(True, alpha=0.3)
    
    # Std over time
    axes[1].plot(df['time'], df['std'], 'o-', lw=2, color='orange')
    axes[1].set_xlabel('Time Period')
    axes[1].set_ylabel('Std Dev')
    axes[1].set_title('Std Parameter Over Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

Cross-Validation
----------------

**Validate distribution fit with cross-validation:**

.. code-block:: python

    from sklearn.model_selection import KFold
    
    def cross_validate_distribution(data, dist_name, n_splits=5):
        """K-fold cross-validation for distribution fit"""
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        log_likelihoods = []
        
        for train_idx, test_idx in kf.split(data):
            # Split data
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            # Fit on training
            dist = get_distribution(dist_name)
            dist.fit(train_data)
            
            # Evaluate on test
            test_log_lik = np.sum(dist.logpdf(test_data))
            log_likelihoods.append(test_log_lik)
        
        return {
            'mean_log_lik': np.mean(log_likelihoods),
            'std_log_lik': np.std(log_likelihoods),
            'log_likelihoods': log_likelihoods
        }
    
    # Example
    data = np.random.gamma(2, 3, 1000)
    
    # Compare distributions
    candidates = ['gamma', 'lognormal', 'weibull']
    
    for dist_name in candidates:
        cv_results = cross_validate_distribution(data, dist_name)
        print(f"{dist_name}:")
        print(f"  Mean log-likelihood: {cv_results['mean_log_lik']:.2f}")
        print(f"  Std: {cv_results['std_log_lik']:.2f}")

Bootstrap Hypothesis Testing
----------------------------

**Test if two datasets come from same distribution:**

.. code-block:: python

    def bootstrap_distribution_test(data1, data2, n_bootstrap=1000):
        """Bootstrap test for distribution equality"""
        
        n1, n2 = len(data1), len(data2)
        
        # Test statistic: difference in means
        observed_diff = np.mean(data1) - np.mean(data2)
        
        # Pool data under null hypothesis
        pooled = np.concatenate([data1, data2])
        
        # Bootstrap
        boot_diffs = []
        
        for _ in range(n_bootstrap):
            # Resample from pooled data
            perm = np.random.permutation(pooled)
            boot_data1 = perm[:n1]
            boot_data2 = perm[n1:]
            
            boot_diff = np.mean(boot_data1) - np.mean(boot_data2)
            boot_diffs.append(boot_diff)
        
        boot_diffs = np.array(boot_diffs)
        
        # P-value (two-tailed)
        p_value = np.mean(np.abs(boot_diffs) >= np.abs(observed_diff))
        
        return {
            'observed_diff': observed_diff,
            'p_value': p_value,
            'reject_h0': p_value < 0.05,
            'boot_diffs': boot_diffs
        }
    
    # Example
    data1 = np.random.normal(10, 2, 500)
    data2 = np.random.normal(10.5, 2, 500)  # Slightly different mean
    
    result = bootstrap_distribution_test(data1, data2)
    print(f"Observed difference: {result['observed_diff']:.4f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Reject H0: {result['reject_h0']}")

Prediction Intervals
--------------------

**Generate prediction intervals for future observations:**

.. code-block:: python

    def prediction_interval(dist, confidence=0.95, n_future=100):
        """Generate prediction intervals"""
        
        alpha = 1 - confidence
        
        # Lower and upper bounds
        lower = dist.ppf(alpha / 2)
        upper = dist.ppf(1 - alpha / 2)
        
        # Generate future samples
        future_samples = dist.rvs(size=n_future)
        
        # Check coverage
        in_interval = (future_samples >= lower) & (future_samples <= upper)
        coverage = np.mean(in_interval)
        
        return {
            'lower': lower,
            'upper': upper,
            'interval_width': upper - lower,
            'coverage': coverage,
            'expected_coverage': confidence
        }
    
    # Example
    dist = get_distribution('normal')
    dist.fit(data)
    
    pi = prediction_interval(dist, confidence=0.95)
    print(f"95% Prediction Interval: [{pi['lower']:.2f}, {pi['upper']:.2f}]")
    print(f"Actual coverage: {pi['coverage']:.1%}")

Tolerance Intervals
-------------------

**Statistical tolerance intervals:**

.. code-block:: python

    def tolerance_interval(dist, coverage=0.95, confidence=0.95):
        """
        Tolerance interval: contains at least `coverage` proportion
        of population with `confidence` probability
        """
        from scipy import stats as sp_stats
        
        # For normal distribution (exact)
        if dist.info.name == 'normal':
            n = 1000  # Assumed sample size
            alpha = 1 - confidence
            
            # K factor from tolerance interval tables
            # Simplified approximation
            z = sp_stats.norm.ppf(1 - alpha/2)
            k = z * np.sqrt(1 + 1/n)
            
            mean = dist.mean()
            std = dist.std()
            
            lower = mean - k * std
            upper = mean + k * std
            
            return (lower, upper)
        else:
            # Non-parametric approach
            alpha_p = (1 - coverage) / 2
            lower = dist.ppf(alpha_p)
            upper = dist.ppf(1 - alpha_p)
            return (lower, upper)
    
    # Example
    dist = get_distribution('normal')
    dist.fit(data)
    
    tol_int = tolerance_interval(dist, coverage=0.95, confidence=0.95)
    print(f"Tolerance Interval: [{tol_int[0]:.2f}, {tol_int[1]:.2f}]")
    print("This interval contains 95% of population with 95% confidence")

Next Steps
----------

- :doc:`../examples/basic` - Basic examples
- :doc:`../examples/advanced` - Advanced examples
- :doc:`../examples/real_world` - Real-world applications
