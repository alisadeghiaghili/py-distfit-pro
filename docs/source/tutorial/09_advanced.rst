Tutorial 9: Advanced Techniques
================================

Advanced usage patterns and best practices.

Automatic Distribution Selection
--------------------------------

**Find the best distribution automatically:**

.. code-block:: python

    from distfit_pro import get_distribution, list_continuous_distributions
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    def auto_select_distribution(data, method='aic', top_n=3):
        """
        Automatically select best distribution
        
        Parameters:
        -----------
        data : array-like
            Data to fit
        method : str
            Selection criterion: 'aic', 'bic', 'ks', 'ad'
        top_n : int
            Return top N distributions
        """
        results = []
        
        for dist_name in list_continuous_distributions():
            try:
                dist = get_distribution(dist_name)
                dist.fit(data, method='mle')
                
                # Calculate criterion
                n = len(data)
                k = len(dist.params)
                log_lik = np.sum(dist.logpdf(data))
                
                if method == 'aic':
                    score = 2 * k - 2 * log_lik
                elif method == 'bic':
                    score = k * np.log(n) - 2 * log_lik
                elif method == 'ks':
                    ks_result = GOFTests.kolmogorov_smirnov(data, dist)
                    score = ks_result.statistic
                elif method == 'ad':
                    ad_result = GOFTests.anderson_darling(data, dist)
                    score = ad_result.statistic
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                results.append({
                    'name': dist_name,
                    'dist': dist,
                    'score': score,
                    'params': dist.params
                })
            except:
                pass
        
        # Sort (lower is better)
        results.sort(key=lambda x: x['score'])
        
        return results[:top_n]
    
    # Use it
    data = np.random.gamma(2, 3, 1000)
    
    top_dists = auto_select_distribution(data, method='aic', top_n=3)
    
    print("Top 3 distributions:")
    for i, result in enumerate(top_dists, 1):
        print(f"{i}. {result['name']}: AIC={result['score']:.2f}")
    
    # Use best distribution
    best = top_dists[0]
    print(f"\nBest fit: {best['name']}")
    print(f"Parameters: {best['params']}")

Cross-Validation
----------------

**Validate model on held-out data:**

.. code-block:: python

    from sklearn.model_selection import KFold
    
    def cv_distribution(data, dist_name, n_splits=5):
        """
        Cross-validate distribution fit
        
        Returns average log-likelihood on test sets
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        log_liks = []
        
        for train_idx, test_idx in kf.split(data):
            train_data = data[train_idx]
            test_data = data[test_idx]
            
            # Fit on training
            dist = get_distribution(dist_name)
            dist.fit(train_data, method='mle')
            
            # Evaluate on test
            test_log_lik = np.mean(dist.logpdf(test_data))
            log_liks.append(test_log_lik)
        
        return np.mean(log_liks), np.std(log_liks)
    
    # Compare distributions with CV
    data = np.random.gamma(2, 3, 1000)
    
    for dist_name in ['gamma', 'lognormal', 'weibull']:
        mean_ll, std_ll = cv_distribution(data, dist_name)
        print(f"{dist_name}: log-lik = {mean_ll:.4f} ± {std_ll:.4f}")

Mixture Models (Manual)
-----------------------

**Fit simple mixture manually:**

.. code-block:: python

    # Two-component mixture
    # 60% N(5, 1) + 40% N(10, 1.5)
    
    data_mix = np.concatenate([
        np.random.normal(5, 1, 600),
        np.random.normal(10, 1.5, 400)
    ])
    
    # Fit two distributions
    # (In practice, use EM algorithm or sklearn.mixture)
    
    # Separate components (if known)
    component1 = data_mix[data_mix < 7.5]
    component2 = data_mix[data_mix >= 7.5]
    
    dist1 = get_distribution('normal')
    dist1.fit(component1)
    
    dist2 = get_distribution('normal')
    dist2.fit(component2)
    
    # Mixture PDF
    def mixture_pdf(x, w1=0.6):
        return w1 * dist1.pdf(x) + (1-w1) * dist2.pdf(x)
    
    # Plot
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.hist(data_mix, bins=50, density=True, alpha=0.5, label='Data')
    
    x = np.linspace(data_mix.min(), data_mix.max(), 200)
    plt.plot(x, mixture_pdf(x), 'r-', lw=2, label='Mixture')
    plt.plot(x, 0.6*dist1.pdf(x), 'g--', label='Component 1')
    plt.plot(x, 0.4*dist2.pdf(x), 'b--', label='Component 2')
    
    plt.legend()
    plt.title('Mixture Model')
    plt.show()

Time Series of Distributions
----------------------------

**Track parameter evolution:**

.. code-block:: python

    # Simulate changing distribution over time
    time_points = 10
    window_size = 200
    
    params_over_time = {'loc': [], 'scale': []}
    
    for t in range(time_points):
        # Mean increases, std stays constant
        mean_t = 10 + t * 0.5
        data_t = np.random.normal(mean_t, 2, window_size)
        
        dist_t = get_distribution('normal')
        dist_t.fit(data_t)
        
        params_over_time['loc'].append(dist_t.params['loc'])
        params_over_time['scale'].append(dist_t.params['scale'])
    
    # Plot evolution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(params_over_time['loc'], 'o-', lw=2)
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Mean')
    axes[0].set_title('Mean Evolution')
    axes[0].grid(True)
    
    axes[1].plot(params_over_time['scale'], 'o-', lw=2)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Std Deviation')
    axes[1].set_title('Std Evolution')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

Robust Fitting
--------------

**Handle outliers automatically:**

.. code-block:: python

    def robust_fit(data, dist_name, contamination=0.05):
        """
        Fit distribution after removing outliers
        """
        # Detect outliers
        dist_initial = get_distribution(dist_name)
        dist_initial.fit(data, method='quantile')  # Robust initial fit
        
        from distfit_pro.core.diagnostics import Diagnostics
        outliers = Diagnostics.detect_outliers(
            data, dist_initial, method='zscore', threshold=3
        )
        
        # Remove outliers
        mask = np.ones(len(data), dtype=bool)
        mask[outliers.outlier_indices] = False
        clean_data = data[mask]
        
        print(f"Removed {len(outliers.outlier_indices)} outliers")
        print(f"Fitting on {len(clean_data)} clean observations")
        
        # Refit on clean data
        dist_final = get_distribution(dist_name)
        dist_final.fit(clean_data, method='mle')
        
        return dist_final, clean_data
    
    # Use it
    data_contaminated = np.concatenate([
        np.random.normal(10, 2, 950),
        np.random.uniform(30, 40, 50)  # 5% outliers
    ])
    
    dist_robust, clean = robust_fit(data_contaminated, 'normal')
    print(dist_robust.summary())

Custom Distribution Class
-------------------------

**Create your own distribution:**

.. code-block:: python

    from distfit_pro.core.distributions import BaseDistribution, DistributionInfo
    from scipy import stats
    import numpy as np
    
    class MyCustomDistribution(BaseDistribution):
        def __init__(self):
            super().__init__()
            self._scipy_dist = stats.gennorm  # Generalized normal
        
        @property
        def info(self) -> DistributionInfo:
            return DistributionInfo(
                name="custom_gennorm",
                display_name="My Generalized Normal",
                parameters={"beta": "shape", "loc": "location", "scale": "scale"},
                support="(-∞, +∞)",
                use_cases=["Flexible tail behavior"],
                characteristics=["Subsumes Laplace, Normal, Uniform"]
            )
        
        def pdf(self, x):
            return self._scipy_dist.pdf(x, self.params['beta'], 
                                       self.params['loc'], 
                                       self.params['scale'])
        
        def cdf(self, x):
            return self._scipy_dist.cdf(x, self.params['beta'],
                                       self.params['loc'],
                                       self.params['scale'])
        
        def ppf(self, q):
            return self._scipy_dist.ppf(q, self.params['beta'],
                                       self.params['loc'],
                                       self.params['scale'])
        
        def fit_mle(self, data, **kwargs):
            params = self._scipy_dist.fit(data)
            return {'beta': params[0], 'loc': params[1], 'scale': params[2]}
        
        def fit_moments(self, data):
            # Fallback to MLE for complex distributions
            return self.fit_mle(data)
    
    # Use custom distribution
    data = np.random.standard_t(5, 1000)
    
    custom_dist = MyCustomDistribution()
    custom_dist.fit(data)
    
    print(custom_dist.summary())

Batch Processing
----------------

**Process multiple datasets:**

.. code-block:: python

    def batch_fit_distributions(datasets, dist_name='normal'):
        """
        Fit same distribution to multiple datasets
        """
        results = []
        
        for i, data in enumerate(datasets):
            dist = get_distribution(dist_name)
            try:
                dist.fit(data)
                
                # GOF test
                ks_result = GOFTests.kolmogorov_smirnov(data, dist)
                
                results.append({
                    'dataset': i,
                    'n': len(data),
                    'params': dist.params,
                    'ks_pvalue': ks_result.p_value,
                    'good_fit': not ks_result.reject_null
                })
            except Exception as e:
                results.append({
                    'dataset': i,
                    'error': str(e)
                })
        
        return results
    
    # Generate multiple datasets
    datasets = [np.random.normal(10+i, 2, 500) for i in range(5)]
    
    results = batch_fit_distributions(datasets)
    
    print("\nBatch Fitting Results:")
    for r in results:
        if 'error' not in r:
            print(f"Dataset {r['dataset']}: "
                  f"mean={r['params']['loc']:.2f}, "
                  f"p-value={r['ks_pvalue']:.4f}")

Export Results
--------------

**Save to CSV/JSON:**

.. code-block:: python

    import pandas as pd
    import json
    
    # Fit distribution
    dist.fit(data)
    
    # To dict
    result_dict = {
        'distribution': dist.info.name,
        'parameters': dist.params,
        'statistics': {
            'mean': dist.mean(),
            'std': dist.std(),
            'median': dist.median()
        }
    }
    
    # Save JSON
    with open('fit_result.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    # Save CSV (for batch results)
    df_results = pd.DataFrame(batch_results)
    df_results.to_csv('batch_results.csv', index=False)

Next Steps
----------

- :doc:`../examples/index` - Real-world examples
- :doc:`../api/index` - Complete API reference
- :doc:`../contributing` - Contribute to DistFit Pro
