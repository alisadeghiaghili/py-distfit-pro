Tutorial 9: Advanced Topics
===========================

Advanced techniques for expert users.

Custom Distributions
--------------------

**Create your own distribution class:**

.. code-block:: python

    from distfit_pro.core.distributions import BaseDistribution, DistributionInfo
    import numpy as np
    from scipy import stats
    
    class CustomDistribution(BaseDistribution):
        def __init__(self):
            super().__init__()
            self._scipy_dist = stats.expon  # Use scipy for some methods
        
        @property
        def info(self) -> DistributionInfo:
            return DistributionInfo(
                name="custom",
                display_name="My Custom Distribution",
                parameters={"rate": "λ (rate parameter)"},
                support="(0, ∞)",
                use_cases=["Custom application"],
                characteristics=["Special behavior"]
            )
        
        def pdf(self, x):
            # Custom PDF
            rate = self.params['rate']
            return rate * np.exp(-rate * x)
        
        def cdf(self, x):
            rate = self.params['rate']
            return 1 - np.exp(-rate * x)
        
        def ppf(self, q):
            rate = self.params['rate']
            return -np.log(1 - q) / rate
        
        def fit_mle(self, data, **kwargs):
            # MLE for exponential: rate = 1/mean
            return {'rate': 1 / np.mean(data[data > 0])}
        
        def fit_moments(self, data):
            return self.fit_mle(data)
    
    # Use custom distribution
    data = np.random.exponential(2, 1000)
    
    custom_dist = CustomDistribution()
    custom_dist.fit(data)
    print(custom_dist.summary())

Model Selection Workflow
------------------------

**Automated best distribution selection:**

.. code-block:: python

    from distfit_pro import get_distribution, list_continuous_distributions
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    
    def find_best_distribution(data, candidates=None, verbose=True):
        """
        Find best fitting distribution using multiple criteria.
        """
        if candidates is None:
            candidates = list_continuous_distributions()
        
        results = []
        
        for dist_name in candidates:
            try:
                # Fit distribution
                dist = get_distribution(dist_name)
                dist.fit(data, method='mle')
                
                # Calculate metrics
                n = len(data)
                k = len(dist.params)
                log_lik = np.sum(dist.logpdf(data))
                
                aic = 2 * k - 2 * log_lik
                bic = k * np.log(n) - 2 * log_lik
                
                # GOF test
                ks_result = GOFTests.kolmogorov_smirnov(data, dist)
                
                results.append({
                    'name': dist_name,
                    'dist': dist,
                    'aic': aic,
                    'bic': bic,
                    'ks_statistic': ks_result.statistic,
                    'ks_p_value': ks_result.p_value,
                    'log_likelihood': log_lik
                })
                
            except Exception as e:
                if verbose:
                    print(f"Failed to fit {dist_name}: {e}")
        
        # Sort by AIC
        results.sort(key=lambda x: x['aic'])
        
        if verbose:
            print(f"\n{'Distribution':<20} {'AIC':<12} {'BIC':<12} {'KS p-value':<12}")
            print("-" * 60)
            for r in results[:10]:  # Top 10
                print(f"{r['name']:<20} {r['aic']:<12.2f} {r['bic']:<12.2f} {r['ks_p_value']:<12.4f}")
        
        return results
    
    # Example usage
    np.random.seed(42)
    data = np.random.gamma(2, 3, 1000)
    
    results = find_best_distribution(data)
    best = results[0]
    
    print(f"\nBest distribution: {best['name']}")
    print(f"AIC: {best['aic']:.2f}")

Mixture Models
--------------

**Fit mixture of distributions:**

.. code-block:: python

    from scipy import stats
    from scipy.optimize import minimize
    
    def fit_gaussian_mixture(data, n_components=2):
        """
        Fit Gaussian mixture model using EM algorithm.
        
        Simple implementation for demonstration.
        """
        # Initialize parameters
        means = np.random.choice(data, n_components)
        stds = np.ones(n_components) * np.std(data)
        weights = np.ones(n_components) / n_components
        
        # EM iterations
        for iteration in range(100):
            # E-step: Calculate responsibilities
            responsibilities = np.zeros((len(data), n_components))
            
            for k in range(n_components):
                responsibilities[:, k] = weights[k] * stats.norm.pdf(
                    data, means[k], stds[k]
                )
            
            responsibilities /= responsibilities.sum(axis=1, keepdims=True)
            
            # M-step: Update parameters
            for k in range(n_components):
                Nk = responsibilities[:, k].sum()
                weights[k] = Nk / len(data)
                means[k] = (responsibilities[:, k] * data).sum() / Nk
                stds[k] = np.sqrt(
                    (responsibilities[:, k] * (data - means[k])**2).sum() / Nk
                )
        
        return {
            'means': means,
            'stds': stds,
            'weights': weights
        }
    
    # Generate mixture data
    np.random.seed(42)
    component1 = np.random.normal(5, 1, 500)
    component2 = np.random.normal(15, 2, 500)
    mixture_data = np.concatenate([component1, component2])
    
    # Fit mixture
    params = fit_gaussian_mixture(mixture_data, n_components=2)
    
    print("Mixture parameters:")
    for i in range(2):
        print(f"Component {i+1}: μ={params['means'][i]:.2f}, "
              f"σ={params['stds'][i]:.2f}, w={params['weights'][i]:.2f}")

Truncated Distributions
-----------------------

**Fit distributions to truncated data:**

.. code-block:: python

    from scipy.stats import truncnorm
    
    def fit_truncated_normal(data, lower=None, upper=None):
        """
        Fit truncated normal distribution.
        """
        if lower is None:
            lower = data.min()
        if upper is None:
            upper = data.max()
        
        # Initial estimates from data
        loc_init = np.mean(data)
        scale_init = np.std(data)
        
        # Convert to standard truncated normal parameters
        a = (lower - loc_init) / scale_init
        b = (upper - loc_init) / scale_init
        
        # Fit
        params = truncnorm.fit(data, a, b)
        
        return {
            'a': params[0],
            'b': params[1],
            'loc': params[2],
            'scale': params[3],
            'lower_bound': lower,
            'upper_bound': upper
        }
    
    # Example: ages (18-65)
    ages = np.random.normal(40, 10, 1000)
    ages_truncated = ages[(ages >= 18) & (ages <= 65)]
    
    params = fit_truncated_normal(ages_truncated, lower=18, upper=65)
    print(f"Truncated normal: μ={params['loc']:.2f}, σ={params['scale']:.2f}")

Censored Data
-------------

**Handle left/right censored observations:**

.. code-block:: python

    def fit_censored_normal(data, censored_mask, censor_type='right', censor_value=None):
        """
        Fit normal distribution to right-censored data.
        
        Parameters:
        -----------
        data : array
            Observed values (use censor_value for censored obs)
        censored_mask : array
            True for censored observations
        censor_type : str
            'right' or 'left'
        censor_value : float
            Censoring threshold
        """
        from scipy.optimize import minimize
        
        if censor_value is None:
            censor_value = data[censored_mask][0]
        
        def neg_log_likelihood(params):
            mu, sigma = params
            if sigma <= 0:
                return np.inf
            
            # Uncensored observations: log PDF
            uncensored_ll = np.sum(
                stats.norm.logpdf(data[~censored_mask], mu, sigma)
            )
            
            # Censored observations: log survival function
            if censor_type == 'right':
                censored_ll = np.sum(
                    stats.norm.logsf(censor_value, mu, sigma)
                ) * censored_mask.sum()
            else:  # left censored
                censored_ll = np.sum(
                    stats.norm.logcdf(censor_value, mu, sigma)
                ) * censored_mask.sum()
            
            return -(uncensored_ll + censored_ll)
        
        # Initial values
        mu_init = np.mean(data[~censored_mask])
        sigma_init = np.std(data[~censored_mask])
        
        # Optimize
        result = minimize(
            neg_log_likelihood,
            [mu_init, sigma_init],
            method='Nelder-Mead'
        )
        
        return {'mu': result.x[0], 'sigma': result.x[1]}
    
    # Example: detection limit
    true_data = np.random.lognormal(2, 0.5, 1000)
    detection_limit = 5
    
    # Censor values below detection limit
    observed = true_data.copy()
    censored_mask = observed < detection_limit
    observed[censored_mask] = detection_limit
    
    # Fit with censoring
    params = fit_censored_normal(
        np.log(observed),
        censored_mask,
        censor_type='left',
        censor_value=np.log(detection_limit)
    )
    
    print(f"Censored fit: μ={params['mu']:.2f}, σ={params['sigma']:.2f}")
    print(f"True: μ={2:.2f}, σ={0.5:.2f}")

Bayesian Parameter Estimation
-----------------------------

**Simple Bayesian inference:**

.. code-block:: python

    def bayesian_normal_inference(data, prior_mu=0, prior_sigma=10):
        """
        Bayesian inference for normal distribution with conjugate prior.
        """
        n = len(data)
        sample_mean = np.mean(data)
        sample_var = np.var(data, ddof=1)
        
        # Posterior for mu (assuming known sigma)
        # Prior: μ ~ N(prior_mu, prior_sigma²)
        # Likelihood: X ~ N(μ, σ²)
        
        posterior_var = 1 / (1/prior_sigma**2 + n/sample_var)
        posterior_mean = posterior_var * (
            prior_mu/prior_sigma**2 + n*sample_mean/sample_var
        )
        
        return {
            'posterior_mean': posterior_mean,
            'posterior_std': np.sqrt(posterior_var),
            'prior_mean': prior_mu,
            'prior_std': prior_sigma
        }
    
    # Example
    data = np.random.normal(10, 2, 100)
    
    # Weakly informative prior
    result = bayesian_normal_inference(data, prior_mu=0, prior_sigma=10)
    
    print(f"Prior: μ ~ N({result['prior_mean']}, {result['prior_std']:.2f})")
    print(f"Posterior: μ ~ N({result['posterior_mean']:.2f}, {result['posterior_std']:.2f})")
    print(f"95% Credible Interval: [{result['posterior_mean'] - 1.96*result['posterior_std']:.2f}, "
          f"{result['posterior_mean'] + 1.96*result['posterior_std']:.2f}]")

Time Series of Distributions
----------------------------

**Track distribution changes over time:**

.. code-block:: python

    def rolling_distribution_fit(time_series, window=100, dist_name='normal'):
        """
        Fit distribution to rolling windows.
        """
        n = len(time_series)
        results = []
        
        for i in range(window, n):
            window_data = time_series[i-window:i]
            
            dist = get_distribution(dist_name)
            dist.fit(window_data)
            
            results.append({
                'index': i,
                'params': dist.params.copy(),
                'mean': dist.mean(),
                'std': dist.std()
            })
        
        return results
    
    # Example: changing volatility
    np.random.seed(42)
    n = 1000
    
    # Generate data with increasing volatility
    volatility = np.linspace(1, 3, n)
    returns = np.random.normal(0, volatility)
    
    # Rolling fit
    rolling_results = rolling_distribution_fit(returns, window=100)
    
    # Plot volatility changes
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    indices = [r['index'] for r in rolling_results]
    rolling_std = [r['std'] for r in rolling_results]
    
    ax.plot(indices, rolling_std, label='Estimated σ')
    ax.plot(range(n), volatility, '--', label='True σ', alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Volatility')
    ax.set_title('Rolling Distribution Fit')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.show()

Batch Processing
----------------

**Efficiently fit multiple datasets:**

.. code-block:: python

    from joblib import Parallel, delayed
    
    def fit_dataset(data_id, data, dist_name='normal'):
        """Fit single dataset."""
        dist = get_distribution(dist_name)
        dist.fit(data)
        
        return {
            'id': data_id,
            'params': dist.params,
            'aic': 2 * len(dist.params) - 2 * np.sum(dist.logpdf(data))
        }
    
    def batch_fit(datasets, dist_name='normal', n_jobs=-1):
        """Fit multiple datasets in parallel."""
        results = Parallel(n_jobs=n_jobs)(
            delayed(fit_dataset)(i, data, dist_name)
            for i, data in enumerate(datasets)
        )
        return results
    
    # Example: 100 datasets
    np.random.seed(42)
    datasets = [np.random.gamma(2, 3, 500) for _ in range(100)]
    
    import time
    start = time.time()
    results = batch_fit(datasets, dist_name='gamma', n_jobs=-1)
    elapsed = time.time() - start
    
    print(f"Fitted {len(datasets)} datasets in {elapsed:.2f}s")
    print(f"Average AIC: {np.mean([r['aic'] for r in results]):.2f}")

Custom Goodness-of-Fit Test
---------------------------

**Implement your own GOF test:**

.. code-block:: python

    def custom_gof_test(data, dist):
        """
        Custom GOF test based on moments.
        """
        # Sample moments
        m1 = np.mean(data)
        m2 = np.mean(data**2)
        m3 = np.mean(data**3)
        
        # Theoretical moments from fitted distribution
        # (approximated by simulation)
        simulated = dist.rvs(size=10000, random_state=42)
        t1 = np.mean(simulated)
        t2 = np.mean(simulated**2)
        t3 = np.mean(simulated**3)
        
        # Test statistic: sum of squared differences
        test_stat = (
            ((m1 - t1) / np.std(data))**2 +
            ((m2 - t2) / np.std(data**2))**2 +
            ((m3 - t3) / np.std(data**3))**2
        )
        
        # Bootstrap p-value
        boot_stats = []
        for _ in range(1000):
            boot_data = dist.rvs(size=len(data))
            bm1 = np.mean(boot_data)
            bm2 = np.mean(boot_data**2)
            bm3 = np.mean(boot_data**3)
            
            boot_stat = (
                ((bm1 - t1) / np.std(boot_data))**2 +
                ((bm2 - t2) / np.std(boot_data**2))**2 +
                ((bm3 - t3) / np.std(boot_data**3))**2
            )
            boot_stats.append(boot_stat)
        
        p_value = np.mean(np.array(boot_stats) >= test_stat)
        
        return {
            'statistic': test_stat,
            'p_value': p_value,
            'reject': p_value < 0.05
        }
    
    # Test
    data = np.random.normal(10, 2, 500)
    dist = get_distribution('normal')
    dist.fit(data)
    
    result = custom_gof_test(data, dist)
    print(f"Custom GOF test: stat={result['statistic']:.4f}, p={result['p_value']:.4f}")

Performance Optimization
------------------------

**Tips for faster fitting:**

1. **Use appropriate sample size**
   
   .. code-block:: python
   
       # For quick prototyping, sample large datasets
       if len(data) > 10000:
           sample = np.random.choice(data, 10000, replace=False)
       else:
           sample = data

2. **Cache distributions**
   
   .. code-block:: python
   
       # Don't recreate distribution objects
       dist = get_distribution('normal')  # Once
       
       for dataset in datasets:
           dist.fit(dataset)  # Reuse object

3. **Use moments for speed**
   
   .. code-block:: python
   
       # MLE is accurate but slow
       dist.fit(data, method='moments')  # Much faster

4. **Parallel processing**
   
   Already shown in batch processing example.

Next Steps
----------

- :doc:`../api/index` - Complete API reference
- :doc:`../examples/real_world` - Complex real-world examples
- :doc:`../contributing` - Contribute new features
