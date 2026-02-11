Real-World Applications
=======================

Complete, realistic examples from industry.

Insurance Claims Analysis
-------------------------

**Modeling claim amounts for risk assessment:**

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    from distfit_pro.core.bootstrap import Bootstrap
    import numpy as np
    import pandas as pd
    
    # Simulated insurance claims ($)
    np.random.seed(42)
    claims = np.random.lognormal(np.log(5000), 1.2, 2000)
    claims = claims[claims < 100000]  # Remove extreme outliers
    
    print(f"Total claims: {len(claims)}")
    print(f"Total payout: ${np.sum(claims):,.0f}")
    print(f"Average claim: ${np.mean(claims):,.0f}")
    
    # Try multiple distributions
    candidates = ['lognormal', 'gamma', 'weibull', 'pareto']
    
    results = {}
    for dist_name in candidates:
        dist = get_distribution(dist_name)
        try:
            dist.fit(claims, method='mle')
            
            # GOF test
            ks = GOFTests.kolmogorov_smirnov(claims, dist)
            
            # AIC
            n = len(claims)
            k = len(dist.params)
            log_lik = np.sum(dist.logpdf(claims))
            aic = 2 * k - 2 * log_lik
            
            results[dist_name] = {
                'dist': dist,
                'aic': aic,
                'p_value': ks.p_value
            }
        except:
            pass
    
    # Select best
    best_name = min(results.keys(), key=lambda x: results[x]['aic'])
    best_dist = results[best_name]['dist']
    
    print(f"\nBest distribution: {best_name}")
    print(f"AIC: {results[best_name]['aic']:.0f}")
    
    # Risk metrics
    print("\n=== RISK METRICS ===")
    
    # Value at Risk (95%)
    var_95 = best_dist.ppf(0.95)
    print(f"VaR (95%): ${var_95:,.0f}")
    
    # Conditional VaR (Expected Shortfall)
    cvar_95 = best_dist.conditional_var(0.95)
    print(f"CVaR (95%): ${cvar_95:,.0f}")
    
    # Probability of large claim (> $20,000)
    prob_large = 1 - best_dist.cdf(np.array([20000]))[0]
    print(f"P(claim > $20K): {prob_large:.2%}")
    
    # Bootstrap CI for VaR
    print("\nBootstrap 95% CI for VaR:")
    ci = Bootstrap.parametric(claims, best_dist, n_bootstrap=1000)
    
    # Recalculate VaR for bootstrap samples
    var_samples = []
    for i in range(1000):
        boot_sample = best_dist.rvs(size=len(claims))
        boot_dist = get_distribution(best_name)
        boot_dist.fit(boot_sample)
        var_samples.append(boot_dist.ppf(0.95))
    
    var_ci = np.percentile(var_samples, [2.5, 97.5])
    print(f"VaR CI: [${var_ci[0]:,.0f}, ${var_ci[1]:,.0f}]")

Manufacturing Process Control
------------------------------

**Statistical process control with control charts:**

.. code-block:: python

    # Measurements from production line (mm)
    np.random.seed(42)
    
    # Normal operation
    normal_data = np.random.normal(100, 0.5, 500)
    
    # Process shift at sample 300
    shifted_data = np.random.normal(100.3, 0.5, 200)
    
    all_measurements = np.concatenate([normal_data[:300], shifted_data])
    
    # Fit to baseline (first 100 samples)
    baseline = all_measurements[:100]
    dist = get_distribution('normal')
    dist.fit(baseline)
    
    print("Baseline Distribution:")
    print(f"  Target: {dist.mean():.3f} mm")
    print(f"  Std: {dist.std():.3f} mm")
    
    # Control limits (3-sigma)
    ucl = dist.mean() + 3 * dist.std()
    lcl = dist.mean() - 3 * dist.std()
    
    print(f"\nControl Limits:")
    print(f"  UCL: {ucl:.3f} mm")
    print(f"  LCL: {lcl:.3f} mm")
    
    # Detect out-of-control points
    out_of_control = (all_measurements > ucl) | (all_measurements < lcl)
    print(f"\nOut-of-control points: {np.sum(out_of_control)}")
    
    # Plot control chart
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(14, 6))
    plt.plot(all_measurements, 'o-', markersize=3, alpha=0.6)
    plt.axhline(dist.mean(), color='green', linestyle='-', 
                lw=2, label='Target')
    plt.axhline(ucl, color='red', linestyle='--', lw=2, label='UCL')
    plt.axhline(lcl, color='red', linestyle='--', lw=2, label='LCL')
    
    # Highlight out-of-control
    plt.scatter(np.where(out_of_control)[0],
                all_measurements[out_of_control],
                color='red', s=100, marker='x', 
                linewidth=3, label='Out of control')
    
    # Mark process shift
    plt.axvline(300, color='orange', linestyle=':', 
                lw=2, alpha=0.7, label='Process shift')
    
    plt.xlabel('Sample Number')
    plt.ylabel('Measurement (mm)')
    plt.title('Statistical Process Control Chart')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('control_chart.png', dpi=300)
    plt.show()

A/B Test Analysis
-----------------

**Comparing conversion rates:**

.. code-block:: python

    # A/B test data
    # Variant A: 5000 visitors, 230 conversions
    # Variant B: 5000 visitors, 267 conversions
    
    conversions_A = np.array([1]*230 + [0]*(5000-230))
    conversions_B = np.array([1]*267 + [0]*(5000-267))
    
    conv_rate_A = np.mean(conversions_A)
    conv_rate_B = np.mean(conversions_B)
    
    print(f"Variant A: {conv_rate_A:.2%}")
    print(f"Variant B: {conv_rate_B:.2%}")
    print(f"Lift: {(conv_rate_B/conv_rate_A - 1)*100:.1f}%")
    
    # Bayesian approach with Beta distribution
    # Prior: Beta(1, 1) = Uniform
    # Posterior: Beta(1 + successes, 1 + failures)
    
    from scipy.stats import beta
    
    # Posteriors
    alpha_A = 1 + np.sum(conversions_A)
    beta_A = 1 + len(conversions_A) - np.sum(conversions_A)
    
    alpha_B = 1 + np.sum(conversions_B)
    beta_B = 1 + len(conversions_B) - np.sum(conversions_B)
    
    # Sample from posteriors
    n_samples = 10000
    samples_A = beta.rvs(alpha_A, beta_A, size=n_samples)
    samples_B = beta.rvs(alpha_B, beta_B, size=n_samples)
    
    # Probability B > A
    prob_B_better = np.mean(samples_B > samples_A)
    
    print(f"\nBayesian Analysis:")
    print(f"P(B > A): {prob_B_better:.1%}")
    
    # Expected lift
    lift_samples = (samples_B - samples_A) / samples_A
    expected_lift = np.median(lift_samples)
    lift_ci = np.percentile(lift_samples, [2.5, 97.5])
    
    print(f"Expected lift: {expected_lift*100:.1f}%")
    print(f"95% CI: [{lift_ci[0]*100:.1f}%, {lift_ci[1]*100:.1f}%]")
    
    if prob_B_better > 0.95:
        print("\n✅ B is significantly better! Deploy variant B.")
    else:
        print("\n⚠️  Not enough evidence. Continue testing.")
