Real-World Examples
===================

Practical examples from various domains.

Reliability Engineering
-----------------------

**Problem:** Analyze component failure times.

.. code-block:: python

    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Failure time data (hours)
    failure_times = np.array([
        120, 145, 167, 189, 201, 234, 267, 289,
        312, 345, 378, 401, 423, 456, 489, 512,
        534, 567, 589, 612, 645, 678, 701, 723
    ])
    
    # Fit Weibull (common in reliability)
    dist = get_distribution('weibull')
    dist.fit(failure_times)
    
    print("Weibull Fit Results:")
    print(dist.summary())
    
    # GOF test
    ks_result = GOFTests.kolmogorov_smirnov(failure_times, dist)
    print(f"\nKS test p-value: {ks_result.p_value:.4f}")
    
    # Reliability metrics
    print("\nReliability Metrics:")
    for t in [200, 400, 600]:
        R_t = dist.reliability(t)
        h_t = dist.hazard_rate(t)
        print(f"  t={t}h: R(t)={R_t:.4f}, h(t)={h_t:.6f}")
    
    # Mean Time To Failure
    mttf = dist.mean_time_to_failure()
    print(f"\nMTTF: {mttf:.1f} hours")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability function
    t = np.linspace(0, 1000, 200)
    R = np.array([dist.reliability(ti) for ti in t])
    
    axes[0].plot(t, R, 'b-', lw=2)
    axes[0].set_xlabel('Time (hours)')
    axes[0].set_ylabel('Reliability R(t)')
    axes[0].set_title('Reliability Function')
    axes[0].grid(True, alpha=0.3)
    
    # Hazard rate
    h = np.array([dist.hazard_rate(ti) for ti in t])
    
    axes[1].plot(t, h, 'r-', lw=2)
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_ylabel('Hazard Rate h(t)')
    axes[1].set_title('Hazard Function')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

Financial Risk Analysis
-----------------------

**Problem:** Estimate Value-at-Risk for stock returns.

.. code-block:: python

    # Daily returns (simulated)
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=1000) * 0.02
    
    # Fit Student-t (captures heavy tails)
    dist = get_distribution('studentt')
    dist.fit(returns)
    
    print("Student-t Fit:")
    print(f"df = {dist.params['df']:.2f}")
    print(f"loc = {dist.params['loc']:.6f}")
    print(f"scale = {dist.params['scale']:.6f}")
    
    # Risk metrics
    print("\nRisk Metrics:")
    
    # VaR at different confidence levels
    for alpha in [0.01, 0.05, 0.10]:
        var = dist.ppf(alpha)
        cvar = dist.conditional_var(alpha)
        
        print(f"\n{int((1-alpha)*100)}% Confidence:")
        print(f"  VaR:  {var:.4f} ({var*100:.2f}%)")
        print(f"  CVaR: {cvar:.4f} ({cvar*100:.2f}%)")

Quality Control
---------------

**Problem:** Monitor defect rates in manufacturing.

.. code-block:: python

    # Number of defects per batch (100 batches)
    defects = np.random.poisson(lam=2.5, size=100)
    
    # Fit Poisson
    dist = get_distribution('poisson')
    dist.fit(defects)
    
    print(f"Estimated defect rate: {dist.params['mu']:.3f} per batch")
    
    # Control limits (3-sigma)
    mean = dist.mean()
    std = dist.std()
    
    ucl = mean + 3 * std  # Upper control limit
    lcl = max(0, mean - 3 * std)  # Lower control limit
    
    print(f"\nControl Limits:")
    print(f"  UCL: {ucl:.2f}")
    print(f"  Mean: {mean:.2f}")
    print(f"  LCL: {lcl:.2f}")
    
    # Check if process is in control
    out_of_control = np.sum((defects > ucl) | (defects < lcl))
    print(f"\nBatches out of control: {out_of_control}/100")
    
    if out_of_control == 0:
        print("✅ Process is in statistical control")
    else:
        print("⚠ Process may be out of control")

Survey Analysis
---------------

**Problem:** Analyze income data with sampling weights.

.. code-block:: python

    from distfit_pro.core.weighted import WeightedFitting
    
    # Stratified sample
    # Stratum 1 (High income): 20% of pop, 40% of sample
    # Stratum 2 (Mid income): 50% of pop, 40% of sample
    # Stratum 3 (Low income): 30% of pop, 20% of sample
    
    income_high = np.random.lognormal(11.5, 0.4, 400)
    income_mid = np.random.lognormal(10.8, 0.5, 400)
    income_low = np.random.lognormal(10.2, 0.6, 200)
    
    income = np.concatenate([income_high, income_mid, income_low])
    
    # Sampling weights (inverse of sampling probability)
    weights = np.concatenate([
        np.ones(400) * (0.20 / 0.40),  # High: pop_prop / sample_prop
        np.ones(400) * (0.50 / 0.40),  # Mid
        np.ones(200) * (0.30 / 0.20)   # Low
    ])
    
    # Weighted fit
    dist = get_distribution('lognormal')
    params_weighted = WeightedFitting.fit_weighted_mle(
        income, weights, dist
    )
    
    dist.params = params_weighted
    dist.fitted = True
    
    # Population estimates
    print("Population Income Estimates:")
    print(f"Median: ${dist.median()/1000:.1f}K")
    print(f"Mean: ${dist.mean()/1000:.1f}K")
    
    # Percentiles
    for p in [10, 25, 50, 75, 90]:
        val = dist.ppf(p/100)
        print(f"{p}th percentile: ${val/1000:.1f}K")
    
    # Effective sample size
    ess = WeightedFitting.effective_sample_size(weights)
    print(f"\nEffective sample size: {ess:.0f} (actual: {len(income)})")
