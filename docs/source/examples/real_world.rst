Real-World Examples
===================

Complete end-to-end examples using real-world scenarios.

Example 1: Quality Control in Manufacturing
--------------------------------------------

**Scenario:** A factory measures bolt diameters. Need to:

1. Fit distribution to measurements
2. Test goodness-of-fit
3. Detect defective bolts (outliers)
4. Estimate defect rate with confidence intervals
5. Set quality control limits

**Complete Code:**

.. code-block:: python

    import numpy as np
    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    from distfit_pro.core.diagnostics import Diagnostics
    from distfit_pro.core.bootstrap import Bootstrap
    import matplotlib.pyplot as plt
    
    # Simulated measurements (mm)
    np.random.seed(42)
    measurements = np.concatenate([
        np.random.normal(10.0, 0.05, 950),  # Good bolts
        np.random.uniform(9.5, 10.5, 50)    # 5% defects
    ])
    np.random.shuffle(measurements)
    
    print(f"Total measurements: {len(measurements)}")
    print(f"Mean: {np.mean(measurements):.4f} mm")
    print(f"Std: {np.std(measurements):.4f} mm")
    
    # Step 1: Fit distribution
    print("\n" + "="*60)
    print("STEP 1: Fit Distribution")
    print("="*60)
    
    dist = get_distribution('normal')
    dist.fit(measurements, method='mle')
    
    print(dist.summary())
    
    # Step 2: Test goodness-of-fit
    print("\n" + "="*60)
    print("STEP 2: Goodness-of-Fit Tests")
    print("="*60)
    
    gof_results = GOFTests.run_all_tests(measurements, dist)
    print(GOFTests.summary_table(gof_results))
    
    # Step 3: Residual analysis
    print("\n" + "="*60)
    print("STEP 3: Residual Analysis")
    print("="*60)
    
    residuals = Diagnostics.residual_analysis(measurements, dist)
    print(residuals.summary())
    
    # Step 4: Detect outliers (defects)
    print("\n" + "="*60)
    print("STEP 4: Outlier Detection")
    print("="*60)
    
    # Use 2.5 sigma for stricter QC
    outliers = Diagnostics.detect_outliers(
        measurements, dist,
        method='zscore',
        threshold=2.5
    )
    
    print(outliers.summary())
    
    defect_rate = len(outliers.outlier_indices) / len(measurements)
    print(f"\nEstimated defect rate: {defect_rate*100:.2f}%")
    
    # Step 5: Bootstrap confidence interval for defect rate
    print("\n" + "="*60)
    print("STEP 5: Bootstrap CI for Defect Rate")
    print("="*60)
    
    def calculate_defect_rate(data):
        d = get_distribution('normal')
        d.fit(data)
        outl = Diagnostics.detect_outliers(data, d, method='zscore', threshold=2.5)
        return len(outl.outlier_indices) / len(data)
    
    boot_defect_rates = []
    for i in range(1000):
        boot_sample = np.random.choice(measurements, size=len(measurements), replace=True)
        boot_defect_rates.append(calculate_defect_rate(boot_sample))
    
    boot_defect_rates = np.array(boot_defect_rates)
    defect_ci_lower = np.percentile(boot_defect_rates, 2.5)
    defect_ci_upper = np.percentile(boot_defect_rates, 97.5)
    
    print(f"Defect Rate: {defect_rate*100:.2f}%")
    print(f"95% CI: [{defect_ci_lower*100:.2f}%, {defect_ci_upper*100:.2f}%]")
    
    # Step 6: Set control limits
    print("\n" + "="*60)
    print("STEP 6: Quality Control Limits")
    print("="*60)
    
    # 3-sigma limits
    ucl = dist.ppf(0.9985)  # Upper control limit
    lcl = dist.ppf(0.0015)  # Lower control limit
    target = dist.mean()
    
    print(f"Target: {target:.4f} mm")
    print(f"LCL (3σ): {lcl:.4f} mm")
    print(f"UCL (3σ): {ucl:.4f} mm")
    print(f"Control range: [{lcl:.4f}, {ucl:.4f}] mm")
    
    # Step 7: Visualization
    print("\n" + "="*60)
    print("STEP 7: Visualization")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogram + PDF
    ax = axes[0, 0]
    ax.hist(measurements, bins=50, density=True, alpha=0.7, label='Data')
    x = np.linspace(measurements.min(), measurements.max(), 200)
    ax.plot(x, dist.pdf(x), 'r-', linewidth=2, label='Fitted Normal')
    ax.axvline(target, color='g', linestyle='--', label='Target')
    ax.axvline(lcl, color='orange', linestyle=':', label='LCL/UCL')
    ax.axvline(ucl, color='orange', linestyle=':')
    ax.set_xlabel('Diameter (mm)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Fit')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Q-Q Plot
    ax = axes[0, 1]
    qq_data = Diagnostics.qq_diagnostics(measurements, dist)
    ax.scatter(qq_data['theoretical'], qq_data['sample'], alpha=0.5)
    lims = [qq_data['theoretical'].min(), qq_data['theoretical'].max()]
    ax.plot(lims, lims, 'r--', label='Perfect fit')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f"Q-Q Plot (r={qq_data['correlation']:.4f})")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Control Chart
    ax = axes[1, 0]
    ax.scatter(range(len(measurements)), measurements, alpha=0.5, s=10)
    ax.axhline(target, color='g', linestyle='-', linewidth=2, label='Target')
    ax.axhline(ucl, color='r', linestyle='--', label='UCL')
    ax.axhline(lcl, color='r', linestyle='--', label='LCL')
    # Mark outliers
    outlier_idx = outliers.outlier_indices
    ax.scatter(outlier_idx, measurements[outlier_idx], color='red', s=50, 
               marker='x', label='Outliers')
    ax.set_xlabel('Measurement Index')
    ax.set_ylabel('Diameter (mm)')
    ax.set_title('Control Chart')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Bootstrap distribution of defect rate
    ax = axes[1, 1]
    ax.hist(boot_defect_rates * 100, bins=30, density=True, alpha=0.7)
    ax.axvline(defect_rate * 100, color='r', linewidth=2, label='Observed')
    ax.axvline(defect_ci_lower * 100, color='orange', linestyle='--', label='95% CI')
    ax.axvline(defect_ci_upper * 100, color='orange', linestyle='--')
    ax.set_xlabel('Defect Rate (%)')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap Distribution of Defect Rate')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('quality_control_analysis.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'quality_control_analysis.png'")
    
    # Final Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"• Distribution: Normal({dist.params['loc']:.4f}, {dist.params['scale']:.4f})")
    print(f"• GOF Tests: {'PASS' if not any(r.reject_null for r in gof_results.values()) else 'FAIL'}")
    print(f"• Defect Rate: {defect_rate*100:.2f}% (95% CI: {defect_ci_lower*100:.2f}%-{defect_ci_upper*100:.2f}%)")
    print(f"• Control Limits: [{lcl:.4f}, {ucl:.4f}] mm")
    
    if defect_rate < 0.01:
        print("\n✅ Process is in control. Defect rate < 1%")
    elif defect_rate < 0.05:
        print("\n⚠️  Process needs monitoring. Defect rate 1-5%")
    else:
        print("\n❌ Process out of control. Defect rate > 5%")

Example 2: Financial Risk Analysis
-----------------------------------

**Scenario:** Estimate Value-at-Risk (VaR) for stock portfolio.

**Complete Code:**

.. code-block:: python

    import numpy as np
    from distfit_pro import get_distribution
    from distfit_pro.core.gof_tests import GOFTests
    from distfit_pro.core.bootstrap import Bootstrap
    import matplotlib.pyplot as plt
    
    # Simulated daily returns
    np.random.seed(42)
    returns = np.random.standard_t(df=5, size=1000) * 0.02  # Heavy tails
    
    print(f"Returns statistics:")
    print(f"  Mean: {np.mean(returns)*100:.2f}%")
    print(f"  Std: {np.std(returns)*100:.2f}%")
    print(f"  Skewness: {np.mean((returns - np.mean(returns))**3) / np.std(returns)**3:.2f}")
    
    # Try multiple distributions
    print("\n" + "="*60)
    print("STEP 1: Model Selection")
    print("="*60)
    
    candidates = {
        'normal': get_distribution('normal'),
        'studentt': get_distribution('studentt'),
        'laplace': get_distribution('laplace'),
        'logistic': get_distribution('logistic')
    }
    
    results = {}
    for name, dist in candidates.items():
        try:
            dist.fit(returns, method='mle')
            
            # AIC
            k = len(dist.params)
            log_lik = np.sum(dist.logpdf(returns))
            aic = 2 * k - 2 * log_lik
            
            # KS test
            ks = GOFTests.kolmogorov_smirnov(returns, dist)
            
            results[name] = {
                'dist': dist,
                'aic': aic,
                'ks_pvalue': ks.p_value
            }
            
            print(f"\n{name.upper()}:")
            print(f"  AIC: {aic:.2f}")
            print(f"  KS p-value: {ks.p_value:.4f}")
        except Exception as e:
            print(f"\n{name.upper()}: FAILED ({e})")
    
    # Best model
    best_name = min(results.items(), key=lambda x: x[1]['aic'])[0]
    best_dist = results[best_name]['dist']
    
    print(f"\n✅ Best model: {best_name.upper()} (lowest AIC)")
    
    # VaR calculation
    print("\n" + "="*60)
    print("STEP 2: Value-at-Risk (VaR) Estimation")
    print("="*60)
    
    var_levels = [0.95, 0.99, 0.995]
    
    print(f"\nVaR estimates ({best_name}):")
    for level in var_levels:
        var = best_dist.ppf(1 - level)
        print(f"  VaR({level*100:.1f}%): {var*100:.2f}%")
    
    # Bootstrap CI for VaR
    print("\n" + "="*60)
    print("STEP 3: Bootstrap CI for VaR(99%)")
    print("="*60)
    
    boot_var99 = []
    for i in range(1000):
        boot_sample = best_dist.rvs(size=len(returns), random_state=i)
        d = get_distribution(best_name)
        d.fit(boot_sample)
        boot_var99.append(d.ppf(0.01))
    
    boot_var99 = np.array(boot_var99)
    var99 = best_dist.ppf(0.01)
    var99_ci = (np.percentile(boot_var99, 2.5), np.percentile(boot_var99, 97.5))
    
    print(f"VaR(99%): {var99*100:.2f}%")
    print(f"95% CI: [{var99_ci[0]*100:.2f}%, {var99_ci[1]*100:.2f}%]")
    
    # Expected Shortfall (CVaR)
    print("\n" + "="*60)
    print("STEP 4: Expected Shortfall (CVaR)")
    print("="*60)
    
    # Simulate to estimate CVaR
    sim_returns = best_dist.rvs(size=100000, random_state=42)
    
    for level in var_levels:
        var = best_dist.ppf(1 - level)
        cvar = np.mean(sim_returns[sim_returns <= var])
        print(f"  CVaR({level*100:.1f}%): {cvar*100:.2f}%")
    
    # Visualization
    print("\n" + "="*60)
    print("STEP 5: Visualization")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # PDF with VaR
    ax = axes[0, 0]
    ax.hist(returns, bins=50, density=True, alpha=0.7, label='Returns')
    x = np.linspace(returns.min(), returns.max(), 200)
    ax.plot(x, best_dist.pdf(x), 'r-', linewidth=2, label=f'Fitted {best_name}')
    ax.axvline(var99, color='orange', linestyle='--', linewidth=2, label='VaR(99%)')
    ax.fill_betweenx([0, ax.get_ylim()[1]], returns.min(), var99, alpha=0.3, color='red')
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution Fit: {best_name}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Q-Q plot
    ax = axes[0, 1]
    qq_data = Diagnostics.qq_diagnostics(returns, best_dist)
    ax.scatter(qq_data['theoretical'], qq_data['sample'], alpha=0.5)
    lims = [qq_data['theoretical'].min(), qq_data['theoretical'].max()]
    ax.plot(lims, lims, 'r--')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title(f"Q-Q Plot (r={qq_data['correlation']:.4f})")
    ax.grid(alpha=0.3)
    
    # Bootstrap VaR distribution
    ax = axes[1, 0]
    ax.hist(boot_var99 * 100, bins=30, density=True, alpha=0.7)
    ax.axvline(var99 * 100, color='r', linewidth=2, label='Estimated VaR')
    ax.axvline(var99_ci[0] * 100, color='orange', linestyle='--', label='95% CI')
    ax.axvline(var99_ci[1] * 100, color='orange', linestyle='--')
    ax.set_xlabel('VaR(99%) (%)')
    ax.set_ylabel('Density')
    ax.set_title('Bootstrap Distribution of VaR(99%)')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Tail comparison
    ax = axes[1, 1]
    # Empirical tail
    sorted_returns = np.sort(returns)
    empirical_prob = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    ax.plot(sorted_returns * 100, 1 - empirical_prob, 'b.', alpha=0.5, label='Empirical')
    # Theoretical tail
    theoretical_prob = best_dist.cdf(sorted_returns)
    ax.plot(sorted_returns * 100, 1 - theoretical_prob, 'r-', linewidth=2, label='Theoretical')
    ax.set_yscale('log')
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Tail Probability')
    ax.set_title('Tail Comparison (Log Scale)')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('financial_risk_analysis.png', dpi=300, bbox_inches='tight')
    print("Figure saved as 'financial_risk_analysis.png'")
    
    # Final summary
    print("\n" + "="*60)
    print("RISK SUMMARY")
    print("="*60)
    print(f"• Best Model: {best_name} (AIC={results[best_name]['aic']:.2f})")
    print(f"• Daily VaR(99%): {var99*100:.2f}% (95% CI: {var99_ci[0]*100:.2f}% to {var99_ci[1]*100:.2f}%)")
    print(f"• With $1M portfolio: ${abs(var99)*1e6:,.0f} at risk")
    print(f"• Annual VaR(99%): ~{var99*np.sqrt(252)*100:.2f}% (assuming 252 trading days)")

**Output Interpretation:**

The analysis provides:

1. **Model selection** - Student's t usually wins for stock returns (heavy tails)
2. **VaR estimates** - Maximum expected loss at different confidence levels
3. **Bootstrap CI** - Uncertainty in VaR estimates
4. **CVaR** - Expected loss beyond VaR (more conservative)
5. **Visual diagnostics** - Check model fit quality

Next Steps
----------

- Modify for your own data
- Try different distributions
- Adjust confidence levels
- Add more diagnostics
