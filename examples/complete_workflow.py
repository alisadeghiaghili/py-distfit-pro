"""Complete workflow example for distfit-pro.

This script demonstrates a realistic distribution fitting workflow:
1. Load and explore data
2. Fit multiple distributions
3. Compare models
4. Validate assumptions
5. Use results for predictions

Author: Ali Sadeghi Aghili
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution, DistributionFitter
from distfit_pro.core.config import config

# Set preferences
config.set_language('en')  # or 'fa' for Persian
config.set_verbosity('verbose')  # detailed explanations

print("="*70)
print("Distribution Fitting - Complete Workflow")
print("="*70)

# ============================================================================
# STEP 1: Generate sample data (replace with your real data)
# ============================================================================
np.random.seed(42)

# Simulating product lifetimes (hours)
lifetimes = np.random.weibull(2.5, 500) * 1000

print(f"\nData summary:")
print(f"  n = {len(lifetimes)}")
print(f"  mean = {np.mean(lifetimes):.2f} hours")
print(f"  std = {np.std(lifetimes):.2f} hours")
print(f"  min = {np.min(lifetimes):.2f}, max = {np.max(lifetimes):.2f}")

# ============================================================================
# STEP 2: Fit single distribution (quick test)
# ============================================================================
print("\n" + "="*70)
print("Quick single distribution fit")
print("="*70)

# Try Weibull distribution
weibull = get_distribution('weibull')
weibull.fit(lifetimes, method='mle')

print(f"\nFitted Weibull parameters:")
for param, value in weibull.params.items():
    print(f"  {param} = {value:.4f}")

print(f"\nModel quality:")
print(f"  Log-likelihood: {weibull.log_likelihood():.2f}")
print(f"  AIC: {weibull.aic():.2f}")
print(f"  BIC: {weibull.bic():.2f}")

# ============================================================================
# STEP 3: Fit multiple distributions automatically
# ============================================================================
print("\n" + "="*70)
print("Automatic multi-distribution fitting")
print("="*70)

fitter = DistributionFitter(lifetimes)

# Let it suggest distributions based on data characteristics
suggestions = fitter.suggest_distributions()

# Fit all suggested distributions
results = fitter.fit(
    distributions=suggestions,
    method='mle',
    criterion='aic',  # or 'bic', 'loo_cv'
    n_jobs=1,  # use -1 for parallel
    verbose=True
)

# Show results
print(results.summary())

# ============================================================================
# STEP 4: Visualize fits
# ============================================================================
print("\n" + "="*70)
print("Creating diagnostic plots...")
print("="*70)

# Comparison plots (PDF, CDF, Q-Q, P-P)
fig1 = results.plot(kind='comparison', show_top_n=3, show=False)
plt.savefig('comparison_plots.png', dpi=150, bbox_inches='tight')
print("  Saved: comparison_plots.png")

# Diagnostic plots (residuals, influence)
fig2 = results.plot(kind='diagnostics', show=False)
plt.savefig('diagnostic_plots.png', dpi=150, bbox_inches='tight')
print("  Saved: diagnostic_plots.png")

# Interactive dashboard (requires plotly)
try:
    fig3 = results.plot(kind='interactive', show=False)
    fig3.write_html('interactive_dashboard.html')
    print("  Saved: interactive_dashboard.html")
except:
    print("  (Plotly not available, skipping interactive plot)")

# ============================================================================
# STEP 5: Use best model for predictions
# ============================================================================
print("\n" + "="*70)
print("Using best model for predictions")
print("="*70)

best = results.get_best()

# Calculate reliability (probability of survival)
time_points = [500, 1000, 1500, 2000]
print("\nReliability analysis (probability component survives):")
for t in time_points:
    reliability = best.sf(t)  # survival function
    print(f"  At {t} hours: {reliability*100:.2f}%")

# Calculate MTTF (Mean Time To Failure)
mttf = best.mean()
print(f"\nMean Time To Failure: {mttf:.2f} hours")

# Warranty analysis (what % will fail within warranty period?)
warranty_period = 800  # hours
failure_prob = best.cdf(warranty_period)
print(f"\nWarranty analysis:")
print(f"  Period: {warranty_period} hours")
print(f"  Expected failures: {failure_prob*100:.2f}%")
print(f"  Cost estimate: ${failure_prob * len(lifetimes) * 50:.2f}")
print(f"    (assuming $50 per warranty claim)")

# Generate predictions for Monte Carlo simulation
print("\nGenerating predictions for simulation...")
simulated_lifetimes = best.rvs(size=10000, random_state=42)
print(f"  Generated {len(simulated_lifetimes)} samples")
print(f"  Mean: {np.mean(simulated_lifetimes):.2f} hours")
print(f"  5th percentile: {np.percentile(simulated_lifetimes, 5):.2f} hours")
print(f"  95th percentile: {np.percentile(simulated_lifetimes, 95):.2f} hours")

# ============================================================================
# STEP 6: Bootstrap confidence intervals (optional)
# ============================================================================
print("\n" + "="*70)
print("Bootstrap confidence intervals")
print("="*70)

try:
    from distfit_pro.bootstrap import bootstrap_ci
    
    ci = bootstrap_ci(
        data=lifetimes,
        distribution=best,
        n_bootstrap=1000,
        confidence_level=0.95,
        random_state=42
    )
    
    print("\n95% Confidence intervals for parameters:")
    for param, (lower, upper) in ci.items():
        point = best.params[param]
        print(f"  {param}: [{lower:.4f}, {upper:.4f}]  (point: {point:.4f})")
        
except Exception as e:
    print(f"  Bootstrap skipped: {e}")

print("\n" + "="*70)
print("Analysis complete!")
print("="*70)
