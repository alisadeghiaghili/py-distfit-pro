"""
Fitting a Single Distribution: Complete Workflow
=================================================

What you'll learn:
- Proper data preparation and validation
- Fit with Maximum Likelihood Estimation (MLE)
- Interpret parameter estimates
- Check goodness of fit
- Visualize results

Real-world context:
You're analyzing customer purchase amounts and want to model
the distribution for pricing optimization.
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

# ============================================================================
# STEP 1: Load and prepare data
# ============================================================================
print("📁 Step 1: Data Preparation")
print("=" * 60)

# Simulate customer purchase amounts (in $)
np.random.seed(123)
purchase_amounts = np.random.gamma(shape=2, scale=50, size=1000)

print(f"Data loaded: {len(purchase_amounts)} purchases")
print(f"Range: ${purchase_amounts.min():.2f} - ${purchase_amounts.max():.2f}")
print(f"Mean: ${purchase_amounts.mean():.2f}")
print(f"Median: ${np.median(purchase_amounts):.2f}")
print()

# Check for issues
if np.any(np.isnan(purchase_amounts)):
    print("⚠️  Warning: Data contains NaN values")
if np.any(np.isinf(purchase_amounts)):
    print("⚠️  Warning: Data contains infinite values") 
if len(purchase_amounts) < 30:
    print("⚠️  Warning: Small sample (n<30) - estimates may be unstable")

print("✅ Data validation passed\n")

# ============================================================================
# STEP 2: Choose and fit distribution
# ============================================================================
print("🔧 Step 2: Fit Distribution")
print("=" * 60)

# Gamma is good for positive skewed data (purchase amounts, wait times)
dist = get_distribution('gamma')

print("Fitting Gamma distribution with MLE...")
dist.fit(purchase_amounts, method='mle')
print("✅ Fitting completed\n")

# ============================================================================
# STEP 3: View and interpret results
# ============================================================================
print("📊 Step 3: Parameter Estimates")
print("=" * 60)
print(dist.summary())
print()

print("📝 Interpretation:")
params = dist.params
print(f"   Shape (alpha): {params['alpha']:.3f}")
print(f"   → Controls distribution shape")
print(f"   → alpha > 1 means mode exists\n")

print(f"   Scale (beta): {params['beta']:.3f}")
print(f"   → Stretches the distribution")
print(f"   → Mean = alpha * beta = {params['alpha'] * params['beta']:.2f}")
print()

# ============================================================================
# STEP 4: Goodness of fit
# ============================================================================
print("✅ Step 4: Goodness of Fit")
print("=" * 60)

aic = dist.aic()
bic = dist.bic()
log_lik = dist.log_likelihood()

print(f"Log-Likelihood: {log_lik:.2f}")
print(f"AIC: {aic:.2f}  (lower is better)")
print(f"BIC: {bic:.2f}  (lower is better, penalizes complexity)")
print()

print("📝 Note: Compare AIC/BIC across distributions to choose best fit")
print("     See examples/04_model_selection/ for systematic comparison\n")

# ============================================================================
# STEP 5: Use the fitted distribution
# ============================================================================
print("💰 Step 5: Business Insights")
print("=" * 60)

# What % of customers spend less than $100?
prob_under_100 = dist.cdf(100)
print(f"P(purchase < $100) = {prob_under_100*100:.1f}%")

# What's the 75th percentile purchase amount?
percentile_75 = dist.ppf(0.75)
print(f"75th percentile: ${percentile_75:.2f}")

# What's the expected purchase amount?
mean_purchase = dist.mean()
print(f"Expected value: ${mean_purchase:.2f}")
print()

# ============================================================================
# STEP 6: Visualization (optional)
# ============================================================================
print("📈 Step 6: Visualization")
print("=" * 60)

try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot 1: Histogram + PDF
    axes[0].hist(purchase_amounts, bins=50, density=True, 
                 alpha=0.6, color='skyblue', edgecolor='black')
    
    x = np.linspace(0, purchase_amounts.max(), 200)
    axes[0].plot(x, dist.pdf(x), 'r-', lw=2, label='Fitted Gamma')
    axes[0].set_xlabel('Purchase Amount ($)')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Histogram + Fitted PDF')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Plot 2: Empirical CDF + Fitted CDF
    sorted_data = np.sort(purchase_amounts)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    
    axes[1].plot(sorted_data, empirical_cdf, 'o', 
                 alpha=0.3, markersize=3, label='Empirical')
    axes[1].plot(x, dist.cdf(x), 'r-', lw=2, label='Fitted Gamma')
    axes[1].set_xlabel('Purchase Amount ($)')
    axes[1].set_ylabel('Cumulative Probability')
    axes[1].set_title('CDF Comparison')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('gamma_fit_results.png', dpi=100, bbox_inches='tight')
    print("✅ Plots saved to 'gamma_fit_results.png'")
    print("   Close the plot window to continue...\n")
    plt.show()
    
except ImportError:
    print("⚠️  Matplotlib not installed - skipping visualization")
    print("   Install with: pip install matplotlib\n")

# ============================================================================
# SUMMARY
# ============================================================================
print("🎯 Summary")
print("=" * 60)
print("""
Complete workflow:
✅ 1. Prepared and validated data
✅ 2. Chose appropriate distribution (Gamma for positive skewed)
✅ 3. Fitted with MLE
✅ 4. Interpreted parameters
✅ 5. Checked goodness of fit
✅ 6. Extracted business insights
✅ 7. Visualized results

Next steps:
- Try other distributions (lognormal, weibull)
- Compare models (see 04_model_selection/)
- Add confidence intervals (see 03_advanced_fitting/)
"""
)
