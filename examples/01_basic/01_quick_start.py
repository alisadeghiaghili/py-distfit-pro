"""
Quick Start: 5-Minute Introduction to DistFit-Pro
==================================================

What you'll learn:
- Fit a distribution to data in 3 lines of code
- View parameter estimates and goodness-of-fit
- Generate samples from the fitted distribution

Real-world context:
You have measurement data (heights, times, weights) and want to
know which probability distribution best describes it.
"""

import numpy as np
from distfit_pro import get_distribution

# ============================================================================
# STEP 1: Generate some data
# ============================================================================
# In real life, this would be your actual measurements
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=500)

print("📊 Our data:")
print(f"   Sample size: {len(data)}")
print(f"   Mean: {np.mean(data):.2f}")
print(f"   Std Dev: {np.std(data):.2f}")
print()

# ============================================================================
# STEP 2: Fit a Normal distribution (3 lines!)
# ============================================================================
print("🔧 Fitting Normal distribution...")
print()

# Get distribution object
dist = get_distribution('normal')

# Fit to data (MLE by default)
dist.fit(data)

# View results
print(dist.summary())
print()

# ============================================================================
# STEP 3: Use the fitted distribution
# ============================================================================
print("📈 Using the fitted distribution:")
print()

# What's the probability of X < 85?
prob_below_85 = dist.cdf(85)
print(f"   P(X < 85) = {prob_below_85:.4f} ({prob_below_85*100:.1f}%)")

# What value has 90% of data below it?
percentile_90 = dist.ppf(0.90)
print(f"   90th percentile = {percentile_90:.2f}")

# Generate 5 new samples from fitted distribution
samples = dist.rvs(5, random_state=42)
print(f"   5 random samples: {samples}")
print()

# ============================================================================
# STEP 4: Model comparison (optional)
# ============================================================================
print("🔍 Is Normal the best fit?")
print()

# Try Lognormal too
dist_lognorm = get_distribution('lognormal')
dist_lognorm.fit(data)

print(f"   Normal AIC:    {dist.aic():.2f}")
print(f"   Lognormal AIC: {dist_lognorm.aic():.2f}")
print()
print("   ✅ Lower AIC is better → Normal wins!")
print()

# ============================================================================
# NEXT STEPS
# ============================================================================
print("""
🚀 Next steps:
   1. Try other distributions: 'exponential', 'weibull', 'gamma'
   2. Use method='mom' for faster (less accurate) fitting
   3. See 02_fitting_single_dist.py for visualization
   4. Check docs/api/ for all available methods
"""
)

# ============================================================================
# REAL-WORLD TIP
# ============================================================================
print("""
💡 Real-world tip:
   Always check multiple distributions! What looks normal might actually
   be lognormal or gamma. Use AIC/BIC to compare.
   
   See examples/04_model_selection/ for systematic comparison.
"""
)
