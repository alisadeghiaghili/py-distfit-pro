#!/usr/bin/env python3
"""
Distribution Family Comparison
===============================

Compares distributions within families:
1. Location-scale family (Normal, Logistic, Cauchy)
2. Exponential family (Exponential, Gamma, Weibull)
3. Heavy-tailed family (t, Cauchy, Pareto)

Shows how to identify distribution families from data characteristics.
"""

import numpy as np
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("DISTRIBUTION FAMILY COMPARISON")
print("="*70)

# =============================================================================
# FAMILY 1: Location-Scale Distributions
# =============================================================================
print("\n" + "="*70)
print("FAMILY 1: LOCATION-SCALE DISTRIBUTIONS")
print("="*70)
print("""
These distributions have:
- loc parameter (center/location)
- scale parameter (spread)
- Different tail behaviors
""")

# Generate data from normal
data_normal = np.random.normal(10, 2, 1000)

print("\nFitting to Normal data:")
print("-"*70)

for name in ['normal', 'logistic', 'cauchy']:
    try:
        dist = get_distribution(name)
        dist.fit(data_normal)
        
        print(f"\n{name.upper()}:")
        print(f"  loc: {dist.params.get('loc', 'N/A'):.4f}")
        print(f"  scale: {dist.params.get('scale', 'N/A'):.4f}")
        print(f"  AIC: {dist.aic():.2f}")
        
        try:
            kurt = dist.kurtosis()
            print(f"  Kurtosis: {kurt:.4f}", end="")
            if abs(kurt) < 0.5:
                print(" (normal tails)")
            elif kurt > 0:
                print(" (heavy tails)")
            else:
                print(" (light tails)")
        except:
            print("  Kurtosis: undefined (very heavy tails)")
            
    except Exception as e:
        print(f"\n{name.upper()}: Failed - {e}")

print("\n" + "="*70)
print("INTERPRETATION:")
print("- Normal has lowest AIC (data is normal)")
print("- Logistic has slightly heavier tails (kurtosis > 0)")
print("- Cauchy has undefined moments (extremely heavy tails)")
print("="*70)

# =============================================================================
# FAMILY 2: Exponential Family (Positive Data)
# =============================================================================
print("\n" + "="*70)
print("FAMILY 2: EXPONENTIAL FAMILY (Positive Data)")
print("="*70)
print("""
These distributions:
- Support: x > 0
- Model waiting times, lifetimes
- Exponential is special case of others
""")

# Generate data from Weibull with shape > 1 (increasing hazard)
from scipy.stats import weibull_min
data_weibull = weibull_min.rvs(c=2.5, scale=5, size=1000)

print("\nFitting to Weibull data (shape=2.5, scale=5):")
print("-"*70)

for name in ['exponential', 'gamma', 'weibull_min']:
    try:
        dist = get_distribution(name)
        dist.fit(data_weibull)
        
        print(f"\n{name.upper()}:")
        for param, value in dist.params.items():
            print(f"  {param}: {value:.4f}")
        print(f"  Mean: {dist.mean():.4f}")
        print(f"  AIC: {dist.aic():.2f}")
        
    except Exception as e:
        print(f"\n{name.upper()}: Failed - {e}")

print("\n" + "="*70)
print("RELATIONSHIPS:")
print("- Exponential: constant hazard rate (memoryless)")
print("- Gamma: sum of exponentials, more flexible")
print("- Weibull: varying hazard rate (shape parameter)")
print("  * shape < 1: decreasing hazard (infant mortality)")
print("  * shape = 1: constant hazard (= exponential)")
print("  * shape > 1: increasing hazard (wear-out)")
print("="*70)

# =============================================================================
# FAMILY 3: Heavy-Tailed Distributions
# =============================================================================
print("\n" + "="*70)
print("FAMILY 3: HEAVY-TAILED DISTRIBUTIONS")
print("="*70)
print("""
These distributions:
- Have outliers more often
- Useful for financial data, extreme events
- Moments may not exist
""")

# Generate data from t-distribution (df=3, heavy tails)
from scipy.stats import t as t_dist
data_heavy = t_dist.rvs(df=3, loc=0, scale=1, size=1000)

print("\nFitting to t-distribution data (df=3):")
print("-"*70)

for name in ['normal', 't', 'cauchy']:
    try:
        dist = get_distribution(name)
        dist.fit(data_heavy)
        
        print(f"\n{name.upper()}:")
        for param, value in dist.params.items():
            print(f"  {param}: {value:.4f}")
        print(f"  AIC: {dist.aic():.2f}")
        
        # Check if moments exist
        try:
            mean = dist.mean()
            var = dist.var()
            print(f"  Mean: {mean:.4f}")
            print(f"  Variance: {var:.4f}")
        except:
            print("  Mean/Variance: undefined")
            
    except Exception as e:
        print(f"\n{name.upper()}: Failed - {e}")

print("\n" + "="*70)
print("TAIL BEHAVIOR:")
print("- Normal: exponentially decreasing tails")
print("- t(df): polynomial tails, converges to normal as df→∞")
print("- Cauchy: very heavy tails, no mean/variance")
print("- Pareto: power-law tails, extreme outliers")
print("="*70)

# =============================================================================
# Summary
# =============================================================================
print("\n" + "="*70)
print("DISTRIBUTION SELECTION FLOWCHART")
print("="*70)
print("""
1. Check data range:
   - Negative values allowed? → Normal, t, Logistic, Cauchy
   - Only positive? → Exponential family, Lognormal
   - Bounded [0,1]? → Beta, Uniform

2. Check symmetry:
   - Symmetric? → Normal, t, Logistic, Uniform
   - Right-skewed? → Gamma, Weibull, Lognormal, Exponential
   - Left-skewed? → Beta (with α > β)

3. Check tail behavior:
   - Normal tails? → Normal, Logistic
   - Heavy tails? → t, Cauchy, Pareto
   - Light tails? → Beta, Uniform

4. Check for outliers:
   - Many outliers? → Heavy-tailed (t, Cauchy)
   - Few outliers? → Normal, Logistic
   - No outliers? → Bounded distributions

5. Domain knowledge:
   - Failure times? → Weibull, Exponential
   - Financial returns? → t-distribution
   - Proportions? → Beta
   - Extreme values? → Gumbel, Frechet
""")
print("="*70)
print("\nTIP: Always compare multiple candidates using AIC/BIC!")
print("="*70)
