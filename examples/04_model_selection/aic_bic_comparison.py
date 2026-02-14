#!/usr/bin/env python3
"""
AIC/BIC Model Comparison
=======================

Use information criteria to select the best distribution.

AIC (Akaike Information Criterion):
  - Balances fit quality vs model complexity
  - AIC = 2k - 2*log(L)
  - Lower is better

BIC (Bayesian Information Criterion):
  - Penalizes complexity more than AIC
  - BIC = k*log(n) - 2*log(L)
  - Lower is better
  - Preferred for large samples

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("🎯 MODEL SELECTION: AIC & BIC")
print("="*70)


# ============================================================================
# Understanding AIC and BIC
# ============================================================================

print("\n" + "="*70)
print("📚 Information Criteria Explained")
print("="*70)
print("""
Both AIC and BIC help choose between competing models:

  🎯 GOAL: Find simplest model that explains data well
  
  AIC (Akaike Information Criterion):
    • Formula: 2k - 2*log(L)
    • k = number of parameters
    • L = likelihood
    • Penalizes complexity moderately
    • Good for prediction
  
  BIC (Bayesian Information Criterion):
    • Formula: k*log(n) - 2*log(L)
    • n = sample size
    • Penalizes complexity MORE as n grows
    • Favors simpler models for large samples
    • Good for explanation/inference
  
  RULE: Lower values = better model
  
  TIP: Difference > 10 is "strong evidence" for better model
""")


# ============================================================================
# Example 1: Right-Skewed Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Right-Skewed Data (Income Distribution)")
print("="*70)

# Generate right-skewed data (lognormal)
n = 1000
true_dist = 'lognormal'
data = np.random.lognormal(mean=10.5, sigma=0.6, size=n)

print(f"\n📊 Data: {n} samples (right-skewed)")
print(f"  Mean:    ${data.mean()/1000:.1f}k")
print(f"  Median:  ${np.median(data)/1000:.1f}k")
print(f"  Skewness: {((data - data.mean())**3).mean() / data.std()**3:.2f}")

# Candidate distributions for right-skewed data
candidates = ['normal', 'lognormal', 'gamma', 'weibull_min', 'expon']

results = []

print("\n🔍 Testing candidate distributions...\n")
print("  Distribution    Parameters    AIC        BIC        Log-Lik")
print("  " + "-"*68)

for name in candidates:
    try:
        dist = get_distribution(name)
        dist.fit(data)
        
        aic = dist.aic()
        bic = dist.bic()
        ll = dist.log_likelihood()
        n_params = len(dist.params)
        
        results.append({
            'name': name,
            'params': n_params,
            'aic': aic,
            'bic': bic,
            'loglik': ll,
            'dist': dist
        })
        
        print(f"  {name:15s}  {n_params} params     {aic:>9.2f}  {bic:>9.2f}  {ll:>10.2f}")
        
    except Exception as e:
        print(f"  {name:15s}  FAILED: {str(e)[:40]}")

# Rank by AIC
print("\n" + "="*70)
print("🥇 RANKING BY AIC")
print("="*70)

results_aic = sorted(results, key=lambda x: x['aic'])

print("\n  Rank  Distribution      AIC        ΔAIC     Evidence")
print("  " + "-"*60)

best_aic = results_aic[0]['aic']
for rank, r in enumerate(results_aic, 1):
    delta = r['aic'] - best_aic
    
    if delta == 0:
        evidence = "BEST"
    elif delta < 2:
        evidence = "Weak"
    elif delta < 10:
        evidence = "Moderate"
    else:
        evidence = "Strong against"
    
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"  {marker} {rank}.  {r['name']:15s}  {r['aic']:>9.2f}  {delta:>7.2f}  {evidence}")

print(f"\n✅ Best model by AIC: {results_aic[0]['name']}")
print(f"   (True distribution was: {true_dist})")

# Rank by BIC
print("\n" + "="*70)
print("🥈 RANKING BY BIC")
print("="*70)

results_bic = sorted(results, key=lambda x: x['bic'])

print("\n  Rank  Distribution      BIC        ΔBIC     Evidence")
print("  " + "-"*60)

best_bic = results_bic[0]['bic']
for rank, r in enumerate(results_bic, 1):
    delta = r['bic'] - best_bic
    
    if delta == 0:
        evidence = "BEST"
    elif delta < 2:
        evidence = "Weak"
    elif delta < 10:
        evidence = "Moderate"
    else:
        evidence = "Strong against"
    
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"  {marker} {rank}.  {r['name']:15s}  {r['bic']:>9.2f}  {delta:>7.2f}  {evidence}")

print(f"\n✅ Best model by BIC: {results_bic[0]['name']}")

if results_aic[0]['name'] != results_bic[0]['name']:
    print(f"\n⚠️  AIC and BIC chose different models!")
    print(f"   AIC prefers: {results_aic[0]['name']}")
    print(f"   BIC prefers: {results_bic[0]['name']}")
    print(f"   → BIC favors simpler models (penalizes parameters more)")


# ============================================================================
# Example 2: Symmetric Data
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Symmetric Data (Normal vs Heavy-Tailed)")
print("="*70)

# Generate near-normal data with slight heavy tails
data_sym = np.random.standard_t(df=10, size=500) * 15 + 100

print(f"\n📊 Data: {len(data_sym)} samples (symmetric, slightly heavy-tailed)")
print(f"  Mean: {data_sym.mean():.2f}")
print(f"  Std:  {data_sym.std():.2f}")

# Compare normal vs t-distribution
candidates_sym = ['normal', 't', 'logistic']

results_sym = []

print("\n🔍 Testing symmetric distributions...\n")
print("  Distribution    AIC        BIC        Params")
print("  " + "-"*50)

for name in candidates_sym:
    try:
        dist = get_distribution(name)
        dist.fit(data_sym)
        
        results_sym.append({
            'name': name,
            'aic': dist.aic(),
            'bic': dist.bic(),
            'params': len(dist.params),
            'dist': dist
        })
        
        print(f"  {name:15s}  {dist.aic():>9.2f}  {dist.bic():>9.2f}  {len(dist.params)}")
        
    except:
        pass

best = min(results_sym, key=lambda x: x['aic'])

print(f"\n✅ Best fit: {best['name']}")
print(f"   → Accounts for heavy tails better than Normal")


# ============================================================================
# Visualization
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Comparison Visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Data histogram with top 3 fits
ax = axes[0, 0]
ax.hist(data, bins=50, density=True, alpha=0.3, color='gray',
        edgecolor='black', label='Data')

x = np.linspace(data.min(), data.max(), 200)
colors = ['red', 'blue', 'green']
for i, r in enumerate(results_aic[:3]):
    y = r['dist'].pdf(x)
    ax.plot(x, y, color=colors[i], linewidth=2, 
            label=f"{r['name']} (rank {i+1})")

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Top 3 Models by AIC', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: AIC comparison bar chart
ax = axes[0, 1]
names = [r['name'] for r in results_aic]
aics = [r['aic'] for r in results_aic]
colors_bar = ['green' if i == 0 else 'orange' if i == 1 else 'red' if i == 2 else 'gray' 
              for i in range(len(names))]
ax.barh(names, aics, color=colors_bar, edgecolor='black')
ax.set_xlabel('AIC (lower is better)', fontsize=11)
ax.set_title('AIC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Plot 3: BIC comparison bar chart  
ax = axes[1, 0]
bics = [r['bic'] for r in results_bic]
names_bic = [r['name'] for r in results_bic]
colors_bic = ['green' if i == 0 else 'orange' if i == 1 else 'red' if i == 2 else 'gray' 
              for i in range(len(names_bic))]
ax.barh(names_bic, bics, color=colors_bic, edgecolor='black')
ax.set_xlabel('BIC (lower is better)', fontsize=11)
ax.set_title('BIC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Plot 4: AIC vs BIC scatter
ax = axes[1, 1]
for r in results:
    ax.scatter(r['aic'], r['bic'], s=100, alpha=0.7)
    ax.annotate(r['name'], (r['aic'], r['bic']), 
                fontsize=8, ha='right', va='bottom')

ax.set_xlabel('AIC', fontsize=11)
ax.set_ylabel('BIC', fontsize=11)
ax.set_title('AIC vs BIC Trade-off', fontweight='bold')
ax.grid(True, alpha=0.3)

# Mark best
best_aic_val = results_aic[0]['aic']
best_bic_val = results_bic[0]['bic']
for r in results:
    if r['aic'] == best_aic_val:
        ax.scatter(r['aic'], r['bic'], s=300, facecolors='none', 
                  edgecolors='red', linewidths=3, label='Best AIC')
    if r['bic'] == best_bic_val:
        ax.scatter(r['aic'], r['bic'], s=200, facecolors='none', 
                  edgecolors='blue', linewidths=2, label='Best BIC')

ax.legend(fontsize=9)

plt.tight_layout()

print("\n✅ Visualizations created!")
plt.show()


print("\n" + "="*70)
print("🎓 Guidelines for Using AIC/BIC")
print("="*70)
print("""
📊 WHEN TO USE WHICH:

  AIC (Akaike Information Criterion):
    ✅ Focus on prediction accuracy
    ✅ Willing to accept more complex models
    ✅ Machine learning applications
    ✅ Small to medium samples
  
  BIC (Bayesian Information Criterion):
    ✅ Focus on finding "true" model
    ✅ Prefer simpler, more interpretable models
    ✅ Scientific research / inference
    ✅ Large samples (BIC penalty grows with n)

🎯 INTERPRETATION:

  ΔAIC or ΔBIC:   Evidence Against Worse Model:
  ────────────────────────────────────────────
    0-2          Weak (models essentially equivalent)
    2-6          Moderate
    6-10         Strong
    >10          Very strong (decisive)

⚠️  IMPORTANT:
  • Information criteria are relative (compare models on same data)
  • Lower is always better
  • Consider both AIC and BIC
  • Large ΔAIC/BIC (≫10) = clear winner
  • Small difference (<2) = models roughly equivalent

Next: See goodness_of_fit.py for statistical tests!
""")
