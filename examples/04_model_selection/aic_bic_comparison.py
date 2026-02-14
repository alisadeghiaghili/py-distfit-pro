#!/usr/bin/env python3
"""
Model Selection using AIC and BIC
=================================

Compare distributions using information criteria:
  - AIC (Akaike Information Criterion): Balances fit quality and complexity
  - BIC (Bayesian Information Criterion): Stronger penalty for complexity

Lower is better for both!

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("📊 MODEL SELECTION: AIC & BIC Comparison")
print("="*70)


# ============================================================================
# Theory: AIC vs BIC
# ============================================================================

print("\n" + "="*70)
print("📚 Theory: Information Criteria")
print("="*70)

theory = """
1. AIC (Akaike Information Criterion):
   Formula: AIC = 2k - 2ln(L)
   Where:
     k = number of parameters
     L = maximum likelihood
   
   • Balances goodness-of-fit with model complexity
   • Tends to select more complex models
   • Good for prediction
   • Lower AIC = Better model

2. BIC (Bayesian Information Criterion):
   Formula: BIC = k·ln(n) - 2ln(L)
   Where:
     k = number of parameters
     n = sample size
     L = maximum likelihood
   
   • Stronger penalty for complexity (especially large n)
   • Tends to select simpler models
   • Good for explanation/interpretation
   • Lower BIC = Better model

3. Key Differences:
   • BIC penalty grows with sample size: k·ln(n)
   • AIC penalty is constant: 2k
   • BIC → simpler models (as n grows)
   • AIC → better prediction
   • Both prefer simple models with good fit

4. When to Use:
   • AIC: Prediction is main goal
   • BIC: Want simpler, interpretable model
   • Both: Compare and see if they agree!
"""

print(theory)


# ============================================================================
# Example 1: Simple Case (Normal vs t-distribution)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Normal vs Student's t")
print("="*70)
print("""
Question: Does data have heavy tails?
  - Normal: 2 parameters (mean, std)
  - Student's t: 3 parameters (df, mean, std)
""")

# Generate data from t-distribution (heavy tails)
data_t = np.random.standard_t(df=5, size=500)

print(f"\n📊 Data: {len(data_t)} observations from t(df=5)")
print(f"  Mean: {data_t.mean():.3f}")
print(f"  Std:  {data_t.std():.3f}")
print(f"  Kurtosis: {((data_t - data_t.mean())**4).mean() / data_t.std()**4:.3f}")
print(f"  (Kurtosis > 3 suggests heavy tails)")

# Fit both models
models = []

# Normal
dist_normal = get_distribution('normal')
dist_normal.fit(data_t)
models.append(('Normal', dist_normal, 2))

# Student's t
dist_t = get_distribution('t')
dist_t.fit(data_t)
models.append(("Student's t", dist_t, 3))

print("\n" + "="*70)
print("Model Comparison")
print("="*70)

print(f"\n{'Model':<15} {'Params':<8} {'Log-Lik':<12} {'AIC':<10} {'BIC':<10}")
print("-"*70)

for name, dist, n_params in models:
    ll = dist.log_likelihood()
    aic = dist.aic()
    bic = dist.bic()
    
    print(f"{name:<15} {n_params:<8} {ll:<12.2f} {aic:<10.2f} {bic:<10.2f}")

# Determine winner
aic_winner = min(models, key=lambda x: x[1].aic())
bic_winner = min(models, key=lambda x: x[1].bic())

print("\n🏆 Results:")
print(f"  AIC prefers: {aic_winner[0]} (AIC = {aic_winner[1].aic():.2f})")
print(f"  BIC prefers: {bic_winner[0]} (BIC = {bic_winner[1].bic():.2f})")

if aic_winner[0] == bic_winner[0]:
    print(f"  ✅ Both criteria agree: {aic_winner[0]} is better!")
else:
    print(f"  ⚠️  Criteria disagree (BIC prefers simpler model)")


# ============================================================================
# Example 2: Multiple Candidates (Right-Skewed Data)
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Multiple Distribution Candidates")
print("="*70)
print("""
Scenario: Positive, right-skewed data (e.g., income, prices)
Candidates: Lognormal, Gamma, Weibull, Exponential
""")

# Generate lognormal data
data_skewed = np.random.lognormal(mean=3.0, sigma=0.6, size=800)

print(f"\n📊 Data: {len(data_skewed)} observations")
print(f"  Mean:     {data_skewed.mean():.2f}")
print(f"  Median:   {np.median(data_skewed):.2f}")
print(f"  Skewness: {((data_skewed - data_skewed.mean())**3).mean() / data_skewed.std()**3:.2f}")
print(f"  (Right-skewed: mean > median, skewness > 0)")

# Test multiple distributions
candidates = [
    ('Lognormal', 'lognormal', 3),
    ('Gamma', 'gamma', 3),
    ('Weibull', 'weibull_min', 3),
    ('Exponential', 'expon', 2),
]

results = []

print("\n" + "="*70)
print("Fitting Candidates...")
print("="*70)

for name, dist_name, n_params in candidates:
    try:
        dist = get_distribution(dist_name)
        dist.fit(data_skewed)
        
        results.append({
            'name': name,
            'dist': dist,
            'n_params': n_params,
            'log_lik': dist.log_likelihood(),
            'aic': dist.aic(),
            'bic': dist.bic(),
        })
        
        print(f"  ✓ {name:<15} fitted")
    except Exception as e:
        print(f"  ✗ {name:<15} failed: {e}")

print("\n" + "="*70)
print("Ranking by Information Criteria")
print("="*70)

# Sort by AIC
results_aic = sorted(results, key=lambda x: x['aic'])

print(f"\n🏆 AIC Ranking (lower = better):")
print(f"\n{'Rank':<6} {'Model':<15} {'Params':<8} {'AIC':<12} {'ΔAIC':<10}")
print("-"*70)

aic_best = results_aic[0]['aic']
for rank, r in enumerate(results_aic, 1):
    delta_aic = r['aic'] - aic_best
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:<4} {r['name']:<15} {r['n_params']:<8} {r['aic']:<12.2f} {delta_aic:<10.2f}")

# Sort by BIC
results_bic = sorted(results, key=lambda x: x['bic'])

print(f"\n🏆 BIC Ranking (lower = better):")
print(f"\n{'Rank':<6} {'Model':<15} {'Params':<8} {'BIC':<12} {'ΔBIC':<10}")
print("-"*70)

bic_best = results_bic[0]['bic']
for rank, r in enumerate(results_bic, 1):
    delta_bic = r['bic'] - bic_best
    marker = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else "  "
    print(f"{marker} {rank:<4} {r['name']:<15} {r['n_params']:<8} {r['bic']:<12.2f} {delta_bic:<10.2f}")

print("\n📌 Interpretation of ΔAIC and ΔBIC:")
print("  • Δ < 2:   Substantial support (models are similar)")
print("  • Δ 2-10:  Considerably less support")
print("  • Δ > 10:  Essentially no support")

# Agreement check
if results_aic[0]['name'] == results_bic[0]['name']:
    print(f"\n✅ Strong consensus: {results_aic[0]['name']} is the best model!")
else:
    print(f"\n⚠️  AIC prefers {results_aic[0]['name']}, BIC prefers {results_bic[0]['name']}")
    print(f"   (BIC penalizes complexity more, so may prefer simpler model)")


# ============================================================================
# Visualization: Model Comparison
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Visualizations...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model Selection using AIC and BIC', fontsize=16, fontweight='bold')

# Plot 1: Example 1 - Normal vs t
ax = axes[0, 0]
ax.hist(data_t, bins=40, density=True, alpha=0.5, color='skyblue', 
        edgecolor='black', label='Data')

x_range = np.linspace(data_t.min(), data_t.max(), 200)
for name, dist, _ in models:
    ax.plot(x_range, dist.pdf(x_range), linewidth=2, label=name)

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Example 1: Normal vs Student\'s t', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Plot 2: AIC comparison bars (Example 2)
ax = axes[0, 1]
names = [r['name'] for r in results_aic]
aic_values = [r['aic'] for r in results_aic]
colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' 
          for i in range(len(names))]

ax.barh(names, aic_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('AIC (lower = better)', fontsize=11)
ax.set_title('AIC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Plot 3: BIC comparison bars
ax = axes[1, 0]
names = [r['name'] for r in results_bic]
bic_values = [r['bic'] for r in results_bic]
colors = ['green' if i == 0 else 'orange' if i == 1 else 'red' 
          for i in range(len(names))]

ax.barh(names, bic_values, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('BIC (lower = better)', fontsize=11)
ax.set_title('BIC Comparison', fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()

# Plot 4: PDF overlay of top candidates
ax = axes[1, 1]
ax.hist(data_skewed, bins=50, density=True, alpha=0.3, color='gray', 
        edgecolor='black', label='Data')

x_skewed = np.linspace(data_skewed.min(), data_skewed.max(), 200)
for r in results_aic[:3]:  # Top 3
    ax.plot(x_skewed, r['dist'].pdf(x_skewed), linewidth=2, label=r['name'])

ax.set_xlabel('Value', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Top 3 Candidates Overlay', fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Plots created!")
print("   Close plot window to continue...")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. INFORMATION CRITERIA:
   • AIC and BIC balance fit quality vs model complexity
   • Lower values = better models
   • Use to compare non-nested models

2. AIC vs BIC:
   • AIC: Better for prediction, prefers complex models
   • BIC: Better for interpretation, prefers simple models
   • BIC penalty increases with sample size

3. INTERPRETATION:
   • ΔAIC/ΔBIC < 2:  Models are similar
   • ΔAIC/ΔBIC 2-10: Some evidence for best model
   • ΔAIC/ΔBIC > 10: Strong evidence for best model

4. BEST PRACTICES:
   ✓ Try multiple reasonable candidates
   ✓ Use both AIC and BIC
   ✓ If they agree → strong evidence
   ✓ If they disagree → consider both models
   ✓ Visualize top candidates

5. IN distfit-pro:
   dist.fit(data)
   aic = dist.aic()  # Lower is better
   bic = dist.bic()  # Lower is better

6. REMEMBER:
   • Information criteria ≠ goodness-of-fit tests
   • They rank models, don't test absolute fit
   • Always check residuals and QQ-plots!

Next: See goodness_of_fit.py for statistical tests!
""")
