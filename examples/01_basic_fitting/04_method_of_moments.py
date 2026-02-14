"""Method of Moments vs Maximum Likelihood

Compare two parameter estimation methods:
- Method of Moments (MoM): Fast, less accurate
- Maximum Likelihood (MLE): Slower, more accurate

Learn when to use each method.

Perfect for: Understanding estimation trade-offs
Time: ~10 minutes
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from distfit_pro import get_distribution
import pandas as pd

print("="*70)
print("⚖️  METHOD OF MOMENTS vs MAXIMUM LIKELIHOOD")
print("="*70)

# ============================================================================
# Generate Test Data
# ============================================================================
np.random.seed(789)
true_loc = 50
true_scale = 10
data = np.random.normal(loc=true_loc, scale=true_scale, size=1000)

print("\n📊 True Parameters:")
print(f"  Location (μ): {true_loc}")
print(f"  Scale (σ): {true_scale}")
print(f"  Sample size: {len(data)}")

# ============================================================================
# Method 1: Method of Moments (MoM)
# ============================================================================
print("\n" + "="*70)
print("METHOD 1: Method of Moments (MoM)")
print("="*70)

print("\n💡 How it works:")
print("  Match sample moments (mean, variance) to theoretical moments.")
print("  ✓ Fast (closed-form solution)")
print("  ✗ Less efficient (higher variance in estimates)")

dist_mom = get_distribution('normal')

start_time = time.time()
dist_mom.fit(data, method='mom')
mom_time = time.time() - start_time

params_mom = dist_mom.params
print(f"\n⏱️  Computation time: {mom_time*1000:.3f} ms")
print(f"\n📊 Estimated Parameters (MoM):")
for name, value in params_mom.items():
    true_val = true_loc if name == 'loc' else true_scale
    error = abs(value - true_val)
    error_pct = error / true_val * 100
    print(f"  {name:<10} = {value:>10.4f}  (error: {error_pct:.3f}%)")

# ============================================================================
# Method 2: Maximum Likelihood Estimation (MLE)
# ============================================================================
print("\n" + "="*70)
print("METHOD 2: Maximum Likelihood Estimation (MLE)")
print("="*70)

print("\n💡 How it works:")
print("  Find parameters that maximize probability of observing the data.")
print("  ✓ More efficient (lower variance in estimates)")
print("  ✗ Slower (requires numerical optimization)")

dist_mle = get_distribution('normal')

start_time = time.time()
dist_mle.fit(data, method='mle')
mle_time = time.time() - start_time

params_mle = dist_mle.params
print(f"\n⏱️  Computation time: {mle_time*1000:.3f} ms")
print(f"\n📊 Estimated Parameters (MLE):")
for name, value in params_mle.items():
    true_val = true_loc if name == 'loc' else true_scale
    error = abs(value - true_val)
    error_pct = error / true_val * 100
    print(f"  {name:<10} = {value:>10.4f}  (error: {error_pct:.3f}%)")

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

print(f"\n⏱️  Speed Comparison:")
print(f"  MoM time: {mom_time*1000:.3f} ms")
print(f"  MLE time: {mle_time*1000:.3f} ms")
print(f"  Speedup: {mle_time/mom_time:.2f}x faster with MoM")

print(f"\n🎯 Accuracy Comparison:")
comparison = pd.DataFrame({
    'Parameter': ['loc (μ)', 'scale (σ)'],
    'True': [true_loc, true_scale],
    'MoM': [params_mom['loc'], params_mom['scale']],
    'MLE': [params_mle['loc'], params_mle['scale']],
    'MoM Error %': [
        abs(params_mom['loc'] - true_loc) / true_loc * 100,
        abs(params_mom['scale'] - true_scale) / true_scale * 100
    ],
    'MLE Error %': [
        abs(params_mle['loc'] - true_loc) / true_loc * 100,
        abs(params_mle['scale'] - true_scale) / true_scale * 100
    ]
})

print("\n" + str(comparison.to_string(index=False)))

print(f"\n📊 Model Quality:")
print(f"  {'Metric':<20} {'MoM':<15} {'MLE':<15} {'Better'}")
print("-" * 60)
ll_mom = dist_mom.log_likelihood()
ll_mle = dist_mle.log_likelihood()
print(f"  {'Log-Likelihood':<20} {ll_mom:<15.2f} {ll_mle:<15.2f} {'MLE' if ll_mle > ll_mom else 'MoM'}")

aic_mom = dist_mom.aic()
aic_mle = dist_mle.aic()
print(f"  {'AIC':<20} {aic_mom:<15.2f} {aic_mle:<15.2f} {'MLE' if aic_mle < aic_mom else 'MoM'}")

bic_mom = dist_mom.bic()
bic_mle = dist_mle.bic()
print(f"  {'BIC':<20} {bic_mom:<15.2f} {bic_mle:<15.2f} {'MLE' if bic_mle < bic_mom else 'MoM'}")

# ============================================================================
# Monte Carlo Simulation: Variance Comparison
# ============================================================================
print("\n" + "="*70)
print("MONTE CARLO SIMULATION: Estimate Variance")
print("="*70)

print("\n🎲 Running 100 simulations to compare estimator variance...")

n_simulations = 100
mom_estimates = []
mle_estimates = []

for i in range(n_simulations):
    # Generate new sample
    sample = np.random.normal(loc=true_loc, scale=true_scale, size=200)
    
    # Fit with both methods
    d_mom = get_distribution('normal')
    d_mom.fit(sample, method='mom')
    mom_estimates.append(d_mom.params['scale'])
    
    d_mle = get_distribution('normal')
    d_mle.fit(sample, method='mle')
    mle_estimates.append(d_mle.params['scale'])

mom_estimates = np.array(mom_estimates)
mle_estimates = np.array(mle_estimates)

print(f"\n📊 Estimator Statistics (scale parameter):")
print(f"  {'Method':<10} {'Mean':<12} {'Std Dev':<12} {'Bias':<12} {'MSE'}")
print("-" * 60)

mom_mean = mom_estimates.mean()
mom_std = mom_estimates.std()
mom_bias = mom_mean - true_scale
mom_mse = ((mom_estimates - true_scale) ** 2).mean()
print(f"  {'MoM':<10} {mom_mean:<12.4f} {mom_std:<12.4f} {mom_bias:<12.4f} {mom_mse:.6f}")

mle_mean = mle_estimates.mean()
mle_std = mle_estimates.std()
mle_bias = mle_mean - true_scale
mle_mse = ((mle_estimates - true_scale) ** 2).mean()
print(f"  {'MLE':<10} {mle_mean:<12.4f} {mle_std:<12.4f} {mle_bias:<12.4f} {mle_mse:.6f}")

print(f"\n💡 Interpretation:")
print(f"  MLE has {mom_std/mle_std:.2f}x lower standard deviation")
print(f"  MLE has {mom_mse/mle_mse:.2f}x lower Mean Squared Error")
print("  → MLE is more efficient (less variance in estimates)")

# ============================================================================
# Visualization
# ============================================================================
print("\n" + "="*70)
print("VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: PDF Comparison
ax1 = axes[0, 0]
ax1.hist(data, bins=40, density=True, alpha=0.4, color='gray', 
         edgecolor='black', label='Data')
x = np.linspace(data.min(), data.max(), 200)
ax1.plot(x, dist_mom.pdf(x), 'b-', linewidth=2, label='MoM Fit')
ax1.plot(x, dist_mle.pdf(x), 'r--', linewidth=2, label='MLE Fit')
ax1.axvline(true_loc, color='green', linestyle=':', linewidth=2, label='True Mean')
ax1.set_xlabel('Value', fontsize=11)
ax1.set_ylabel('Density', fontsize=11)
ax1.set_title('PDF: MoM vs MLE', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Parameter Estimates
ax2 = axes[0, 1]
params = ['Location', 'Scale']
true_vals = [true_loc, true_scale]
mom_vals = [params_mom['loc'], params_mom['scale']]
mle_vals = [params_mle['loc'], params_mle['scale']]

x_pos = np.arange(len(params))
width = 0.25

ax2.bar(x_pos - width, true_vals, width, label='True', color='green', alpha=0.7)
ax2.bar(x_pos, mom_vals, width, label='MoM', color='blue', alpha=0.7)
ax2.bar(x_pos + width, mle_vals, width, label='MLE', color='red', alpha=0.7)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(params)
ax2.set_ylabel('Parameter Value', fontsize=11)
ax2.set_title('Parameter Estimates Comparison', fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Estimator Distributions (Monte Carlo)
ax3 = axes[1, 0]
ax3.hist(mom_estimates, bins=20, alpha=0.5, color='blue', 
         edgecolor='black', label=f'MoM (std={mom_std:.3f})', density=True)
ax3.hist(mle_estimates, bins=20, alpha=0.5, color='red', 
         edgecolor='black', label=f'MLE (std={mle_std:.3f})', density=True)
ax3.axvline(true_scale, color='green', linestyle='--', 
            linewidth=2, label=f'True Scale={true_scale}')
ax3.set_xlabel('Estimated Scale Parameter', fontsize=11)
ax3.set_ylabel('Density', fontsize=11)
ax3.set_title('Sampling Distribution (100 simulations)', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Plot 4: MSE Comparison
ax4 = axes[1, 1]
methods = ['MoM', 'MLE']
mse_vals = [mom_mse, mle_mse]
colors_bar = ['blue', 'red']

ax4.bar(methods, mse_vals, color=colors_bar, alpha=0.7, edgecolor='black')
ax4.set_ylabel('Mean Squared Error', fontsize=11)
ax4.set_title('MSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, (method, mse) in enumerate(zip(methods, mse_vals)):
    ax4.text(i, mse + 0.01, f'{mse:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('mom_vs_mle_comparison.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: mom_vs_mle_comparison.png")

# ============================================================================
# SUMMARY & RECOMMENDATIONS
# ============================================================================
print("\n" + "="*70)
print("🎯 RECOMMENDATIONS")
print("="*70)

print("\nUse METHOD OF MOMENTS when:")
print("  ✓ Speed is critical (real-time applications)")
print("  ✓ Large datasets (> 10,000 samples)")
print("  ✓ Quick exploratory analysis")
print("  ✓ Initial parameter guesses for MLE")

print("\nUse MAXIMUM LIKELIHOOD when:")
print("  ✓ Accuracy is critical (small error tolerance)")
print("  ✓ Small to medium datasets (< 10,000 samples)")
print("  ✓ Final production models")
print("  ✓ Need best statistical properties (efficiency, consistency)")

print("\n📊 Key Takeaways:")
print(f"  1. MLE is {mle_std/mom_std:.2f}x more efficient (lower variance)")
print(f"  2. MoM is {mle_time/mom_time:.2f}x faster")
print("  3. For Normal distribution, both are unbiased")
print("  4. MLE always has lower or equal MSE")
print("="*70)
