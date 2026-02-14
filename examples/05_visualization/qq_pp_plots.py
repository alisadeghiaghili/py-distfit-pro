#!/usr/bin/env python3
"""
Q-Q and P-P Plots
=================

Quantile-Quantile and Probability-Probability diagnostic plots.

Author: Ali Sadeghi Aghili
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("📊 Q-Q & P-P DIAGNOSTIC PLOTS")
print("="*70)

# Generate data: good fit vs bad fit
data_good = np.random.normal(100, 15, 800)
data_bad = np.random.lognormal(4, 0.5, 800)

# Fit distributions
dist_good = get_distribution('normal')
dist_good.fit(data_good)

dist_bad_normal = get_distribution('normal')  # Wrong fit
dist_bad_normal.fit(data_bad)

dist_bad_correct = get_distribution('lognormal')  # Correct fit
dist_bad_correct.fit(data_bad)

print("\n✅ Fitted distributions")

# Create figure
fig = plt.figure(figsize=(14, 10))

# ============================================================================
# Good Fit: Q-Q Plot
# ============================================================================
ax1 = plt.subplot(2, 3, 1)

percentiles = np.linspace(0.01, 0.99, len(data_good))
theoretical_q = dist_good.ppf(percentiles)
empirical_q = np.sort(data_good)

ax1.scatter(theoretical_q, empirical_q, alpha=0.5, s=15, color='blue')
min_val = min(theoretical_q.min(), empirical_q.min())
max_val = max(theoretical_q.max(), empirical_q.max())
ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
         label='Perfect Fit Line')

ax1.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold')
ax1.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold')
ax1.set_title('Q-Q Plot: GOOD FIT', fontsize=11, fontweight='bold', color='green')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_aspect('equal', 'box')

# ============================================================================
# Good Fit: P-P Plot
# ============================================================================
ax2 = plt.subplot(2, 3, 2)

theoretical_p = dist_good.cdf(np.sort(data_good))
empirical_p = np.arange(1, len(data_good)+1) / len(data_good)

ax2.scatter(theoretical_p, empirical_p, alpha=0.5, s=15, color='blue')
ax2.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Fit Line')

ax2.set_xlabel('Theoretical Cumulative Prob', fontsize=10, fontweight='bold')
ax2.set_ylabel('Empirical Cumulative Prob', fontsize=10, fontweight='bold')
ax2.set_title('P-P Plot: GOOD FIT', fontsize=11, fontweight='bold', color='green')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_aspect('equal', 'box')

# ============================================================================
# Good Fit: Residuals
# ============================================================================
ax3 = plt.subplot(2, 3, 3)

residuals = empirical_q - theoretical_q
ax3.scatter(theoretical_q, residuals, alpha=0.5, s=15, color='blue')
ax3.axhline(0, color='r', linestyle='--', linewidth=2)
ax3.fill_between([theoretical_q.min(), theoretical_q.max()], 
                  [-2*data_good.std()/np.sqrt(len(data_good))]*2,
                  [2*data_good.std()/np.sqrt(len(data_good))]*2,
                  alpha=0.2, color='green', label='±2 SE')

ax3.set_xlabel('Theoretical Quantiles', fontsize=10, fontweight='bold')
ax3.set_ylabel('Residuals', fontsize=10, fontweight='bold')
ax3.set_title('Residual Plot: GOOD FIT', fontsize=11, fontweight='bold', color='green')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

# ============================================================================
# Bad Fit: Q-Q Plot (Normal on Lognormal data)
# ============================================================================
ax4 = plt.subplot(2, 3, 4)

percentiles_bad = np.linspace(0.01, 0.99, len(data_bad))
theoretical_q_bad = dist_bad_normal.ppf(percentiles_bad)
empirical_q_bad = np.sort(data_bad)

ax4.scatter(theoretical_q_bad, empirical_q_bad, alpha=0.5, s=15, color='orange')
min_val = min(theoretical_q_bad.min(), empirical_q_bad.min())
max_val = max(theoretical_q_bad.max(), empirical_q_bad.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
         label='Perfect Fit Line')

ax4.set_xlabel('Theoretical Quantiles (Normal)', fontsize=10, fontweight='bold')
ax4.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold')
ax4.set_title('Q-Q Plot: BAD FIT (Wrong Model)', fontsize=11, 
              fontweight='bold', color='red')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# ============================================================================
# Correct Fit: Q-Q Plot (Lognormal on Lognormal data)
# ============================================================================
ax5 = plt.subplot(2, 3, 5)

theoretical_q_correct = dist_bad_correct.ppf(percentiles_bad)

ax5.scatter(theoretical_q_correct, empirical_q_bad, alpha=0.5, s=15, color='green')
min_val = min(theoretical_q_correct.min(), empirical_q_bad.min())
max_val = max(theoretical_q_correct.max(), empirical_q_bad.max())
ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2,
         label='Perfect Fit Line')

ax5.set_xlabel('Theoretical Quantiles (Lognormal)', fontsize=10, fontweight='bold')
ax5.set_ylabel('Sample Quantiles', fontsize=10, fontweight='bold')
ax5.set_title('Q-Q Plot: GOOD FIT (Correct Model)', fontsize=11,
              fontweight='bold', color='green')
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3)
ax5.set_aspect('equal', 'box')

# ============================================================================
# Explanation Panel
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

explanation = """
Q-Q & P-P PLOT GUIDE
{'='*40}

Q-Q PLOT (Quantile-Quantile):
{'-'*40}
  • Compares quantiles of two distributions
  • Points on red line = perfect fit
  • Deviations show where model fails
  
  PATTERNS:
  • S-curve: skewness difference
  • Points above line: data > theory
  • Points below line: data < theory
  • Curved tails: heavy/light tail issues

P-P PLOT (Probability-Probability):
{'-'*40}
  • Compares cumulative probabilities
  • More sensitive to center differences
  • Less sensitive to tail differences
  
  vs Q-Q:
  • Q-Q better for tail assessment
  • P-P better for overall fit

RESIDUAL PLOT:
{'-'*40}
  • Shows deviations from perfect fit
  • Should be random around zero
  • Patterns indicate model problems

✅ USE Q-Q PLOTS TO:
  1. Check distribution assumptions
  2. Identify outliers
  3. Assess tail behavior
  4. Compare model alternatives
"""

ax6.text(0.05, 0.95, explanation, transform=ax6.transAxes,
         fontsize=8, verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

plt.suptitle('Diagnostic Plots: Q-Q, P-P, and Residuals',
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

print("\n✅ Diagnostic plots created!")
print("\nInterpretation:")
print("  Top row:    Good fit (points follow line)")
print("  Bottom row: Bad fit (Normal) vs Good fit (Lognormal)")

plt.show()

print("\nNext: Check 06_real_world/ for applied examples!")
