#!/usr/bin/env python3
"""
Custom Parameters Example
========================

Manually set distribution parameters without fitting.
Useful for:
  - Using parameters from literature/specifications
  - Simulation with known distributions
  - Testing hypothetical scenarios

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

print("="*70)
print("🔧 MANUAL PARAMETER SETTING")
print("="*70)


# ============================================================================
# Method 1: Set Parameters Directly (No Fitting)
# ============================================================================

print("\n" + "="*70)
print("Method 1: Direct Parameter Assignment")
print("="*70)

# Create distribution object
dist_normal = get_distribution('normal')

# Set parameters manually (instead of fitting)
dist_normal.params = {'loc': 100, 'scale': 15}

print("✅ Parameters set manually (no fitting needed):")
for param, val in dist_normal.params.items():
    print(f"  {param}: {val}")

# Now you can use all distribution methods
print(f"\nMean: {dist_normal.mean():.2f}")
print(f"Std:  {dist_normal.std():.2f}")
print(f"P(X < 110) = {dist_normal.cdf(110):.4f}")


# ============================================================================
# Method 2: Use Known Specifications
# ============================================================================

print("\n" + "="*70)
print("Method 2: From Product Specifications")
print("="*70)
print("""
Scenario: Manufacturing tolerance specification
  - Target dimension: 50.0 mm
  - Tolerance: ±0.5 mm (99.7% within spec)
  - Assumes normal distribution
""")

# For normal: 99.7% within ±3σ (3-sigma rule)
# So if tolerance is ±0.5, then σ ≈ 0.5/3 = 0.167

target = 50.0
tolerance = 0.5
sigma = tolerance / 3  # 3-sigma rule

dist_spec = get_distribution('normal')
dist_spec.params = {'loc': target, 'scale': sigma}

print(f"✅ Distribution created from specifications:")
print(f"  Target (μ):    {target} mm")
print(f"  Std Dev (σ):  {sigma:.4f} mm")
print(f"\n  99.7% of parts between: [{target-3*sigma:.2f}, {target+3*sigma:.2f}] mm")

# Calculate defect rate (outside tolerance)
defect_rate = 1 - (dist_spec.cdf(target+tolerance) - dist_spec.cdf(target-tolerance))
print(f"  Expected defect rate: {defect_rate*100:.2f}%")


# ============================================================================
# Method 3: Literature/Research Parameters
# ============================================================================

print("\n" + "="*70)
print("Method 3: From Published Research")
print("="*70)
print("""
Scenario: Using parameters from a research paper
  - Paper reports: "Height follows N(170, 10²) cm"
  - You want to simulate or calculate probabilities
""")

dist_literature = get_distribution('normal')
dist_literature.params = {'loc': 170, 'scale': 10}  # mean=170cm, std=10cm

print("✅ Distribution from literature:")
print(f"  Mean height: {dist_literature.mean():.1f} cm")
print(f"  Std:         {dist_literature.std():.1f} cm")

# Calculate interesting probabilities
prob_tall = 1 - dist_literature.cdf(180)  # P(Height > 180cm)
print(f"\n  P(Height > 180 cm) = {prob_tall:.4f} or {prob_tall*100:.2f}%")

# Generate synthetic data for simulation
synthetic_data = dist_literature.rvs(size=1000, random_state=42)
print(f"\n  Generated {len(synthetic_data)} synthetic samples")
print(f"  Sample mean: {synthetic_data.mean():.2f} cm")


# ============================================================================
# Method 4: Scenario Testing (What-If Analysis)
# ============================================================================

print("\n" + "="*70)
print("Method 4: What-If Scenario Testing")
print("="*70)
print("""
Scenario: Compare different quality improvement strategies
  - Current: μ=100, σ=15 (baseline)
  - Strategy A: Reduce mean to 95 (centering)
  - Strategy B: Reduce std to 10 (tighter control)
  - Strategy C: Both improvements
""")

scenarios = [
    ('Current Baseline', {'loc': 100, 'scale': 15}),
    ('Strategy A (Center)', {'loc': 95, 'scale': 15}),
    ('Strategy B (Tighten)', {'loc': 100, 'scale': 10}),
    ('Strategy C (Both)', {'loc': 95, 'scale': 10}),
]

# Specification: upper limit = 110
upper_limit = 110

print(f"\nGoal: Minimize defects (values > {upper_limit})\n")

for name, params in scenarios:
    dist = get_distribution('normal')
    dist.params = params
    
    defect_rate = 1 - dist.cdf(upper_limit)
    
    print(f"{name:20s}  μ={params['loc']:5.1f}  σ={params['scale']:5.1f}  "
          f"Defects={defect_rate*100:5.2f}%")

print("\n→ Strategy C (both improvements) gives lowest defect rate!")


# ============================================================================
# Visualization: Compare Scenarios
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Scenario Comparison Plot...")
print("="*70)

fig, ax = plt.subplots(figsize=(12, 6))

x_range = np.linspace(50, 150, 500)

colors = ['blue', 'green', 'orange', 'red']
for (name, params), color in zip(scenarios, colors):
    dist = get_distribution('normal')
    dist.params = params
    
    y = dist.pdf(x_range)
    ax.plot(x_range, y, linewidth=2, label=name, color=color)

# Add specification limit
ax.axvline(upper_limit, color='red', linestyle='--', linewidth=2, 
           label=f'Specification Limit ({upper_limit})')

ax.set_xlabel('Value', fontsize=12)
ax.set_ylabel('Probability Density', fontsize=12)
ax.set_title('Scenario Comparison: Quality Improvement Strategies', 
             fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()

print("\n✅ Visualization created!")
print("   Close plot window to continue...")

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. You don't always need to fit data - sometimes you KNOW the parameters

2. Use manual parameters for:
   • Specifications from engineering drawings
   • Published research results
   • Hypothetical scenario testing
   • Simulation with known distributions

3. Simply set: dist.params = {'param1': value1, 'param2': value2}

4. All methods (pdf, cdf, rvs, etc.) work the same way

5. Great for "what-if" analysis without collecting new data!

Next: See 03_advanced_fitting/ for weighted and censored data fitting!
""")
