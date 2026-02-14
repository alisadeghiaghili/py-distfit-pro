"""
Discrete Distributions: Complete Guide
======================================

What you'll learn:
- When to use discrete distributions
- Common applications (counts, events)
- Parameter interpretation
- Comparison with continuous analogs

Distributions covered:
- Poisson
- Binomial
- Geometric
- Negative Binomial
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

np.random.seed(42)

# ============================================================================
# 1. POISSON DISTRIBUTION
# ============================================================================
print("📦 1. POISSON DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Counting events in fixed time/space
  • Rare events
  • Website visits per hour
  • Defects per product
  • Goals per game
  
Key property:
  • Mean = Variance = λ (rate parameter)
  • Models "how many" events
  
Parameters:
  • mu (λ): average event rate
""")

# Example: Website visits per minute
visits_per_minute = np.random.poisson(12, 500)

dist_poisson = get_distribution('poisson')
dist_poisson.fit(visits_per_minute)

rate = dist_poisson.params['mu']
print(f"Average visits per minute: {rate:.2f}")
print(f"Variance: {dist_poisson.var():.2f}")
print(f"\nP(exactly 10 visits) = {dist_poisson.pmf(10)*100:.2f}%")
print(f"P(at least 15 visits) = {dist_poisson.sf(14)*100:.2f}%")

# Poisson approximation to Binomial
print("\n💡 When to use Poisson vs Binomial:")
print("   Use Poisson when: n is large, p is small, np is moderate")
print("   Example: n=1000 trials, p=0.01 → Poisson(λ=10)\n")

# ============================================================================
# 2. BINOMIAL DISTRIBUTION
# ============================================================================
print("🎲 2. BINOMIAL DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Fixed number of trials (n)
  • Each trial is success/failure
  • Constant probability (p)
  • Independent trials
  
Examples:
  • Coin flips
  • Quality control (k defects in n items)
  • A/B test conversions
  
Parameters:
  • n: number of trials
  • p: success probability
""")

# Example: Quality control (10% defect rate, inspect 50 items)
n_trials = 50
p_defect = 0.10

# Generate data: number of defects found in 500 inspection batches
defects_found = np.random.binomial(n_trials, p_defect, 500)

dist_binom = get_distribution('binomial')
dist_binom.fit(defects_found)

print(f"Trials per batch: {n_trials}")
print(f"Expected defects per batch: {dist_binom.mean():.2f}")
print(f"Std dev: {dist_binom.std():.2f}")

print(f"\nP(0 defects) = {dist_binom.pmf(0)*100:.2f}% (all pass)")
print(f"P(> 8 defects) = {dist_binom.sf(8)*100:.2f}% (batch fails)")

# Normal approximation
if n_trials * p_defect > 5 and n_trials * (1-p_defect) > 5:
    print("\n✅ Normal approximation valid (np > 5 and n(1-p) > 5)")
    print(f"   Can approximate with Normal({n_trials*p_defect}, "
          f"{np.sqrt(n_trials*p_defect*(1-p_defect)):.2f})")
print()

# ============================================================================
# 3. GEOMETRIC DISTRIBUTION  
# ============================================================================
print("🎯 3. GEOMETRIC DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Number of trials until first success
  • Memoryless discrete analog of Exponential
  • Customer conversions
  • Machine cycles until failure
  
Key property:
  • Memoryless: P(X > n+k | X > n) = P(X > k)
  
Parameters:
  • p: success probability per trial
""")

# Example: Sales calls until first sale (20% close rate)
p_sale = 0.20

# Generate data: calls needed to close (for 500 salespeople)
calls_until_sale = np.random.geometric(p_sale, 500)

dist_geom = get_distribution('geometric')
dist_geom.fit(calls_until_sale)

print(f"Close rate: {p_sale*100:.0f}%")
print(f"Average calls until sale: {dist_geom.mean():.2f}")
print(f"Median calls until sale: {dist_geom.median():.0f}")

print(f"\nP(sale on 1st call) = {dist_geom.pmf(1)*100:.1f}%")
print(f"P(need > 5 calls) = {dist_geom.sf(5)*100:.1f}%")

print("\n💡 Memoryless property:")
print("   If you've made 3 calls with no sale,")
print("   probability of 2 more calls needed is same as initially\n")

# ============================================================================
# 4. NEGATIVE BINOMIAL DISTRIBUTION
# ============================================================================  
print("🔄 4. NEGATIVE BINOMIAL DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Number of trials until r successes
  • Overdispersed count data (variance > mean)
  • Alternative to Poisson for clustered data
  • Modeling "contagion" or "success runs"
  
Key difference from Poisson:
  • Poisson: Var = Mean
  • Negative Binomial: Var > Mean (more flexible)
  
Parameters:
  • n (r): number of successes to wait for
  • p: success probability
""")

# Example: Customer complaints (clustered, not Poisson)
# Generate overdispersed data
complaints_per_day = np.random.negative_binomial(5, 0.3, 500)

dist_nbinom = get_distribution('negative_binomial')
dist_nbinom.fit(complaints_per_day)

mean_complaints = dist_nbinom.mean()
var_complaints = dist_nbinom.var()

print(f"Mean complaints per day: {mean_complaints:.2f}")
print(f"Variance: {var_complaints:.2f}")
print(f"Variance/Mean ratio: {var_complaints/mean_complaints:.2f}")

if var_complaints > mean_complaints * 1.5:
    print("  → Strong overdispersion: complaints are clustered")
    print("  → Negative Binomial better than Poisson")

print(f"\nP(0 complaints) = {dist_nbinom.pmf(0)*100:.1f}%")
print(f"P(> 20 complaints) = {dist_nbinom.sf(20)*100:.1f}% (bad day!)\n")

# ============================================================================
# COMPARISON: Poisson vs Negative Binomial
# ============================================================================
print("⚖️  Poisson vs Negative Binomial")
print("=" * 70)

# Fit both to same data
dist_poisson_comp = get_distribution('poisson')
dist_poisson_comp.fit(complaints_per_day)

print("Fitting overdispersed data:")
print(f"  Negative Binomial AIC: {dist_nbinom.aic():.2f}")
print(f"  Poisson AIC: {dist_poisson_comp.aic():.2f}")

if dist_nbinom.aic() < dist_poisson_comp.aic():
    print("  \n✅ Negative Binomial wins (lower AIC)")
    print("     Data shows overdispersion/clustering\n")
else:
    print("  \n✅ Poisson wins (lower AIC)")
    print("     Data follows Poisson assumptions\n")

# ============================================================================
# VISUAL COMPARISON
# ============================================================================
print("📈 Visual Comparison")
print("=" * 70)

try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Discrete Distributions Overview', fontsize=16, fontweight='bold')
    
    # 1. Poisson
    ax = axes[0, 0]
    x_pois = np.arange(0, max(visits_per_minute)+1)
    ax.bar(x_pois, [dist_poisson.pmf(k) for k in x_pois], 
           alpha=0.6, color='skyblue', edgecolor='black')
    ax.set_title('Poisson\n(Website Visits)', fontweight='bold')
    ax.set_xlabel('Number of Visits')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3, axis='y')
    
    # 2. Binomial
    ax = axes[0, 1]
    x_binom = np.arange(0, n_trials+1)
    ax.bar(x_binom, [dist_binom.pmf(k) for k in x_binom],
           alpha=0.6, color='lightcoral', edgecolor='black')
    ax.set_title('Binomial\n(Quality Control)', fontweight='bold')
    ax.set_xlabel('Number of Defects')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3, axis='y')
    
    # 3. Geometric
    ax = axes[1, 0]
    x_geom = np.arange(1, 20)
    ax.bar(x_geom, [dist_geom.pmf(k) for k in x_geom],
           alpha=0.6, color='lightgreen', edgecolor='black')
    ax.set_title('Geometric\n(Calls Until Sale)', fontweight='bold')
    ax.set_xlabel('Number of Trials')
    ax.set_ylabel('Probability')
    ax.grid(alpha=0.3, axis='y')
    
    # 4. Negative Binomial vs Poisson
    ax = axes[1, 1]
    x_nb = np.arange(0, 40)
    ax.bar(x_nb-0.2, [dist_nbinom.pmf(k) for k in x_nb], 
           width=0.4, alpha=0.6, color='wheat', 
           edgecolor='black', label='Neg Binomial')
    ax.bar(x_nb+0.2, [dist_poisson_comp.pmf(k) for k in x_nb],
           width=0.4, alpha=0.6, color='plum',
           edgecolor='black', label='Poisson')
    ax.set_title('Negative Binomial vs Poisson\n(Overdispersed Data)', 
                 fontweight='bold')
    ax.set_xlabel('Count')
    ax.set_ylabel('Probability')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('discrete_distributions.png', dpi=100, bbox_inches='tight')
    print("✅ Saved comparison plot to 'discrete_distributions.png'")
    print("   Close window to continue...\n")
    plt.show()
    
except ImportError:
    print("⚠️  Matplotlib not installed - skipping visualization\n")

# ============================================================================
# QUICK REFERENCE
# ============================================================================
print("📝 Quick Reference")
print("=" * 70)
print("""
| Distribution | Models | Mean | Variance |
|-------------|--------|------|----------|
| Poisson | Events in interval | λ | λ |
| Binomial | Successes in n trials | np | np(1-p) |
| Geometric | Trials until success | 1/p | (1-p)/p² |
| Neg Binomial | Trials until r successes | r(1-p)/p | r(1-p)/p² |

Choosing:
• Fixed n, count successes → Binomial
• Count events in time/space → Poisson  
• Wait for first event → Geometric
• Overdispersed counts → Negative Binomial

Next:
- See 03_specialized.py for reliability distributions
- See 04_model_selection/ for choosing best fit
"""
)
