"""
Continuous Distributions: Complete Guide
========================================

What you'll learn:
- When to use each continuous distribution
- Typical real-world applications
- How to fit and interpret parameters
- Visual comparison

Distributions covered:
- Normal (Gaussian)
- Lognormal  
- Exponential
- Gamma
- Weibull
- Beta
- Student's t
"""

import numpy as np
import matplotlib.pyplot as plt
from distfit_pro import get_distribution

# Set random seed
np.random.seed(42)

# ============================================================================
# 1. NORMAL DISTRIBUTION
# ============================================================================
print("📊 1. NORMAL (GAUSSIAN) DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Measurement errors
  • Heights, weights, IQ scores
  • Central Limit Theorem applies
  • Data is symmetric
  
Parameters:
  • loc (μ): mean/center
  • scale (σ): standard deviation
""")

# Generate and fit
heights = np.random.normal(170, 10, 500)  # Heights in cm
dist_normal = get_distribution('normal')
dist_normal.fit(heights)

print(f"Fitted parameters: μ={dist_normal.params['loc']:.2f} cm, "
      f"σ={dist_normal.params['scale']:.2f} cm")
print(f"Interpretation: Average height {dist_normal.mean():.1f} cm, "
      f"most people within {dist_normal.std():.1f} cm\n")

# ============================================================================
# 2. LOGNORMAL DISTRIBUTION  
# ============================================================================
print("📈 2. LOGNORMAL DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Incomes, salaries
  • Stock prices
  • File sizes
  • Particle sizes
  • Data is positive and right-skewed
  
Parameters:
  • s: shape (related to variance)
  • scale: scale (related to median)
""")

# Generate and fit
incomes = np.random.lognormal(10.5, 0.5, 500)  # Annual income in thousands
dist_lognorm = get_distribution('lognormal')
dist_lognorm.fit(incomes)

print(f"Mean income: ${dist_lognorm.mean():.2f}k")
print(f"Median income: ${dist_lognorm.median():.2f}k")
print(f"Note: Mean > Median due to right skew (few very high earners)\n")

# ============================================================================
# 3. EXPONENTIAL DISTRIBUTION
# ============================================================================
print("⏱️  3. EXPONENTIAL DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Time between events (arrivals, failures)
  • Radioactive decay
  • Service times
  • Memoryless processes
  
Parameters:
  • scale (1/λ): mean time between events
""")

# Generate and fit
wait_times = np.random.exponential(5, 500)  # Customer wait times in minutes
dist_exp = get_distribution('exponential')
dist_exp.fit(wait_times)

print(f"Average wait time: {dist_exp.mean():.2f} minutes")
print(f"P(wait < 3 min) = {dist_exp.cdf(3)*100:.1f}%")
print(f"Memoryless property: P(wait > 8 | waited 5) = P(wait > 3)\n")

# ============================================================================
# 4. GAMMA DISTRIBUTION
# ============================================================================
print("🔷 4. GAMMA DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Sum of exponential random variables
  • Time until k events occur
  • Insurance claims
  • Rainfall amounts
  • Positive skewed data
  
Parameters:
  • alpha (shape): controls shape
  • beta (scale): stretches distribution
""")

# Generate and fit
claim_amounts = np.random.gamma(2, 1000, 500)  # Insurance claims in $
dist_gamma = get_distribution('gamma')
dist_gamma.fit(claim_amounts)

print(f"Mean claim: ${dist_gamma.mean():.2f}")
print(f"Std dev: ${dist_gamma.std():.2f}")
print(f"95th percentile: ${dist_gamma.ppf(0.95):.2f} (used for reserves)\n")

# ============================================================================
# 5. WEIBULL DISTRIBUTION
# ============================================================================
print("🔧 5. WEIBULL DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Reliability analysis (time to failure)
  • Wind speed modeling
  • Survival analysis
  • Hazard rate changes over time
  
Parameters:
  • c (shape): 
    - c < 1: decreasing failure rate (early failures)
    - c = 1: constant (exponential)
    - c > 1: increasing (wear-out)
  • scale: characteristic life
""")

# Generate and fit
failure_times = np.random.weibull(1.5, 500) * 1000  # Component lifetimes (hours)
dist_weibull = get_distribution('weibull')
dist_weibull.fit(failure_times)

shape = dist_weibull.params['c']
print(f"Shape parameter: {shape:.3f}")
if shape > 1:
    print("  → Increasing failure rate (wear-out period)")
elif shape < 1:
    print("  → Decreasing failure rate (early failures)")
else:
    print("  → Constant failure rate (exponential)")

print(f"\nMean lifetime: {dist_weibull.mean():.1f} hours")
print(f"Median lifetime: {dist_weibull.median():.1f} hours\n")

# ============================================================================
# 6. BETA DISTRIBUTION
# ============================================================================
print("🎯 6. BETA DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Proportions, percentages
  • Probabilities (Bayesian analysis)
  • Project completion rates
  • Bounded data [0, 1]
  
Parameters:
  • alpha, beta: shape parameters
    - Both > 1: bell-shaped
    - Both < 1: U-shaped  
    - One large: skewed
""")

# Generate and fit
test_scores = np.random.beta(8, 2, 500)  # Exam scores (as fraction)
dist_beta = get_distribution('beta')
dist_beta.fit(test_scores)

print(f"Mean score: {dist_beta.mean()*100:.1f}%")
print(f"Most students score between "
      f"{dist_beta.ppf(0.25)*100:.0f}% and {dist_beta.ppf(0.75)*100:.0f}%\n")

# ============================================================================
# 7. STUDENT'S T DISTRIBUTION
# ============================================================================
print("📊 7. STUDENT'S T DISTRIBUTION")
print("=" * 70)
print("""
Use when:
  • Small samples from normal population
  • Unknown population variance
  • Robust to outliers (vs Normal)
  • Financial returns (heavy tails)
  
Parameters:
  • df: degrees of freedom
    - df → ∞: approaches Normal
    - small df: heavier tails
""")

# Generate and fit
returns = np.random.standard_t(5, 500) * 0.02 + 0.001  # Daily stock returns
dist_t = get_distribution('t')
dist_t.fit(returns)

df = dist_t.params['df']
print(f"Degrees of freedom: {df:.1f}")
if df < 10:
    print("  → Heavy tails: more outliers than Normal")
    print("  → Good for financial data with fat tails")

print(f"\nMean return: {dist_t.mean()*100:.3f}%")
print(f"Std dev: {dist_t.std()*100:.3f}%\n")

# ============================================================================
# VISUAL COMPARISON
# ============================================================================
print("📈 Visual Comparison")
print("=" * 70)

try:
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Continuous Distributions Overview', fontsize=16, fontweight='bold')
    
    distributions = [
        (heights, dist_normal, 'Normal\n(Heights)', 'skyblue'),
        (incomes, dist_lognorm, 'Lognormal\n(Incomes)', 'lightcoral'),
        (wait_times, dist_exp, 'Exponential\n(Wait Times)', 'lightgreen'),
        (claim_amounts, dist_gamma, 'Gamma\n(Insurance)', 'wheat'),
        (failure_times, dist_weibull, 'Weibull\n(Reliability)', 'plum'),
        (test_scores, dist_beta, 'Beta\n(Test Scores)', 'lightsalmon'),
        (returns*1000, dist_t, "Student's t\n(Returns)", 'lightsteelblue'),
    ]
    
    for idx, (data, dist, title, color) in enumerate(distributions):
        ax = axes[idx // 4, idx % 4]
        
        # Histogram
        ax.hist(data, bins=30, density=True, alpha=0.6, 
                color=color, edgecolor='black', linewidth=0.5)
        
        # Fitted PDF
        x = np.linspace(data.min(), data.max(), 200)
        ax.plot(x, dist.pdf(x), 'r-', lw=2)
        
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Density', fontsize=9)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
    
    # Hide empty subplot
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig('continuous_distributions.png', dpi=100, bbox_inches='tight')
    print("✅ Saved comparison plot to 'continuous_distributions.png'")
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
| Distribution | Best For | Key Property |
|-------------|----------|-------------|
| Normal | Symmetric data | Bell curve, CLT |
| Lognormal | Positive skewed | Multiplicative processes |
| Exponential | Time between events | Memoryless |
| Gamma | Sum of exponentials | Flexible shape |
| Weibull | Failure analysis | Varying hazard rate |
| Beta | Bounded [0,1] | Very flexible shape |
| Student's t | Heavy tails | Robust to outliers |

Next steps:
- See 02_discrete.py for discrete distributions
- See 04_model_selection/ for choosing best fit
- See docs/api/ for all parameters
"""
)
