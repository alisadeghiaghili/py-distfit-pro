#!/usr/bin/env python3
"""
Finance Applications: Distribution Fitting for Risk Analysis
============================================================

Real-world finance examples:
  - Stock returns distribution
  - Value at Risk (VaR) calculation
  - Portfolio risk assessment
  - Fat-tail analysis

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

print("="*70)
print("💰 FINANCE: DISTRIBUTION FITTING FOR RISK ANALYSIS")
print("="*70)


# ============================================================================
# Example 1: Stock Returns Distribution
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 1: Stock Returns Analysis")
print("="*70)

print("""
Scenario: Analyze daily stock returns
  - Are returns normally distributed?
  - Presence of fat tails?
  - Appropriate distribution for risk modeling
""")

# Simulate realistic stock returns (slightly fat-tailed)
# Use t-distribution to capture fat tails
returns_clean = np.random.standard_t(df=5, size=950) * 0.02
returns_crashes = np.random.standard_t(df=3, size=50) * 0.06  # Extreme events
returns = np.concatenate([returns_clean, returns_crashes])
np.random.shuffle(returns)

print(f"\n📊 Stock Returns Data: {len(returns)} daily observations")
print(f"  Mean return:    {returns.mean()*100:6.3f}%")
print(f"  Std deviation:  {returns.std()*100:6.3f}%")
print(f"  Skewness:       {stats.skew(returns):6.3f}")
print(f"  Kurtosis:       {stats.kurtosis(returns):6.3f} (excess)")
print(f"  Min return:     {returns.min()*100:6.2f}%")
print(f"  Max return:     {returns.max()*100:6.2f}%")

# Test multiple distributions
print("\n🔬 Testing distributions...")

candidates = [
    ('Normal', 'normal'),
    ('t-distribution', 't'),
    ('Logistic', 'logistic'),
]

fitted_dists = []
for name, dist_name in candidates:
    d = get_distribution(dist_name)
    d.fit(returns)
    ks_stat, ks_pval = stats.kstest(returns, d.cdf)
    fitted_dists.append((name, d, ks_stat, ks_pval))
    print(f"  {name:<20} AIC={d.aic():8.2f}  KS p-value={ks_pval:.4f}")

# Select best
best_name, best_dist, _, _ = min(fitted_dists, key=lambda x: x[1].aic())
print(f"\n🏆 Best model: {best_name} (AIC={best_dist.aic():.2f})")

# Calculate Value at Risk (VaR)
confidence_levels = [0.95, 0.99]

print("\n💸 Value at Risk (VaR):")
print(f"  (Maximum expected loss at given confidence)\n")

for conf in confidence_levels:
    var_empirical = np.percentile(returns, (1-conf)*100)
    var_parametric = best_dist.ppf(1-conf)
    
    print(f"  {int(conf*100)}% VaR:")
    print(f"    Empirical:  {var_empirical*100:6.2f}% loss")
    print(f"    Parametric: {var_parametric*100:6.2f}% loss")
    print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Stock Returns Distribution Analysis', fontsize=16, fontweight='bold')

# Plot 1: Histogram with fitted distributions
ax = axes[0, 0]
ax.hist(returns*100, bins=50, density=True, alpha=0.6, color='skyblue', 
        edgecolor='black', label='Returns Data')

x = np.linspace(returns.min(), returns.max(), 300)
for name, d, _, _ in fitted_dists:
    ax.plot(x*100, d.pdf(x)/100, linewidth=2, label=name)

ax.set_xlabel('Daily Return (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Returns Distribution', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.axvline(0, color='black', linestyle=':', linewidth=1)

# Plot 2: Q-Q plot (best model)
ax = axes[0, 1]
percentiles = np.linspace(0.01, 0.99, len(returns))
theoretical_q = best_dist.ppf(percentiles)
empirical_q = np.sort(returns)

ax.scatter(theoretical_q*100, empirical_q*100, alpha=0.5, s=10, color='green')
min_v, max_v = min(theoretical_q.min(), empirical_q.min())*100, \
               max(theoretical_q.max(), empirical_q.max())*100
ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2)
ax.set_xlabel(f'Theoretical Quantiles (%) - {best_name}', fontsize=10)
ax.set_ylabel('Sample Quantiles (%)', fontsize=10)
ax.set_title('Q-Q Plot: Goodness of Fit', fontweight='bold')
ax.grid(True, alpha=0.3)

# Plot 3: Tail analysis
ax = axes[1, 0]
tail_cutoff = np.percentile(returns, 5)
tail_data = returns[returns <= tail_cutoff]

ax.hist(tail_data*100, bins=20, density=True, alpha=0.6, color='red', 
        edgecolor='black', label='Left Tail Data')

x_tail = np.linspace(tail_data.min(), tail_cutoff, 100)
ax.plot(x_tail*100, best_dist.pdf(x_tail)/100, 'b-', linewidth=2, 
        label=f'{best_name} Fit')

ax.set_xlabel('Daily Return (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Left Tail Analysis (Worst 5%)', fontweight='bold', color='red')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: VaR visualization
ax = axes[1, 1]
ax.hist(returns*100, bins=50, density=True, alpha=0.4, color='gray', 
        edgecolor='black')
ax.plot(x*100, best_dist.pdf(x)/100, 'b-', linewidth=2, label=best_name)

# Mark VaR levels
for conf in confidence_levels:
    var_val = best_dist.ppf(1-conf)
    ax.axvline(var_val*100, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(var_val*100, ax.get_ylim()[1]*0.9, f'{int(conf*100)}% VaR\n{var_val*100:.2f}%',
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

ax.set_xlabel('Daily Return (%)', fontsize=11, fontweight='bold')
ax.set_ylabel('Density', fontsize=11, fontweight='bold')
ax.set_title('Value at Risk (VaR) Levels', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
print("\n📊 Visualization created!")
plt.savefig('/tmp/finance_returns_analysis.png', dpi=150, bbox_inches='tight')


# ============================================================================
# Example 2: Portfolio Risk Assessment
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE 2: Portfolio Risk Assessment")
print("="*70)

print("""
Scenario: Two-asset portfolio optimization
  - Asset A: Conservative (low risk)
  - Asset B: Aggressive (high risk)
  - Find optimal mix for target return
""")

# Simulate two assets
returns_A = np.random.normal(0.08/252, 0.15/np.sqrt(252), 1000)  # 8% annual, 15% vol
returns_B = np.random.normal(0.15/252, 0.30/np.sqrt(252), 1000)  # 15% annual, 30% vol

print(f"\n📊 Asset Statistics (daily):")
print(f"\n  Asset A (Conservative):")
print(f"    Mean return: {returns_A.mean()*252*100:5.2f}% (annualized)")
print(f"    Volatility:  {returns_A.std()*np.sqrt(252)*100:5.2f}% (annualized)")
print(f"\n  Asset B (Aggressive):")
print(f"    Mean return: {returns_B.mean()*252*100:5.2f}% (annualized)")
print(f"    Volatility:  {returns_B.std()*np.sqrt(252)*100:5.2f}% (annualized)")

# Portfolio combinations
weights_A = np.linspace(0, 1, 21)
portfolio_returns = []
portfolio_risks = []

for w_A in weights_A:
    w_B = 1 - w_A
    portfolio_ret = returns_A * w_A + returns_B * w_B
    portfolio_returns.append(portfolio_ret.mean() * 252 * 100)
    portfolio_risks.append(portfolio_ret.std() * np.sqrt(252) * 100)

# Find optimal (max Sharpe ratio, assuming risk-free rate = 0)
sharpe_ratios = np.array(portfolio_returns) / np.array(portfolio_risks)
optimal_idx = np.argmax(sharpe_ratios)
optimal_w_A = weights_A[optimal_idx]
optimal_w_B = 1 - optimal_w_A

print(f"\n🎯 Optimal Portfolio:")
print(f"    Weight A: {optimal_w_A*100:5.1f}%")
print(f"    Weight B: {optimal_w_B*100:5.1f}%")
print(f"    Expected return: {portfolio_returns[optimal_idx]:5.2f}%")
print(f"    Risk (volatility): {portfolio_risks[optimal_idx]:5.2f}%")
print(f"    Sharpe ratio: {sharpe_ratios[optimal_idx]:5.3f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(portfolio_risks, portfolio_returns, 'b-', linewidth=2.5, 
        label='Efficient Frontier')
ax.scatter(portfolio_risks[optimal_idx], portfolio_returns[optimal_idx], 
           s=200, c='red', marker='*', edgecolors='black', linewidth=2,
           label='Optimal Portfolio', zorder=5)

# Mark pure assets
ax.scatter(returns_A.std()*np.sqrt(252)*100, returns_A.mean()*252*100,
           s=100, c='green', marker='o', edgecolors='black', linewidth=1.5,
           label='Asset A', zorder=4)
ax.scatter(returns_B.std()*np.sqrt(252)*100, returns_B.mean()*252*100,
           s=100, c='orange', marker='o', edgecolors='black', linewidth=1.5,
           label='Asset B', zorder=4)

# Annotations
for i in [0, 5, 10, 15, 20]:
    w_A = weights_A[i]
    ax.annotate(f'{int(w_A*100)}-{int((1-w_A)*100)}', 
                xy=(portfolio_risks[i], portfolio_returns[i]),
                xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.7)

ax.set_xlabel('Risk (Volatility) %', fontsize=12, fontweight='bold')
ax.set_ylabel('Expected Return %', fontsize=12, fontweight='bold')
ax.set_title('Portfolio Efficient Frontier', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
print("\n📊 Efficient frontier plotted!")
plt.savefig('/tmp/finance_portfolio_frontier.png', dpi=150, bbox_inches='tight')

plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways - Finance Applications")
print("="*70)
print("""
1. STOCK RETURNS:
   • Often NOT normally distributed
   • Fat tails (extreme events more likely)
   • t-distribution often better fit
   • Important for risk management

2. VALUE AT RISK (VaR):
   • VaR(95%) = maximum loss in 95% of cases
   • Critical for regulatory compliance
   • Parametric VaR uses fitted distribution
   • More accurate than historical VaR

3. DISTRIBUTION CHOICE MATTERS:
   • Normal: Underestimates tail risk
   • t-distribution: Better for fat tails
   • Choice affects VaR by 20-50%!

4. PORTFOLIO OPTIMIZATION:
   • Diversification reduces risk
   • Efficient frontier shows optimal trade-offs
   • Sharpe ratio balances return vs risk

5. BEST PRACTICES:
   ✓ Always test normality assumption
   ✓ Check Q-Q plot for tail behavior
   ✓ Use appropriate distribution for VaR
   ✓ Backtest risk models regularly
   ✓ Account for fat tails in stress testing

6. REGULATORY CONTEXT:
   • Basel III requires VaR calculation
   • 99% VaR common for banks
   • Expected Shortfall (ES) supplements VaR

Next: See reliability_engineering.py!
""")
