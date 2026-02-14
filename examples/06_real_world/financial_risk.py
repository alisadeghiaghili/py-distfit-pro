#!/usr/bin/env python3
"""
Financial Risk Management Example
=================================

Calculate Value-at-Risk (VaR) using distribution fitting.
Common in: Finance, Investment, Risk Management.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
import matplotlib.pyplot as plt

np.random.seed(42)

print("="*70)
print("💹 FINANCIAL RISK: Value-at-Risk (VaR) Calculation")
print("="*70)
print("""
Scenario: Portfolio risk management
  - Daily returns of an investment portfolio
  - Need to estimate VaR for risk reporting
  - Regulatory requirement: 95% and 99% VaR
""")


# ============================================================================
# Data: Daily Portfolio Returns (%)
# ============================================================================

# Simulate daily returns (slightly heavy-tailed, typical of real markets)
# Using Student's t with df=5 for heavier tails than normal
from scipy import stats

n_days = 1000
returns_pct = stats.t.rvs(df=5, loc=0.05, scale=1.5, size=n_days)

print(f"\n📊 Portfolio Data: {n_days} trading days")
print(f"  Mean return:  {returns_pct.mean():.3f}%")
print(f"  Std Dev:      {returns_pct.std():.3f}%")
print(f"  Min:          {returns_pct.min():.3f}%")
print(f"  Max:          {returns_pct.max():.3f}%")
print(f"  Median:       {np.median(returns_pct):.3f}%")

# Calculate basic risk metrics
positive_days = (returns_pct > 0).sum()
print(f"\n  Positive return days: {positive_days}/{n_days} ({positive_days/n_days*100:.1f}%)")
print(f"  Negative return days: {n_days-positive_days}/{n_days} ({(n_days-positive_days)/n_days*100:.1f}%)")


# ============================================================================
# Compare Distribution Fits
# ============================================================================

print("\n" + "="*70)
print("🔍 Testing Distribution Models")
print("="*70)

candidates = ['normal', 't', 'lognormal']
results = {}

for name in candidates:
    try:
        dist = get_distribution(name)
        dist.fit(returns_pct)
        
        aic = dist.aic()
        results[name] = {'dist': dist, 'aic': aic}
        
        print(f"\n{name:12s}  AIC = {aic:8.2f}")
        
    except Exception as e:
        print(f"\n{name:12s}  Failed: {e}")

# Select best model
best_name = min(results.keys(), key=lambda k: results[k]['aic'])
best_dist = results[best_name]['dist']

print(f"\n✅ Best fit: {best_name} (lowest AIC)")
print(best_dist.summary())


# ============================================================================
# Value-at-Risk (VaR) Calculation
# ============================================================================

print("\n" + "="*70)
print("📊 Value-at-Risk (VaR) Analysis")
print("="*70)
print("""
VaR Definition:
  "With X% confidence, we won't lose more than VaR in one day"
  
Example: 95% VaR = -$50,000
  → 95% of days, losses will be < $50,000
  → Only 5% of days will see losses > $50,000
""")

confidence_levels = [0.90, 0.95, 0.99]
portfolio_value = 1_000_000  # $1 million

print(f"\nPortfolio Value: ${portfolio_value:,.0f}")
print(f"\n" + "="*70)
print(f"VaR Estimates (using {best_name} distribution):")
print("="*70)
print(f"\n  Confidence    VaR (%)      VaR ($)         Interpretation")
print("  " + "-"*65)

for conf in confidence_levels:
    # VaR is the negative of the percentile (we care about losses)
    var_pct = best_dist.ppf(1 - conf)  # Lower tail
    var_dollar = portfolio_value * var_pct / 100
    
    print(f"    {conf*100:.0f}%        {var_pct:7.3f}%    ${var_dollar:10,.0f}      "
          f"{100-conf*100:.0f}% chance of worse loss")

# Compare with empirical VaR (actual data)
print(f"\n" + "="*70)
print("Comparison: Model-based vs Empirical VaR")
print("="*70)
print(f"\n  Confidence    Model VaR    Empirical VaR    Difference")
print("  " + "-"*60)

for conf in confidence_levels:
    model_var = best_dist.ppf(1 - conf)
    empirical_var = np.percentile(returns_pct, (1 - conf) * 100)
    diff = model_var - empirical_var
    
    print(f"    {conf*100:.0f}%         {model_var:7.3f}%      {empirical_var:7.3f}%       {diff:+6.3f}%")


# ============================================================================
# Conditional VaR (CVaR) / Expected Shortfall
# ============================================================================

print(f"\n" + "="*70)
print("📉 Conditional VaR (Expected Shortfall)")
print("="*70)
print("""
CVaR = Average loss GIVEN that VaR is exceeded
"Tail risk": If things go wrong, how bad will it be on average?
""")

conf = 0.95
var_95 = best_dist.ppf(1 - conf)

# CVaR: average of all returns worse than VaR
worst_returns = returns_pct[returns_pct < var_95]
cvar_empirical = worst_returns.mean() if len(worst_returns) > 0 else var_95

print(f"\n95% VaR:  {var_95:.3f}%")
print(f"95% CVaR: {cvar_empirical:.3f}%")
print(f"\nInterpretation:")
print(f"  - On a bad day (worst 5%), we expect to lose {var_95:.2f}% or more")
print(f"  - Average loss on those bad days: {cvar_empirical:.2f}%")
print(f"  - In dollars: ${portfolio_value * cvar_empirical / 100:,.0f}")


# ============================================================================
# Stress Testing
# ============================================================================

print(f"\n" + "="*70)
print("⚠️  Stress Testing: Extreme Scenarios")
print("="*70)

extreme_percentiles = [0.001, 0.005, 0.01]  # Very rare events

print(f"\n  Probability    Daily Loss    Portfolio Loss    Scenario")
print("  " + "-"*70)

for p in extreme_percentiles:
    loss_pct = best_dist.ppf(p)
    loss_dollar = portfolio_value * loss_pct / 100
    frequency = f"~{int(1/p)} days"
    
    print(f"    {p*100:.2f}%         {loss_pct:7.2f}%     ${loss_dollar:11,.0f}     "
          f"Once every {frequency}")

print(f"\n  ⚠️  These are VERY rare but possible losses!")


# ============================================================================
# Visualization: Risk Dashboard
# ============================================================================

print("\n" + "="*70)
print("📊 Creating Risk Dashboard...")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Return Distribution + VaR markers
ax = axes[0, 0]
ax.hist(returns_pct, bins=50, density=True, alpha=0.6, color='skyblue', edgecolor='black')

x = np.linspace(returns_pct.min(), returns_pct.max(), 500)
y = best_dist.pdf(x)
ax.plot(x, y, 'r-', linewidth=2, label=f'Fitted {best_name}')

# Mark VaR levels
for conf in [0.95, 0.99]:
    var = best_dist.ppf(1 - conf)
    ax.axvline(var, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(var, ax.get_ylim()[1]*0.9, f'{conf*100:.0f}% VaR', 
            rotation=90, va='top', ha='right', fontsize=9)

ax.set_xlabel('Daily Return (%)', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Return Distribution & VaR Thresholds', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Cumulative returns
ax = axes[0, 1]
cumulative_returns = np.cumprod(1 + returns_pct/100) - 1
ax.plot(cumulative_returns * 100, linewidth=1.5)
ax.axhline(0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Trading Day', fontsize=11)
ax.set_ylabel('Cumulative Return (%)', fontsize=11)
ax.set_title('Portfolio Performance Over Time', fontweight='bold')
ax.grid(True, alpha=0.3)

# 3. Q-Q Plot (check model fit)
ax = axes[1, 0]
percentiles = np.linspace(0.01, 0.99, len(returns_pct))
theoretical = best_dist.ppf(percentiles)
empirical = np.sort(returns_pct)

ax.scatter(theoretical, empirical, alpha=0.5, s=20)
min_val = min(theoretical.min(), empirical.min())
max_val = max(theoretical.max(), empirical.max())
ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax.set_xlabel('Theoretical Quantiles', fontsize=11)
ax.set_ylabel('Empirical Quantiles', fontsize=11)
ax.set_title('Q-Q Plot: Model Fit Check', fontweight='bold')
ax.grid(True, alpha=0.3)

# 4. VaR comparison table (as text)
ax = axes[1, 1]
ax.axis('off')

summary = f"""
RISK SUMMARY
{'='*40}

Portfolio Value: ${portfolio_value:,.0f}
Daily Return: {returns_pct.mean():.2f}% ± {returns_pct.std():.2f}%

VALUE-AT-RISK (VaR)
{'-'*40}
"""

for conf in confidence_levels:
    var_pct = best_dist.ppf(1 - conf)
    var_dollar = portfolio_value * var_pct / 100
    summary += f"\n{conf*100:.0f}% VaR: {var_pct:6.2f}%  (${var_dollar:,.0f})"

summary += f"""

{'='*40}

CONDITIONAL VaR (95%)
{'-'*40}
Average loss when VaR exceeded:
  {cvar_empirical:.2f}% (${portfolio_value * cvar_empirical / 100:,.0f})

{'='*40}

Best Fit Model: {best_name}
AIC: {results[best_name]['aic']:.2f}
"""

ax.text(0.1, 0.9, summary, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle('Financial Risk Dashboard', fontsize=14, fontweight='bold')
plt.tight_layout()

print("\n✅ Dashboard created!")
plt.show()


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. VaR = "With X% confidence, we won't lose more than VaR"

2. Regulatory standards:
   - Basel III: 99% VaR (banks)
   - Common practice: 95% VaR

3. CVaR (Expected Shortfall) > VaR:
   - More conservative (accounts for tail risk)
   - Preferred by modern risk management

4. Distribution matters:
   - Normal underestimates tail risk
   - Student's t captures heavy tails better
   - Always validate with Q-Q plots!

5. VaR limitations:
   - Assumes historical patterns continue
   - Doesn't predict black swan events
   - Complement with stress testing

Next: See quality_control.py for manufacturing SPC!
""")
