# Goodness-of-Fit Testing

## What is Goodness-of-Fit (GOF)?

GOF tests answer: **"Does this distribution actually fit my data?"**

Without GOF tests, you might fit any distribution regardless of fit quality. GOF tests quantify how well the fit matches reality.

## Statistical Tests Available

### 1. Kolmogorov-Smirnov Test (KS)

**What it tests:** Distance between data and fitted CDF

```python
from distfit_pro.core.gof_tests import KolmogorovSmirnov

ks_result = KolmogorovSmirnov.test(data, dist)
print(f"Test Statistic: {ks_result.statistic:.4f}")
print(f"P-value: {ks_result.pvalue:.4f}")
print(f"Result: {ks_result.result}")  # 'Pass' or 'Fail'
```

**When to use:**
- General purpose GOF test
- Continuous distributions
- Medium sample sizes (30 < n < 10000)
- Sensitive to location and shape

**Interpretation:**
- p-value > 0.05: Fit is good ✅
- p-value < 0.05: Fit is poor ❌

### 2. Anderson-Darling Test (AD)

**What it tests:** Distance with emphasis on tails

```python
from distfit_pro.core.gof_tests import AndersonDarling

ad_result = AndersonDarling.test(data, dist)
print(f"Test Statistic: {ad_result.statistic:.4f}")
print(f"Critical Value (5%): {ad_result.critical_values[2]:.4f}")
print(f"Result: {ad_result.result}")
```

**When to use:**
- When tails are important
- More sensitive than KS
- Better for small samples
- Extreme value analysis

### 3. Chi-Square Test (χ²)

**What it tests:** Frequency counts in bins

```python
from distfit_pro.core.gof_tests import ChiSquare

chi2_result = ChiSquare.test(data, dist, bins=10)
print(f"Test Statistic: {chi2_result.statistic:.4f}")
print(f"P-value: {chi2_result.pvalue:.4f}")
print(f"Degrees of freedom: {chi2_result.dof}")
```

**When to use:**
- Discrete or grouped data
- Large sample sizes (n > 100)
- Binned data
- Goodness-of-fit for categories

### 4. Cramér-von Mises Test (CVM)

**What it tests:** Quadratic distance to CDF

```python
from distfit_pro.core.gof_tests import CramerVonMises

cvm_result = CramerVonMises.test(data, dist)
print(f"Test Statistic: {cvm_result.statistic:.4f}")
print(f"P-value: {cvm_result.pvalue:.4f}")
```

**When to use:**
- Alternative to KS
- More balanced (middle vs tails)
- All sample sizes

## Running All Tests

### Quick Summary

```python
from distfit_pro.core.gof_tests import GOFTests

# Run all 4 tests at once
results = GOFTests.run_all_tests(data, dist)

# Print summary table
print(GOFTests.summary_table(results))

# Output:
# Test              | Statistic | P-value | Result
# ─────────────────────────────────────────────────
# KS                | 0.0245    | 0.982   | PASS ✓
# Anderson-Darling  | 0.234     | 0.843   | PASS ✓
# Chi-Square        | 12.45     | 0.195   | PASS ✓
# Cramér-von Mises  | 0.0156    | 0.921   | PASS ✓
```

## Interpretation Guide

### All Tests Pass (p > 0.05)

✅ **Excellent fit!** The distribution describes your data well.

```python
if all(r.pvalue > 0.05 for r in results.values()):
    print("All tests pass - distribution is appropriate!")
    dist_name = dist.name
else:
    print("Some tests failed - try another distribution")
```

### Most Tests Pass

⚠️ **Good fit, but minor issues.** Consider:
- Sample size (small samples less reliable)
- Outliers (remove and retest)
- Test sensitivity

```python
passing = sum(1 for r in results.values() if r.pvalue > 0.05)
print(f"Passed {passing}/4 tests")
```

### Most Tests Fail (p < 0.05)

❌ **Poor fit.** Try:
- Different distribution
- Remove outliers
- Transform data
- Mixture model

## Workflow: Testing Multiple Distributions

```python
import numpy as np
from distfit_pro import get_distribution
from distfit_pro.core.gof_tests import GOFTests

# Your data
data = np.random.weibull(2, 1000)

# Candidates
candidates = ['normal', 'weibull', 'gamma', 'lognormal']

# Compare
results_dict = {}
for name in candidates:
    dist = get_distribution(name)
    dist.fit(data)
    
    results = GOFTests.run_all_tests(data, dist)
    passing = sum(1 for r in results.values() if r.pvalue > 0.05)
    
    results_dict[name] = {
        'aic': dist.aic,
        'bic': dist.bic,
        'gof_pass': passing,
        'results': results
    }

# Rank by GOF tests passed
ranked = sorted(results_dict.items(), 
                key=lambda x: x[1]['gof_pass'], 
                reverse=True)

for name, metrics in ranked:
    print(f"{name:12} AIC={metrics['aic']:7.1f} GOF={metrics['gof_pass']}/4 tests")
```

## When GOF Tests Disagree

### Different tests show different results?

This is normal! Tests measure different aspects:

- **KS & CVM:** Overall distance
- **AD:** Emphasizes tails
- **Chi-Square:** Frequency counts

```python
# Solution: Look at consensus
from collections import Counter

passing_counts = Counter()
for test_name, result in results.items():
    status = 'pass' if result.pvalue > 0.05 else 'fail'
    passing_counts[status] += 1

if passing_counts['pass'] >= 3:  # 3 out of 4
    print("Consensus: Good fit")
else:
    print("Consensus: Poor fit")
```

## Sample Size Considerations

### Small Sample (n < 30)
- ⚠️ Tests less reliable
- Larger confidence intervals
- False negatives more likely (good fits rejected)

```python
if len(data) < 30:
    print("Warning: Small sample size. Results less reliable.")
    print("Consider bootstrap instead.")
```

### Large Sample (n > 5000)
- ✅ Tests very sensitive
- Small deviations detected
- May reject good fits that aren't perfect

```python
if len(data) > 5000:
    print("Large sample: Tests very sensitive.")
    print("Visually inspect Q-Q plot too.")
    dist.plot(plot_type='qq')
```

## Next Steps

🔍 See [Tutorial 3: GOF Tests](../tutorials/03_goodness_of_fit.md)

🔍 See [Tutorial 4: Bootstrap CI](../tutorials/04_bootstrap_ci.md)
