# Goodness-of-Fit Tests

## Overview

Four statistical tests to assess if a distribution fits your data.

## Test Comparison

| Test | Speed | Sensitivity | Best For |
|------|-------|-------------|----------|
| **KS** | Fast | Medium | General purpose |
| **AD** | Fast | High | Tails important |
| **Chi2** | Medium | Medium | Binned data |
| **CVM** | Medium | Medium | Balanced |

---

## Kolmogorov-Smirnov Test

```python
from distfit_pro.core.gof_tests import KolmogorovSmirnov

result = KolmogorovSmirnov.test(data, dist)
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")
print(f"Critical value: {result.critical_value}")
```

**Interprets statistic as:** Maximum distance between empirical and theoretical CDF

**P-value > 0.05:** Fit is adequate ✅

---

## Anderson-Darling Test

```python
from distfit_pro.core.gof_tests import AndersonDarling

result = AndersonDarling.test(data, dist)
print(f"Statistic: {result.statistic}")
print(f"Critical values: {result.critical_values}")
print(f"Significance levels: {result.significance_levels}")
```

**Note:** Compare statistic to critical values (not p-value)

**Statistic < critical_value[2] (5% level):** Fit is adequate ✅

---

## Chi-Square Test

```python
from distfit_pro.core.gof_tests import ChiSquare

result = ChiSquare.test(data, dist, bins=10)
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")
print(f"Degrees of freedom: {result.dof}")
```

**Parameters:**
- `bins` (int): Number of bins to group data

**P-value > 0.05:** Fit is adequate ✅

---

## Cramér-von Mises Test

```python
from distfit_pro.core.gof_tests import CramerVonMises

result = CramerVonMises.test(data, dist)
print(f"Statistic: {result.statistic}")
print(f"P-value: {result.pvalue}")
```

**Similar to KS but:** More balanced (emphasizes center over extremes)

**P-value > 0.05:** Fit is adequate ✅

---

## Running All Tests

```python
from distfit_pro.core.gof_tests import GOFTests

results = GOFTests.run_all_tests(data, dist)

# Print nice table
print(GOFTests.summary_table(results))

# Access individual results
for test_name, result in results.items():
    print(f"{test_name}: p-value={result.pvalue:.4f}")
```

---

## Interpreting Results

### All Tests Pass (p > 0.05)

✅ **Excellent fit** - Use this distribution with confidence

### Most Tests Pass (3/4)

⚠️ **Good fit** - Acceptable, but verify visually with Q-Q plot

### Some Tests Pass (2/4)

⚠️ **Marginal fit** - Consider alternative distributions

### Most Tests Fail (< 2/4)

❌ **Poor fit** - Try different distribution or transform data

---

## Advanced

### Custom Significance Level

```python
# Instead of default 0.05
alpha = 0.01  # Stricter

result = KolmogorovSmirnov.test(data, dist)
if result.pvalue > alpha:
    print("Passes at 1% significance")
```

### Handling Disagreement

When tests disagree:

```python
# Vote by consensus
passing = sum(1 for r in results.values() if r.pvalue > 0.05)

if passing >= 3:  # 3+ out of 4 pass
    print("Consensus: Good fit")
else:
    print("Consensus: Poor fit")
```

---

## See Also

- [Goodness-of-Fit Guide](../user_guide/05_goodness_of_fit.md)
- [API Reference](index.md)
