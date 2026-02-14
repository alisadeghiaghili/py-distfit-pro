#!/usr/bin/env python3
"""
Multilingual Output Example
==========================

Generate distribution summaries in multiple languages.
Supported: English, Persian (Farsi), German.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution

np.random.seed(42)

print("="*70)
print("🌍 MULTILINGUAL OUTPUT DEMONSTRATION")
print("="*70)

# Generate sample data
data = np.random.normal(loc=100, scale=15, size=1000)


# ============================================================================
# Example 1: English (Default)
# ============================================================================

print("\n" + "="*70)
print("🇬🇧 ENGLISH (Default)")
print("="*70)

dist_en = get_distribution('normal')
dist_en.fit(data)

print(dist_en.summary())  # Default is English

# Or explicitly specify locale
print("\n" + "-"*70)
print(dist_en.summary(locale='en'))


# ============================================================================
# Example 2: Persian (Farsi)
# ============================================================================

print("\n" + "="*70)
print("🇮🇷 PERSIAN (فارسی)")
print("="*70)

dist_fa = get_distribution('normal')
dist_fa.fit(data)

print(dist_fa.summary(locale='fa'))


# ============================================================================
# Example 3: German
# ============================================================================

print("\n" + "="*70)
print("🇩🇪 GERMAN (Deutsch)")
print("="*70)

dist_de = get_distribution('normal')
dist_de.fit(data)

print(dist_de.summary(locale='de'))


# ============================================================================
# Example 4: Different Distributions in Multiple Languages
# ============================================================================

print("\n" + "="*70)
print("🔀 DIFFERENT DISTRIBUTIONS IN MULTIPLE LANGUAGES")
print("="*70)

# Exponential distribution
data_exp = np.random.exponential(scale=5, size=1000)

print("\n" + "-"*70)
print("EXPONENTIAL DISTRIBUTION")
print("-"*70)

dist_exp = get_distribution('expon')
dist_exp.fit(data_exp)

print("\n🇬🇧 English:")
print(dist_exp.summary(locale='en'))

print("\n🇮🇷 Persian:")
print(dist_exp.summary(locale='fa'))

print("\n🇩🇪 German:")
print(dist_exp.summary(locale='de'))


# ============================================================================
# Example 5: Lognormal Distribution
# ============================================================================

data_lognorm = np.random.lognormal(mean=3, sigma=0.5, size=1000)

print("\n" + "="*70)
print("LOGNORMAL DISTRIBUTION")
print("="*70)

dist_lognorm = get_distribution('lognormal')
dist_lognorm.fit(data_lognorm)

for locale, flag in [('en', '🇬🇧'), ('fa', '🇮🇷'), ('de', '🇩🇪')]:
    print(f"\n{flag} {locale.upper()}:")
    print(dist_lognorm.summary(locale=locale))


# ============================================================================
# Example 6: Error Handling (Invalid Locale)
# ============================================================================

print("\n" + "="*70)
print("⚠️ ERROR HANDLING: Invalid Locale")
print("="*70)

try:
    print(dist_en.summary(locale='invalid_locale'))
except ValueError as e:
    print(f"\n❌ Error caught: {e}")
    print("\nSupported locales: en, fa, de")
    print("Falls back to English if unsupported locale provided.")


# ============================================================================
# Example 7: Statistical Methods in Different Languages
# ============================================================================

print("\n" + "="*70)
print("📊 STATISTICAL METHODS OUTPUT")
print("="*70)

print("""
Note: Statistical methods (pdf, cdf, etc.) return numeric values.
Only summary() method provides translated text output.
""")

print("\n" + "-"*70)
print("🇬🇧 English Summary:")
print(dist_en.summary(locale='en'))

print("\n" + "-"*70)
print("Numeric methods (language-independent):")
print(f"  Mean:   {dist_en.mean():.2f}")
print(f"  Median: {dist_en.median():.2f}")
print(f"  Std:    {dist_en.std():.2f}")
print(f"  PDF(100): {dist_en.pdf(100):.6f}")
print(f"  CDF(100): {dist_en.cdf(100):.6f}")


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Use summary(locale='XX') to get output in different languages:
   - 'en': English (default)
   - 'fa': Persian/Farsi
   - 'de': German

2. All distribution types support multilingual output

3. Statistical methods (pdf, cdf, mean, etc.) are language-independent
   (they return numbers, not text)

4. Only summary() provides translated text

5. Falls back to English for unsupported locales

6. Easy to add custom locales (see custom_locale.py example)

Next: See custom_locale.py to add your own language!
""")
