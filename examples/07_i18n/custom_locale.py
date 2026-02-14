#!/usr/bin/env python3
"""
Custom Locale Creation Example
==============================

Create your own language translations for distfit-pro.

Author: Ali Sadeghi Aghili
"""

import numpy as np
from distfit_pro import get_distribution
from distfit_pro.i18n import LocaleManager

np.random.seed(42)

print("="*70)
print("🌍 CUSTOM LOCALE CREATION")
print("="*70)
print("""
Scenario: Adding Spanish (es) support to distfit-pro

Steps:
  1. Create translation dictionary
  2. Register with LocaleManager
  3. Use your custom locale
""")


# ============================================================================
# Step 1: Create Spanish Translation Dictionary
# ============================================================================

print("\n" + "="*70)
print("STEP 1: Define Spanish Translations")
print("="*70)

spanish_translations = {
    # Distribution names
    'Normal': 'Normal',
    'Exponential': 'Exponencial',
    'Lognormal': 'Lognormal',
    'Gamma': 'Gamma',
    'Weibull': 'Weibull',
    'Beta': 'Beta',
    'Uniform': 'Uniforme',
    'Chi-Square': 'Chi-Cuadrado',
    "Student's t": 't de Student',
    'Poisson': 'Poisson',
    
    # Summary labels
    'Distribution': 'Distribución',
    'Fitted Parameters': 'Parámetros Ajustados',
    'Statistics': 'Estadísticas',
    'Model Fit': 'Ajuste del Modelo',
    
    # Parameter names
    'loc': 'ubicación',
    'scale': 'escala',
    'shape': 'forma',
    'df': 'grados de libertad',
    'mu': 'mu',
    'sigma': 'sigma',
    'alpha': 'alfa',
    'beta': 'beta',
    'lambda': 'lambda',
    'rate': 'tasa',
    
    # Statistics
    'Mean': 'Media',
    'Median': 'Mediana',
    'Mode': 'Moda',
    'Variance': 'Varianza',
    'Std Dev': 'Desviación Estándar',
    'Skewness': 'Asimetría',
    'Kurtosis': 'Curtosis',
    
    # Model fit
    'Log-Likelihood': 'Log-Verosimilitud',
    'AIC': 'AIC',
    'BIC': 'BIC',
    'Sample Size': 'Tamaño de Muestra',
    'Fitted': 'Ajustado',
    'Yes': 'Sí',
    'No': 'No',
}

print("\n✅ Spanish translations defined")
print(f"   {len(spanish_translations)} terms translated")


# ============================================================================
# Step 2: Register Spanish Locale
# ============================================================================

print("\n" + "="*70)
print("STEP 2: Register Spanish Locale")
print("="*70)

# Get the locale manager instance
locale_manager = LocaleManager()

# Register Spanish locale
locale_manager.register_locale('es', spanish_translations)

print("\n✅ Spanish locale registered as 'es'")
print(f"   Available locales: {locale_manager.available_locales()}")


# ============================================================================
# Step 3: Use Custom Spanish Locale
# ============================================================================

print("\n" + "="*70)
print("STEP 3: Use Spanish Locale")
print("="*70)

# Generate sample data
data = np.random.normal(loc=100, scale=15, size=1000)

# Fit distribution
dist = get_distribution('normal')
dist.fit(data)

print("\n🇪🇸 Spanish Output:")
print(dist.summary(locale='es'))


# ============================================================================
# Step 4: Compare All Languages
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: All Available Languages")
print("="*70)

locales = [
    ('en', '🇬🇧 English'),
    ('fa', '🇮🇷 Persian'),
    ('de', '🇩🇪 German'),
    ('es', '🇪🇸 Spanish (Custom)'),
]

for locale_code, locale_name in locales:
    print(f"\n{locale_name}:")
    print("-" * 70)
    print(dist.summary(locale=locale_code))


# ============================================================================
# Step 5: Custom Locale for Another Distribution
# ============================================================================

print("\n" + "="*70)
print("EXAMPLE: Exponential Distribution in Spanish")
print("="*70)

data_exp = np.random.exponential(scale=5, size=1000)
dist_exp = get_distribution('expon')
dist_exp.fit(data_exp)

print("\n🇪🇸 Distribución Exponencial:")
print(dist_exp.summary(locale='es'))


# ============================================================================
# Step 6: Partial Translation (Fallback to English)
# ============================================================================

print("\n" + "="*70)
print("STEP 6: Partial Translation Example")
print("="*70)
print("""
Note: If a term is not in your translation dictionary,
distfit-pro automatically falls back to English for that term.
""")

# Create minimal translation (only a few terms)
minimal_spanish = {
    'Distribution': 'Distribución',
    'Normal': 'Normal',
    'Mean': 'Media',
    # Other terms will fall back to English
}

# Register minimal locale
locale_manager.register_locale('es_minimal', minimal_spanish)

print("\n🇪🇸 Spanish (Minimal - with English fallback):")
print(dist.summary(locale='es_minimal'))

print("\nℹ️  Notice: Some terms are in Spanish, others in English")


# ============================================================================
# Step 7: Programmatic Translation Updates
# ============================================================================

print("\n" + "="*70)
print("STEP 7: Adding Translations Dynamically")
print("="*70)

# You can update translations after registration
locale_manager.add_translation('es', 'Fitted Parameters', 'Parámetros del Modelo')
locale_manager.add_translation('es', 'Sample Size', 'Número de Muestras')

print("\n✅ Updated Spanish translations")
print("\n🇪🇸 Spanish (Updated):")
print(dist.summary(locale='es'))


print("\n" + "="*70)
print("🎓 Key Takeaways")
print("="*70)
print("""
1. Create custom locale in 3 steps:
   a) Define translation dictionary
   b) Register with LocaleManager
   c) Use with summary(locale='XX')

2. Translation dictionary structure:
   {
       'English Term': 'Translated Term',
       ...
   }

3. Automatic fallback to English for missing translations

4. You can update translations dynamically:
   locale_manager.add_translation('XX', 'English', 'Translation')

5. Common terms to translate:
   - Distribution names
   - Parameter names (loc, scale, shape, etc.)
   - Statistics (Mean, Median, Std Dev, etc.)
   - Model fit metrics (AIC, BIC, Log-Likelihood)

6. Best practice: Start with full translation dictionary
   to avoid mixed-language output

7. To add permanently: Submit PR to distfit-pro with your locale!

Example languages to add:
  - French (fr)
  - Spanish (es) ← We just did this!
  - Arabic (ar)
  - Chinese (zh)
  - Japanese (ja)
  - Russian (ru)
  - ... any language you need!

""")

print("="*70)
print("🚀 Ready to contribute your locale to distfit-pro?")
print("="*70)
print("""
To add your language permanently:

1. Fork the repo: github.com/alisadeghiaghili/py-distfit-pro

2. Add your locale file:
   distfit_pro/i18n/locales/your_language.py

3. Follow the pattern of existing locale files:
   - en.py (English)
   - fa.py (Persian)
   - de.py (German)

4. Submit a Pull Request

5. Help make distfit-pro multilingual! 🌍
""")
