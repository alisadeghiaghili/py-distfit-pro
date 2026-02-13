#!/usr/bin/env python3
"""
Quick fix script for distributions.py parameter names.
Run this in py-distfit-pro-main directory.
"""

import os

FILE_PATH = 'distfit_pro/core/distributions.py'

if not os.path.exists(FILE_PATH):
    print(f"❌ File not found: {FILE_PATH}")
    print("Run this script from the py-distfit-pro-main directory!")
    exit(1)

with open(FILE_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

original_content = content

# === WEIBULL FIXES ===
print("🔧 Fixing Weibull...")
content = content.replace(
    "self._params = {'alpha': shape, 'a': shape, 'scale': scale}",
    "self._params = {'c': shape, 'scale': scale}"
)
content = content.replace(
    "alpha = self._params['alpha']\n        scale = self._params['scale']\n        if alpha > 1:\n            return scale * ((alpha - 1) / alpha) ** (1 / alpha)",
    "c = self._params['c']\n        scale = self._params['scale']\n        if c > 1:\n            return scale * ((c - 1) / c) ** (1 / c)"
)
content = content.replace(
    "return {'c': self._params['alpha'], 'loc': 0, 'scale': self._params['scale']}",
    "return {'c': self._params['c'], 'loc': 0, 'scale': self._params['scale']}"
)

# === LOGNORMAL FIXES ===  
print("🔧 Fixing Lognormal...")
# Fix _fit_mle
content = content.replace(
    "self._params = {'alpha': shape, 'a': shape, 'scale': scale}",
    "self._params = {'s': shape, 'scale': scale}"
)
# Fix _fit_mom
content = content.replace(
    "self._params = {'alpha': s, 'a': s, 'scale': scale}",
    "self._params = {'s': s, 'scale': scale}"
)
# Fix mode()
content = content.replace(
    "alpha = self._params['alpha']\n        scale = self._params['scale']\n        return scale * np.exp(-alpha**2)",
    "s = self._params['s']\n        scale = self._params['scale']\n        return scale * np.exp(-s**2)"
)
# Fix _get_scipy_params
content = content.replace(
    "return {'s': self._params['alpha'], 'loc': 0, 'scale': self._params['scale']}",
    "return {'s': self._params['s'], 'loc': 0, 'scale': self._params['scale']}"
)

if content == original_content:
    print("❌ No changes made - file might already be fixed or patterns don't match!")
    exit(1)

# Write back
with open(FILE_PATH, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ Fixed Weibull: alpha → c")
print("✅ Fixed Lognormal: alpha → s") 
print("")
print("⚠️  Hypergeometric still needs manual fix (scipy param mapping)")
print("")
print("Now run: pytest tests/test_distributions_basic.py -v")
