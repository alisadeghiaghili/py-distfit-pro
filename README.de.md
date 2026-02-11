# DistFit Pro 🎯

**Professionelle Verteilungsanpassung für Python**

Ein umfassendes, produktionsreifes Paket, das die besten Funktionen von EasyFit und R's fitdistrplus kombiniert, mit modernen Verbesserungen in statistischer Methodik, Benutzererfahrung und Software-Engineering.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[English](README.md) | [Persian/فارسی](README.fa.md) | **Deutsch**

---

## 🚀 Warum DistFit Pro?

### Bessere statistische Philosophie
- ✅ **Modellauswahl über AIC/BIC/WAIC/LOO-CV** statt nur p-Werten
- ✅ **Bayessche Modellmittelung** für robuste Inferenz
- ✅ **Automatische Schweifverhalten-Erkennung** und Ausreißer-Diagnose
- ✅ **Korrektur für multiples Testen** zur Vermeidung von falsch-positiven Ergebnissen

### Bessere Benutzererfahrung
- ✅ **Scikit-learn-ähnliche API** - intuitiv und konsistent
- ✅ **Umfangreiche Visualisierungen** mit matplotlib/seaborn/plotly
- ✅ **Selbsterklärende Ausgaben** - jeder Schritt ist dokumentiert
- ✅ **Mehrsprachige Unterstützung** - Deutsch, Englisch, Persisch
- ✅ **Umfassende Dokumentation** und Tutorials

### Bessere Erweiterbarkeit
- ✅ **Benutzerdefinierte Verteilungen** leicht gemacht
- ✅ **Mischmodelle** integriert
- ✅ **Hierarchische/mehrstufige Anpassung** unterstützt
- ✅ **Modulare Architektur** für einfache Erweiterung

### Bessere Leistung
- ✅ **Optimiert für große Datensätze**
- ✅ **Parallele Verarbeitung** über joblib
- ✅ **GPU-Beschleunigung** (optional, über CuPy)
- ✅ **Effiziente Algorithmen** mit numba JIT

---

## 📦 Installation

```bash
pip install distfit-pro
```

Für Entwicklung:
```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e ".[dev]"
```

---

## 🎯 Schnellstart

```python
import numpy as np
from distfit_pro import set_language, DistributionFitter

# Sprache auf Deutsch setzen
set_language('de')

# Beispieldaten generieren
np.random.seed(42)
data = np.random.lognormal(mean=2, sigma=0.5, size=1000)

# Verteilungen anpassen
fitter = DistributionFitter(data)
results = fitter.fit(
    distributions=['lognormal', 'gamma', 'weibull', 'normal'],
    method='mle',  # oder 'moments', 'quantile'
    n_jobs=-1  # parallele Verarbeitung
)

# Selbsterklärende Ergebnisse ausgeben (auf Deutsch!)
print(results.summary())

# Visualisieren
results.plot(kind='comparison')  # P-P, Q-Q, PDF, CDF
results.plot(kind='diagnostics')  # Residuen, Schweifverhalten

# Bestes Modell mit Erklärung erhalten
best = results.best_model
print(best.explain())  # ✅ Ausgabe auf Deutsch!

# Parameter und Statistiken abrufen
print(best.params)      # Angepasste Parameter
print(best.mean())      # Verteilungsmittelwert
print(best.variance())  # Verteilungsvarianz
```

---

## 🌐 Mehrsprachige Unterstützung

DistFit Pro unterstützt **drei Sprachen** für alle Ausgaben:

```python
from distfit_pro import set_language

# 🇩🇪 Deutsch
set_language('de')
print(dist.explain())
# Ausgabe:
# 📊 Geschätzte Parameter:
#    • Einkommen
#    • Aktienkurse
# 🔍 Eigenschaften:
#    • Rechtsschief
#    • Nur positive Werte

# 🇬🇧 English
set_language('en')
print(dist.explain())
# Output:
# 📊 Estimated Parameters:
#    • Income
#    • Stock prices
# 🔍 Characteristics:
#    • Right-skewed
#    • Positive values only

# 🇮🇷 Persian/Farsi
set_language('fa')
print(dist.explain())
# خروجی:
# 📊 پارامترهای برآورد شده:
#    • درآمد
#    • قیمت سهام
# 🔍 ویژگی‌ها:
#    • راست‌چوله
#    • فقط مقادیر مثبت
```

---

## 📚 Kernfunktionen

### 1. Umfassende Verteilungsunterstützung

**Stetige Verteilungen (30+):**
- Normal, Log-Normal, Exponential, Gamma, Weibull
- Beta, Chi-Quadrat, Student-t, F, Cauchy
- Pareto, Gumbel, GEV, Rayleigh, Rice
- Burr, Inverse Gamma, Log-Logistisch, Nakagami
- Und mehr...

**Diskrete Verteilungen (15+):**
- Poisson, Binomial, Negativ-Binomial
- Geometrisch, Hypergeometrisch, Multinomial
- Zero-inflated Varianten

### 2. Fortgeschrittene Schätzmethoden

- **Maximum-Likelihood (MLE)** - Standard, effizient
- **Momentenmethode** - robust gegenüber Ausreißern
- **Quantil-Anpassung** - passt spezifische Perzentile an
- **Maximum Goodness-of-Fit** - optimiert GOF-Statistik
- **Bayessche Schätzung** - volle Posterior mit Unsicherheit

### 3. Modellauswahlkriterien

- **AIC/BIC** - bestraftes Likelihood
- **WAIC** - Bayessches Informationskriterium
- **LOO-CV** - Leave-One-Out Kreuzvalidierung
- **K-fache CV** - robuste Kreuzvalidierung
- **Bayessche Modellmittelung** - gewichtetes Ensemble

### 4. Zensierte und gestutzte Daten

Unterstützung für:
- Rechtszensierte Daten (Survival-Analyse)
- Linksgestutzte Daten
- Intervallzensierte Daten

### 5. Mischmodelle

Anpassung von Mischverteilungen mittels EM-Algorithmus mit automatischer Komponentenauswahl.

### 6. Umfangreiche Diagnostik

- Anpassungsgütetests (KS, AD, CVM, χ²)
- Residuenanalyse
- Schweifverhaltensbewertung
- Ausreißererkennung
- Einflussanalyse
- Kreuzvalidierungsscores

### 7. Bootstrap-Konfidenzintervalle

Parametrischer und nichtparametrischer Bootstrap mit paralleler Verarbeitung.

### 8. Interaktive Visualisierungen

Statische Plots (matplotlib/seaborn) und interaktive Plots (plotly).

---

## 🔬 Erweiterte Beispiele

### Beispiel 1: Zuverlässigkeitstechnik

```python
import numpy as np
from distfit_pro import set_language, DistributionFitter

set_language('de')

# Ausfallzeitdaten (rechtszensiert)
ausfallzeiten = np.array([120, 145, 167, 189, 201, 234, 267, 289, 312, 345])
zensiert = np.array([0, 0, 0, 1, 0, 1, 0, 0, 1, 0])  # 1=zensiert

fitter = DistributionFitter(
    data=ausfallzeiten,
    censoring=zensiert,
    censoring_type='right'
)

results = fitter.fit(
    distributions=['weibull', 'lognormal', 'gamma', 'exponential'],
    method='mle'
)

# Zuverlässigkeitsfunktionen
zuverlaessigkeit = results.best_model.reliability(t=200)
ausfallrate = results.best_model.hazard_rate(t=200)
mttf = results.best_model.mean_time_to_failure()

print(f"Zuverlässigkeit bei t=200h: {zuverlaessigkeit:.3f}")
print(f"Ausfallrate bei t=200h: {ausfallrate:.4f}")
print(f"MTTF: {mttf:.1f}h")
```

### Beispiel 2: Finanzrisiko (VaR-Schätzung)

```python
set_language('de')

# Aktienrenditen
renditen = lade_aktienrenditen('AAPL')

fitter = DistributionFitter(renditen)
results = fitter.fit(
    distributions=['normal', 'student_t', 'cauchy', 'gev'],
    method='mle'
)

# Value at Risk (99% Konfidenz)
var_99 = results.best_model.ppf(0.01)  # 1. Perzentil
cvar_99 = results.best_model.conditional_var(0.01)  # Expected Shortfall

print(f"VaR(99%): {var_99:.2%}")
print(f"CVaR(99%): {cvar_99:.2%}")
```

---

## 🧪 Entwicklungsstatus

**Aktuelle Version:** v0.1.0-alpha

### ✅ Implementiert (v0.1.0):
- Kern-Verteilungsklassen (30 Verteilungen)
- Modellauswahl (AIC, BIC, LOO-CV)
- Grundlegende Anpassungsfunktionalität
- Selbsterklärende Ausgaben
- **Mehrsprachige Unterstützung** (EN/FA/DE)
- Visualisierungsmodul (matplotlib + plotly)

### 🔨 In Arbeit:
- Erweiterte Diagnostik
- Bootstrap-CI-Implementierung
- Unterstützung zensierter Daten

### 📋 Geplant:
- Bayessche Inferenz (PyMC-Integration)
- Mischmodelle
- Interaktive Dashboards
- Umfassende Testsuite
- Vollständige Dokumentationsseite

---

## 🤝 Mitwirken

Beiträge sind willkommen! Bitte beachten Sie [CONTRIBUTING.md](CONTRIBUTING.md).

---

## 📄 Lizenz

MIT-Lizenz - siehe [LICENSE](LICENSE) Datei.

---

## 📞 Kontakt

**Ali Sadeghi Aghili**  
- Website: [zil.ink/thedatascientist](https://zil.ink/thedatascientist)  
- LinkTree: [linktr.ee/aliaghili](https://linktr.ee/aliaghili)
- GitHub: [@alisadeghiaghili](https://github.com/alisadeghiaghili)

---

## 🙏 Danksagungen

Inspiriert von:
- R's `fitdistrplus` Paket
- MathWave's EasyFit Software
- SciPy's statistische Verteilungen

Gebaut mit modernen Verbesserungen in statistischer Methodik und Software-Engineering-Praktiken.

---

**Mit ❤️ und ☕ von Ali Sadeghi Aghili erstellt**