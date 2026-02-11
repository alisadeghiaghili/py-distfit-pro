# DistFit Pro 🎯

<div dir="rtl">

کتابخانه پیشرفته و چندزبانه برای فیت توزیع‌های آماری به داده

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ ویژگی‌ها

### 🌍 **پشتیبانی چندزبانه کامل**
- فارسی (fa) 🇮🇷
- انگلیسی (en) 🇬🇧
- آلمانی (de) 🇩🇪

### 📊 **30 توزیع آماری**
توزیع‌های پیوسته:
- **Normal** (نرمال)
- **Lognormal** (لوگ نرمال)
- **Weibull** (وایبول)
- **Gamma** (گاما)
- **Exponential** (نمایی)
- **Beta** (بتا)
- **Pareto** (پارتو)
- **Student-t**
- **Chi-Square** (خی‌دو)
- **F-distribution**
- و 20 توزیع دیگر...

توزیع‌های گسسته:
- **Poisson** (پواسون)
- **Binomial** (دوجمله‌ای)
- **Geometric** (هندسی)
- **Negative Binomial**
- **Hypergeometric**

### 📈 **نمودارهای حرفه‌ای**
- **مقایسه‌ای**: PDF, CDF, P-P, Q-Q
- **تشخیصی**: Residuals, Tail Behavior, Influence
- **تعاملی**: Plotly Dashboard
- **چندزبانه**: تمام برچسب‌ها به زبان انتخابی شما
- **RTL Fix**: متن فارسی به درستی نمایش داده می‌شود

### 🧠 **هوشمند و Self-Explanatory**
- توضیحات مفهومی برای هر توزیع
- پیشنهاد خودکار توزیع‌های مناسب
- تشخیص مشکلات fit و ارائه راهکار
- محاسبه خودکار فاصله اطمینان (CI)

---

## 🚀 نصب

```bash
pip install distfit-pro
```

یا برای نسخه توسعه‌دهندگان:

```bash
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro
pip install -e .
```

### 📦 وابستگی‌ها

وابستگی‌های اصلی:
```bash
pip install numpy scipy pandas matplotlib joblib tqdm
```

برای پشتیبانی کامل فارسی (RTL):
```bash
pip install arabic-reshaper python-bidi
```

برای نمودارهای تعاملی:
```bash
pip install plotly
```

---

## 📚 مثال‌های کاربردی

### مثال 1️⃣: فیت ساده یک توزیع

```python
import numpy as np
from distfit_pro import get_distribution, set_language

# تنظیم زبان به فارسی
set_language('fa')

# تولید داده‌ی تصادفی از توزیع لوگ نرمال
data = np.random.lognormal(mean=2, sigma=0.5, size=1000)

# فیت توزیع لوگ نرمال
dist = get_distribution('lognormal')
dist.fit(data, method='mle')

# نمایش آمارها (به فارسی!)
print(dist.summary())

# توضیحات مفهومی (به فارسی!)
print(dist.explain())
```

**خروجی:**
```
══════════════════════════════════════
📊 توزیع Lognormal
══════════════════════════════════════

پارامترهای برآورد شده:
   • میانگین (μ): 2.0134
   • انحراف معیار (σ): 0.4987

آمارهای مکانی:
   • میانگین (μ): 8.9234
   • میانه: 7.5012
   • مد (نما): 5.2341
...
```

---

### مثال 2️⃣: فیت خودکار چند توزیع

```python
import numpy as np
from distfit_pro import DistributionFitter, set_language

# تنظیم زبان
set_language('fa')

# تولید داده
data = np.random.gamma(shape=2, scale=2, size=1000)

# فیت خودکار
fitter = DistributionFitter(data)
results = fitter.fit()  # خودش بهترین توزیع‌ها را پیشنهاد می‌دهد!

# نمایش نتایج (به فارسی)
print(results.summary())

# نمودارهای مقایسه‌ای (برچسب‌ها به فارسی!)
results.plot(kind='comparison')

# نمودارهای تشخیصی
results.plot(kind='diagnostics')

# داشبورد تعاملی
results.plot(kind='interactive')
```

**خروجی:**
```
🚀 شروع فیت 5 توزیع...
   • روش تخمین: MLE
   • معیار انتخاب: AIC
   • تعداد کور: همه

═════════════════════════════════════
🔍 نتایج فیت توزیع‌های آماری
═════════════════════════════════════

📊 خلاصه داده:
   • تعداد: 1000
   • میانگین: 3.9821 (فاصله اطمینان ۹۵٪: [3.7234, 4.2408])
   • انحراف معیار: 2.8123
   • چولگی: 1.3421 → راست‌چوله (دنباله سمت راست بلند)

🏆 رتبه‌بندی مدل‌ها:

رتبه   توزیع         AIC        Δ          وضعیت
──────────────────────────────────────────
1      Gamma          3245.21    0.00       ✅
2      Lognormal      3246.89    1.68       ✅
3      Weibull        3251.34    6.13       ⚠️
4      Exponential    3278.92    33.71      ❌
5      Normal         3295.47    50.26      ❌

✨ مدل برتر: Gamma
...
```

---

### مثال 3️⃣: مقایسه چند توزیع

```python
import numpy as np
from distfit_pro import DistributionFitter, set_language

# تنظیم زبان
set_language('fa')

# تولید داده
data = np.random.weibull(a=1.5, size=1000) * 10

# فیت با لیست خاص توزیع‌ها
fitter = DistributionFitter(data)
results = fitter.fit(
    distributions=['weibull', 'lognormal', 'gamma', 'exponential'],
    method='mle',
    criterion='bic',
    n_jobs=-1  # استفاده از تمام کورها
)

# نمایش نتایج
print(results.summary())

# دسترسی به بهترین مدل
best = results.best_model
print(f"\nبهترین توزیع: {best.info.display_name}")
print(f"پارامترها: {best.params}")
```

---

### مثال 4️⃣: تغییر زبان به صورت پویا

```python
import numpy as np
from distfit_pro import get_distribution, set_language

data = np.random.normal(loc=50, scale=10, size=500)
dist = get_distribution('normal')
dist.fit(data)

# خروجی به فارسی
set_language('fa')
print("═" * 50)
print("خروجی فارسی")
print("═" * 50)
print(dist.summary())

# خروجی به انگلیسی
set_language('en')
print("\n" + "=" * 50)
print("ENGLISH OUTPUT")
print("=" * 50)
print(dist.summary())

# خروجی به آلمانی
set_language('de')
print("\n" + "=" * 50)
print("DEUTSCHE AUSGABE")
print("=" * 50)
print(dist.summary())
```

---

## 📝 مستندات کامل

برای مستندات دقیق‌تر و مثال‌های بیشتر:

- [📖 مستندات کامل](https://github.com/alisadeghiaghili/py-distfit-pro/wiki)
- [📚 API Reference](https://github.com/alisadeghiaghili/py-distfit-pro/wiki/API)
- [🎯 مثال‌های کاربردی](https://github.com/alisadeghiaghili/py-distfit-pro/tree/main/examples)

---

## ⚙️ توزیع‌های پشتیبانی شده

### توزیع‌های پیوسته (25 تا)

| توزیع | کاربرد | پارامترها |
|---------|---------|-------------|
| Normal | داده‌های متقارن | loc, scale |
| Lognormal | داده‌های مثبت و چوله | s, loc, scale |
| Weibull | Reliability, Lifetime | c, loc, scale |
| Gamma | داده‌های مثبت و چوله | a, loc, scale |
| Exponential | زمان بین رویدادها | loc, scale |
| Beta | درصدها، نرخ‌ها | a, b, loc, scale |
| Pareto | درآمد، ثروت | b, loc, scale |
| Student-t | داده‌های دنباله سنگین | df, loc, scale |
| Uniform | توزیع یکنواخت | loc, scale |
| Chi-Square | آزمون‌های آماری | df, loc, scale |
| ... | ... | ... |

### توزیع‌های گسسته (5 تا)

| توزیع | کاربرد | پارامترها |
|---------|---------|-------------|
| Poisson | تعداد رویدادها | λ (mu) |
| Binomial | موفقیت/شکست | n, p |
| Geometric | زمان تا موفقیت اول | p |
| Negative Binomial | تعداد شکست‌ها | n, p |
| Hypergeometric | نمونه‌گیری بدون جایگذاری | M, n, N |

---

## 🛠️ قابلیت‌های پیشرفته

### 1️⃣ فاصله اطمینان (Confidence Intervals)

```python
# Bootstrap CI برای پارامترها
ci = dist.bootstrap_ci(n_bootstrap=1000, alpha=0.05)
print(f"فاصله اطمینان ۹۵٪: {ci}")
```

### 2️⃣ تشخیص مشکلات Fit

```python
# آزمون Goodness-of-Fit
ks_stat, p_value = dist.goodness_of_fit(data, test='ks')
print(f"KS Statistic: {ks_stat:.4f}, p-value: {p_value:.4f}")

if p_value < 0.05:
    print("⚠️ مدل مناسب نیست!")
```

### 3️⃣ مقایسه با معیارهای مختلف

```python
# مقایسه با AIC, BIC, LOO-CV
results_aic = fitter.fit(criterion='aic')
results_bic = fitter.fit(criterion='bic')
results_loo = fitter.fit(criterion='loo_cv')
```

---

## 🤝 مشارکت

مشارکت‌ها استقبال می‌شوند! برای مشارکت:

1. 🐛 **گزارش باگ**: [Issues](https://github.com/alisadeghiaghili/py-distfit-pro/issues)
2. 💡 **پیشنهاد قابلیت**: [Discussions](https://github.com/alisadeghiaghili/py-distfit-pro/discussions)
3. 🚀 **Pull Request**: Fork کنید و PR بفرستید!

---

## 📝 لایسنس

MIT License - برای جزئیات به [LICENSE](LICENSE) مراجعه کنید.

---

## 👨‍💻 سازنده

**Ali Sadeghi Aghili**
- 🔗 GitHub: [@alisadeghiaghili](https://github.com/alisadeghiaghili)
- 🌐 Website: [zil.ink/thedatascientist](https://zil.ink/thedatascientist)
- 🔗 LinkTree: [linktr.ee/aliaghili](https://linktr.ee/aliaghili)

---

## ⭐ پشتیبانی از پروژه

اگر این پروژه برای شما مفید بود، لطفاً یک ⭐ بهش بدین!

---

## 📚 منابع و آموزش

- [🎯 Tutorial: Beginner's Guide](https://github.com/alisadeghiaghili/py-distfit-pro/wiki/Tutorial)
- [📈 Examples Gallery](https://github.com/alisadeghiaghili/py-distfit-pro/tree/main/examples)
- [🤔 FAQ](https://github.com/alisadeghiaghili/py-distfit-pro/wiki/FAQ)
- [📝 Changelog](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/CHANGELOG.md)

---

<div align="center">

**ساخته شده با ❤️ در ایران**

DistFit Pro © 2026

</div>

</div>
