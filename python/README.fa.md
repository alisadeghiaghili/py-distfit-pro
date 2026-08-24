# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

<div lang="fa" dir="rtl">

## وضعیت

`veridist` با نسخهٔ 0.0.0.dev0 یک هستهٔ قراردادی پیش‌آلفا و در حال توسعه است.
این نسخه قراردادهای تحویل کران‌دار، بازپخش‌پذیری، بودجهٔ گذر، تلاش مجدد
تراکنشی، سازگاری checkpoint، شکست‌های نوع‌دار، پیامدهای اجرا و منشأ دادهٔ
پالایش‌شده از اطلاعات حساس را تعریف و آزمایش می‌کند.

این نسخه یک برآوردگر آزمایشی MLE نمایی فقط برای پارامتر نرخ و برای طول عمرهای
دقیق و راست‌سانسورشدهٔ مستقل دارد. هرگاه MLE متناهی وجود داشته باشد، برآورد
نقطه‌ای برمی‌گرداند و در غیر این صورت شکست‌های نوع‌دار می‌دهد. استنباط ارائه
نمی‌شود. حالت الگوریتمی کاهش‌دهنده ثابت و از مرتبهٔ O(1) است، اما بسته آداپتور
دادهٔ عملیاتی عرضه نمی‌کند و اجرای برون‌حافظه‌ای عملیاتی را ادعا نمی‌کند.
همچنین ادعایی دربارهٔ دوام پایدار checkpoint ندارد. منبع‌ها و مخزن‌های checkpoint حافظه‌ای فقط
fixture قراردادی هستند، نه اجزای عملیاتی ذخیره‌سازی یا هماهنگ‌سازی.

</div>

<div lang="fa" dir="rtl">

## نصب نسخهٔ ارزیابی

پس از clone کردن مخزن، پروژهٔ تو‌در‌توی Python را نصب کنید:

</div>

```console
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro/python
python -m pip install .
```

<div lang="fa" dir="rtl">

یا wheel مشخصی را که خودتان ساخته‌اید یا از یک اجرای تأییدشده گرفته‌اید نصب
کنید:

</div>

```console
python -m pip install /path/to/veridist-0.0.0.dev0-py3-none-any.whl
```

<div lang="fa" dir="rtl">

این پروژه کاربران را به نصب نام یک بستهٔ منتشرنشده از public index هدایت
نمی‌کند.

## آزمودن عمودی آزمایشی

</div>

```python
from veridist.domain import ExactLifetime, RightCensoredLifetime
from veridist.families import ExponentialFitSuccess, fit_exponential
from veridist.reporting import ReportLocale, render_exponential_report

fit = fit_exponential([ExactLifetime(1.0), RightCensoredLifetime(1.0)])
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"

report = render_exponential_report(fit, ReportLocale.FA)
assert 'lang="fa" dir="rtl"' in report
```

<div lang="fa" dir="rtl">

برای جزئیات، [زنجیرهٔ مستندسازی](docs/README.md) و
[دفتر شواهد](../docs/v1-readiness.md) را ببینید؛ قابلیت‌های پیاده‌شده و
محدودیت‌ها در آن‌ها جدا شده‌اند.

## مجوز

MIT؛ متن کامل در [LICENSE](LICENSE) است.

</div>
