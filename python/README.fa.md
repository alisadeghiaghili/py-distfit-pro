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
نمی‌شود. مسیر CSV عمومی آن سخت‌گیرانه است: CSV با UTF-8 و دقیقاً سرستون‌های
`time,event_observed`، توکن رخداد `1` و توکن راست‌سانسوری `0`. این مسیر یک
گذر از iterator با بودجهٔ منطقی payload نگه‌داشته‌شده اجرا و نتیجهٔ اجرایی
بسته و نوع‌دار برمی‌گرداند. شواهد نگه‌داری‌شده فقط payload داخلی کران‌دار را
برای ماتریس اندازه‌گیری‌شدهٔ 10k/100k/1m ردیف و بودجه‌های
32KiB/64KiB/128KiB نشان می‌دهند؛ از آن‌ها پشتیبانی عمومی از دادهٔ بزرگ یا
توان عملیاتی بالا نتیجه نمی‌شود. این ادعای سقف RSS قابل‌حمل، لغو، retry،
checkpoint یا برون‌حافظه‌ای عمومی هم نیست.

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
from pathlib import Path
from tempfile import TemporaryDirectory
from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path, schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
```

<div lang="fa" dir="rtl">

برای جزئیات، [زنجیرهٔ مستندسازی](docs/README.md) و
[دفتر شواهد](../docs/v1-readiness.md) را ببینید؛ قابلیت‌های پیاده‌شده و
محدودیت‌ها در آن‌ها جدا شده‌اند.

## مجوز

MIT؛ متن کامل در [LICENSE](LICENSE) است.

</div>
