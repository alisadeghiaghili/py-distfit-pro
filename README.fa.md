# Veridist

**برازش دادهٔ طول‌عمر با نتیجه‌ای قابل بررسی و بازتولید.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![Coverage ≥95%](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=coverage%20%E2%89%A595%25)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

<div lang="fa" dir="rtl">

## از زمان خرابی تا یک نتیجهٔ قابل بررسی

Veridist به مهندسان قابلیت اطمینان و پژوهشگران کمک می‌کند مدل طول‌عمر برازش کنند، داده‌های پایان‌یافته پیش از خرابی را تحلیل کنند و اطلاعات اجرای محاسبه را نگه دارند.

### سه کاربرد اصلی

| هدف شما | نتیجه |
| --- | --- |
| تحلیل زمان خرابی | برازش نمایی، Weibull-minimum و Lognormal با مکان ثابت |
| بررسی نتیجه | فرض‌های مشخص، خطاهای نوع‌دار و اطلاعات اجرا |
| بازیابی اجرای متوقف‌شده | checkpoint محلی برای محاسبهٔ نمایی سازگار |

</div>

```console
python -m pip install veridist
```

<div lang="fa" dir="rtl">

[اجرای نمونهٔ نخست](#نخستین-برازش-را-اجرا-کنید) · [انتخاب کار بعدی](#کار-بعدی-را-انتخاب-کنید) · [راهنمای بسته](python/README.fa.md)

## نخستین برازش را اجرا کنید

این نمونه CSV خودش را می‌سازد و بلافاصله پس از نصب اجرا می‌شود. `1` یعنی رخداد
مشاهده شده است. `0` یعنی تا پایان مشاهده رخداد اتفاق نیفتاده است؛ به این وضعیت
راست‌سانسوری می‌گویند.

</div>

<details>
<summary>نمایش مثال کامل قابل اجرا</summary>

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32_768, 32_768),
    ).fit

assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

</details>

```text
rate=0.5; events=1; censored=1
```

<div lang="fa" dir="rtl">

`rate` معکوس واحد زمانی CSV شماست. در این مثال واحدی برای زمان مشخص نشده، پس
نرخ `0.5` به‌ازای هر واحد زمان ورودی است. برازش موفق فقط می‌گوید محاسبهٔ اعلام‌شده
تمام شده است؛ مناسب‌بودن مدل نمایی را به‌تنهایی ثابت نمی‌کند.

## کار بعدی را انتخاب کنید

| کار شما | از این‌جا شروع کنید |
| --- | --- |
| برآورد مدل قابلیت‌اطمینان از CSV کوچک طول‌عمر | [برازش CSV سخت‌گیرانه](python/docs/source/exponential-right-censoring.md) |
| کار با مشاهده‌ای که هنوز خراب نشده است | [راست‌سانسوری و فرض‌های آن](python/docs/source/exponential-right-censoring.md#model-and-estimate) |
| بررسی نتیجه یا مشکل ورودی | [API CSV و حالت‌های خطا](python/docs/source/api.md) |
| محاسبهٔ توزیع پشتیبانی‌شده یا پردازش chunkهای خودتان | [خانواده‌های اسکالر و likelihood جریانی](python/docs/source/families-log-density-likelihood.md) |
| ادامهٔ اجرای محلیِ سازگار پس از وقفه | [راهنمای checkpoint و resume](python/examples/checkpoint_resume.py) |

Veridist اکنون برازش‌های نمایی، Weibull-minimum و Lognormal با مکان ثابت را برای
دادهٔ طول‌عمر دقیق و راست‌سانسورشدهٔ مستقل پشتیبانی می‌کند. نقطهٔ ورود CSV از
مسیر نمایی استفاده می‌کند؛ عملیات اسکالر توزیع‌ها سطحی جدا از برازش هستند.

## اگر اجرا متوقف شد

برای یک ماشین و فایل‌سیستم محلی، `SQLiteCheckpointStore` می‌تواند بخش ثبت‌شدهٔ
اجرا را نگه دارد و اجرای سازگار را ادامه دهد. revision منبع را ثابت نگه دارید و
همان store محلی را دوباره باز کنید. [نمونهٔ checkpoint قابل‌اجرا](python/examples/checkpoint_resume.py)
راه‌اندازی و گذر دوم را نشان می‌دهد.

این قابلیت سرویس توزیع‌شده نیست. برای تصمیم‌گیری، [محدودیت‌های شناخته‌شده](python/KNOWN_LIMITS.fa.md)
را بخوانید.

## چرا می‌توان به این صفحه اعتماد کرد

| شاهد | معنای آن |
| --- | --- |
| PyPI و Python | بستهٔ منتشرشده و سازگاری اعلام‌شدهٔ Python |
| CI | آزمون‌ها، نصب بسته، مستندات و بررسی مرورگر روی `main` اجرا می‌شوند |
| Coverage ≥95% | CI حداقل ۹۵٪ پوشش line و branch را الزام می‌کند؛ badge وضعیت گذر/شکست همین گیت است، نه یک درصد ساختگی |
| Mutation gate | هستهٔ آماری مهم در برابر mutation آزموده می‌شود |
| Release evidence | نامزد انتشار دوباره build و قابلیت بازتولید artifactها بررسی می‌شود |

شواهد انتشار، مسیرهای اعلام‌شده را در 10k، 100k و 1m ردیف و شرایط مشخص پوشش
می‌دهد. این شواهد وعدهٔ عمومی دربارهٔ throughput، حافظه، اجرای توزیع‌شده، Parquet،
dataframe یا «بهترین» مدل نیست.

## پیش از استفادهٔ عملیاتی

[محدودیت‌های شناخته‌شده](python/KNOWN_LIMITS.fa.md)، [مرجع API](python/docs/source/api.md)
و [شواهد انتشار](docs/v1-readiness.md) را بخوانید. ورودی UTF-8 سخت‌گیرانه با
`time,event_observed` است، checkpointها SQLite محلی‌اند و سطح استنباط از registry
کامل توزیع‌ها محدودتر است.

محتوای `distfit_pro` برای زمینهٔ تاریخی در مخزن نگه‌داری می‌شود و وعدهٔ سازگاری
با Veridist نیست؛ وضعیت آن در [دفتر مهاجرت legacy](docs/migration/README.md) ثبت شده است.

## ارجاع به Veridist

نسخه‌ای را ارجاع دهید که نتیجه را با آن تولید کرده‌اید. الگوهای آمادهٔ [IEEE، BibTeX و APA](docs/citing-veridist.md) همراه با [CITATION.cff](CITATION.cff) نگه‌داری می‌شوند.

## کمک و مطالعهٔ بیشتر

برای نصب و مثال‌ها [راهنمای بسته](python/README.fa.md)، برای سطح آماری
[راهنمای خانواده‌ها](python/docs/source/families-log-density-likelihood.md) و برای
تصمیم پذیرش [محدودیت‌ها](python/KNOWN_LIMITS.fa.md) را ببینید. خطای بازتولیدپذیر
را در [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) و مشکل
امنیتی را در [SECURITY.md](SECURITY.md) گزارش کنید. مشارکت‌کنندگان باید
[CONTRIBUTING.md](CONTRIBUTING.md) و [قراردادهای مهندسی](docs/conventions.md) را بخوانند.

بسته با BUSL-1.1 و مجوز استفادهٔ اضافی Apache-2.0 در [LICENSE](LICENSE) منتشر
می‌شود. تاریخچهٔ انتشار در [python/CHANGELOG.md](python/CHANGELOG.md) است.

</div>
