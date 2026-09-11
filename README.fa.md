# veridist

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python](https://img.shields.io/pypi/pyversions/veridist.svg)](https://pypi.org/project/veridist/)
[![CI](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=coverage%20%E2%89%A595%25)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Mutation gate](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml)
[![Release evidence](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-7b1fa2.svg)](LICENSE)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

<div lang="fa" dir="rtl">

## برازش توزیع با زنجیرهٔ شواهد قابل‌مشاهده

`veridist` 1.0.0 یک بستهٔ evidence-first برای برازش توزیع است؛ برای تیم‌هایی
که به نتیجهٔ طول‌عمر قابل‌بررسی، بازتولیدپذیر و دارای مرز روشن نیاز دارند. ورودی
CSV سخت‌گیرانه را به نتیجه و گزارش اجرای نوع‌دار تبدیل می‌کند، بی‌آن‌که معنای
داده را حدس بزند یا یک پاسخ کلیِ «بهترین توزیع» عرضه کند.

## چرا Veridist

| نیاز | آنچه Veridist می‌دهد |
| --- | --- |
| نقطهٔ شروع روشن | سلول‌های MLE نمایی، Weibull-minimum و Lognormal با مکان ثابت برای طول‌عمرهای دقیق و راست‌سانسورشدهٔ مستقل |
| ورودی قابل اتکا | قرارداد UTF-8 سخت‌گیرانهٔ `time,event_observed`؛ `1` رخداد و `0` راست‌سانسوری مستقل است |
| نتیجهٔ قابل ممیزی | برآورد و شکست‌های نوع‌دار، واقعیت اجرای یک‌گذر و منشأ دادهٔ پالایش‌شده |
| شواهد پیش از پذیرش | گیت‌های CI و پوشش، mutation، انتشار بازتولیدپذیر و شواهد release |

این بسته برای تیم‌های قابلیت‌اطمینان، مهندسی و علم داده با مدل طول‌عمر مشخص
ساخته شده است، نه برای جست‌وجوی اکتشافی و عمومی میان همهٔ توزیع‌ها.

## شروع در ۶۰ ثانیه

بستهٔ منتشرشده را نصب کنید:

</div>

```console
python -m pip install veridist
```

<div lang="fa" dir="rtl">

نمونهٔ اجراییِ کامل، یک CSV کوچک با یک مشاهدهٔ راست‌سانسورشده می‌سازد و فرض‌های
مدل نتیجه را کنترل می‌کند.

</div>

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import CsvLifetimeLimits, CsvLifetimeSchema, PublicSourceId, fit_exponential_csv
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    result = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32_768, 32_768),
    )

fit = result.fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

<div lang="fa" dir="rtl">

برای نصب از clone، به `veridist/python` بروید و `python -m pip install .` را
اجرا کنید. راهنمای بسته با همین نمونه به [English](python/README.md)،
[فارسی](python/README.fa.md) و [Deutsch](python/README.de.md) در دسترس است.

## انتخاب مسیر کار

| هدف | از این‌جا شروع کنید |
| --- | --- |
| برازش CSV طول‌عمر سخت‌گیرانه | [نخستین برازش و قرارداد CSV](python/docs/source/exponential-right-censoring.md) |
| دادهٔ راست‌سانسورشدهٔ مستقل | [مدل و حالت‌های شکست](python/docs/source/exponential-right-censoring.md#model-and-estimate) |
| بررسی نتیجه یا شکست نوع‌دار | [API نمایی CSV](python/docs/source/api.md) |
| محاسبهٔ توزیع اعلام‌شده یا جریان chunkها | [خانواده‌ها و likelihood جریانی](python/docs/source/families-log-density-likelihood.md) |
| برنامه‌ریزی checkpoint و ادامهٔ اجرا | [آزمون‌های قرارداد checkpoint/resume](python/tests/contract/test_v1_checkpointed_csv.py) |

checkpoint فقط وضعیت SQLite محلی است. اجرای سازگار می‌تواند پیشوند commit‌شده را
ادامه دهد؛ این سرویس checkpoint توزیع‌شده نیست.

## اعتبارسنجی نتیجه

نتیجه را همراه با فرض‌های مدل و گزارش اجرا بخوانید. برآورد نقطه‌ای متناهی، فاصلهٔ
اطمینان یا نتیجهٔ goodness-of-fit نیست. سلول نماییِ بدون سانسور آزمون‌های refit
Monte Carlo از نوع KS، AD و CvM، AIC/BIC و انتخاب مدل adequacy-gated دارد؛ این
استنباط به همان سلول اعلام‌شده و generator متعلق به فراخواننده محدود است.

CI اصلی همهٔ نسخه‌های پشتیبانی‌شدهٔ Python را اجرا و دست‌کم ۹۵٪ پوشش line و
branch را الزام می‌کند. بج Coverage ≥95% وضعیت گذر/شکست همین قرارداد الزام‌شده
را روی `main` نشان می‌دهد و درصدی ساختگی نشان نمی‌دهد.

## شواهد، مقیاس و مرزهای تولید

اعتبارسنجی انتشار SHA نامزد را bind می‌کند، بسته را دو بار می‌سازد و تطابق
بایت‌به‌بایت می‌خواهد. مرز release شامل ماتریس ۲۷سلولیِ complete، retry-resume
و cancellation در 10k، 100k و 1m ردیف روی Linux، macOS و Windows است.

این شواهد ادعای عمومی دربارهٔ throughput، RSS، اجرای توزیع‌شده، Parquet، Arrow،
dataframe، پایگاه‌داده، سانسور گسترده، عملیات برداری یا انتخاب بهترین توزیع ایجاد
نمی‌کنند. پیش از استفادهٔ عملیاتی، [محدودیت‌های شناخته‌شده](python/KNOWN_LIMITS.fa.md)
و [دفتر شواهد](docs/v1-readiness.md) را بخوانید.

محتوای `distfit_pro` هنوز برای زمینهٔ تاریخی و ممیزی‌شده در مخزن هست. این محتوا
وعدهٔ سازگاری با `veridist` نیست؛ وضعیت آن در
[دفتر مهاجرت legacy](docs/migration/README.md) ثبت شده است.

## نقشهٔ مستندات

| برای چه کاری | مرجع |
| --- | --- |
| شروع با بسته | [راهنمای بسته](python/README.fa.md) و [آموزش CSV](python/docs/source/exponential-right-censoring.md) |
| یکپارچه‌سازی API | [مرجع API](python/docs/source/api.md) و [محدودیت‌ها](python/KNOWN_LIMITS.fa.md) |
| بررسی سطح آماری | [خانواده‌ها و likelihood](python/docs/source/families-log-density-likelihood.md) |
| بررسی کیفیت و release | [برنامهٔ آزمون](docs/v1-test-plan.md)، [readiness](docs/v1-readiness.md) و [ADRها](docs/adr/README.md) |
| مشارکت | [راهنمای مشارکت](CONTRIBUTING.md)، [قراردادهای مهندسی](docs/conventions.md) و [toolchain مستندات](python/docs/README.md) |

## پشتیبانی، امنیت و مجوز

برای خطای بازتولیدپذیر یا پیشنهاد قابلیت از
[GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) استفاده کنید.
آسیب‌پذیری را مطابق [SECURITY.md](SECURITY.md) گزارش کنید. تاریخچهٔ انتشار در
[python/CHANGELOG.md](python/CHANGELOG.md) است. مخزن و بستهٔ تو در تو از
BUSL-1.1 با مجوز استفادهٔ اضافی Apache-2.0 مطابق [LICENSE](LICENSE) استفاده
می‌کنند.

</div>
