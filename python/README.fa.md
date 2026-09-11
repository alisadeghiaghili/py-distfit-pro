# veridist

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python](https://img.shields.io/pypi/pyversions/veridist.svg)](https://pypi.org/project/veridist/)
[![CI](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage ≥95%](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=coverage%20%E2%89%A595%25)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Mutation gate](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml)
[![Release evidence](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-7b1fa2.svg)](LICENSE)

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

<div lang="fa" dir="rtl">

## مدل طول‌عمر مشخص را با قرارداد روشن برازش دهید

نسخهٔ 1.0.0 یک انتشار عمومی با شواهد قابل‌بازبینی است. `veridist` CSV طول‌عمر
سخت‌گیرانه را به برازش نوع‌دار و گزارش اجرا تبدیل می‌کند تا مدل، ورودی و مرزهای
عملیاتی روشن بمانند.

## نصب و نخستین موفقیت

نسخهٔ منتشرشده را نصب کنید:

</div>

```console
python -m pip install veridist
```

<div lang="fa" dir="rtl">

برای source checkout:

</div>

```console
git clone https://github.com/alisadeghiaghili/veridist.git
cd veridist/python
python -m pip install .
```

<div lang="fa" dir="rtl">

یا wheel یک اجرای تأییدشده را نصب کنید:

</div>

```console
python -m pip install /path/to/veridist-1.0.0-py3-none-any.whl
```

<div lang="fa" dir="rtl">

پس از نصب، این برازش کامل CSV را اجرا کنید:

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
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
```

<div lang="fa" dir="rtl">

## مسیر درست را انتخاب کنید

| نیاز | ابزار |
| --- | --- |
| برازش CSV طول‌عمر سخت‌گیرانه | `fit_exponential_csv` و [آموزش CSV](docs/source/exponential-right-censoring.md) |
| محاسبهٔ توزیع اسکالر اعلام‌شده | `FAMILY_REGISTRY` و `evaluate_log_density`؛ [راهنمای خانواده‌ها](docs/source/families-log-density-likelihood.md) |
| کاهش حالت‌دقیق روی chunkهای متعلق به فراخواننده | `reduce_log_likelihood_chunks`؛ [API جریان](docs/source/api.md#generic-stream-source-api) |
| طراحی checkpoint و resume محلی | `SQLiteCheckpointStore` و [قرارداد CSV](tests/contract/test_v1_checkpointed_csv.py) |

## قابلیت و شواهد

سطح برازش، سلول‌های MLE نمایی، Weibull-minimum و Lognormal با مکان ثابت برای
طول عمرهای دقیق و راست‌سانسورشدهٔ مستقل دارد. راه‌حل متناهی برآورد نقطه‌ای و
شرایط آماری یا عملیاتی نامعتبر شکست‌های نوع‌دار برمی‌گردانند. استنباط به همین
سلول اعلام‌شده محدود است: سلول نماییِ بدون سانسور KS، AD و CvM با refit Monte
Carlo، AIC/BIC و انتخاب adequacy-gated با generator فراخواننده دارد.

مسیر CSV عمومی آن سخت‌گیرانه است: UTF-8 و دقیقاً `time,event_observed`، توکن
رخداد `1` و توکن راست‌سانسوری `0`. یک گذر از iterator انجام می‌دهد و CSV عمومی
نیست. نتیجه را همراه با گزارش اجرا و فرض‌های مدل بخوانید.

CI نسخه‌های پشتیبانی‌شدهٔ Python، حداقل ۹۵٪ پوشش line و branch، کیفیت، نصب بسته
و مستندات را کنترل می‌کند. بج Coverage ≥95% گذر/شکست همین قرارداد الزام‌شده را
روی `main` نشان می‌دهد؛ در این صفحه عدد ثابت coverage وجود ندارد.

## مقیاس و مرزهای تولید

شواهد نگه‌داری‌شده ماتریس اندازه‌گیری‌شدهٔ 10k/100k/1m ردیف و بودجه‌های
32KiB/64KiB/128KiB را برای مسیرهای اعلام‌شده پوشش می‌دهند. این شواهد، پشتیبانی
کلی از big-data، throughput، RSS قابل‌حمل، dataframe، Parquet، Arrow،
پایگاه‌داده، اجرای توزیع‌شده، سانسور گسترده، عملیات برداری یا انتخاب عمومی مدل
را ثابت نمی‌کنند. `SQLiteCheckpointStore` وضعیت محلی پایدار است، نه سرویس
توزیع‌شده.

پیش از استفادهٔ عملیاتی [KNOWN_LIMITS.fa.md](KNOWN_LIMITS.fa.md) و
[دفتر شواهد](../docs/v1-readiness.md) را بخوانید.

خانواده‌های `normal`، `gamma`، `weibull_min`، `lognormal` و `gumbel_right` در
registry اسکالر هستند. چگالیِ لگاریتمی binary64 موفق و `reduce_log_likelihood_chunks`
مجموع صحیحِ دقیق را برای سطح اعلام‌شده نگه می‌دارند؛ این‌ها API عمومی برازش یا
آرایه نیستند.

## مستندات، مشارکت و پشتیبانی

[مرجع API](docs/source/api.md)، [راهنمای خانواده‌ها](docs/source/families-log-density-likelihood.md)
و [toolchain مستندات](docs/README.md) مسیرهای اصلی‌اند. برای تغییرات از
[راهنمای مشارکت](../CONTRIBUTING.md) و [قراردادهای مهندسی](../docs/conventions.md)
شروع کنید. خطاهای بازتولیدپذیر را در
[GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) و
آسیب‌پذیری‌ها را مطابق [SECURITY.md](../SECURITY.md) گزارش کنید.

BUSL-1.1 با مجوز استفادهٔ اضافی Apache-2.0 برای استفادهٔ شخصی و غیرتجاری؛
[LICENSE](LICENSE) را ببینید.

</div>
