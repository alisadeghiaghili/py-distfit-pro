# veridist

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

<div lang="fa" dir="rtl">

## وضعیت

`veridist` با نسخهٔ 0.9.0 یک انتشار عمومی با شواهد قابل‌بازبینی است.
این نسخه قراردادهای تحویل کران‌دار، بازپخش‌پذیری، بودجهٔ گذر، تلاش مجدد
تراکنشی، سازگاری checkpoint، شکست‌های نوع‌دار، پیامدهای اجرا و منشأ دادهٔ
پالایش‌شده از اطلاعات حساس را تعریف و آزمایش می‌کند.

این نسخه سلول‌های MLE نمایی، Weibull-minimum و Lognormal با مکان ثابت برای
طول عمرهای دقیق و راست‌سانسورشدهٔ مستقل دارد. هرگاه راه‌حل متناهی وجود داشته
باشد، برآورد نقطه‌ای برمی‌گرداند و در غیر این صورت شکست‌های نوع‌دار می‌دهد.
برای سلول نمایی بدون سانسور، آزمون‌های KS، AD و CvM با Monte Carlo و refit،
AIC/BIC، خلاصهٔ calibration و انتخاب مدل adequacy-gated نیز ارائه می‌شود؛ این
استنباط به همین سلول اعلام‌شده و generator متعلق به فراخواننده محدود است.
مسیر CSV عمومی آن سخت‌گیرانه است: CSV با UTF-8 و دقیقاً سرستون‌های
`time,event_observed`، توکن رخداد `1` و توکن راست‌سانسوری `0`. این مسیر یک
گذر از iterator با بودجهٔ منطقی payload نگه‌داشته‌شده اجرا و نتیجهٔ اجرایی
بسته و نوع‌دار برمی‌گرداند. شواهد نگه‌داری‌شده فقط payload داخلی کران‌دار را
برای ماتریس اندازه‌گیری‌شدهٔ 10k/100k/1m ردیف و بودجه‌های
32KiB/64KiB/128KiB نشان می‌دهند؛ از آن‌ها پشتیبانی عمومی از دادهٔ بزرگ یا
توان عملیاتی بالا نتیجه نمی‌شود. این ادعای سقف RSS قابل‌حمل، لغو، retry،
checkpoint یا برون‌حافظه‌ای عمومی هم نیست.

`IterableDataSource` آداپتور عمومیِ قابل‌استفاده‌مجدد برای iterableهای chunk
متعلق به فراخواننده است. فرادادهٔ تغییرناپذیر آن تک‌گذر یا بازپخش‌پذیر بودن را
صریح اعلام می‌کند: دریافت دوبارهٔ منبع تک‌گذر با خطای نوع‌دار بودجهٔ گذر شکست
می‌خورد و منبع بازپخش‌پذیر به iterator factory نیاز دارد. `BoundedChunkBuffer`
بایت‌های chunkهای صف‌شده و در اختیار مصرف‌کننده را تا `BufferedChunk.release()`
حساب می‌کند؛ فراخواننده باید chunk دریافت‌شده را، معمولاً در `finally`، آزاد کند.
تنها آداپتور فایلِ ارائه‌شده همچنان آداپتور سخت‌گیرانهٔ طول‌عمر CSV است؛ این
قابلیت آداپتورهای عمومی CSV، Parquet، Arrow، dataframe یا out-of-core نیست.

برای یک reducer خالص و ترتیبی، `veridist.engine.SQLiteCheckpointStore` وضعیت
checkpoint محلی پایدار را با compare-and-swap نسلی و قفل‌گذاری SQLite بین
فرایندها فراهم می‌کند. این قابلیت فقط برای یک میزبان و filesystem محلی است؛
`fit_exponential_checkpointed_csv` ادامهٔ صریح و وابسته به source revision را
برای مسیر CSV طول‌عمر سخت‌گیرانه فراهم می‌کند.
این کار ادعای CSV عمومی، Parquet، Arrow، dataframe، پایگاه‌داده یا out-of-core
گسترده ایجاد نمی‌کند.

سطح اسکالر جداگانه فراداده‌های تغییرناپذیر `FAMILY_REGISTRY` را برای پنج خانوادهٔ `normal`، `gamma`، `weibull_min`، `lognormal` و `gumbel_right`، عملیات log-density، CDF، survival، quantile و نمونه‌گیری با RNG متعلق به فراخواننده و کاهش‌دهندهٔ حالتِ دقیق `reduce_log_likelihood_chunks` دارد. عملیات آرایه‌ای نیستند و ماتریس برازش و استنباط از registry محدودتر است. هر چگالیِ لگاریتمی binary64 موفق متناهی است؛ کاهش‌دهنده آن را به‌صورت تعداد صحیحِ دقیقِ واحدهای زیرنرمال انباشته می‌کند و فقط مجموع نهایی را یک‌بار به binary64 گرد می‌کند. کران شمارش unsigned-64 به کران ۲۱۶۲ بیت برای مجموع صحیحِ دقیق می‌انجامد. شواهد ۱۰k/۱۰۰k/۱m فقط مخصوص جریان‌های `normal` آزمایش‌شده است.

اندازه‌گیری scale برای candidate عمداً دستی است: workflow با نام
`veridist-scale-evidence` ابتدا SHA کامل و checkout تمیزِ candidate را bind و
قراردادهای شواهد را اجرا می‌کند؛ سپس مسیر likelihood با iterable عمومی و مسیر
سخت‌گیرانهٔ CSV/نمایی را روی Linux و Windows اندازه می‌گیرد. artifact فقط پس از
اعتبارسنجی fail-closedِ SHA و schema خود نگه‌داری می‌شود. وجود این workflow یا
artifact تاریخی به‌تنهایی شاهد یک candidate تازه نیست و ادعای throughput یا RSS
ایجاد نمی‌کند.

</div>

<div lang="fa" dir="rtl">

## نصب نسخهٔ ارزیابی

پس از clone کردن مخزن، پروژهٔ تو‌در‌توی Python را نصب کنید:

</div>

```console
git clone https://github.com/alisadeghiaghili/veridist.git
cd veridist/python
python -m pip install .
```

<div lang="fa" dir="rtl">

یا wheel مشخصی را که خودتان ساخته‌اید یا از یک اجرای تأییدشده گرفته‌اید نصب
کنید:

</div>

```console
python -m pip install /path/to/veridist-0.9.0-py3-none-any.whl
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

BUSL-1.1 با مجوز استفادهٔ اضافی Apache-2.0 برای استفادهٔ شخصی و غیرتجاری؛
متن کامل در [LICENSE](LICENSE) است.

</div>
