# محدودیت‌های شناخته‌شدهٔ Veridist 0.9

این سند مرز انتشار 0.9 برای نسخهٔ بستهٔ `0.9.0` را تعریف می‌کند.

- `FIT-CSV-EXP`: مسیر CSV فقط مدل نمایی با مکان ثابت و پارامتر نرخ را برای
  طول‌عمرهای دقیق و مستقلِ راست‌سانسورشده برازش می‌دهد. برازش Weibull-minimum
  و Lognormal با اشیای طول‌عمر نوع‌دار قابل‌فراخوانی است و API عمومی فایل نیست.
  وزن تحلیلی، متغیر کمکی، truncation، چپ‌سانسوری، سانسور بازه‌ای و مکان آزاد
  پشتیبانی نمی‌شوند.
- `CSV-STRICT`: adapter فایل همراه بسته فقط CSV با UTF-8 و دقیقاً دو ستون
  `time,event_observed` را می‌پذیرد؛ `1` رخداد دقیق و `0` راست‌سانسوری مستقل
  است. این یک CSV reader یا spreadsheet reader عمومی نیست.
- `SCALAR-FAMILIES`: خانواده‌های normal، gamma، Weibull-minimum، lognormal و
  right-Gumbel عملیات اسکالر log-density، CDF، survival، quantile و sampling
  دارند. ارزیابی array، API یکنواخت برازش و استنباط برای همهٔ خانواده‌ها ارائه
  نمی‌شود.
- `STREAM-SOURCE`: `IterableDataSource`، iterableهای chunk متعلق به فراخواننده
  را سازگار می‌کند. adapter برای Parquet، Arrow، dataframe، database یا network
  همراه بسته نیست. ادامهٔ پایدار به مسیر CSV طول‌عمر سخت‌گیرانه و SQLite محلی
  محدود است و checkpoint توزیع‌شده نیست.
- `INFERENCE-EXP`: آزمون‌های KS/AD/CvM با Monte Carlo و refit و انتخاب
  adequacy-gated فقط برای نمونه‌های نمایی مثبت، متناهی و بدون سانسور هستند.
  ثبات انتخاب bootstrap یا calibration بیرون از grid آزموده‌شده ادعا نمی‌شود.
- `MEMORY-BOUND`: کران delivery، payload صف و lease فعال مصرف‌کننده را تا
  release صریح محاسبه می‌کند. این کران منطقی payload نگه‌داری‌شده است، نه سقف
  قابل‌حمل RSS.
- `SCALE-EVIDENCE`: اندازه‌گیری فقط برای adapter، خانواده، workload، platform،
  نسخهٔ Python، حد chunk و SHA همان candidate معتبر است. این شواهد throughput
  همگانی، پشتیبانی generic big-data یا قابلیت گستردهٔ out-of-core را ثابت
  نمی‌کنند.
- `LICENSE`: مجوز بسته BUSL-1.1 با مجوز استفادهٔ اضافی Apache-2.0 مندرج در
  `LICENSE` است و در 2030-09-05 به Apache-2.0 تغییر می‌کند.

[English](KNOWN_LIMITS.md) | [Deutsch](KNOWN_LIMITS.de.md)
