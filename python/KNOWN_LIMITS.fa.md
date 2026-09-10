# محدودیت‌های شناخته‌شدهٔ Veridist 0.5

این سند مرز انتشار موردنظر 0.5 را شرح می‌دهد. تا زمانی که نسخهٔ بسته
`0.5.0` است، این متن ادعای انتشار محسوب
نمی‌شود.

- `FIT-CSV-EXP`: برازش به مدل نمایی با مکان ثابت و فقط پارامتر نرخ برای
  طول‌عمرهای دقیق و مستقلِ راست‌سانسورشده محدود است. استنباط، فاصلهٔ اطمینان،
  آزمون برازش، رتبه‌بندی مدل، وزن، متغیر کمکی، برش، چپ‌سانسوری، سانسور بازه‌ای
  و پارامتر مکان آزاد ارائه نمی‌شود.
- `CSV-STRICT`: adapter فایل همراه بسته فقط CSV با UTF-8 و دقیقاً دو ستون
  `time,event_observed` را می‌پذیرد؛ `1` رخداد دقیق و `0` راست‌سانسوری مستقل
  است. این یک CSV reader یا spreadsheet reader عمومی نیست.
- `SCALAR-FAMILIES`: خانواده‌های normal، gamma، Weibull-minimum، lognormal و
  right-Gumbel فقط log-density اسکالر متناهی را ارزیابی می‌کنند. ارزیابی array،
  برازش، CDF، PPF، استنباط، likelihood سانسورشده و رتبه‌بندی ارائه نمی‌شود.
- `STREAM-SOURCE`: `IterableDataSource`، iterableهای chunk متعلق به فراخواننده
  را سازگار می‌کند. adapter برای Parquet، Arrow، dataframe، database یا network
  همراه بسته نیست و acquisition از نوع checkpoint-replayable رد می‌شود.
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
