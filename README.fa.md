# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

<div lang="fa" dir="rtl">

`veridist` 0.9.1 یک بستهٔ evidence-first برای برازش توزیع است که دامنهٔ اعلام‌شدهٔ
آن با قراردادهای اجرایی CI، پوشش تست، mutation، انتشار و مقیاس کنترل می‌شود.

دامنهٔ عمومی فعلی شامل CSV سخت‌گیرانهٔ UTF-8 و اجرای قابل‌ادامه با SQLite محلی،
سلول‌های MLE نمایی، Weibull-minimum و Lognormal با مکان ثابت برای طول‌عمرهای دقیق
و مستقلِ راست‌سانسورشده، عملیات اسکالر خانواده‌های اعلام‌شده و آزمون نیکویی برازش
Monte Carlo با refit و انتخاب مدل adequacy-gated برای نمونه‌های نمایی بدون سانسور
است. مرزهای دقیق در سند محدودیت‌ها آمده و این نسخه ادعای عمومی RSS، throughput،
سانسور گسترده یا «بهترین برازش» همگانی ندارد.

</div>

برای نصب build ارزیابی پس از clone کردن مخزن:

```console
cd veridist/python
python -m pip install .
```

صفحه‌های بسته شامل مثال اجرایی، قرارداد adapter و محدودیت‌ها به
[English](python/README.md)، [فارسی](python/README.fa.md) و
[Deutsch](python/README.de.md) هستند. [تغییرات candidate](python/CHANGELOG.md) و
[محدودیت‌های شناخته‌شده](python/KNOWN_LIMITS.fa.md) و
[نقشهٔ راه v1](docs/v1-roadmap.md) مرز انتشار و کار باقی‌مانده تا 1.0 را مشخص
می‌کنند.

## امنیت و مجوز

آسیب‌پذیری را مطابق [SECURITY.md](SECURITY.md) گزارش کنید. مجوز مخزن و بستهٔ
تو در تو BUSL-1.1 با مجوز استفادهٔ اضافی Apache-2.0 است که در
[LICENSE](LICENSE) آمده است. این مستندات پیش‌آلفا مجوز را تغییر نمی‌دهد.
