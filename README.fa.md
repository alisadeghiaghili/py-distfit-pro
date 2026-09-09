# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

<div lang="fa" dir="rtl">

`veridist` یک بستهٔ evidence-first و پیش‌آلفا برای برازش توزیع است. نسخهٔ رسمی
بسته `0.0.0.dev0` است؛ این یک انتشار رسمی نیست و از index عمومی بسته‌ها نصب
نمی‌شود.

دامنهٔ عمومی فعلی عمداً محدود است: CSV سخت‌گیرانهٔ UTF-8، برآوردگر MLE نمایی با
مکان ثابت و فقط پارامتر نرخ برای طول‌عمرهای دقیق و مستقلِ راست‌سانسورشده، پنج
ارزیاب scalar log-density، و کاهش streaming likelihood با حالت دقیق. این بسته
برازش عمومی، استنباط، آزمون برازش، رتبه‌بندی، سانسور گسترده، CSV عمومی، یا ادعای
کلیِ out-of-core و کارایی ارائه نمی‌کند.

</div>

برای نصب build ارزیابی پس از clone کردن مخزن:

```console
cd py-distfit-pro/python
python -m pip install .
```

صفحه‌های بسته شامل مثال اجرایی، قرارداد adapter و محدودیت‌ها به
[English](python/README.md)، [فارسی](python/README.fa.md) و
[Deutsch](python/README.de.md) هستند. قرارداد انتشار لازم در
[ADR-0020](docs/adr/ADR-0020-veridist-0.5-release-contract.md) تا گذر همهٔ
گیت‌های وابسته به candidate، نسخه را روی `0.0.0.dev0` نگه می‌دارد.

## امنیت و مجوز

آسیب‌پذیری را مطابق [SECURITY.md](SECURITY.md) گزارش کنید. مجوز مخزن و بستهٔ
تو در تو BUSL-1.1 با مجوز استفادهٔ اضافی Apache-2.0 است که در
[LICENSE](LICENSE) آمده است. این مستندات پیش‌آلفا مجوز را تغییر نمی‌دهد.
