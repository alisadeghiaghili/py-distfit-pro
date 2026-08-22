# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

## وضعیت

`veridist` با نسخهٔ 0.0.0.dev0 یک هستهٔ قراردادی پیش‌آلفا و در حال توسعه است.
در وضعیت فعلی، قراردادهای تحویل کران‌دار، بازپخش‌پذیری، بودجهٔ گذر، retry
تراکنشی، سازگاری checkpoint، خطاهای نوع‌دار، outcomeهای اجرا و provenance
حذف‌شده از اطلاعات حساس را تعریف و آزمایش می‌کند.

این نسخه API برازش توزیع ارائه نمی‌کند. آداپتور دادهٔ عملیاتی عرضه نمی‌کند.
دوام پایدار checkpoint را ادعا نمی‌کند. منبع‌ها و checkpoint storeهای حافظه‌ای
فقط fixture قراردادی هستند و storage یا orchestrator عملیاتی محسوب نمی‌شوند.

## نصب نسخهٔ ارزیابی

پس از clone کردن مخزن، پروژهٔ تو‌در‌توی Python را نصب کنید:

```console
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro/python
python -m pip install .
```

یا wheel مشخصی را که خودتان ساخته‌اید یا از یک اجرای تأییدشده گرفته‌اید نصب
کنید:

```console
python -m pip install /path/to/veridist-0.0.0.dev0-py3-none-any.whl
```

این پروژه کاربران را به نصب نام یک بستهٔ منتشرنشده از public index هدایت
نمی‌کند.

## بررسی مرز بسته

```python
import veridist

assert veridist.__version__ == "0.0.0.dev0"
```

برای جزئیات، [زنجیرهٔ مستندسازی](docs/README.md) و
[دفتر شواهد](../docs/v1-readiness.md) را ببینید؛ قابلیت‌های پیاده‌شده و
محدودیت‌ها در آن‌ها جدا شده‌اند.

## مجوز

MIT؛ متن کامل در [LICENSE](LICENSE) است.
