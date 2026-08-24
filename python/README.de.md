# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

## Status

`veridist` 0.0.0.dev0 ist ein Pre-Alpha-Vertragskern in aktiver Entwicklung.
Der aktuelle Stand spezifiziert und testet begrenzte Datenlieferung,
Wiederholbarkeit, Pass-Budgets, transaktionale Wiederholungen,
Checkpoint-Kompatibilität, typisierte Fehler, Ausführungsergebnisse und
redigierte Provenienz.

Dieser Stand enthält einen experimentellen, rein ratenparametrisierten
exponentiellen MLE für exakte und unabhängig rechtszensierte Lebensdauern.
Wenn ein endlicher MLE existiert, wird eine Punktschätzung zurückgegeben;
andernfalls entstehen typisierte Fehlschläge. Inferenz wird nicht bereitgestellt.
Der Reduzierer besitzt einen festen algorithmischen O(1)-Zustand. Das Paket
enthält jedoch keine produktionsreifen Datenadapter und beansprucht keine
produktive Out-of-Core-Ausführung oder dauerhafte Checkpoint-Persistenz.
Quellen und Checkpoint-Speicher im Arbeitsspeicher bleiben Vertrags-Fixtures,
keine produktiven Speicher- oder Orchestrierungskomponenten.

## Evaluierungsstand installieren

Nach dem Klonen wird das verschachtelte Python-Projekt installiert:

```console
git clone https://github.com/alisadeghiaghili/py-distfit-pro.git
cd py-distfit-pro/python
python -m pip install .
```

Alternativ kann ein selbst gebautes oder aus einem bestimmten geprüften Lauf
bezogenes Wheel installiert werden:

```console
python -m pip install /path/to/veridist-0.0.0.dev0-py3-none-any.whl
```

Das Projekt fordert nicht zur Installation eines unveröffentlichten
Paketnamens aus einem öffentlichen Index auf.

## Experimentelle Vertikale ausprobieren

```python
from veridist.domain import ExactLifetime, RightCensoredLifetime
from veridist.families import ExponentialFitSuccess, fit_exponential
from veridist.reporting import ReportLocale, render_exponential_report

fit = fit_exponential([ExactLifetime(1.0), RightCensoredLifetime(1.0)])
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5

report = render_exponential_report(fit, ReportLocale.FA)
assert 'lang="fa" dir="rtl"' in report
```

Weitere Einzelheiten stehen in der
[Dokumentations-Toolchain](docs/README.md) und im
[Evidenzregister](../docs/v1-readiness.md). Dort werden implementierte Prüfungen
und ausdrückliche Grenzen getrennt ausgewiesen.

## Lizenz

MIT; siehe [LICENSE](LICENSE).
