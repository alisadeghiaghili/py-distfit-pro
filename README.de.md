# veridist

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python](https://img.shields.io/pypi/pyversions/veridist.svg)](https://pypi.org/project/veridist/)
[![CI](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)
[![Coverage](https://codecov.io/gh/alisadeghiaghili/veridist/graph/badge.svg)](https://app.codecov.io/gh/alisadeghiaghili/veridist)
[![Mutation gate](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/mutation.yml)
[![Release evidence](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml/badge.svg?branch=main)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-release-evidence.yml)
[![License](https://img.shields.io/badge/license-BUSL--1.1-7b1fa2.svg)](LICENSE)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

## Verteilungsanpassung mit sichtbarer Evidenz

`veridist` 1.0.0 ist ein evidenzorientiertes Paket zur Verteilungsanpassung
für Teams, die ein Lebensdauer-Ergebnis prüfen, reproduzieren und klar
eingrenzen müssen. Es macht aus einer strikten CSV mit Ereigniszeiten ein
typisiertes Ergebnis samt Ausführungsprotokoll, statt Eingabesemantik zu raten
oder pauschal eine „beste Verteilung“ auszugeben.

## Warum Veridist

| Bedarf | Veridist liefert |
| --- | --- |
| Einen klaren Einstieg | Exponential-, Weibull-Minimum- und Lognormal-MLE-Zellen mit festem Ort für exakte und unabhängig rechtszensierte Lebensdauern |
| Vertrauenswürdige Eingaben | Einen strikten UTF-8-Vertrag `time,event_observed`: `1` bedeutet Ereignis, `0` unabhängige Rechtszensur |
| Prüfbare Ergebnisse | Typisierte Schätzungen und Fehler, Fakten zur Ein-Pass-Ausführung und redigierte Quellprovenienz |
| Evidenz vor der Übernahme | CI- und Coverage-Gates, Mutationstests, reproduzierbare Release-Prüfungen und Release-Evidenz |

Das Paket richtet sich an Reliability-, Engineering- und Data-Science-Teams
mit einem deklarierten Lebensdauermodell. Es ist keine breite explorative
Workbench für beliebige Anpassungen.

## In 60 Sekunden starten

Das veröffentlichte Paket installieren:

```console
python -m pip install veridist
```

Dieses vollständige Beispiel erzeugt eine kleine CSV mit einer rechtszensierten
Beobachtung, passt die unterstützte Exponential-Vertikale an und prüft ihre
Modellannahmen.

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

Für einen Checkout `cd veridist/python` und danach `python -m pip install .`
ausführen. Die Paket-Anleitung enthält dasselbe ausführbare Beispiel auf
[English](python/README.md), [فارسی](python/README.fa.md) und
[Deutsch](python/README.de.md).

## Workflow auswählen

| Ziel | Hier beginnen |
| --- | --- |
| Eine kleine, strikte Lebensdauer-CSV anpassen | [Erste Anpassung und CSV-Vertrag](python/docs/source/exponential-right-censoring.md) |
| Daten mit unabhängiger Rechtszensur anpassen | [Modell und Fehlerfälle](python/docs/source/exponential-right-censoring.md#model-and-estimate) |
| Ergebnis oder typisierten Fehler prüfen | [CSV-Exponential-API](python/docs/source/api.md) |
| Deklarierte Verteilung auswerten oder eigene Chunks reduzieren | [Skalare Familien und Streaming-Likelihood](python/docs/source/families-log-density-likelihood.md) |
| Lokales Checkpointing und Fortsetzen planen | [Checkpoint/Resume-Vertragstests](python/tests/contract/test_v1_checkpointed_csv.py) |

Der Checkpoint-Speicher ist lokaler SQLite-Zustand. Ein kompatibler Lauf kann
seinen gespeicherten Präfix fortsetzen; er ist kein verteilter Checkpoint-Dienst.

## Ergebnis vor dem Vertrauen validieren

Ein Ergebnis ist zusammen mit Modellannahmen und Ausführungsprotokoll zu lesen.
Eine endliche Punktschätzung ist weder ein Konfidenzintervall noch ein
Goodness-of-Fit-Urteil. Die unzensierte Exponentialzelle bietet Refit-Monte-
Carlo-KS-, AD- und CvM-Tests, AIC/BIC und adequacy-gesteuerte Auswahl. Diese
Inferenz bleibt auf die deklarierte Zelle und einen aufruferseitigen Generator
beschränkt.

Die Haupt-CI testet unterstützte Python-Versionen und erzwingt mindestens 95 %
Line- und Branch-Coverage. Das Coverage-Badge zeigt den live von Codecov
veröffentlichten Python-3.11-Bericht, keinen fest geschriebenen README-Wert.

## Evidenz, Skalierung und Produktionsgrenzen

Die Release-Validierung bindet einen Kandidaten-SHA, baut das Paket zweimal und
fordert byteidentische Artefakte. Die Release-Grenze umfasst eine 27-Zellen-
Matrix für vollständige Läufe, Retry-Resume und Abbruch bei 10k, 100k und 1m
Zeilen auf Linux, macOS und Windows.

Das belegt keine allgemeine Durchsatz-, RSS-, verteilte, Parquet-, Arrow-,
Dataframe-, Datenbank-, breite Zensierungs-, vektorisierte oder universelle
Best-Fit-Fähigkeit. Vor einem Produktionseinsatz [bekannte Grenzen](python/KNOWN_LIMITS.de.md)
und das [Evidenzregister](docs/v1-readiness.md) lesen.

`distfit_pro`-Material bleibt für historischen und auditierten Kontext im
Repository. Es ist kein Kompatibilitätsversprechen für `veridist`; der Status
steht im [Legacy-Migrationsregister](docs/migration/README.md).

## Dokumentation nach Aufgabe

| Rolle oder Aufgabe | Referenz |
| --- | --- |
| Paket ausprobieren | [Paket-Anleitung](python/README.de.md) und [CSV-Tutorial](python/docs/source/exponential-right-censoring.md) |
| API integrieren | [API-Referenz](python/docs/source/api.md) und [Grenzen](python/KNOWN_LIMITS.de.md) |
| Statistische Oberfläche prüfen | [Familien und Likelihood](python/docs/source/families-log-density-likelihood.md) |
| Qualität und Release-Evidenz prüfen | [Testplan](docs/v1-test-plan.md), [Readiness-Register](docs/v1-readiness.md) und [ADRs](docs/adr/README.md) |
| Beitragen | [Beitragsleitfaden](CONTRIBUTING.md), [Engineering-Konventionen](docs/conventions.md) und [Dokumentations-Toolchain](python/docs/README.md) |

## Support, Sicherheit und Lizenz

Reproduzierbare Fehler und Funktionsvorschläge gehören in die
[GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues).
Sicherheitslücken gemäß [SECURITY.md](SECURITY.md) melden. Die Release-Historie
steht in [python/CHANGELOG.md](python/CHANGELOG.md). Repository und
verschachteltes Paket verwenden BUSL-1.1 mit einer zusätzlichen Apache-2.0-
Nutzungserlaubnis gemäß [LICENSE](LICENSE).
