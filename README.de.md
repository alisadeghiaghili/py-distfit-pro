# Veridist

**Lebensdauermodelle nachvollziehbar anpassen und reproduzieren.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![Coverage ≥95%](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=coverage%20%E2%89%A595%25)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

## Von Ausfallzeiten zu nachvollziehbaren Ergebnissen

Veridist hilft Zuverlässigkeitsingenieuren und Forschenden, Lebensdauermodelle anzupassen, vor dem Ausfall beendete Beobachtungen auszuwerten und die Berechnung zu dokumentieren.

### Drei Einsatzmöglichkeiten

| Ihr Ziel | Das Ergebnis |
| --- | --- |
| Ausfallzeiten analysieren | Exponential-, Weibull-Minimum- und Lognormal-Fits mit festem Ort |
| Ergebnisse prüfen | Explizite Annahmen, typisierte Fehler und Ausführungsdaten |
| Unterbrochene Arbeit fortsetzen | Lokale Checkpoints für kompatible Exponential-Reduktionen |

Es validiert eine definierte Lebensdauer-CSV und gibt entweder einen typisierten
Fit oder einen typisierten Fehler mit Ausführungsprotokoll zurück. Das Beispiel
ausführen und für Installation und API die [Paket-Anleitung](python/README.de.md)
verwenden.

```console
python -m pip install veridist
```

[Beispiel ausführen](#schnellstart) · [Nächsten Schritt wählen](#nächsten-schritt-wählen) · [Paket-Anleitung](python/README.de.md)

## Schnellstart

Das Beispiel erzeugt zwei Beobachtungen. Die erste ist ein Ausfall zur Zeit
`1`; die zweite läuft zur Zeit `1` noch, ist also rechtszensiert und kein
Ausfall.

<details>
<summary>Vollständiges Beispiel anzeigen</summary>

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from veridist import (
    CsvLifetimeLimits,
    CsvLifetimeSchema,
    PublicSourceId,
    fit_exponential_csv,
)
from veridist.families import ExponentialFitSuccess

with TemporaryDirectory() as directory:
    path = Path(directory) / "lifetimes.csv"
    path.write_text("time,event_observed\n1,1\n1,0\n", encoding="utf-8")
    fit = fit_exponential_csv(
        path,
        schema=CsvLifetimeSchema("time", "event_observed"),
        source_id=PublicSourceId("src_0123456789abcdef0123456789abcdef"),
        limits=CsvLifetimeLimits(32_768, 32_768),
    ).fit

assert isinstance(fit, ExponentialFitSuccess)
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

</details>

```text
rate=0.5; events=1; censored=1
```

`rate` ist die Zahl der Ausfälle pro Einheit der Spalte `time`; bei Stunden
also Ausfälle pro Stunde. Das Modell setzt unabhängige Rechtszensur voraus:
das Ende der Beobachtung darf nicht von der unbeobachteten Ausfallzeit abhängen.
Ein erfolgreicher Fit beweist keine Modellangemessenheit; zuerst
[Annahmen und Beurteilung](python/docs/source/exponential-right-censoring.md#model-and-estimate) lesen.

## Einsatzfälle und vorhandene Fähigkeiten

- Ein deklariertes Lebensdauermodell für kontrollierte Ausfalldaten anpassen.
- Rechtszensierte Daten ausdrücken: `1` ist ein beobachtetes Ereignis, `0`
  ein bis zum Beobachtungsende nicht eingetretenes Ereignis.
- Auditierbare Batch-Ausführung mit Ein-Pass-Protokoll und lokalem Checkpoint.

Verfügbar sind Exponential-, Weibull-Minimum- und Lognormal-Modelle mit festem
Ort für exakte und unabhängig rechtszensierte Lebensdauern. Die öffentliche CSV
ist UTF-8 mit genau `time,event_observed`; sie ist kein allgemeiner CSV-Reader
und keine universelle Best-Fit-Suche.

## Nächsten Schritt wählen

| Wenn Sie… | Hier beginnen |
| --- | --- |
| erste strikte Lebensdauer-CSV anpassen | [CSV-Eingabe und erster Fit](python/docs/source/exponential-right-censoring.md) |
| Zensur, Schätzung und Fehler verstehen | [Modellannahmen](python/docs/source/exponential-right-censoring.md#model-and-estimate) |
| API integrieren | [API-Referenz](python/docs/source/api.md) |
| kompatiblen lokalen Lauf fortsetzen | [Checkpoint- und Resume-Anleitung](python/docs/checkpoint-resume.md) |
| Eingabe- oder Betriebsfehler lösen | [Grenzen und Fehlersuche](python/KNOWN_LIMITS.de.md) |

## Große Eingaben und lokales Fortsetzen

Der unterstützte CSV-Pfad arbeitet in einem Iterator-Durchlauf. SQLite kann
einen unterbrochenen lokalen, kompatiblen Lauf mit derselben Source-Revision,
demselben Store und bestätigten Präfix fortsetzen. Das gilt für einen Host und
lokales Dateisystem, nicht für mehrere Worker oder verteilte Stores. Die
[Checkpoint-Anleitung](python/docs/checkpoint-resume.md) enthält ein ausführbares
Zwei-Pass-Beispiel.

Die Release-Evidenz umfasst deklarierte Szenarien mit 10k, 100k und 1m Zeilen
und konkreten Speicherbudgets. Sie belegt keinen allgemeinen Durchsatz, kein
portables RSS, keine verteilte Ausführung, kein Parquet, Arrow, Dataframe,
Datenbank, breite Zensierung oder universelle Modellwahl.

## Umfang und Qualitätsevidenz

Das CI-Badge umfasst Tests, Paketinstallation und Dokumentation. **Coverage
≥95%** bedeutet: Der verlinkte Workflow erzwingt mindestens 95% globale Line-
und Branch-Coverage und zeigt nur Pass/Fail dieses Gates, keinen gemessenen
Prozentwert. Vor Produktion [Grenzen](python/KNOWN_LIMITS.de.md) und
[Evidenzregister](docs/v1-readiness.md) lesen.

`distfit_pro` bleibt für historischen und auditierten Kontext im Repository und
ist kein Kompatibilitätsversprechen für Veridist; siehe
[Migrationsregister](docs/migration/README.md).

## Veridist zitieren

Zitieren Sie die Version, mit der das Ergebnis erzeugt wurde. [IEEE-, BibTeX- und APA-Vorlagen](docs/citing-veridist.md) werden zusammen mit [CITATION.cff](CITATION.cff) gepflegt.

## Hilfe, Beiträge, Zitat und Lizenz

Fehler in [GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues)
und Sicherheitslücken gemäß [SECURITY.md](SECURITY.md) melden. Für Änderungen
[Beitragsleitfaden](CONTRIBUTING.md), für Releases
[CHANGELOG](python/CHANGELOG.md) verwenden. In Berichten Paketversion,
Source-Commit und Modellannahmen angeben.

[BUSL-1.1](LICENSE) gilt mit der dort beschriebenen zusätzlichen Apache-2.0-
Nutzungserlaubnis.
