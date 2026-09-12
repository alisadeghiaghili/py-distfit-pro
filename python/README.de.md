# Veridist

**Lebensdauermodelle nachvollziehbar anpassen und reproduzieren.**

[![PyPI](https://img.shields.io/pypi/v/veridist.svg)](https://pypi.org/project/veridist/)
[![Python 3.11–3.14](https://img.shields.io/badge/Python-3.11%E2%80%933.14-3776AB)](https://github.com/alisadeghiaghili/veridist/blob/main/docs/capability-matrix.md)
[![Coverage ≥95%](https://img.shields.io/github/actions/workflow/status/alisadeghiaghili/veridist/v1-ci.yml?branch=main&label=coverage%20%E2%89%A595%25)](https://github.com/alisadeghiaghili/veridist/actions/workflows/v1-ci.yml)

[English](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/veridist/blob/main/python/README.de.md)

## Von Ausfallzeiten zu nachvollziehbaren Ergebnissen

Veridist hilft Zuverlässigkeitsingenieuren und Forschenden, Lebensdauermodelle anzupassen, vor dem Ausfall beendete Beobachtungen auszuwerten und die Berechnung zu dokumentieren.

### Drei Einsatzmöglichkeiten

| Ihr Ziel | Das Ergebnis |
| --- | --- |
| Ausfallzeiten analysieren | Exponential-, Weibull-Minimum- und Lognormal-Fits mit festem Ort |
| Ergebnisse prüfen | Explizite Annahmen, typisierte Fehler und Ausführungsdaten |
| Unterbrochene Arbeit fortsetzen | Lokale Checkpoints für kompatible Exponential-Reduktionen |

[Ergebnis ansehen](#ergebnis-ansehen) · [Pfad wählen](#passenden-pfad-wählen) · [Grenzen lesen](KNOWN_LIMITS.de.md)

## Installation und erster Erfolg

Das veröffentlichte Paket installieren:

```console
python -m pip install veridist
```

## Ergebnis ansehen

Nach der Installation diesen vollständigen CSV-Fit ausführen:

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
        limits=CsvLifetimeLimits(32768, 32768),
    ).fit
assert isinstance(fit, ExponentialFitSuccess)
assert fit.rate == 0.5
assert fit.inference == "not_provided"
assert fit.censoring_assumption == "independent_right_censoring"
print(f"rate={fit.rate}; events={fit.event_count}; censored={fit.censored_count}")
```

</details>

```text
rate=0.5; events=1; censored=1
```

## Passenden Pfad wählen

| Bedarf | Einstieg |
| --- | --- |
| Strikte Lebensdauer-CSV anpassen | `fit_exponential_csv` und [CSV-Tutorial](docs/source/exponential-right-censoring.md) |
| Deklarierte skalare Verteilung auswerten | `FAMILY_REGISTRY` und `evaluate_log_density`; [Familienleitfaden](docs/source/families-log-density-likelihood.md) |
| Zustandsgenauen Reduzierer über eigene Chunks verwenden | `reduce_log_likelihood_chunks`; [Stream-API](docs/source/api.md#generic-stream-source-api) |
| Lokales Checkpointing und Resume entwerfen | [Ausführbares SQLite-Rezept](examples/checkpoint_resume.py) und [Grenzen](KNOWN_LIMITS.de.md) |

## Unterstützte Aufgaben

| Für | Ergebnis |
| --- | --- |
| Reliability Engineering | Lebensdauer-Ergebnis mit prüfbaren Ausführungsfakten |
| Rechtszensierte Analyse | Explizite Semantik für Ereignis `1` und unabhängige Rechtszensur `0` |
| Auditierbare Batches | Begrenzter Ein-Pass-Lauf und lokaler Neustart |

## Fähigkeit und Evidenz

Die Anpassungsoberfläche enthält Exponential-, Weibull-Minimum- und Lognormal-MLE-Zellen
mit festem Ort für exakte und unabhängig rechtszensierte Lebensdauern.
Eine endliche Lösung liefert eine Punktschätzung; ungültige statistische oder
betriebliche Bedingungen erzeugen typisierte Fehlschläge. Inferenz ist auf diese
deklarierte Zelle beschränkt: Die unzensierte Exponentialzelle unterstützt
Refit-Monte-Carlo-KS, AD und CvM, AIC/BIC sowie adequacy-gesteuerte Auswahl mit
einem aufruferseitigen Generator.

Der öffentliche CSV-Pfad ist strikt: UTF-8 mit exakt `time,event_observed`,
Ereignis-Token `1` und Rechtszensur-Token `0`. Er benötigt einen
Iterator-Durchlauf und ist kein allgemeines CSV. Ein Ergebnis immer zusammen
mit Ausführungsprotokoll und Modellannahmen lesen.

Die CI prüft unterstützte Python-Versionen, mindestens 95 % Line- und Branch-
Coverage, Qualität, Paketinstallation und Dokumentation. Das Badge Coverage
≥95% zeigt den Pass/Fail-Status dieses erzwungenen Vertrags auf `main`; diese
Seite enthält keinen statischen Coverage-Wert.

## Skalierung und Fortsetzen

Hinterlegte Evidenz umfasst die gemessene Matrix aus 10k/100k/1m Zeilen und
32KiB/64KiB/128KiB für die deklarierten Pfade. Sie belegt keine allgemeine
Big-Data-Unterstützung, keinen Durchsatz, keine portable RSS-Grenze, keine
Dataframe-, Parquet-, Arrow-, Datenbank- oder verteilte Ausführung, keine breite
Zensierung, keine vektorisierten Operationen und keine universelle Modellwahl.
`SQLiteCheckpointStore` ist dauerhafter lokaler Zustand, kein verteilter Dienst.
Für das Fortsetzen Source-Revision stabil halten, denselben lokalen Store öffnen
und den bestätigten Präfix fortsetzen.
Das [ausführbare SQLite-Rezept](examples/checkpoint_resume.py) zeigt die nötige
Initialisierung und einen kompatiblen zweiten Durchlauf.

## Produktionsreife

Vor Produktionseinsatz [KNOWN_LIMITS.de.md](KNOWN_LIMITS.de.md) und das
[Evidenzregister](../docs/v1-readiness.md) lesen.

## Veridist zitieren

Zitieren Sie die Version, mit der das Ergebnis erzeugt wurde. Verwenden Sie die gepflegten [IEEE-, APA-, BibTeX-, RIS-, EndNote-, CSL-JSON-, Chicago-, MLA-, Harvard- und Vancouver-Vorlagen](../docs/citing-veridist.md) mit [CITATION.cff](../CITATION.cff).

## Dokumentation, Beiträge und Support

Für Integration die [API-Referenz](docs/source/api.md), für Statistik den
[Familienleitfaden](docs/source/families-log-density-likelihood.md) und für
Dokumentation die [Toolchain](docs/README.md) verwenden. Änderungen beginnen
mit dem [Beitragsleitfaden](../CONTRIBUTING.md) und den
[Engineering-Konventionen](../docs/conventions.md). Reproduzierbare Fehler in
[GitHub Issues](https://github.com/alisadeghiaghili/veridist/issues) und
Sicherheitslücken gemäß [SECURITY.md](../SECURITY.md) melden.

BUSL-1.1 mit einer zusätzlichen Apache-2.0-Nutzungserlaubnis für persönliche,
nicht-kommerzielle Nutzung; siehe [LICENSE](LICENSE).
