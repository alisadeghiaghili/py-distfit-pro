# veridist

[English](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.md) | [فارسی](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.fa.md) | [Deutsch](https://github.com/alisadeghiaghili/py-distfit-pro/blob/main/python/README.de.md)

## Status

`veridist` 0.5.0 ist die erste öffentlich dokumentierte Vertragsversion mit überprüfbaren Nachweisen.
Der aktuelle Stand spezifiziert und testet begrenzte Datenlieferung,
Wiederholbarkeit, Pass-Budgets, transaktionale Wiederholungen,
Checkpoint-Kompatibilität, typisierte Fehler, Ausführungsergebnisse und
redigierte Provenienz.

Dieser Stand enthält einen experimentellen, rein ratenparametrisierten
exponentiellen MLE für exakte und unabhängig rechtszensierte Lebensdauern.
Wenn ein endlicher MLE existiert, wird eine Punktschätzung zurückgegeben;
andernfalls entstehen typisierte Fehlschläge. Inferenz wird nicht bereitgestellt.
Der öffentliche CSV-Pfad ist strikt: UTF-8-CSV mit exakt
`time,event_observed`, Ereignis-Token `1` und Rechtszensur-Token `0`. Er führt
einen Iterator-Durchlauf mit einem deklarierten logischen Budget für behaltene
Payloads aus und gibt ein geschlossenes, typisiertes Ausführungsergebnis zurück.
Dies ist keine Behauptung über allgemeines CSV, portable RSS-Grenzen,
Durchsatz, Abbruch, Retry, Checkpoints oder breite Out-of-Core-Unterstützung.
Die hinterlegte Evidenz belegt begrenzte interne Payload nur für die gemessene
Matrix aus 10k/100k/1m Zeilen und 32KiB/64KiB/128KiB; daraus folgt keine
allgemeine Big-Data- oder Hochdurchsatzfähigkeit.

`IterableDataSource` ist der wiederverwendbare öffentliche Stream-Adapter für
aufruferseitige Chunk-Iterables. Seine unveränderlichen Metadaten erklären
Einmaligkeit oder Wiederholbarkeit ausdrücklich: Eine zweite Akquisition einer
Einmalquelle schlägt mit einem typisierten Pass-Budget-Fehler fehl; eine
wiederholbare Quelle benötigt eine Iterator-Factory. `BoundedChunkBuffer`
berechnet Bytes für eingereihte und vom Consumer gehaltene Chunks bis
`BufferedChunk.release()`; empfangene Chunks müssen, üblicherweise in
`finally`, freigegeben werden. Der einzige ausgelieferte Datei-Adapter bleibt
der strikte CSV-Lebensdauer-Adapter. Dies ergänzt weder allgemeines CSV noch
Parquet-, Arrow-, Dataframe-, Datenbank- oder breite Out-of-Core-Adapter.

Die separate skalare Oberfläche bietet unveränderliche `FAMILY_REGISTRY`-
Metadaten für Normal-, Gamma-, Weibull-Minimum-, Lognormal- und Rechts-Gumbel-
Familien, die skalare `evaluate_log_density` und die exakte Zustandsreduktion
`reduce_log_likelihood_chunks`. Sie ist keine generische Anpassungs-,
Inferenz-, Anpassungsgüte-, Ranking-, Array- oder Zensierungs-API.
Erfolgreiche binary64-Terme werden exakt repräsentiert; die Endsumme wird
einmal gerundet. Die unsigned-64-Zählgrenze ergibt eine 2162-Bit-Grenze für die
exakte Summe. Die Evidenz für 10k/100k/1m gilt nur für getestete Normalströme.

Die Scale-Messung eines Kandidaten ist bewusst manuell: Der Workflow
`veridist-scale-evidence` bindet zunächst den vollständigen SHA eines sauberen
Kandidaten und führt die Evidenzverträge aus. Erst dann misst er den öffentlichen
Iterable-Likelihood-Pfad und den strikten CSV/Exponential-Pfad unter Linux und
Windows. Artefakte werden erst nach ihrer eigenen fail-closed SHA- und
Schema-Validierung aufbewahrt. Das bloße Vorhandensein dieses Workflows oder eines
historischen Artefakts ist kein Beleg für einen neuen Kandidaten und keine
Durchsatz- oder RSS-Behauptung.

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

Weitere Einzelheiten stehen in der
[Dokumentations-Toolchain](docs/README.md) und im
[Evidenzregister](../docs/v1-readiness.md). Dort werden implementierte Prüfungen
und ausdrückliche Grenzen getrennt ausgewiesen.

## Lizenz

BUSL-1.1 mit einer zusätzlichen Apache-2.0-Nutzungserlaubnis für persönliche,
nicht-kommerzielle Nutzung; siehe [LICENSE](LICENSE).
