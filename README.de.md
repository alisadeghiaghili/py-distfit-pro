# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

`veridist` 1.0.0 ist ein evidenzorientiertes Paket zur Verteilungsanpassung.
Sein deklarierter Umfang wird durch ausführbare CI-, Coverage-, Mutation-,
Release- und Skalierungsverträge abgesichert.

Der öffentliche Umfang enthält striktes UTF-8-CSV und fortsetzbare lokale
SQLite-Ausführung, Exponential-, Weibull-Minimum- und Lognormal-MLE-Zellen mit
festem Ort für exakte und unabhängig rechtszensierte Lebensdauern, skalare
Operationen sowie Refit-Monte-Carlo-Anpassungstests und adequacy-gesteuerte
Modellauswahl für unzensierte Exponentialdaten. Die bekannten Grenzen sind
verbindlich; diese Version behauptet keine allgemeinen RSS-, Durchsatz-,
breiten Zensierungs- oder universellen Best-Fit-Eigenschaften.

## Installation

Das veröffentlichte Paket von PyPI installieren:

```console
python -m pip install veridist
```

Für Entwicklung aus einem Repository-Checkout:

```console
cd python
python -m pip install .
```

Die Paket-Landingpages enthalten ausfuehrbares Beispiel, Adapter-Vertraege und
Grenzen auf [English](python/README.md), [فارسی](python/README.fa.md) und
[Deutsch](python/README.de.md). Das [Kandidaten-Changelog](python/CHANGELOG.md)
und die [bekannten Grenzen](python/KNOWN_LIMITS.de.md) sowie die
[v1-Roadmap](docs/v1-roadmap.md) definieren die Release-Grenze und die
verbleibende Arbeit.

## Sicherheit und Lizenz

Schwachstellen gemaess [SECURITY.md](SECURITY.md) melden. Repository und
verschachteltes Paket verwenden BUSL-1.1 mit der in [LICENSE](LICENSE)
beschriebenen zusaetzlichen Apache-2.0-Nutzungserlaubnis.
