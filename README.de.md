# veridist

[English](README.md) | [فارسی](README.fa.md) | [Deutsch](README.de.md)

`veridist` ist ein evidenzorientiertes Pre-Alpha-Paket zur
Verteilungsanpassung. Die formale Paketversion ist `0.0.0.dev0`; es ist keine
veroeffentlichte Distribution und nicht ueber einen oeffentlichen Paketindex
verfuegbar.

Der öffentliche Umfang ist absichtlich eng: striktes UTF-8-CSV, ein
Exponential-MLE mit festem Ort und ausschließlich Rate für exakte sowie
unabhaengig rechtszensierte Lebensdauern, fuenf skalare Log-Dichte-Auswerter und
exakte Streaming-Likelihood-Reduktion. Es bietet keine allgemeine Anpassung,
Inferenz, Guetepruefung, Rangfolge, breite Zensierung, allgemeinen CSV-Leser
oder allgemeine Out-of-Core- und Leistungsbehauptung.

Einen Evaluierungs-Build nach dem Klonen installieren:

```console
cd py-distfit-pro/python
python -m pip install .
```

Die Paket-Landingpages enthalten ausfuehrbares Beispiel, Adapter-Vertraege und
Grenzen auf [English](python/README.md), [فارسی](python/README.fa.md) und
[Deutsch](python/README.de.md). Der verbindliche Release-Vertrag in
[ADR-0020](docs/adr/ADR-0020-veridist-0.5-release-contract.md) belaesst die
Version bei `0.0.0.dev0`, bis alle kandidatspezifischen Gates erfuellt sind.

## Sicherheit und Lizenz

Schwachstellen gemaess [SECURITY.md](SECURITY.md) melden. Repository und
verschachteltes Paket verwenden BUSL-1.1 mit der in [LICENSE](LICENSE)
beschriebenen zusaetzlichen Apache-2.0-Nutzungserlaubnis. Diese
Pre-Alpha-Dokumentation aendert keine Lizenz.
