# Bekannte Grenzen von Veridist 0.5

Dieses Dokument beschreibt die vorgesehene Release-Grenze für 0.5. Solange die
Paketversion `0.0.0.dev0` lautet, bleibt das Release ein Kandidat; diese Aussagen
sind keine Veröffentlichungsbehauptung.

- `FIT-CSV-EXP`: Die Anpassung ist auf ein Exponentialmodell mit festem Ort und
  ausschließlich Rate für exakte und unabhängig rechtszensierte Lebensdauern
  begrenzt. Inferenz, Konfidenzintervalle, Anpassungstests, Modellrangfolge,
  Gewichte, Kovariaten, Trunkierung, Links- oder Intervallzensierung und ein
  freier Ortsparameter werden nicht angeboten.
- `CSV-STRICT`: Der mitgelieferte Dateiadapter akzeptiert nur UTF-8-CSV mit
  exakt `time,event_observed`; `1` bezeichnet ein exaktes Ereignis und `0`
  unabhängige Rechtszensierung. Er ist kein allgemeiner CSV- oder
  Tabellenkalkulationsleser.
- `SCALAR-FAMILIES`: Normal-, Gamma-, Weibull-Minimum-, Lognormal- und
  Rechts-Gumbel-Familien werten nur endliche skalare Log-Dichten aus. Arrays,
  Anpassung, CDF, PPF, Inferenz, zensierte Likelihood und Rangfolge fehlen.
- `STREAM-SOURCE`: `IterableDataSource` adaptiert Chunk-Iterables des Aufrufers.
  Das Paket enthält keinen Parquet-, Arrow-, Dataframe-, Datenbank- oder
  Netzwerkadapter. Checkpoint-wiederholbare Akquisition wird abgelehnt.
- `MEMORY-BOUND`: Die Liefergrenze umfasst Nutzdaten in der Warteschlange und
  aktive Verbraucher-Leases bis zur ausdrücklichen Freigabe. Sie ist eine
  logische Grenze für gehaltene Nutzdaten, keine portable RSS-Obergrenze.
- `SCALE-EVIDENCE`: Messungen gelten nur für den exakten Adapter, die Familie,
  Arbeitslast, Plattform, Python-Version, Chunk-Grenze und Kandidaten-SHA. Sie
  belegen weder universellen Durchsatz noch allgemeine Big-Data-Unterstützung
  oder eine breite Out-of-Core-Fähigkeit.
- `LICENSE`: Das Paket verwendet BUSL-1.1 mit der in `LICENSE` beschriebenen
  zusätzlichen Apache-2.0-Nutzungserlaubnis und wechselt am 2030-09-05 zu
  Apache-2.0.

[English](KNOWN_LIMITS.md) | [فارسی](KNOWN_LIMITS.fa.md)
