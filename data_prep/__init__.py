"""
Data prep: fetch each source into the raw layer, then join once.

Two layers, in dependency order:

1. **Fetchers** — one module per source, each writing dated parquet snapshots
   under `data/raw/<source>/`. Pure I/O: no cross-source joins, no derived
   values. See `raw_io` for the storage contract and why it is partitioned by
   refresh cadence and auth rather than by processing stage.
       scrape_fangraphs  -> raw/projections/{steamer,atc}
       fantrax_api       -> raw/fantrax, raw/standings
       market            -> raw/market/{ottoneu,adp,espn,hkb}
       mlb_api           -> raw/identity

2. **The join** — `build.build_players()` reads the latest snapshot of each
   source and produces the single wide `players` table every downstream
   analysis reads. ALL identity reconciliation lives there, once.

Nothing is re-exported here on purpose: import from the submodule you mean, so
the layer a call belongs to is visible at the call site.
"""
