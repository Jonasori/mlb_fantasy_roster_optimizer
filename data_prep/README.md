# Data prep (`data_prep`)

Fetch each source independently into a **raw layer** of dated snapshots, then
**join once** into the wide `players` table all analysis reads.

```
data/raw/projections/{steamer,atc}/YYYY-MM-DD.parquet   daily,  browser cookies
data/raw/fantrax/YYYY-MM-DD.parquet                     on txn, PASTED cookies
data/raw/standings/YYYY-MM-DD.parquet                   on txn, PASTED cookies
data/raw/market/{ottoneu,adp,espn,hkb}/YYYY-MM-DD.parquet  daily, no auth
data/raw/identity/YYYY-MM-DD.parquet                    ~never, no auth
                              |
                              v  build.build_players()
data/players.parquet          the one wide table
```

Storage is partitioned by **refresh cadence and auth**, not by processing stage.
That is deliberate. The previous design baked projections, Fantrax rosters and
ages into one "silver" table, so every refresh had to satisfy the union of their
requirements: a stale Fantrax cookie blocked getting fresh projections, and the
table could only ever hold one projection system. Both problems disappear when
each source lands on its own.

## Run

```bash
uv run fetch status         # how stale is each source?
uv run fetch market         # no auth — safe anytime
uv run fetch identity       # no auth
uv run fetch projections    # FanGraphs, via browser cookies
uv run fetch fantrax        # needs FRESH cookies in config.json
uv run fetch build          # join latest snapshots -> data/players.parquet
uv run fetch build --system steamer    # same snapshots, different projections
uv run fetch all            # every source, then build
```

Switching projection systems is a read from a different directory — no rebuild,
no re-auth. Reproduce a past day with `build_players(on_or_before=date(...))`.

Only **Fantrax** needs hand-pasted cookies in repo-root `config.json` (see
[docs/FANTRAX_API.md](docs/FANTRAX_API.md)); they expire every few weeks. Every
other source is free to refresh.

## Layers

**Fetchers** — one module per source, each writing raw snapshots verbatim. No
cross-source joins, no derived values. `raw_io` holds the storage contract
(`write_raw`, `read_latest_raw`, `available_dates`, `snapshot_ages`).

| Module | Writes |
|---|---|
| `scrape_fangraphs` | `raw/projections/{steamer,atc}` |
| `fantrax_api` | `raw/fantrax`, `raw/standings` |
| `market` | `raw/market/{ottoneu,adp,espn,hkb}` |
| `mlb_api` | `raw/identity` |
| `ceiling` | `raw/projections/oopsypeak`, `raw/savant` |

`ceiling` is the one fetcher that is also an analysis: it ranks dynasty players
by TAIL upside rather than expected value and does **not** feed the `players`
table. `uv run python -m data_prep.ceiling --help`.

**The join** — `build.build_players()` is the *only* place identity is
reconciled across sources. Keys in order of trust: `MLBAMID` (MLB's own),
`PlayerId` (FanGraphs, and Ottoneu's `fg_id`), then normalized name as a last
resort. No provider covers everyone, so each merge is a **cascade**: strong key
first, name for the remainder. That measurably beats either key alone — Ottoneu
prices 951 players by FanGraphs id but 987 by id-then-name, because ~245 of its
rows are minor leaguers carrying only a FanGraphs *minor*-league id.

Only projections are load-bearing. A missing Fantrax snapshot yields a table
with no ownership; a missing market snapshot yields one without the dynasty
axis.

## `players` table contract

Identity: `Name` (with the `-H`/`-P` suffix distinguishing a two-way player's
two sides), `Team`, `Position`, `player_type`, `PlayerId`, `MLBAMID`,
`fantrax_id`, `age`, `birth_date`.

Projections: `PA R HR RBI SB OPS` / `IP W SV K ERA WHIP`, plus `WAR`. All 12
scoring stats are always present and non-null — zero for the opposite player
type — so the MEW formula needs no hitter/pitcher branching.

Fantrax: `owner` (null = free agent), `roster_status`, `injury_status`,
`injury_detail`, `fantrax_score`, `pct_rostered`.

Market: `market_value` (Ottoneu median salary), `salary_momentum` (last-10
auctions vs. median), `adp`, `pct_owned`, `dynasty_value`, `dynasty_rank`.

Downstream columns (`FV`, `MEW`, `BV`, `optimal_slot`) are added by `optimizer/`,
not here.

## Docs

| Doc | Topic |
|-----|--------|
| [docs/FANGRAPHS_LOADING.md](docs/FANGRAPHS_LOADING.md) | FanGraphs field expectations |
| [docs/FANTRAX_API.md](docs/FANTRAX_API.md) | Auth, endpoints, parsing notes |
| [docs/MLB_STATS_API.md](docs/MLB_STATS_API.md) | Identity batch fetch |
