# Data prep (`data_prep`)

Builds the **silver** `players` table: FanGraphs projections, merged with Fantrax rosters/positions/pool stats, then MLB Stats API ages. Downstream optimizers (v1/v2) read the written file or call `build_silver_table` in process.

## Run: write silver table to disk

From this directory:

```bash
cd data_prep
uv sync
uv run build-silver-table
```

Default output: **`data/silver_table.parquet`** (path is `data_prep/data/silver_table.parquet` relative to the repo).

CSV instead:

```bash
uv run build-silver-table --output data/silver_table.csv
```

Custom projection files (otherwise uses `config.json` + latest `data/pulled_YYYYMMDD/`):

```bash
uv run build-silver-table \
  --hitter data/pulled_20260323/fangraphs-atc-pt-adjusted-hitters.csv \
  --pitcher data/pulled_20260323/fangraphs-atc-pt-adjusted-pitchers.csv \
  -o data/silver_table.parquet
```

Skip MLB Stats API age fetch (Fantrax ages only):

```bash
uv run build-silver-table --skip-mlb-api
```

### Programmatic

```python
from data_prep import (
    build_silver_table,
    write_silver_table,
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    SILVER_TABLE_DEFAULT_PATH,
)

players = build_silver_table(HITTER_PROJ_PATH, PITCHER_PROJ_PATH)
write_silver_table(players, SILVER_TABLE_DEFAULT_PATH)
```

Requires valid **Fantrax cookies** in repo-root `config.json` (see [docs/FANTRAX_API.md](docs/FANTRAX_API.md)).

---

## Config and data layout

- **Config:** `mlb_fantasy_roster_optimizer/config.json`.
- **Projections:** Downloaded CSVs live under **`data_prep/data/`**, usually in `pulled_YYYYMMDD/`. The package picks the newest `pulled_*` folder (see `data_prep.config.find_latest_projection_folder`).
- **Silver default path:** `data_prep.config.SILVER_TABLE_DEFAULT_PATH` → `data/silver_table.parquet`.

---

## Playing time adjustment: how the external DB is used

PT adjustment is **optional** and **separate** from `build_silver_table`. It reads raw FanGraphs-style CSVs and writes adjusted CSVs; the silver build then consumes whatever hitter/pitcher paths `config.json` points at (`use_adjusted` / file names).

### Where the databases live

| Source | Default path | Purpose |
|--------|----------------|--------|
| **Historical PA / IP** | `{parent_of_repo}/mlb_player_comps_dashboard/mlb_stats.db` | Actual 2023–2024 (by season) PA for hitters, IP (and GS) for pitchers, keyed by `player_id` = FanGraphs `MLBAMID` |
| **Ages for PT step** | `data_prep/data/optimizer.db` | SQLite `players` table: `MLBAMID`, `age` — used only inside `adjust_projections()` for the age factor |

Default `MLB_STATS_DB` is computed in `data_prep/playing_time.py` as:

```text
Path(__file__).resolve().parent.parent.parent.parent / "mlb_player_comps_dashboard" / "mlb_stats.db"
```

i.e. **sibling directory to `mlb_fantasy_roster_optimizer`** named `mlb_player_comps_dashboard`. If that file is missing, historical stats are treated as empty (no PA/IP shrink from history).

Override when calling:

```python
from data_prep.playing_time import adjust_projections, MLB_STATS_DB

adjust_projections(mlb_stats_db="/path/to/mlb_stats.db", optimizer_db="/path/to/optimizer.db")
```

### End-to-end trace (`adjust_projections`)

1. **Load inputs:** Read hitter and pitcher CSVs from `hitters_input` / `pitchers_input` (defaults under `data_prep/data/`).
2. **`_load_historical(mlb_stats_db)`:** If `mlb_stats.db` exists, run SQL aggregations for seasons 2024 and 2023 (hitters: `game_logs.pa`; pitchers: `pitcher_game_logs.ip`, `gs`). If the file is missing, both DataFrames are empty.
3. **`_load_ages(optimizer_db)`:** If `optimizer.db` exists and has a `players` table with `mlbamid` and `age`, load ages; else empty.
4. **`_adjust_hitters` / `_adjust_pitchers`:**
   - Merge historical wide columns (`pa_2024`, `pa_2023` or `ip_*`) onto rows by `MLBAMID`.
   - Merge ages onto rows by `MLBAMID`.
   - For each row, **`_compute_historical_ratio`:** compare average of available actual PA (or IP) to **projected** PA (or IP); ratio = `min(1.0, actual_avg / projected)`; if no history, ratio = 1.0.
   - **`_compute_age_factor`:** If age above threshold (hitter 32, pitcher 33), apply a linear penalty (5% per year over, floor 0.5); missing age → 1.0.
   - **`_compute_talent_factor`:** If WAR below league 25th percentile for that file, multiply by 0.85; else 1.0.
   - **`_blend`:** New PA (or IP) = `0.65 * projected + 0.35 * (projected * hist_ratio * age_factor * talent_factor)`.
   - **Counting stats:** Every column listed in `HITTER_COUNTING_COLS` / `PITCHER_COUNTING_COLS` present in the CSV is multiplied by `(new_PA / old_PA)` or `(new_IP / old_IP)` so rates implied by FanGraphs totals stay consistent. **OPS, ERA, WHIP, etc. are not scaled** (they are not in those lists).
5. **Write** `hitters_output` and `pitchers_output` CSVs.

So: **`mlb_stats.db` drives how much we trust projected volume vs recent reality; `optimizer.db` only feeds the age penalty inside PT adjustment**, not the silver merge (silver ages come from Fantrax + MLB API in `build_silver_table`).

More detail: [docs/PLAYING_TIME_ADJUSTMENT.md](docs/PLAYING_TIME_ADJUSTMENT.md).

---

## Documentation ported from v1 `implementation_specs`

| Doc | Topic |
|-----|--------|
| [docs/PLAYING_TIME_ADJUSTMENT.md](docs/PLAYING_TIME_ADJUSTMENT.md) | PT module, DBs, blend formula |
| [docs/FANGRAPHS_LOADING.md](docs/FANGRAPHS_LOADING.md) | CSV expectations + `load_projections` behavior |
| [docs/FANTRAX_API.md](docs/FANTRAX_API.md) | Auth, endpoints, parsing notes |
| [docs/MLB_STATS_API.md](docs/MLB_STATS_API.md) | Ages batch fetch (`mlb_api`) |

Original paths: `v1/implementation_specs 14-27-50-693/*.md`.

---

## Silver table contract

Columns align with v2’s silver input: `Name` (-H/-P), `Team`, `Position`, `player_type`, counting/rate stats, `WAR`, `owner`, `roster_status`, optional `fantrax_score`, `pct_rostered`, `age`, `MLBAMID`. No `FV`, `optimal_slot`, `PV`, `MEW`, `EWA` (gold layer).
