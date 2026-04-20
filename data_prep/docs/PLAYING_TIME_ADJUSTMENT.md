# Playing time adjustment

Ported from v1 `implementation_specs …/02c_playing_time_adjustment.md`, updated for `data_prep`.

## Overview

This module adjusts FanGraphs-style projection CSVs for playing-time bias. Projection systems tend to overstate PA/IP.

**Module:** `data_prep.playing_time`  
**Default inputs:** `data_prep/data/fangraphs-atc-projections-{hitters,pitchers}.csv`  
**Default outputs:** `data_prep/data/fangraphs-atc-pt-adjusted-{hitters,pitchers}.csv`

Output CSVs keep the **same columns** as the inputs. **PA/IP change**, and **listed counting stats are scaled proportionally** so volume stays consistent with FanGraphs’ “rate × playing time” totals. **Rate columns** (e.g. OPS, ERA, WHIP) are **not** multiplied.

---

## Cross-references

**Depends on:**

- External: `mlb_stats.db` (see default path below) — historical PA/IP
- `data_prep/data/optimizer.db` — optional ages for the age factor (`players.mlbamid`, `players.age`)

**Data flow:**

```text
ATC CSVs + mlb_stats.db + optimizer.db → adjust_projections() → adjusted CSVs
```

---

## Research basis

Jeff Zimmerman (FanGraphs 2024 projection review):

1. Systems overproject volume league-wide.
2. Prior playing time, age, and talent level help predict bias.
3. Blend used in code: **65% projection + 35% adjusted** component.

---

## Data sources

### 1. Projection CSVs (input)

FanGraphs exports. Key columns for the adjustment:

- Hitters: `MLBAMID`, `PA`, `WAR`
- Pitchers: `MLBAMID`, `IP`, `WAR` (and `GS` in raw data; used in historical pivot where present)

### 2. Historical actual PA/IP

**Default file:** same parent directory as `mlb_fantasy_roster_optimizer`, then `mlb_player_comps_dashboard/mlb_stats.db`.

Resolved in `data_prep/playing_time.py` as:

```python
_DATA_PREP_ROOT = Path(__file__).resolve().parent.parent  # data_prep project (pyproject + data/)
_REPO_ROOT = _DATA_PREP_ROOT.parent  # mlb_fantasy_roster_optimizer
MLB_STATS_DB = _REPO_ROOT.parent / "mlb_player_comps_dashboard" / "mlb_stats.db"
```

If the file is missing, historical frames are empty → historical ratio is effectively 1.0 for everyone.

```sql
-- Hitter season totals (simplified)
SELECT p.player_id AS mlbam_id, g.season, SUM(g.pa) AS pa
FROM players p JOIN game_logs g ON p.player_id = g.player_id
WHERE g.season IN (2024, 2023)
GROUP BY p.player_id, g.season;

-- Pitcher season totals (simplified)
SELECT p.player_id AS mlbam_id, g.season, SUM(g.ip) AS ip, SUM(g.gs) AS gs
FROM pitchers p JOIN pitcher_game_logs g ON p.player_id = g.player_id
WHERE g.season IN (2024, 2023)
GROUP BY p.player_id, g.season;
```

Join key: `mlbam_id` = FanGraphs `MLBAMID`.

### 3. Ages (PT step only)

**Source:** `data_prep/data/optimizer.db`, table `players`, columns `MLBAMID` / `age`.

Used only for the **age factor** inside `adjust_projections`. If missing, age factor is 1.0.  
**Silver table** ages in the main pipeline come from Fantrax + MLB Stats API (`build_silver_table`), not from this DB.

---

## Public API

```python
from data_prep.playing_time import adjust_projections

adjust_projections(
    hitters_input=...,
    pitchers_input=...,
    hitters_output=...,
    pitchers_output=...,
    mlb_stats_db=...,  # override default
    optimizer_db=...,  # default data_prep/data/optimizer.db
)
```

Returns `{"hitters": int, "pitchers": int, "pa_reduction": float, "ip_reduction": float}`.

---

## Implementation note (vs older spec text)

Older docs sometimes said “only PA/IP change.” The **implemented** behavior also **rescales counting stats** listed in `HITTER_COUNTING_COLS` / `PITCHER_COUNTING_COLS` by the PA or IP ratio so totals stay coherent with FanGraphs’ pre-scaled stat columns.
