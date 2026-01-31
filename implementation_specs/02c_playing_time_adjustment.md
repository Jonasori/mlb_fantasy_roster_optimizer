# Playing Time Adjustment Module

## Overview

This standalone module adjusts ATC projections for playing time bias. All projection systems (including ATC) systematically overproject playing time. This module corrects that using historical data.

**Module:** `optimizer/playing_time.py`  
**Input:** `data/fangraphs-atc-projections-{hitters,pitchers}.csv`  
**Output:** `data/fangraphs-atc-pt-adjusted-{hitters,pitchers}.csv`

The output files are **structurally identical** to the FanGraphs originals—same columns, same order. Only PA (hitters) and IP (pitchers) values are adjusted. This allows them to be used as drop-in replacements.

---

## Cross-References

**Depends on:**
- [01f_mlb_stats_api.md](01f_mlb_stats_api.md) — ages loaded from `data/optimizer.db` (populated by data pipeline)
- External: `../mlb_player_comps_dashboard/mlb_stats.db` — historical PA/IP data

**Data flow:**
```
ATC CSVs + mlb_stats.db + optimizer.db → playing_time module → adjusted CSVs
```

---

## Research Basis

Jeff Zimmerman's analysis (FanGraphs 2024 Projection Review):

1. **All systems overproject** — FanGraphs overprojected by 78.8 PA per hitter; average overshoot was 10,000+ PA league-wide
2. **Marcels ranked #1** for established players — undershot totals by only 800 PA (vs. 10,000+ overshoot by others)

**Three predictive factors:**
1. Prior 2 seasons of actual playing time (strongest predictor)
2. Age (older players systematically overprojected)
3. Talent (weaker players receive inflated estimates)

**Optimal blend:** 65% projection + 35% adjustment (improved RMSE from 153.2 to 146.3)

---

## Data Sources

### 1. ATC Projections (Input)

Already downloaded:
- `data/fangraphs-atc-projections-hitters.csv`
- `data/fangraphs-atc-projections-pitchers.csv`

Key columns for adjustment:
- Hitters: `MLBAMID`, `PA`, `WAR`
- Pitchers: `MLBAMID`, `IP`, `GS`, `WAR`

All other columns are preserved unchanged.

### 2. Historical Actual PA/IP

**Source:** `../mlb_player_comps_dashboard/mlb_stats.db`

```sql
-- Hitter season totals
SELECT 
    p.player_id as mlbam_id,
    g.season,
    SUM(g.pa) as pa
FROM players p
JOIN game_logs g ON p.player_id = g.player_id
WHERE g.season IN (2024, 2023)
GROUP BY p.player_id, g.season;

-- Pitcher season totals
SELECT 
    p.player_id as mlbam_id,
    g.season,
    SUM(g.ip) as ip,
    SUM(g.gs) as gs
FROM pitchers p
JOIN pitcher_game_logs g ON p.player_id = g.player_id
WHERE g.season IN (2024, 2023)
GROUP BY p.player_id, g.season;
```

**Cross-reference key:** `mlbam_id` = FanGraphs `MLBAMID`

### 3. Player Ages

**Source:** `data/optimizer.db` (populated by data pipeline)

```sql
SELECT MLBAMID as mlbam_id, age FROM players WHERE age IS NOT NULL
```

**Note:** This module reads ages from the database. If ages are missing (e.g., `skip_mlb_api=True` during data refresh), the module skips the age adjustment factor and still applies historical and talent adjustments.

---

## Module Structure

Single file: `optimizer/playing_time.py`

Exports one public function: `adjust_projections()`

Called at the start of `notebook.py` when `USE_ADJUSTED_PROJECTIONS=True`.

**Key constraint:** This module has NO dependencies on the existing optimizer package. It reads files and outputs CSVs.

---

## Public API

```python
from optimizer.playing_time import adjust_projections

adjust_projections(
    hitters_input="data/fangraphs-atc-projections-hitters.csv",
    pitchers_input="data/fangraphs-atc-projections-pitchers.csv",
    hitters_output="data/fangraphs-atc-pt-adjusted-hitters.csv",
    pitchers_output="data/fangraphs-atc-pt-adjusted-pitchers.csv",
)
```

Returns a dict with summary: `{"hitters": int, "pitchers": int, "pa_reduction": float, "ip_reduction": float}`

---

## Notebook Integration

In `notebook.py`, set `USE_ADJUSTED_PROJECTIONS=True` in the config cell:

```python
USE_ADJUSTED_PROJECTIONS = True  # Apply playing time adjustments
```

This automatically calls `adjust_projections()` and points the data loader at the adjusted files.

---

## Output Format

The output CSVs are **structurally identical** to the FanGraphs originals:

- Same columns, same order
- Only `PA` (hitters) and `IP` (pitchers) values are changed

This makes them drop-in replacements for the raw projections.

---

## Dependencies

This module uses only:
- `pandas` — DataFrame operations
- `sqlite3` — Database access (stdlib)
- `numpy` — Numeric operations

No external API calls. All data comes from files and databases populated by the data pipeline.
