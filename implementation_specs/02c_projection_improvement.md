# Projection Improvement Module: Playing Time & Combination

## Overview

This is a **standalone module** that improves projection accuracy through:
1. Playing time adjustment (PA/IP correction using historical data)
2. Projection combination (stat-specific weighted averaging)

The module operates independently and outputs an enhanced projections CSV that can be consumed by the existing optimizer.

**Output:** `data/combined_projections.csv` — ready to use as input for `load_projections()`

---

## Research Basis

### Playing Time Adjustment

Jeff Zimmerman's analysis (FanGraphs 2024 Projection Review) found:

1. **All systems overproject** — FanGraphs overprojected by 78.8 PA per hitter; average overshoot was 10,000+ PA across all hitters
2. **Simple averaging ranked #1** — RMSE: 145.7
3. **Marcels (#1 for established players)** — Undershot totals by only 800 PA vs. 10,000+ PA overshoot by others

**Three predictive factors:**
1. Prior 2 seasons of actual playing time (strongest predictor)
2. Age (older players systematically overprojected)
3. Talent (weaker players receive inflated estimates)

**Optimal blend:** 65% projection + 35% adjustment (improved RMSE from 153.2 to 146.3)

### Projection Combination

The "forecast combination puzzle" (Smith & Wallis 2009): Simple averaging often beats complex weighting.

**ATC approach** (Ariel Cohen): Different weights per statistic based on historical accuracy. Consistently #1 in FantasyPros accuracy rankings.

---

## Data Sources

### 1. Historical Actual PA/IP

**Source:** `../mlb_player_comps_dashboard/mlb_stats.db`

This SQLite database contains game-by-game stats from the MLB Stats API:

```sql
-- Hitter season totals (2023, 2024, 2025 available)
SELECT 
    p.player_id as mlbam_id,
    p.name,
    g.season,
    SUM(g.pa) as pa,
    SUM(g.r) as r,
    SUM(g.hr) as hr,
    SUM(g.rbi) as rbi,
    SUM(g.sb) as sb
FROM players p
JOIN game_logs g ON p.player_id = g.player_id
GROUP BY p.player_id, g.season;

-- Pitcher season totals
SELECT 
    p.player_id as mlbam_id,
    p.name,
    g.season,
    SUM(g.ip) as ip,
    SUM(g.w) as w,
    SUM(g.sv) as sv,
    SUM(g.k) as k,
    SUM(g.gs) as gs
FROM pitchers p
JOIN pitcher_game_logs g ON p.player_id = g.player_id
GROUP BY p.player_id, g.season;
```

**Current data:** 911 unique hitters, 1302 unique pitchers across 2023-2025.

### 2. Player Ages (Birthdates)

**Source:** pybaseball's Chadwick Bureau register

```python
from pybaseball import chadwick_register

# Download full register (~490,000 players)
register = chadwick_register()

# Filter to MLB players with recent activity
mlb_players = register[
    (register['mlb_played_last'] >= 2020) & 
    (register['key_mlbam'].notna())
]

# Relevant columns:
# - key_mlbam: MLBAM ID (matches player_id in mlb_stats.db and MLBAMID in FanGraphs)
# - birth_year, birth_month, birth_day
# - name_first, name_last
```

**Cross-reference key:** `key_mlbam` = `player_id` = FanGraphs `MLBAMID`

### 3. Projection Source CSVs

**Source:** FanGraphs projections (requires free account for export)

| System | Hitters URL | Pitchers URL |
|--------|-------------|--------------|
| **Steamer** | `fangraphs.com/projections?pos=all&stats=bat&type=steamer` | `fangraphs.com/projections?pos=all&stats=pit&type=steamer` |
| **ZiPS** | `fangraphs.com/projections?pos=all&stats=bat&type=zips` | `fangraphs.com/projections?pos=all&stats=pit&type=zips` |
| **THE BAT** | `fangraphs.com/projections?pos=all&stats=bat&type=thebat` | `fangraphs.com/projections?pos=all&stats=pit&type=thebat` |
| **Depth Charts** | `fangraphs.com/projections?pos=all&stats=bat&type=rfangraphsdc` | `fangraphs.com/projections?pos=all&stats=pit&type=rfangraphsdc` |

**Download instructions:**
1. Navigate to URL
2. Click "Export Data" button (top-right of table)
3. Save to `data/projections/{system}_{hitters|pitchers}.csv`

**Required columns:**
- Hitters: `Name`, `MLBAMID`, `Team`, `PA`, `R`, `HR`, `RBI`, `SB`, `OPS`, `WAR`
- Pitchers: `Name`, `MLBAMID`, `Team`, `IP`, `W`, `SV`, `SO`, `ERA`, `WHIP`, `GS`, `WAR`

---

## Module Structure

```
optimizer/
  projection_improvement/
    __init__.py
    config.py           # Constants and configuration
    data_sources.py     # Load historical, ages, projections
    playing_time.py     # PA/IP adjustment algorithm
    combination.py      # Projection averaging/weighting
    main.py             # CLI entry point
```

**Key constraint:** This module has NO dependencies on the rest of the optimizer package. It reads external data and outputs a CSV.

---

## Function Specifications

### config.py

```python
"""Configuration constants for projection improvement."""

from pathlib import Path

# External data paths
MLB_STATS_DB = Path("../mlb_player_comps_dashboard/mlb_stats.db")
PROJECTIONS_DIR = Path("data/projections")
OUTPUT_PATH = Path("data/combined_projections.csv")

# Playing time constants
FULL_SEASON_PA = 600          # Full-time hitter plays ~600 PA
FULL_SEASON_IP_SP = 180       # Full-time starter pitches ~180 IP
FULL_SEASON_IP_RP = 70        # Full-time reliever pitches ~70 IP

# Age penalty thresholds
AGE_PENALTY_START_HITTER = 32
AGE_PENALTY_START_PITCHER = 33
AGE_PENALTY_PER_YEAR = 0.05   # 5% per year over threshold

# Talent penalty
TALENT_PENALTY_PERCENTILE = 0.25  # Bottom quartile
TALENT_PENALTY_FACTOR = 0.85      # 15% reduction

# Blend ratio (from Zimmerman's research)
PROJECTION_WEIGHT = 0.65
ADJUSTMENT_WEIGHT = 0.35

# Default projection system weights (if not calibrating from data)
# These are starting points; adjust based on backtesting
DEFAULT_WEIGHTS = {
    "PA": {"steamer": 0.30, "zips": 0.25, "thebat": 0.20, "depthcharts": 0.25},
    "IP": {"steamer": 0.30, "zips": 0.25, "thebat": 0.20, "depthcharts": 0.25},
    "HR": {"steamer": 0.35, "zips": 0.35, "thebat": 0.30},
    "SB": {"steamer": 0.30, "zips": 0.40, "thebat": 0.30},
    "R":  {"steamer": 0.33, "zips": 0.33, "thebat": 0.34},
    "RBI": {"steamer": 0.33, "zips": 0.33, "thebat": 0.34},
    "OPS": {"steamer": 0.35, "zips": 0.30, "thebat": 0.35},
    "W":  {"steamer": 0.33, "zips": 0.33, "thebat": 0.34},
    "SV": {"steamer": 0.33, "zips": 0.33, "thebat": 0.34},
    "K":  {"steamer": 0.35, "zips": 0.35, "thebat": 0.30},
    "ERA": {"steamer": 0.35, "zips": 0.30, "thebat": 0.35},
    "WHIP": {"steamer": 0.35, "zips": 0.30, "thebat": 0.35},
}
```

### data_sources.py

```python
"""Load data from external sources."""

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
from pybaseball import chadwick_register


def load_historical_actuals(db_path: Path, seasons: list[int]) -> dict[str, pd.DataFrame]:
    """
    Load historical PA/IP from mlb_stats.db.
    
    Args:
        db_path: Path to mlb_stats.db
        seasons: List of seasons to load (e.g., [2023, 2024])
    
    Returns:
        {
            "hitters": DataFrame with columns [mlbam_id, name, season, pa, r, hr, rbi, sb],
            "pitchers": DataFrame with columns [mlbam_id, name, season, ip, w, sv, k, gs]
        }
    
    Implementation:
        1. Connect to SQLite database
        2. Aggregate game_logs by player_id and season
        3. Aggregate pitcher_game_logs by player_id and season
        4. Return both DataFrames
    """


def load_player_ages() -> pd.DataFrame:
    """
    Load player birthdates from Chadwick register.
    
    Returns:
        DataFrame with columns [mlbam_id, name, birth_year, birth_month, birth_day, age]
        
        Age is calculated as of today's date.
    
    Implementation:
        1. Download chadwick_register() from pybaseball
        2. Filter to mlb_played_last >= 2020 AND key_mlbam IS NOT NULL
        3. Rename key_mlbam -> mlbam_id
        4. Compute age from birth_year/month/day
        5. Return DataFrame
    """


def load_projection_csv(filepath: Path, player_type: str) -> pd.DataFrame:
    """
    Load a single FanGraphs projection CSV.
    
    Args:
        filepath: Path to CSV file
        player_type: "hitter" or "pitcher"
    
    Returns:
        DataFrame with standardized columns.
        Pitchers: SO renamed to K.
        MLBAMID preserved for joining.
    
    Assertions:
        - Required columns present
        - No null MLBAMID values
    """


def load_all_projections(projections_dir: Path) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Load all projection CSVs from directory.
    
    Expected files:
        steamer_hitters.csv, steamer_pitchers.csv,
        zips_hitters.csv, zips_pitchers.csv,
        thebat_hitters.csv, thebat_pitchers.csv,
        depthcharts_hitters.csv, depthcharts_pitchers.csv
    
    Returns:
        {
            "steamer": {"hitters": df, "pitchers": df},
            "zips": {"hitters": df, "pitchers": df},
            ...
        }
    
    Missing files are skipped with a warning (not all systems required).
    """
```

### playing_time.py

```python
"""Playing time adjustment using historical data."""

import pandas as pd


def compute_historical_pt_ratio(
    actual_pa_2024: int | None,
    actual_pa_2023: int | None,
    full_season_pa: int,
) -> float:
    """
    Compute ratio of actual to expected playing time.
    
    Args:
        actual_pa_2024: Actual PA from 2024 (or None if no data)
        actual_pa_2023: Actual PA from 2023 (or None)
        full_season_pa: Expected PA for a full-time player (e.g., 600)
    
    Returns:
        Float in range [0.0, 1.0].
        - If no history: return 1.0 (trust projection)
        - Otherwise: min(1.0, avg(actual) / full_season_pa)
    
    The cap at 1.0 prevents giving bonus for high prior PT.
    """


def compute_age_factor(age: int | None, age_penalty_start: int) -> float:
    """
    Compute age-based playing time factor.
    
    Args:
        age: Player age (or None)
        age_penalty_start: Age at which penalty begins (32 for hitters, 33 for pitchers)
    
    Returns:
        Float in range [0.5, 1.0].
        - Under threshold: 1.0
        - Over threshold: max(0.5, 1.0 - 0.05 * years_over)
    
    Example: age 35, threshold 32 → 1.0 - 0.05*3 = 0.85
    """


def compute_talent_factor(sgp: float, sgp_25th_percentile: float) -> float:
    """
    Compute talent-based playing time factor.
    
    Args:
        sgp: Player's SGP value
        sgp_25th_percentile: 25th percentile SGP for player type
    
    Returns:
        1.0 if SGP >= threshold, else 0.85 (15% penalty for weak players)
    """


def adjust_playing_time(
    projected_pt: float,
    historical_ratio: float,
    age_factor: float,
    talent_factor: float,
) -> float:
    """
    Apply three-factor adjustment and blend with projection.
    
    Formula:
        adjusted_pt = projected_pt * historical_ratio * age_factor * talent_factor
        blended_pt = 0.65 * projected_pt + 0.35 * adjusted_pt
    
    Returns:
        Blended playing time value
    """


def apply_playing_time_adjustments(
    projections: pd.DataFrame,
    historical_actuals: dict[str, pd.DataFrame],
    ages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply playing time adjustments to all players.
    
    Args:
        projections: DataFrame with PA, IP, MLBAMID, player_type columns
        historical_actuals: Output from load_historical_actuals()
        ages: Output from load_player_ages()
    
    Returns:
        projections DataFrame with PA_adjusted and IP_adjusted columns added.
        Original PA/IP preserved for comparison.
    
    Implementation:
        1. Join historical actuals by mlbam_id (left join)
        2. Join ages by mlbam_id (left join)
        3. Compute SGP percentiles for hitters and pitchers
        4. For each player, compute:
           - historical_ratio
           - age_factor
           - talent_factor
        5. Apply adjust_playing_time()
        6. Return DataFrame
    
    Print summary:
        "Playing time adjustments applied:"
        "  Hitters: avg PA change = +X.X / -X.X"
        "  Pitchers: avg IP change = +X.X / -X.X"
    """
```

### combination.py

```python
"""Projection combination methods."""

import pandas as pd
import numpy as np


def simple_average(
    projections: dict[str, pd.DataFrame],
    stat_columns: list[str],
    join_key: str = "MLBAMID",
) -> pd.DataFrame:
    """
    Combine projections using simple equal-weighted average.
    
    This is the baseline method. Research shows it often beats complex
    weighting due to estimation error in weight calculation.
    
    Args:
        projections: Dict mapping system name to DataFrame
        stat_columns: Columns to average (e.g., ["PA", "HR", "SB", "OPS"])
        join_key: Column to join on
    
    Returns:
        DataFrame with averaged stat columns.
        Name, Team, and other metadata come from first system.
    
    Implementation:
        1. Merge all DataFrames on join_key (outer join)
        2. For each stat, compute mean across available systems
        3. Return combined DataFrame
    """


def weighted_average(
    projections: dict[str, pd.DataFrame],
    weights: dict[str, dict[str, float]],
    join_key: str = "MLBAMID",
) -> pd.DataFrame:
    """
    Combine projections using stat-specific weights.
    
    This is the ATC-style approach.
    
    Args:
        projections: Dict mapping system name to DataFrame
        weights: Dict mapping stat name to weight dict
            Example: {"HR": {"steamer": 0.4, "zips": 0.35, "thebat": 0.25}}
            Weights for each stat MUST sum to 1.0.
        join_key: Column to join on
    
    Returns:
        DataFrame with weighted stat columns.
    
    Assertions:
        - Weights sum to 1.0 for each stat
        - All weight sources exist in projections
    """


def calibrate_weights_from_history(
    historical_projections: dict[int, dict[str, pd.DataFrame]],
    historical_actuals: dict[int, pd.DataFrame],
    stats: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute optimal weights using inverse MSE.
    
    Args:
        historical_projections: {year: {system: df}} 
            e.g., {2023: {"steamer": df, "zips": df}, 2024: {...}}
        historical_actuals: {year: df}
        stats: Stats to calibrate (e.g., ["HR", "SB", "ERA"])
    
    Returns:
        Dict mapping stat to weight dict.
        
    Implementation (for each stat):
        1. For each system, compute prediction errors across all years
        2. Compute MSE for each system
        3. Weight = (1/MSE) / sum(1/MSE) for all systems
        4. Return normalized weights
    
    Note: Requires 3+ years of historical data for stable weights.
    """
```

### main.py

```python
"""CLI entry point for projection improvement."""

import argparse
from pathlib import Path

import pandas as pd

from .config import (
    DEFAULT_WEIGHTS,
    MLB_STATS_DB,
    OUTPUT_PATH,
    PROJECTIONS_DIR,
)
from .data_sources import (
    load_all_projections,
    load_historical_actuals,
    load_player_ages,
)
from .playing_time import apply_playing_time_adjustments
from .combination import simple_average, weighted_average


def main():
    """
    Main entry point.
    
    Usage:
        python -m optimizer.projection_improvement.main
        python -m optimizer.projection_improvement.main --method weighted
        python -m optimizer.projection_improvement.main --output custom_output.csv
    
    Steps:
        1. Load historical actuals from mlb_stats.db
        2. Load player ages from Chadwick register
        3. Load all projection CSVs
        4. Combine projections (simple or weighted average)
        5. Apply playing time adjustments
        6. Compute SGP for combined projections
        7. Output combined_projections.csv
    
    Output columns (matching existing load_projections() format):
        Name, Team, Position, MLBAMID,
        PA, R, HR, RBI, SB, OPS,        # Hitters
        IP, W, SV, K, ERA, WHIP, GS,    # Pitchers
        WAR, player_type,
        PA_original, IP_original,        # Before adjustment
        age                              # From Chadwick
    """
    parser = argparse.ArgumentParser(description="Improve projections")
    parser.add_argument(
        "--method",
        choices=["simple", "weighted"],
        default="simple",
        help="Combination method (default: simple)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--skip-pt-adjustment",
        action="store_true",
        help="Skip playing time adjustment",
    )
    args = parser.parse_args()

    print("=== Projection Improvement Pipeline ===\n")

    # Step 1: Load historical actuals
    print("Step 1: Loading historical actuals...")
    historical = load_historical_actuals(MLB_STATS_DB, [2023, 2024])
    print(f"  Loaded {len(historical['hitters'])} hitter-seasons")
    print(f"  Loaded {len(historical['pitchers'])} pitcher-seasons\n")

    # Step 2: Load ages
    print("Step 2: Loading player ages...")
    ages = load_player_ages()
    print(f"  Loaded ages for {len(ages)} players\n")

    # Step 3: Load projections
    print("Step 3: Loading projection CSVs...")
    projections = load_all_projections(PROJECTIONS_DIR)
    systems = list(projections.keys())
    print(f"  Loaded {len(systems)} systems: {', '.join(systems)}\n")

    # Step 4: Combine
    print(f"Step 4: Combining projections ({args.method})...")
    if args.method == "simple":
        combined_hitters = simple_average(
            {s: projections[s]["hitters"] for s in systems},
            ["PA", "R", "HR", "RBI", "SB", "OPS", "WAR"],
        )
        combined_pitchers = simple_average(
            {s: projections[s]["pitchers"] for s in systems},
            ["IP", "W", "SV", "K", "ERA", "WHIP", "GS", "WAR"],
        )
    else:
        combined_hitters = weighted_average(
            {s: projections[s]["hitters"] for s in systems if s != "depthcharts"},
            {k: v for k, v in DEFAULT_WEIGHTS.items() if k in ["PA", "R", "HR", "RBI", "SB", "OPS"]},
        )
        combined_pitchers = weighted_average(
            {s: projections[s]["pitchers"] for s in systems if s != "depthcharts"},
            {k: v for k, v in DEFAULT_WEIGHTS.items() if k in ["IP", "W", "SV", "K", "ERA", "WHIP"]},
        )
    
    # Add player_type
    combined_hitters["player_type"] = "hitter"
    combined_pitchers["player_type"] = "pitcher"
    
    # Combine
    combined = pd.concat([combined_hitters, combined_pitchers], ignore_index=True)
    print(f"  Combined: {len(combined_hitters)} hitters, {len(combined_pitchers)} pitchers\n")

    # Step 5: Apply playing time adjustments
    if not args.skip_pt_adjustment:
        print("Step 5: Applying playing time adjustments...")
        combined = apply_playing_time_adjustments(combined, historical, ages)
    else:
        print("Step 5: Skipping playing time adjustments\n")

    # Step 6: Output
    print(f"Step 6: Writing output to {args.output}...")
    combined.to_csv(args.output, index=False)
    print(f"  Wrote {len(combined)} players\n")

    print("=== Done ===")


if __name__ == "__main__":
    main()
```

---

## Expected File Layout

```
data/
  projections/
    steamer_hitters.csv
    steamer_pitchers.csv
    zips_hitters.csv
    zips_pitchers.csv
    thebat_hitters.csv
    thebat_pitchers.csv
    depthcharts_hitters.csv      # Optional
    depthcharts_pitchers.csv     # Optional
  combined_projections.csv       # OUTPUT

../mlb_player_comps_dashboard/
  mlb_stats.db                   # Historical PA/IP data
```

---

## Validation Checklist

```python
# After loading historical actuals:
assert len(historical["hitters"]) > 500, "Expected 500+ hitter-seasons"
assert len(historical["pitchers"]) > 800, "Expected 800+ pitcher-seasons"

# After loading ages:
assert ages["age"].notna().mean() > 0.95, "Expected 95%+ players with age"
assert (ages["age"] > 15).all() and (ages["age"] < 50).all(), "Ages in valid range"

# After combining projections:
assert combined["MLBAMID"].notna().all(), "All players must have MLBAMID"
assert combined["PA"].notna().all() or combined["IP"].notna().all(), "All have PA or IP"

# After playing time adjustment:
hitters = combined[combined["player_type"] == "hitter"]
assert (hitters["PA_adjusted"] <= hitters["PA_original"] * 1.1).all(), "Adjustments shouldn't increase PA much"
```

---

## Usage

```bash
# Ensure projection CSVs are downloaded to data/projections/

# Run with simple averaging (recommended baseline)
python -m optimizer.projection_improvement.main

# Run with weighted averaging
python -m optimizer.projection_improvement.main --method weighted

# Skip playing time adjustment (if you want raw combination only)
python -m optimizer.projection_improvement.main --skip-pt-adjustment
```

The output `data/combined_projections.csv` can be used directly with the existing optimizer:

```python
# In your notebook or optimizer usage
projections = pd.read_csv("data/combined_projections.csv")
# ... use with existing optimizer functions
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project.optional-dependencies]
projection = [
    "pybaseball>=2.2.0",
]
```

Install with: `uv sync --extra projection`
