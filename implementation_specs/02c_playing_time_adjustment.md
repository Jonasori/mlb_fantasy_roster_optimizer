# Playing Time Adjustment Module

## Overview

This standalone module adjusts ATC projections for playing time bias. All projection systems (including ATC) systematically overproject playing time. This module corrects that using historical data.

**Module:** `optimizer/playing_time/`  
**Input:** `data/fangraphs-atc-projections-{hitters,pitchers}.csv`  
**Output:** `data/adjusted_projections.csv`

---

## Cross-References

**Depends on:**
- [01f_mlb_stats_api.md](01f_mlb_stats_api.md) — ages loaded from `data/optimizer.db` (populated by data pipeline)
- External: `../mlb_player_comps_dashboard/mlb_stats.db` — historical PA/IP data

**Data flow:**
```
ATC CSVs + mlb_stats.db + optimizer.db → playing_time module → adjusted_projections.csv
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

Key columns:
- Hitters: `Name`, `MLBAMID`, `Team`, `PA`, `R`, `HR`, `RBI`, `SB`, `OPS`, `WAR`
- Pitchers: `Name`, `MLBAMID`, `Team`, `IP`, `W`, `SV`, `SO`, `ERA`, `WHIP`, `GS`, `WAR`

### 2. Historical Actual PA/IP

**Source:** `../mlb_player_comps_dashboard/mlb_stats.db`

```sql
-- Hitter season totals
SELECT 
    p.player_id as mlbam_id,
    p.name,
    g.season,
    SUM(g.pa) as pa
FROM players p
JOIN game_logs g ON p.player_id = g.player_id
WHERE g.season IN (2024, 2023)
GROUP BY p.player_id, g.season;

-- Pitcher season totals
SELECT 
    p.player_id as mlbam_id,
    p.name,
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

**Source:** `data/optimizer.db` (populated by MLB Stats API via [01f_mlb_stats_api.md](01f_mlb_stats_api.md))

The database `players` table has an `age` column with 100% coverage for projected players. The data pipeline fetches ages from the MLB Stats API and syncs them before this module runs.

```sql
SELECT MLBAMID, age FROM players WHERE age IS NOT NULL
```

**Note:** This module reads ages from the database. If ages are missing (e.g., `skip_mlb_api=True` during data refresh), the module skips the age adjustment factor and still applies historical and talent adjustments.

---

## Module Structure

```
optimizer/playing_time/
    __init__.py
    config.py           # Constants
    load.py             # Load ATC, historical, ages
    adjust.py           # Adjustment algorithm
    main.py             # CLI entry point
```

**Key constraint:** This module has NO dependencies on the existing optimizer package. It reads files and outputs a CSV.

---

## Function Specifications

### config.py

```python
"""Configuration constants for playing time adjustment."""

from pathlib import Path

# Paths
MLB_STATS_DB = Path("../mlb_player_comps_dashboard/mlb_stats.db")
OPTIMIZER_DB = Path("data/optimizer.db")
ATC_HITTERS = Path("data/fangraphs-atc-projections-hitters.csv")
ATC_PITCHERS = Path("data/fangraphs-atc-projections-pitchers.csv")
OUTPUT_PATH = Path("data/adjusted_projections.csv")

# Playing time baselines
FULL_SEASON_PA = 600          # Full-time hitter
FULL_SEASON_IP_SP = 180       # Full-time starter
FULL_SEASON_IP_RP = 70        # Full-time reliever

# Age penalty thresholds
AGE_PENALTY_START_HITTER = 32
AGE_PENALTY_START_PITCHER = 33
AGE_PENALTY_PER_YEAR = 0.05   # 5% reduction per year over threshold

# Talent penalty
TALENT_PENALTY_PERCENTILE = 0.25  # Bottom quartile
TALENT_PENALTY_FACTOR = 0.85      # 15% reduction

# Blend ratio (from Zimmerman's research)
PROJECTION_WEIGHT = 0.65
ADJUSTMENT_WEIGHT = 0.35
```

### load.py

```python
"""Load data from external sources."""

import sqlite3
from pathlib import Path

import pandas as pd


def load_atc_hitters(filepath: Path) -> pd.DataFrame:
    """
    Load ATC hitter projections.
    
    Returns:
        DataFrame with columns: Name, MLBAMID, Team, PA, R, HR, RBI, SB, OPS, WAR
        Plus player_type = 'hitter'
    
    Implementation:
        1. Read CSV
        2. Select required columns
        3. Add player_type = 'hitter'
        4. Assert MLBAMID has no nulls
        5. Assert PA > 0 for all rows
    """


def load_atc_pitchers(filepath: Path) -> pd.DataFrame:
    """
    Load ATC pitcher projections.
    
    Returns:
        DataFrame with columns: Name, MLBAMID, Team, IP, W, SV, K, ERA, WHIP, GS, WAR
        Plus player_type = 'pitcher', Position = 'SP' or 'RP'
    
    Implementation:
        1. Read CSV
        2. Rename SO -> K
        3. Add Position = 'SP' if GS >= 3 else 'RP'
        4. Add player_type = 'pitcher'
        5. Assert MLBAMID has no nulls
        6. Assert IP > 0 for all rows
    """


def load_atc_projections() -> pd.DataFrame:
    """
    Load and combine ATC hitter and pitcher projections.
    
    Returns:
        Combined DataFrame with all columns aligned.
        Hitters have IP=0, Pitchers have PA=0.
    
    Print:
        "Loaded ATC projections: {N} hitters, {M} pitchers"
    """


def load_historical_actuals(db_path: Path, seasons: list[int]) -> dict[str, pd.DataFrame]:
    """
    Load historical PA/IP from mlb_stats.db.
    
    Args:
        db_path: Path to mlb_stats.db
        seasons: List of seasons to load (e.g., [2023, 2024])
    
    Returns:
        {
            "hitters": DataFrame [mlbam_id, season, pa],
            "pitchers": DataFrame [mlbam_id, season, ip, gs]
        }
    
    Implementation:
        1. Connect to SQLite
        2. Query game_logs aggregated by player_id, season
        3. Query pitcher_game_logs aggregated by player_id, season
        4. Close connection
    
    Print:
        "Loaded historical: {N} hitter-seasons, {M} pitcher-seasons"
    """


def load_ages(db_path: Path) -> pd.DataFrame:
    """
    Load player ages from optimizer database.
    
    Args:
        db_path: Path to data/optimizer.db
    
    Returns:
        DataFrame with columns [mlbam_id, age]
        Only players with non-null age and MLBAMID.
    
    Implementation:
        conn = sqlite3.connect(db_path)
        query = '''
            SELECT MLBAMID as mlbam_id, age 
            FROM players 
            WHERE age IS NOT NULL AND MLBAMID IS NOT NULL
        '''
        df = pd.read_sql(query, conn)
        conn.close()
        return df
    
    Note:
        Ages are populated by the data pipeline via MLB Stats API.
        See 01f_mlb_stats_api.md for details.
        
        If database doesn't exist or has no ages, returns empty DataFrame
        and the adjustment will skip the age factor.
    
    Print:
        "Loaded ages for {N} players from database"
        or "WARNING: No ages found in database. Age adjustment will be skipped."
    """
```

### adjust.py

```python
"""Playing time adjustment algorithm."""

import pandas as pd
import numpy as np

from .config import (
    AGE_PENALTY_PER_YEAR,
    AGE_PENALTY_START_HITTER,
    AGE_PENALTY_START_PITCHER,
    ADJUSTMENT_WEIGHT,
    FULL_SEASON_IP_RP,
    FULL_SEASON_IP_SP,
    FULL_SEASON_PA,
    PROJECTION_WEIGHT,
    TALENT_PENALTY_FACTOR,
    TALENT_PENALTY_PERCENTILE,
)


def compute_historical_ratio(
    actual_current: float | None,
    actual_prior: float | None,
    full_season: float,
) -> float:
    """
    Compute ratio of actual to expected playing time.
    
    Args:
        actual_current: Actual PA/IP from most recent season (e.g., 2024)
        actual_prior: Actual PA/IP from prior season (e.g., 2023)
        full_season: Expected full-season PA/IP (600 for hitters, 180/70 for pitchers)
    
    Returns:
        Float in range [0.0, 1.0].
        
    Implementation:
        - If both seasons available: avg = (current + prior) / 2
        - If only one season: use that value
        - If neither: return 1.0 (trust projection for rookies)
        - Ratio = min(1.0, avg / full_season)
        
    The cap at 1.0 prevents bonus for historically high PT.
    """


def compute_age_factor(
    age: int | None,
    is_pitcher: bool,
) -> float:
    """
    Compute age-based adjustment factor.
    
    Args:
        age: Player age (or None if unknown)
        is_pitcher: True for pitchers, False for hitters
    
    Returns:
        Float in range [0.5, 1.0].
        
    Implementation:
        threshold = 33 if is_pitcher else 32
        
        if age is None or age < threshold:
            return 1.0
        
        years_over = age - threshold + 1
        return max(0.5, 1.0 - 0.05 * years_over)
        
    Examples:
        - Age 30 hitter → 1.0
        - Age 35 hitter → 1.0 - 0.05*4 = 0.80
        - Age 38 pitcher → 1.0 - 0.05*6 = 0.70
    """


def compute_talent_factor(
    player_value: float,
    percentile_25: float,
) -> float:
    """
    Compute talent-based adjustment factor.
    
    Args:
        player_value: Player's value metric (WAR or computed SGP)
        percentile_25: 25th percentile value for player type
    
    Returns:
        1.0 if player_value >= percentile_25
        0.85 if player_value < percentile_25 (15% penalty for weak players)
    """


def adjust_playing_time(
    projected: float,
    historical_ratio: float,
    age_factor: float,
    talent_factor: float,
) -> float:
    """
    Apply three-factor adjustment and blend with projection.
    
    Formula:
        adjusted = projected * historical_ratio * age_factor * talent_factor
        blended = 0.65 * projected + 0.35 * adjusted
    
    Returns:
        Blended playing time value (float)
    """


def apply_adjustments(
    projections: pd.DataFrame,
    historical: dict[str, pd.DataFrame],
    ages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply playing time adjustments to all players.
    
    Args:
        projections: DataFrame from load_atc_projections()
        historical: Dict from load_historical_actuals()
        ages: DataFrame from load_ages()
    
    Returns:
        projections with:
        - PA_original, IP_original columns (preserved originals)
        - PA, IP columns (adjusted values)
        - age column (joined from ages df)
    
    Implementation:
        1. Pivot historical to wide format:
           - hitters: mlbam_id, pa_2024, pa_2023
           - pitchers: mlbam_id, ip_2024, ip_2023, gs_2024
        
        2. Join historical to projections on MLBAMID (left join)
        3. Join ages to projections on MLBAMID (left join)
        
        4. Compute value percentiles:
           - hitter_war_25 = hitters['WAR'].quantile(0.25)
           - pitcher_war_25 = pitchers['WAR'].quantile(0.25)
        
        5. For each player, compute:
           - historical_ratio
           - age_factor
           - talent_factor
           - adjusted PT
        
        6. Store originals in PA_original, IP_original
        7. Replace PA, IP with adjusted values
    
    Print summary:
        "Playing time adjustments:"
        "  Hitters: {N} adjusted, avg change = {X:+.1f} PA"
        "  Pitchers: {M} adjusted, avg change = {Y:+.1f} IP"
        "  {K} players missing historical data (rookies)"
        "  {J} players missing age data"
    """
```

### main.py

```python
"""CLI entry point for playing time adjustment."""

import argparse
from pathlib import Path

from .config import ATC_HITTERS, ATC_PITCHERS, MLB_STATS_DB, OPTIMIZER_DB, OUTPUT_PATH
from .load import load_ages, load_atc_projections, load_historical_actuals
from .adjust import apply_adjustments


def main():
    """
    Adjust ATC projections for playing time bias.
    
    Usage:
        python -m optimizer.playing_time.main
        python -m optimizer.playing_time.main --output custom.csv
        python -m optimizer.playing_time.main --no-age-adjustment
    """
    parser = argparse.ArgumentParser(description="Adjust ATC projections for playing time")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-age-adjustment",
        action="store_true",
        help="Skip age-based adjustment",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=OPTIMIZER_DB,
        help=f"Path to optimizer database (default: {OPTIMIZER_DB})",
    )
    args = parser.parse_args()

    print("=== Playing Time Adjustment ===\n")

    # Step 1: Load ATC projections
    print("Step 1: Loading ATC projections...")
    projections = load_atc_projections()

    # Step 2: Load historical actuals
    print("\nStep 2: Loading historical actuals...")
    historical = load_historical_actuals(MLB_STATS_DB, [2024, 2023])

    # Step 3: Load ages from database
    print("\nStep 3: Loading ages from database...")
    if args.no_age_adjustment:
        print("  Skipping (--no-age-adjustment)")
        ages = pd.DataFrame(columns=["mlbam_id", "age"])
    else:
        ages = load_ages(args.db)

    # Step 4: Apply adjustments
    print("\nStep 4: Applying adjustments...")
    adjusted = apply_adjustments(projections, historical, ages)

    # Step 5: Write output
    print(f"\nStep 5: Writing to {args.output}...")
    adjusted.to_csv(args.output, index=False)
    print(f"  Wrote {len(adjusted)} players")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
```

---

## Output Format

The output CSV matches the format expected by the existing `load_projections()`:

```
Name,Team,MLBAMID,PA,R,HR,RBI,SB,OPS,IP,W,SV,K,ERA,WHIP,GS,WAR,player_type,Position,age,PA_original,IP_original
Aaron Judge,NYY,592450,612,...  # PA reduced from 646
...
```

Key columns:
- `PA`, `IP` — Adjusted values
- `PA_original`, `IP_original` — Original ATC values (for comparison)
- `age` — From MLB Stats API (100% coverage)

---

## Validation Checklist

```python
# After loading ATC:
assert projections["MLBAMID"].notna().all(), "All players must have MLBAMID"
assert (projections["PA"] > 0).any(), "Some hitters must have PA > 0"
assert (projections["IP"] > 0).any(), "Some pitchers must have IP > 0"

# After loading historical:
assert len(historical["hitters"]) > 500, "Expected 500+ hitter-seasons"
assert len(historical["pitchers"]) > 800, "Expected 800+ pitcher-seasons"

# After adjustment:
hitters = adjusted[adjusted["player_type"] == "hitter"]
pitchers = adjusted[adjusted["player_type"] == "pitcher"]

# Adjustments should generally reduce PT, not increase it
assert hitters["PA"].sum() <= hitters["PA_original"].sum() * 1.05, \
    "Total PA should not increase significantly"
assert pitchers["IP"].sum() <= pitchers["IP_original"].sum() * 1.05, \
    "Total IP should not increase significantly"

# Spot check: older players should have reduced PT
old_hitters = hitters[hitters["age"] >= 35]
if len(old_hitters) > 0:
    assert (old_hitters["PA"] <= old_hitters["PA_original"]).mean() > 0.8, \
        "Most old hitters should have reduced PA"
```

---

## Usage

```bash
# Basic usage (fetches ages from MLB Stats API)
python -m optimizer.playing_time.main

# Custom output path
python -m optimizer.playing_time.main --output data/my_adjusted.csv

# Skip age adjustment (faster, for testing)
python -m optimizer.playing_time.main --no-age-adjustment
```

---

## Age Data Source

Ages are loaded from `data/optimizer.db`, which is populated by the data pipeline.

See [01f_mlb_stats_api.md](01f_mlb_stats_api.md) for how ages are fetched from the MLB Stats API and synced to the database.

**Coverage:** 100% for all ATC projection players (verified by test script).

---

## Dependencies

This module uses only:
- `pandas` — DataFrame operations
- `sqlite3` — Database access (stdlib)
- `numpy` — Numeric operations

No external API calls. All data comes from files and databases populated by the data pipeline.
