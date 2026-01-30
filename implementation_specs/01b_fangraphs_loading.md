# FanGraphs Projection Loading

## Overview

This document specifies loading FanGraphs Steamer projections. FanGraphs is the source of truth for player projections; Fantrax API provides roster/age data.

**Module:** `optimizer/data_loader.py` (projection loading section)

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style, fail-fast philosophy
- [01a_config.md](01a_config.md) — `compute_sgp_value()`, category constants

**Used by:**
- [01d_database.md](01d_database.md) — `load_projections()` output synced to database
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — `compute_team_totals()`, `compute_quality_scores()`
- [03_trade_engine.md](03_trade_engine.md) — `compute_team_totals()` for win probability

---

## Why FanGraphs?

| Data | FanGraphs | Fantrax |
|------|-----------|---------|
| **PA (Plate Appearances)** | ✅ | ❌ (only AB) |
| **WAR** | ✅ | ❌ |
| **MLBAMID** | ✅ | ❌ |
| **All player projections** | ✅ (~1,500) | ❌ (rostered only) |
| **GS (Games Started)** | ✅ | ❌ |

PA is critical because team OPS requires PA-weighted averaging, and SGP calculation requires PA weighting.

---

## CSV File Specifications

### Downloading

1. **Hitters:** https://www.fangraphs.com/projections.aspx?pos=all&stats=bat&type=steamer
2. **Pitchers:** https://www.fangraphs.com/projections.aspx?pos=all&stats=pit&type=steamer
3. Click "Export Data" → save to `data/` directory

### Hitter CSV Columns

| Column | Type | Notes |
|--------|------|-------|
| `Name` | str | Add `-H` suffix during load |
| `Team` | str | MLB team (blank = free agent) |
| `PA` | int | **Critical** for OPS weighting and SGP |
| `R`, `HR`, `RBI`, `SB` | int | Counting stats |
| `OPS` | float | **Use directly, do NOT recompute** |
| `WAR` | float | For reference/generic value |
| `MLBAMID` | int | For position lookup (optional) |

### Pitcher CSV Columns

| Column | Type | Notes |
|--------|------|-------|
| `Name` | str | Add `-P` suffix during load |
| `Team` | str | MLB team |
| `IP` | float | **Critical** for ERA/WHIP weighting and SGP |
| `W`, `SV` | int | Counting stats |
| `SO` | int | **Rename to `K`** during load |
| `ERA`, `WHIP` | float | Ratio stats |
| `GS` | int | For SP/RP determination: `SP if GS >= 3 else RP` |
| `WAR` | float | For reference |
| `MLBAMID` | int | For position lookup (optional) |

---

## Function Specifications

### Position Loading (Optional)

Positions can come from an external MLB stats database or default to DH/SP/RP.

```python
def load_positions_from_db(db_path: str) -> dict[int, str]:
    """
    Load player positions from SQLite database.
    
    Args:
        db_path: Path to mlb_stats.db (external database, OPTIONAL)
    
    Returns:
        Dict mapping MLBAMID (int) to position (str).
        Example: {592450: 'OF', 677951: 'SS', ...}
    
    Query: SELECT player_id, position FROM players
    
    Note:
        This is OPTIONAL. If database is unavailable, return empty dict.
        Positions will default to DH for hitters.
        Pitcher positions come from GS (not database).
    
    Print: "Loaded positions for {N} players from database"
    """
```

### Hitter Projections

```python
def load_hitter_projections(
    filepath: str,
    positions: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Load hitter projections from FanGraphs CSV.
    
    Args:
        filepath: Path to hitter projections CSV
        positions: Optional dict mapping MLBAMID to position
    
    Returns:
        DataFrame with columns:
            Name (with -H suffix), Team, Position, PA, R, HR, RBI, SB, OPS,
            WAR, player_type='hitter'
    
    Implementation:
        1. Read CSV
        2. Append '-H' to all names: df["Name"] = df["Name"] + "-H"
        3. Fill null Team with 'FA' (free agent)
        4. Position = positions.get(MLBAMID, 'DH') if positions provided, else 'DH'
        5. Add player_type = 'hitter'
        6. Drop duplicate names (keep first, print note)
        7. Assert no nulls in required columns
    
    Print: "Loaded {N} hitter projections ({M} with positions, {K} defaulted to DH)"
    """
```

### Pitcher Projections

```python
def load_pitcher_projections(filepath: str) -> pd.DataFrame:
    """
    Load pitcher projections from FanGraphs CSV.
    
    Args:
        filepath: Path to pitcher projections CSV
    
    Returns:
        DataFrame with columns:
            Name (with -P suffix), Team, Position, IP, W, SV, K, ERA, WHIP,
            GS, WAR, player_type='pitcher'
    
    Implementation:
        1. Read CSV
        2. Append '-P' to all names
        3. Rename SO → K
        4. Position = 'SP' if GS >= 3 else 'RP'
        5. Add player_type = 'pitcher'
        6. Drop duplicate names (keep first, print note)
        7. Assert no nulls in required columns
    
    Print: "Loaded {N} pitcher projections ({X} SP, {Y} RP)"
    """
```

### Combined Projections

```python
def load_projections(
    hitter_path: str,
    pitcher_path: str,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and combine all projections.
    
    Args:
        hitter_path: Path to hitter projections CSV
        pitcher_path: Path to pitcher projections CSV
        db_path: Optional path to external position database
    
    Returns DataFrame with columns:
        Name, Team, Position, player_type,
        PA, R, HR, RBI, SB, OPS,     # Hitter stats (0 for pitchers)
        IP, W, SV, K, ERA, WHIP, GS, # Pitcher stats (0 for hitters)
        WAR, SGP
    
    Implementation:
        1. Load positions from DB (if path provided)
        2. Load hitters and pitchers
        3. Align columns (fill missing with 0)
        4. Concatenate vertically
        5. Compute SGP for each player using compute_sgp_value()
    
    Two-way players appear as two rows:
        "Shohei Ohtani-H" with hitting stats
        "Shohei Ohtani-P" with pitching stats
    
    Print:
        "Combined projections: {N} players ({H} hitters, {P} pitchers)"
        "SGP range: {min:.1f} to {max:.1f} (mean: {mean:.1f})"
    """
```

---

## Team Total Computation

**Critical shared function** used by optimizer and trade engine.

```python
def compute_team_totals(
    player_names: Iterable[str],
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute a team's projected totals in all 10 scoring categories.
    
    Args:
        player_names: Iterable of player names (must include -H/-P suffix)
        projections: Combined projections DataFrame
    
    Returns:
        Dict mapping category to team total.
        Example: {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ...}
    
    Implementation:
        1. Filter projections to players on team
        2. Separate hitters (player_type == 'hitter') from pitchers
        3. Counting stats for hitters: sum(R, HR, RBI, SB)
        4. OPS: sum(PA * OPS) / sum(PA)  # PA-weighted average
        5. Counting stats for pitchers: sum(W, SV, K)
        6. ERA: sum(IP * ERA) / sum(IP)  # IP-weighted average
        7. WHIP: sum(IP * WHIP) / sum(IP)
    
    Assertions (with messages):
        - All player_names found in projections
        - sum(PA) > 0 for hitters
        - sum(IP) > 0 for pitchers
    """


def compute_all_opponent_totals(
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """
    Compute totals for all 6 opponents.
    
    Args:
        opponent_rosters: Dict mapping team_id to set of player names
        projections: Combined projections DataFrame
    
    Returns:
        Dict mapping team_id to their category totals.
        {1: {'R': 800, 'HR': 230, ...}, 2: {...}, ...}
    
    Implementation:
        Use tqdm for progress: "Computing opponent totals"
    """
```

---

## Quality Score Computation

Used for candidate filtering in the optimizer.

```python
def compute_quality_scores(projections: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a scalar quality score for each player.
    
    This is used for prefiltering candidates before MILP optimization.
    
    Methodology:
        Hitters: Mean z-score of (R, HR, RBI, SB, OPS)
                 All 5 hitting categories included.
        
        Pitchers: Mean z-score of (W, SV, K, -ERA, -WHIP)
                  Negate ERA/WHIP before z-scoring so lower = better = higher z.
    
    Returns:
        DataFrame with columns: Name, player_type, quality_score
    
    Implementation:
        1. Separate hitters and pitchers
        2. For each stat, compute z-score: (value - mean) / std
        3. For ERA/WHIP: negate before z-scoring
        4. Average z-scores across relevant categories
        5. Combine and return
    
    Note:
        OPS IS included for hitters (it's one of our scoring categories).
        Quality score is used for filtering, not final valuation.
    """
```

---

## Known Name Corrections

Some names differ between FanGraphs and Fantrax (accents, apostrophes, nicknames).

**Note:** These corrections are for FanGraphs names. Applied AFTER loading CSVs, BEFORE suffix is added.
The Fantrax corrections are in `01c_fantrax_api.md`.

```python
# FanGraphs name corrections (applied during FanGraphs CSV loading)
# Maps: FanGraphs spelling → canonical spelling
FANGRAPHS_NAME_CORRECTIONS = {
    "Logan OHoppe": "Logan O'Hoppe",
    # Add more as discovered from FanGraphs data
}
```

Corrections applied in `load_hitter_projections()` and `load_pitcher_projections()` BEFORE adding `-H`/`-P` suffix.

---

## Handling Missing Players

When loading rosters, some rostered players may not have FanGraphs projections (new call-ups, etc.).

```python
def validate_roster_against_projections(
    roster_names: set[str],
    projections: pd.DataFrame,
    roster_label: str = "roster",
) -> set[str]:
    """
    Validate roster names exist in projections. Drop missing with warning.
    
    Args:
        roster_names: Set of player names (with -H/-P suffix)
        projections: Combined projections DataFrame
        roster_label: Label for print messages (e.g., "my roster", "opponent 1")
    
    Returns:
        Set of valid roster names (those found in projections)
    
    Implementation:
        proj_names = set(projections["Name"])
        missing = roster_names - proj_names
        
        if len(missing) > 0:
            print(f"WARNING: Dropping {len(missing)} {roster_label} players not in projections:")
            for name in sorted(missing):
                print(f"  - {strip_name_suffix(name)}")
        
        return roster_names - missing
    """
```

---

## Validation Checklist

```python
# After loading projections:
assert all(projections['Name'].str.endswith(('-H', '-P'))), "All names must have suffix"
assert 'OPS' in projections.columns, "OPS must come from CSV, not be computed"
assert 'K' in projections.columns, "SO must be renamed to K"
assert set(projections[projections['player_type']=='pitcher']['Position'].unique()) <= {'SP', 'RP'}

# After computing team totals:
assert totals['OPS'] < 2.0, f"OPS looks wrong: {totals['OPS']} (should be ~0.7-0.9)"
assert totals['ERA'] > 0, f"ERA must be positive: {totals['ERA']}"
```
