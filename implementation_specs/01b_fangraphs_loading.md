# FanGraphs Projection Loading

## Overview

This document specifies loading FanGraphs Steamer projections. FanGraphs is the source of truth for player projections; Fantrax API provides roster/age data.

**Module:** `optimizer/data_loader.py` (projection loading section)

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style, fail-fast philosophy
- [01a_config.md](01a_config.md) — `compute_sgp_value()`, category constants, `SLOT_ELIGIBILITY`, `HITTING_SLOTS`, `PITCHING_SLOTS`

**Used by:**
- [01d_database.md](01d_database.md) — `load_projections()` output synced to database
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — `compute_team_totals()`, `compute_optimal_lineup()`, `compute_quality_scores()`
- [03_trade_engine.md](03_trade_engine.md) — `compute_team_totals()` for win probability (lineup-aware)

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
    hitter_path: str | None = None,
    pitcher_path: str | None = None,
    db_path: str | None = None,
) -> pd.DataFrame:
    """
    Load and combine all projections.
    
    Args:
        hitter_path: Path to hitter projections CSV (if None, uses config)
        pitcher_path: Path to pitcher projections CSV (if None, uses config)
        db_path: Optional path to external position database
    
    Returns DataFrame with columns:
        Name, Team, Position, player_type,
        PA, R, HR, RBI, SB, OPS,     # Hitter stats (0 for pitchers)
        IP, W, SV, K, ERA, WHIP, GS, # Pitcher stats (0 for hitters)
        WAR, SGP
    
    Implementation:
        1. If hitter_path/pitcher_path are None, load from config:
           from optimizer.config import HITTER_PROJ_PATH, PITCHER_PROJ_PATH
        2. Load positions from DB (if path provided)
        3. Load hitters and pitchers
        4. Align columns (fill missing with 0)
        5. Concatenate vertically
        6. Compute SGP for each player using compute_sgp_value()
    
    Two-way players appear as two rows:
        "Shohei Ohtani-H" with hitting stats
        "Shohei Ohtani-P" with pitching stats
    
    Print:
        "Combined projections: {N} players ({H} hitters, {P} pitchers)"
        "SGP range: {min:.1f} to {max:.1f} (mean: {mean:.1f})"
    """
```

---

## Optimal Lineup Computation

**Key function** that determines which players should start, given positional constraints. This is called by `compute_team_totals()` to ensure only starters contribute to category totals.

```python
def compute_optimal_lineup(
    roster_names: set[str],
    projections: pd.DataFrame,
) -> set[str]:
    """
    Given a roster, determine the optimal starting lineup.

    Solves an assignment problem: assign players to starting slots to maximize
    total SGP of starters. Multi-position players can fill any eligible slot.

    Args:
        roster_names: Set of player names on the roster (with -H/-P suffix)
        projections: Combined projections DataFrame (must have SGP, Position)

    Returns:
        Set of player names who should start.
        Size = sum(HITTING_SLOTS.values()) + sum(PITCHING_SLOTS.values())
        Bench players are NOT included in the returned set.

    Implementation:
        Uses a small MILP (solves in milliseconds):

        Decision variables:
            a[i,s] ∈ {0,1} = 1 if player i starts in slot s

        Objective:
            maximize Σ_{i,s} SGP[i] * a[i,s]

        Constraints:
            1. Each slot type filled to required count:
               Σ_i a[i,s] = HITTING_SLOTS[s]  for hitting slots
               Σ_i a[i,s] = PITCHING_SLOTS[s]  for pitching slots
               (e.g., OF needs 3 players, SP needs 5)

            2. Each player in at most one slot:
               Σ_s a[i,s] <= 1  for each player i

            3. Eligibility enforced by variable creation:
               Only create a[i,s] for (player, slot) pairs where player is eligible.
               Hitters can only fill hitting slots; pitchers can only fill pitching slots.
               Multi-position players (e.g., "SS,2B") can fill any eligible slot.

        If the roster cannot fill all slots, raise AssertionError with message
        indicating which slot cannot be filled. This can happen when evaluating
        hypothetical roster changes (e.g., dropping the only catcher).

    Assertions:
        - len(return) == sum(HITTING_SLOTS.values()) + sum(PITCHING_SLOTS.values())
        - All returned names are in roster_names
        - Returned lineup satisfies all slot constraints (each slot filled exactly once)

    Print: (none - this is called frequently)
    """
```

---

## Team Total Computation

**Critical shared function** used by optimizer and trade engine. Computes totals from **starters only** (not full roster).

```python
def compute_team_totals(
    roster_names: Iterable[str],
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute a team's projected totals in all 10 scoring categories.

    IMPORTANT: Only starters contribute to totals. This function first
    computes the optimal starting lineup, then sums stats from starters only.
    Bench players do not contribute.

    Args:
        roster_names: Iterable of player names on roster (must include -H/-P suffix)
        projections: Combined projections DataFrame

    Returns:
        Dict mapping category to team total (from starters only).
        Example: {'R': 823, 'HR': 245, ..., 'ERA': 3.85, ...}

    Implementation:
        1. Convert roster_names to set
        2. Call compute_optimal_lineup(roster_names, projections) → starters
        3. Filter projections to starters only (NOT full roster)
        4. Separate hitters (player_type == 'hitter') from pitchers
        5. Counting stats for hitters: sum(R, HR, RBI, SB)
        6. OPS: sum(PA * OPS) / sum(PA)  # PA-weighted average
        7. Counting stats for pitchers: sum(W, SV, K)
        8. ERA: sum(IP * ERA) / sum(IP)  # IP-weighted average
        9. WHIP: sum(IP * WHIP) / sum(IP)

    Assertions (with messages):
        - All roster_names found in projections
        - len(starters) == sum(HITTING_SLOTS.values()) + sum(PITCHING_SLOTS.values())
        - sum(PA) > 0 for starting hitters
        - sum(IP) > 0 for starting pitchers
    """


def compute_all_opponent_totals(
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """
    Compute totals for all 6 opponents.

    Each opponent's totals are computed from their optimal starting lineup,
    not their full roster. This ensures fair comparison since only starters
    generate fantasy points.

    Args:
        opponent_rosters: Dict mapping team_id to set of player names
        projections: Combined projections DataFrame

    Returns:
        Dict mapping team_id to their category totals (from starters only).
        {1: {'R': 800, 'HR': 230, ...}, 2: {...}, ...}

    Implementation:
        For each opponent:
            compute_team_totals(opponent_roster, projections)

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
