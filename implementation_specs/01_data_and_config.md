# Data Loading and Configuration

## Overview

This document specifies the data loading pipeline, league configuration, and shared utility functions used by both the free agent optimizer and trade engine.

**Module:** `optimizer/data_loader.py`

---

## League Configuration Constants

Define these at the top of the module:

```python
# === LEAGUE CONFIGURATION ===

# Scoring categories
HITTING_CATEGORIES = ['R', 'HR', 'RBI', 'SB', 'OPS']
PITCHING_CATEGORIES = ['W', 'SV', 'K', 'ERA', 'WHIP']
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Categories where lower is better
NEGATIVE_CATEGORIES = {'ERA', 'WHIP'}

# Ratio stats and their weighting denominators
RATIO_STATS = {
    'OPS': 'PA',   # Team OPS = sum(PA * OPS) / sum(PA)
    'ERA': 'IP',   # Team ERA = sum(IP * ERA) / sum(IP)
    'WHIP': 'IP',  # Team WHIP = sum(IP * WHIP) / sum(IP)
}

# Roster construction
ROSTER_SIZE = 26  # Active roster only

# Starting lineup slots (for positional validity checking)
HITTING_SLOTS = {
    'C': 1,
    '1B': 1,
    '2B': 1,
    'SS': 1,
    '3B': 1,
    'OF': 3,
    'UTIL': 1,
}  # Total: 9 hitting slots

PITCHING_SLOTS = {
    'SP': 5,
    'RP': 2,
}  # Total: 7 pitching slots

# Position eligibility: which player positions can fill which lineup slots
SLOT_ELIGIBILITY = {
    'C': {'C'},
    '1B': {'1B'},
    '2B': {'2B'},
    'SS': {'SS'},
    '3B': {'3B'},
    'OF': {'OF'},
    'UTIL': {'C', '1B', '2B', 'SS', '3B', 'OF', 'DH'},
    'SP': {'SP'},
    'RP': {'RP'},
}

# Roster composition bounds
MIN_HITTERS = 12
MAX_HITTERS = 16
MIN_PITCHERS = 10
MAX_PITCHERS = 14

# League structure
NUM_OPPONENTS = 6  # 7-team league (me + 6 opponents)
```

---

## Data File Specifications

### FanGraphs Projection CSVs

#### Hitter Projections: `fangraphs-steamer-projections-hitters.csv`

**Source:** Export from FanGraphs Steamer projections page.

**Columns we use:**
| Column | Type | Description |
|--------|------|-------------|
| `Name` | str | Player name |
| `Team` | str | MLB team abbreviation |
| `PA` | int | Plate appearances (OPS weighting) |
| `R` | int | Runs scored |
| `HR` | int | Home runs |
| `RBI` | int | Runs batted in |
| `SB` | int | Stolen bases |
| `OPS` | float | On-base plus slugging (**already computed, do NOT recompute**) |
| `MLBAMID` | int | MLB Advanced Media player ID (for position lookup) |

**Critical:** The OPS column exists in the FanGraphs export. Use it directly.

#### Pitcher Projections: `fangraphs-steamer-projections-pitchers.csv`

**Source:** Export from FanGraphs Steamer projections page.

**Columns we use:**
| Column | Type | Description |
|--------|------|-------------|
| `Name` | str | Player name |
| `Team` | str | MLB team abbreviation |
| `IP` | float | Innings pitched (ERA/WHIP weighting) |
| `W` | int | Wins |
| `SV` | int | Saves |
| `SO` | int | Strikeouts (**rename to `K` during load**) |
| `ERA` | float | Earned run average |
| `WHIP` | float | Walks + hits per inning pitched |
| `GS` | int | Games started (for SP/RP determination) |
| `MLBAMID` | int | MLB Advanced Media player ID |

### Position Database

**Path:** `../mlb_player_comps_dashboard/mlb_stats.db` (SQLite)

**Query:**
```sql
SELECT player_id, position FROM players WHERE player_id = <MLBAMID>
```

**Position values:** `C`, `1B`, `2B`, `SS`, `3B`, `OF`, `DH`

The database already consolidates LF/CF/RF into `OF`.

### Fantrax Roster Exports

Raw exports from Fantrax have a multi-section format:

```csv
"","Hitting"
"ID","Pos","Player","Team","Eligible","Status","Age",...
"*04my4*","C","Cal Raleigh","SEA","C","Act","29",...
...
"","Pitching"
"ID","Pos","Player","Team","Eligible","Status","Age",...
"*12345*","SP","Max Fried","ATL","SP","Act","30",...
```

**Key columns:**
- `Player`: Player name (may differ from FanGraphs spelling)
- `Team`: MLB team abbreviation
- `Status`: `Act` (active), `Res` (reserve/bench), `Min` (minors), `IR` (injured)

**Status mapping:**
```python
FANTRAX_STATUS_MAP = {
    "Act": "active",
    "Res": "active",   # Reserve still counts toward 26-man
    "Min": "prospect",
    "IR": "IR",
}
```

---

## Function Specifications

### Position Loading

```python
def load_positions_from_db(db_path: str) -> dict[int, str]:
    """
    Load player positions from SQLite database.
    
    Args:
        db_path: Path to mlb_stats.db
    
    Returns:
        Dict mapping MLBAMID (int) to position (str).
        Example: {592450: 'OF', 677951: 'SS', ...}
    
    Print:
        "Loaded positions for {N} players from database"
    """
```

### Projection Loading

```python
def load_hitter_projections(filepath: str, positions: dict[int, str]) -> pd.DataFrame:
    """
    Load hitter projections from FanGraphs CSV.
    
    Args:
        filepath: Path to hitter projections CSV
        positions: Dict mapping MLBAMID to position
    
    Returns:
        DataFrame with columns:
            Name (with -H suffix), Team, Position, PA, R, HR, RBI, SB, OPS,
            player_type='hitter'
    
    Implementation:
        1. Read CSV
        2. Append '-H' to all names
        3. Join position from positions dict using MLBAMID
        4. Players not in positions dict get Position='DH'
        5. Add player_type='hitter'
        6. Handle duplicate names (keep first occurrence, print note)
        7. Assert no nulls in required columns
    
    Print:
        "Loaded {N} hitter projections ({M} with positions from DB, {K} defaulted to DH)"
    """


def load_pitcher_projections(filepath: str) -> pd.DataFrame:
    """
    Load pitcher projections from FanGraphs CSV.
    
    Args:
        filepath: Path to pitcher projections CSV
    
    Returns:
        DataFrame with columns:
            Name (with -P suffix), Team, Position, IP, W, SV, K, ERA, WHIP,
            player_type='pitcher'
    
    Implementation:
        1. Read CSV
        2. Append '-P' to all names
        3. Rename SO -> K
        4. Position = 'SP' if GS >= 3 else 'RP'
        5. Add player_type='pitcher'
        6. Handle duplicate names (keep first occurrence, print note)
        7. Assert no nulls in required columns
    
    Print:
        "Loaded {N} pitcher projections ({X} SP, {Y} RP)"
    """


def load_projections(hitter_path: str, pitcher_path: str, db_path: str) -> pd.DataFrame:
    """
    Load and combine hitter and pitcher projections.
    
    Returns:
        Combined DataFrame with columns:
            Name, Team, Position, player_type,
            PA, R, HR, RBI, SB, OPS,    (hitting - 0 for pitchers)
            IP, W, SV, K, ERA, WHIP     (pitching - 0 for hitters)
        
        Two-way players (e.g., Ohtani) appear as separate rows:
        - "Shohei Ohtani-H" with hitting stats, pitching cols = 0
        - "Shohei Ohtani-P" with pitching stats, hitting cols = 0
    
    Implementation:
        1. positions = load_positions_from_db(db_path)
        2. hitters = load_hitter_projections(hitter_path, positions)
        3. pitchers = load_pitcher_projections(pitcher_path)
        4. Align columns (fill missing with 0)
        5. Concatenate vertically
        6. Set Name as index for fast lookup (optional)
    
    Print:
        "Combined projections: {N} total players ({H} hitters, {P} pitchers)"
    """
```

### Fantrax Roster Conversion

```python
def parse_fantrax_roster(filepath: str) -> pd.DataFrame:
    """
    Parse a single Fantrax roster CSV file.
    
    Handles the multi-section format (separate Hitting/Pitching sections).
    
    Returns:
        DataFrame with columns: name, team, player_type, status
        - name: Player name with -H/-P suffix based on section
        - team: MLB team abbreviation
        - player_type: 'hitter' or 'pitcher' (from section)
        - status: 'active', 'prospect', or 'IR'
    """


def convert_fantrax_rosters_from_dir(
    raw_rosters_dir: str,
    my_team_filename: str = 'my_team.csv',
    output_dir: str | None = None,
) -> tuple[str, str]:
    """
    Convert all Fantrax rosters in a directory to pipeline format.
    
    Args:
        raw_rosters_dir: Directory containing Fantrax CSV files
        my_team_filename: Filename of my team's roster
        output_dir: Where to write output files (default: parent of raw_rosters_dir)
    
    Returns:
        Tuple of (my_roster_path, opponent_rosters_path)
    
    Implementation:
        1. Find all CSV files in raw_rosters_dir
        2. Identify my_team file vs opponent files
        3. Assign team_ids 1-6 to opponents (alphabetical order)
        4. Parse each file with parse_fantrax_roster()
        5. Write my-roster.csv and opponent-rosters.csv
    
    Print:
        "Converting {N} roster files from {dir}..."
        "  My team: {filename}"
        "  Opponents: {list of filenames}"
        "Wrote my-roster.csv ({A} active, {I} inactive)"
        "Wrote opponent-rosters.csv ({N} total players across 6 teams)"
    """
```

### Name Matching and Correction

```python
def normalize_name(name: str) -> str:
    """
    Normalize player name for fuzzy comparison.
    
    CRITICAL: Preserves -H/-P suffix!
    
    Handles:
        - Accented characters (Rodríguez → rodriguez)
        - Apostrophe variations (O'Hoppe, OHoppe → o'hoppe)
        - Name suffixes like Jr., Sr. (removed)
    
    Example:
        "Julio Rodríguez-H" → "julio rodriguez-h"
        "Ronald Acuña Jr.-H" → "ronald acuna-h"
    """


def find_name_mismatches(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
) -> pd.DataFrame:
    """
    Find roster names that don't match projections and suggest corrections.
    
    Returns:
        DataFrame with columns: roster_name, suggested_match, similarity_score
        Only includes names without exact matches.
    
    Implementation:
        Uses normalize_name() for comparison (preserving -H/-P suffix).
        Computes string similarity for suggestions.
    """


def apply_name_corrections(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
    min_similarity: float = 0.9,
) -> None:
    """
    Apply automatic name corrections to a roster file.
    
    High-confidence matches (>= min_similarity) are applied automatically.
    Lower-confidence matches are printed for manual review.
    
    Known corrections (hardcoded):
        "Logan OHoppe-H" → "Logan O'Hoppe-H"
        "Leodalis De Vries-H" → "Leo De Vries-H"
        "Leodalis De Vries-P" → "Leo De Vries-P"
    
    Print:
        "Applied {N} automatic corrections to {filepath}"
        "*** MANUAL REVIEW NEEDED ({M} names) ***" (if any low-confidence)
    """
```

### Roster Loading (Post-Correction)

```python
def load_my_roster(filepath: str, projections: pd.DataFrame) -> tuple[set[str], set[str]]:
    """
    Load my roster from CSV (assumes name corrections already applied).
    
    Returns:
        Tuple of (active_names, inactive_names)
        - active_names: Set of player names on active roster
        - inactive_names: Set of player names on prospect/IR slots
    
    Assertions:
        All active_names must exist in projections['Name'].
        If not, crash with list of unmatched names.
    
    Print:
        "My roster: {A} active, {I} inactive (prospects/IR)"
    """


def load_opponent_rosters(filepath: str, projections: pd.DataFrame) -> dict[int, set[str]]:
    """
    Load opponent rosters from CSV.
    
    Returns:
        Dict mapping team_id to set of active player names.
        Example: {1: {'Mike Trout-H', 'Mookie Betts-H', ...}, 2: {...}, ...}
    
    Assertions:
        - team_ids are exactly {1, 2, 3, 4, 5, 6}
        - All player names exist in projections
        - No player appears on multiple teams
    
    Print:
        "Loaded 6 opponent rosters: {N1}, {N2}, {N3}, {N4}, {N5}, {N6} active players each"
    """


def load_all_data(
    hitter_proj_path: str,
    pitcher_proj_path: str,
    my_roster_path: str,
    opponent_rosters_path: str,
    db_path: str = "../mlb_player_comps_dashboard/mlb_stats.db",
) -> tuple[pd.DataFrame, set[str], dict[int, set[str]]]:
    """
    Load all data files and validate. Main entry point for data loading.
    
    Returns:
        Tuple of (projections, my_active_roster, opponent_rosters)
    
    This function calls individual load functions in sequence.
    All validation happens in those functions.
    """
```

---

## Team Total Computation

This is a critical shared utility used by both free agent optimizer and trade engine.

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
        1. Filter projections to players in player_names
        2. Separate into hitters and pitchers by player_type
        
        For counting stats (R, HR, RBI, SB, W, SV, K):
            total = column.sum()
        
        For ratio stats:
            OPS = sum(PA * OPS) / sum(PA)  [over hitters]
            ERA = sum(IP * ERA) / sum(IP)  [over pitchers]
            WHIP = sum(IP * WHIP) / sum(IP)  [over pitchers]
    
    Assertions:
        - sum(PA) > 0 for hitters (OPS undefined otherwise)
        - sum(IP) > 0 for pitchers (ERA/WHIP undefined otherwise)
        - All player_names found in projections
    """


def compute_all_opponent_totals(
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """
    Compute totals for all 6 opponents.
    
    Returns:
        Dict mapping team_id to their category totals.
        {1: {'R': 800, 'HR': 230, ...}, 2: {...}, ...}
    
    Uses tqdm progress bar.
    """
```

---

## Quality Score Computation

Used by the free agent optimizer for candidate filtering.

```python
def compute_quality_scores(projections: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a scalar quality score for each player.
    
    Methodology:
        Hitters: Mean z-score of (R, HR, RBI, SB)
                 Do NOT include OPS (ratio stat)
        
        Pitchers: Mean z-score of (W, SV, K, -ERA, -WHIP)
                  Negate ERA/WHIP before z-scoring
    
    Returns:
        DataFrame with columns: Name, player_type, quality_score
    
    Assertions:
        Standard deviation > 0 for each stat being z-scored
    """
```

---

## Projection Uncertainty Estimation

Used by the trade engine for probabilistic modeling.

```python
def estimate_projection_uncertainty(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> dict[str, float]:
    """
    Estimate the standard deviation of team totals for each category.
    
    Args:
        my_totals: My team's category totals.
        opponent_totals: Dict mapping opponent_id to their category totals.
    
    Returns:
        Dict mapping category to standard deviation.
        Example: {'R': 45.2, 'HR': 18.3, ..., 'ERA': 0.35, ...}
    
    Implementation:
        For each category:
            1. Collect all 7 team values (my_totals[c] + opponent_totals[1..6][c])
            2. Compute σ_c = std(values)
            3. If σ_c < MIN_STD (0.001), use MIN_STD to avoid division by zero
    
    Note:
        This empirical approach captures real league variation.
        The σ_c values are used in the Rosenof (2025) formulation
        for normalizing matchup gaps.
    """
```

---

## Validation Checklist

Before data loading is complete, verify:

- [ ] All player names have -H or -P suffix
- [ ] OPS used directly from CSV (not recomputed)
- [ ] SO renamed to K for pitchers
- [ ] Pitcher positions derived from GS (SP if GS >= 3)
- [ ] Hitter positions from database (DH fallback)
- [ ] Duplicate names in projections handled (keep first)
- [ ] All roster names validated against projections
- [ ] sum(PA) > 0 and sum(IP) > 0 asserted for all teams
- [ ] Ratio stats computed as weighted averages
