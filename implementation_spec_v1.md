# Dynasty Roto Roster Optimizer: Implementation Spec

## Overview

Build a roster optimization library for a dynasty fantasy baseball league with rotisserie scoring. The code runs in a marimo notebook, with all logic in two importable Python files. The optimizer uses Mixed-Integer Linear Programming (MILP) to find the roster that maximizes wins against known opponents across scoring categories.

**Key design decision:** All rostered players' stats count toward team totals (not just starters). The starting lineup slots exist only to enforce positional requirements—a valid roster must be able to field a legal starting lineup. Bench players contribute stats equally to starters.

---

## Constraints on Implementation Style

**Mandatory:**
- No classes. No OOP. Only functions and plain data structures (dicts, lists, numpy arrays, pandas DataFrames).
- Fail fast: use `assert` liberally. If something is wrong, crash immediately with a clear message. Never write try/except for error handling. Never write fallback logic.
- Use `print()` for status updates. Use `tqdm` for progress bars on any loop that might take more than a few seconds.
- All configuration is passed as function arguments. No global state.

**File structure:**
```
mlb_fantasy_roster_optimizer/
├── optimizer/
│   ├── __init__.py           # Empty or minimal exports
│   ├── roster_optimizer.py   # All optimization logic
│   └── visualizations.py     # All plotting functions
├── data/
│   ├── raw_rosters/          # Raw Fantrax CSV exports (input)
│   │   ├── my_team.csv
│   │   ├── team_aidan.csv
│   │   ├── team_oliver.csv
│   │   └── ... (6 opponent teams)
│   ├── fangraphs-steamer-projections-hitters.csv
│   ├── fangraphs-steamer-projections-pitchers.csv
│   ├── my-roster.csv         # Generated from raw_rosters/
│   └── opponent-rosters.csv  # Generated from raw_rosters/
├── notebook.py               # Marimo notebook (at project root)
└── pyproject.toml
```

The marimo notebook lives at the project root and imports from the `optimizer` package.

---

## Package Management

Use `uv` for package management. Create `pyproject.toml` at the project root:

```toml
[project]
name = "roster-optimizer"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "pulp>=2.8",
    "highspy>=1.5",
    "matplotlib>=3.7",
    "tqdm>=4.65",
    "marimo>=0.8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["optimizer"]
```

Install with: `uv sync`

Run the notebook with: `marimo edit notebook.py`

---

## League Settings

These are specific to this league. Define as module-level constants at the top of `roster_optimizer.py`:

```python
# === LEAGUE CONFIGURATION ===

# Scoring categories
HITTING_CATEGORIES = ['R', 'HR', 'RBI', 'SB', 'OPS']
PITCHING_CATEGORIES = ['W', 'SV', 'K', 'ERA', 'WHIP']
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Categories where lower is better
NEGATIVE_CATEGORIES = {'ERA', 'WHIP'}

# Ratio stats and their weighting denominators
# Key = category, Value = column used for weighting
RATIO_STATS = {
    'OPS': 'PA',   # Team OPS = sum(PA_i * OPS_i) / sum(PA_i)
    'ERA': 'IP',   # Team ERA = sum(IP_i * ERA_i) / sum(IP_i)
    'WHIP': 'IP',  # Team WHIP = sum(IP_i * WHIP_i) / sum(IP_i)
}

# Roster construction
ROSTER_SIZE = 26  # Active roster only (excludes 15 prospect slots and 1 IR slot)

# Starting lineup slots
# Maps slot type -> number of slots of that type
# NOTE: These slots enforce positional validity only. ALL rostered players
# contribute stats, not just starters. The slot constraints ensure the roster
# CAN field a valid starting lineup.
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

# Total starting slots: 16
# Remaining roster spots (26 - 16 = 10) are bench
# Bench players' stats count equally toward team totals

# Position eligibility: which player positions can fill which lineup slots
# Note: Database already consolidates LF/CF/RF into 'OF', so we only see 'OF' in practice.
# DH players can ONLY fill UTIL slots (not position-specific slots).
SLOT_ELIGIBILITY = {
    'C': {'C'},
    '1B': {'1B'},
    '2B': {'2B'},
    'SS': {'SS'},
    '3B': {'3B'},
    'OF': {'OF'},        # Database already normalizes LF/CF/RF to OF
    'UTIL': {'C', '1B', '2B', 'SS', '3B', 'OF', 'DH'},  # Any hitter position
    'SP': {'SP'},
    'RP': {'RP'},
}

# League structure
NUM_OPPONENTS = 6  # Hardcoded: 7-team league (me + 6 opponents)

# Roster composition bounds (for full 26-man roster, not just starters)
MIN_HITTERS = 12
MAX_HITTERS = 16
MIN_PITCHERS = 10
MAX_PITCHERS = 14
```

---

## Data Files

### FanGraphs Projection CSVs

Download from FanGraphs Steamer projections.

#### `fangraphs-steamer-projections-hitters.csv`

Exported from FanGraphs Steamer hitter projections. Actual columns in file (partial list):

```
Name,Team,G,PA,AB,H,1B,2B,3B,HR,R,RBI,BB,...,OBP,SLG,...,OPS,...,MLBAMID
```

**Columns we use:**
- `Name`: Player name (used for matching to rosters)
- `Team`: MLB team abbreviation
- `PA`: Plate appearances (used for OPS weighting)
- `R`: Runs scored
- `HR`: Home runs
- `RBI`: Runs batted in
- `SB`: Stolen bases
- `OBP`: On-base percentage
- `SLG`: Slugging percentage
- `OPS`: On-base plus slugging (**already computed in file, do NOT recompute**)
- `MLBAMID`: MLB Advanced Media player ID (**used for joining position data**)

**Position data:** FanGraphs export doesn't include position. Position is loaded from the SQLite database at `../mlb_player_comps_dashboard/mlb_stats.db`:

```sql
SELECT player_id, position FROM players WHERE player_id = <MLBAMID>
```

The database has single positions: `C`, `1B`, `2B`, `SS`, `3B`, `OF`, `DH`. If a hitter's `MLBAMID` is not found in the database, assign `Position = 'DH'` as a fallback (DH players can only fill UTIL slots).

#### `fangraphs-steamer-projections-pitchers.csv`

Exported from FanGraphs Steamer pitcher projections. Actual columns (partial list):

```
Name,Team,W,L,QS,ERA,G,GS,SV,HLD,BS,IP,...,SO,...,WHIP,...,MLBAMID
```

**Columns we use:**
- `Name`: Player name
- `Team`: MLB team abbreviation
- `IP`: Innings pitched (used for ERA/WHIP weighting)
- `W`: Wins
- `SV`: Saves
- `SO`: Strikeouts (**rename to `K` during load**)
- `ERA`: Earned run average
- `WHIP`: Walks + hits per inning pitched
- `GS`: Games started (used for SP/RP determination)
- `MLBAMID`: MLB Advanced Media player ID

**Position assignment:** Determine SP vs RP from the data:
- If `GS >= 3`, assign `Position = 'SP'`
- Otherwise, assign `Position = 'RP'`

(No database lookup needed for pitchers—position is derived from projection data.)

### Fantrax Roster Conversion

**Important:** Rosters are exported from Fantrax and must be converted to the pipeline format. The raw Fantrax CSVs contain team information that enables robust name matching.

#### Raw Fantrax Export Format

Fantrax roster exports have a multi-section format with "Hitting" and "Pitching" sections:

```csv
"","Hitting"
"ID","Pos","Player","Team","Eligible","Status","Age",...
"*04my4*","C","Cal Raleigh","SEA","C","Act","29",...
"*04fhb*","OF","Julio Rodriguez","SEA","OF","Act","25",...
...
"","Pitching"
"ID","Pos","Player","Team","Eligible","Status","Age",...
"*12345*","SP","Max Fried","ATL","SP","Act","30",...
```

**Key columns from Fantrax:**
- `Player`: Player name (may differ from FanGraphs spelling)
- `Team`: MLB team abbreviation (critical for matching!)
- `Status`: One of `Act` (active), `Res` (reserve/bench), `Min` (minors), `IR` (injured)

#### Fantrax Status Mapping

```python
FANTRAX_STATUS_MAP = {
    "Act": "active",   # Active/starting player
    "Res": "active",   # Reserve/bench (still counts toward 26-man roster)
    "Min": "prospect", # Minor league player
    "IR": "IR",        # Injured reserve
}
```

#### Conversion Functions

```python
def parse_fantrax_roster(filepath: str) -> pd.DataFrame:
    """
    Parse a single Fantrax roster CSV file.
    
    Handles the multi-section format (separate Hitting/Pitching sections).
    
    Returns:
        DataFrame with columns: name, fantrax_status, status, player_type
        where status is mapped to pipeline format (active, prospect, IR)
    """

def convert_fantrax_rosters(
    my_team_path: str,
    opponent_team_paths: dict[int, str],
    output_dir: str,
) -> tuple[str, str]:
    """
    Convert Fantrax roster exports to the pipeline's expected format.
    
    Args:
        my_team_path: Path to my team's Fantrax roster CSV
        opponent_team_paths: Dict mapping team_id (1-6) to path of opponent's roster CSV
        output_dir: Directory to write output files (e.g., 'data/')
    
    Returns:
        Tuple of (my_roster_path, opponent_rosters_path)
    """

def convert_fantrax_rosters_from_dir(
    raw_rosters_dir: str,
    my_team_filename: str = 'my_team.csv',
    output_dir: str = None,
) -> tuple[str, str]:
    """
    Convenience wrapper that auto-discovers opponent roster files in a directory.
    
    Looks for all CSV files in raw_rosters_dir that are not the my_team file
    and assigns them team_ids 1-6 in alphabetical order.
    
    Args:
        raw_rosters_dir: Directory containing Fantrax roster CSV files
        my_team_filename: Filename of my team's roster (default: 'my_team.csv')
        output_dir: Directory to write output files. If None, uses parent of raw_rosters_dir.
    
    Returns:
        Tuple of (my_roster_path, opponent_rosters_path)
    
    Example:
        convert_fantrax_rosters_from_dir('data/raw_rosters/')
        # Auto-discovers: my_team.csv + 6 opponent files (team_*.csv)
        # Assigns team_ids alphabetically: team_aidan.csv → 1, team_future.csv → 2, etc.
    """
```

#### Name Correction Functions

After conversion, names may not match FanGraphs projections due to accent differences, apostrophe variations, etc. These functions help diagnose and fix mismatches:

```python
def find_name_mismatches(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
) -> pd.DataFrame:
    """
    Find roster names that don't match FanGraphs projections and suggest corrections.
    
    Uses normalized name comparison to find likely matches.
    
    Returns:
        DataFrame with columns: roster_name, suggested_match, similarity_score
        Only includes names that don't have an exact match.
    """

def apply_name_corrections(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
    min_similarity: float = 0.9,
) -> None:
    """
    Apply automatic name corrections to a roster file based on fuzzy matching.
    
    High-confidence corrections (similarity >= min_similarity) are applied automatically.
    Lower-confidence matches are printed for manual review.
    
    Handles special cases:
    - Accents: "Andres Munoz" → "Andrés Muñoz"
    - Apostrophes: "Logan OHoppe" → "Logan O'Hoppe"
    - Known corrections: "Leodalis De Vries" → "Leo De Vries"
    - Ohtani split: When Ohtani-H and Ohtani-P are on different teams,
      auto-merges both to the hitter's team with a warning
    """
```

#### Known Name Corrections

Some names require explicit correction that normalization can't catch:

```python
known_corrections = {
    "Logan OHoppe": "Logan O'Hoppe",
    "Leodalis De Vries": "Leo De Vries",
}
```

#### Ohtani Split Handling

In some leagues, Ohtani's hitting and pitching are owned by different teams. The pipeline handles this automatically:

1. Detects when `Shohei Ohtani-H` and `Shohei Ohtani-P` are on different fantasy teams
2. Auto-merges both entries to the hitter's team (team 6 in the example)
3. Prints a warning explaining that Ohtani will contribute BOTH hitting and pitching stats to that team
4. The other team loses Ohtani entirely from their roster

This is a limitation—the pipeline was designed assuming one team owns all of a two-way player's stats.

### Roster CSVs (Generated)

These files are generated by `convert_fantrax_rosters()` from raw Fantrax exports.

#### `my-roster.csv`

```csv
name,team,player_type,status
Cal Raleigh,SEA,hitter,active
Julio Rodriguez,SEA,hitter,active
Luis Castillo,SEA,pitcher,active
Max Fried,NYY,pitcher,active
Some Prospect,MIL,hitter,prospect
Injured Guy,NYY,pitcher,IR
```

**Columns:**
- `name`: Player name from Fantrax (may have accent differences from FanGraphs)
- `team`: MLB team abbreviation (used for robust name matching)
- `player_type`: `hitter` or `pitcher` (critical for disambiguating name collisions like Luis Castillo!)
- `status`: One of `active`, `prospect`, or `IR`

Only `active` players are included in optimization. The `team` + `player_type` columns enable robust matching that handles:
- Accent differences: `Julio Rodriguez (SEA)` → `Julio Rodríguez`
- Name collisions: `Luis Castillo (SEA, pitcher)` vs `Luis Castillo (MIL, hitter)`

#### `opponent-rosters.csv`

```csv
team_id,name,team,player_type,status
1,Mike Trout,LAA,hitter,active
1,Freddie Freeman,LAD,hitter,active
2,Juan Soto,NYM,hitter,active
2,Luis Castillo,SEA,pitcher,active
```

**Columns:**
- `team_id`: Integer 1-6 identifying the fantasy team (opponent)
- `name`: Player name from Fantrax
- `team`: MLB team abbreviation
- `player_type`: `hitter` or `pitcher`
- `status`: One of `active`, `prospect`, or `IR`

**Note:** Two-way players like Ohtani may appear with `-H` (hitter) or `-P` (pitcher) suffixes in Fantrax exports. The matching logic uses `player_type` to match them to the correct projection entry.

**Sample opponent rosters** (generated from Fantrax exports):

```csv
team_id,name,status
1,Freddie Freeman,active
1,Trea Turner,active
1,Corey Seager,active
1,Marcus Semien,active
1,Rafael Devers,active
1,Kyle Tucker,active
1,Adley Rutschman,active
1,Ozzie Albies,active
1,Pete Alonso,active
1,Bo Bichette,active
1,Zack Wheeler,active
1,Gerrit Cole,active
1,Spencer Strider,active
1,Corbin Burnes,active
1,Tyler Glasnow,active
1,Blake Snell,active
1,Sandy Alcantara,active
1,Kevin Gausman,active
1,Josh Hader,active
1,Edwin Diaz,active
2,Fernando Tatis Jr.,active
2,Wander Franco,active
2,Matt Olson,active
2,Dansby Swanson,active
2,Austin Riley,active
2,Bryce Harper,active
2,Yordan Alvarez,active
2,William Contreras,active
2,J.T. Realmuto,active
2,Ketel Marte,active
2,Shane McClanahan,active
2,Max Scherzer,active
2,Justin Verlander,active
2,Shota Imanaga,active
2,Yoshinobu Yamamoto,active
2,George Kirby,active
2,Luis Castillo,active
2,Nestor Cortes,active
2,Felix Bautista,active
2,Alexis Diaz,active
3,Starling Marte,active
3,Xander Bogaerts,active
3,Alex Bregman,active
3,Nolan Arenado,active
3,Salvador Perez,active
3,Christian Walker,active
3,Byron Buxton,active
3,Seiya Suzuki,active
3,Brandon Lowe,active
3,Javier Baez,active
3,Ronel Blanco,active
3,MacKenzie Gore,active
3,Jesus Luzardo,active
3,Shane Bieber,active
3,Framber Valdez,active
3,Jack Flaherty,active
3,Nathan Eovaldi,active
3,Joe Ryan,active
3,Robert Suarez,active
3,Jordan Romano,active
4,Anthony Volpe,active
4,CJ Abrams,active
4,Spencer Torkelson,active
4,Yandy Diaz,active
4,Riley Greene,active
4,Steven Kwan,active
4,Michael Harris II,active
4,Jackson Chourio,active
4,Eloy Jimenez,active
4,MJ Melendez,active
4,Sonny Gray,active
4,Hunter Greene,active
4,Grayson Rodriguez,active
4,Nick Lodolo,active
4,Mitch Keller,active
4,Reid Detmers,active
4,Cade Smith,active
4,Tanner Houck,active
4,Andres Munoz,active
4,Clay Holmes,active
5,Jordan Westburg,active
5,Matt McLain,active
5,Triston Casas,active
5,Josh Jung,active
5,Jarren Duran,active
5,Lane Thomas,active
5,Jackson Merrill,active
5,Wyatt Langford,active
5,Anthony Santander,active
5,Sean Murphy,active
5,Bryan Woo,active
5,Jared Jones,active
5,Bobby Miller,active
5,Gavin Williams,active
5,Andrew Abbott,active
5,Kodai Senga,active
5,Tanner Bibee,active
5,Bryce Miller,active
5,Raisel Iglesias,active
5,Pete Fairbanks,active
6,Ezequiel Tovar,active
6,Masyn Winn,active
6,Vinnie Pasquantino,active
6,Alec Bohm,active
6,Luis Robert Jr.,active
6,Jake Burger,active
6,James Outman,active
6,Evan Carter,active
6,Colton Cowser,active
6,Yainer Diaz,active
6,Cristopher Sanchez,active
6,Jacob deGrom,active
6,Eury Perez,active
6,Clarke Schmidt,active
6,Michael King,active
6,Ranger Suarez,active
6,Pablo Lopez,active
6,Marcus Stroman,active
6,Mason Miller,active
6,Jhoan Duran,active
```

**Note:** Sample rosters use real player names. Update these files with your actual league rosters before running the optimizer.

---

## Name Matching

Player names must match exactly between roster CSVs and FanGraphs projections. The `apply_name_corrections()` function handles common mismatches automatically.

### Name Normalization for Matching

```python
def normalize_name(name: str) -> str:
    """
    Normalize player name for fuzzy comparison.
    
    Handles:
    - Accented characters (Rodríguez → rodriguez)
    - Apostrophe variations (O'Hoppe, OHoppe → o'hoppe)
    
    IMPORTANT: Does NOT strip -H/-P suffix - that's part of the unique name!
    "Aaron Judge-H" normalizes to "aaron judge-h" (suffix preserved)
    """
    import unicodedata
    
    # Remove accents
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    
    # Lowercase
    normalized = normalized.lower()
    
    # Handle OHoppe → O'Hoppe
    if "ohoppe" in normalized:
        normalized = normalized.replace("ohoppe", "o'hoppe")
    
    # Standardize apostrophes
    normalized = normalized.replace("'", "'").replace("'", "'")
    
    return normalized.strip()
```

### `find_name_mismatches()` - Finding Unmatched Names

```python
def find_name_mismatches(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
) -> pd.DataFrame:
    """
    Find roster names that don't match FanGraphs projections and suggest corrections.
    
    Returns:
        DataFrame with columns: roster_name, suggested_match, similarity_score
        Only includes names that don't have an exact match.
    
    CRITICAL IMPLEMENTATION DETAIL:
        The internal normalize_name() function MUST preserve -H/-P suffixes!
        Otherwise "Julio Rodriguez-H" will match "Julio Rodríguez-P" incorrectly.
        
        Correct normalization:
        - "Julio Rodriguez-H" → "julio rodriguez-h"
        - "Julio Rodríguez-P" → "julio rodriguez-p"
        - These do NOT match (different suffixes)
        
        Wrong normalization (strips suffix):
        - "Julio Rodriguez-H" → "julio rodriguez"
        - "Julio Rodríguez-P" → "julio rodriguez"
        - These incorrectly match!
    """
```

**Internal normalization function:**
```python
def normalize_name(name: str) -> str:
    """Normalize name for fuzzy matching, PRESERVING -H/-P suffix."""
    import unicodedata
    import re
    
    # Extract and preserve the -H/-P suffix if present
    suffix_to_preserve = ""
    if name.endswith("-H") or name.endswith("-P"):
        suffix_to_preserve = name[-2:]
        name = name[:-2]
    
    # Remove accents
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    
    # Lowercase and handle apostrophes
    normalized = normalized.lower()
    normalized = re.sub(r"([a-z])([A-Z])", r"\1'\2", name.lower())
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = normalized.replace("'", "'").replace("'", "'")
    
    # Remove name suffixes like Jr., Sr. (but NOT -H/-P)
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii"]:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    
    # Re-attach the -H/-P suffix (lowercased)
    return normalized.strip() + suffix_to_preserve.lower()
```

**Known corrections dict (must include suffixes):**
```python
known_corrections = {
    "Logan OHoppe-H": "Logan O'Hoppe-H",
    "Leodalis De Vries-H": "Leo De Vries-H",
    "Leodalis De Vries-P": "Leo De Vries-P",
}
```

### `apply_name_corrections()` - Auto-Correcting Names

```python
def apply_name_corrections(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
    min_similarity: float = 0.9,
) -> None:
    """
    Apply automatic name corrections to a roster file based on fuzzy matching.
    
    Flow:
        1. Call find_name_mismatches() to get unmatched names with suggestions
        2. Filter to high-confidence matches (similarity >= min_similarity)
        3. Build correction_map: {old_name: new_name}
        4. Read roster CSV, apply corrections, write back
        5. Print low-confidence matches for manual review
    
    CRITICAL: Because find_name_mismatches preserves -H/-P suffixes,
    "Julio Rodriguez-H" will only match "Julio Rodríguez-H" (the hitter),
    never "Julio Rodríguez-P" (the pitcher).
    """
```

**Example corrections (note suffixes are preserved):**
```
Julio Rodriguez-H    → Julio Rodríguez-H ✓
Andres Munoz-P       → Andrés Muñoz-P ✓
Logan OHoppe-H       → Logan O'Hoppe-H ✓
Ronald Acuna Jr.-H   → Ronald Acuña Jr.-H ✓
```

### Manual Review Cases

Names with low similarity scores are printed for manual review but not auto-corrected:

```
*** MANUAL REVIEW NEEDED (2 names) ***
These names have low-confidence matches and need manual correction:
  'Cole Hertzler-P' -> 'Cole Patten-P' (score: 0.33)
  'Tre Morgan-H' -> 'Tre' Morgan-H' (score: 0.33)
```

These are typically prospects not in FanGraphs projections, which is fine since prospects are excluded from optimization anyway.

---

## Core Functions in `roster_optimizer.py`

### Data Loading

```python
def load_positions_from_db(db_path: str) -> dict[int, str]:
    """
    Load player positions from SQLite database.
    
    Args:
        db_path: Path to mlb_stats.db (typically '../mlb_player_comps_dashboard/mlb_stats.db')
    
    Returns:
        Dict mapping MLBAMID (int) to position (str).
        Example: {592450: 'OF', 677951: 'SS', ...}
    
    Implementation:
        1. Connect to SQLite database
        2. Query: SELECT player_id, position FROM players
        3. Return as dict
    
    Print:
        "Loaded positions for {N} players from database"
    """
```

```python
def load_hitter_projections(filepath: str, positions: dict[int, str]) -> pd.DataFrame:
    """
    Load hitter projections from FanGraphs CSV export.
    
    Args:
        filepath: Path to fangraphs-steamer-projections-hitters.csv
        positions: Dict mapping MLBAMID to position (from load_positions_from_db)
    
    Returns:
        DataFrame with columns: Name, Team, Position, PA, R, HR, RBI, SB, OBP, SLG, OPS
        Plus a 'player_type' column set to 'hitter' for all rows.
    
    Implementation:
        1. Read CSV with pd.read_csv()
        2. OPS column already exists—use it directly (do NOT recompute)
        3. Join position from positions dict using MLBAMID column
        4. For players not found in positions dict, assign Position = 'DH' (fallback)
        5. Add player_type = 'hitter'
        6. Select and rename columns to standardized names
        7. Assert no null values in required columns: Name, PA, R, HR, RBI, SB, OPS
        8. Assert no duplicate names within hitters
    
    Print:
        "Loaded {N} hitter projections ({M} with positions from DB, {K} defaulted to DH)"
    """
```

```python
def load_pitcher_projections(filepath: str) -> pd.DataFrame:
    """
    Load pitcher projections from FanGraphs CSV export.
    
    Args:
        filepath: Path to fangraphs-steamer-projections-pitchers.csv
    
    Returns:
        DataFrame with columns: Name, Team, Position, IP, W, SV, K, ERA, WHIP
        Plus a 'player_type' column set to 'pitcher' for all rows.
    
    Implementation:
        1. Read CSV with pd.read_csv()
        2. Rename SO column to K
        3. Determine Position: 'SP' if GS >= 3, else 'RP'
        4. Add player_type = 'pitcher'
        5. Select and rename columns to standardized names
        6. Assert no null values in required columns: Name, IP, W, SV, K, ERA, WHIP
        7. Assert no duplicate names within pitchers
    
    Print:
        "Loaded {N} pitcher projections ({X} SP, {Y} RP)"
    """
```

```python
def load_projections(hitter_path: str, pitcher_path: str, db_path: str) -> pd.DataFrame:
    """
    Load and combine hitter and pitcher projections.
    
    Args:
        hitter_path: Path to hitter projections CSV
        pitcher_path: Path to pitcher projections CSV
        db_path: Path to SQLite database with position data
    
    Returns:
        Combined DataFrame with all players.
        Hitters have pitching columns (IP, W, SV, K, ERA, WHIP) set to 0.
        Pitchers have hitting columns (PA, R, HR, RBI, SB, OBP, SLG, OPS) set to 0.
    
    Implementation:
        1. Load positions from database
        2. Load hitters and pitchers separately
        3. Add missing columns with value 0 to each
        4. Concatenate into single DataFrame
        5. Reset index
        6. Two-way players: Ohtani appears separately in hitters and pitchers files,
           which is fine—they have different stats. Assert no duplicates WITHIN each
           player_type, but allow same name to appear once as hitter and once as pitcher.
    
    Print:
        "Combined projections: {N} total players ({H} hitters, {P} pitchers)"
    """
```

```python
def load_my_roster(filepath: str, projections: pd.DataFrame) -> tuple[set[str], set[str]]:
    """
    Load my roster from CSV.
    
    Args:
        filepath: Path to my-roster.csv (must have name corrections already applied)
        projections: Combined projections DataFrame (for name validation)
    
    Returns:
        Tuple of:
        - active_names: Set of player names on active roster
        - inactive_names: Set of player names on prospect/IR slots (for reference)
    
    Implementation:
        1. Read CSV
        2. Split by status column
        3. Assert all names in active_names exist in projections['Name']
           If any don't match, crash with message listing unmatched names
    
    Print:
        "My roster: {A} active, {I} inactive (prospects/IR)"
    """
```

```python
def load_opponent_rosters(filepath: str, projections: pd.DataFrame) -> dict[int, set[str]]:
    """
    Load opponent rosters from CSV.
    
    Args:
        filepath: Path to opponent-rosters.csv (must have name corrections already applied)
        projections: Combined projections DataFrame (for name validation)
    
    Returns:
        Dict mapping team_id to set of active player names.
        Example: {1: {'Mike Trout', 'Mookie Betts', ...}, 2: {...}, ...}
    
    Implementation:
        1. Read CSV
        2. Filter to status == 'active'
        3. Group by team_id
        4. Assert team_ids are exactly {1, 2, 3, 4, 5, 6}
        5. Assert all player names exist in projections['Name']
           If any don't match, crash with message listing unmatched names
        6. Assert no player appears on multiple teams
    
    Print:
        "Loaded 6 opponent rosters: {N1}, {N2}, {N3}, {N4}, {N5}, {N6} active players each"
    """
```

```python
def load_all_data(
    hitter_proj_path: str,
    pitcher_proj_path: str,
    my_roster_path: str,
    opponent_rosters_path: str,
    db_path: str = "../mlb_player_comps_dashboard/mlb_stats.db",
) -> tuple[pd.DataFrame, set[str], dict[int, set[str]]]:
    """
    Load all data files and validate.
    
    This is the main entry point for data loading.
    
    Args:
        hitter_proj_path: Path to hitter projections CSV
        pitcher_proj_path: Path to pitcher projections CSV
        my_roster_path: Path to my roster CSV
        opponent_rosters_path: Path to opponent rosters CSV
        db_path: Path to SQLite database with position data
    
    Returns:
        projections: Combined projections DataFrame
        my_active_roster: Set of player names on my active roster
        opponent_rosters: Dict mapping team_id to set of active player names
    
    Implementation:
        Call the individual load functions in sequence.
        All validation happens in those functions.
    """
```

### Team Total Computation

```python
from typing import Iterable

def compute_team_totals(
    player_names: Iterable[str],
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute a team's projected totals in all 10 scoring categories.
    
    Args:
        player_names: Iterable of player names on the team (set, list, etc.)
        projections: Combined projections DataFrame
    
    Returns:
        Dict mapping category name to team total.
        Example: {'R': 823, 'HR': 245, 'RBI': 780, 'SB': 95, 'OPS': 0.758,
                  'W': 78, 'SV': 55, 'K': 1250, 'ERA': 3.85, 'WHIP': 1.22}
    
    Implementation:
        1. Convert player_names to set if not already
        2. Filter projections to players in player_names
        3. Separate into hitters (player_type == 'hitter') and pitchers
        
        For counting stats (R, HR, RBI, SB, W, SV, K):
            total = column.sum()
        
        For ratio stats:
            OPS:  sum(PA * OPS) / sum(PA) over hitters
            ERA:  sum(IP * ERA) / sum(IP) over pitchers
            WHIP: sum(IP * WHIP) / sum(IP) over pitchers
    
    Assertions:
        - sum(PA) > 0 for hitters (else OPS undefined)
        - sum(IP) > 0 for pitchers (else ERA/WHIP undefined)
        - All player_names found in projections (crash with list of missing names)
    
    Note: This function works for both my roster and opponent rosters.
    The old compute_my_totals function is removed—just use this one.
    """
```

```python
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
        {1: {'R': 800, 'HR': 230, ...}, 2: {'R': 815, ...}, ...}
    
    Implementation:
        Loop over opponent_rosters, call compute_team_totals for each.
        Use tqdm progress bar.
    
    Assertions:
        Each opponent must have sum(PA) > 0 and sum(IP) > 0.
        (These assertions happen inside compute_team_totals.)
    """
```

### Candidate Prefiltering

```python
def compute_quality_scores(projections: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a scalar quality score for each player.
    
    Methodology:
        Hitters: Mean z-score of (R, HR, RBI, SB).
                 Do NOT include OPS (ratio stat, z-score doesn't apply cleanly).
        
        Pitchers: Mean z-score of (W, SV, K, -ERA, -WHIP).
                  Negate ERA and WHIP before z-scoring so lower = better becomes higher = better.
    
    Implementation:
        1. Separate hitters and pitchers using player_type column
        2. For hitters: z-score each of R, HR, RBI, SB; take mean across the 4
        3. For pitchers: z-score W, SV, K; z-score -ERA, -WHIP; take mean across all 5
        4. Combine into single DataFrame
    
    Returns:
        DataFrame with columns [Name, player_type, quality_score]
    
    Assertions:
        - Standard deviation > 0 for each stat being z-scored
    """
```

```python
def filter_candidates(
    projections: pd.DataFrame,
    quality_scores: pd.DataFrame,
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    top_n_per_position: int = 30,
    top_n_per_category: int = 10,
) -> pd.DataFrame:
    """
    Filter to candidate players for optimization.
    
    Args:
        projections: Combined projections DataFrame
        quality_scores: DataFrame with quality_score per player
        my_roster_names: Set of player names currently on my roster
        opponent_roster_names: Set of ALL player names on ANY opponent roster (unavailable)
        top_n_per_position: How many players to keep per position
        top_n_per_category: How many top players per category to ensure are included
    
    Include:
        1. All players currently on my roster (must be candidates even if low quality)
        2. Top N available players at each position by quality score
        3. Top M available players in each scoring category (to capture specialists)
    
    Exclude:
        - All players on opponent rosters (they're unavailable)
    
    Position handling:
        Database stores single positions (no multi-position). Each player has exactly
        one position. When selecting top N for a slot type, include players whose
        Position is in that slot's SLOT_ELIGIBILITY set.
        
        For example, for the UTIL slot, include players with any hitter position.
        For the OF slot, include players with Position='OF'.
    
    Returns:
        Filtered projections DataFrame containing only candidate players.
    
    Implementation:
        1. Create available_pool = projections excluding opponent_roster_names
        2. Join quality_scores to available_pool
        3. For each slot type in SLOT_ELIGIBILITY keys:
           - Find players whose Position is in SLOT_ELIGIBILITY[slot]
           - Take top N by quality_score
           - Add to candidate set
        4. For each scoring category:
           - Take top M players by that category's value
           - Add to candidate set
        5. Add all my_roster_names to candidate set
        6. Return projections filtered to candidate set
    
    Print:
        "Filtered to {X} candidates from {Y} total players"
        "  - {H} hitters, {P} pitchers"
    """
```

---

## MILP Formulation

This section describes the mathematical optimization model. I provide both mathematical notation and equivalent Python pseudocode for each component.

**Key insight:** All rostered players contribute stats (not just starters). The slot assignment variables (`a[i,s]`) exist only to ensure the roster CAN field a valid starting lineup—they don't affect stat contributions. The beat constraints sum over all rostered players using `x[i]`.

### Index Sets

```
I     = set of candidate player indices (0 to len(candidates)-1)
I_H   = subset of I where player_type == 'hitter'
I_P   = subset of I where player_type == 'pitcher'
J     = {1, 2, 3, 4, 5, 6} = opponent team_ids (hardcoded 6 opponents)
C_H   = {'R', 'HR', 'RBI', 'SB', 'OPS'} = hitting categories
C_P   = {'W', 'SV', 'K', 'ERA', 'WHIP'} = pitching categories
C     = C_H ∪ C_P = all 10 categories
S_H   = {'C', '1B', '2B', 'SS', '3B', 'OF', 'UTIL'} = hitting slot types
S_P   = {'SP', 'RP'} = pitching slot types
S     = S_H ∪ S_P = all slot types
```

Python:
```python
I = list(range(len(candidates)))
I_H = [i for i in I if candidates.iloc[i]['player_type'] == 'hitter']
I_P = [i for i in I if candidates.iloc[i]['player_type'] == 'pitcher']
J = [1, 2, 3, 4, 5, 6]
# ... etc
```

### Decision Variables

**Player selection:**
```
x[i] ∈ {0, 1}    for all i ∈ I
```
`x[i] = 1` means player i is on my roster.

**Slot assignment:**
```
a[i,s] ∈ {0, 1}    for all i ∈ I, s ∈ S where player i is eligible for slot s
```
`a[i,s] = 1` means player i is assigned to start in slot type s.

Only create `a[i,s]` variables for eligible (player, slot) pairs. Eligibility is determined by:
- Check if player's Position is in `SLOT_ELIGIBILITY[s]`

(Database stores single positions, no need to parse multi-position strings.)

**Beat indicators:**
```
y[j,c] ∈ {0, 1}    for all j ∈ J, c ∈ C
```
`y[j,c] = 1` means I beat opponent j in category c.

Python:
```python
import pulp

# Player selection
x = {i: pulp.LpVariable(f"x_{i}", cat='Binary') for i in I}

# Slot assignment (only for eligible pairs)
a = {}
for i in I:
    player_position = candidates.iloc[i]['Position']
    for s in S:
        if player_position in SLOT_ELIGIBILITY[s]:
            a[i, s] = pulp.LpVariable(f"a_{i}_{s}", cat='Binary')

# Beat indicators
y = {(j, c): pulp.LpVariable(f"y_{j}_{c}", cat='Binary') for j in J for c in C}
```

### Objective Function

Maximize total opponent-category wins:

```
maximize  Σ_{j ∈ J} Σ_{c ∈ C} y[j,c]
```

With 6 opponents and 10 categories, the maximum possible is 60.

Python:
```python
prob = pulp.LpProblem("roster_optimization", pulp.LpMaximize)
prob += pulp.lpSum(y[j, c] for j in J for c in C)
```

### Constraints

#### C1: Roster Size

Exactly 26 players on the roster:

```
Σ_{i ∈ I} x[i] = 26
```

Python:
```python
prob += pulp.lpSum(x[i] for i in I) == ROSTER_SIZE
```

#### C2: Slot Assignment Requires Rostering

A player can only be assigned to a slot if they're on the roster:

```
a[i,s] ≤ x[i]    for all (i,s) where a[i,s] exists
```

Python:
```python
for (i, s), var in a.items():
    prob += var <= x[i]
```

#### C3: Each Player Assigned to At Most One Slot

A player cannot fill multiple starting slots simultaneously:

```
Σ_{s : (i,s) ∈ a} a[i,s] ≤ 1    for all i ∈ I
```

Python:
```python
for i in I:
    slots_for_player = [a[i, s] for s in S if (i, s) in a]
    if slots_for_player:
        prob += pulp.lpSum(slots_for_player) <= 1
```

#### C4: Starting Slots Must Be Filled

Each slot type must have exactly the required number of players assigned:

```
Σ_{i : (i,s) ∈ a} a[i,s] = n_s    for all s ∈ S
```

Where `n_s` is the number of slots of type s (e.g., `n_OF = 3`, `n_SP = 5`).

Python:
```python
all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
for s, required_count in all_slots.items():
    players_for_slot = [a[i, s] for i in I if (i, s) in a]
    assert len(players_for_slot) >= required_count, f"Not enough candidates eligible for {s} slot"
    prob += pulp.lpSum(players_for_slot) == required_count
```

#### C5: Roster Composition Bounds

Total hitters and pitchers must be within reasonable bounds:

```
MIN_HITTERS ≤ Σ_{i ∈ I_H} x[i] ≤ MAX_HITTERS
MIN_PITCHERS ≤ Σ_{i ∈ I_P} x[i] ≤ MAX_PITCHERS
```

Python:
```python
prob += pulp.lpSum(x[i] for i in I_H) >= MIN_HITTERS
prob += pulp.lpSum(x[i] for i in I_H) <= MAX_HITTERS
prob += pulp.lpSum(x[i] for i in I_P) >= MIN_PITCHERS
prob += pulp.lpSum(x[i] for i in I_P) <= MAX_PITCHERS
```

#### C6: Beat Constraints for Counting Stats

For counting stats (R, HR, RBI, SB, W, SV, K) where higher is better:

Let `M[i,c]` = player i's projected value in category c.
Let `O[j,c]` = opponent j's total in category c.

```
Σ_{i ∈ I_relevant} M[i,c] * x[i]  ≥  O[j,c] + ε - B * (1 - y[j,c])    for all j ∈ J
```

Where:
- `I_relevant` = `I_H` for hitting categories, `I_P` for pitching categories
- `ε = 0.5` (ensures strict inequality for integer stats; ties lose)
- `B = 10000` (big-M constant, larger than any plausible category total)

**How this works:**
- If `y[j,c] = 1`: constraint becomes `my_total ≥ O[j,c] + 0.5`, i.e., I must beat them.
- If `y[j,c] = 0`: constraint becomes `my_total ≥ O[j,c] - 9999.5`, always satisfied.

The optimizer maximizes Σy, so it sets `y[j,c] = 1` whenever the constraint allows (i.e., whenever I actually beat them).

Python:
```python
EPSILON_COUNTING = 0.5
BIG_M_COUNTING = 10000

counting_stats = {'R', 'HR', 'RBI', 'SB', 'W', 'SV', 'K'}

for c in counting_stats:
    I_relevant = I_H if c in HITTING_CATEGORIES else I_P
    for j in J:
        my_total = pulp.lpSum(candidates.iloc[i][c] * x[i] for i in I_relevant)
        opp_total = opponent_totals[j][c]
        prob += my_total >= opp_total + EPSILON_COUNTING - BIG_M_COUNTING * (1 - y[j, c])
```

#### C7: Beat Constraints for OPS (Ratio Stat, Higher is Better)

For OPS, we need to linearize the comparison of weighted averages.

My OPS = `Σ(PA[i] * OPS[i] * x[i]) / Σ(PA[i] * x[i])`

I beat opponent j if my OPS > opponent j's OPS.

**Linearization:**
```
my_OPS > O[j, OPS]
⟺  Σ(PA[i] * OPS[i] * x[i]) / Σ(PA[i] * x[i]) > O[j, OPS]
⟺  Σ(PA[i] * OPS[i] * x[i]) > O[j, OPS] * Σ(PA[i] * x[i])
⟺  Σ PA[i] * (OPS[i] - O[j, OPS]) * x[i] > 0
```

Define transformed coefficient:
```
M_tilde[i, j, OPS] = PA[i] * (OPS[i] - O[j, OPS])
```

Constraint:
```
Σ_{i ∈ I_H} M_tilde[i, j, OPS] * x[i]  ≥  ε - B * (1 - y[j, OPS])    for all j ∈ J
```

Where `ε = 0.001` (small positive for continuous values), `B = 5000`.

Python:
```python
EPSILON_RATIO = 0.001
BIG_M_RATIO = 5000

for j in J:
    opp_ops = opponent_totals[j]['OPS']
    my_weighted_diff = pulp.lpSum(
        candidates.iloc[i]['PA'] * (candidates.iloc[i]['OPS'] - opp_ops) * x[i]
        for i in I_H
    )
    prob += my_weighted_diff >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, 'OPS'])
```

#### C8: Beat Constraints for ERA and WHIP (Ratio Stats, Lower is Better)

For ERA, I beat opponent j if my ERA < opponent j's ERA.

**Linearization:**
```
my_ERA < O[j, ERA]
⟺  Σ(IP[i] * ERA[i] * x[i]) / Σ(IP[i] * x[i]) < O[j, ERA]
⟺  Σ(IP[i] * ERA[i] * x[i]) < O[j, ERA] * Σ(IP[i] * x[i])
⟺  Σ IP[i] * (O[j, ERA] - ERA[i]) * x[i] > 0
```

Define transformed coefficient:
```
M_tilde[i, j, ERA] = IP[i] * (O[j, ERA] - ERA[i])
```

Note the sign flip compared to OPS: `(O[j,c] - stat[i])` instead of `(stat[i] - O[j,c])`.

Constraint:
```
Σ_{i ∈ I_P} M_tilde[i, j, ERA] * x[i]  ≥  ε - B * (1 - y[j, ERA])    for all j ∈ J
```

Same for WHIP.

Python:
```python
for c in ['ERA', 'WHIP']:
    for j in J:
        opp_val = opponent_totals[j][c]
        my_weighted_diff = pulp.lpSum(
            candidates.iloc[i]['IP'] * (opp_val - candidates.iloc[i][c]) * x[i]
            for i in I_P
        )
        prob += my_weighted_diff >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, c])
```

### Solver Invocation

```python
# Use HiGHS via Python bindings (highspy), NOT the command-line version
solver = pulp.HiGHS(msg=1, timeLimit=300)
status = prob.solve(solver)

assert status == pulp.LpStatusOptimal, (
    f"Solver failed with status: {pulp.LpStatus[status]}. "
    f"This likely means the constraints are infeasible—check that enough "
    f"candidates are eligible for each position slot."
)
```

**Important:** Use `pulp.HiGHS()` (Python bindings via highspy), NOT `pulp.HiGHS_CMD()` (command-line). The command-line version requires a separate HiGHS installation, while the Python bindings are included with the `highspy` package.

**Pre-solve validation:** Before solving, check each slot type and assert there are enough eligible candidates:

```python
all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
for s, required_count in all_slots.items():
    eligible_count = sum(1 for i in I if (i, s) in a)
    assert eligible_count >= required_count, (
        f"Not enough candidates for {s} slot: need {required_count}, "
        f"but only {eligible_count} candidates are eligible. "
        f"Consider increasing top_n_per_position or adding more players to your roster."
    )
```

**Solver notes:**
- HiGHS does NOT support warm-starting in PuLP (only CBC, CPLEX, GUROBI, XPRESS do).
- For sensitivity analysis, each MILP solve starts from scratch (~1-2s each).
- If highspy is not installed, the solver will fail. No fallback—just crash.

---

## MILP Function

```python
def build_and_solve_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    current_roster_names: set[str],
) -> tuple[list[str], dict]:
    """
    Build and solve the MILP for optimal roster construction.
    
    Args:
        candidates: DataFrame of candidate players (filtered projections).
                    Must have columns: Name, Position, player_type,
                    and all stat columns (R, HR, ..., ERA, WHIP, PA, IP).
        opponent_totals: Dict mapping team_id to dict of category totals.
                         Example: {1: {'R': 800, 'HR': 230, ...}, ...}
        current_roster_names: Set of player names currently on my roster.
                              (Not used in constraints, but useful for logging changes.)
    
    Returns:
        optimal_roster_names: List of player Names for the optimal 26-man roster
        solution_info: Dict with keys:
            - 'objective': float, number of opponent-category wins (max 60)
            - 'solve_time': float, seconds to solve
            - 'status': str, solver status
    
    Implementation follows the MILP formulation above.
    
    Print progress:
        - "Building MILP with {N} candidate players..."
        - "  Variables: {X} player, {Y} slot assignment, {Z} beat indicators"
        - "  Constraints: {W} total"
        - "Solving..."
        - "Solved in {T:.1f}s — objective: {obj}/60 opponent-category wins"
    """
```

---

## Solution Extraction and Output

```python
def extract_solution(
    prob: pulp.LpProblem,
    candidates: pd.DataFrame,
    x_vars: dict[int, pulp.LpVariable],
) -> list[str]:
    """
    Extract player Names of rostered players from solved MILP.
    
    Returns:
        List of player Names where x[i] = 1 in the optimal solution.
    
    Implementation:
        return [candidates.iloc[i]['Name'] for i in x_vars if pulp.value(x_vars[i]) > 0.5]
    """
```

**Note:** Use `compute_team_totals(roster_names, projections)` to compute totals for the optimal roster. The function accepts any iterable of player names.

```python
def compute_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute projected standings for each category.
    
    Returns:
        DataFrame with columns: [category, my_value, opp_1, opp_2, ..., opp_6, my_rank, wins]
        
        my_rank: 1 = first place, 7 = last place in that category
        wins: number of opponents I beat in that category (0-6)
    
    For negative categories (ERA, WHIP), lower value = better rank.
    """
```

```python
def print_roster_summary(
    roster_names: list[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    old_roster_names: set[str] = None,
) -> None:
    """
    Print a formatted summary of the optimal roster.
    
    Sections:
    
    1. ROSTER (26 players)
       Split into Hitters and Pitchers.
       Show: position, name, team, and relevant stats.
       
    2. CHANGES (if old_roster_names provided)
       List players added and dropped.
    
    3. STANDINGS PROJECTION
       Table showing my value vs each opponent in each category.
       Mark wins with asterisk (*) or similar indicator.
       
    4. SUMMARY
       "Total opponent-category wins: X / 60"
       "Projected roto points: Y / 70" (sum of (8 - rank) across 10 categories)
    
    Output format example:
    
    === OPTIMAL ROSTER ===
    
    HITTERS (14):
    Pos   Name                  Team   PA    R     HR    RBI   SB    OPS
    ----- --------------------- ------ ----- ----- ----- ----- ----- ------
    C     J.T. Realmuto         PHI    520   65    20    70    12    .795
    1B    Freddie Freeman       LAD    600   95    28    95    8     .895
    ...
    
    PITCHERS (12):
    Pos   Name                  Team   IP     W     SV    K     ERA    WHIP
    ----- --------------------- ------ ------ ----- ----- ----- ------ ------
    SP    Zack Wheeler          PHI    195    15    0     220   3.15   1.05
    ...
    
    === CHANGES FROM CURRENT ROSTER ===
    ADDED:   Mike Trout (OF), Corbin Burnes (SP)
    DROPPED: Random Scrub (OF), Bad Pitcher (RP)
    
    === PROJECTED STANDINGS ===
    Category   Me       Opp1    Opp2    Opp3    Opp4    Opp5    Opp6    Rank  Wins
    ---------- -------- ------- ------- ------- ------- ------- ------- ----- ----
    R          823*     780     810     795     750     800     815     2     5
    HR         267*     245     260     255     230     250     240     1     6
    ...
    
    === SUMMARY ===
    Opponent-category wins: 47 / 60
    Projected roto points: 58 / 70
    """
```

---

## Visualization Functions in `visualizations.py`

All functions return a matplotlib Figure object. Do not call `plt.show()` inside any function—marimo handles display.

### Team Comparison Radar

```python
def plot_team_radar(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Radar chart comparing all 7 teams across all 10 categories.
    
    Display:
        - One polygon per team (7 total)
        - My team: thick solid line, distinct color (blue)
        - Opponents: thin dashed lines, muted colors
        - Legend identifying each team
    
    Normalization:
        Convert each category to percentile rank among the 7 teams.
        This puts all categories on [0, 1] scale.
        For negative categories (ERA, WHIP), flip so that better = higher on chart.
    
    Implementation notes:
        - Use matplotlib's polar projection
        - 10 spokes (one per category), evenly spaced around circle
        - Close the polygon by appending the first point at the end
    """
```

### Category Margins

```python
def plot_category_margins(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Grouped bar chart showing my margin over each opponent in each category.
    
    X-axis: 10 categories
    Bars: 6 bars per category (one per opponent), showing my_value - opponent_value
    Colors: Green if positive (I win), red if negative (I lose)
    
    For negative categories (ERA, WHIP), flip sign: opponent_value - my_value,
    so positive still means I win.
    
    This shows where wins are comfortable vs narrow.
    """
```

### Win Matrix Heatmap

```python
def plot_win_matrix(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> plt.Figure:
    """
    Heatmap showing win/loss for each opponent-category pair.
    
    Rows: 6 opponents
    Columns: 10 categories
    Cell color: Green if I win, red if I lose
    Cell text: Margin (my_value - opponent_value), formatted appropriately
    
    For negative categories, flip sign for display consistency
    (positive margin = I win).
    
    Use matplotlib imshow or pcolormesh.
    Add text annotations in each cell.
    """
```

### Player Contribution Breakdown

```python
def plot_category_contributions(
    roster_names: list[str],
    projections: pd.DataFrame,
    category: str,
) -> plt.Figure:
    """
    Horizontal bar chart showing each player's contribution to one category.
    
    For counting stats: Player's raw value.
    For ratio stats (OPS, ERA, WHIP): Player's "impact" on team ratio.
        Impact = weight * (player_value - team_average_without_player)
        Or simplified: weight * (player_value - team_value)
        This shows who's helping vs hurting the team ratio.
    
    Sort bars by contribution magnitude (largest contributors at top).
    Color bars by positive (helps team) vs negative (hurts team) for ratio stats.
    For counting stats, all contributions are positive.
    
    Useful for identifying:
        - Concentration risk (one player dominates SB)
        - Ratio stat drags (bad WHIP pitcher hurting team)
    """
```

### Roster Diff Visualization

```python
def plot_roster_changes(
    old_roster_names: set[str],
    new_roster_names: set[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visual comparison of old vs new roster.
    
    Layout: Two sections stacked vertically
    
    Top section: Two-column table
        Left column: "DROPPED" - players removed, with key stats
        Right column: "ADDED" - players added, with key stats
    
    Bottom section: Bar chart showing net change in each projected category total
        Green bars = improvement, red bars = decline
    
    Use matplotlib subplots or gridspec for layout.
    """
```

### Player Contribution Radar Chart

```python
def plot_player_contribution_radar(
    roster_names: list[str],
    projections: pd.DataFrame,
    player_type: str = "hitter",
    top_n: int = 12,
) -> plt.Figure:
    """
    Radar chart showing each player's contributions across all relevant categories.
    
    Each player is a polygon on the radar. This shows all category contributions
    at once, instead of needing separate bar charts per category.
    
    Args:
        roster_names: List of player names on the roster
        projections: Combined projections DataFrame
        player_type: "hitter" or "pitcher" - determines which 5 categories to show
        top_n: Maximum number of players to show (to avoid visual clutter)
    
    Categories:
        - Hitters: R, HR, RBI, SB, OPS (5 axes)
        - Pitchers: W, SV, K, ERA, WHIP (5 axes)
    
    Normalization:
        Values normalized to [0, 1] scale based on min/max within the roster.
        For ratio stats (OPS, ERA, WHIP), compute impact = weight * (player - team_avg).
        For negative categories (ERA, WHIP), flip sign so positive = good.
        Players closer to the edge are better in that category.
    
    Implementation:
        1. Filter players by player_type
        2. For each category, compute contribution:
           - Counting stats: raw value
           - Ratio stats: weight * (value - team_avg), flipped for ERA/WHIP
        3. Normalize each category to [0, 1]
        4. Select top N players by total contribution
        5. Plot using matplotlib polar projection
        6. Each player gets a different color line
        7. Strip -H/-P suffix for display names in legend
    
    Returns:
        Figure with radar chart. Call separately for hitters and pitchers.
    
    Usage:
        # Two calls for full roster view
        plot_player_contribution_radar(roster, projections, "hitter", top_n=10)
        plot_player_contribution_radar(roster, projections, "pitcher", top_n=10)
    """
```

### Sensitivity Analysis

```python
def compute_player_sensitivity(
    optimal_roster_names: list[str],
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute sensitivity of objective to each player.
    
    For each player in candidates:
        - If player is ON optimal roster: solve MILP forcing x[i] = 0 (exclude them)
        - If player is NOT on optimal roster: solve MILP forcing x[i] = 1 (include them)
        - Compare resulting objective to unconstrained optimum
    
    Returns:
        DataFrame with columns:
            - Name: player name
            - player_type: hitter or pitcher
            - Position: player's position(s)
            - on_optimal_roster: bool
            - forced_objective: objective value when this player is forced in/out
            - objective_delta: forced_objective - optimal_objective
              (negative means forcing this player's inclusion/exclusion makes us worse)
    
    Implementation:
        This requires solving len(candidates) MILPs.
        Use tqdm with description "Computing player sensitivities".
        Each MILP should be fast (~1-2s), so total time ~5-15 minutes for 300 candidates.
        
        The forcing constraint is simply:
            - To exclude player i: add constraint x[i] == 0
            - To include player i: add constraint x[i] == 1
        
        **Important:** HiGHS does NOT support warm-starting in PuLP. Each solve starts
        from scratch. The implementation should rebuild the MILP for each player
        (or use PuLP's constraint removal if available, but this doesn't warm-start).
        
        Practical approach: Build a fresh MILP for each candidate with the forcing
        constraint included from the start. This is cleaner than trying to modify
        a base problem.
    
    Print at start:
        "Computing sensitivity for {N} candidates (estimated time: {T} minutes)"
        "Note: Each solve starts fresh (HiGHS doesn't support warm-starting)"
    """
```

```python
def plot_player_sensitivity(
    sensitivity_df: pd.DataFrame,
    top_n: int = 15,
) -> plt.Figure:
    """
    Horizontal bar chart showing most impactful players.
    
    Two panels, stacked vertically:
    
    Top panel: "Most Valuable Rostered Players"
        Players currently on optimal roster, sorted by objective_delta (most negative first).
        These are players whose removal hurts the most.
        Bars extend left (negative delta = losing them hurts).
        Show top N.
    
    Bottom panel: "Best Available Non-Rostered"  
        Players NOT on optimal roster, sorted by objective_delta.
        Usually most are zero (optimizer already found best).
        Interesting if some players are close substitutes.
        Show top N by |objective_delta|.
    
    Label bars with player name and position.
    Color by magnitude.
    """
```

### Constraint Slack Analysis

```python
def plot_constraint_analysis(
    candidates: pd.DataFrame,
    optimal_roster_names: list[str],
    projections: pd.DataFrame,
) -> plt.Figure:
    """
    Visualize which roster constraints are binding.
    
    Bar chart showing:
        - For each position slot: how many eligible players are rostered vs required
        - For hitter/pitcher bounds: current count vs min/max
    
    Color coding:
        - Red: at minimum (binding constraint, might want more)
        - Yellow: at maximum (binding constraint, can't add more)
        - Green: between min and max (slack available)
    
    This helps identify if you're constrained by:
        - Lack of catchers in candidate pool
        - Too many hitters (at MAX_HITTERS)
        - etc.
    """
```

---

## Notebook Usage Example

The marimo notebook should follow this pattern:

```python
# Cell 1: Imports
import pandas as pd
import matplotlib.pyplot as plt
from optimizer.roster_optimizer import (
    # Fantrax roster conversion
    convert_fantrax_rosters_from_dir,
    load_projections,
    apply_name_corrections,
    # Data loading
    load_all_data,
    compute_all_opponent_totals,
    compute_team_totals,
    compute_quality_scores,
    filter_candidates,
    build_and_solve_milp,
    compute_standings,
    print_roster_summary,
)
from optimizer.visualizations import (
    plot_team_radar,
    plot_category_margins,
    plot_win_matrix,
    plot_category_contributions,
    plot_player_contribution_radar,
    plot_roster_changes,
    compute_player_sensitivity,
    plot_player_sensitivity,
)

# Cell 2: Configuration (paths)
DATA_DIR = "data/"
RAW_ROSTERS_DIR = DATA_DIR + "raw_rosters/"
HITTER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-hitters.csv"
PITCHER_PROJ_PATH = DATA_DIR + "fangraphs-steamer-projections-pitchers.csv"
MY_ROSTER_PATH = DATA_DIR + "my-roster.csv"
OPPONENT_ROSTERS_PATH = DATA_DIR + "opponent-rosters.csv"
DB_PATH = "../mlb_player_comps_dashboard/mlb_stats.db"

# Cell 3: Convert Fantrax rosters (runs every time notebook is executed)
# Auto-discovers my_team.csv and team_*.csv files in raw_rosters/
# Assigns team_ids 1-6 alphabetically to opponent files
converted_my_roster_path, converted_opponent_rosters_path = convert_fantrax_rosters_from_dir(
    raw_rosters_dir=RAW_ROSTERS_DIR,
)

# Cell 4: Apply name corrections
# Load projections temporarily for name validation
_projections_for_correction = load_projections(
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    DB_PATH,
)

# Auto-correct accents, apostrophes, known mismatches
# Also handles Ohtani split (merges to hitter's team with warning)
apply_name_corrections(converted_my_roster_path, _projections_for_correction)
apply_name_corrections(
    converted_opponent_rosters_path, _projections_for_correction, is_opponent_file=True
)

# Cell 5: Load data (now with corrected names)
projections, my_roster_names, opponent_rosters = load_all_data(
    HITTER_PROJ_PATH,
    PITCHER_PROJ_PATH,
    MY_ROSTER_PATH,
    OPPONENT_ROSTERS_PATH,
    DB_PATH,
)

# Cell 5: Opponent totals
opponent_totals = compute_all_opponent_totals(opponent_rosters, projections)
pd.DataFrame(opponent_totals).T  # Display as table

# Cell 6: Filter candidates
quality_scores = compute_quality_scores(projections)
opponent_roster_names = set().union(*opponent_rosters.values())

candidates = filter_candidates(
    projections,
    quality_scores,
    my_roster_names,
    opponent_roster_names,
    top_n_per_position=30,
    top_n_per_category=10,
)

# Cell 7: Solve
optimal_roster_names, solution_info = build_and_solve_milp(
    candidates,
    opponent_totals,
    my_roster_names,
)

# Cell 8: Results
my_totals = compute_team_totals(optimal_roster_names, projections)
print_roster_summary(
    optimal_roster_names,
    projections,
    my_totals,
    opponent_totals,
    old_roster_names=my_roster_names,
)

# Cell 9: Visualizations
plot_team_radar(my_totals, opponent_totals)

# Cell 10
plot_win_matrix(my_totals, opponent_totals)

# Cell 11: Deep dive on a specific category
plot_category_contributions(optimal_roster_names, projections, 'SB')

# Cell 12: Player contribution radar charts (shows all categories at once)
plot_player_contribution_radar(list(optimal_roster_names), projections, "hitter", top_n=10)

# Cell 13
plot_player_contribution_radar(list(optimal_roster_names), projections, "pitcher", top_n=10)

# Cell 14: Sensitivity (slow—run separately if needed, ~5-15 minutes)
sensitivity = compute_player_sensitivity(optimal_roster_names, candidates, opponent_totals)
plot_player_sensitivity(sensitivity)
```

---

## Edge Cases and Implementation Notes

**The agent must handle these correctly:**

1. **Position handling:** Database stores single positions (`C`, `1B`, `2B`, `SS`, `3B`, `OF`, `DH`). The database already consolidates LF/CF/RF into `OF`. No need to split on `/`.

2. **Slot eligibility check:** When checking if player i can fill slot s, test if player's Position is in `SLOT_ELIGIBILITY[s]`.

3. **OPS already exists:** The FanGraphs hitter CSV already has an `OPS` column. Do NOT recompute it.

4. **Strikeouts column:** The FanGraphs pitcher CSV has `SO` for strikeouts. Rename to `K` during load.

5. **Pitcher position assignment:** Compute from GS (games started):
   - `Position = 'SP'` if `GS >= 3`
   - `Position = 'RP'` otherwise

6. **Hitter position fallback:** If a hitter's MLBAMID is not found in the database, assign `Position = 'DH'`. DH players can only fill UTIL slots.

7. **Ratio stat team totals:** Do NOT sum OPS/ERA/WHIP. Compute weighted averages:
   ```python
   team_ops = (hitters['PA'] * hitters['OPS']).sum() / hitters['PA'].sum()
   ```

8. **Ratio stat linearization—sign flips:**
   - OPS (higher is better): coefficient = `PA * (OPS_player - OPS_opponent)`
   - ERA (lower is better): coefficient = `IP * (ERA_opponent - ERA_player)` — note flipped order
   - WHIP (lower is better): coefficient = `IP * (WHIP_opponent - WHIP_player)`

9. **Filter by player_type for category constraints:**
   - Hitting category constraints (R, HR, RBI, SB, OPS) sum only over `I_H`
   - Pitching category constraints (W, SV, K, ERA, WHIP) sum only over `I_P`
   - Common bug: accidentally including pitchers in OPS calculation

10. **Big-M values:**
    - Counting stats: `B = 10000`
    - Ratio stat linearized forms: `B = 5000`
    - Too small = falsely constrains feasible region
    - Too large = numerical precision issues with solver

11. **Epsilon values:**
    - Counting stats: `ε = 0.5` (ensures strict inequality for integer-valued stats)
    - Ratio stats: `ε = 0.001` (small positive for continuous-valued stats)

12. **Variable naming in PuLP:** Use only alphanumeric characters and underscores. `f"x_{i}"` is safe (i is an integer index). Never put player names in variable names.

13. **Solver status check:** Assert `status == pulp.LpStatusOptimal`. Any other status means the problem is infeasible or unbounded—crash with a clear message identifying which position slot is problematic.

14. **Zero IP or zero PA:** If a roster has no pitchers (IP=0), ERA/WHIP are undefined. Assert `sum(IP) > 0` for pitcher pools. Same for PA with hitters. **This applies to opponent rosters too**—assert for all teams.

15. **tqdm in notebooks:** Import from `tqdm.auto` for proper display:
    ```python
    from tqdm.auto import tqdm
    ```

16. **Globally unique names with -H/-P suffix:** ALL player names are suffixed with `-H` (hitters) or `-P` (pitchers) at load time. This makes names globally unique and eliminates all disambiguation complexity:
    - `load_hitter_projections`: Appends `-H` to all names
    - `load_pitcher_projections`: Appends `-P` to all names  
    - `parse_fantrax_roster`: Appends suffix based on Hitting/Pitching section
    - "Luis Castillo-P" (SEA pitcher) is now distinct from "Luis Castillo-H" (MIL hitter)
    - "Shohei Ohtani-H" and "Shohei Ohtani-P" are just two entries like any other player
    - No special two-way player handling needed anywhere

17. **Name normalization for matching:** After adding suffixes, use normalized name + team for matching. Name normalization handles accents and apostrophe variations:
    - "Julio Rodriguez-H" normalizes to "julio rodriguez-h"
    - "Julio Rodríguez-H" also normalizes to "julio rodriguez-h" 
    - They match because accents are stripped

18. **Display stripping:** When displaying names to users, strip the -H/-P suffix for cleaner output (position already indicates player type).

18. **Duplicate names in projections:** FanGraphs includes minor league players who may share names with MLB players (e.g., multiple "Luis Rodriguez" entries). Keep only the first occurrence when loading (first = higher projected). Print a note about removed duplicates.

19. **Roto points calculation:** Points = Σ (8 - rank) for each category. If rank is 1, you get 7 points. If rank is 7, you get 1 point. Maximum is 70 (first in all 10 categories).

20. **Bench players contribute stats:** All rostered players' stats count toward team totals. The starting lineup slot constraints only ensure positional validity—bench players are not treated differently.

21. **Roster changes output:** When showing dropped/added players, display full stats (PA, R, HR, RBI, SB, OPS for hitters; IP, W, SV, K, ERA, WHIP for pitchers) so users can understand WHY the optimizer is making changes. Flag suspicious low-PA/IP players with warning symbols.

---

## Final Checklist

Before the agent considers the implementation complete:

### Core Implementation
1. ☐ All functions are module-level functions (no classes, no methods)
2. ☐ All assertions have descriptive error messages
3. ☐ No try/except blocks anywhere
4. ☐ No fallback logic or default error handling
5. ☐ tqdm used for any loop over opponents, candidates, or sensitivity calculations
6. ☐ print() used for status updates at key stages

### Data Loading
7. ☐ OPS used directly from CSV (NOT recomputed)
8. ☐ SO column renamed to K during pitcher load
9. ☐ Hitter positions loaded from SQLite database via MLBAMID join
10. ☐ Hitters missing from database default to Position='DH'
11. ☐ Pitcher position (SP/RP) computed from GS during load
12. ☐ Duplicate names in projections handled (keep first occurrence, print note)

### Fantrax Conversion
13. ☐ `parse_fantrax_roster()` parses multi-section Hitting/Pitching format
14. ☐ `parse_fantrax_roster()` extracts `player_type` (hitter/pitcher) from section
15. ☐ Fantrax status mapping (Act→active, Res→active, Min→prospect, IR→IR)
16. ☐ `convert_fantrax_rosters()` outputs: name, team, player_type, status columns
17. ☐ `player_type` column enables disambiguation (Luis Castillo pitcher vs hitter)

### Name Uniqueness and Matching
18. ☐ `load_hitter_projections()` appends `-H` to all player names
19. ☐ `load_pitcher_projections()` appends `-P` to all player names
20. ☐ `parse_fantrax_roster()` appends `-H/-P` based on Hitting/Pitching section
21. ☐ Names are globally unique (no special two-way player handling needed)
22. ☐ `_normalize_name_for_comparison()` removes accents, handles apostrophes, PRESERVES `-H/-P` suffix
23. ☐ `find_name_mismatches()` internal normalize_name() PRESERVES `-H/-P` suffix (CRITICAL!)
24. ☐ `known_corrections` dict includes `-H/-P` suffixes in both keys and values
25. ☐ Display functions strip `-H/-P` suffix for cleaner output

### MILP Formulation
26. ☐ Ratio stat linearization has correct sign for negative categories (ERA, WHIP)
27. ☐ Beat constraints filter by player_type (I_H for hitting, I_P for pitching)
28. ☐ Position eligibility computed via set membership check
29. ☐ Uses `pulp.HiGHS()` (Python bindings), NOT `pulp.HiGHS_CMD()`
30. ☐ Infeasibility error identifies which position slot is problematic
31. ☐ Assertions for sum(PA) > 0 and sum(IP) > 0 apply to ALL teams

### Output
30. ☐ Roster changes section shows full player stats (PA/IP, R/W, etc.)
31. ☐ Low PA/IP players flagged with warning symbols in output
32. ☐ All visualizations return Figure objects (no plt.show() calls)
33. ☐ `plot_player_contribution_radar()` implemented for multi-category player view

### Project Structure
34. ☐ `pyproject.toml` exists with correct dependencies and hatch config
35. ☐ `optimizer/__init__.py` exists
36. ☐ `data/raw_rosters/` directory for Fantrax exports
36. ☐ Sample roster conversion tested and working