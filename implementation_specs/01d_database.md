# Database Schema and Sync

## Overview

A SQLite database serves as the **primary source of truth** for all player data. DataFrames used in the optimizer and trade engine are query results from this database, not independent data structures.

**Module:** `optimizer/database.py`  
**Database:** `data/optimizer.db`

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style
- [01a_config.md](01a_config.md) — constants for validation
- [01b_fangraphs_loading.md](01b_fangraphs_loading.md) — `load_projections()` output to sync
- [01c_fantrax_api.md](01c_fantrax_api.md) — `fetch_all_fantrax_data()` output to sync
- [01e_dynasty_valuation.md](01e_dynasty_valuation.md) — `compute_dynasty_sgp()` for dynasty values
- [01f_mlb_stats_api.md](01f_mlb_stats_api.md) — `fetch_player_ages()` for age data

**Used by:**
- [02_free_agent_optimizer.md](02_free_agent_optimizer.md) — `get_projections()`, `get_roster_names()`
- [03_trade_engine.md](03_trade_engine.md) — `get_projections()`, `get_roster_names()`
- [05_notebook_integration.md](05_notebook_integration.md) — `refresh_all_data()` entry point
- [06_streamlit_dashboard.md](06_streamlit_dashboard.md) — `refresh_all_data()` entry point

---

## Design Philosophy

1. **Database is primary** — All data queries pull from the database
2. **DataFrames are views** — Any DataFrame in the optimizer is a query result
3. **One wide `players` table** — Merges FanGraphs projections + API metadata
4. **Separate `standings` table** — Fantasy league standings
5. **No historical tracking** — Current state only (overwrite on refresh)
6. **Data sources**: FanGraphs (projections), MLB API (ages), Fantrax (ownership, positions)

---

## Schema

### Players Table

```sql
CREATE TABLE IF NOT EXISTS players (
    -- Primary key is the suffixed name (guaranteed unique)
    name TEXT PRIMARY KEY,              -- With -H/-P suffix: "Mike Trout-H"
    display_name TEXT NOT NULL,         -- Without suffix: "Mike Trout"
    player_type TEXT NOT NULL,          -- 'hitter' or 'pitcher'
    
    -- MLB identifiers
    mlbamid INTEGER,                    -- MLBAM ID from FanGraphs (for MLB API lookups)
    
    -- Position (from Fantrax API or derived)
    position TEXT NOT NULL,             -- C, 1B, 2B, SS, 3B, OF, DH, SP, RP
    
    -- Team
    team TEXT,                          -- MLB team abbreviation, or NULL
    
    -- From FanGraphs Projections
    pa INTEGER DEFAULT 0,               -- Plate appearances (hitters)
    r INTEGER DEFAULT 0,                -- Runs
    hr INTEGER DEFAULT 0,               -- Home runs
    rbi INTEGER DEFAULT 0,              -- RBI
    sb INTEGER DEFAULT 0,               -- Stolen bases
    ops REAL DEFAULT 0,                 -- On-base plus slugging
    
    ip REAL DEFAULT 0,                  -- Innings pitched (pitchers)
    w INTEGER DEFAULT 0,                -- Wins
    sv INTEGER DEFAULT 0,               -- Saves
    k INTEGER DEFAULT 0,                -- Strikeouts
    era REAL DEFAULT 0,                 -- Earned run average
    whip REAL DEFAULT 0,                -- Walks + hits per inning
    
    war REAL DEFAULT 0,                 -- Wins above replacement
    
    -- From Fantrax
    age INTEGER,                        -- Player age
    owner TEXT,                         -- Fantasy team name, or NULL if free agent
    roster_status TEXT,                 -- 'active', 'reserve', 'IR', 'minors'
    adp REAL,                           -- Average draft position
    
    -- Computed values
    sgp REAL,                           -- Standing Gain Points
    dynasty_sgp REAL,                   -- Age-adjusted SGP
    quality_score REAL,                 -- For candidate filtering
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_players_owner ON players(owner);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_players_sgp ON players(sgp DESC);
CREATE INDEX IF NOT EXISTS idx_players_player_type ON players(player_type);
CREATE INDEX IF NOT EXISTS idx_players_mlbamid ON players(mlbamid);
```

### Standings Table

```sql
CREATE TABLE IF NOT EXISTS standings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date DATE NOT NULL,
    
    team_name TEXT NOT NULL,
    overall_rank INTEGER,
    total_points REAL,
    
    -- Category values (actual season totals)
    cat_r INTEGER, cat_hr INTEGER, cat_rbi INTEGER, cat_sb INTEGER, cat_ops REAL,
    cat_w INTEGER, cat_sv INTEGER, cat_k INTEGER, cat_era REAL, cat_whip REAL,
    
    -- Category ranks (1-7)
    rank_r INTEGER, rank_hr INTEGER, rank_rbi INTEGER, rank_sb INTEGER, rank_ops INTEGER,
    rank_w INTEGER, rank_sv INTEGER, rank_k INTEGER, rank_era INTEGER, rank_whip INTEGER,
    
    UNIQUE(as_of_date, team_name)
);
```

---

## Initialization

```python
def initialize_database(db_path: str = "data/optimizer.db") -> None:
    """
    Create database and tables if they don't exist.
    
    Implementation:
        1. Create parent directories if needed
        2. Connect to SQLite
        3. Execute CREATE TABLE IF NOT EXISTS for both tables
        4. Create indexes
    
    Print: "Initialized database at {db_path}"
    """
```

---

## Sync Functions

### Sync FanGraphs Projections

```python
def sync_fangraphs_to_db(
    projections: pd.DataFrame,
    db_path: str = "data/optimizer.db",
) -> int:
    """
    Sync FanGraphs projections to players table.
    
    Args:
        projections: DataFrame from load_projections() with columns:
            Name, Team, Position, player_type, MLBAMID, PA, R, HR, RBI, SB, OPS,
            IP, W, SV, K, ERA, WHIP, WAR, SGP
        db_path: Path to SQLite database
    
    Returns:
        Number of players synced
    
    Implementation:
        Use INSERT ... ON CONFLICT(name) DO UPDATE to:
        - Create new players if they don't exist
        - Update projection columns (pa, r, hr, etc.) AND mlbamid for existing players
        - PRESERVE Fantrax columns (owner, roster_status, age, dynasty_sgp)
        - PRESERVE position column (do NOT overwrite with FanGraphs position)
        
        CRITICAL: Do NOT use INSERT OR REPLACE - it deletes and recreates
        the row, wiping out ownership data.
        
        CRITICAL: Do NOT update position in ON CONFLICT clause! FanGraphs
        hitter CSV has no position data (defaults to UTIL). Overwriting
        would erase good Fantrax positions. Positions should only be set
        on initial INSERT, then updated by sync_player_pool_to_db().
        
        NOTE: mlbamid IS updated on conflict - it comes from FanGraphs and
        is needed for MLB API age lookups (see 01f_mlb_stats_api.md).
    
    Print: "Synced {N} FanGraphs projections to database"
    """
```

### Sync Fantrax Rosters

```python
def sync_fantrax_rosters_to_db(
    roster_sets: dict[str, set[str]],
    roster_details: dict[str, list[dict]],
    db_path: str = "data/optimizer.db",
) -> int:
    """
    Sync Fantrax roster ownership to players table.
    
    Note: Ages come from MLB Stats API, not Fantrax.
    
    Args:
        roster_sets: Dict mapping team_name to set of player names (with suffix)
        roster_details: Dict mapping team_name to list of player dicts with roster_status, etc.
        db_path: Path to SQLite database
    
    Returns:
        Number of players updated
    
    Implementation:
        1. Clear all ownership: UPDATE players SET owner = NULL
        2. For each team's roster:
           - UPDATE players SET owner = team_name, roster_status = ...
             WHERE name = player_name
        3. Report players in rosters but not in database (logged as warnings)
    
    Print: 
        "Cleared ownership for all players"
        "Updated ownership for {N} rostered players across 7 teams"
        "WARNING: {M} rostered players not found in projections" (if any)
    """
```

### Sync Player Pool (Positions)

```python
def sync_player_pool_to_db(
    player_pool: pd.DataFrame,
    db_path: str = "data/optimizer.db",
) -> dict[str, int]:
    """
    Sync positions from Fantrax player pool to all players in database.
    
    Why this is needed:
        - FanGraphs hitter CSV has no position column → defaults to "UTIL"
        - sync_fantrax_rosters_to_db() only updates ROSTERED players
        - Free agents (owner IS NULL) never get position updates otherwise
    
    Note: Ages come from MLB Stats API (sync_ages_to_db), not Fantrax.
    
    Args:
        player_pool: DataFrame from fetch_player_pool() with name, position columns
        db_path: Path to SQLite database
    
    Returns:
        Dict with counts: {"position_updated": N, "not_found": K}
    
    Implementation:
        1. Build normalized name lookup from database (handles accents, etc.)
        2. Single pass through player_pool:
           - Apply name corrections
           - Add -H or -P suffix based on player_type
           - Match to database via normalized name
           - UPDATE players SET position = ? WHERE name = db_name
           
        Only updates existing rows (doesn't create new players).
    
    Print: 
        "Synced player pool: {N} positions updated"
        "  ({K} players not found in database)" (if any)
    """
```

### Sync Ages from MLB API

```python
def sync_ages_to_db(ages_df: pd.DataFrame, db_path: str = "data/optimizer.db") -> int:
    """
    Sync ages from MLB Stats API to database.
    
    Args:
        ages_df: DataFrame from fetch_player_ages() with mlbam_id, age columns
        db_path: Path to database
    
    Returns:
        Number of players updated
    
    Implementation:
        UPDATE players SET age = ? WHERE mlbamid = ?
        
        Only updates players that exist (matched by mlbamid).
    """
```

### Compute Dynasty SGP

```python
def compute_dynasty_sgp_in_db(db_path: str = "data/optimizer.db") -> int:
    """
    Compute dynasty_sgp for all players with age data.
    
    Implementation:
        1. Query all players with age and sgp
        2. For each player, compute dynasty_sgp using compute_dynasty_sgp()
        3. UPDATE players SET dynasty_sgp = ... WHERE name = ...
        4. Players without age: dynasty_sgp = sgp
    
    Returns:
        Number of players with dynasty_sgp computed
    
    Print: 
        "Computed dynasty SGP for {N} players with age data"
        "{M} players missing age - dynasty_sgp = sgp"
    """
```

---

## Query Functions

These functions return DataFrames from database queries. **All optimizer/trade engine code should use these functions rather than loading CSVs directly.**

```python
def get_projections(db_path: str = "data/optimizer.db") -> pd.DataFrame:
    """
    Get all players as a projections DataFrame.
    
    Returns DataFrame with columns matching load_projections() output:
        Name, Team, Position, player_type, PA, R, HR, RBI, SB, OPS,
        IP, W, SV, K, ERA, WHIP, WAR, SGP, age, dynasty_sgp, quality_score
    
    This is the PRIMARY way to get projection data.
    """


def get_roster_names(
    team_name: str,
    db_path: str = "data/optimizer.db",
    active_only: bool = True,
) -> set[str]:
    """
    Get player names on a team's roster.
    
    Args:
        team_name: Fantasy team name (e.g., "The Big Dumpers")
        db_path: Database path
        active_only: If True, exclude IR/minors players
    
    Returns:
        Set of player names (with -H/-P suffix)
    """


def get_all_roster_names(
    db_path: str = "data/optimizer.db",
    active_only: bool = True,
) -> dict[str, set[str]]:
    """
    Get roster names for all teams.
    
    Returns:
        Dict mapping team_name to set of player names
        {"The Big Dumpers": {"Mike Trout-H", ...}, ...}
    """


def get_free_agents(
    db_path: str = "data/optimizer.db",
    position: str | None = None,
    min_sgp: float | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    """
    Query free agents (players with owner IS NULL).
    
    Args:
        position: Filter to specific position
        min_sgp: Minimum SGP threshold
        limit: Maximum rows to return
    
    Returns:
        DataFrame sorted by SGP descending
    """


def get_standings(db_path: str = "data/optimizer.db") -> pd.DataFrame:
    """
    Get most recent standings.
    
    Returns DataFrame with team standings and category ranks.
    """
```

---

## Data Refresh Pipeline

```python
def refresh_all_data(
    hitter_proj_path: str = "data/fangraphs-atc-projections-hitters.csv",
    pitcher_proj_path: str = "data/fangraphs-atc-projections-pitchers.csv",
    db_path: str = "data/optimizer.db",
    skip_fantrax: bool = False,
    skip_mlb_api: bool = False,
) -> dict:
    """
    Complete data refresh: FanGraphs + MLB API + Fantrax → database.
    
    This is the MAIN ENTRY POINT for loading data.
    
    Pipeline:
        1. Initialize database (create tables if needed)
        2. Load FanGraphs projections → sync to database
        3. Fetch ages from MLB Stats API → sync to database
        4. If not skip_fantrax:
           a. Fetch Fantrax data (rosters, standings, player pool)
           b. Sync rosters (ownership)
           c. Sync player pool (positions)
           d. Sync standings
        5. Compute dynasty_sgp for all players
        6. Return data from database queries
    
    Args:
        hitter_proj_path: Path to FanGraphs hitter projections CSV
        pitcher_proj_path: Path to FanGraphs pitcher projections CSV
        db_path: Path to SQLite database
        skip_fantrax: If True, skip Fantrax API calls (use existing DB data)
        skip_mlb_api: If True, skip MLB Stats API call for ages
    
    Returns:
        {
            "projections": pd.DataFrame,
            "my_roster": set[str],
            "opponent_rosters": dict[str, set[str]],
            "standings": pd.DataFrame,
        }
    """
```

---

## Usage Pattern

The optimizer and trade engine should follow this pattern:

```python
# Load all data (refreshes database)
data = refresh_all_data()

# Access data via returned dict (which came from DB queries)
projections = data["projections"]
my_roster = data["my_roster"]
opponent_rosters = data["opponent_rosters"]

# Or query database directly for specific needs
free_agents = get_free_agents(min_sgp=5.0, limit=100)
```

---

## Validation Checklist

```python
# After refresh_all_data:
import sqlite3

conn = sqlite3.connect(db_path)

player_count = pd.read_sql("SELECT COUNT(*) as n FROM players", conn).iloc[0, 0]
assert player_count > 1000, f"Expected >1000 players, got {player_count}"

rostered_count = pd.read_sql("SELECT COUNT(*) as n FROM players WHERE owner IS NOT NULL", conn).iloc[0, 0]
assert 150 <= rostered_count <= 200, f"Expected ~182 rostered, got {rostered_count}"

with_sgp = pd.read_sql("SELECT COUNT(*) as n FROM players WHERE sgp IS NOT NULL", conn).iloc[0, 0]
assert with_sgp == player_count, "All players should have SGP"

with_age = pd.read_sql("SELECT COUNT(*) as n FROM players WHERE age IS NOT NULL", conn).iloc[0, 0]
print(f"Players with age data: {with_age}/{player_count}")

# Position validation: most hitters should have real positions, not UTIL
hitter_count = pd.read_sql(
    "SELECT COUNT(*) as n FROM players WHERE player_type = 'hitter'", conn
).iloc[0, 0]
util_count = pd.read_sql(
    "SELECT COUNT(*) as n FROM players WHERE player_type = 'hitter' AND position = 'UTIL'", conn
).iloc[0, 0]
util_pct = util_count / hitter_count * 100
print(f"Hitters with UTIL position: {util_count}/{hitter_count} ({util_pct:.1f}%)")
# After position sync, UTIL should be <60% (only minor leaguers not in Fantrax top 5000)
assert util_pct < 60, f"Too many UTIL hitters ({util_pct:.1f}%) - position sync may have failed"

conn.close()
```
