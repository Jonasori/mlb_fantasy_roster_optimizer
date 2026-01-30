"""
SQLite database schema, sync functions, and queries.

The database is the PRIMARY source of truth for all player data.
DataFrames used in the optimizer and trade engine are query results from this database.
"""

import sqlite3
from datetime import date
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from .data_loader import (
    MY_TEAM_NAME,
    compute_quality_scores,
    load_projections,
    normalize_name,
    strip_name_suffix,
)
from .fantrax_api import (
    FANTRAX_NAME_CORRECTIONS,
    create_session,
    fetch_all_fantrax_data,
    get_player_type,
    test_auth,
)
from .mlb_api import fetch_player_ages

# =============================================================================
# SCHEMA
# =============================================================================

PLAYERS_SCHEMA = """
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
    
    -- From FanGraphs Projections (primary projection source)
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
    
    -- Historical actual stats (for playing time adjustment)
    pa_actual_2024 INTEGER,             -- Actual PA from 2024 season
    pa_actual_2023 INTEGER,             -- Actual PA from 2023 season
    ip_actual_2024 REAL,                -- Actual IP from 2024 season
    ip_actual_2023 REAL,                -- Actual IP from 2023 season
    
    -- Playing time adjusted projections
    pa_adjusted INTEGER,                -- PA after playing time adjustment
    ip_adjusted REAL,                   -- IP after playing time adjustment
    
    -- Multiple projection source support (for combination)
    pa_steamer INTEGER,                 -- Steamer projected PA
    pa_zips INTEGER,                    -- ZiPS projected PA
    pa_thebat INTEGER,                  -- THE BAT projected PA
    pa_depthcharts INTEGER,             -- FanGraphs Depth Charts PA
    ip_steamer REAL,                    -- Steamer projected IP
    ip_zips REAL,                       -- ZiPS projected IP
    ip_thebat REAL,                     -- THE BAT projected IP
    ip_depthcharts REAL,                -- FanGraphs Depth Charts IP
    
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
)
"""

PLAYERS_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_players_owner ON players(owner);
CREATE INDEX IF NOT EXISTS idx_players_position ON players(position);
CREATE INDEX IF NOT EXISTS idx_players_sgp ON players(sgp DESC);
CREATE INDEX IF NOT EXISTS idx_players_player_type ON players(player_type);
CREATE INDEX IF NOT EXISTS idx_players_mlbamid ON players(mlbamid);
"""

STANDINGS_SCHEMA = """
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
)
"""


# =============================================================================
# INITIALIZATION
# =============================================================================


def initialize_database(db_path: str = "data/optimizer.db") -> None:
    """
    Create database and tables if they don't exist.
    If schema is outdated, deletes and recreates the database.
    """
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)

    # Check if existing database has correct schema
    if db_file.exists():
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(players)")
        existing_columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        # Required columns - if any missing, delete and recreate
        required = {"mlbamid", "pa_adjusted", "ip_adjusted"}
        if not required.issubset(existing_columns):
            print(f"  Schema outdated, recreating database...")
            db_file.unlink()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.executescript(PLAYERS_SCHEMA)
    cursor.executescript(PLAYERS_INDEXES)
    cursor.executescript(STANDINGS_SCHEMA)

    conn.commit()
    conn.close()

    print(f"Initialized database at {db_path}")


# =============================================================================
# SYNC FUNCTIONS
# =============================================================================


def sync_fangraphs_to_db(
    projections: pd.DataFrame,
    db_path: str = "data/optimizer.db",
) -> int:
    """
    Sync FanGraphs projections to players table.

    Args:
        projections: DataFrame from load_projections() with columns:
            Name, Team, Position, player_type, PA, R, HR, RBI, SB, OPS,
            IP, W, SV, K, ERA, WHIP, WAR, SGP
        db_path: Path to SQLite database

    Returns:
        Number of players synced
    """
    # Compute quality scores
    quality_df = compute_quality_scores(projections)
    projections = projections.merge(
        quality_df[["Name", "quality_score"]], on="Name", how="left"
    )

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    synced = 0
    for _, row in tqdm(
        projections.iterrows(), total=len(projections), desc="Syncing projections"
    ):
        # Use INSERT ... ON CONFLICT to preserve ownership data (owner, roster_status, age, dynasty_sgp)
        # NOTE: mlbamid IS updated on conflict - it comes from FanGraphs and is needed for MLB API age lookups
        cursor.execute(
            """
            INSERT INTO players 
            (name, display_name, player_type, mlbamid, position, team,
             pa, r, hr, rbi, sb, ops,
             ip, w, sv, k, era, whip,
             war, sgp, quality_score, last_updated)
            VALUES (?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(name) DO UPDATE SET
                display_name = excluded.display_name,
                player_type = excluded.player_type,
                mlbamid = excluded.mlbamid,
                -- NOTE: Do NOT update position here! Positions come from Fantrax sync.
                -- FanGraphs hitter CSV has no position data (defaults to UTIL).
                -- Overwriting would erase good Fantrax positions.
                team = excluded.team,
                pa = excluded.pa,
                r = excluded.r,
                hr = excluded.hr,
                rbi = excluded.rbi,
                sb = excluded.sb,
                ops = excluded.ops,
                ip = excluded.ip,
                w = excluded.w,
                sv = excluded.sv,
                k = excluded.k,
                era = excluded.era,
                whip = excluded.whip,
                war = excluded.war,
                sgp = excluded.sgp,
                quality_score = excluded.quality_score,
                last_updated = CURRENT_TIMESTAMP
        """,
            (
                row["Name"],
                strip_name_suffix(row["Name"]),
                row["player_type"],
                int(row["MLBAMID"]) if pd.notna(row.get("MLBAMID")) else None,
                row["Position"],
                row["Team"] if row["Team"] != "FA" else None,
                int(row["PA"]),
                int(row["R"]),
                int(row["HR"]),
                int(row["RBI"]),
                int(row["SB"]),
                float(row["OPS"]),
                float(row["IP"]),
                int(row["W"]),
                int(row["SV"]),
                int(row["K"]),
                float(row["ERA"]),
                float(row["WHIP"]),
                float(row["WAR"]),
                float(row["SGP"]),
                float(row["quality_score"])
                if pd.notna(row.get("quality_score"))
                else None,
            ),
        )
        synced += 1

    conn.commit()
    conn.close()

    print(f"Synced {synced} FanGraphs projections to database")
    return synced


def sync_fantrax_rosters_to_db(
    roster_sets: dict[str, set[str]],
    roster_details: dict[str, list[dict]],
    db_path: str = "data/optimizer.db",
) -> int:
    """
    Sync Fantrax roster ownership to players table.

    Uses normalized name matching to handle accent differences between
    Fantrax (unaccented) and FanGraphs (accented) names.

    Args:
        roster_sets: Dict mapping team_name to set of player names (with suffix)
        roster_details: Dict mapping team_name to list of player dicts with age, position, etc.
        db_path: Path to SQLite database

    Returns:
        Number of players updated
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Clear all ownership
    cursor.execute("UPDATE players SET owner = NULL, roster_status = NULL")
    print("Cleared ownership for all players")

    # Build normalized name lookup from database: normalized_name -> actual_name
    # This allows us to match "eugenio suarez-h" to "Eugenio Suárez-H"
    cursor.execute("SELECT name FROM players")
    all_db_names = [row[0] for row in cursor.fetchall()]

    normalized_to_actual = {}
    for actual_name in all_db_names:
        normalized = normalize_name(actual_name)
        normalized_to_actual[normalized] = actual_name

    # Build lookup from roster_details: name -> (age, status, position)
    player_info = {}
    for team_name, players in roster_details.items():
        for player in players:
            raw_name = player["name"]

            # Skip players with no name (empty roster slots)
            if raw_name is None:
                continue

            # Apply exceptional corrections (truly different names only)
            if raw_name in FANTRAX_NAME_CORRECTIONS:
                raw_name = FANTRAX_NAME_CORRECTIONS[raw_name]

            # Add suffix only if not already present (Fantrax sometimes includes it)
            if raw_name.endswith("-H") or raw_name.endswith("-P"):
                suffixed_name = raw_name
            else:
                suffix = "-P" if player["player_type"] == "pitcher" else "-H"
                suffixed_name = raw_name + suffix

            # Map status_id to human-readable status for database storage
            status_id = player.get("status_id", "")
            status = {"1": "active", "2": "reserve", "3": "minors", "4": "IR"}.get(
                status_id, "unknown"
            )

            player_info[suffixed_name] = {
                "age": player.get("age"),
                "status": status,
                "position": player.get("position"),
            }

    updated = 0
    not_found = []

    for team_name, names in roster_sets.items():
        for fantrax_name in names:
            info = player_info.get(fantrax_name, {})

            # Try to find matching DB name via normalization
            normalized = normalize_name(fantrax_name)
            db_name = normalized_to_actual.get(normalized)

            if db_name is None:
                not_found.append(fantrax_name)
                continue

            cursor.execute(
                """
                UPDATE players 
                SET owner = ?, 
                    roster_status = ?,
                    age = COALESCE(?, age),
                    position = COALESCE(?, position)
                WHERE name = ?
            """,
                (
                    team_name,
                    info.get("status"),
                    info.get("age"),
                    info.get("position"),
                    db_name,
                ),
            )
            updated += 1

    conn.commit()
    conn.close()

    print(f"Updated ownership for {updated} rostered players across 7 teams")
    if not_found:
        print(f"WARNING: {len(not_found)} rostered players not found in projections:")
        for name in sorted(not_found)[:10]:
            print(f"  - {strip_name_suffix(name)}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    return updated


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
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build normalized name lookup from database: normalized_name -> actual_name
    # This allows matching "eugenio suarez-h" to "Eugenio Suárez-H"
    cursor.execute("SELECT name FROM players")
    all_db_names = [row[0] for row in cursor.fetchall()]

    normalized_to_actual = {}
    for actual_name in all_db_names:
        normalized = normalize_name(actual_name)
        normalized_to_actual[normalized] = actual_name

    position_updated = 0
    not_found = 0

    for _, row in tqdm(
        player_pool.iterrows(), total=len(player_pool), desc="Syncing player pool"
    ):
        raw_name = row["name"]

        # Skip if no name
        if raw_name is None or pd.isna(raw_name):
            continue

        # Apply name corrections
        if raw_name in FANTRAX_NAME_CORRECTIONS:
            raw_name = FANTRAX_NAME_CORRECTIONS[raw_name]

        # Determine player type and add suffix
        position = row.get("position", "")
        player_type = row.get("player_type", get_player_type(position))
        suffix = "-P" if player_type == "pitcher" else "-H"
        suffixed_name = raw_name + suffix

        # Try to find matching DB name via normalization
        normalized = normalize_name(suffixed_name)
        db_name = normalized_to_actual.get(normalized)

        if db_name is None:
            not_found += 1
            continue

        # Only update position (ages come from MLB Stats API)
        has_position = position and not pd.isna(position) and position != ""
        if has_position:
            cursor.execute(
                "UPDATE players SET position = ? WHERE name = ?",
                (position, db_name),
            )
            if cursor.rowcount > 0:
                position_updated += 1

    conn.commit()
    conn.close()

    print(f"Synced player pool: {position_updated} positions updated")
    if not_found > 0:
        print(f"  ({not_found} players not found in database)")

    return {
        "position_updated": position_updated,
        "not_found": not_found,
    }


# Dynasty SGP constants (from 01e_dynasty_valuation.md)
DYNASTY_DISCOUNT_RATE = 0.25
DYNASTY_PROJECTION_YEARS = 4

HITTER_AGING_FACTORS = {
    22: 0.0,
    23: 0.0,
    24: 0.0,
    25: 0.0,
    26: 0.0,
    27: -0.8,
    28: -0.9,
    29: -1.0,
    30: -1.1,
    31: -1.2,
    32: -1.4,
    33: -1.6,
    34: -1.8,
    35: -2.2,
    36: -2.6,
    37: -3.0,
    38: -3.5,
    39: -4.0,
    40: -4.5,
}

PITCHER_AGING_FACTORS = {
    22: 0.0,
    23: 0.0,
    24: 0.0,
    25: 0.0,
    26: 0.0,
    27: 0.0,
    28: 0.0,
    29: 0.0,
    30: 0.0,
    31: 0.0,
    32: -0.6,
    33: -0.8,
    34: -1.0,
    35: -1.3,
    36: -1.6,
    37: -2.0,
    38: -2.5,
    39: -3.0,
    40: -3.5,
}


def _compute_dynasty_sgp_row(row: pd.Series) -> float:
    """Compute dynasty SGP for a single player row."""
    sgp = row["sgp"]
    age = row["age"]
    player_type = row["player_type"]

    # If no age, dynasty_sgp = sgp
    if pd.isna(age):
        return sgp

    age = int(age)
    aging_factors = (
        HITTER_AGING_FACTORS if player_type == "hitter" else PITCHER_AGING_FACTORS
    )

    dynasty_sgp = 0.0
    current_sgp = sgp

    for year in range(DYNASTY_PROJECTION_YEARS):
        discount = 1.0 / ((1 + DYNASTY_DISCOUNT_RATE) ** year)
        dynasty_sgp += max(0, current_sgp) * discount

        # Apply aging decline for next year
        future_age = age + year + 1
        decline = aging_factors.get(future_age, -4.5)  # Default to steep decline
        current_sgp += decline

    return dynasty_sgp


def compute_dynasty_sgp_in_db(db_path: str = "data/optimizer.db") -> int:
    """
    Compute dynasty_sgp for all players with age data.

    Returns:
        Number of players with dynasty_sgp computed
    """
    conn = sqlite3.connect(db_path)

    df = pd.read_sql(
        """
        SELECT name, sgp, age, player_type, position
        FROM players
        WHERE sgp IS NOT NULL
        """,
        conn,
    )

    # Compute dynasty SGP
    df["dynasty_sgp"] = df.apply(_compute_dynasty_sgp_row, axis=1)

    # Update database
    cursor = conn.cursor()
    with_age = 0
    without_age = 0

    for _, row in df.iterrows():
        cursor.execute(
            "UPDATE players SET dynasty_sgp = ? WHERE name = ?",
            (float(row["dynasty_sgp"]), row["name"]),
        )

        if pd.notna(row["age"]):
            with_age += 1
        else:
            without_age += 1

    conn.commit()
    conn.close()

    print(f"Computed dynasty SGP for {with_age} players with age data")
    print(f"{without_age} players missing age - dynasty_sgp = sgp")

    return with_age


def sync_standings_to_db(
    standings: pd.DataFrame,
    db_path: str = "data/optimizer.db",
) -> int:
    """
    Sync standings to standings table.

    Returns:
        Number of teams synced
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    today = date.today().isoformat()
    synced = 0

    for _, row in standings.iterrows():
        cursor.execute(
            """
            INSERT OR REPLACE INTO standings 
            (as_of_date, team_name, overall_rank, total_points,
             cat_r, cat_hr, cat_rbi, cat_sb, cat_ops,
             cat_w, cat_sv, cat_k, cat_era, cat_whip,
             rank_r, rank_hr, rank_rbi, rank_sb, rank_ops,
             rank_w, rank_sv, rank_k, rank_era, rank_whip)
            VALUES (?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?)
        """,
            (
                today,
                row.get("team_name"),
                row.get("overall_rank"),
                row.get("total_points"),
                row.get("r"),
                row.get("hr"),
                row.get("rbi"),
                row.get("sb"),
                row.get("ops"),
                row.get("w"),
                row.get("sv"),
                row.get("k"),
                row.get("era"),
                row.get("whip"),
                row.get("r_rank"),
                row.get("hr_rank"),
                row.get("rbi_rank"),
                row.get("sb_rank"),
                row.get("ops_rank"),
                row.get("w_rank"),
                row.get("sv_rank"),
                row.get("k_rank"),
                row.get("era_rank"),
                row.get("whip_rank"),
            ),
        )
        synced += 1

    conn.commit()
    conn.close()

    print(f"Synced standings for {synced} teams")
    return synced


def sync_ages_to_db(ages_df: pd.DataFrame, db_path: str = "data/optimizer.db") -> int:
    """
    Sync ages from MLB Stats API to database.

    Args:
        ages_df: DataFrame from fetch_player_ages() with mlbam_id, age columns
        db_path: Path to database

    Returns:
        Number of players updated
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    updated = 0
    for _, row in tqdm(ages_df.iterrows(), total=len(ages_df), desc="Syncing ages"):
        mlbam_id = row.get("mlbam_id")
        age = row.get("age")

        if mlbam_id is None or pd.isna(mlbam_id):
            continue
        if age is None or pd.isna(age):
            continue

        cursor.execute(
            "UPDATE players SET age = ? WHERE mlbamid = ?",
            (int(age), int(mlbam_id)),
        )
        if cursor.rowcount > 0:
            updated += 1

    conn.commit()
    conn.close()

    print(f"Synced ages for {updated} players from MLB Stats API")
    return updated


# =============================================================================
# HISTORICAL STATS SYNC
# =============================================================================


def sync_historical_stats_to_db(
    actuals_df: pd.DataFrame,
    year: int,
    db_path: str = "data/optimizer.db",
) -> dict[str, int]:
    """
    Sync historical actual stats (PA/IP) to database.

    This is essential for playing time adjustment - historical PA/IP is
    the strongest predictor of future playing time.

    Args:
        actuals_df: DataFrame with actual stats. Required columns:
            - Name: Player name (with or without -H/-P suffix)
            - PA: Plate appearances (for hitters)
            - IP: Innings pitched (for pitchers)
            - player_type: 'hitter' or 'pitcher' (optional, inferred from PA/IP)
        year: Year of the data (2024, 2023, etc.)
        db_path: Path to SQLite database

    Returns:
        Dict with counts: {"hitters_updated": N, "pitchers_updated": M, "not_found": K}

    Note:
        Only 2024 and 2023 data are stored (most recent 2 years).
        Older data has diminishing predictive value.
    """
    assert year in (2024, 2023), f"Only 2024 and 2023 supported, got {year}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Build normalized name lookup
    cursor.execute("SELECT name FROM players")
    all_db_names = [row[0] for row in cursor.fetchall()]
    normalized_to_actual = {normalize_name(n): n for n in all_db_names}

    pa_col = f"pa_actual_{year}"
    ip_col = f"ip_actual_{year}"

    hitters_updated = 0
    pitchers_updated = 0
    not_found = 0

    for _, row in tqdm(
        actuals_df.iterrows(), total=len(actuals_df), desc=f"Syncing {year} actuals"
    ):
        name = row.get("Name") or row.get("name")
        if name is None or pd.isna(name):
            continue

        # Determine if hitter or pitcher
        pa = row.get("PA") or row.get("pa")
        ip = row.get("IP") or row.get("ip")

        is_hitter = pa is not None and not pd.isna(pa) and pa > 0
        is_pitcher = ip is not None and not pd.isna(ip) and ip > 0

        if not is_hitter and not is_pitcher:
            continue

        # Add suffix if not present
        if not (name.endswith("-H") or name.endswith("-P")):
            suffix = "-H" if is_hitter else "-P"
            name = name + suffix

        # Find in database
        normalized = normalize_name(name)
        db_name = normalized_to_actual.get(normalized)

        if db_name is None:
            not_found += 1
            continue

        # Update
        if is_hitter:
            cursor.execute(
                f"UPDATE players SET {pa_col} = ? WHERE name = ?",
                (int(pa), db_name),
            )
            if cursor.rowcount > 0:
                hitters_updated += 1
        else:
            cursor.execute(
                f"UPDATE players SET {ip_col} = ? WHERE name = ?",
                (float(ip), db_name),
            )
            if cursor.rowcount > 0:
                pitchers_updated += 1

    conn.commit()
    conn.close()

    print(
        f"Synced {year} actuals: {hitters_updated} hitters, {pitchers_updated} pitchers"
    )
    if not_found > 0:
        print(f"  ({not_found} players not found in database)")

    return {
        "hitters_updated": hitters_updated,
        "pitchers_updated": pitchers_updated,
        "not_found": not_found,
    }


# =============================================================================
# QUERY FUNCTIONS
# =============================================================================


def get_projections(
    db_path: str = "data/optimizer.db",
    use_adjusted_playing_time: bool = False,
) -> pd.DataFrame:
    """
    Get all players as a projections DataFrame.

    Returns DataFrame with columns matching load_projections() output:
        Name, Team, Position, player_type, PA, R, HR, RBI, SB, OPS,
        IP, W, SV, K, ERA, WHIP, WAR, SGP, age, dynasty_sgp, quality_score

    Args:
        db_path: Path to SQLite database
        use_adjusted_playing_time: If True and adjusted values exist,
            use pa_adjusted/ip_adjusted instead of pa/ip for PA/IP columns.

    This is the PRIMARY way to get projection data.
    """
    conn = sqlite3.connect(db_path)

    df = pd.read_sql(
        """
        SELECT 
            name as Name,
            team as Team,
            position as Position,
            player_type,
            mlbamid as MLBAMID,
            pa as PA,
            r as R,
            hr as HR,
            rbi as RBI,
            sb as SB,
            ops as OPS,
            ip as IP,
            w as W,
            sv as SV,
            k as K,
            era as ERA,
            whip as WHIP,
            war as WAR,
            sgp as SGP,
            age,
            dynasty_sgp,
            quality_score,
            owner,
            pa_adjusted,
            ip_adjusted,
            pa_actual_2024,
            pa_actual_2023,
            ip_actual_2024,
            ip_actual_2023
        FROM players
    """,
        conn,
    )

    conn.close()

    # Fill null Team with 'FA'
    df["Team"] = df["Team"].fillna("FA")

    # Optionally swap in adjusted playing time
    if use_adjusted_playing_time:
        # Use adjusted PA where available, else original
        df["PA_original"] = df["PA"]
        df["IP_original"] = df["IP"]

        df["PA"] = df["pa_adjusted"].fillna(df["PA"]).astype(int)
        df["IP"] = df["ip_adjusted"].fillna(df["IP"])

        adjusted_pa_count = df["pa_adjusted"].notna().sum()
        adjusted_ip_count = df["ip_adjusted"].notna().sum()
        print(
            f"Using adjusted playing time: {adjusted_pa_count} PA, {adjusted_ip_count} IP values"
        )

    return df


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
    conn = sqlite3.connect(db_path)

    if active_only:
        query = """
            SELECT name FROM players 
            WHERE owner = ? AND (roster_status IN ('active', 'reserve') OR roster_status IS NULL)
        """
    else:
        query = "SELECT name FROM players WHERE owner = ?"

    df = pd.read_sql(query, conn, params=(team_name,))
    conn.close()

    return set(df["name"])


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
    conn = sqlite3.connect(db_path)

    if active_only:
        query = """
            SELECT owner, name FROM players 
            WHERE owner IS NOT NULL AND (roster_status IN ('active', 'reserve') OR roster_status IS NULL)
        """
    else:
        query = "SELECT owner, name FROM players WHERE owner IS NOT NULL"

    df = pd.read_sql(query, conn)
    conn.close()

    rosters = {}
    for owner, group in df.groupby("owner"):
        rosters[owner] = set(group["name"])

    return rosters


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
    conn = sqlite3.connect(db_path)

    query = "SELECT * FROM players WHERE owner IS NULL"
    params = []

    if position:
        query += " AND position = ?"
        params.append(position)

    if min_sgp:
        query += " AND sgp >= ?"
        params.append(min_sgp)

    query += " ORDER BY sgp DESC"

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, conn, params=params if params else None)
    conn.close()

    return df


def get_standings(db_path: str = "data/optimizer.db") -> pd.DataFrame:
    """
    Get most recent standings.

    Returns DataFrame with team standings and category ranks.
    """
    conn = sqlite3.connect(db_path)

    df = pd.read_sql(
        """
        SELECT * FROM standings 
        WHERE as_of_date = (SELECT MAX(as_of_date) FROM standings)
        ORDER BY overall_rank
    """,
        conn,
    )

    conn.close()
    return df


# =============================================================================
# DATA REFRESH PIPELINE
# =============================================================================


def refresh_all_data(
    hitter_proj_path: str = "data/fangraphs-steamer-projections-hitters.csv",
    pitcher_proj_path: str = "data/fangraphs-steamer-projections-pitchers.csv",
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
            "projections": pd.DataFrame,           # From get_projections()
            "my_roster": set[str],                 # From get_roster_names(MY_TEAM_NAME)
            "opponent_rosters": dict[str, set[str]], # From get_all_roster_names() minus my team
            "standings": pd.DataFrame,             # From get_standings()
        }
    """
    print("=== Data Refresh Pipeline ===")

    # Step 1: Initialize database
    print("\nStep 1: Initializing database...")
    initialize_database(db_path)

    # Step 2: Load FanGraphs projections
    print("\nStep 2: Loading FanGraphs projections...")
    projections = load_projections(hitter_proj_path, pitcher_proj_path)

    # Step 3: Sync to database
    print("\nStep 3: Syncing projections to database...")
    sync_fangraphs_to_db(projections, db_path)

    # Step 4: Fetch ages from MLB Stats API
    if not skip_mlb_api:
        print("\nStep 4: Fetching ages from MLB Stats API...")
        # Get all valid MLBAM IDs from projections
        mlbam_ids = projections["MLBAMID"].dropna().astype(int).tolist()
        if len(mlbam_ids) > 0:
            ages_df = fetch_player_ages(mlbam_ids)
            sync_ages_to_db(ages_df, db_path)
        else:
            print("  Skipped - no MLBAM IDs in projections")
    else:
        print("\nStep 4: Skipping MLB Stats API call (using existing DB data)")

    if not skip_fantrax:
        # Step 5: Fetch Fantrax data
        print("\nStep 5: Fetching Fantrax data...")
        session = create_session()
        assert test_auth(session), "Fantrax authentication failed - update cookies"

        fantrax_data = fetch_all_fantrax_data(session)

        # Step 6: Sync rosters
        print("\nStep 6: Syncing rosters to database...")
        sync_fantrax_rosters_to_db(
            fantrax_data["roster_sets"],
            fantrax_data["rosters"],
            db_path,
        )

        # Step 7: Sync positions from player pool
        print("\nStep 7: Syncing positions from player pool...")
        if len(fantrax_data["player_pool"]) > 0:
            sync_player_pool_to_db(fantrax_data["player_pool"], db_path)
        else:
            print("  Skipped - player pool is empty")

        # Step 8: Sync standings
        print("\nStep 8: Syncing standings...")
        sync_standings_to_db(fantrax_data["standings"], db_path)
    else:
        print("\nSkipping Fantrax API calls (using existing DB data)")

    # Step 9: Compute dynasty values (requires age data)
    print("\nStep 9: Computing dynasty values...")
    compute_dynasty_sgp_in_db(db_path)

    # Step 10: Return data from database
    print("\n=== Refresh Complete ===")

    # Query fresh data from database
    projections = get_projections(db_path)
    all_rosters = get_all_roster_names(db_path)
    my_roster = all_rosters.pop(MY_TEAM_NAME, set())
    standings = get_standings(db_path)

    print(f"\nData Summary:")
    print(f"  Projections: {len(projections)} players")
    print(f"  My roster: {len(my_roster)} players")
    print(f"  Opponent rosters: {len(all_rosters)} teams")

    return {
        "projections": projections,
        "my_roster": my_roster,
        "opponent_rosters": all_rosters,
        "standings": standings,
    }
