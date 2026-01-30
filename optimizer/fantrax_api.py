"""
Fantrax API integration for roster, standings, and player age data.

The API is the single source of truth for roster ownership and provides player positions.
"""

import json
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

from .data_loader import (
    FANTRAX_LEAGUE_ID,
    FANTRAX_STATUS_MAP,
    FANTRAX_TEAM_IDS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

COOKIE_FILE = Path("data/fantrax_cookies.json")
FANTRAX_API_URL = "https://www.fantrax.com/fxpa/req"

# Exceptional name corrections for cases that normalization can't handle
# (different names entirely, not just accent/suffix variations)
# Most matching is done via normalize_name() - this is for true edge cases only
FANTRAX_NAME_CORRECTIONS = {
    "Logan OHoppe": "Logan O'Hoppe",  # Missing apostrophe
    "Leodalis De Vries": "Leo De Vries",  # Completely different first name
}


# =============================================================================
# PLAYER TYPE DETERMINATION
# =============================================================================


def get_player_type(position: str) -> str:
    """
    Determine if player is hitter or pitcher based on Fantrax position.

    Args:
        position: Position string from Fantrax (e.g., "SS", "SP", "SP,RP")

    Returns:
        "pitcher" if position contains SP or RP, else "hitter"
    """
    return (
        "pitcher"
        if position in ("SP", "RP") or "SP" in position or "RP" in position
        else "hitter"
    )


# =============================================================================
# AUTHENTICATION
# =============================================================================


def load_cookies() -> dict[str, str]:
    """
    Load cookies from file.

    Returns: Dict with JSESSIONID and FX_RM.
    """
    assert COOKIE_FILE.exists(), (
        f"Cookie file not found: {COOKIE_FILE}\n"
        f"To fix:\n"
        f"  1. Log into https://www.fantrax.com\n"
        f"  2. Open DevTools → Application → Cookies\n"
        f"  3. Copy JSESSIONID and FX_RM\n"
        f"  4. Save to {COOKIE_FILE}"
    )

    with open(COOKIE_FILE) as f:
        return json.load(f)


def create_session() -> requests.Session:
    """
    Create authenticated Fantrax session.

    Returns: Configured requests.Session with cookies set.
    """
    cookies = load_cookies()
    session = requests.Session()

    # Set cookies on session with domain
    session.cookies.set("JSESSIONID", cookies["JSESSIONID"], domain=".fantrax.com")
    session.cookies.set("FX_RM", cookies["FX_RM"], domain=".fantrax.com")

    return session


def test_auth(session: requests.Session) -> bool:
    """
    Test if session is authenticated.

    Makes a simple API call and checks for auth error.

    Returns: True if authenticated, False otherwise.
    """
    response = session.post(
        FANTRAX_API_URL,
        params={"leagueId": FANTRAX_LEAGUE_ID},
        json={
            "msgs": [
                {
                    "method": "getFantasyLeagueInfo",
                    "data": {"leagueId": FANTRAX_LEAGUE_ID},
                }
            ]
        },
    )
    resp = response.json()

    if "pageError" in resp and resp["pageError"].get("code") == "WARNING_NOT_LOGGED_IN":
        return False
    return True


# =============================================================================
# DATA FETCHING
# =============================================================================


def fetch_team_roster(session: requests.Session, team_id: str) -> list[dict]:
    """
    Fetch roster for a single team.

    API: getTeamRosterInfo with view="STATS"

    Returns list of player dicts with:
        name, position, team (MLB team), status, player_type
    """
    response = session.post(
        FANTRAX_API_URL,
        params={"leagueId": FANTRAX_LEAGUE_ID},
        json={
            "msgs": [
                {
                    "method": "getTeamRosterInfo",
                    "data": {
                        "leagueId": FANTRAX_LEAGUE_ID,
                        "teamId": team_id,
                        "view": "STATS",
                    },
                }
            ]
        },
    )

    full_response = response.json()
    resp0 = full_response.get("responses", [{}])[0]

    # Check for API errors
    if "pageError" in resp0 and "data" not in resp0:
        error_code = resp0["pageError"].get("code", "unknown")
        print(f"WARNING: Roster API error for team {team_id}: {error_code}")
        return []

    data = resp0.get("data", {})

    players = []
    tables = data.get("tables", []) or data.get("tableList", [])
    for table in tables:
        for row in table.get("rows", []):
            scorer = row.get("scorer", {})
            pos = scorer.get("posShortNames", "")
            status_id = str(row.get("statusId", ""))

            # Data is in cells array:
            # [0]: Age, [1]: %D, [2]: ADP, [3]: AB, [4]: H, [5]: R,
            # [6]: HR, [7]: RBI, [8]: SB, [9]: OPS, [10]: GP
            cells = row.get("cells", [])

            def get_cell(idx: int, as_float: bool = False) -> int | float | None:
                if len(cells) <= idx:
                    return None
                cell = cells[idx]
                if not isinstance(cell, dict):
                    return None
                content = cell.get("content", "")
                if not content or content == "-":
                    return None
                content = content.replace("%", "").strip()
                if not content:
                    return None
                if as_float:
                    # Remove decimal point for digit check, but allow negative sign
                    cleaned = content.lstrip("-").replace(".", "")
                    if cleaned.isdigit():
                        return float(content)
                    return None
                return int(content) if content.isdigit() else None

            players.append(
                {
                    "name": scorer.get("name"),
                    "position": pos,
                    "team": scorer.get("teamShortName"),
                    "status": FANTRAX_STATUS_MAP.get(status_id, "unknown"),
                    "player_type": get_player_type(pos),
                    "age": get_cell(0),
                    "adp": get_cell(2, as_float=True),
                    "rookie": scorer.get("rookie", False),
                    "minors_eligible": scorer.get("minorsEligible", False),
                    "fantrax_id": scorer.get("scorerId"),
                    "eligible_positions": scorer.get("posIds", []),
                }
            )

    return players


def fetch_all_rosters(session: requests.Session) -> dict[str, list[dict]]:
    """
    Fetch rosters for ALL teams.

    Returns:
        Dict mapping team_name to list of player dicts.
        {"The Big Dumpers": [{"name": "Cal Raleigh", "position": "C", ...}, ...], ...}
    """
    print("Fetching rosters for 7 teams...")
    all_rosters = {}

    for team_name, team_id in tqdm(FANTRAX_TEAM_IDS.items(), desc="Fetching rosters"):
        players = fetch_team_roster(session, team_id)
        all_rosters[team_name] = players

        hitters = sum(1 for p in players if p["player_type"] == "hitter")
        pitchers = sum(1 for p in players if p["player_type"] == "pitcher")
        print(f"  {team_name}: {len(players)} players ({hitters} H, {pitchers} P)")

    return all_rosters


# =============================================================================
# STANDINGS PARSING HELPERS
# =============================================================================


def _empty_standings_df() -> pd.DataFrame:
    """Return empty DataFrame with expected standings columns."""
    return pd.DataFrame(
        columns=[
            "team_id",
            "team_name",
            "overall_rank",
            "total_points",
            "r",
            "hr",
            "rbi",
            "sb",
            "ops",
            "w",
            "sv",
            "k",
            "era",
            "whip",
        ]
    )


def _parse_standings_data(data: dict) -> list[dict]:
    """
    Parse standings from API response data.

    Fantrax API returns data in various structures. The most reliable is:
    - tableList contains category tables (Runs Scored, Home Runs, etc.)
    - Each category table has rows with cells[3] containing team name + teamId
    - cells[0] = rank, cells[1] = points for that category
    """
    tables = data.get("tables") or data.get("tableList") or []

    # Build team info from category tables (tables with Rotisserie3 type have team data)
    # These tables have cells[3] with teamId and team name
    teams_by_id = {}

    for table in tables:
        if not isinstance(table, dict):
            continue

        table_type = table.get("tableType", "")
        caption = table.get("caption", "")

        # Skip section headings and look for category tables
        if table_type == "SECTION_HEADING":
            continue

        for row in table.get("rows", []):
            if not isinstance(row, dict):
                continue

            cells = row.get("cells", [])

            # Look for cell with teamId (usually cells[3] in category tables)
            team_cell = None
            team_cell_idx = -1
            for idx, cell in enumerate(cells):
                if isinstance(cell, dict) and "teamId" in cell:
                    team_cell = cell
                    team_cell_idx = idx
                    break

            if team_cell is None:
                continue

            team_id = team_cell.get("teamId")
            team_name = team_cell.get("content", "Unknown")

            if team_id not in teams_by_id:
                teams_by_id[team_id] = {
                    "team_id": team_id,
                    "team_name": team_name,
                    "overall_rank": 1,  # Default, will update if found
                    "total_points": 0,
                }

    # Convert to list and sort by team name
    rows = list(teams_by_id.values())
    rows.sort(key=lambda x: x.get("team_name", ""))

    # Assign ranks based on position (all tied at 1 pre-season)
    for i, row in enumerate(rows):
        row["overall_rank"] = i + 1

    return rows


def fetch_standings(session: requests.Session) -> pd.DataFrame:
    """
    Fetch current league standings.

    API: getStandings

    Returns DataFrame with columns:
        team_id, team_name, overall_rank, total_points,
        r, hr, rbi, sb, ops, w, sv, k, era, whip,  # Category values
        r_rank, hr_rank, ...                        # Category ranks (1-7)
    """
    response = session.post(
        FANTRAX_API_URL,
        params={"leagueId": FANTRAX_LEAGUE_ID},
        json={
            "msgs": [
                {"method": "getStandings", "data": {"leagueId": FANTRAX_LEAGUE_ID}}
            ]
        },
    )

    full_response = response.json()
    resp0 = full_response.get("responses", [{}])[0]

    # Check for API errors
    if "pageError" in resp0 and "data" not in resp0:
        error_code = resp0["pageError"].get("code", "unknown")
        print(f"WARNING: Standings API error: {error_code}")
        return _empty_standings_df()

    data = resp0.get("data", {})
    rows = _parse_standings_data(data)

    if not rows:
        print("WARNING: Could not parse standings from API response")
        print(f"  Available keys in response: {list(data.keys())}")
        return _empty_standings_df()

    df = pd.DataFrame(rows)

    # Ensure overall_rank exists and has values
    if "overall_rank" not in df.columns or df["overall_rank"].isna().all():
        df["overall_rank"] = list(range(1, len(df) + 1))

    # Fill any remaining NaN ranks with sequential values
    if df["overall_rank"].isna().any():
        df["overall_rank"] = df["overall_rank"].fillna(pd.Series(range(1, len(df) + 1)))

    print(f"Standings ({len(df)} teams):")
    for _, row in df.sort_values("overall_rank").iterrows():
        rank = int(row["overall_rank"]) if pd.notna(row["overall_rank"]) else "?"
        pts = row.get("total_points", "N/A")
        print(f"  {rank}. {row['team_name']} - {pts} pts")

    return df


def fetch_player_pool(
    session: requests.Session,
    max_results: int | None = None,
) -> pd.DataFrame:
    """
    Fetch players from Fantrax database.

    API: getPlayerStats (single request, max 5000 players)

    Note: The Fantrax API does not support pagination for this endpoint.
    The offset parameter is ignored. We use maxResultsPerPage=5000 to get
    the top 5000 players by Fantrax score in a single request.

    This is sufficient because:
    - Rostered players get their data from fetch_all_rosters()
    - The bottom ~4,700 players have Fantrax score of 0 (irrelevant)
    - All relevant MLB players are in the top 5000

    Args:
        max_results: Limit total players (None = 5000, the API max)

    Returns DataFrame with columns:
        name, position, mlb_team, age, adp, pct_rostered,
        fantrax_rank, fantrax_score, is_free_agent
    """
    print("Fetching player pool from Fantrax...")

    # API max is 5000 players per request; pagination doesn't work
    request_size = min(max_results, 5000) if max_results else 5000

    response = session.post(
        FANTRAX_API_URL,
        params={"leagueId": FANTRAX_LEAGUE_ID},
        json={
            "msgs": [
                {
                    "method": "getPlayerStats",
                    "data": {
                        "leagueId": FANTRAX_LEAGUE_ID,
                        "maxResultsPerPage": request_size,
                    },
                }
            ]
        },
    )

    full_response = response.json()
    resp0 = full_response.get("responses", [{}])[0]

    # Check for API errors
    if "pageError" in resp0 and "data" not in resp0:
        error_code = resp0["pageError"].get("code", "unknown")
        print(f"WARNING: Player pool API error: {error_code}")
        return pd.DataFrame()

    data = resp0.get("data", {})

    # Get pagination info for reporting
    pagination = data.get("paginatedResultSet", {})
    total_available = pagination.get("totalNumResults", 0)
    print(
        f"  API reports {total_available:,} total players (fetching top {request_size:,})"
    )

    # Player data is in statsTable
    rows = data.get("statsTable", [])

    # Fallback to tables/tableList
    if not rows:
        tables = data.get("tables", []) or data.get("tableList", [])
        rows = tables[0].get("rows", []) if tables else []

    all_players = []
    for row in rows:
        scorer = row.get("scorer", {})
        pos = scorer.get("posShortNames", "")

        # Data is in cells array:
        # cells[0]: Rank, cells[1]: Status, cells[2]: Age,
        # cells[3]: Score, cells[4]: %D, cells[5]: ADP,
        # cells[6]: % Rostered, cells[7]: Trend
        cells = row.get("cells", [])

        def get_cell(idx: int, as_float: bool = False) -> int | float | None:
            if len(cells) <= idx:
                return None
            cell = cells[idx]
            if not isinstance(cell, dict):
                return None
            content = cell.get("content", "")
            if not content or content == "-":
                return None
            # Remove % sign if present
            content = content.replace("%", "").strip()
            if not content:
                return None
            if as_float:
                # Remove decimal point for digit check, but allow negative sign
                cleaned = content.lstrip("-").replace(".", "")
                if cleaned.isdigit():
                    return float(content)
                return None
            return int(content) if content.isdigit() else None

        # Extract status to determine if free agent
        status_cell = cells[1] if len(cells) > 1 else {}
        status = status_cell.get("content", "") if isinstance(status_cell, dict) else ""
        is_free_agent = status == "FA"

        all_players.append(
            {
                "name": scorer.get("name"),
                "position": pos,
                "mlb_team": scorer.get("teamShortName"),
                "age": get_cell(2),
                "fantrax_score": get_cell(3, as_float=True),
                "pct_drafted": get_cell(4, as_float=True),
                "adp": get_cell(5, as_float=True),
                "pct_rostered": get_cell(6, as_float=True),
                "roster_trend": get_cell(7, as_float=True),
                "fantrax_rank": get_cell(0),
                "is_free_agent": is_free_agent,
                "player_type": get_player_type(pos),
                "rookie": scorer.get("rookie", False),
                "minors_eligible": scorer.get("minorsEligible", False),
                "fantrax_id": scorer.get("scorerId"),
            }
        )

    df = pd.DataFrame(all_players)
    print(f"  Fetched {len(df):,} players")

    return df


# =============================================================================
# ROSTER EXTRACTION WITH SUFFIXES
# =============================================================================


def extract_roster_sets(
    all_rosters: dict[str, list[dict]],
    apply_corrections: bool = True,
) -> dict[str, set[str]]:
    """
    Convert roster dicts to name sets with -H/-P suffixes.

    Args:
        all_rosters: Dict from fetch_all_rosters()
        apply_corrections: Whether to apply name corrections

    Returns:
        Dict mapping team_name to set of player names WITH -H/-P suffix.
        Example: {"The Big Dumpers": {"Cal Raleigh-H", "Gerrit Cole-P", ...}}
    """
    roster_sets = {}

    for team_name, players in all_rosters.items():
        names = set()
        active_count = 0

        for player in players:
            # Skip non-active players (minors, IR)
            if player["status"] not in ("active", "reserve"):
                continue

            name = player["name"]

            # Skip players with no name (empty roster slots)
            if name is None:
                continue

            # Apply name corrections if enabled (BEFORE suffix)
            if apply_corrections and name in FANTRAX_NAME_CORRECTIONS:
                name = FANTRAX_NAME_CORRECTIONS[name]

            # Add suffix only if not already present (Fantrax sometimes includes it)
            if name.endswith("-H") or name.endswith("-P"):
                suffixed_name = name
            else:
                suffix = "-P" if player["player_type"] == "pitcher" else "-H"
                suffixed_name = name + suffix

            names.add(suffixed_name)
            active_count += 1

        roster_sets[team_name] = names

    print("Extracted roster sets for 7 teams")
    for team, names in roster_sets.items():
        print(f"  {team}: {len(names)} active players")

    return roster_sets


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def fetch_all_fantrax_data(session: requests.Session) -> dict:
    """
    Fetch ALL available data from Fantrax.

    Returns:
        {
            "rosters": dict[str, list[dict]],  # team_name -> player list
            "roster_sets": dict[str, set[str]],  # team_name -> name set with suffixes
            "standings": pd.DataFrame,
            "player_pool": pd.DataFrame,
        }

    This is the main entry point for Fantrax data.
    """
    # Test authentication
    assert test_auth(session), "Fantrax authentication failed - update cookies"

    # Fetch all data
    rosters = fetch_all_rosters(session)
    roster_sets = extract_roster_sets(rosters)
    standings = fetch_standings(session)
    player_pool = fetch_player_pool(session)

    # Validation
    assert len(rosters) == 7, f"Expected 7 teams, got {len(rosters)}"

    total_players = sum(len(players) for players in rosters.values())
    assert total_players >= 150, f"Expected ~182 rostered players, got {total_players}"

    for team, names in roster_sets.items():
        assert all(n.endswith(("-H", "-P")) for n in names), (
            f"{team} has names without suffix"
        )

    print(f"\n=== Fantrax Data Summary ===")
    print(f"Rosters: {len(rosters)} teams, {total_players} total players")
    print(f"Player pool: {len(player_pool)} players")
    print(f"Standings: {len(standings)} teams")

    return {
        "rosters": rosters,
        "roster_sets": roster_sets,
        "standings": standings,
        "player_pool": player_pool,
    }
