# Fantrax API Integration

## Overview

This document specifies the Fantrax API integration for fetching live roster, standings, and player age data. The API is the **single source of truth** for roster ownership and provides player positions.

**Module:** `optimizer/fantrax_api.py`

---

## What Fantrax Provides

| Data | Available | Endpoint | Notes |
|------|-----------|----------|-------|
| **Rosters** | ✅ | `getTeamRosterInfo` | All 7 teams with full player data |
| **Position** | ✅ | Both | C, 1B, 2B, SS, 3B, OF, SP, RP (from `posShortNames`) |
| **Age** | ✅ | Both endpoints | For all ~9,718 players |
| **Standings** | ✅ | `getStandings` | Live category ranks/values |
| **ADP** | ✅ | Both | Average Draft Position |
| **Projections** | ⚠️ | Rostered only | Missing PA, WAR → use FanGraphs |

**Key insight:** Fantrax API provides positions directly (`posShortNames` field). This eliminates the need for an external position database for roster data.

---

## Authentication

The league is private. Requires cookies from browser session.

**Required cookies:**
- `JSESSIONID` — session cookie (expires ~24 hours)
- `FX_RM` — "remember me" cookie (longer-lived)

**Setup:**
1. Log into https://www.fantrax.com
2. Open DevTools → Application → Cookies → fantrax.com
3. Copy `JSESSIONID` and `FX_RM` values
4. Save to `data/fantrax_cookies.json`:

```json
{
    "JSESSIONID": "node0rb32jtbscpp0mcgmjh03qggo152156.node0",
    "FX_RM": "_qpxzU1kSFh5dARIORBFCA1lFAgIKAwUdEhsVBxwcBlRCAQU="
}
```

---

## Configuration

```python
from pathlib import Path
import requests

# Import league constants from data_loader (single source of truth)
from .data_loader import (
    FANTRAX_LEAGUE_ID,
    MY_TEAM_NAME,
    MY_TEAM_ID,
    FANTRAX_TEAM_IDS,
    FANTRAX_STATUS_MAP,
)

COOKIE_FILE = Path("data/fantrax_cookies.json")
FANTRAX_API_URL = "https://www.fantrax.com/fxpa/req"

# Known name corrections: Fantrax spelling → FanGraphs spelling
# Applied BEFORE adding -H/-P suffix
FANTRAX_NAME_CORRECTIONS = {
    "Julio Rodriguez": "Julio Rodríguez",
    "Ronald Acuna Jr.": "Ronald Acuña Jr.",
    "Luis Garcia": "Luis García",
    "Logan OHoppe": "Logan O'Hoppe",
    "Leodalis De Vries": "Leo De Vries",
}
```

---

## Player Type Determination

The Fantrax API provides position via `posShortNames` field (e.g., "SS", "OF", "SP", "RP", "SP,RP").

**Rule for determining player_type:**
```python
def get_player_type(position: str) -> str:
    """
    Determine if player is hitter or pitcher based on Fantrax position.
    
    Args:
        position: Position string from Fantrax (e.g., "SS", "SP", "SP,RP")
    
    Returns:
        "pitcher" if position contains SP or RP, else "hitter"
    """
    return "pitcher" if position in ("SP", "RP") or "SP" in position or "RP" in position else "hitter"
```

This is used to determine the `-H` or `-P` suffix.

---

## API Details

**Base URL:** `https://www.fantrax.com/fxpa/req`  
**Method:** POST with JSON payload  
**Params:** `leagueId` query parameter

**Request format:**
```python
response = session.post(
    FANTRAX_API_URL,
    params={"leagueId": FANTRAX_LEAGUE_ID},
    json={
        "msgs": [
            {"method": method_name, "data": data_dict}
        ]
    }
)
```

---

## Function Specifications

### Authentication

```python
def load_cookies() -> dict[str, str]:
    """
    Load cookies from file.
    
    Returns: Dict with JSESSIONID and FX_RM.
    
    Assertion (with actionable message):
        assert COOKIE_FILE.exists(), (
            f"Cookie file not found: {COOKIE_FILE}\\n"
            f"To fix:\\n"
            f"  1. Log into https://www.fantrax.com\\n"
            f"  2. Open DevTools → Application → Cookies\\n"
            f"  3. Copy JSESSIONID and FX_RM\\n"
            f"  4. Save to {COOKIE_FILE}"
        )
    """


def create_session() -> requests.Session:
    """
    Create authenticated Fantrax session.
    
    Implementation:
        1. Load cookies from file
        2. Create requests.Session
        3. Set cookies on session with domain=".fantrax.com"
    
    Returns: Configured requests.Session with cookies set.
    """


def test_auth(session: requests.Session) -> bool:
    """
    Test if session is authenticated.
    
    Makes a simple API call and checks for auth error.
    
    Returns: True if authenticated, False otherwise.
    
    Implementation:
        response = session.post(
            FANTRAX_API_URL,
            params={"leagueId": FANTRAX_LEAGUE_ID},
            json={"msgs": [{"method": "getFantasyLeagueInfo", "data": {"leagueId": FANTRAX_LEAGUE_ID}}]}
        )
        resp = response.json()
        
        if "pageError" in resp and resp["pageError"].get("code") == "WARNING_NOT_LOGGED_IN":
            return False
        return True
    """
```

### Data Fetching

```python
def fetch_team_roster(session: requests.Session, team_id: str) -> list[dict]:
    """
    Fetch roster for a single team.
    
    API: getTeamRosterInfo with view="STATS"
    
    Returns list of player dicts with:
        name, position, team (MLB team), status, player_type
    
    Implementation:
        response = session.post(
            FANTRAX_API_URL,
            params={"leagueId": FANTRAX_LEAGUE_ID},
            json={"msgs": [{"method": "getTeamRosterInfo", 
                           "data": {"leagueId": FANTRAX_LEAGUE_ID, "teamId": team_id, "view": "STATS"}}]}
        )
        
        data = response.json()["responses"][0]["data"]
        
        players = []
        for table in data.get("tables", []):
            for row in table.get("rows", []):
                scorer = row.get("scorer", {})
                pos = scorer.get("posShortNames", "")
                players.append({
                    "name": scorer.get("name"),
                    "position": pos,
                    "team": scorer.get("teamShortName"),
                    "status": FANTRAX_STATUS_MAP.get(row.get("statusId"), "unknown"),
                    "player_type": "pitcher" if pos in ("SP", "RP") else "hitter",
                })
        return players
    """


def fetch_all_rosters(session: requests.Session) -> dict[str, list[dict]]:
    """
    Fetch rosters for ALL teams.
    
    Returns:
        Dict mapping team_name to list of player dicts.
        {"The Big Dumpers": [{"name": "Cal Raleigh", "position": "C", ...}, ...], ...}
    
    Print:
        "Fetching rosters for 7 teams..."
        "  {team_name}: {count} players ({hitters} H, {pitchers} P)"
    """


def fetch_standings(session: requests.Session) -> pd.DataFrame:
    """
    Fetch current league standings.
    
    API: getStandings
    
    Returns DataFrame with columns:
        team_id, team_name, overall_rank, total_points,
        r, hr, rbi, sb, ops, w, sv, k, era, whip,  # Category values
        r_rank, hr_rank, ...                        # Category ranks (1-7)
    
    Print:
        "Standings (7 teams):"
        "  1. {team} - {points} pts"
    """


def fetch_player_pool(
    session: requests.Session,
    max_results: int | None = None,
) -> pd.DataFrame:
    """
    Fetch ALL players in Fantrax database.
    
    API: getPlayerStats (paginated)
    
    Args:
        max_results: Limit total players (None = all ~9,718)
    
    Returns DataFrame with columns:
        name, position, mlb_team, age, adp, pct_rostered,
        fantrax_rank, fantrax_score, is_free_agent
    
    Use this for:
        - Player ages (for ALL players including free agents)
        - ADP data
        - Ownership status
    
    Print: "Fetched {N} players from Fantrax"
    """
```

### Roster Extraction with Suffixes

```python
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
    
    Implementation:
        For each player:
            1. Apply name corrections if enabled (BEFORE suffix)
            2. Determine suffix from player_type
            3. Combine: name + suffix
        
        Filter to active/reserve status only (not minors/IR).
    
    Print:
        "Extracted roster sets for 7 teams"
        "  {team}: {count} active players"
    """
```

### Convenience Functions

```python
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
    
    Print: Summary of all data fetched.
    """
```

---

## Name Correction Flow

Name corrections must happen BEFORE adding the `-H/-P` suffix:

```
Fantrax: "Julio Rodriguez"
    ↓ Apply FANTRAX_NAME_CORRECTIONS
Corrected: "Julio Rodríguez"  
    ↓ Add suffix based on position
Final: "Julio Rodríguez-H"
```

This ensures the suffixed name matches FanGraphs projections.

---

## Validation Checklist

```python
# After testing auth:
assert test_auth(session), "Fantrax authentication failed - update cookies"

# After fetching rosters:
all_rosters = fetch_all_rosters(session)
assert len(all_rosters) == 7, f"Expected 7 teams, got {len(all_rosters)}"

total_players = sum(len(players) for players in all_rosters.values())
assert total_players >= 150, f"Expected ~182 rostered players, got {total_players}"

# After extracting roster sets:
roster_sets = extract_roster_sets(all_rosters)
for team, names in roster_sets.items():
    assert all(n.endswith(('-H', '-P')) for n in names), f"{team} has names without suffix"
```
