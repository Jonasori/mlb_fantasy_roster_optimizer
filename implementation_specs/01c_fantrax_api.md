# Fantrax API Integration

## Overview

This document specifies the Fantrax API integration for fetching live roster, standings, and player age data. The API is the **single source of truth** for roster ownership and provides player positions.

**Module:** `optimizer/fantrax_api.py`

---

## Cross-References

**Depends on:**
- [00_agent_guidelines.md](00_agent_guidelines.md) — code style, no try/except
- [01a_config.md](01a_config.md) — `FANTRAX_LEAGUE_ID`, `FANTRAX_TEAM_IDS`

**Used by:**
- [01d_database.md](01d_database.md) — `fetch_all_fantrax_data()` synced to database
- [01e_dynasty_valuation.md](01e_dynasty_valuation.md) — age data from `fetch_player_pool()`

---

## ⚠️ Critical: getPlayerStats Pagination is Broken

**DO NOT attempt to paginate `getPlayerStats`.** The API ignores all pagination parameters. Use `maxResultsPerPage=5000` in a single request (see `fetch_player_pool()` implementation below).

---

## What Fantrax Provides

| Data | Available | Endpoint | Notes |
|------|-----------|----------|-------|
| **Rosters** | ✅ | `getTeamRosterInfo` | All 7 teams with full player data |
| **Position** | ✅ | Both | C, 1B, 2B, SS, 3B, OF, SP, RP (from `posShortNames`) |
| **Age** | ✅ | Both | In `cells` array (roster[0], pool[2]) |
| **Standings** | ✅ | `getStandings` | Live category ranks/values in `tableList` |
| **ADP** | ✅ | Both | Average Draft Position in `cells` |
| **Fantrax Score** | ✅ | `getPlayerStats` | League-specific value formula |
| **% Rostered** | ✅ | `getPlayerStats` | Current + trend |
| **Rookie Status** | ✅ | Both | `scorer.rookie` boolean |
| **Minors Eligible** | ✅ | Both | `scorer.minorsEligible` boolean |
| **Waiver Order** | ✅ | `getStandings` | In Point Totals table cells[2] |
| **Projected Stats** | ✅ | `getTeamRosterInfo` | AB, H, R, HR, RBI, SB, OPS for rostered players |
| **Full Projections** | ⚠️ | N/A | Missing PA, WAR, pitching → use FanGraphs |

**Key insight:** Fantrax API provides positions directly (`posShortNames` field). This eliminates the need for an external position database for roster data.

**API Robustness:** The Fantrax API is undocumented and response structures vary unpredictably.

**Critical lessons learned:**

1. **Type checking is mandatory** - Never assume a value is a dict. Check with `isinstance(x, dict)` before calling `.get()`. The API returns strings, lists, or dicts depending on context.

2. **Key name variations** - The same data appears under different keys:
   - Tables: `tables` OR `tableList`
   - Team ID: `id` OR `teamId`
   - Team name: `name` OR `teamName`
   - Rank: `rank` OR `position` OR `overallRank`
   - Points: `fpts` OR `totalPoints` OR `pts`

3. **Nested vs flat structures** - Team data may be:
   - Nested: `row["team"]["name"]`
   - Flat: `row["teamName"]`
   - Mixed: varies by endpoint

4. **Empty/null handling** - Players can have `None` names (empty roster slots), fields can be missing entirely.

5. **Error responses** - Check for `pageError` key before accessing `data`:
   ```python
   if "pageError" in resp0 and "data" not in resp0:
       # Handle error, return empty result
   ```

6. **`getPlayerStats` pagination is BROKEN** - Use `maxResultsPerPage=5000` in a single request (see `fetch_player_pool()`).

7. **Two-way players already have -H/-P suffix** - Fantrax returns some player names with the suffix already attached. **Do NOT add another suffix** or you'll create names like "Shohei Ohtani-H-H" that won't match the database. Always check `if name.endswith("-H") or name.endswith("-P")` before adding a suffix.

**Defensive parsing pattern:**
```python
def _parse_team_row(row, index: int) -> dict | None:
    if not isinstance(row, dict):
        return None
    team_data = row.get("team", row)
    if not isinstance(team_data, dict):
        team_data = row
    # ... extract fields with fallbacks
```

**Critical: Data is in `cells` array, not direct fields!**

The API returns row data in a `cells` array. Use `tableHeader.cells` to understand column meanings.

### Cell Mapping Reference

**`getPlayerStats` (Player Pool):** cells[0]=Rank, cells[1]=Status, cells[2]=Age, cells[3]=Score, cells[4-7]=Drafted/ADP/Rostered/Trend

**`getTeamRosterInfo` (Roster):** cells[0]=Age, cells[1-2]=Drafted/ADP, cells[3-10]=AB/H/R/HR/RBI/SB/OPS/GP (projected stats)

**`getStandings`:** Table 2 (Point Totals) has cells[0]=Points, cells[2]=Waiver Order, cells[3-16]=Category values. Category tables (6-15) have cells[0]=Rank, cells[3]=Team info, cells[4]=Stat value.

**Scorer object fields:** `rookie`, `minorsEligible`, `scorerId`, `posIds`, `icons`

**Row fields:** `statusId` (roster status), `eligiblePosIds`, `posId` (current slot)

Always check `isinstance(x, dict)` before calling `.get()` on cell values.

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
4. Add to `config.json` under `fantrax.cookies`:

```json
{
    "fantrax": {
        "cookies": {
            "JSESSIONID": "node0rb32jtbscpp0mcgmjh03qggo152156.node0",
            "FX_RM": "_qpxzU1kSFh5dARIORBFCA1lFAgIKAwUdEhsVBxwcBlRCAQU="
        }
    }
}
```

---

## Configuration

```python
import json
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

# Import league constants from config (single source of truth)
from .config import FANTRAX_COOKIES
from .data_loader import (
    FANTRAX_LEAGUE_ID,
    FANTRAX_TEAM_IDS,
)

FANTRAX_API_URL = "https://www.fantrax.com/fxpa/req"

# Exceptional name corrections for cases that normalization can't handle
# Most matching uses normalize_name() from data_loader - this is for true edge cases only
FANTRAX_NAME_CORRECTIONS = {
    "Logan OHoppe": "Logan O'Hoppe",  # Missing apostrophe
    "Leodalis De Vries": "Leo De Vries",  # Completely different first name
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

## Observed Response Structures

**Note:** These are based on actual API observations. The API is undocumented and may change.

### Top-level response
```python
{
    "data": {...},           # Metadata (not the main data!)
    "roles": ["02"],
    "pageError": {...},      # Present on errors
    "responses": [           # Main data is here
        {
            "data": {...},   # The actual response data
            "pageError": {...}  # Or error info
        }
    ]
}
```

### Error response (e.g., expired cookies)
```python
{
    "responses": [
        {
            "pageError": {
                "code": "NOT_MEMBER_OF_LEAGUE",
                "title": "Not Member of League / League not Found",
                ...
            }
            # Note: NO "data" key when there's an error
        }
    ]
}
```

### getStandings response (observed keys)
```python
# responses[0]["data"] contains:
{
    "goBackDays": ...,
    "fantasyTeamInfo": [...],    # Can be list of strings OR dicts!
    "displayedSelections": ...,
    "miscData": ...,
    "tableList": [...],          # Note: "tableList" not "tables"
    "displayedLists": ...
}
```

### getTeamRosterInfo response
```python
# responses[0]["data"] contains:
{
    "tables": [  # Or "tableList"
        {
            "rows": [
                {
                    "scorer": {
                        "name": "Mike Trout",
                        "posShortNames": "OF",
                        "teamShortName": "LAA",
                        "age": 32
                    },
                    "statusId": "1"  # "1" = active, "2" = reserve, "3" = minors, "4" = IR
                }
            ]
        }
    ]
}
```

---

## Function Specifications

### Authentication

```python
def load_cookies() -> dict[str, str]:
    """
    Load cookies from config.
    
    Returns: Dict with JSESSIONID and FX_RM.
    
    Assertion (with actionable message):
        assert "JSESSIONID" in FANTRAX_COOKIES, "Config must have 'fantrax.cookies.JSESSIONID'"
        assert "FX_RM" in FANTRAX_COOKIES, "Config must have 'fantrax.cookies.FX_RM'"
        return FANTRAX_COOKIES
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
        name, position, team (MLB team), status, player_type, age
    
    Implementation:
        1. Check for pageError before accessing data
        2. Try both "tables" and "tableList" keys
        3. Player info is under row["scorer"]
        
        full_response = response.json()
        resp0 = full_response.get("responses", [{}])[0]
        
        # Check for API errors
        if "pageError" in resp0 and "data" not in resp0:
            print(f"WARNING: Roster API error: {resp0['pageError'].get('code')}")
            return []
        
        data = resp0.get("data", {})
        tables = data.get("tables", []) or data.get("tableList", [])
        
        players = []
        for table in tables:
            for row in table.get("rows", []):
                scorer = row.get("scorer", {})
                pos = scorer.get("posShortNames", "")
                players.append({
                    "name": scorer.get("name"),
                    "position": pos,
                    "team": scorer.get("teamShortName"),
                    "status": {"1": "active", "2": "reserve", "3": "minors", "4": "IR"}.get(
                        str(row.get("statusId", "")), "unknown"
                    ),
                    "player_type": get_player_type(pos),
                    "age": scorer.get("age"),
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
    
    Implementation:
        Uses helper functions for clean separation:
        - _empty_standings_df() - returns consistent empty DataFrame
        - _parse_team_row(row, index) - extracts team data with type checking
        - _parse_standings_data(data) - tries structures in priority order
        
        Structure priority (first match wins):
        1. tableList[] with tableId="standings" (most complete, has category data)
        2. "fantasyTeamInfo" key (common fallback, basic team info only)
        3. Any table with team/teamId in rows (last resort)
        
        CRITICAL: Always check isinstance(x, dict) before .get() calls.
        The API returns strings in some contexts where dicts are expected.
    
    Print:
        "Standings ({N} teams):"
        "  1. {team} - {points} pts"
    """


def fetch_player_pool(
    session: requests.Session,
    max_results: int | None = None,
) -> pd.DataFrame:
    """
    Fetch players from Fantrax database.
    
    API: getPlayerStats (single request, max 5000 players)
    
    ⚠️ CRITICAL: Pagination parameters are ignored. Use `maxResultsPerPage=5000` in a single request.
    
    Args:
        max_results: Limit total players (None = 5000, the API max)
    
    Returns DataFrame with columns:
        name, position, mlb_team, age, adp, pct_rostered,
        fantrax_rank, fantrax_score, is_free_agent
    
    Implementation:
        Single request with maxResultsPerPage=5000 (max allowed).
        The API reports 9,719 total but only 5,000 are fetchable.
        This is sufficient since rostered players come from fetch_all_rosters()
        and bottom ~4,700 players have Fantrax score = 0.
    
    Print:
        "Fetching player pool from Fantrax..."
        "  API reports {N} total players (fetching top {M})"
        "  Fetched {N} players"
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
            1. Skip if status not in ("active", "reserve") — excludes minors/IR
            2. Skip if name is None — handles empty roster slots
            3. Apply name corrections if enabled (BEFORE suffix)
            4. Check if name already ends with -H or -P (two-way players)
            5. If NOT already suffixed, add suffix from player_type
            6. Add to name set
            
        ⚠️ CRITICAL: Fantrax returns two-way players with suffix already attached!
        Example: "Shohei Ohtani-H" (hitter slot), "Shohei Ohtani-P" (pitcher slot)
        If you add another suffix, you get "Shohei Ohtani-H-H" which won't match the database.
    
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

## Name Matching Strategy

**Principle**: Use `normalize_name()` for matching, not a correction dictionary.

The `normalize_name()` function (in `data_loader.py`) handles:
- Accented characters: `Rodríguez` → `rodriguez`
- Suffixes: `Jr.`, `Sr.`, `II`, `III` → removed
- Case: lowercased
- Preserves -H/-P suffix

**Matching flow in `sync_fantrax_rosters_to_db`**:

```
1. Build lookup from DB: normalized_name → actual_name
   "eugenio suarez-h" → "Eugenio Suárez-H"

2. For each Fantrax player:
   Fantrax: "Eugenio Suarez-H"
       ↓ normalize_name()
   Normalized: "eugenio suarez-h"
       ↓ Lookup in normalized_to_actual
   DB name: "Eugenio Suárez-H"
```

The `FANTRAX_NAME_CORRECTIONS` dict is only for truly exceptional cases
where names are fundamentally different (not just accent variations).

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
