"""
Test script for Fantrax API - fetches all rosters and validates authentication.

Usage:
    1. Update COOKIES below with fresh values from browser if needed
    2. Run: uv run python test_fantrax_api.py

Cookies are saved to data/fantrax_cookies.json for reuse.
"""

import json
from pathlib import Path

import requests

LEAGUE_ID = "f7cc72ecmkfnc7kl"
COOKIE_FILE = Path("data/fantrax_cookies.json")

# All teams in the league
TEAM_IDS = {
    "Aidangonnawin": "oluroo3mmkfnc7kw",
    "Future AL MVP, Evan Carter": "a80hb31gmkfnc7kw",
    "Oliver Wendell Homers": "ea7d523tmkfnc7kw",
    "paranoia_in_z_major": "2n4v2dn8mkfnc7kw",
    "Reasonable Doubtfielders": "03xns3dwmkfnc7kw",
    "Shohei Me The (Betting) Money!": "9z72hkg2mkfnc7kw",
    "The Big Dumpers": "zhh2uwcamkfnc7kw",
}

MY_TEAM_NAME = "The Big Dumpers"

# =============================================================================
# COOKIES - Update these if authentication fails
# Get from browser: DevTools > Application > Cookies > fantrax.com
# =============================================================================

COOKIES = {
    "JSESSIONID": "node0rb32jtbscpp0mcgmjh03qggo152156.node0",
    "FX_RM": "_qpxzU1kSFh5dARIORBFCA1lFAgIKAwUdEhsVBxwcBlRCAQU=",
}

# =============================================================================


def create_session(cookies: dict[str, str]) -> requests.Session:
    """Create authenticated session."""
    session = requests.Session()
    for name, value in cookies.items():
        session.cookies.set(name, value, domain=".fantrax.com")
    return session


def test_auth(session: requests.Session) -> bool:
    """Test if session is authenticated."""
    response = session.post(
        "https://www.fantrax.com/fxpa/req",
        params={"leagueId": LEAGUE_ID},
        json={
            "msgs": [
                {"method": "getFantasyLeagueInfo", "data": {"leagueId": LEAGUE_ID}}
            ]
        },
    )
    resp = response.json()
    if "pageError" in resp and resp["pageError"].get("code") == "WARNING_NOT_LOGGED_IN":
        return False
    return True


def fetch_team_roster(session: requests.Session, team_id: str) -> list[dict]:
    """Fetch a single team's roster."""
    response = session.post(
        "https://www.fantrax.com/fxpa/req",
        params={"leagueId": LEAGUE_ID},
        json={
            "msgs": [
                {
                    "method": "getTeamRosterInfo",
                    "data": {"leagueId": LEAGUE_ID, "teamId": team_id, "view": "STATS"},
                }
            ]
        },
    )

    resp = response.json()
    data = resp["responses"][0].get("data", {})

    STATUS_MAP = {"1": "active", "2": "reserve", "3": "IR", "9": "minors"}

    players = []
    for table in data.get("tables", []):
        for row in table.get("rows", []):
            scorer = row.get("scorer", {})
            pos = scorer.get("posShortNames", "")
            players.append(
                {
                    "name": scorer.get("name"),
                    "position": pos,
                    "team": scorer.get("teamShortName"),
                    "status": STATUS_MAP.get(row.get("statusId"), "unknown"),
                    "player_type": "pitcher" if pos in ("SP", "RP") else "hitter",
                }
            )
    return players


def fetch_all_rosters(session: requests.Session) -> dict[str, list[dict]]:
    """Fetch all team rosters."""
    all_rosters = {}
    for team_name, team_id in TEAM_IDS.items():
        players = fetch_team_roster(session, team_id)
        all_rosters[team_name] = players
        h = len([p for p in players if p["player_type"] == "hitter"])
        p = len([p for p in players if p["player_type"] == "pitcher"])
        print(f"  {team_name}: {len(players)} players ({h} H, {p} P)")
    return all_rosters


def main():
    print("Testing Fantrax API...")
    print(f"League ID: {LEAGUE_ID}")
    print()

    session = create_session(COOKIES)

    if not test_auth(session):
        print("❌ AUTHENTICATION FAILED - Cookies expired or invalid")
        print()
        print("To fix:")
        print("  1. Log into https://www.fantrax.com in your browser")
        print("  2. Open DevTools (Cmd+Option+I)")
        print("  3. Go to Application > Cookies > fantrax.com")
        print("  4. Copy JSESSIONID and FX_RM values")
        print("  5. Update COOKIES dict in this file")
        return

    print("✅ Authentication successful!")

    # Save cookies for future use
    COOKIE_FILE.parent.mkdir(exist_ok=True)
    with open(COOKIE_FILE, "w") as f:
        json.dump(COOKIES, f, indent=2)
    print(f"   Cookies saved to {COOKIE_FILE}")
    print()

    # Fetch all rosters
    print("=" * 60)
    print("FETCHING ALL TEAM ROSTERS")
    print("=" * 60)

    all_rosters = fetch_all_rosters(session)

    print()
    print("=" * 60)
    print(f"MY TEAM: {MY_TEAM_NAME}")
    print("=" * 60)

    my_roster = all_rosters.get(MY_TEAM_NAME, [])
    active = [p for p in my_roster if p["status"] in ("active", "reserve")]

    print(f"Active roster ({len(active)} players):")
    for p in active[:15]:
        suffix = "-P" if p["player_type"] == "pitcher" else "-H"
        print(f"  {p['name']}{suffix}  {p['position']:8}  {p['team']}")
    if len(active) > 15:
        print(f"  ... and {len(active) - 15} more")

    print()
    print("=" * 60)
    print("SUCCESS - Fantrax API is working!")
    print("=" * 60)


if __name__ == "__main__":
    main()
