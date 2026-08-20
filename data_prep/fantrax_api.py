"""
Fantrax fetchers that write raw-layer snapshots.

Two sources, because they are two different grains and there is no honest way to
put them in one table:

    data/raw/fantrax/YYYY-MM-DD.parquet     player grain: ownership, positions,
                                            roster slot, injuries, % rostered
    data/raw/standings/YYYY-MM-DD.parquet   team grain: banked YTD category totals

Both are pure Fantrax: one tidy frame per source, no cross-source joins and no
derived values. In particular this module does NOT merge projections, does NOT
do FanGraphs name reconciliation and does NOT add -H/-P suffixes — that is
`build.py`'s job, downstream of the raw layer. `FANTRAX_NAME_CORRECTIONS` and
`get_player_type` live here because the knowledge is Fantrax-specific; both are
applied by `build.merge_fantrax`, not baked into the snapshot, so the raw layer
keeps Fantrax's own spelling verbatim.

Auth reads cookies straight from a logged-in browser (see `browser_auth`), so
there is nothing to paste; `config.json` cookies remain a headless fallback.
Losing the session is still the most common failure mode, so
`refresh_fantrax_snapshots` verifies auth before fetching anything.

Note there is no single Fantrax login cookie, and the set has already changed
once: a live session authenticated on `FX_RM` + `ui`/`uig` + Cloudflare's
`cf_clearance`/`__cf_bm` with no `JSESSIONID` at all. Never assert on one
cookie name — send the whole jar and let `test_auth` decide.
"""

import datetime
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm

from .browser_auth import load_browser_cookies
from .config import FANTRAX_COOKIES, FANTRAX_LEAGUE_ID, FANTRAX_TEAM_IDS
from .raw_io import write_raw

# =============================================================================
# CONFIGURATION
# =============================================================================

FANTRAX_API_URL = "https://www.fantrax.com/fxpa/req"

# Exceptional name corrections for cases that normalization can't handle
# (different names entirely, not just accent/suffix variations).
# Most matching is done via normalize_name() — this is for true edge cases only.
# Consumed by build.py during FanGraphs reconciliation, NOT applied to the
# snapshot: the raw layer stores names as Fantrax spells them.
FANTRAX_NAME_CORRECTIONS = {
    "Logan OHoppe": "Logan O'Hoppe",  # Missing apostrophe
    "Leodalis De Vries": "Leo De Vries",  # Completely different first name
}

# Fantrax's own roster-slot enum, from the API's `statusTotals` block.
# Do NOT guess these.
ROSTER_STATUS_BY_ID = {
    "1": "active",
    "2": "reserve",
    "3": "IR",
    "9": "minors",
}

# The player-grain snapshot's column contract. `fantrax_id` is Fantrax's own
# player id; the previous name-based merge captured then silently dropped it, so
# it is preserved here deliberately — never drop it. NOTE: build.merge_fantrax
# carries it onto `players` but does not yet JOIN on it (matching is still the
# name cascade), and merge_market still matches ADP on name. Wiring those to
# fantrax_id is the obvious next step now that the column survives.
FANTRAX_SNAPSHOT_COLUMNS = [
    "fantrax_id",
    "name",
    "Position",
    "mlb_team",
    "player_type",
    "age",
    "owner",
    "roster_status",
    "status_id",
    "injury_status",
    "injury_detail",
    "rookie",
    "minors_eligible",
    "eligible_positions",
    "adp",
    "fantrax_score",
    "pct_rostered",
    "roster_trend",
    "fantrax_rank",
    "is_free_agent",
]


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
# HELPERS
# =============================================================================

# Fantrax encodes player state as `scorer.icons`, a list of {tooltip, typeId}.
# Injury-relevant codes (catalogued from live packets across all 7 rosters):
#   typeId "1" -> Day-to-Day (e.g. "Oblique - Day-to-Day"); also non-injury
#                 absences like "Paternity Leave - Day-to-Day"
#   typeId "2" -> On the Injured List (e.g. "Injured List - 10-day IL - Oblique")
# All other typeIds are lineup/handedness/batting-order/news markers (not injury).
_ICON_INJURED_LIST = "2"
_ICON_DAY_TO_DAY = "1"


def _parse_injury(scorer: dict) -> tuple[str | None, str | None]:
    """Extract injury state from a Fantrax scorer's icons.

    Returns:
        (injury_status, injury_detail) where injury_status is "IL", "DTD",
        or None, and injury_detail is the raw Fantrax tooltip (or None).
        IL takes precedence over DTD when both icons are present.
    """
    icons = scorer.get("icons", []) or []
    il_detail = None
    dtd_detail = None
    for icon in icons:
        if not isinstance(icon, dict):
            continue
        type_id = str(icon.get("typeId", ""))
        if type_id == _ICON_INJURED_LIST:
            il_detail = icon.get("tooltip")
        elif type_id == _ICON_DAY_TO_DAY:
            dtd_detail = icon.get("tooltip")
    if il_detail is not None:
        return "IL", il_detail
    if dtd_detail is not None:
        return "DTD", dtd_detail
    return None, None


def _parse_cell(cells: list, idx: int, as_float: bool = False) -> int | float | None:
    """Extract a typed value from a Fantrax API cells array."""
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
        cleaned = content.lstrip("-").replace(".", "")
        if cleaned.isdigit():
            return float(content)
        return None
    return int(content) if content.isdigit() else None


def _response_data(response: requests.Response, what: str) -> dict:
    """Unwrap the first message of a Fantrax `fxpa/req` response.

    Fails loudly on the two ways this goes wrong: an expired cookie (Fantrax
    answers 200 with a `pageError`) and a shape change.
    """
    payload = response.json()
    responses = payload.get("responses", [])
    assert responses, (
        f"Fantrax returned no 'responses' for {what}. "
        f"Top-level keys: {sorted(payload.keys())}. "
        f"This usually means the session is not logged in — log in to "
        f"fantrax.com in your browser and re-run."
    )
    resp0 = responses[0]
    assert "data" in resp0, (
        f"Fantrax returned no data for {what}: "
        f"pageError={resp0.get('pageError')}. "
        f"Log in to fantrax.com in your browser, then re-run."
    )
    return resp0["data"]


# =============================================================================
# AUTHENTICATION
# =============================================================================


def create_session(use_browser: bool = True) -> requests.Session:
    """
    Create a Fantrax session, preferring cookies read from a logged-in browser.

    Browser cookies are tried first and are the supported path: there is nothing
    to paste and nothing to re-paste when the session rolls over. `config.json`
    remains a fallback for headless use.

    Do NOT require any single named cookie here. Fantrax has no one tell-tale
    login cookie and the set has already changed once — it authenticated on
    `FX_RM` + `ui`/`uig` + Cloudflare's `cf_clearance`/`__cf_bm` with NO
    `JSESSIONID` present at all, so an assert on `JSESSIONID` (which this used
    to have) rejects a perfectly good session. Send whatever the browser holds
    and let `test_auth` be the only judge of whether it works.

    Args:
        use_browser: Read cookies from the browser. False forces config.json.

    Returns: Configured requests.Session.
    """
    if use_browser:
        session = requests.Session()
        session.cookies = load_browser_cookies("fantrax.com", label="Fantrax")
        return session

    assert FANTRAX_COOKIES, (
        "config.json has no 'fantrax.cookies' values and browser cookies were "
        "skipped. Either log in to fantrax.com in a browser and call with "
        "use_browser=True, or paste cookie values into config.json."
    )
    session = requests.Session()
    for name, value in FANTRAX_COOKIES.items():
        session.cookies.set(name, value, domain=".fantrax.com")
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
# PLAYER-GRAIN FETCHING
# =============================================================================


def _parse_roster_rows(data: dict, owner: str) -> list[dict]:
    """Parse one team's `getTeamRosterInfo` payload into snapshot rows.

    Args:
        data: The `responses[0].data` block.
        owner: Fantrax team name, written to every row's `owner`.

    Returns:
        One dict per rostered player, keyed by FANTRAX_SNAPSHOT_COLUMNS names.
    """
    rows = []
    tables = data.get("tables", []) or data.get("tableList", [])
    for table in tables:
        for row in table.get("rows", []):
            scorer = row.get("scorer", {})
            if not scorer:
                continue
            pos = scorer.get("posShortNames", "")
            status_id = str(row.get("statusId", ""))
            injury_status, injury_detail = _parse_injury(scorer)
            cells = row.get("cells", [])

            rows.append(
                {
                    "fantrax_id": scorer.get("scorerId"),
                    "name": scorer.get("name"),
                    "Position": pos,
                    "mlb_team": scorer.get("teamShortName"),
                    "player_type": get_player_type(pos),
                    "age": _parse_cell(cells, 0),
                    "owner": owner,
                    # Every player the roster endpoint returns is OWNED,
                    # whatever slot they sit in (active/reserve/IR/minors).
                    "roster_status": ROSTER_STATUS_BY_ID.get(status_id, "unknown"),
                    "status_id": status_id,
                    "injury_status": injury_status,
                    "injury_detail": injury_detail,
                    "rookie": scorer.get("rookie", False),
                    "minors_eligible": scorer.get("minorsEligible", False),
                    "eligible_positions": scorer.get("posIds", []),
                    "adp": _parse_cell(cells, 2, as_float=True),
                    "is_free_agent": False,
                }
            )
    return rows


def fetch_team_roster(
    session: requests.Session, team_id: str, owner: str
) -> list[dict]:
    """
    Fetch one team's roster.

    API: getTeamRosterInfo with view="STATS"

    Args:
        session: Authenticated session from create_session().
        team_id: Fantrax team id.
        owner: Fantrax team name, stamped onto each row.

    Returns:
        List of player dicts (see _parse_roster_rows).
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
    data = _response_data(response, f"roster of team {owner} ({team_id})")
    rows = _parse_roster_rows(data, owner)
    assert rows, (
        f"Parsed 0 players from {owner}'s roster ({team_id}). The response had "
        f"keys {sorted(data.keys())} — the getTeamRosterInfo table shape "
        f"probably changed."
    )
    return rows


def fetch_rosters(session: requests.Session) -> pd.DataFrame:
    """
    Fetch all 7 teams' rosters as one player-grain DataFrame.

    Returns:
        DataFrame with one row per owned player and an `owner` column holding
        the Fantrax team name. Columns are a subset of
        FANTRAX_SNAPSHOT_COLUMNS (the pool-only fields are absent).
    """
    print(f"Fetching rosters for {len(FANTRAX_TEAM_IDS)} teams...")
    rows: list[dict] = []

    for team_name, team_id in tqdm(FANTRAX_TEAM_IDS.items(), desc="Fetching rosters"):
        team_rows = fetch_team_roster(session, team_id, team_name)
        rows.extend(team_rows)
        hitters = sum(1 for r in team_rows if r["player_type"] == "hitter")
        pitchers = len(team_rows) - hitters
        print(f"  {team_name}: {len(team_rows)} players ({hitters} H, {pitchers} P)")

    df = pd.DataFrame(rows)
    print(f"  {len(df)} owned players across {df['owner'].nunique()} teams")
    return df


def _parse_pool_rows(data: dict) -> list[dict]:
    """Parse a `getPlayerStats` payload into snapshot rows.

    Cell layout differs from the roster table: 0=rank, 1=status, 2=age,
    3=Fantrax score, 4=% rostered, 5=roster trend.
    """
    rows = data.get("statsTable", [])
    assert len(rows) > 0, (
        f"Fantrax returned no player data in 'statsTable'. "
        f"Response keys: {sorted(data.keys())}"
    )

    out = []
    for row in rows:
        scorer = row.get("scorer", {})
        pos = scorer.get("posShortNames", "")
        injury_status, injury_detail = _parse_injury(scorer)
        cells = row.get("cells", [])

        status_cell = cells[1] if len(cells) > 1 else {}
        status = status_cell.get("content", "") if isinstance(status_cell, dict) else ""

        out.append(
            {
                "fantrax_id": scorer.get("scorerId"),
                "name": scorer.get("name"),
                "Position": pos,
                "mlb_team": scorer.get("teamShortName"),
                "player_type": get_player_type(pos),
                "age": _parse_cell(cells, 2),
                "injury_status": injury_status,
                "injury_detail": injury_detail,
                "rookie": scorer.get("rookie", False),
                "minors_eligible": scorer.get("minorsEligible", False),
                "eligible_positions": scorer.get("posIds", []),
                "fantrax_score": _parse_cell(cells, 3, as_float=True),
                "pct_rostered": _parse_cell(cells, 4, as_float=True),
                "roster_trend": _parse_cell(cells, 5, as_float=True),
                "fantrax_rank": _parse_cell(cells, 0),
                "is_free_agent": status == "FA",
            }
        )
    return out


def fetch_player_pool(
    session: requests.Session,
    max_results: int | None = None,
) -> pd.DataFrame:
    """
    Fetch the league's whole player pool (rostered players included).

    API: getPlayerStats (single request; pagination params are unreliable, so
    one call with maxResultsPerPage=5000 — the API max — is the way).

    Args:
        session: Authenticated session from create_session().
        max_results: Limit total players (None = 5000, the API max).

    Returns:
        DataFrame with one row per player; `is_free_agent` flags the unowned
        ones. Columns are a subset of FANTRAX_SNAPSHOT_COLUMNS (the roster-only
        fields are absent).
    """
    print("Fetching player pool from Fantrax...")

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
    data = _response_data(response, "player pool")

    total_available = data.get("paginatedResultSet", {}).get("totalNumResults", 0)
    print(
        f"  API reports {total_available:,} total players (fetching top {request_size:,})"
    )

    df = pd.DataFrame(_parse_pool_rows(data))
    print(
        f"  Fetched {len(df):,} players ({int(df['is_free_agent'].sum()):,} free agents)"
    )
    return df


def assemble_fantrax_snapshot(
    rosters: pd.DataFrame, player_pool: pd.DataFrame
) -> pd.DataFrame:
    """
    Combine the roster and player-pool fetches into one tidy player-grain frame.

    Both inputs come from Fantrax, so this is not a cross-source join: it is the
    de-duplication that makes the snapshot one row per player. The pool endpoint
    returns rostered players too, and it is the only place `fantrax_score` and
    `pct_rostered` exist, so the pool cannot simply be filtered to free agents.

    Roster rows take precedence field-by-field (the roster endpoint is
    authoritative for positions and slot), with the pool filling anything the
    roster leaves null.

    Args:
        rosters: From fetch_rosters().
        player_pool: From fetch_player_pool().

    Returns:
        DataFrame with exactly FANTRAX_SNAPSHOT_COLUMNS, one row per
        `fantrax_id`. `owner` and `roster_status` are null for unowned players.
    """
    assert len(rosters) > 0, (
        "assemble_fantrax_snapshot: rosters frame is empty. fetch_rosters "
        "returned nothing, which means the fetch failed rather than that the "
        "league is empty."
    )
    assert len(player_pool) > 0, (
        "assemble_fantrax_snapshot: player_pool frame is empty. "
        "fetch_player_pool returned nothing — the fetch failed."
    )
    for label, frame in (("rosters", rosters), ("player_pool", player_pool)):
        assert "fantrax_id" in frame.columns, (
            f"assemble_fantrax_snapshot: {label} has no 'fantrax_id' column "
            f"(got {list(frame.columns)}). fantrax_id is the join key to every "
            f"other Fantrax-keyed source — it must not be dropped."
        )
        n_missing = frame["fantrax_id"].isna().sum()
        assert n_missing == 0, (
            f"assemble_fantrax_snapshot: {n_missing} of {len(frame)} {label} "
            f"rows have a null fantrax_id, so they cannot be de-duplicated or "
            f"joined. Check that scorer.scorerId is still in the API payload."
        )

    dupes = rosters.loc[rosters["fantrax_id"].duplicated(keep=False), "fantrax_id"]
    assert dupes.empty, (
        f"assemble_fantrax_snapshot: {dupes.nunique()} player(s) appear on more "
        f"than one roster: {sorted(set(dupes))}. One player cannot be owned "
        f"twice — the roster fetch returned overlapping teams."
    )

    combined = pd.concat([rosters, player_pool], ignore_index=True)
    missing = [c for c in FANTRAX_SNAPSHOT_COLUMNS if c not in combined.columns]
    assert not missing, (
        f"assemble_fantrax_snapshot: neither fetch supplied columns {missing}. "
        f"Available: {sorted(combined.columns)}. Update the parsers or "
        f"FANTRAX_SNAPSHOT_COLUMNS."
    )

    # groupby.first() takes the first NON-NULL value per column, and roster rows
    # come first, so this is "roster wins, pool fills the gaps" in one pass.
    snapshot = (
        combined[FANTRAX_SNAPSHOT_COLUMNS]
        .groupby("fantrax_id", as_index=False, sort=False)
        .first()
    )

    owned = snapshot["owner"].notna()
    assert owned.sum() == len(rosters), (
        f"assemble_fantrax_snapshot: {owned.sum()} players carry an owner but "
        f"{len(rosters)} roster rows went in — the de-duplication lost or "
        f"merged owned players."
    )
    assert snapshot.loc[owned, "roster_status"].notna().all(), (
        "assemble_fantrax_snapshot: some owned players have a null "
        "roster_status. Every roster row should decode a statusId via "
        "ROSTER_STATUS_BY_ID."
    )
    assert snapshot["roster_status"][~owned].isna().all(), (
        "assemble_fantrax_snapshot: some unowned players carry a roster_status. "
        "roster_status must be null wherever owner is null."
    )

    print(
        f"Assembled Fantrax snapshot: {len(snapshot)} players "
        f"({int(owned.sum())} owned, {len(snapshot) - int(owned.sum())} unowned)"
    )
    return snapshot


# =============================================================================
# STANDINGS PARSING HELPERS
# =============================================================================

# Fantrax standings header `shortName` → our standings column name. The
# "Standings - Stat Totals" table carries the REAL season-to-date totals
# (not the roto points), and each row's `cells` align by index to the table's
# `header.cells`. AB and IP are captured as playing-time weights for ratio
# blending downstream.
_STANDINGS_SHORTNAME_TO_COL: dict[str, str] = {
    "R": "r",
    "HR": "hr",
    "RBI": "rbi",
    "SB": "sb",
    "OPS": "ops",
    "ERA": "era",
    "WHIP": "whip",
    "K": "k",
    "W": "w",
    "SV": "sv",
    "AB": "ab",
    "IP": "ip",
}


def _to_float(content) -> float | None:
    """Parse a Fantrax cell value to float, tolerating thousands separators."""
    text = str(content).replace(",", "").replace("%", "").strip()
    if not text or text == "-":
        return None
    # Reject anything that isn't a plain number rather than raising: the
    # standings tables mix in labels, and banked.py validates the result.
    if text.lstrip("-").replace(".", "", 1).isdigit():
        return float(text)
    return None


def _parse_ip(content) -> float | None:
    """Convert baseball IP notation to decimal innings (418.2 → 418.667).

    Fantrax reports IP with the fractional digit as thirds of an inning
    (.1 = 1/3, .2 = 2/3), not as a true decimal. A plain float parse would be
    off by up to ~0.27 IP — negligible as a blend weight, but converting is
    cheap and correct.
    """
    raw = _to_float(content)
    if raw is None:
        return None
    whole = int(raw)
    frac_digit = round((raw - whole) * 10)
    if frac_digit in (1, 2):
        return whole + frac_digit / 3.0
    return raw


def _team_cell(row: dict) -> dict | None:
    """Return the team identity cell for a standings row (in fixedCells)."""
    for cell in row.get("fixedCells", []):
        if isinstance(cell, dict) and "teamId" in cell:
            return cell
    # Fallback for table variants that inline the team in `cells`.
    for cell in row.get("cells", []):
        if isinstance(cell, dict) and "teamId" in cell:
            return cell
    return None


def _parse_standings_data(data: dict) -> list[dict]:
    """
    Parse league standings into one row per team with real category totals.

    The Fantrax getStandings response contains several tables; the
    **"Standings - Stat Totals"** table holds the authoritative season-to-date
    values (R, HR, …, OPS, ERA, WHIP, K, W, SV, plus AB/IP). We map the table's
    `header.cells` shortNames to column indices and read each row's `cells`
    accordingly, taking team identity from `row.fixedCells`. This is exact (no
    heuristics) — the earlier title-based heuristic accidentally read the
    *roto-points* table for ERA/WHIP, corrupting those rates.

    If the Stat-Totals table is absent (unexpected response shape), returns
    team rows without category columns; downstream
    ``optimizer.banked.standings_to_banked_totals`` then safely falls back to
    rest-of-season-only.
    """
    tables = data.get("tables") or data.get("tableList") or []

    stat_table = None
    for table in tables:
        if (
            isinstance(table, dict)
            and "stat totals" in str(table.get("caption", "")).lower()
        ):
            stat_table = table
            break

    rows_out: list[dict] = []

    if stat_table is None:
        # Best-effort team identity only (keeps the standings display working).
        seen: set = set()
        for table in tables:
            if not isinstance(table, dict):
                continue
            for row in table.get("rows", []):
                if not isinstance(row, dict):
                    continue
                tc = _team_cell(row)
                if tc is None or tc.get("teamId") in seen:
                    continue
                seen.add(tc.get("teamId"))
                rows_out.append(
                    {
                        "team_id": tc.get("teamId"),
                        "team_name": tc.get("content", "Unknown"),
                        "overall_rank": 1,
                        "total_points": 0,
                    }
                )
    else:
        header_cells = stat_table.get("header", {}).get("cells", [])
        col_index = {
            c.get("shortName"): i
            for i, c in enumerate(header_cells)
            if isinstance(c, dict) and c.get("shortName")
        }
        for row in stat_table.get("rows", []):
            if not isinstance(row, dict):
                continue
            tc = _team_cell(row)
            if tc is None:
                continue
            cells = row.get("cells", [])
            entry = {
                "team_id": tc.get("teamId"),
                "team_name": tc.get("content", "Unknown"),
                "overall_rank": 1,
                "total_points": 0,
            }
            for short_name, col in _STANDINGS_SHORTNAME_TO_COL.items():
                i = col_index.get(short_name)
                if i is None or i >= len(cells):
                    continue
                content = (
                    cells[i].get("content") if isinstance(cells[i], dict) else cells[i]
                )
                value = _parse_ip(content) if col == "ip" else _to_float(content)
                if value is not None:
                    entry[col] = value
            rows_out.append(entry)

    rows_out.sort(key=lambda x: x.get("team_name", ""))
    for i, row in enumerate(rows_out):
        row["overall_rank"] = i + 1

    return rows_out


def fetch_standings(session: requests.Session) -> pd.DataFrame:
    """
    Fetch current league standings (banked year-to-date category totals).

    API: getStandings

    Returns:
        DataFrame with columns team_id, team_name, overall_rank, total_points
        and the lowercase category totals r, hr, rbi, sb, ops, w, sv, k, era,
        whip plus ab / ip when the Stat-Totals table is present. This is the
        contract ``optimizer.banked.standings_to_banked_totals`` consumes; it
        range-validates the rate stats and falls back to rest-of-season-only if
        the category columns are missing.
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
    data = _response_data(response, "standings")
    rows = _parse_standings_data(data)

    assert rows, (
        f"Could not parse any team from the standings response "
        f"(keys: {sorted(data.keys())}). The getStandings table shape changed."
    )

    df = pd.DataFrame(rows)
    n_teams = len(FANTRAX_TEAM_IDS)
    assert len(df) == n_teams, (
        f"Parsed {len(df)} standings rows but the league has {n_teams} teams: "
        f"{sorted(df['team_name'])}. A partial parse would bank the wrong "
        f"totals, so refusing to continue."
    )

    missing_cats = [
        c for c in ("r", "hr", "rbi", "sb", "ops", "era", "whip") if c not in df.columns
    ]
    if missing_cats:
        print(
            f"WARNING: standings are missing category columns {missing_cats} — "
            f"the 'Stat Totals' table was absent from the response. The "
            f"optimizer will run rest-of-season-only."
        )

    print(f"Standings ({len(df)} teams):")
    for _, row in df.sort_values("overall_rank").iterrows():
        print(f"  {int(row['overall_rank'])}. {row['team_name']}")

    return df


# =============================================================================
# ENTRY POINT
# =============================================================================


def refresh_fantrax_snapshots(
    date: datetime.date | None = None,
    max_pool_results: int | None = None,
) -> tuple[Path, Path]:
    """
    Authenticate once, then write both Fantrax raw snapshots.

    Args:
        date: Snapshot date. Defaults to today.
        max_pool_results: Cap on player-pool rows (None = 5000, the API max).

    Returns:
        (fantrax_snapshot_path, standings_snapshot_path)
    """
    print("=== Refreshing Fantrax raw snapshots ===")
    session = create_session()
    assert test_auth(session), (
        "Fantrax authentication failed: the browser has fantrax.com cookies "
        "but Fantrax says they are not a logged-in session.\n"
        "Fix: open fantrax.com in your browser, log in, load your league page "
        "once (that is what mints the session), then re-run. If you use a "
        "browser other than Brave/Chrome, check browser_auth.COOKIE_BROWSERS."
    )
    print("Authenticated.")

    rosters = fetch_rosters(session)
    player_pool = fetch_player_pool(session, max_results=max_pool_results)
    snapshot = assemble_fantrax_snapshot(rosters, player_pool)
    fantrax_path = write_raw(snapshot, "fantrax", date)

    standings = fetch_standings(session)
    standings_path = write_raw(standings, "standings", date)

    print("=== Fantrax snapshots written ===")
    return fantrax_path, standings_path


def main() -> None:
    """CLI entry: write today's `fantrax` and `standings` raw snapshots."""
    refresh_fantrax_snapshots()


if __name__ == "__main__":
    main()
