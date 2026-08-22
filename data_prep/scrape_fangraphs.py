"""
Scrape FanGraphs projections via their internal JSON API into the raw layer.

Uses browser cookies for authentication, fetches the in-season rest-of-season
Steamer and ATC DC projections, and writes ONE dated parquet snapshot per
projection system holding hitters and pitchers together:

    data/raw/projections/steamer/YYYY-MM-DD.parquet
    data/raw/projections/atc/YYYY-MM-DD.parquet

These are the most recent (rest-of-season) feeds, not the preseason-frozen
projections — see PROJECTION_TYPES.

Raw-layer discipline: one source, structurally tidy, no cross-source joins and
no derived values. So hitters and pitchers are concatenated and tagged with
`player_type`, but nothing is merged in (positions, ages, market value), no
stat is renamed to its fantasy alias (`SO` stays `SO`), no `-H`/`-P` suffix is
appended, and the opposite type's stat columns are left NULL rather than
zero-filled. All of that is `build.py`'s job.

Usage:
    uv run python -m data_prep.scrape_fangraphs
"""

from pathlib import Path

import browser_cookie3
import pandas as pd
import requests

from .raw_io import write_raw

FANGRAPHS_API_URL = "https://www.fangraphs.com/api/projections"

# Logical name -> FanGraphs API `type` param. We pull the in-season
# rest-of-season feeds (NOT the preseason-frozen "steamer"/"atc" feeds, which
# never update once the season starts). FanGraphs prefixes RoS feeds with "r";
# "ratcdc" is ATC's in-season DC variant (the only working updated ATC feed —
# the full-season "atcdc" endpoint returns HTTP 500). These return remaining
# (rest-of-season) PA/IP and totals, not full-season projections.
#
# "oopsypeak" is NOT a rest-of-season feed: it is OOPSY's projected CAREER PEAK
# season, normalised to 600 PA / 198 IP (or 70 IP for relievers) and
# park-neutral. It is the talent input for dynasty ceiling work
# (`data_prep.ceiling`), not for this season's lineup, which is why
# ROS_SYSTEMS below — what `uv run fetch projections` refreshes — excludes it.
PROJECTION_TYPES = {
    "steamer": "steamerr",
    "atc": "ratcdc",
    "oopsypeak": "oopsypeak",
}

# The subset `scrape_projections()` refreshes by default. Keeping this separate
# from PROJECTION_TYPES means registering a non-RoS feed (oopsypeak) cannot
# silently change what the daily projections fetch pulls.
ROS_SYSTEMS: list[str] = ["steamer", "atc"]

# ── Columns kept in the snapshot ───────────────────────────────────────────

# Deliberately a subset of the ~74-column FanGraphs export. Two reasons, and
# the second is load-bearing: (1) nothing downstream reads the percentile /
# ADP / advanced-rate columns, and (2) hitter and pitcher feeds REUSE stat
# keys with opposite meanings — hitter `R`/`HR`/`SO` are runs scored, homers
# hit and strikeouts taken, while pitcher `R`/`HR`/`SO` are runs and homers
# ALLOWED and strikeouts recorded. Concatenating the full frames would silently
# stack those into one column. Restricting each side to its own scoring
# categories keeps the shared columns genuinely shared.
SHARED_COLUMNS = [
    "Name",
    "Team",
    "player_type",
    "PlayerId",  # FanGraphs player id — the join key for market data.
    "MLBAMID",
]

HITTER_STAT_COLUMNS = ["PA", "AB", "R", "HR", "RBI", "SB", "OPS", "WAR"]

PITCHER_STAT_COLUMNS = ["IP", "W", "SV", "SO", "ERA", "WHIP", "WAR"]

# The API's stat keys already match these names; only name and ids differ.
API_RENAMES = {
    "PlayerName": "Name",
    "xMLBAMID": "MLBAMID",
    "playerids": "PlayerId",
}


# Browsers to probe for a logged-in FanGraphs session, in priority order.
# browser_cookie3 exposes one loader function per browser; whichever has the
# wordpress_logged_in cookie wins. (Chromium-based browsers like Arc store
# cookies under Chrome's path on macOS and are usually picked up by `chrome`.)
_COOKIE_BROWSERS: tuple[str, ...] = (
    "brave",
    "chrome",
    "edge",
    "vivaldi",
    "opera",
    "firefox",
    "safari",
)


def get_fangraphs_session() -> requests.Session:
    """Create a requests session authenticated from a logged-in browser.

    Auto-detects which installed browser holds a FanGraphs login, so it works
    regardless of which browser you use (not hardcoded to Brave). Each browser
    is probed for ``.fangraphs.com`` cookies; the first one carrying the
    ``wordpress_logged_in`` cookie is used.

    Returns:
        Session with FanGraphs cookies and appropriate headers.
    """
    print("Loading FanGraphs cookies (auto-detecting browser)...")
    cj = None
    chosen = None
    for name in _COOKIE_BROWSERS:
        loader = getattr(browser_cookie3, name, None)
        if loader is None:
            continue
        # Each browser may be absent or its cookie store locked; probe and skip.
        try:
            candidate = loader(domain_name=".fangraphs.com")
        except Exception as exc:  # noqa: BLE001 - browser_cookie3 raises many types
            print(f"  {name}: unavailable ({type(exc).__name__})")
            continue
        names = [c.name for c in candidate]
        has_auth = any("wordpress_logged_in" in n for n in names)
        print(
            f"  {name}: {len(names)} fangraphs cookie(s)"
            f"{' — logged in' if has_auth else ''}"
        )
        if has_auth:
            cj = candidate
            chosen = name
            break

    assert cj is not None, (
        "No FanGraphs login (wordpress_logged_in cookie) found in any supported "
        "browser (Brave, Chrome, Edge, Vivaldi, Opera, Firefox, Safari).\n"
        "  1. Log into fangraphs.com in one of those browsers.\n"
        "  2. The rest-of-season feeds (steamerr / ratcdc) require a FanGraphs "
        "MEMBERSHIP — a free account is not enough.\n"
        "  3. macOS permissions (see the per-browser lines above):\n"
        "     • 'chrome: BrowserCookieError' → Chrome cookies are keychain-"
        "encrypted; run from a normal GUI Terminal and approve the keychain "
        "prompt (a headless/SSH session can't decrypt them).\n"
        "     • 'safari: PermissionError' → grant the app running this "
        "(Terminal / your IDE) Full Disk Access in System Settings → Privacy & "
        "Security, then retry."
    )
    print(f"  Using {chosen} cookies")

    is_member = any(c.name == "fg_is_member" for c in cj)
    if is_member:
        print("  FanGraphs member session detected")

    session = requests.Session()
    session.cookies = cj
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Referer": "https://www.fangraphs.com/projections",
        }
    )
    return session


def fetch_projections(
    session: requests.Session,
    proj_type: str,
    stats_type: str,
) -> list[dict]:
    """Fetch projections JSON from the FanGraphs internal API.

    Args:
        session: Authenticated requests session.
        proj_type: Projection system ("steamer" or "atc").
        stats_type: Player type ("bat" for hitters, "pit" for pitchers).

    Returns:
        List of player dicts from the API.
    """
    assert proj_type in PROJECTION_TYPES, (
        f"Unknown projection type '{proj_type}'. Expected one of: {list(PROJECTION_TYPES)}"
    )
    assert stats_type in ("bat", "pit"), (
        f"stats_type must be 'bat' or 'pit', got '{stats_type}'"
    )

    params = {
        "type": PROJECTION_TYPES[proj_type],
        "stats": stats_type,
        "pos": "all",
    }

    label = f"{proj_type} {'hitters' if stats_type == 'bat' else 'pitchers'}"
    print(f"  Fetching {label}...")

    resp = session.get(FANGRAPHS_API_URL, params=params)

    assert resp.status_code == 200, (
        f"FanGraphs API returned {resp.status_code} for {label}. "
        f"Response: {resp.text[:300]}. "
        f"If this is 403/Cloudflare, your browser cookies may have expired — "
        f"visit fangraphs.com in that browser and retry."
    )

    data = resp.json()
    assert isinstance(data, list), (
        f"Expected list from API, got {type(data).__name__}. Preview: {str(data)[:200]}"
    )
    assert len(data) > 0, f"API returned 0 rows for {label}"

    print(f"    Got {len(data)} players")
    return data


def build_type_frame(
    api_data: list[dict],
    stat_columns: list[str],
    player_type: str,
    label: str,
) -> pd.DataFrame:
    """Convert one projections feed (hitters or pitchers) to a tidy frame.

    Args:
        api_data: Player rows as returned by `fetch_projections`.
        stat_columns: HITTER_STAT_COLUMNS or PITCHER_STAT_COLUMNS.
        player_type: "hitter" or "pitcher", written to the `player_type` column.
        label: Feed description used in assertion messages.

    Returns:
        DataFrame with exactly SHARED_COLUMNS + `stat_columns`, MLBAMID as a
        nullable integer (so it never stringifies as "677951.0") and PlayerId
        as a string (market data joins on it as a string).

    Note:
        A missing column is a hard error, not a silently empty one: if FanGraphs
        renames a field the failure must surface here, where the message can
        say which field vanished.
    """
    assert player_type in ("hitter", "pitcher"), (
        f"player_type must be 'hitter' or 'pitcher', got '{player_type}'"
    )

    df = pd.DataFrame(api_data).rename(columns=API_RENAMES)
    df["player_type"] = player_type

    columns = SHARED_COLUMNS + stat_columns
    missing = [col for col in columns if col not in df.columns]
    assert not missing, (
        f"FanGraphs {label} response is missing expected column(s): {missing}.\n"
        f"  Columns returned: {sorted(df.columns)}\n"
        f"FanGraphs likely renamed a field — update API_RENAMES or the "
        f"SHARED_COLUMNS / HITTER_STAT_COLUMNS / PITCHER_STAT_COLUMNS lists "
        f"in this module."
    )

    df = df[columns].copy()

    # PlayerId is the ONLY join key to market-value data; a null here becomes an
    # unmatched player later, far from the cause.
    n_missing_id = int(df["PlayerId"].isna().sum())
    assert n_missing_id == 0, (
        f"FanGraphs {label} returned {n_missing_id} of {len(df)} rows with a "
        f"null PlayerId. PlayerId is the join key for market data — a null "
        f"there silently drops the player from Ottoneu value matching. "
        f"Check whether the API's `playerids` field was renamed."
    )

    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"]).astype("Int64")
    df["PlayerId"] = df["PlayerId"].astype(str)
    return df


def build_snapshot(
    hitter_data: list[dict],
    pitcher_data: list[dict],
    system: str,
) -> pd.DataFrame:
    """Combine one system's hitter and pitcher feeds into a raw snapshot frame.

    Args:
        hitter_data: `fetch_projections(..., "bat")` output.
        pitcher_data: `fetch_projections(..., "pit")` output.
        system: Projection system name, used in messages only.

    Returns:
        DataFrame with SHARED_COLUMNS + HITTER_STAT_COLUMNS +
        PITCHER_STAT_COLUMNS. Each row carries values only for its own
        `player_type`; the other type's stat columns are NULL (NOT zero —
        zero-filling is the join step's decision, in build.py).
    """
    hitters = build_type_frame(
        hitter_data, HITTER_STAT_COLUMNS, "hitter", f"{system} hitters"
    )
    pitchers = build_type_frame(
        pitcher_data, PITCHER_STAT_COLUMNS, "pitcher", f"{system} pitchers"
    )

    snapshot = pd.concat([hitters, pitchers], ignore_index=True)

    expected = SHARED_COLUMNS + HITTER_STAT_COLUMNS + PITCHER_STAT_COLUMNS
    # WAR is in both stat lists; dedupe while preserving order.
    expected = list(dict.fromkeys(expected))
    assert list(snapshot.columns) == expected, (
        f"{system} snapshot columns {list(snapshot.columns)} != expected "
        f"{expected}. Concatenating the two feeds should produce exactly the "
        f"union of the shared and per-type column lists."
    )
    assert set(snapshot["player_type"]) == {"hitter", "pitcher"}, (
        f"{system} snapshot must contain both player types, got "
        f"{sorted(set(snapshot['player_type']))}. One of the two feeds came "
        f"back empty."
    )

    print(
        f"  Built {system} snapshot: {len(snapshot)} rows "
        f"({len(hitters)} hitters, {len(pitchers)} pitchers)"
    )
    return snapshot


def scrape_projections(systems: list[str] | None = None) -> dict[str, Path]:
    """Scrape FanGraphs projections and write one raw snapshot per system.

    Args:
        systems: Projection systems to scrape. Defaults to ROS_SYSTEMS
            (steamer + atc) — NOT every key of PROJECTION_TYPES, which also
            holds the peak feed used only by the ceiling script.

    Returns:
        Dict mapping system name to the parquet path written.
    """
    if systems is None:
        systems = list(ROS_SYSTEMS)

    session = get_fangraphs_session()

    written: dict[str, Path] = {}
    for system in systems:
        print(f"\n{'=' * 50}")
        print(f"Projection system: {system.upper()}")
        print(f"{'=' * 50}")

        hitter_data = fetch_projections(session, system, "bat")
        pitcher_data = fetch_projections(session, system, "pit")
        snapshot = build_snapshot(hitter_data, pitcher_data, system)
        written[system] = write_raw(snapshot, f"projections/{system}")

    print(f"\nDone. {len(written)} projection snapshot(s) written.")
    return written


def main() -> None:
    """Entry point: scrape every projection system into today's raw snapshots."""
    scrape_projections()


if __name__ == "__main__":
    main()
