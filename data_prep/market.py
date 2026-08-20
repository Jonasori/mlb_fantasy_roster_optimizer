"""
Market-driven player value fetchers: Ottoneu auction salaries, Fantrax ADP,
ESPN ownership, HarryKnowsBall dynasty ranks.

All fetchers are unauthenticated GETs returning tidy DataFrames. The `fetch_*`
functions are pure — they hit the network and return a frame, nothing else.
Persistence to the raw layer is a separate step (`fetch_market_source` /
`fetch_all_market`), so each source lands independently under
`data/raw/market/<name>/<YYYY-MM-DD>.parquet` and one dead scraper cannot block
the other three. Joining these frames onto `players` belongs to build.py.
"""

import json
import re
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from . import raw_io
from .names import strip_diacritics

OTTONEU_VALUES_URL = "https://ottoneu.fangraphs.com/averageValues?export=csv"
FANTRAX_ADP_URL = "https://www.fantrax.com/fxea/general/getAdp?sport=MLB"
ESPN_PLAYERS_URL = (
    "https://lm-api-reads.fantasy.espn.com/apis/v3/games/flb/seasons/{season}/players"
)
HKB_RANKINGS_URL = "https://harryknowsball.com/rankings"

REQUEST_TIMEOUT = 60

OTTONEU_COLUMN_MAP = {
    "Name": "name",
    "OttoneuID": "ottoneu_id",
    "FG MajorLeagueID": "fg_id",
    "FG MinorLeagueID": "fg_minor_id",
    "MLB Org": "mlb_org",
    "Position(s)": "positions",
    "Avg Salary": "avg_salary",
    "Median Salary": "median_salary",
    "Min Salary": "min_salary",
    "Max Salary": "max_salary",
    "Last 10": "last_10",
    "Roster%": "roster_pct",
}
OTTONEU_SALARY_COLUMNS = [
    "avg_salary",
    "median_salary",
    "min_salary",
    "max_salary",
    "last_10",
]


def parse_dollar_series(values: pd.Series) -> pd.Series:
    """Convert dollar strings like '$78.64' to floats. Blanks become NaN."""
    stripped = values.astype(str).str.replace(r"[$,]", "", regex=True).str.strip()
    numeric = pd.to_numeric(stripped.replace({"": None, "nan": None}), errors="coerce")
    return numeric.astype(float)


def flip_last_first(name: str) -> str:
    """
    Convert a Fantrax 'Last, First' name to 'First Last'.

    Any -H / -P style tag on the surname is preserved on the flipped name
    ('Ohtani-H, Shohei' -> 'Shohei Ohtani-H'), matching the repo's suffix
    convention. Names without a comma are returned unchanged.
    """
    if "," not in name:
        return name.strip()
    last, first = name.split(",", 1)
    return f"{first.strip()} {last.strip()}".strip()


def fetch_ottoneu_values() -> pd.DataFrame:
    """
    Fetch league-wide average Ottoneu auction salaries (no auth required).

    Returns:
        DataFrame with columns: name, ottoneu_id, fg_id, fg_minor_id, mlb_org,
        positions, avg_salary, median_salary, min_salary, max_salary, last_10,
        roster_pct, salary_momentum.

        `fg_id` is the FanGraphs major-league player id as a string — the join
        key to the `PlayerId` column of FanGraphs projection CSVs.
        `salary_momentum` = last_10 - median_salary, i.e. how far recent auction
        prices sit above (positive) or below (negative) the long-run price.
    """
    print(f"Fetching Ottoneu average values from {OTTONEU_VALUES_URL} ...")
    response = requests.get(OTTONEU_VALUES_URL, timeout=REQUEST_TIMEOUT)
    assert response.status_code == 200, (
        f"Ottoneu returned HTTP {response.status_code} for {OTTONEU_VALUES_URL}. "
        f"The export endpoint may have moved or be rate limiting."
    )

    df = pd.read_csv(
        StringIO(response.text),
        dtype={"OttoneuID": str, "FG MajorLeagueID": str, "FG MinorLeagueID": str},
    )
    missing = set(OTTONEU_COLUMN_MAP) - set(df.columns)
    assert not missing, (
        f"Ottoneu CSV is missing expected columns {sorted(missing)}. "
        f"Got: {list(df.columns)}. Update OTTONEU_COLUMN_MAP to match."
    )

    df = df.rename(columns=OTTONEU_COLUMN_MAP)[list(OTTONEU_COLUMN_MAP.values())]
    for column in OTTONEU_SALARY_COLUMNS:
        df[column] = parse_dollar_series(df[column])
    df["roster_pct"] = pd.to_numeric(df["roster_pct"], errors="coerce")
    df["salary_momentum"] = df["last_10"] - df["median_salary"]

    assert len(df) > 100, (
        f"Only {len(df)} Ottoneu rows parsed — expected >1000. "
        f"The CSV export is probably truncated or behind a login now."
    )
    print(
        f"  Loaded {len(df)} Ottoneu players "
        f"({df['fg_id'].notna().sum()} with a FanGraphs major-league id), "
        f"top salary ${df['median_salary'].max():.0f}"
    )
    return df


def fetch_fantrax_adp() -> pd.DataFrame:
    """
    Fetch Fantrax MLB average draft position (no auth required).

    Returns:
        DataFrame with columns: name ('First Last'), position, fantrax_id, adp,
        sorted ascending by ADP.
    """
    print(f"Fetching Fantrax ADP from {FANTRAX_ADP_URL} ...")
    response = requests.get(FANTRAX_ADP_URL, timeout=REQUEST_TIMEOUT)
    assert response.status_code == 200, (
        f"Fantrax returned HTTP {response.status_code} for {FANTRAX_ADP_URL}."
    )

    records = response.json()
    assert isinstance(records, list) and len(records) > 100, (
        f"Expected a JSON list of >100 ADP records, got "
        f"{type(records).__name__} of length {len(records)}."
    )

    df = pd.DataFrame(
        {
            "name": [flip_last_first(strip_diacritics(r["name"])) for r in records],
            "position": [r["pos"] for r in records],
            "fantrax_id": [r["id"] for r in records],
            "adp": pd.to_numeric([r["ADP"] for r in records], errors="coerce"),
        }
    ).sort_values("adp", ignore_index=True)

    print(f"  Loaded {len(df)} ADPs (best: {df.loc[0, 'name']} at {df.loc[0, 'adp']})")
    return df


def fetch_espn_ownership(season: int = 2026, limit: int = 2000) -> pd.DataFrame:
    """
    Fetch ESPN fantasy baseball roster-ownership percentages.

    Args:
        season: ESPN season year.
        limit: Max players to return, taken from the top of the ownership list.

    Returns:
        DataFrame with columns: name, espn_id, pct_owned, pct_change.

    Note:
        The `x-fantasy-filter` keys must sit at the JSON *root*, not nested under
        "players" — nested filters are silently ignored and all ~23k players come
        back. `pct_change` is NaN under the `players_wl` view; only
        `kona_player_info` carries ownership.percentChange, at ~100 kB of stats
        per player (>100 MB here), so it is not worth the payload.
    """
    url = ESPN_PLAYERS_URL.format(season=season)
    espn_filter = {
        "filterActive": {"value": True},
        "limit": limit,
        "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
    }
    print(f"Fetching ESPN ownership for {season} (top {limit} by % owned) ...")
    response = requests.get(
        url,
        params={"view": "players_wl"},
        headers={"x-fantasy-filter": json.dumps(espn_filter)},
        timeout=REQUEST_TIMEOUT,
    )
    assert response.status_code == 200, (
        f"ESPN returned HTTP {response.status_code} for {url}. "
        f"Filter sent: {json.dumps(espn_filter)}"
    )

    records = response.json()
    assert isinstance(records, list) and len(records) > 100, (
        f"Expected a JSON list of >100 ESPN players, got "
        f"{type(records).__name__} of length {len(records)}. "
        f"The x-fantasy-filter header is likely malformed."
    )

    ownership = [r.get("ownership") or {} for r in records]
    df = pd.DataFrame(
        {
            "name": [strip_diacritics(str(r["fullName"])) for r in records],
            "espn_id": [r["id"] for r in records],
            "pct_owned": pd.to_numeric(
                [o.get("percentOwned") for o in ownership], errors="coerce"
            ),
            "pct_change": pd.to_numeric(
                [o.get("percentChange") for o in ownership], errors="coerce"
            ),
        }
    )

    print(
        f"  Loaded {len(df)} ESPN players "
        f"({df['pct_owned'].notna().sum()} with ownership, "
        f"{df['pct_change'].notna().sum()} with pct_change)"
    )
    return df


def fetch_hkb_dynasty_values() -> pd.DataFrame:
    """
    Fetch HarryKnowsBall crowdsourced MLB dynasty ranks and values.

    There is no public API. The /rankings page is a Next.js app that embeds the
    full ranking list in its `__NEXT_DATA__` script tag, so this reads that JSON
    blob rather than parsing HTML.

    Returns:
        DataFrame with columns: name, rank, value (HKB's 0-10000 dynasty value),
        value_change_30d, sorted by rank.
    """
    print(f"Fetching HarryKnowsBall dynasty ranks from {HKB_RANKINGS_URL} ...")
    response = requests.get(HKB_RANKINGS_URL, timeout=REQUEST_TIMEOUT)
    assert response.status_code == 200, (
        f"harryknowsball.com returned HTTP {response.status_code} for "
        f"{HKB_RANKINGS_URL}."
    )

    match = re.search(
        r'id="__NEXT_DATA__"[^>]*>(.*?)</script>', response.text, re.DOTALL
    )
    assert match is not None, (
        "No __NEXT_DATA__ script tag found on harryknowsball.com/rankings. "
        "The site has changed framework or now renders rankings client-side — "
        "re-investigate for a /_next/data/<buildId>/rankings.json route."
    )

    players = json.loads(match.group(1))["props"]["pageProps"]["players"]
    assert len(players) > 100, (
        f"Only {len(players)} players in the HKB __NEXT_DATA__ payload — "
        f"expected >1000. The rankings page shape has changed."
    )

    df = pd.DataFrame(
        {
            "name": [strip_diacritics(str(p["name"])) for p in players],
            "rank": [p["rank"] for p in players],
            "value": [p.get("value") for p in players],
            "value_change_30d": [p.get("valueChange30Days") for p in players],
        }
    ).sort_values("rank", ignore_index=True)

    print(f"  Loaded {len(df)} HKB ranks (#1: {df.loc[0, 'name']})")
    return df


# Raw-layer source name -> fetcher. Keys are the directory under
# data/raw/market/, so adding a source is one line here.
MARKET_FETCHERS = {
    "ottoneu": fetch_ottoneu_values,
    "adp": fetch_fantrax_adp,
    "espn": fetch_espn_ownership,
    "hkb": fetch_hkb_dynasty_values,
}


def fetch_market_source(name: str) -> Path:
    """
    Fetch one market source and write today's raw snapshot.

    Args:
        name: Key of MARKET_FETCHERS ("ottoneu", "adp", "espn", "hkb").

    Returns:
        Path written, `data/raw/market/<name>/<YYYY-MM-DD>.parquet`.
    """
    assert name in MARKET_FETCHERS, (
        f"Unknown market source '{name}'. Known sources: "
        f"{sorted(MARKET_FETCHERS)}. Add a fetcher to MARKET_FETCHERS first."
    )
    return raw_io.write_raw(MARKET_FETCHERS[name](), f"market/{name}")


def fetch_all_market() -> dict[str, Path]:
    """
    Fetch every market source and write each one's raw snapshot.

    Fails fast on the first dead source (per AGENTS.md). Re-run the survivors
    individually with `fetch_market_source(name)` — the raw layer keeps each
    source's snapshots independent, so a broken scraper only staleness-ages its
    own directory.

    Returns:
        Dict of source name -> path written, one entry per MARKET_FETCHERS key.
    """
    print(f"=== Fetching {len(MARKET_FETCHERS)} market sources ===")
    written = {name: fetch_market_source(name) for name in MARKET_FETCHERS}
    print(f"=== Wrote {len(written)} market snapshots: {', '.join(written)} ===")
    return written
