"""
Scrape FanGraphs projections via their internal JSON API.

Uses Brave browser cookies for authentication, fetches the in-season
rest-of-season Steamer and ATC DC projections for hitters and pitchers, and
writes CSVs that match the exact format produced by the FanGraphs "Export
Data" button. These are the most recent (rest-of-season) feeds, not the
preseason-frozen projections — see PROJECTION_TYPES.

Usage:
    uv run python -m data_prep.scrape_fangraphs           # scrape to data/pulled_YYYYMMDD/
    uv run python -m data_prep.scrape_fangraphs --dry-run  # print what would be scraped
"""

import argparse
import unicodedata
from datetime import date
from pathlib import Path

import browser_cookie3
import pandas as pd
import requests

FANGRAPHS_API_URL = "https://www.fangraphs.com/api/projections"

# Logical name -> FanGraphs API `type` param. We pull the in-season
# rest-of-season feeds (NOT the preseason-frozen "steamer"/"atc" feeds, which
# never update once the season starts). FanGraphs prefixes RoS feeds with "r";
# "ratcdc" is ATC's in-season DC variant (the only working updated ATC feed —
# the full-season "atcdc" endpoint returns HTTP 500). These return remaining
# (rest-of-season) PA/IP and totals, not full-season projections.
PROJECTION_TYPES = {
    "steamer": "steamerr",
    "atc": "ratcdc",
}

# ── Column order in FanGraphs CSV exports ──────────────────────────────────

CSV_HITTER_COLUMNS = [
    "Name",
    "Team",
    "G",
    "PA",
    "AB",
    "H",
    "1B",
    "2B",
    "3B",
    "HR",
    "R",
    "RBI",
    "BB",
    "IBB",
    "SO",
    "HBP",
    "SF",
    "SH",
    "GDP",
    "SB",
    "CS",
    "AVG",
    "BB%",
    "K%",
    "BB/K",
    "OBP",
    "SLG",
    "wOBA",
    "OPS",
    "ISO",
    "Spd",
    "BABIP",
    "UBR",
    "wSB",
    "wRC",
    "wRAA",
    "wRC+",
    "BsR",
    "Fld",
    "Off",
    "Def",
    "WAR",
    "ADP",
    "InterSD",
    "InterSK",
    "IntraSD",
    "Vol",
    "Skew",
    "Dim",
    "FPTS",
    "FPTS/G",
    "SPTS",
    "SPTS/G",
    "P10",
    "P20",
    "P30",
    "P40",
    "P50",
    "P60",
    "P70",
    "P80",
    "P90",
    "TT10",
    "TT20",
    "TT30",
    "TT40",
    "TT50",
    "TT60",
    "TT70",
    "TT80",
    "TT90",
    "NameASCII",
    "PlayerId",
    "MLBAMID",
]

CSV_PITCHER_COLUMNS = [
    "Name",
    "Team",
    "W",
    "L",
    "QS",
    "ERA",
    "G",
    "GS",
    "SV",
    "HLD",
    "BS",
    "IP",
    "TBF",
    "H",
    "R",
    "ER",
    "HR",
    "BB",
    "IBB",
    "HBP",
    "SO",
    "K/9",
    "BB/9",
    "K/BB",
    "HR/9",
    "K%",
    "BB%",
    "K-BB%",
    "AVG",
    "WHIP",
    "BABIP",
    "LOB%",
    "GB%",
    "HR/FB",
    "FIP",
    "WAR",
    "RA9-WAR",
    "ADP",
    "InterSD",
    "InterSK",
    "IntraSD",
    "Vol",
    "Skew",
    "Dim",
    "FPTS",
    "FPTS/IP",
    "SPTS",
    "SPTS/IP",
    "P10",
    "P20",
    "P30",
    "P40",
    "P50",
    "P60",
    "P70",
    "P80",
    "P90",
    "TT10",
    "TT20",
    "TT30",
    "TT40",
    "TT50",
    "TT60",
    "TT70",
    "TT80",
    "TT90",
    "NameASCII",
    "PlayerId",
    "MLBAMID",
]

# ── API→CSV column renames ─────────────────────────────────────────────────

HITTER_RENAMES = {
    "PlayerName": "Name",
    "xMLBAMID": "MLBAMID",
    "playerids": "PlayerId",
    "FPTS_G": "FPTS/G",
    "SPTS_G": "SPTS/G",
    "BaseRunning": "BsR",
    "UZR": "Fld",
    "wBsR": "wSB",
}

PITCHER_RENAMES = {
    "PlayerName": "Name",
    "xMLBAMID": "MLBAMID",
    "playerids": "PlayerId",
    "FPTS_IP": "FPTS/IP",
    "SPTS_IP": "SPTS/IP",
}

PERCENTILE_RENAMES = {
    "q10": "P10",
    "q20": "P20",
    "q30": "P30",
    "q40": "P40",
    "q50": "P50",
    "q60": "P60",
    "q70": "P70",
    "q80": "P80",
    "q90": "P90",
    "tt_q10": "TT10",
    "tt_q20": "TT20",
    "tt_q30": "TT30",
    "tt_q40": "TT40",
    "tt_q50": "TT50",
    "tt_q60": "TT60",
    "tt_q70": "TT70",
    "tt_q80": "TT80",
    "tt_q90": "TT90",
}

# ── Output filenames ───────────────────────────────────────────────────────

# Files carry a _ros suffix to make the rest-of-season vintage explicit and to
# avoid colliding with any legacy preseason-frozen pulls on disk.
FILE_NAMES = {
    ("steamer", "bat"): "fangraphs-steamer-projections-hitters_ros.csv",
    ("steamer", "pit"): "fangraphs-steamer-projections-pitchers_ros.csv",
    ("atc", "bat"): "fangraphs-atc-projections-hitters_ros.csv",
    ("atc", "pit"): "fangraphs-atc-projections-pitchers_ros.csv",
}


def _strip_diacritics(text: str) -> str:
    """Convert accented characters to their ASCII equivalents."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _clean_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert MLBAMID and PlayerId to clean types matching FanGraphs CSV export.

    MLBAMID: nullable integer (no .0 suffix).
    PlayerId: string.
    """
    if "MLBAMID" in df.columns:
        df["MLBAMID"] = df["MLBAMID"].astype("Int64")
    if "PlayerId" in df.columns:
        df["PlayerId"] = df["PlayerId"].astype(str)
    return df


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
        f"If this is 403/Cloudflare, your Brave cookies may have expired — "
        f"visit fangraphs.com in Brave and retry."
    )

    data = resp.json()
    assert isinstance(data, list), (
        f"Expected list from API, got {type(data).__name__}. Preview: {str(data)[:200]}"
    )
    assert len(data) > 0, f"API returned 0 rows for {label}"

    print(f"    Got {len(data)} players")
    return data


def _build_hitter_csv(api_data: list[dict]) -> pd.DataFrame:
    """Convert hitter API JSON to CSV-format DataFrame.

    Maps API column names to FanGraphs CSV export column names,
    computes NameASCII, and orders columns to match the export format.
    """
    df = pd.DataFrame(api_data)

    all_renames = {**HITTER_RENAMES, **PERCENTILE_RENAMES}
    existing_renames = {k: v for k, v in all_renames.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    df["NameASCII"] = df["Name"].apply(_strip_diacritics)

    for col in CSV_HITTER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = _clean_id_columns(df)
    return df[CSV_HITTER_COLUMNS].copy()


def _build_pitcher_csv(api_data: list[dict]) -> pd.DataFrame:
    """Convert pitcher API JSON to CSV-format DataFrame.

    Maps API column names to FanGraphs CSV export column names,
    computes NameASCII, and orders columns to match the export format.
    """
    df = pd.DataFrame(api_data)

    all_renames = {**PITCHER_RENAMES, **PERCENTILE_RENAMES}
    existing_renames = {k: v for k, v in all_renames.items() if k in df.columns}
    df = df.rename(columns=existing_renames)

    df["NameASCII"] = df["Name"].apply(_strip_diacritics)

    for col in CSV_PITCHER_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df = _clean_id_columns(df)
    return df[CSV_PITCHER_COLUMNS].copy()


def scrape_projections(
    output_dir: Path | str | None = None,
    systems: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, Path]:
    """Scrape FanGraphs projections and save as CSVs.

    Args:
        output_dir: Directory to write CSVs. Defaults to data/pulled_YYYYMMDD/.
        systems: Which projection systems to scrape. Defaults to ["steamer", "atc"].
        dry_run: If True, print what would be done without writing files.

    Returns:
        Dict mapping file description to output path.
    """
    if systems is None:
        systems = ["steamer", "atc"]

    data_dir = Path(__file__).resolve().parent.parent / "data"
    if output_dir is None:
        today = date.today().strftime("%Y%m%d")
        output_dir = data_dir / f"pulled_{today}"
    else:
        output_dir = Path(output_dir)

    if dry_run:
        print(f"DRY RUN — would write to: {output_dir}")
        for system in systems:
            for stats in ("bat", "pit"):
                fname = FILE_NAMES[(system, stats)]
                print(f"  {fname}")
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    session = get_fangraphs_session()

    written: dict[str, Path] = {}

    for system in systems:
        print(f"\n{'=' * 50}")
        print(f"Projection system: {system.upper()}")
        print(f"{'=' * 50}")

        for stats_type, builder, label in [
            ("bat", _build_hitter_csv, "hitters"),
            ("pit", _build_pitcher_csv, "pitchers"),
        ]:
            api_data = fetch_projections(session, system, stats_type)
            df = builder(api_data)

            fname = FILE_NAMES[(system, stats_type)]
            out_path = output_dir / fname
            df.to_csv(out_path, index=False)

            print(f"    Wrote {len(df)} rows → {out_path.name}")
            written[f"{system}_{label}"] = out_path

    print(f"\nDone. {len(written)} files written to {output_dir}")
    return written


def main() -> None:
    """CLI entry point for scraping FanGraphs projections."""
    parser = argparse.ArgumentParser(
        description="Scrape FanGraphs projections (Steamer + ATC)"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Output directory (default: data/pulled_YYYYMMDD/)",
    )
    parser.add_argument(
        "--systems",
        nargs="+",
        default=["steamer", "atc"],
        choices=list(PROJECTION_TYPES.keys()),
        help="Projection systems to scrape (default: steamer atc)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    args = parser.parse_args()

    scrape_projections(
        output_dir=args.output_dir,
        systems=args.systems,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
