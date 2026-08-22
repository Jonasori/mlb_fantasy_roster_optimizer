"""
The raw layer: dated parquet snapshots, one directory per source.

Storage is partitioned by REFRESH CADENCE AND AUTH, not by processing stage.
That is the whole point. The previous design baked projections, Fantrax rosters
and identity into one "silver" table, which forced every refresh to satisfy the
union of their requirements — a stale Fantrax cookie blocked getting fresh
projections, and the table could only ever hold one projection system. Both
problems disappear once each source lands independently:

    data/raw/projections/steamer/2026-08-20.parquet   daily,  browser cookies
    data/raw/projections/atc/2026-08-20.parquet       daily,  browser cookies
    data/raw/fantrax/2026-08-20.parquet               on txn, PASTED COOKIES
    data/raw/market/ottoneu/2026-08-20.parquet        daily,  no auth
    data/raw/identity/2026-08-20.parquet              ~never, no auth

Switching projection systems becomes "read a different directory". Refreshing
one source never touches another. `build_players` (build.py) then reads the
latest snapshot of each and produces the single wide table all analysis uses.

Filenames are ISO dates so lexical sort == chronological sort.

SEASON PARTITIONING. A few sources describe one specific SEASON rather than a
current state, and for those the snapshot date is not enough to identify the
data. `ytd` is the clear case: `statsapi_stats.fetch_stats_range` accepts any
season, so a 2024 backfill lands at `ytd/2024-10-01.parquet`. Ask for
`read_latest_raw("ytd", on_or_before=date(2026, 3, 1))` and you get a FULL 2024
SEASON returned as 2026 year-to-date — real data, wrong meaning, no error.

So sources in `SEASONAL_SOURCES` nest under a season:

    data/raw/ytd/2024/2024-10-01.parquet
    data/raw/ytd/2026/2026-08-21.parquet

`write_raw` refuses to write a seasonal source without a season, and
`available_dates` refuses to read one without a season once season directories
exist. Both failures are asserts, so the mixed-season read cannot happen
quietly.
"""

import datetime
import re
from pathlib import Path

import pandas as pd

# Repo root is the parent of this package.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

# The joined wide table produced by build_players.
PLAYERS_TABLE_PATH = DATA_DIR / "players.parquet"

# Sources whose contents are specific to one season, so the snapshot date alone
# does not identify them. These MUST be written and read with an explicit
# season. Everything else is current-state and stays flat.
SEASONAL_SOURCES: frozenset[str] = frozenset({"ytd", "savant"})

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")
_SEASON_RE = re.compile(r"^(19|20)\d{2}$")


def season_dirs(source: str) -> list[int]:
    """Seasons already on disk for a source, ascending. Empty if none.

    Used to detect a seasonal source being read without a season, which is the
    mixed-season bug this partitioning exists to prevent.
    """
    directory = RAW_DIR / source
    if not directory.is_dir():
        return []
    return sorted(
        int(entry.name)
        for entry in directory.iterdir()
        if entry.is_dir() and _SEASON_RE.match(entry.name)
    )


def _require_season(source: str, season: int | None, caller: str) -> None:
    """Assert that a seasonal source was given a season.

    Two triggers: the source is declared seasonal, or it already has season
    directories on disk (which catches a source added to SEASONAL_SOURCES after
    data was written flat).
    """
    if season is not None:
        return
    assert source not in SEASONAL_SOURCES, (
        f"{caller}: source '{source}' is in SEASONAL_SOURCES and needs an "
        f"explicit season=. Its contents describe one season, so a date alone "
        f"cannot identify them — a 2024 snapshot read as year-to-date would "
        f"return a full 2024 season with no error. Pass season=<YYYY>."
    )
    existing = season_dirs(source)
    assert not existing, (
        f"{caller}: source '{source}' already has season directories "
        f"{existing} on disk but no season= was passed. Reading it flat would "
        f"see none of them. Pass season=<YYYY>, or add '{source}' to "
        f"SEASONAL_SOURCES so this is enforced everywhere."
    )


def raw_path(
    source: str, date: datetime.date | None = None, season: int | None = None
) -> Path:
    """Path for one source's snapshot on one date.

    Args:
        source: Source directory, slash-separated for nesting
            (e.g. "fantrax", "projections/steamer", "market/ottoneu").
        date: Snapshot date. Defaults to today.
        season: Season the contents describe. Required for SEASONAL_SOURCES;
            must be None for everything else.

    Returns:
        Path to `data/raw/<source>/<YYYY-MM-DD>.parquet`, or
        `data/raw/<source>/<season>/<YYYY-MM-DD>.parquet` when season is given.
        May not exist.
    """
    if date is None:
        date = datetime.date.today()
    directory = RAW_DIR / source if season is None else RAW_DIR / source / str(season)
    return directory / f"{date.isoformat()}.parquet"


def write_raw(
    df: pd.DataFrame,
    source: str,
    date: datetime.date | None = None,
    season: int | None = None,
) -> Path:
    """Write a source's snapshot for one date, creating parent dirs.

    Overwrites same-day snapshots: re-fetching today replaces today, it does
    not accumulate duplicates.

    Args:
        df: Raw fetched frame. Written as-is — no joins, no renames.
        source: Source directory (see `raw_path`).
        date: Snapshot date. Defaults to today.
        season: Season the contents describe. Required for SEASONAL_SOURCES.

    Returns:
        The path written.
    """
    assert len(df) > 0, (
        f"write_raw: refusing to write an empty frame for source '{source}'. "
        f"An empty fetch means the upstream call failed — fix the fetcher "
        f"rather than persisting a snapshot that would silently poison the join."
    )
    _require_season(source, season, "write_raw")
    path = raw_path(source, date, season)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    # Shorten for display only when the path really is under the repo. An
    # unconditional relative_to() raises ValueError for anything outside it,
    # which turned a progress message into a hard failure whenever RAW_DIR
    # pointed elsewhere (tests, or a relocated data dir).
    display = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
    print(f"  wrote {len(df)} rows -> {display}")
    return path


def available_dates(
    source: str,
    on_or_before: datetime.date | None = None,
    season: int | None = None,
) -> list[datetime.date]:
    """All snapshot dates for a source, oldest first. Empty if none exist.

    Args:
        source: Source directory (see `raw_path`).
        on_or_before: Ignore snapshots newer than this date, so a caller asking
            "does this source have anything usable?" gets the same answer
            `read_latest_raw` will act on. Without it a caller reproducing a
            past day sees a source as present, then hard-fails reading it.
        season: Season to look in. Required for SEASONAL_SOURCES.
    """
    _require_season(source, season, "available_dates")
    directory = RAW_DIR / source if season is None else RAW_DIR / source / str(season)
    if not directory.is_dir():
        return []
    dates = []
    for entry in directory.iterdir():
        match = _DATE_RE.match(entry.name)
        if match:
            date = datetime.date.fromisoformat(match.group(1))
            if on_or_before is None or date <= on_or_before:
                dates.append(date)
    return sorted(dates)


def read_latest_raw(
    source: str,
    on_or_before: datetime.date | None = None,
    season: int | None = None,
) -> tuple[pd.DataFrame, datetime.date]:
    """Read a source's most recent snapshot.

    Args:
        source: Source directory (see `raw_path`).
        on_or_before: Ignore snapshots newer than this date. Used to reproduce
            a past day's decisions from the snapshots that existed then.
        season: Season to read. Required for SEASONAL_SOURCES — without it a
            cross-season read would return a different season's data silently.

    Returns:
        (frame, snapshot_date) — the date is returned so callers can report and
        assert on staleness instead of silently analyzing old data.
    """
    dates = available_dates(source, season=season)
    if on_or_before is not None:
        dates = [d for d in dates if d <= on_or_before]
    assert dates, (
        f"read_latest_raw: no snapshots for source '{source}'"
        + (f" season {season}" if season is not None else "")
        + f" in {raw_path(source, season=season).parent}"
        + (f" on or before {on_or_before}" if on_or_before else "")
        + ". Run the matching fetcher first (`uv run fetch --help`)."
    )
    date = dates[-1]
    return pd.read_parquet(raw_path(source, date, season)), date


def snapshot_ages(sources: list[str]) -> pd.DataFrame:
    """How stale is each source? One row per source, oldest first.

    Seasonal sources are reported for their NEWEST season on disk, and the
    season is named in the `season` column so a stale-season source is visible
    rather than looking merely stale.

    Returns:
        DataFrame with columns: source, season, latest, days_old, n_snapshots.
        Sources with no snapshots get latest=NaT and days_old=NA.
    """
    today = datetime.date.today()
    rows = []
    for source in sources:
        seasons = season_dirs(source)
        season = seasons[-1] if seasons else None
        # A seasonal source with no season directories yet has nothing to
        # report; calling available_dates would (correctly) assert on the
        # missing season and take the whole status display down with it.
        if season is None and source in SEASONAL_SOURCES:
            dates: list[datetime.date] = []
        else:
            dates = available_dates(source, season=season)
        latest = dates[-1] if dates else None
        rows.append(
            {
                "source": source,
                "season": season if season is not None else pd.NA,
                "latest": pd.Timestamp(latest) if latest else pd.NaT,
                "days_old": (today - latest).days if latest else pd.NA,
                "n_snapshots": len(dates),
            }
        )
    return pd.DataFrame(rows).sort_values("days_old", ascending=False, na_position="first")
