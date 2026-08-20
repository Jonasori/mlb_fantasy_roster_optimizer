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

_DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})\.parquet$")


def raw_path(source: str, date: datetime.date | None = None) -> Path:
    """Path for one source's snapshot on one date.

    Args:
        source: Source directory, slash-separated for nesting
            (e.g. "fantrax", "projections/steamer", "market/ottoneu").
        date: Snapshot date. Defaults to today.

    Returns:
        Path to `data/raw/<source>/<YYYY-MM-DD>.parquet` (may not exist).
    """
    if date is None:
        date = datetime.date.today()
    return RAW_DIR / source / f"{date.isoformat()}.parquet"


def write_raw(df: pd.DataFrame, source: str, date: datetime.date | None = None) -> Path:
    """Write a source's snapshot for one date, creating parent dirs.

    Overwrites same-day snapshots: re-fetching today replaces today, it does
    not accumulate duplicates.

    Args:
        df: Raw fetched frame. Written as-is — no joins, no renames.
        source: Source directory (see `raw_path`).
        date: Snapshot date. Defaults to today.

    Returns:
        The path written.
    """
    assert len(df) > 0, (
        f"write_raw: refusing to write an empty frame for source '{source}'. "
        f"An empty fetch means the upstream call failed — fix the fetcher "
        f"rather than persisting a snapshot that would silently poison the join."
    )
    path = raw_path(source, date)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"  wrote {len(df)} rows -> {path.relative_to(REPO_ROOT)}")
    return path


def available_dates(source: str) -> list[datetime.date]:
    """All snapshot dates for a source, oldest first. Empty if none exist."""
    directory = RAW_DIR / source
    if not directory.is_dir():
        return []
    dates = []
    for entry in directory.iterdir():
        match = _DATE_RE.match(entry.name)
        if match:
            dates.append(datetime.date.fromisoformat(match.group(1)))
    return sorted(dates)


def read_latest_raw(
    source: str, on_or_before: datetime.date | None = None
) -> tuple[pd.DataFrame, datetime.date]:
    """Read a source's most recent snapshot.

    Args:
        source: Source directory (see `raw_path`).
        on_or_before: Ignore snapshots newer than this date. Used to reproduce
            a past day's decisions from the snapshots that existed then.

    Returns:
        (frame, snapshot_date) — the date is returned so callers can report and
        assert on staleness instead of silently analyzing old data.
    """
    dates = available_dates(source)
    if on_or_before is not None:
        dates = [d for d in dates if d <= on_or_before]
    assert dates, (
        f"read_latest_raw: no snapshots for source '{source}' in "
        f"{(RAW_DIR / source)}"
        + (f" on or before {on_or_before}" if on_or_before else "")
        + ". Run the matching fetcher first (`uv run fetch --help`)."
    )
    date = dates[-1]
    return pd.read_parquet(raw_path(source, date)), date


def snapshot_ages(sources: list[str]) -> pd.DataFrame:
    """How stale is each source? One row per source, oldest first.

    Returns:
        DataFrame with columns: source, latest, days_old, n_snapshots.
        Sources with no snapshots get latest=NaT and days_old=NA.
    """
    today = datetime.date.today()
    rows = []
    for source in sources:
        dates = available_dates(source)
        latest = dates[-1] if dates else None
        rows.append(
            {
                "source": source,
                "latest": pd.Timestamp(latest) if latest else pd.NaT,
                "days_old": (today - latest).days if latest else pd.NA,
                "n_snapshots": len(dates),
            }
        )
    return pd.DataFrame(rows).sort_values("days_old", ascending=False, na_position="first")
