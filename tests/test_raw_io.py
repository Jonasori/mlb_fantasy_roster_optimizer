"""
Offline tests for the raw snapshot layer's season partitioning.
No network access. Per AGENTS.md: no classes, no fixtures, no mocking.

The point of these tests is the cross-season read. Before season partitioning,
`read_latest_raw("ytd", on_or_before=<early 2026>)` with a 2024 backfill on disk
returned a FULL 2024 SEASON as 2026 year-to-date: real data, wrong meaning, no
error. `test_two_seasons_stay_separate` is the regression guard for exactly that.
"""

import datetime

import pandas as pd
import pytest

from data_prep import raw_io
from data_prep.raw_io import (
    SEASONAL_SOURCES,
    available_dates,
    raw_path,
    read_latest_raw,
    season_dirs,
    snapshot_ages,
    write_raw,
)


def _redirect_raw(tmp_path, monkeypatch) -> None:
    """Point the raw layer at a temp dir so tests never touch data/raw."""
    monkeypatch.setattr(raw_io, "RAW_DIR", tmp_path / "raw")


def test_ytd_is_declared_seasonal():
    assert "ytd" in SEASONAL_SOURCES, (
        "ytd must stay in SEASONAL_SOURCES: fetch_stats_range accepts any "
        "season, so an unpartitioned backfill is indistinguishable from the "
        "current year-to-date. Removing it reopens the cross-season read bug."
    )
    assert "savant" in SEASONAL_SOURCES, (
        "savant leaderboards are per-season; keep it in SEASONAL_SOURCES."
    )


def test_raw_path_nests_by_season():
    flat = raw_path("fantrax", datetime.date(2026, 8, 21))
    seasonal = raw_path("ytd", datetime.date(2024, 10, 1), season=2024)
    assert flat.parent.name == "fantrax", (
        f"Non-seasonal source should stay flat, got {flat}"
    )
    assert seasonal.parent.name == "2024", (
        f"Seasonal source should nest under its season, got {seasonal}"
    )
    assert seasonal.name == "2024-10-01.parquet", (
        f"Filename should remain the ISO date, got {seasonal.name}"
    )


def test_two_seasons_stay_separate(tmp_path, monkeypatch):
    """The regression guard: a past-season backfill must not be readable as now.

    2024's snapshot is dated LATER in its own year than 2026's is in ours, so a
    flat layout ordered by date would let an `on_or_before` read in early 2026
    fall through to the 2024 file.
    """
    _redirect_raw(tmp_path, monkeypatch)

    old = pd.DataFrame({"MLBAMID": [1], "R": [95.0]})
    new = pd.DataFrame({"MLBAMID": [1], "R": [12.0]})
    write_raw(old, "ytd", datetime.date(2024, 10, 1), season=2024)
    write_raw(new, "ytd", datetime.date(2026, 4, 15), season=2026)

    frame_2024, date_2024 = read_latest_raw("ytd", season=2024)
    frame_2026, date_2026 = read_latest_raw("ytd", season=2026)

    assert date_2024 == datetime.date(2024, 10, 1), f"Got {date_2024}"
    assert date_2026 == datetime.date(2026, 4, 15), f"Got {date_2026}"
    assert float(frame_2024["R"].iloc[0]) == 95.0, (
        "2024 read returned the wrong season's rows — season partitioning is "
        "not isolating snapshots."
    )
    assert float(frame_2026["R"].iloc[0]) == 12.0, (
        "2026 read returned the wrong season's rows — this is the cross-season "
        "bug the partitioning exists to prevent."
    )

    # The dangerous query: 'latest on or before an early-2026 date'. Scoped to
    # 2026 it must find 2026's own snapshot and never fall back to 2024.
    _, scoped = read_latest_raw(
        "ytd", on_or_before=datetime.date(2026, 5, 1), season=2026
    )
    assert scoped == datetime.date(2026, 4, 15), (
        f"Scoped on_or_before read leaked across seasons, got {scoped}"
    )
    assert season_dirs("ytd") == [2024, 2026], (
        f"season_dirs should list both seasons, got {season_dirs('ytd')}"
    )
    assert available_dates("ytd", season=2024) == [datetime.date(2024, 10, 1)], (
        "available_dates leaked another season's dates into a scoped listing."
    )


def test_seasonal_source_without_season_asserts(tmp_path, monkeypatch):
    _redirect_raw(tmp_path, monkeypatch)
    frame = pd.DataFrame({"MLBAMID": [1], "R": [1.0]})

    with pytest.raises(AssertionError, match="SEASONAL_SOURCES"):
        write_raw(frame, "ytd", datetime.date(2026, 4, 15))
    with pytest.raises(AssertionError, match="SEASONAL_SOURCES"):
        available_dates("ytd")
    with pytest.raises(AssertionError, match="SEASONAL_SOURCES"):
        read_latest_raw("ytd")


def test_undeclared_source_with_season_dirs_asserts(tmp_path, monkeypatch):
    """Catches a source partitioned on disk but not yet declared seasonal.

    Without this guard, adding season dirs to a source and forgetting to list
    it in SEASONAL_SOURCES makes every flat read silently return zero rows.
    """
    _redirect_raw(tmp_path, monkeypatch)
    frame = pd.DataFrame({"MLBAMID": [1], "R": [1.0]})
    write_raw(frame, "standings", datetime.date(2025, 9, 1), season=2025)

    with pytest.raises(AssertionError, match="season directories"):
        available_dates("standings")


def test_snapshot_ages_survives_missing_seasonal_source(tmp_path, monkeypatch):
    """The status display must not crash on a seasonal source with no data."""
    _redirect_raw(tmp_path, monkeypatch)
    write_raw(
        pd.DataFrame({"MLBAMID": [1]}), "fantrax", datetime.date(2026, 8, 21)
    )
    ages = snapshot_ages(["fantrax", "ytd"])

    assert list(ages["source"]) == ["ytd", "fantrax"] or set(ages["source"]) == {
        "fantrax",
        "ytd",
    }, f"Expected both sources reported, got {list(ages['source'])}"
    ytd_row = ages[ages["source"] == "ytd"].iloc[0]
    assert ytd_row["n_snapshots"] == 0, (
        f"Empty seasonal source should report 0 snapshots, got "
        f"{ytd_row['n_snapshots']}"
    )
    assert pd.isna(ytd_row["latest"]), "Empty source should report NaT latest."


def test_snapshot_ages_reports_newest_season(tmp_path, monkeypatch):
    _redirect_raw(tmp_path, monkeypatch)
    frame = pd.DataFrame({"MLBAMID": [1]})
    write_raw(frame, "ytd", datetime.date(2024, 10, 1), season=2024)
    write_raw(frame, "ytd", datetime.date(2026, 4, 15), season=2026)

    row = snapshot_ages(["ytd"]).iloc[0]
    assert row["season"] == 2026, (
        f"snapshot_ages should report the newest season, got {row['season']}"
    )
    assert row["n_snapshots"] == 1, (
        f"Counts must be per-season, not pooled across seasons; got "
        f"{row['n_snapshots']}"
    )
