"""Season-to-date player stats from MLB StatsAPI's byDateRange leaderboard.

`byDateRange` is a league-wide leaderboard, not a per-player lookup: any date
window costs two requests (one per group) regardless of how many players are
involved. This is what makes arbitrary backtest split dates cheap.
"""

import datetime

import pandas as pd
import statsapi

from .raw_io import write_raw

# One row per player per group; 3000 comfortably exceeds the ~1450 real rows.
_ROW_LIMIT = 3000

# Raw stat key -> our column name. Keys absent from a payload become NaN.
_HITTING_FIELDS: dict[str, str] = {
    "plateAppearances": "PA", "atBats": "AB", "hits": "H", "homeRuns": "HR",
    "runs": "R", "rbi": "RBI", "stolenBases": "SB", "caughtStealing": "CS",
    "baseOnBalls": "BB", "strikeOuts": "SO", "sacFlies": "SF",
}
_PITCHING_FIELDS: dict[str, str] = {
    "battersFaced": "BF", "outs": "outs", "wins": "W", "saves": "SV",
    "earnedRuns": "ER", "hits": "HA", "baseOnBalls": "BBA", "strikeOuts": "SOA",
    "homeRuns": "HRA", "groundOuts": "groundOuts", "airOuts": "airOuts",
}
_RATE_FIELDS: tuple[str, ...] = ("avg", "obp", "slg", "babip")

# Below this, a hitting/pitching frame is almost certainly the ~140-player
# collapse caused by a dropped playerPool=ALL, not a real narrow slate.
_MIN_PLAUSIBLE_ROWS = 50


def _range_params(
    group: str, season: int, start: datetime.date, end: datetime.date
) -> dict:
    """Build the byDateRange query params for one group.

    Pulled out of `fetch_stats_range` so `playerPool=ALL` — whose omission
    silently returns ~140 players instead of ~1450 — is covered by an offline
    test instead of only living behind a comment.
    """
    return {
        "stats": "byDateRange",
        "group": group,
        "season": season,
        "gameType": "R",
        "sportId": 1,
        # MANDATORY: without playerPool=ALL this silently returns only
        # ~140 qualified players.
        "playerPool": "ALL",
        "limit": _ROW_LIMIT,
        "startDate": start.isoformat(),
        "endDate": end.isoformat(),
    }


def parse_rate(series: pd.Series) -> pd.Series:
    """Parse StatsAPI rate strings ('.319') to float, with '.---' as missing.

    StatsAPI returns rate stats as strings and uses '.---' as its missing
    sentinel. A plain astype(float) either raises or silently coerces.
    """
    text = series.astype("string")
    text = text.where(text != ".---")
    return pd.to_numeric(text, errors="coerce").astype(float)


def parse_stat_splits(payload: dict, group: str) -> pd.DataFrame:
    """Flatten one byDateRange response into a per-player frame.

    Args:
        payload: Raw JSON dict from the stats endpoint.
        group: "hitting" or "pitching" — selects the field map.

    Returns:
        One row per player. Adds columns: MLBAMID, name, group, plus the
        mapped counting fields, the parsed rate fields, and (pitching only)
        IP derived from `outs`.
    """
    assert group in ("hitting", "pitching"), (
        f"parse_stat_splits: group must be 'hitting' or 'pitching', got {group!r}."
    )
    fields = _HITTING_FIELDS if group == "hitting" else _PITCHING_FIELDS

    rows = []
    for stat_group in payload.get("stats", []):
        for split in stat_group.get("splits", []):
            stat = split["stat"]
            row = {
                "MLBAMID": int(split["player"]["id"]),
                "name": split["player"]["fullName"],
                "group": group,
            }
            for raw_key, col in fields.items():
                row[col] = stat.get(raw_key)
            for rate in _RATE_FIELDS:
                row[rate] = stat.get(rate)
            rows.append(row)

    assert rows, (
        f"parse_stat_splits: no {group} splits in payload. Check that "
        f"playerPool=ALL was sent — omitting it silently returns ~140 players, "
        f"and an empty window returns none at all."
    )
    frame = pd.DataFrame(rows)

    for col in fields.values():
        frame[col] = pd.to_numeric(frame[col], errors="coerce").astype(float)
    for rate in _RATE_FIELDS:
        frame[rate] = parse_rate(frame[rate])

    if group == "pitching":
        # inningsPitched is "76.1" meaning 76 1/3, NOT 76.1. Always use outs.
        frame["IP"] = frame["outs"] / 3.0

    assert not frame["MLBAMID"].duplicated().any(), (
        f"parse_stat_splits: duplicate MLBAMIDs in {group}. byDateRange should "
        f"aggregate traded players across teams; a duplicate means the response "
        f"is split by team."
    )
    return frame


def fetch_stats_range(
    season: int, start: datetime.date, end: datetime.date
) -> pd.DataFrame:
    """Fetch league-wide cumulative stats between two dates, both groups.

    Two requests total, independent of player count.

    Returns:
        Hitting and pitching rows stacked, distinguished by the `group` column.
    """
    assert start <= end, (
        f"fetch_stats_range: start {start} is after end {end}."
    )
    frames = []
    for group in ("hitting", "pitching"):
        payload = statsapi.get("stats", _range_params(group, season, start, end))
        frame = parse_stat_splits(payload, group)
        assert len(frame) < _ROW_LIMIT, (
            f"fetch_stats_range: {group} returned {len(frame)} rows, at the "
            f"{_ROW_LIMIT} limit — the response is probably truncated. Raise "
            f"_ROW_LIMIT."
        )
        assert len(frame) > _MIN_PLAUSIBLE_ROWS, (
            f"fetch_stats_range: {group} returned only {len(frame)} rows, "
            f"below the {_MIN_PLAUSIBLE_ROWS}-row floor. This is the signature "
            f"of the playerPool=ALL trap (a dropped playerPool silently "
            f"returns ~140 'qualified' players) — check _range_params."
        )
        print(f"  byDateRange {group} {start}..{end}: {len(frame)} players")
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)


def fetch_ytd_snapshot(
    season: int, through: datetime.date | None = None
) -> pd.DataFrame:
    """Fetch season-to-date stats and write them to the raw layer.

    Returns the frame; the snapshot lands at data/raw/ytd/<through>.parquet.
    """
    if through is None:
        through = datetime.date.today()
    print(f"=== Fetching YTD stats for {season} through {through} ===")
    frame = fetch_stats_range(season, datetime.date(season, 1, 1), through)
    path = write_raw(frame, "ytd", through)
    print(f"=== ytd snapshot written: {path} ({len(frame)} rows) ===")
    return frame
