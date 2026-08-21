# In-Season Projection Conditioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct ATC's rest-of-season projections using decomposed in-season evidence, with a backtest harness that proves any correction beats unadjusted ATC in MEW units before it ships.

**Architecture:** Three layers. A new `ytd` raw source pulls league-wide season-to-date stats from MLB StatsAPI's `byDateRange` leaderboard (2 requests per window, any date range). A skill-decomposition module turns raw counts into K%/BB%/ISO/BABIP rather than composites. A backtest harness assembles (projection-at-D, evidence-through-D, actual-after-D) triples and scores candidate correctors in MEW units. Only then does a volume corrector get fitted and wired into `build_players` at its single seam.

**Tech Stack:** Python 3.11+, pandas, numpy, `MLB-StatsAPI`, `requests`, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-20-in-season-projection-conditioning-design.md`

## Global Constraints

Copied verbatim from `AGENTS.md` and the spec. **Every task's requirements implicitly include this section.**

- **No classes.** Module-level functions only; plain `dict`/`list`/`tuple`/`set`, pandas, numpy. No OOP anywhere.
- **No `try`/`except`. No fallback logic.** Crash immediately with a clear message.
- **Every `assert` carries a descriptive message** naming the actual value and the fix.
- **`players = players.copy()`** at the top of any function that adds columns.
- **Column names are API.** Docstrings state required and added columns.
- **`print()` for status at key stages**; `tqdm.auto` for loops over a few seconds.
- **Tests:** no classes, no fixtures, no mocking (`tests/test_market.py` is the reference style). Offline by default.
- **Package layout:** two packages only — `optimizer` (math) and `data_prep` (fetch + join). Do not add a third.
- **Scoring columns that must stay non-NaN:** `PA, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP`.
- **Opposite-type columns must stay exactly `0.0`** (`data_prep/build.py:236`) or MEW picks up a phantom ratio term.
- **Never scale a player to zero volume** — drops him from FV's ratio z-population (`optimizer/player_scoring.py:71`) and permanently benches IL players via `optimizer/players.py::get_startable_slots`.
- **Season end date: 2026-09-27.**
- **Part 2b (rate correction) is OUT OF SCOPE.** Do not implement it. It is gated on Part 0 results.

### Verified API traps — all are silent failures

| Trap | Consequence if ignored |
|---|---|
| `playerPool=ALL` omitted from `byDateRange` | Silently returns ~140 qualified players instead of ~1450 |
| `inningsPitched` is `"76.1"` meaning 76⅓ | Parsing as float gives 76.1 — use the `outs` field ÷ 3 |
| Rate stats return as strings, `".---"` = missing | `astype(float)` raises or coerces wrong |
| Savant leaderboards **ignore** `start_date`/`end_date` | Returns full-season numbers with no error |
| `statcast_search/csv` caps at exactly 25,000 rows | Silent truncation |
| Wayback 503 body has no `__NEXT_DATA__` | Parses as "no data" rather than raising — records false empties |
| Wayback CDX stores some URLs with literal `&amp;` | Substituting `&` returns a *different* capture |

---

## File Structure

| File | Responsibility |
|---|---|
| `data_prep/statsapi_stats.py` (new) | Fetch + parse `byDateRange` league-wide stats. Nothing else. |
| `data_prep/skills.py` (new) | Turn raw counting stats into skill rates. Pure, no I/O. |
| `data_prep/wayback.py` (new) | Harvest dated vendor RoS projections from the Wayback Machine. |
| `data_prep/volume_adjust.py` (new) | Fit and apply the RoS volume correction. |
| `optimizer/backtest.py` (new) | Assemble backtest triples, score in MEW units, run baselines. |
| `data_prep/cli.py` (modify) | Register the `ytd` source and command. |
| `data_prep/build.py` (modify, near line 717) | Call the volume corrector at the single seam. |
| `tests/test_statsapi_stats.py` (new) | Offline parsing tests. |
| `tests/test_skills.py` (new) | Skill-rate arithmetic. |
| `tests/test_backtest.py` (new) | MEW-unit scoring, baselines. |
| `tests/test_volume_adjust.py` (new) | Invariants — the ones nothing else enforces. |

---

## Task 1: StatsAPI `byDateRange` fetch and parse

**Files:**
- Create: `data_prep/statsapi_stats.py`
- Test: `tests/test_statsapi_stats.py`

**Interfaces:**
- Consumes: nothing (first task)
- Produces:
  - `parse_stat_splits(payload: dict, group: str) -> pd.DataFrame` — pure, testable offline
  - `fetch_stats_range(season: int, start: datetime.date, end: datetime.date) -> pd.DataFrame`
  - `parse_rate(series: pd.Series) -> pd.Series`
  - Columns produced: `MLBAMID, name, group, PA, AB, H, HR, R, RBI, SB, CS, BB, SO, SF, avg, obp, slg, BF, outs, IP, W, SV, ER, HA, BBA, SOA, groundOuts, airOuts, babip`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_statsapi_stats.py`:

```python
"""
Offline tests for the StatsAPI byDateRange parser.
No network access. Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import numpy as np
import pandas as pd

from data_prep.statsapi_stats import parse_rate, parse_stat_splits

_HITTING_PAYLOAD = {
    "stats": [
        {
            "splits": [
                {
                    "player": {"id": 663728, "fullName": "Cal Raleigh"},
                    "numTeams": 1,
                    "stat": {
                        "plateAppearances": 392, "atBats": 340, "hits": 53,
                        "homeRuns": 17, "runs": 42, "rbi": 51, "stolenBases": 1,
                        "caughtStealing": 0, "baseOnBalls": 45, "strikeOuts": 125,
                        "sacFlies": 4, "avg": ".156", "obp": ".273", "slg": ".296",
                        "babip": ".196",
                    },
                },
                {
                    "player": {"id": 1, "fullName": "No Rate Guy"},
                    "numTeams": 2,
                    "stat": {
                        "plateAppearances": 0, "atBats": 0, "hits": 0,
                        "homeRuns": 0, "runs": 0, "rbi": 0, "stolenBases": 0,
                        "caughtStealing": 0, "baseOnBalls": 0, "strikeOuts": 0,
                        "sacFlies": 0, "avg": ".---", "obp": ".---", "slg": ".---",
                        "babip": ".---",
                    },
                },
            ]
        }
    ]
}

_PITCHING_PAYLOAD = {
    "stats": [
        {
            "splits": [
                {
                    "player": {"id": 694973, "fullName": "Paul Skenes"},
                    "numTeams": 1,
                    "stat": {
                        "battersFaced": 500, "outs": 229, "inningsPitched": "76.1",
                        "wins": 8, "saves": 0, "earnedRuns": 20, "hits": 55,
                        "baseOnBalls": 20, "strikeOuts": 95, "homeRuns": 6,
                        "groundOuts": 70, "airOuts": 60, "babip": ".280",
                    },
                }
            ]
        }
    ]
}


def test_parse_rate_handles_string_and_sentinel():
    parsed = parse_rate(pd.Series([".319", ".---", None, ".000"]))
    assert parsed.iloc[0] == 0.319, f"'.319' parsed as {parsed.iloc[0]}, expected 0.319"
    assert np.isnan(parsed.iloc[1]), f"'.---' should be NaN, got {parsed.iloc[1]}"
    assert np.isnan(parsed.iloc[2]), f"None should be NaN, got {parsed.iloc[2]}"
    assert parsed.iloc[3] == 0.0, f"'.000' parsed as {parsed.iloc[3]}, expected 0.0"
    assert parsed.dtype == float, f"Expected float dtype, got {parsed.dtype}"


def test_parse_stat_splits_hitting():
    df = parse_stat_splits(_HITTING_PAYLOAD, "hitting")
    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"
    raleigh = df[df["MLBAMID"] == 663728].iloc[0]
    assert raleigh["PA"] == 392, f"PA parsed as {raleigh['PA']}, expected 392"
    assert raleigh["SO"] == 125, f"SO parsed as {raleigh['SO']}, expected 125"
    assert abs(raleigh["slg"] - 0.296) < 1e-9, (
        f"slg parsed as {raleigh['slg']}, expected 0.296"
    )
    assert raleigh["group"] == "hitting", f"group is {raleigh['group']}"
    assert np.isnan(df[df["MLBAMID"] == 1].iloc[0]["avg"]), (
        "'.---' avg should parse to NaN"
    )


def test_parse_stat_splits_pitching_uses_outs_not_innings_string():
    df = parse_stat_splits(_PITCHING_PAYLOAD, "pitching")
    row = df.iloc[0]
    assert row["outs"] == 229, f"outs parsed as {row['outs']}, expected 229"
    # 229/3 = 76.333..., NOT the 76.1 that a naive float() of "76.1" would give.
    assert abs(row["IP"] - 229 / 3) < 1e-9, (
        f"IP is {row['IP']}, expected {229 / 3}. "
        f"inningsPitched '76.1' means 76 1/3 — derive IP from outs."
    )
    assert abs(row["IP"] - 76.1) > 0.2, (
        f"IP is {row['IP']}, which looks like a naive float('76.1'). Use outs/3."
    )
    assert row["BF"] == 500, f"BF parsed as {row['BF']}, expected 500"


def test_parse_stat_splits_no_duplicate_players():
    df = parse_stat_splits(_HITTING_PAYLOAD, "hitting")
    assert not df["MLBAMID"].duplicated().any(), (
        "byDateRange aggregates traded players across teams; duplicate MLBAMIDs "
        "mean the parser is splitting on team."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_statsapi_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_prep.statsapi_stats'`

- [ ] **Step 3: Write the implementation**

Create `data_prep/statsapi_stats.py`:

```python
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
        payload = statsapi.get(
            "stats",
            {
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
            },
        )
        frame = parse_stat_splits(payload, group)
        assert len(frame) < _ROW_LIMIT, (
            f"fetch_stats_range: {group} returned {len(frame)} rows, at the "
            f"{_ROW_LIMIT} limit — the response is probably truncated. Raise "
            f"_ROW_LIMIT."
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_statsapi_stats.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Verify against the live API**

Run:

```bash
uv run python -c "
import datetime
from data_prep.statsapi_stats import fetch_stats_range
df = fetch_stats_range(2026, datetime.date(2026,1,1), datetime.date(2026,6,11))
h = df[df.group=='hitting']; p = df[df.group=='pitching']
print('hitters', len(h), 'pitchers', len(p))
print(h.nlargest(3,'PA')[['name','PA','HR','SO']].to_string(index=False))
print(p.nlargest(3,'IP')[['name','IP','SOA','BBA']].to_string(index=False))
"
```

Expected: 900+ hitters and 700+ pitchers (NOT ~140 — that means `playerPool` was dropped). Top-IP pitchers should show IP with `.333`/`.667` fractions, never `.1`/`.2`.

- [ ] **Step 6: Commit**

```bash
git add data_prep/statsapi_stats.py tests/test_statsapi_stats.py
git commit -m "feat: fetch league-wide season-to-date stats via byDateRange

Two requests per date window regardless of player count, which is what
makes arbitrary backtest split dates cheap.

Guards the three silent-failure traps: playerPool=ALL is mandatory (else
~140 players), inningsPitched '76.1' means 76 1/3 so IP comes from outs/3,
and rate stats are strings with '.---' as the missing sentinel."
```

---

## Task 2: Register `ytd` as a fetchable source

**Files:**
- Modify: `data_prep/cli.py` (SOURCES list ~line 20, command choices ~line 98, dispatch ~line 112)
- Test: `tests/test_statsapi_stats.py` (append)

**Interfaces:**
- Consumes: `fetch_ytd_snapshot` from Task 1
- Produces: `uv run fetch ytd`; `"ytd"` present in `data_prep.cli.SOURCES`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_statsapi_stats.py`:

```python
def test_ytd_registered_as_source():
    from data_prep.cli import SOURCES
    from data_prep.raw_io import RAW_DIR, raw_path

    assert "ytd" in SOURCES, (
        f"'ytd' missing from cli.SOURCES ({SOURCES}); `uv run fetch status` "
        f"will not report its staleness."
    )
    path = raw_path("ytd", datetime.date(2026, 8, 20))
    assert path == RAW_DIR / "ytd" / "2026-08-20.parquet", (
        f"ytd snapshots must land at data/raw/ytd/<date>.parquet, got {path}"
    )
```

Add `import datetime` to the test file's imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_statsapi_stats.py::test_ytd_registered_as_source -v`
Expected: FAIL — `'ytd' missing from cli.SOURCES`

- [ ] **Step 3: Wire it into the CLI**

In `data_prep/cli.py`, add `"ytd"` to `SOURCES`:

```python
SOURCES: list[str] = [
    "projections/steamer",
    "projections/atc",
    "fantrax",
    "standings",
    "identity",
    "ytd",
    "market/ottoneu",
    "market/adp",
    "market/espn",
    "market/hkb",
]
```

Add the command function next to the other `cmd_*` functions:

```python
def cmd_ytd() -> None:
    """Fetch season-to-date player stats from MLB StatsAPI."""
    import datetime

    from .statsapi_stats import fetch_ytd_snapshot

    fetch_ytd_snapshot(datetime.date.today().year)
```

Add `"ytd"` to the `choices` list of the `command` argument, then add the dispatch branch and include it in `all`:

```python
    elif args.command == "ytd":
        cmd_ytd()
```

```python
    elif args.command == "all":
        cmd_projections()
        cmd_fantrax()
        cmd_market()
        cmd_identity()
        cmd_ytd()
        cmd_build(args.system)
```

Update the module docstring's command list to include `uv run fetch ytd  # no auth`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_statsapi_stats.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Fetch a real snapshot**

Run: `uv run fetch ytd && uv run fetch status`

Expected: `data/raw/ytd/2026-08-20.parquet` exists and `ytd` appears in the status table with age 0 days.

- [ ] **Step 6: Commit**

```bash
git add data_prep/cli.py tests/test_statsapi_stats.py
git commit -m "feat: register ytd as a fetchable source

uv run fetch ytd, included in 'all' and in status staleness reporting."
```

---

## Task 3: Skill decomposition

**Files:**
- Create: `data_prep/skills.py`
- Test: `tests/test_skills.py`

**Interfaces:**
- Consumes: the column set produced by `parse_stat_splits` (Task 1)
- Produces: `add_skill_rates(stats: pd.DataFrame) -> pd.DataFrame`
  - Hitting columns added: `K_pct, BB_pct, ISO, BABIP, SBA_rate, n_PA`
  - Pitching columns added: `K_pct, BB_pct, GB_pct, HRFB, BABIP_against, n_BF`

**Why this exists:** the spec's §2.5 finding is that Raleigh's `.569` and Duran's `.622` look identical as OPS and decompose completely differently. Every downstream consumer must see components, never composites.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_skills.py`:

```python
"""
Offline tests for skill-rate decomposition.
Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import numpy as np
import pandas as pd

from data_prep.skills import add_skill_rates

def _hitting_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2],
            "name": ["Full Season", "Zero PA"],
            "group": ["hitting", "hitting"],
            "PA": [400.0, 0.0], "AB": [360.0, 0.0], "H": [90.0, 0.0],
            "HR": [20.0, 0.0], "R": [50.0, 0.0], "RBI": [60.0, 0.0],
            "SB": [8.0, 0.0], "CS": [2.0, 0.0], "BB": [36.0, 0.0],
            "SO": [100.0, 0.0], "SF": [4.0, 0.0],
            "avg": [0.250, np.nan], "obp": [0.320, np.nan],
            "slg": [0.450, np.nan], "babip": [0.280, np.nan],
        }
    )


def _pitching_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [3],
            "name": ["Starter"],
            "group": ["pitching"],
            "BF": [500.0], "outs": [300.0], "IP": [100.0], "W": [8.0],
            "SV": [0.0], "ER": [40.0], "HA": [90.0], "BBA": [30.0],
            "SOA": [125.0], "HRA": [12.0],
            "groundOuts": [120.0], "airOuts": [80.0],
            "avg": [np.nan], "obp": [np.nan], "slg": [np.nan],
            "babip": [0.290],
        }
    )


def test_hitting_skill_rates():
    out = add_skill_rates(_hitting_frame())
    row = out[out["MLBAMID"] == 1].iloc[0]
    assert abs(row["K_pct"] - 100.0 / 400.0) < 1e-9, (
        f"K_pct is {row['K_pct']}, expected {100 / 400}"
    )
    assert abs(row["BB_pct"] - 36.0 / 400.0) < 1e-9, (
        f"BB_pct is {row['BB_pct']}, expected {36 / 400}"
    )
    assert abs(row["ISO"] - (0.450 - 0.250)) < 1e-9, (
        f"ISO is {row['ISO']}, expected slg - avg = 0.200"
    )
    assert abs(row["SBA_rate"] - 10.0 / 400.0) < 1e-9, (
        f"SBA_rate is {row['SBA_rate']}, expected (SB+CS)/PA = {10 / 400}"
    )
    assert row["n_PA"] == 400.0, f"n_PA is {row['n_PA']}, expected 400"


def test_zero_volume_gives_nan_not_zero():
    out = add_skill_rates(_hitting_frame())
    row = out[out["MLBAMID"] == 2].iloc[0]
    for col in ("K_pct", "BB_pct", "SBA_rate"):
        assert np.isnan(row[col]), (
            f"{col} is {row[col]} for a 0-PA player; must be NaN. A zero rate "
            f"reads as 'elite contact' to any downstream shrinkage."
        )
    assert row["n_PA"] == 0.0, f"n_PA is {row['n_PA']}, expected 0"


def test_pitching_skill_rates():
    out = add_skill_rates(_pitching_frame())
    row = out.iloc[0]
    assert abs(row["K_pct"] - 125.0 / 500.0) < 1e-9, (
        f"K_pct is {row['K_pct']}, expected {125 / 500}"
    )
    assert abs(row["BB_pct"] - 30.0 / 500.0) < 1e-9, (
        f"BB_pct is {row['BB_pct']}, expected {30 / 500}"
    )
    assert abs(row["GB_pct"] - 120.0 / 200.0) < 1e-9, (
        f"GB_pct is {row['GB_pct']}, expected groundOuts/(groundOuts+airOuts)"
    )
    assert abs(row["HRFB"] - 12.0 / 80.0) < 1e-9, (
        f"HRFB is {row['HRFB']}, expected HRA/airOuts"
    )
    assert row["n_BF"] == 500.0, f"n_BF is {row['n_BF']}, expected 500"


def test_does_not_mutate_input():
    frame = _hitting_frame()
    before = list(frame.columns)
    add_skill_rates(frame)
    assert list(frame.columns) == before, (
        "add_skill_rates mutated its input; it must copy first (AGENTS.md)."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_skills.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_prep.skills'`

- [ ] **Step 3: Write the implementation**

Create `data_prep/skills.py`:

```python
"""Decompose raw counting stats into skill rates.

The spec's §2.5 finding is that two players with near-identical OPS can
decompose completely differently — one a real skill collapse, one batted-ball
luck — and that the stabilization constants differ by an order of magnitude
between components (K% M~49 PA vs BABIP M~433). Every consumer downstream of
here must see components, never composites.

GB_pct and HRFB are approximations: StatsAPI exposes groundOuts and airOuts,
not true batted-ball classifications, so these are out-rate proxies rather
than the FanGraphs definitions. They are directionally right and internally
consistent, which is what shrinkage needs; do not compare them to published
GB%/HR-FB values.
"""

import numpy as np
import pandas as pd

HITTING_SKILLS: tuple[str, ...] = ("K_pct", "BB_pct", "ISO", "BABIP", "SBA_rate")
PITCHING_SKILLS: tuple[str, ...] = (
    "K_pct", "BB_pct", "GB_pct", "HRFB", "BABIP_against",
)


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Elementwise ratio, NaN where the denominator is zero or missing.

    A zero denominator must NOT yield zero: a 0-PA player with K_pct=0 reads
    as elite contact to any shrinkage that weights by reliability.
    """
    denom = denominator.where(denominator > 0)
    return (numerator / denom).astype(float)


def add_skill_rates(stats: pd.DataFrame) -> pd.DataFrame:
    """Add per-skill rate columns to a parsed stats frame.

    Requires columns: group, and for hitting rows PA/AB/H/HR/SB/CS/BB/SO/slg/avg/babip;
    for pitching rows BF/SOA/BBA/HRA/groundOuts/airOuts/babip.

    Adds columns: K_pct, BB_pct, ISO, BABIP, SBA_rate, n_PA (hitting rows);
    K_pct, BB_pct, GB_pct, HRFB, BABIP_against, n_BF (pitching rows).
    Rows of the other group get NaN in that group's columns.
    """
    stats = stats.copy()
    assert "group" in stats.columns, (
        "add_skill_rates: frame has no 'group' column. Pass the output of "
        "parse_stat_splits, which tags every row 'hitting' or 'pitching'."
    )

    is_hit = stats["group"] == "hitting"
    is_pit = stats["group"] == "pitching"
    assert (is_hit | is_pit).all(), (
        f"add_skill_rates: unexpected group values "
        f"{sorted(set(stats.loc[~(is_hit | is_pit), 'group']))}."
    )

    for col in (*HITTING_SKILLS, *PITCHING_SKILLS, "n_PA", "n_BF"):
        stats[col] = np.nan

    if is_hit.any():
        h = stats.loc[is_hit]
        stats.loc[is_hit, "K_pct"] = _safe_ratio(h["SO"], h["PA"])
        stats.loc[is_hit, "BB_pct"] = _safe_ratio(h["BB"], h["PA"])
        stats.loc[is_hit, "ISO"] = (h["slg"] - h["avg"]).astype(float)
        stats.loc[is_hit, "BABIP"] = h["babip"].astype(float)
        stats.loc[is_hit, "SBA_rate"] = _safe_ratio(h["SB"] + h["CS"], h["PA"])
        stats.loc[is_hit, "n_PA"] = h["PA"].astype(float)

    if is_pit.any():
        p = stats.loc[is_pit]
        stats.loc[is_pit, "K_pct"] = _safe_ratio(p["SOA"], p["BF"])
        stats.loc[is_pit, "BB_pct"] = _safe_ratio(p["BBA"], p["BF"])
        stats.loc[is_pit, "GB_pct"] = _safe_ratio(
            p["groundOuts"], p["groundOuts"] + p["airOuts"]
        )
        stats.loc[is_pit, "HRFB"] = _safe_ratio(p["HRA"], p["airOuts"])
        stats.loc[is_pit, "BABIP_against"] = p["babip"].astype(float)
        stats.loc[is_pit, "n_BF"] = p["BF"].astype(float)

    n_hit = int(is_hit.sum())
    n_pit = int(is_pit.sum())
    print(f"skill rates: {n_hit} hitting rows, {n_pit} pitching rows")
    return stats
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_skills.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Sanity-check against the real snapshot**

Run:

```bash
uv run python -c "
import pandas as pd
from data_prep.raw_io import read_latest_raw
from data_prep.skills import add_skill_rates
ytd, d = read_latest_raw('ytd')
s = add_skill_rates(ytd)
h = s[(s.group=='hitting') & (s.n_PA>=250)]
print('n =', len(h))
print(h[['K_pct','BB_pct','ISO','BABIP']].describe().round(3).to_string())
print()
print(h[h.name.isin(['Cal Raleigh','Jarren Duran','Caleb Durbin'])]
      [['name','n_PA','K_pct','BB_pct','ISO','BABIP']].round(3).to_string(index=False))
"
```

Expected, from spec §2.5: Raleigh `K_pct ≈ 0.319`, `ISO ≈ 0.140`, `BABIP ≈ 0.196`. Duran `BABIP ≈ 0.253`. Pool medians should be roughly K% 0.22, BB% 0.08, ISO 0.16, BABIP 0.29.

- [ ] **Step 6: Commit**

```bash
git add data_prep/skills.py tests/test_skills.py
git commit -m "feat: decompose in-season stats into skill rates

Components, never composites: the spec's §2.5 finding is that Raleigh's
.569 OPS and Duran's .622 decompose completely differently, and their
stabilization constants differ ~9x (K% M~49 PA vs BABIP M~433).

Zero-volume players get NaN, not 0.0 — a 0-PA player with K_pct=0 would
read as elite contact to any reliability-weighted shrinkage."
```

---

## Task 4: Backtest triple assembly

**Files:**
- Create: `optimizer/backtest.py`
- Test: `tests/test_backtest.py`

**Interfaces:**
- Consumes: `fetch_stats_range` (Task 1), `add_skill_rates` (Task 3)
- Produces:
  - `SEASON_END: dict[int, datetime.date]`
  - `assemble_backtest_frame(season, split, projection, group="hitting") -> pd.DataFrame`
  - Columns produced: `MLBAMID, name, proj_<stat>, evid_<skill>, n_evid, actual_<stat>`

**Why the projection is a parameter:** Task 7 harvests more dated projections. This function must not care where one came from.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_backtest.py`:

```python
"""
Offline tests for the backtest harness.
Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import datetime

import numpy as np
import pandas as pd

from optimizer.backtest import SEASON_END, assemble_backtest_frame


def _projection() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2, 3],
            "Name": ["Kept", "Also Kept", "No Evidence"],
            "player_type": ["hitter", "hitter", "hitter"],
            "PA": [200.0, 150.0, 100.0],
            "R": [25.0, 18.0, 12.0], "HR": [8.0, 5.0, 3.0],
            "RBI": [26.0, 19.0, 13.0], "SB": [4.0, 2.0, 1.0],
            "OPS": [0.780, 0.700, 0.650],
        }
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2], "name": ["Kept", "Also Kept"],
            "group": ["hitting", "hitting"],
            "PA": [300.0, 280.0], "AB": [270.0, 250.0], "H": [70.0, 60.0],
            "HR": [12.0, 7.0], "R": [40.0, 33.0], "RBI": [41.0, 30.0],
            "SB": [6.0, 3.0], "CS": [2.0, 1.0], "BB": [27.0, 25.0],
            "SO": [70.0, 62.0], "SF": [3.0, 3.0],
            "avg": [0.259, 0.240], "obp": [0.330, 0.318],
            "slg": [0.440, 0.390], "babip": [0.300, 0.285],
        }
    )


def _actual() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2], "name": ["Kept", "Also Kept"],
            "group": ["hitting", "hitting"],
            "PA": [190.0, 140.0], "AB": [170.0, 125.0], "H": [45.0, 30.0],
            "HR": [9.0, 4.0], "R": [27.0, 16.0], "RBI": [28.0, 17.0],
            "SB": [5.0, 1.0], "CS": [1.0, 1.0], "BB": [18.0, 13.0],
            "SO": [44.0, 33.0], "SF": [2.0, 2.0],
            "avg": [0.265, 0.240], "obp": [0.335, 0.312],
            "slg": [0.455, 0.376], "babip": [0.305, 0.280],
        }
    )


def test_season_end_dates_known():
    assert SEASON_END[2026] == datetime.date(2026, 9, 27), (
        f"2026 season end is {SEASON_END.get(2026)}, expected 2026-09-27"
    )


def test_assemble_joins_and_prefixes():
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    assert len(frame) == 2, (
        f"Expected 2 rows (inner join on players with both evidence and "
        f"outcome), got {len(frame)}"
    )
    assert 3 not in set(frame["MLBAMID"]), (
        "Player 3 has a projection but no evidence and no outcome; he must "
        "not appear — scoring him would credit the model for a phantom."
    )
    for col in ("proj_PA", "proj_OPS", "evid_K_pct", "n_evid", "actual_PA"):
        assert col in frame.columns, f"Missing column {col}: {list(frame.columns)}"

    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert kept["proj_PA"] == 200.0, f"proj_PA is {kept['proj_PA']}, expected 200"
    assert kept["actual_PA"] == 190.0, f"actual_PA is {kept['actual_PA']}, expected 190"
    assert kept["n_evid"] == 300.0, f"n_evid is {kept['n_evid']}, expected 300 PA"
    assert abs(kept["evid_K_pct"] - 70.0 / 300.0) < 1e-9, (
        f"evid_K_pct is {kept['evid_K_pct']}, expected {70 / 300}"
    )


def test_actual_ops_is_derived_not_copied():
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert abs(kept["actual_OPS"] - (0.335 + 0.455)) < 1e-9, (
        f"actual_OPS is {kept['actual_OPS']}, expected obp+slg = 0.790"
    )


def test_rejects_split_outside_season():
    try:
        assemble_backtest_frame(
            2026, datetime.date(2026, 12, 1), _projection(),
            evidence=_evidence(), actual=_actual(),
        )
    except AssertionError as exc:
        assert "split" in str(exc).lower(), (
            f"Expected an assertion about the split date, got: {exc}"
        )
    else:
        raise AssertionError(
            "A split date after the season end must fail loudly — it silently "
            "produces an empty outcome window otherwise."
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'optimizer.backtest'`

- [ ] **Step 3: Write the implementation**

Create `optimizer/backtest.py`:

```python
"""Backtest harness for projection corrections.

Assembles (projection-at-D, evidence-through-D, actual-after-D) triples and
scores candidate correctors. Nothing in this module ships to the optimizer;
it exists so that no correction reaches production on a tuned constant.

Per the spec §3.1, the evidence side is cheap: byDateRange is a league-wide
leaderboard, so any split date costs two requests.
"""

import datetime

import numpy as np
import pandas as pd

from data_prep.skills import HITTING_SKILLS, PITCHING_SKILLS, add_skill_rates
from data_prep.statsapi_stats import fetch_stats_range

# Regular-season end dates. Used to bound the outcome window.
SEASON_END: dict[int, datetime.date] = {
    2021: datetime.date(2021, 10, 3),
    2022: datetime.date(2022, 10, 5),
    2023: datetime.date(2023, 10, 1),
    2024: datetime.date(2024, 9, 29),
    2025: datetime.date(2025, 9, 28),
    2026: datetime.date(2026, 9, 27),
}

# Scoring columns, by side. These mirror the optimizer's category set.
HITTING_STATS: tuple[str, ...] = ("PA", "R", "HR", "RBI", "SB", "OPS")
PITCHING_STATS: tuple[str, ...] = ("IP", "W", "SV", "K", "ERA", "WHIP")


def _derive_outcome_stats(actual: pd.DataFrame, group: str) -> pd.DataFrame:
    """Compute the optimizer's scoring categories from raw outcome counts."""
    out = actual.copy()
    if group == "hitting":
        out["OPS"] = (out["obp"] + out["slg"]).astype(float)
        return out[["MLBAMID", *HITTING_STATS]]

    innings = out["IP"].where(out["IP"] > 0)
    out["ERA"] = (out["ER"] * 9.0 / innings).astype(float)
    out["WHIP"] = ((out["HA"] + out["BBA"]) / innings).astype(float)
    out["K"] = out["SOA"].astype(float)
    return out[["MLBAMID", *PITCHING_STATS]]


def assemble_backtest_frame(
    season: int,
    split: datetime.date,
    projection: pd.DataFrame,
    group: str = "hitting",
    evidence: pd.DataFrame | None = None,
    actual: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build one (projection, evidence, outcome) triple for a split date.

    Args:
        season: Season year.
        split: The cut date. Evidence is season-start..split; the outcome
            window is split+1..season end.
        projection: A dated rest-of-season projection as of `split`. Must
            carry MLBAMID plus the scoring columns for `group`.
        group: "hitting" or "pitching".
        evidence: Pre-fetched evidence frame. Fetched from StatsAPI if None.
        actual: Pre-fetched outcome frame. Fetched from StatsAPI if None.

    Returns:
        One row per player present in all three sources. Adds columns
        prefixed `proj_`, `evid_`, `actual_`, plus `n_evid` (the evidence
        sample size, PA for hitters and BF for pitchers).
    """
    assert group in ("hitting", "pitching"), (
        f"assemble_backtest_frame: group must be 'hitting' or 'pitching', got {group!r}."
    )
    assert season in SEASON_END, (
        f"assemble_backtest_frame: no season end date recorded for {season}. "
        f"Known: {sorted(SEASON_END)}. Add it to SEASON_END."
    )
    end = SEASON_END[season]
    start = datetime.date(season, 1, 1)
    assert start < split < end, (
        f"assemble_backtest_frame: split date {split} must fall strictly "
        f"inside the {season} season ({start}..{end}). Outside it, one of the "
        f"two windows is empty and the backtest silently scores nothing."
    )

    if evidence is None:
        evidence = fetch_stats_range(season, start, split)
    if actual is None:
        actual = fetch_stats_range(
            season, split + datetime.timedelta(days=1), end
        )

    evidence = add_skill_rates(evidence[evidence["group"] == group])
    actual_raw = actual[actual["group"] == group].copy()
    if group == "pitching" and "IP" not in actual_raw.columns:
        actual_raw["IP"] = actual_raw["outs"] / 3.0

    skills = HITTING_SKILLS if group == "hitting" else PITCHING_SKILLS
    n_col = "n_PA" if group == "hitting" else "n_BF"
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS

    evid = evidence[["MLBAMID", *skills, n_col]].rename(
        columns={s: f"evid_{s}" for s in skills} | {n_col: "n_evid"}
    )
    if group == "hitting":
        # The volume corrector's slump term needs the observed composite, even
        # though every *rate* consumer downstream must use the components.
        evid["evid_OPS"] = (
            evidence["obp"].astype(float) + evidence["slg"].astype(float)
        ).values
        # Raw observed counting stats, so the raw_ytd baseline can project them
        # onto the projection's volume. Without these it silently degenerates
        # into the ATC baseline with one column swapped.
        for stat in ("R", "HR", "RBI", "SB"):
            evid[f"evid_{stat}"] = evidence[stat].astype(float).values
    proj = projection[["MLBAMID", *stats]].rename(
        columns={s: f"proj_{s}" for s in stats}
    )
    out = _derive_outcome_stats(actual_raw, group).rename(
        columns={s: f"actual_{s}" for s in stats}
    )

    name_col = "Name" if "Name" in projection.columns else "name"
    names = projection[["MLBAMID", name_col]].rename(columns={name_col: "name"})

    frame = (
        proj.merge(evid, on="MLBAMID", how="inner")
        .merge(out, on="MLBAMID", how="inner")
        .merge(names, on="MLBAMID", how="left")
    )
    frame = frame[~frame["MLBAMID"].duplicated()].reset_index(drop=True)

    assert len(frame) > 0, (
        f"assemble_backtest_frame: no players survived the join for {season} "
        f"@ {split} ({group}). Check that the projection carries MLBAMID and "
        f"that it is not the opposite player type."
    )
    print(
        f"backtest frame {season} @ {split} ({group}): {len(frame)} players "
        f"(projection {len(proj)}, evidence {len(evid)}, outcome {len(out)})"
    )
    return frame
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Assemble the real 2026-06-11 frame**

Run:

```bash
uv run python -c "
import datetime, pandas as pd
from optimizer.backtest import assemble_backtest_frame
proj = pd.read_csv('data_prep/data/pulled_20260611/fangraphs-atc-projections-hitters_ros.csv')
proj = proj[proj.MLBAMID.notna()]
proj['MLBAMID'] = proj.MLBAMID.astype(int)
f = assemble_backtest_frame(2026, datetime.date(2026,6,11), proj, 'hitting')
print(f.shape)
print(f[f.name.isin(['Cal Raleigh','Jarren Duran','Caleb Durbin'])]
      [['name','proj_PA','proj_OPS','n_evid','evid_K_pct','evid_ISO','actual_PA','actual_OPS']]
      .round(3).to_string(index=False))
"
```

Expected: 400+ rows. Raleigh should show a high `evid_K_pct` and low `evid_ISO`, and his `actual_OPS` is the number that decides whether ATC's optimism was justified.

- [ ] **Step 6: Commit**

```bash
git add optimizer/backtest.py tests/test_backtest.py
git commit -m "feat: assemble backtest triples for a split date

(projection-at-D, evidence-through-D, actual-after-D), inner-joined so a
player with a projection but no outcome cannot silently score.

Split dates outside the season assert rather than returning an empty
outcome window."
```

---

## Task 5: MEW-unit scoring and baselines

**Files:**
- Modify: `optimizer/backtest.py`
- Test: `tests/test_backtest.py` (append)

**Interfaces:**
- Consumes: `assemble_backtest_frame` (Task 4); `optimizer.player_scoring.add_mew`
- Produces:
  - `score_in_mew(frame, my_totals, gradient, group="hitting") -> pd.DataFrame`
  - `BASELINES: dict[str, callable]` with keys `"atc"`, `"raw_ytd"`, `"flat_volume"`
  - `run_baselines(frame, my_totals, gradient, group="hitting") -> pd.DataFrame`

**Why this is the load-bearing task:** the spec's §3.1 decision rule is that error is judged in MEW units, not stat units. A method that cuts OPS RMSE 20% but moves no decision has earned nothing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest.py`:

```python
from optimizer.backtest import BASELINES, run_baselines, score_in_mew

_TOTALS = {
    "PA": 7000.0, "IP": 1000.0, "R": 897.0, "HR": 240.0, "RBI": 764.0,
    "SB": 163.0, "OPS": 0.7356, "W": 67.0, "SV": 65.0, "K": 1014.0,
    "ERA": 3.7553, "WHIP": 1.1300,
}
_GRADIENT = {
    "R": 0.02284, "HR": 0.01964, "RBI": 0.00101, "SB": 0.07693,
    "OPS": 0.39234, "W": 0.26798, "SV": 0.06077, "K": 0.00660,
    "ERA": -4.58521, "WHIP": -28.62424,
}


def _scored_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2],
            "name": ["A", "B"],
            "proj_PA": [200.0, 150.0], "proj_R": [25.0, 18.0],
            "proj_HR": [8.0, 5.0], "proj_RBI": [26.0, 19.0],
            "proj_SB": [4.0, 2.0], "proj_OPS": [0.780, 0.700],
            "actual_PA": [190.0, 140.0], "actual_R": [27.0, 16.0],
            "actual_HR": [9.0, 4.0], "actual_RBI": [28.0, 17.0],
            "actual_SB": [5.0, 1.0], "actual_OPS": [0.790, 0.690],
            "n_evid": [300.0, 280.0],
        }
    )


def test_score_in_mew_zero_when_prediction_is_perfect():
    frame = _scored_frame()
    for stat in ("PA", "R", "HR", "RBI", "SB", "OPS"):
        frame[f"pred_{stat}"] = frame[f"actual_{stat}"]
    scored = score_in_mew(frame, _TOTALS, _GRADIENT, "hitting")
    assert scored["mew_error"].abs().max() < 1e-9, (
        f"A perfect prediction must have zero MEW error, got "
        f"{scored['mew_error'].abs().max()}"
    )


def test_score_in_mew_weights_by_gradient():
    """A 10-unit SB error must outweigh a 10-unit RBI error by ~g_SB/g_RBI."""
    frame = _scored_frame()
    for stat in ("PA", "R", "HR", "RBI", "SB", "OPS"):
        frame[f"pred_{stat}"] = frame[f"actual_{stat}"]

    sb_off = frame.copy()
    sb_off["pred_SB"] = sb_off["actual_SB"] + 10.0
    rbi_off = frame.copy()
    rbi_off["pred_RBI"] = rbi_off["actual_RBI"] + 10.0

    sb_err = score_in_mew(sb_off, _TOTALS, _GRADIENT, "hitting")["mew_error"].abs().sum()
    rbi_err = score_in_mew(rbi_off, _TOTALS, _GRADIENT, "hitting")["mew_error"].abs().sum()
    ratio = sb_err / rbi_err
    expected = _GRADIENT["SB"] / _GRADIENT["RBI"]
    assert abs(ratio - expected) < 0.01 * expected, (
        f"SB/RBI MEW-error ratio is {ratio:.1f}, expected ~{expected:.1f} "
        f"(g_SB/g_RBI). Scoring is not gradient-weighted."
    )


def test_baselines_present_and_atc_is_identity():
    assert set(BASELINES) >= {"atc", "raw_ytd", "flat_volume"}, (
        f"Spec §3.1 requires all three mandatory baselines, got {sorted(BASELINES)}"
    )
    frame = _scored_frame()
    predicted = BASELINES["atc"](frame, "hitting")
    assert (predicted["pred_PA"] == frame["proj_PA"]).all(), (
        "The 'atc' baseline must pass the projection through unchanged — it is "
        "the thing every candidate has to beat."
    )


def test_run_baselines_returns_one_row_per_baseline():
    result = run_baselines(_scored_frame(), _TOTALS, _GRADIENT, "hitting")
    assert set(result["baseline"]) >= {"atc", "raw_ytd", "flat_volume"}, (
        f"Missing baselines in result: {sorted(set(result['baseline']))}"
    )
    for col in ("baseline", "mae_mew", "rmse_mew", "n"):
        assert col in result.columns, f"Missing column {col}: {list(result.columns)}"
    assert (result["mae_mew"] >= 0).all(), "MAE cannot be negative"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: FAIL — `ImportError: cannot import name 'score_in_mew'`

- [ ] **Step 3: Write the implementation**

Append to `optimizer/backtest.py`:

```python
def _mew_contribution(
    frame: pd.DataFrame, prefix: str, my_totals: dict, gradient: dict, group: str
) -> pd.Series:
    """MEW for each player from one column family (proj_, pred_, or actual_).

    Mirrors optimizer.player_scoring.add_mew exactly: counting stats enter
    linearly, ratio stats enter volume-weighted against the team's own rate.
    """
    if group == "hitting":
        volume = frame[f"{prefix}PA"].astype(float)
        mew = (
            gradient["R"] * frame[f"{prefix}R"]
            + gradient["HR"] * frame[f"{prefix}HR"]
            + gradient["RBI"] * frame[f"{prefix}RBI"]
            + gradient["SB"] * frame[f"{prefix}SB"]
        )
        mew = mew + gradient["OPS"] * volume * (
            frame[f"{prefix}OPS"] - my_totals["OPS"]
        ) / my_totals["PA"]
        return mew.astype(float)

    volume = frame[f"{prefix}IP"].astype(float)
    mew = (
        gradient["W"] * frame[f"{prefix}W"]
        + gradient["SV"] * frame[f"{prefix}SV"]
        + gradient["K"] * frame[f"{prefix}K"]
    )
    for cat in ("ERA", "WHIP"):
        mew = mew + gradient[cat] * volume * (
            frame[f"{prefix}{cat}"] - my_totals[cat]
        ) / my_totals["IP"]
    return mew.astype(float)


def score_in_mew(
    frame: pd.DataFrame, my_totals: dict, gradient: dict, group: str = "hitting"
) -> pd.DataFrame:
    """Score a prediction in MEW units — the only metric that decides anything.

    Requires columns: pred_<stat> and actual_<stat> for every scoring category
    of `group`.
    Adds columns: mew_pred, mew_actual, mew_error.

    Stat-unit error is reported alongside by the caller, but the spec's §3.1
    decision rule is this column: a method that cuts OPS RMSE by 20% while
    moving no decision has earned nothing.
    """
    frame = frame.copy()
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS
    for stat in stats:
        for prefix in ("pred_", "actual_"):
            col = f"{prefix}{stat}"
            assert col in frame.columns, (
                f"score_in_mew: missing column {col}. Every candidate must "
                f"produce pred_<stat> for all of {stats}."
            )

    frame["mew_pred"] = _mew_contribution(frame, "pred_", my_totals, gradient, group)
    frame["mew_actual"] = _mew_contribution(frame, "actual_", my_totals, gradient, group)
    frame["mew_error"] = frame["mew_pred"] - frame["mew_actual"]
    return frame


def _baseline_atc(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Unadjusted projection. The thing every candidate must beat."""
    frame = frame.copy()
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS
    for stat in stats:
        frame[f"pred_{stat}"] = frame[f"proj_{stat}"]
    return frame


def _baseline_raw_ytd(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Season-to-date rates, unshrunk, projected onto the projection's volume.

    Brown (2008) found this is worse than the league grand mean for batting
    average. It is here to confirm that finding rather than to compete.
    """
    frame = frame.copy()
    stats = HITTING_STATS if group == "hitting" else PITCHING_STATS
    vol_col = "PA" if group == "hitting" else "IP"
    frame[f"pred_{vol_col}"] = frame[f"proj_{vol_col}"]

    if group == "hitting":
        per_pa = frame["proj_PA"] / frame["n_evid"].where(frame["n_evid"] > 0)
        for stat in ("R", "HR", "RBI", "SB"):
            observed = frame.get(f"evid_{stat}")
            frame[f"pred_{stat}"] = (
                observed * per_pa if observed is not None else frame[f"proj_{stat}"]
            )
        frame["pred_OPS"] = frame.get("evid_OPS", frame["proj_OPS"])
        return frame

    for stat in stats:
        if stat != vol_col:
            frame[f"pred_{stat}"] = frame[f"proj_{stat}"]
    return frame


def _baseline_flat_volume(frame: pd.DataFrame, group: str) -> pd.DataFrame:
    """Projection rates, but every player gets the pool's median volume.

    Zimmerman measured ATC's preseason PA RMSE at 156 against 162 for a flat
    510 PA. If a volume model cannot beat this, it is not a model.
    """
    frame = _baseline_atc(frame, group)
    vol_col = "PA" if group == "hitting" else "IP"
    median_volume = float(frame[f"proj_{vol_col}"].median())
    scale = median_volume / frame[f"proj_{vol_col}"].where(
        frame[f"proj_{vol_col}"] > 0
    )
    counting = ("R", "HR", "RBI", "SB") if group == "hitting" else ("W", "SV", "K")
    frame[f"pred_{vol_col}"] = median_volume
    for stat in counting:
        frame[f"pred_{stat}"] = frame[f"proj_{stat}"] * scale
    return frame


BASELINES: dict = {
    "atc": _baseline_atc,
    "raw_ytd": _baseline_raw_ytd,
    "flat_volume": _baseline_flat_volume,
}


def run_baselines(
    frame: pd.DataFrame, my_totals: dict, gradient: dict, group: str = "hitting"
) -> pd.DataFrame:
    """Score every mandatory baseline on one backtest frame.

    Returns:
        One row per baseline: baseline, mae_mew, rmse_mew, and n.
    """
    rows = []
    for name, build in BASELINES.items():
        scored = score_in_mew(build(frame, group), my_totals, gradient, group)
        error = scored["mew_error"].dropna()
        rows.append(
            {
                "baseline": name,
                "mae_mew": float(error.abs().mean()),
                "rmse_mew": float(np.sqrt((error**2).mean())),
                "n": int(len(error)),
            }
        )
    result = pd.DataFrame(rows).sort_values("mae_mew").reset_index(drop=True)
    print(f"baselines ({group}):\n{result.to_string(index=False)}")
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_backtest.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Run the baselines on real data**

Run:

```bash
uv run python -c "
import datetime, pandas as pd
from data_prep.build import build_players
from optimizer.league_state import compute_league_state
from optimizer.backtest import assemble_backtest_frame, run_baselines
players = build_players(system='atc')
state = compute_league_state(players)
proj = pd.read_csv('data_prep/data/pulled_20260611/fangraphs-atc-projections-hitters_ros.csv')
proj = proj[proj.MLBAMID.notna()]; proj['MLBAMID'] = proj.MLBAMID.astype(int)
f = assemble_backtest_frame(2026, datetime.date(2026,6,11), proj, 'hitting')
run_baselines(f, state['my_totals'], state['gradient'], 'hitting')
" 2>&1 | tail -8
```

Expected: `atc` should beat `raw_ytd`. If `flat_volume` beats `atc`, that is a real and important finding about ATC's volume projection, not a bug — record it in the plan's results and tell the user.

If `compute_league_state` has a different signature, read `optimizer/league_state.py` and adapt; the harness output is what matters, not the exact call.

- [ ] **Step 6: Commit**

```bash
git add optimizer/backtest.py tests/test_backtest.py
git commit -m "feat: score backtest predictions in MEW units, with baselines

Spec §3.1's decision rule: error is judged where it changes decisions, not
in stat units. Scoring mirrors add_mew exactly, so a 10-unit SB error
outweighs a 10-unit RBI error by g_SB/g_RBI (~76x at current state).

Three mandatory baselines: unadjusted ATC, raw unshrunk YTD (Brown 2008
says this loses to the grand mean), and flat median volume (Zimmerman:
ATC's PA RMSE is only 4% better than a constant)."
```

---

## Task 6: Wayback projection archive harvester

**Files:**
- Create: `data_prep/wayback.py`
- Test: `tests/test_wayback.py`

**Interfaces:**
- Consumes: `write_raw` from `data_prep.raw_io`
- Produces:
  - `extract_next_data(html: str) -> list[dict] | None`
  - `validate_ros_capture(frame, previous) -> tuple[bool, str]`
  - `list_captures(proj_type, stats, min_length=150_000) -> list[str]`
  - `harvest_capture(timestamp, proj_type, stats) -> pd.DataFrame | None`

**Why the parsing is separated from the fetching:** every trap here is a silent failure. A 503 body parses as "no data" rather than raising, so the extraction step must be independently testable offline.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_wayback.py`:

```python
"""
Offline tests for the Wayback projection harvester.
No network access. Per AGENTS.md: no classes, no fixtures, no mocking.
"""

import json

import pandas as pd

from data_prep.wayback import extract_next_data, validate_ros_capture

def _page(records: list[dict]) -> str:
    payload = {
        "props": {
            "pageProps": {
                "dehydratedState": {
                    "queries": [{"state": {"data": records}}]
                }
            }
        }
    }
    return (
        "<html><body>"
        f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script>'
        "</body></html>"
    )


def test_extract_next_data_reads_embedded_records():
    records = [{"PlayerName": "Bobby Witt Jr.", "PA": 199, "HR": 8}]
    assert extract_next_data(_page(records)) == records, (
        "Failed to extract the embedded dataset from __NEXT_DATA__"
    )


def test_extract_returns_none_for_503_body():
    """A Wayback 503 has no __NEXT_DATA__ and must not look like an empty page."""
    body = "<html><body><h1>503 Service Temporarily Unavailable</h1></body></html>"
    assert extract_next_data(body) is None, (
        "A body with no __NEXT_DATA__ must return None so the caller can "
        "retry. Returning [] would silently record a false empty."
    )


def test_validate_rejects_non_decaying_pa():
    """RoS PA must fall as the season progresses; a rise means full-season data."""
    earlier = pd.DataFrame({"PlayerName": ["A", "B"], "PA": [650.0, 600.0]})
    later_ok = pd.DataFrame({"PlayerName": ["A", "B"], "PA": [397.0, 350.0]})
    later_bad = pd.DataFrame({"PlayerName": ["A", "B"], "PA": [672.0, 640.0]})

    ok, reason = validate_ros_capture(later_ok, earlier)
    assert ok, f"A decaying capture was rejected: {reason}"

    ok, reason = validate_ros_capture(later_bad, earlier)
    assert not ok, (
        "A capture whose PA rose against an earlier one is mislabelled "
        "full-season data and must be rejected."
    )
    assert "pa" in reason.lower(), f"Reason should mention PA, got: {reason}"


def test_validate_rejects_empty_capture():
    ok, reason = validate_ros_capture(pd.DataFrame({"PlayerName": [], "PA": []}), None)
    assert not ok, "An empty capture must be rejected"
    assert "empty" in reason.lower(), f"Reason should mention emptiness, got: {reason}"


def test_validate_accepts_first_capture_with_no_predecessor():
    frame = pd.DataFrame({"PlayerName": ["A"], "PA": [650.0]})
    ok, reason = validate_ros_capture(frame, None)
    assert ok, f"The first capture of a season has nothing to compare to: {reason}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_wayback.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_prep.wayback'`

- [ ] **Step 3: Write the implementation**

Create `data_prep/wayback.py`:

```python
"""Harvest dated rest-of-season projections from the Wayback Machine.

FanGraphs' /projections page is server-rendered and embeds its full dataset
in <script id="__NEXT_DATA__">. The JSON API is paywalled; the page is not.

Every failure mode here is silent, so each is guarded explicitly:
  - A 503 body has no __NEXT_DATA__ and parses as "no data", not an error.
  - Captures are partial pages (587 rows observed against 1327 live).
  - Some captures labelled RoS contain full-season values.
  - CDX stores some URLs with a literal '&amp;'; substituting '&' returns a
    different capture.
"""

import json
import re
import time

import pandas as pd
import requests

_BROWSER_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    )
}
_NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)
_CDX = "https://web.archive.org/cdx/search/cdx"
_PROJECTIONS_URL = "https://www.fangraphs.com/projections"

# Wayback rate-limits hard; 2 of 3 probe requests returned 503 without this.
_MAX_ATTEMPTS = 4
_BACKOFF_SECONDS = 6

# Rest-of-season endpoint codes, matching scrape_fangraphs.PROJECTION_TYPES.
ROS_TYPES: tuple[str, ...] = ("ratcdc", "steamerr", "rthebatx", "rzipsdc")


def extract_next_data(html: str) -> list[dict] | None:
    """Pull the embedded dataset out of a captured page.

    Returns None when the page carries no __NEXT_DATA__ at all — which is
    what a Wayback 503 looks like. Returning an empty list here would record
    a false empty and silently shrink the training set.
    """
    match = _NEXT_DATA_RE.search(html)
    if match is None:
        return None
    payload = json.loads(match.group(1))
    queries = payload["props"]["pageProps"]["dehydratedState"]["queries"]
    return queries[0]["state"]["data"]


def validate_ros_capture(
    frame: pd.DataFrame, previous: pd.DataFrame | None
) -> tuple[bool, str]:
    """Check that a capture is genuinely rest-of-season.

    RoS playing time decays as the season progresses. A capture whose PA rose
    against an earlier one from the same season is mislabelled full-season
    data — observed in the wild on a `type=steamerr` capture.

    Args:
        frame: The capture under test.
        previous: An earlier accepted capture from the same season, or None
            for the first one.

    Returns:
        (is_valid, reason). Reason is "ok" when valid.
    """
    if len(frame) == 0:
        return False, "empty capture — no rows"
    if "PA" not in frame.columns:
        return False, f"no PA column; got {list(frame.columns)[:8]}"
    if previous is None or len(previous) == 0:
        return True, "ok"

    key = "PlayerName" if "PlayerName" in frame.columns else "Name"
    if key not in frame.columns or key not in previous.columns:
        return True, "ok"

    merged = frame[[key, "PA"]].merge(
        previous[[key, "PA"]], on=key, suffixes=("_now", "_prev")
    )
    if len(merged) < 10:
        return True, "ok"

    median_ratio = float((merged["PA_now"] / merged["PA_prev"]).median())
    if median_ratio > 1.05:
        return False, (
            f"PA rose against the earlier capture (median ratio "
            f"{median_ratio:.2f}); this looks like full-season data "
            f"mislabelled as rest-of-season"
        )
    return True, "ok"


def list_captures(
    proj_type: str, stats: str = "bat", min_length: int = 150_000
) -> list[str]:
    """List Wayback timestamps holding a usable capture of one projection feed.

    Args:
        proj_type: A RoS endpoint code, e.g. "ratcdc".
        stats: "bat" or "pit".
        min_length: Skip captures smaller than this; small ones are error
            pages or empty shells.

    Returns:
        Timestamps (YYYYMMDDhhmmss), oldest first, one per distinct day.
    """
    response = requests.get(
        _CDX,
        params={
            "url": "www.fangraphs.com/projections*",
            "matchType": "prefix",
            "filter": "statuscode:200",
            "output": "json",
            "collapse": "timestamp:8",
            "fl": "timestamp,original,length",
        },
        headers=_BROWSER_UA,
        timeout=120,
    )
    rows = response.json()
    assert rows and rows[0][0] == "timestamp", (
        f"list_captures: unexpected CDX response shape: {rows[:2]}"
    )

    wanted_type = f"type={proj_type}"
    wanted_stats = f"stats={stats}"
    timestamps = []
    for timestamp, original, length in rows[1:]:
        # CDX stores some URLs with a literal '&amp;'. Normalise for MATCHING
        # only — never for fetching, where the exact stored string matters.
        normalised = original.replace("&amp;", "&")
        if wanted_type not in normalised or wanted_stats not in normalised:
            continue
        if not length.isdigit() or int(length) < min_length:
            continue
        timestamps.append(timestamp)

    print(f"CDX: {len(timestamps)} usable captures for {proj_type}/{stats}")
    return sorted(timestamps)


def harvest_capture(
    timestamp: str, proj_type: str, stats: str = "bat"
) -> pd.DataFrame | None:
    """Fetch and parse one capture, retrying through Wayback's 503s.

    Returns None if every attempt failed — the caller decides whether a gap
    is tolerable. Never returns an empty frame for a failed fetch.
    """
    url = f"{_PROJECTIONS_URL}?type={proj_type}&stats={stats}&pos=all"
    for attempt in range(_MAX_ATTEMPTS):
        response = requests.get(
            f"https://web.archive.org/web/{timestamp}id_/{url}",
            headers=_BROWSER_UA,
            timeout=120,
        )
        records = extract_next_data(response.text)
        if records:
            frame = pd.DataFrame(records)
            print(f"  {timestamp} {proj_type}/{stats}: {len(frame)} rows")
            return frame
        time.sleep(_BACKOFF_SECONDS * (attempt + 1))

    print(f"  {timestamp} {proj_type}/{stats}: FAILED after {_MAX_ATTEMPTS} attempts")
    return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_wayback.py -v`
Expected: PASS, 5 tests.

- [ ] **Step 5: Harvest and validate the real archive**

Run:

```bash
uv run python -c "
import pandas as pd
from data_prep.wayback import list_captures, harvest_capture, validate_ros_capture
ts = list_captures('ratcdc', 'bat')
print('captures:', len(ts))
kept, prev = [], None
for t in ts[:6]:
    f = harvest_capture(t, 'ratcdc', 'bat')
    if f is None: continue
    ok, why = validate_ros_capture(f, prev)
    print(f'  {t}: {len(f)} rows  valid={ok}  {why}')
    if ok: kept.append((t, f)); prev = f
print('kept', len(kept), 'of', len(ts[:6]))
"
```

Expected: a nonzero count of validated captures. **Record the surviving count** — spec open question 1 asks exactly this, and if it is very small the fitted `M_vs_ATC` must fall back to the reconstruction route. Report the number to the user before proceeding.

- [ ] **Step 6: Commit**

```bash
git add data_prep/wayback.py tests/test_wayback.py
git commit -m "feat: harvest dated RoS projections from the Wayback Machine

FanGraphs' /projections page embeds its full dataset in __NEXT_DATA__; the
paywall is on the JSON API, not the page. Extends the backtest archive from
7 local snapshots to the wider 2023-2026 record.

Guards every silent failure: a 503 body parses as 'no data' so extraction
returns None rather than [], captures whose PA rose against an earlier one
are rejected as mislabelled full-season, and CDX's literal '&amp;' is
normalised for matching but never for fetching."
```

---

## Task 7: Fit and apply the volume correction

**Files:**
- Create: `data_prep/volume_adjust.py`
- Test: `tests/test_volume_adjust.py`

**Interfaces:**
- Consumes: `assemble_backtest_frame`, `score_in_mew` (Tasks 4–5); `add_skill_rates` (Task 3)
- Produces:
  - `fit_volume_correction(frame, group="hitting") -> dict[str, float]`
  - `adjust_projection_volume(players, ytd, coefficients) -> pd.DataFrame`
  - Columns added to `players`: `PA_adj_factor`, `IP_adj_factor`
  - Columns rewritten: `PA, AB, R, HR, RBI, SB` (hitters); `IP, W, SV, K` (pitchers)

**Model** (spec §3.3): a log-linear multiplier on projected volume, from Zimmerman's three validated drivers plus the slump term.

```
log(PA_actual / PA_proj) = b0 + b_age·(age − 30) + b_talent·(proj_OPS − 0.730)
                              + b_slump·(evid_OPS − proj_OPS)
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_volume_adjust.py`:

```python
"""
Offline tests for the RoS volume correction.
Per AGENTS.md: no classes, no fixtures, no mocking.

These tests guard invariants that NOTHING else in the codebase enforces:
nothing checks OPS against PA, and scaling PA does not scale R/HR/RBI/SB.
"""

import numpy as np
import pandas as pd

from data_prep.volume_adjust import adjust_projection_volume

_COEFFS = {
    "b0": -0.10, "b_age": -0.01, "b_talent": 0.50, "b_slump": 0.40,
    "min_factor": 0.25, "max_factor": 2.0,
}


def _players() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["Hitter A", "Hitter B", "Pitcher C"],
            "MLBAMID": [1, 2, 3],
            "player_type": ["hitter", "hitter", "pitcher"],
            "age": [29.0, 34.0, 27.0],
            "PA": [130.0, 120.0, 0.0], "AB": [117.0, 108.0, 0.0],
            "R": [16.0, 14.0, 0.0], "HR": [5.0, 4.0, 0.0],
            "RBI": [17.0, 15.0, 0.0], "SB": [4.0, 1.0, 0.0],
            "OPS": [0.780, 0.700, 0.0],
            "IP": [0.0, 0.0, 40.0], "W": [0.0, 0.0, 3.0],
            "SV": [0.0, 0.0, 0.0], "K": [0.0, 0.0, 42.0],
            "ERA": [0.0, 0.0, 3.50], "WHIP": [0.0, 0.0, 1.10],
        }
    )


def _ytd() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "MLBAMID": [1, 2, 3],
            "group": ["hitting", "hitting", "pitching"],
            "n_PA": [400.0, 380.0, np.nan],
            "n_BF": [np.nan, np.nan, 500.0],
            "ytd_OPS": [0.800, 0.560, np.nan],
        }
    )


def test_counting_stats_scale_with_volume():
    """The invariant nothing else enforces: scaling PA must scale R/HR/RBI/SB."""
    before = _players()
    after = adjust_projection_volume(before, _ytd(), _COEFFS)
    a = after[after["Name"] == "Hitter A"].iloc[0]
    b = before[before["Name"] == "Hitter A"].iloc[0]
    factor = a["PA"] / b["PA"]
    for stat in ("AB", "R", "HR", "RBI", "SB"):
        assert abs(a[stat] - b[stat] * factor) < 1e-9, (
            f"{stat} is {a[stat]} but PA scaled by {factor:.4f} from {b[stat]}, "
            f"so it should be {b[stat] * factor}. A player who plays more games "
            f"and scores the same runs is a silent corruption."
        )
    assert abs(a["PA_adj_factor"] - factor) < 1e-9, (
        f"PA_adj_factor {a['PA_adj_factor']} disagrees with the applied "
        f"factor {factor}"
    )


def test_rates_are_untouched():
    after = adjust_projection_volume(_players(), _ytd(), _COEFFS)
    before = _players()
    for name, col in (("Hitter A", "OPS"), ("Pitcher C", "ERA"), ("Pitcher C", "WHIP")):
        a = after[after["Name"] == name].iloc[0][col]
        b = before[before["Name"] == name].iloc[0][col]
        assert a == b, (
            f"{name}'s {col} changed from {b} to {a}. This is the VOLUME "
            f"corrector; rate correction is Part 2b and out of scope."
        )


def test_opposite_type_columns_stay_exactly_zero():
    after = adjust_projection_volume(_players(), _ytd(), _COEFFS)
    hitters = after[after["player_type"] == "hitter"]
    pitchers = after[after["player_type"] == "pitcher"]
    for col in ("IP", "W", "SV", "K", "ERA", "WHIP"):
        assert (hitters[col] == 0.0).all(), (
            f"Hitter {col} is not exactly 0.0: {hitters[col].tolist()}. "
            f"build.py:236 zeroes these and MEW gains a phantom term otherwise."
        )
    for col in ("PA", "R", "HR", "RBI", "SB", "OPS"):
        assert (pitchers[col] == 0.0).all(), (
            f"Pitcher {col} is not exactly 0.0: {pitchers[col].tolist()}"
        )


def test_never_scales_a_player_to_zero_volume():
    """Zero volume drops a player from FV's z-population and benches IL players."""
    extreme = {**_COEFFS, "b_slump": 50.0}
    after = adjust_projection_volume(_players(), _ytd(), extreme)
    hitters = after[after["player_type"] == "hitter"]
    assert (hitters["PA"] > 0).all(), (
        f"A hitter reached zero PA: {hitters[['Name', 'PA']].to_dict('records')}. "
        f"Clamp to min_factor — zero volume silently removes him from FV's "
        f"ratio z-population and permanently benches him if he is on the IL."
    )
    assert (hitters["PA_adj_factor"] >= _COEFFS["min_factor"] - 1e-9).all(), (
        "min_factor clamp was not applied"
    )


def test_slumping_player_loses_playing_time():
    """MGL: cold hitters lose ~30 PA that projections do not anticipate."""
    after = adjust_projection_volume(_players(), _ytd(), _COEFFS)
    hot = after[after["Name"] == "Hitter A"].iloc[0]["PA_adj_factor"]
    cold = after[after["Name"] == "Hitter B"].iloc[0]["PA_adj_factor"]
    assert cold < hot, (
        f"Hitter B is 140 points of OPS below his projection and Hitter A is "
        f"20 above, yet B's factor ({cold:.3f}) is not below A's ({hot:.3f})."
    )


def test_players_with_no_ytd_are_left_alone():
    players = _players()
    after = adjust_projection_volume(players, _ytd().iloc[:0], _COEFFS)
    hitters = after[after["player_type"] == "hitter"]
    assert (hitters["PA_adj_factor"] == 1.0).all(), (
        f"A player with no YTD evidence must pass through unchanged, got "
        f"{hitters['PA_adj_factor'].tolist()}"
    )


def test_does_not_mutate_input():
    players = _players()
    before_pa = players["PA"].tolist()
    adjust_projection_volume(players, _ytd(), _COEFFS)
    assert players["PA"].tolist() == before_pa, (
        "adjust_projection_volume mutated its input; it must copy first."
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_volume_adjust.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_prep.volume_adjust'`

- [ ] **Step 3: Write the implementation**

Create `data_prep/volume_adjust.py`:

```python
"""Correct rest-of-season playing time using in-season evidence.

Volume is the only gradient-invariant input (spec §1.4): a PA error moves
every counting category AND re-weights the ratio, so its value holds in every
league state. That is why this ships and rate correction is gated.

Model, from Zimmerman's three validated drivers plus MGL's slump-benching
effect:

    log(PA_actual / PA_proj) = b0
                             + b_age    * (age - 30)
                             + b_talent * (proj_OPS - 0.730)
                             + b_slump  * (ytd_OPS - proj_OPS)

b0 absorbs the systematic ~10% over-projection every system shows.
"""

import numpy as np
import pandas as pd

# Reference points, so coefficients read as deviations rather than intercept soup.
_AGE_REFERENCE: float = 30.0
_OPS_REFERENCE: float = 0.730

_HITTER_COUNTING: tuple[str, ...] = ("AB", "R", "HR", "RBI", "SB")
_PITCHER_COUNTING: tuple[str, ...] = ("W", "SV", "K")

REQUIRED_COEFFICIENTS: tuple[str, ...] = (
    "b0", "b_age", "b_talent", "b_slump", "min_factor", "max_factor",
)


def _volume_factor(
    frame: pd.DataFrame, coefficients: dict[str, float]
) -> pd.Series:
    """Multiplier on projected volume. 1.0 where evidence is missing."""
    log_factor = (
        coefficients["b0"]
        + coefficients["b_age"] * (frame["age"] - _AGE_REFERENCE)
        + coefficients["b_talent"] * (frame["proj_OPS"] - _OPS_REFERENCE)
        + coefficients["b_slump"] * (frame["ytd_OPS"] - frame["proj_OPS"])
    )
    factor = np.exp(log_factor)
    # No evidence -> no opinion. Never guess a player into or out of a lineup.
    factor = factor.where(frame["ytd_OPS"].notna(), 1.0)
    factor = factor.where(frame["age"].notna(), 1.0)
    return factor.clip(
        lower=coefficients["min_factor"], upper=coefficients["max_factor"]
    ).astype(float)


def adjust_projection_volume(
    players: pd.DataFrame, ytd: pd.DataFrame, coefficients: dict[str, float]
) -> pd.DataFrame:
    """Rescale rest-of-season volume, carrying counting stats with it.

    Requires columns on `players`: Name, MLBAMID, player_type, age, and all of
    PA, AB, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Requires columns on `ytd`: MLBAMID, group, n_PA, n_BF, ytd_OPS.

    Adds columns: PA_adj_factor, IP_adj_factor.
    Rewrites columns: PA, AB, R, HR, RBI, SB (hitters); IP, W, SV, K (pitchers).

    Rates (OPS, ERA, WHIP) are NOT touched — that is Part 2b, which is gated.
    """
    players = players.copy()
    missing = [c for c in REQUIRED_COEFFICIENTS if c not in coefficients]
    assert not missing, (
        f"adjust_projection_volume: coefficients missing {missing}. Fit them "
        f"with fit_volume_correction against a backtest frame; do not guess."
    )
    assert coefficients["min_factor"] > 0.0, (
        f"adjust_projection_volume: min_factor is {coefficients['min_factor']}; "
        f"it must be strictly positive. Zero volume drops a player from FV's "
        f"ratio z-population and permanently benches him if he is on the IL."
    )

    evidence = ytd[["MLBAMID", "ytd_OPS"]].drop_duplicates("MLBAMID")
    frame = players[["MLBAMID", "age", "OPS"]].rename(columns={"OPS": "proj_OPS"})
    frame = frame.merge(evidence, on="MLBAMID", how="left")
    frame.index = players.index

    factor = _volume_factor(frame, coefficients)
    is_hitter = players["player_type"] == "hitter"
    is_pitcher = players["player_type"] == "pitcher"

    players["PA_adj_factor"] = factor.where(is_hitter, 1.0)
    players["IP_adj_factor"] = factor.where(is_pitcher, 1.0)

    for col in ("PA", *_HITTER_COUNTING):
        players.loc[is_hitter, col] = (
            players.loc[is_hitter, col] * players.loc[is_hitter, "PA_adj_factor"]
        )
    for col in ("IP", *_PITCHER_COUNTING):
        players.loc[is_pitcher, col] = (
            players.loc[is_pitcher, col] * players.loc[is_pitcher, "IP_adj_factor"]
        )

    n_moved = int(((factor - 1.0).abs() > 0.01).sum())
    print(
        f"volume adjustment: {n_moved} of {len(players)} players moved >1% "
        f"(median factor {float(factor.median()):.3f})"
    )
    return players


def fit_volume_correction(
    frame: pd.DataFrame, group: str = "hitting"
) -> dict[str, float]:
    """Fit the volume multiplier by OLS on log(actual / projected) volume.

    Requires columns on `frame`: proj_PA/proj_IP, actual_PA/actual_IP,
    proj_OPS, evid_OPS, age.

    Returns the coefficient dict `adjust_projection_volume` consumes.
    """
    vol = "PA" if group == "hitting" else "IP"
    usable = frame[
        (frame[f"proj_{vol}"] > 0)
        & (frame[f"actual_{vol}"] > 0)
        & frame["age"].notna()
        & frame["evid_OPS"].notna()
    ].copy()
    assert len(usable) >= 50, (
        f"fit_volume_correction: only {len(usable)} usable rows for {group}. "
        f"Fitting four coefficients on this is overfitting; widen the window "
        f"or fall back to the reconstruction route (spec §3.1)."
    )

    target = np.log(usable[f"actual_{vol}"] / usable[f"proj_{vol}"])
    design = np.column_stack(
        [
            np.ones(len(usable)),
            usable["age"] - _AGE_REFERENCE,
            usable["proj_OPS"] - _OPS_REFERENCE,
            usable["evid_OPS"] - usable["proj_OPS"],
        ]
    )
    beta, *_ = np.linalg.lstsq(design, target.values, rcond=None)
    coefficients = {
        "b0": float(beta[0]),
        "b_age": float(beta[1]),
        "b_talent": float(beta[2]),
        "b_slump": float(beta[3]),
        "min_factor": 0.25,
        "max_factor": 2.0,
    }
    residual = target.values - design @ beta
    r_squared = 1.0 - residual.var() / target.values.var()
    print(
        f"fit_volume_correction ({group}, n={len(usable)}): R2={r_squared:.3f} "
        f"{ {k: round(v, 4) for k, v in coefficients.items()} }"
    )
    return coefficients
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_volume_adjust.py -v`
Expected: PASS, 7 tests.

- [ ] **Step 5: Fit on real data and check it beats the baselines**

Run:

```bash
uv run python -c "
import datetime, pandas as pd
from data_prep.build import build_players
from data_prep.volume_adjust import fit_volume_correction
from optimizer.league_state import compute_league_state
from optimizer.backtest import assemble_backtest_frame, run_baselines
players = build_players(system='atc')
state = compute_league_state(players)
proj = pd.read_csv('data_prep/data/pulled_20260611/fangraphs-atc-projections-hitters_ros.csv')
proj = proj[proj.MLBAMID.notna()]; proj['MLBAMID']=proj.MLBAMID.astype(int)
f = assemble_backtest_frame(2026, datetime.date(2026,6,11), proj, 'hitting')
f = f.merge(players[['MLBAMID','age']].dropna().drop_duplicates('MLBAMID'), on='MLBAMID', how='left')
print(fit_volume_correction(f, 'hitting'))
run_baselines(f, state['my_totals'], state['gradient'], 'hitting')
" 2>&1 | tail -12
```

`evid_OPS` and `age` are the two columns the fit needs beyond the backtest frame: the first comes from Task 4's `assemble_backtest_frame`, the second from the merge above.

**Decision gate.** Compare the fitted corrector's MAE in MEW units against `atc`. **If it does not beat unadjusted ATC, stop and report to the user — do not wire it into `build_players`.** That is the entire point of Part 0.

- [ ] **Step 6: Commit**

```bash
git add data_prep/volume_adjust.py tests/test_volume_adjust.py
git commit -m "feat: fit and apply rest-of-season volume correction

Log-linear multiplier from Zimmerman's three validated drivers (age,
projected talent, and an intercept absorbing the systematic ~10%
over-projection) plus MGL's slump-benching term, which is the channel
through which an in-season rate signal actually reaches a decision.

Tests guard the invariants nothing else in the codebase enforces: scaling
PA carries R/HR/RBI/SB with it, opposite-type columns stay exactly 0.0,
rates are untouched (Part 2b is gated), and no player is ever scaled to
zero volume."
```

---

## Task 8: Wire the corrector into `build_players`

**Files:**
- Modify: `data_prep/build.py` (near line 717, the `return players` of `build_players`)
- Test: `tests/test_build.py` (append)

**Interfaces:**
- Consumes: `adjust_projection_volume` (Task 7), the `ytd` raw source (Task 2)
- Produces: `build_players(..., adjust_volume: bool = False)`

**Gate:** only do this task if Task 7's step 5 showed the corrector beating unadjusted ATC in MEW units. The default stays `False` so every existing caller and test is unaffected until the numbers justify flipping it.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_build.py`:

```python
def test_build_players_accepts_adjust_volume_flag():
    import inspect

    from data_prep.build import build_players

    params = inspect.signature(build_players).parameters
    assert "adjust_volume" in params, (
        f"build_players has no adjust_volume parameter; got {list(params)}"
    )
    assert params["adjust_volume"].default is False, (
        f"adjust_volume must default to False so existing callers are "
        f"unaffected until the backtest justifies flipping it; default is "
        f"{params['adjust_volume'].default}"
    )


def test_volume_adjustment_preserves_scoring_columns():
    """A team-wide volume change must not produce NaN or negative volume."""
    import numpy as np
    import pandas as pd

    from data_prep.volume_adjust import adjust_projection_volume

    players = pd.DataFrame(
        {
            "Name": ["H", "P"], "MLBAMID": [1, 2],
            "player_type": ["hitter", "pitcher"], "age": [28.0, 31.0],
            "PA": [130.0, 0.0], "AB": [117.0, 0.0], "R": [16.0, 0.0],
            "HR": [5.0, 0.0], "RBI": [17.0, 0.0], "SB": [4.0, 0.0],
            "OPS": [0.780, 0.0], "IP": [0.0, 40.0], "W": [0.0, 3.0],
            "SV": [0.0, 0.0], "K": [0.0, 42.0], "ERA": [0.0, 3.5],
            "WHIP": [0.0, 1.1],
        }
    )
    ytd = pd.DataFrame(
        {"MLBAMID": [1, 2], "group": ["hitting", "pitching"],
         "n_PA": [400.0, np.nan], "n_BF": [np.nan, 500.0],
         "ytd_OPS": [0.700, np.nan]}
    )
    coefficients = {
        "b0": -0.10, "b_age": -0.01, "b_talent": 0.50, "b_slump": 0.40,
        "min_factor": 0.25, "max_factor": 2.0,
    }
    out = adjust_projection_volume(players, ytd, coefficients)
    for col in ("PA", "IP", "R", "HR", "RBI", "SB", "OPS", "W", "SV", "K", "ERA", "WHIP"):
        assert out[col].notna().all(), f"{col} has NaN after adjustment"
        assert (out[col] >= 0).all(), f"{col} went negative: {out[col].tolist()}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_build.py -v`
Expected: FAIL — `build_players has no adjust_volume parameter`

- [ ] **Step 3: Wire it in**

In `data_prep/build.py`, add the parameter to `build_players` (line ~650):

```python
def build_players(
    system: str = "atc",
    on_or_before=None,
    include_market: bool = True,
    include_identity: bool = True,
    adjust_volume: bool = False,
) -> pd.DataFrame:
```

Add to the docstring's Args:

```
        adjust_volume: Apply the fitted rest-of-season volume correction from
            the ytd snapshot. Defaults to False; see
            docs/superpowers/specs/2026-08-20-in-season-projection-conditioning-design.md.
```

Insert the block **immediately before the `_warn_if_sources_disagree_on_date(used)` call at line ~713** — NOT before `return players` at ~717. The `used` dict is consumed at 713 (staleness warning) and 714 (`players.attrs["snapshot_dates"]`); inserting after those makes `used["ytd"] = ytd_date` dead code and the ytd snapshot date silently vanishes from provenance:

```python
    if adjust_volume:
        from .raw_io import read_latest_raw
        from .skills import add_skill_rates
        from .volume_adjust import VOLUME_COEFFICIENTS, adjust_projection_volume

        ytd_raw, ytd_date = read_latest_raw("ytd", on_or_before)
        print(f"ytd snapshot: {ytd_date}")
        ytd = add_skill_rates(ytd_raw)
        ytd["ytd_OPS"] = (ytd["obp"] + ytd["slg"]).astype(float)
        players = adjust_projection_volume(players, ytd, VOLUME_COEFFICIENTS)
        used["ytd"] = ytd_date
```

Add the fitted constants to `data_prep/volume_adjust.py`, replacing the placeholder values with the numbers Task 7 step 5 actually produced:

```python
# Fitted against the 2026-06-11 backtest window. Refit with
# fit_volume_correction whenever the archive grows; do not hand-tune.
VOLUME_COEFFICIENTS: dict[str, float] = {
    "b0": 0.0,        # <- replace with the fitted value
    "b_age": 0.0,     # <- replace with the fitted value
    "b_talent": 0.0,  # <- replace with the fitted value
    "b_slump": 0.0,   # <- replace with the fitted value
    "min_factor": 0.25,
    "max_factor": 2.0,
}
```

- [ ] **Step 4: Run the full suite**

Run: `uv run pytest -v`
Expected: PASS — all pre-existing tests plus the new ones. The pre-existing 41 must still pass; `adjust_volume` defaults to `False`, so nothing changes for them.

- [ ] **Step 5: Measure the effect on EW**

Run:

```bash
uv run python -c "
from data_prep.build import build_players
from optimizer.league_state import compute_league_state
for flag in (False, True):
    p = build_players(system='atc', adjust_volume=flag)
    s = compute_league_state(p)
    print(f'adjust_volume={flag}: EW={s[\"current_ew\"]:.3f}')
" 2>&1 | grep "adjust_volume="
```

Expected: two EW numbers. **Report both to the user.** Spec §3.3 measured that a team-wide +10% hitter volume bump moved EW 30.52 → 31.59 and shifted `g_RBI` by 10×, so this adjustment is not local — a large EW move is expected and is information, not an error.

- [ ] **Step 6: Commit**

```bash
git add data_prep/build.py data_prep/volume_adjust.py tests/test_build.py
git commit -m "feat: wire volume correction into build_players behind a flag

adjust_volume defaults to False so every existing caller and the 41
pre-existing tests are unaffected. Coefficients are fitted, not hand-tuned;
refit with fit_volume_correction as the archive grows.

The adjustment is not local: sigma is computed from the RoS league mean
including our own team, so the gradient moves too."
```

---

## Self-Review

**1. Spec coverage**

| Spec section | Task |
|---|---|
| §3.1 backtest harness, MEW-unit metric, 3 baselines | Tasks 4, 5 |
| §3.1 Wayback archive extension + traps | Task 6 |
| §3.1 `byDateRange` reconstruction route | Task 1 |
| §3.2 in-season evidence layer, skill decomposition | Tasks 1, 2, 3 |
| §3.2 parsing traps (playerPool, outs, string rates) | Task 1 |
| §3.3 volume corrector + Zimmerman drivers + slump term | Task 7 |
| §3.3 invariants (counting scale, 0.0 columns, no zero volume) | Task 7 |
| §3.3 single seam at build.py:717 | Task 8 |
| §3.4 Part 2b rate correction | **Deliberately absent — gated** |
| §3.5 Statcast / changepoints | **Deliberately absent — out of scope** |

Open question 1 (how much of the archive survives validation) is answered by Task 6 step 5, which instructs the implementer to report the count. Open question 3 (pitcher asymmetry) is exposed by the `group` parameter threaded through Tasks 4, 5, and 7 — the pitcher path is implemented but its calibration is expected to be thin, which the `n >= 50` assert in `fit_volume_correction` will surface loudly rather than silently overfitting.

**2. Placeholder scan** — one intentional placeholder remains: `VOLUME_COEFFICIENTS` in Task 8 step 3 ships as zeros with an explicit instruction to replace them with Task 7's fitted output. This is deliberate: hardcoding invented coefficients would be exactly the knob-tuning the spec exists to prevent. The `n >= 50` assert and the Task 7 decision gate prevent shipping unfitted values silently.

**3. Type consistency** — `add_skill_rates` produces `n_PA`/`n_BF`; Task 4 renames to `n_evid`; Task 7 consumes `n_evid` from backtest frames and `ytd_OPS` from the raw ytd frame. `HITTING_SKILLS`/`PITCHING_SKILLS` are defined in `skills.py` and imported by `backtest.py`. `HITTING_STATS`/`PITCHING_STATS` are defined in `backtest.py` only — note these shadow same-named constants in `data_prep/build.py`, which are a different thing (that module's are column lists for zeroing). They are never imported across modules, so there is no collision, but do not import one where the other is meant.

The `evid_OPS` gap found in this review is fixed: Task 4's `assemble_backtest_frame` now carries `obp + slg` from the evidence side, which is what Task 7's slump term consumes. Note the asymmetry that survives deliberately — the volume corrector's slump term uses the observed *composite* OPS, while every rate consumer must use components. That is not an inconsistency: the slump term models a manager's benching decision, and managers react to the composite.

**One test to add when implementing Task 4**, since the column is now produced there:

```python
def test_evid_ops_is_carried_from_evidence():
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    assert abs(kept["evid_OPS"] - (0.330 + 0.440)) < 1e-9, (
        f"evid_OPS is {kept['evid_OPS']}, expected obp+slg = 0.770"
    )


def test_evid_counting_stats_are_carried():
    """The raw_ytd baseline needs observed counts, not just skill rates."""
    frame = assemble_backtest_frame(
        2026, datetime.date(2026, 6, 11), _projection(),
        evidence=_evidence(), actual=_actual(),
    )
    kept = frame[frame["MLBAMID"] == 1].iloc[0]
    for stat, expected in (("R", 40.0), ("HR", 12.0), ("RBI", 41.0), ("SB", 6.0)):
        assert kept[f"evid_{stat}"] == expected, (
            f"evid_{stat} is {kept[f'evid_{stat}']}, expected {expected}. "
            f"Without these the raw_ytd baseline silently degenerates into the "
            f"ATC baseline with one column swapped."
        )
```
