# Fantrax API integration

Ported from v1 `implementation_specs …/01c_fantrax_api.md`, condensed for **`data_prep.fantrax_api`**.

## Overview

Fantrax is the **source of truth** for roster ownership, roster status, and position strings used in the silver table. Undocumented HTTP JSON API.

**Module:** `data_prep.fantrax_api`  
**Config:** `FANTRAX_LEAGUE_ID`, `FANTRAX_TEAM_IDS`, `FANTRAX_COOKIES` from `data_prep.config` (loaded from shared `config.json`).

---

## Critical: `getPlayerStats` pagination

**Do not rely on pagination parameters.** Use a single request with `maxResultsPerPage=5000` (see `fetch_player_pool`).

---

## What Fantrax provides (for silver build)

| Data | Endpoint / source | Notes |
|------|-------------------|--------|
| Rosters | `getTeamRosterInfo` | All teams; `statusId` for roster slot (see below) |
| Positions | Roster + player pool | `posShortNames` |
| Injury status | Roster + player pool | `scorer.icons` — see below |
| Pool stats | `getPlayerStats` | `fantrax_score`, `%` rostered, etc., for top N players |
| Standings | `getStandings` | Fetched in `fetch_all_fantrax_data`; not required for silver CSV write |

### `statusId` (roster slot) — authoritative values

From the API's own `statusTotals` (do **not** guess these):

| `statusId` | `roster_status` |
|------------|-----------------|
| `1` | `active` |
| `2` | `reserve` |
| `3` | `IR` (Inj Res) |
| `9` | `minors` |

`statusId` is the **fantasy roster slot** the manager chose, **not** a real-world
injury signal: injured players are frequently left in `active`/`reserve` slots.

### Injury status — `scorer.icons`

Real-world injury state lives in `scorer.icons`, a list of `{tooltip, typeId}`:

| `typeId` | Meaning | `injury_status` |
|----------|---------|-----------------|
| `1` | Day-to-Day (e.g. "Oblique - Day-to-Day"); also non-injury absences like Paternity Leave | `DTD` |
| `2` | On the Injured List (e.g. "Injured List - 10-day IL - Oblique") | `IL` |

`_parse_injury()` extracts `(injury_status, injury_detail)` (IL beats DTD). Both
the roster and player-pool parsers populate these; they flow to the silver
`injury_status` / `injury_detail` columns. Other `typeId`s are
lineup/handedness/batting-order/news markers and are ignored.

Full projections (PA, WAR, pitching depth) still come from **FanGraphs** CSVs.

---

## API robustness

1. Check `isinstance(x, dict)` before `.get()` on nested values.
2. Tables may appear under `tables` **or** `tableList`.
3. On errors, `pageError` may be present and `data` absent — handle before reading rows.
4. **Two-way / dual rows:** If a name already ends with `-H` or `-P`, do not append another suffix.

---

## Authentication

Private league → browser cookies on `fantrax.com`:

- `JSESSIONID`
- `FX_RM`

Place under `fantrax.cookies` in repo-root `config.json`. `create_session()` attaches these to a `requests.Session`.

`test_auth` calls `getFantasyLeagueInfo`; `WARNING_NOT_LOGGED_IN` means refresh cookies.

---

## Request shape

```text
POST https://www.fantrax.com/fxpa/req?leagueId=<id>
Content-Type: application/json

{"msgs": [{"method": "<name>", "data": {...}}]}
```

League id and team ids come from `config.json` → `league.fantrax_league_id`, `league.fantrax_team_ids`.

---

## Cell mapping (reference)

**`getPlayerStats` (player pool):** cells[0]=Rank, [1]=Status, [2]=Age, [3]=Score, [4+] rostered/trend fields as implemented in `_parse_cell`.

**`getTeamRosterInfo` (roster):** cells[0]=Age, …; scorer holds `name`, `posShortNames`, `teamShortName`.

---

## Name corrections

`FANTRAX_NAME_CORRECTIONS` maps rare Fantrax strings to names that align with FanGraphs after normalization. Extend in `data_prep/fantrax_api.py` when mismatches appear.

---

## Player type → `-H` / `-P`

`get_player_type(position)` returns `pitcher` if `SP` or `RP` appears in the position string; else `hitter`. Suffix choice follows v1 behavior (see source).

---

## Functions used by `build_silver_table`

- `create_session`, `test_auth`
- `fetch_all_fantrax_data` → rosters, roster_sets, standings, player_pool
- `merge_fantrax` in `data_prep.build` applies pool + roster rows to the projections DataFrame

`extract_slot_assignments` (MILP) remains in **v1 optimizer**, not in `data_prep`.
