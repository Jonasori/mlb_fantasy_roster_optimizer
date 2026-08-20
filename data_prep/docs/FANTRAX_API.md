# Fantrax API integration

Fantrax is the **source of truth** for roster ownership, roster slot, position
eligibility and injury state. Undocumented HTTP JSON API.

**Module:** `data_prep.fantrax_api` — two raw sources, because they are two
different grains and there is no honest way to put them in one table:

```
data/raw/fantrax/YYYY-MM-DD.parquet     player grain: ownership, positions,
                                        roster slot, injuries, % rostered
data/raw/standings/YYYY-MM-DD.parquet   team grain: banked YTD category totals
```

**Fetch:** `uv run fetch fantrax` (or `uv run python -m data_prep.fantrax_api`).
**Join:** `build.merge_fantrax(players, fantrax)`.

Both snapshots are pure Fantrax. This module does **not** merge projections,
does **not** do FanGraphs name reconciliation and does **not** add `-H`/`-P`
suffixes. That is `build.py`'s job, downstream.

---

## Authentication — read this first

**This is the only source in the pipeline that needs hand-pasted credentials,
and expired cookies are by far the most common failure mode in the whole
project.** The league is private, so there is no token and no API key: you copy
two browser cookies for `fantrax.com` into repo-root `config.json`.

```json
{
  "fantrax": {
    "cookies": {
      "JSESSIONID": "…",
      "FX_RM": "…"
    }
  }
}
```

DevTools → Application → Cookies → `fantrax.com`. **They expire every few
weeks**, at which point every Fantrax fetch fails and nothing else in the
pipeline cares — that separation is the reason the raw layer is partitioned by
auth in the first place. `uv run fetch projections`, `market` and `identity`
keep working with a dead Fantrax cookie.

`create_session()` asserts both keys exist before doing anything and attaches
them to a `requests.Session` on domain `.fantrax.com`.

`refresh_fantrax_snapshots()` then calls `test_auth()` — a `getFantasyLeagueInfo`
request — **before** fetching anything, so a stale cookie fails immediately with
paste-these-again instructions instead of half-writing a snapshot.

Two ways expiry shows up:

- `test_auth` sees `pageError.code == "WARNING_NOT_LOGGED_IN"` → returns False.
- A data request returns **HTTP 200** with a `pageError` and no `data` block.
  `_response_data()` catches both that and a missing `responses` array, and
  both messages say to refresh `fantrax.cookies` in `config.json`.

League and team ids come from the same file: `league.fantrax_league_id`,
`league.fantrax_team_ids`. `data_prep.config` asserts the league has exactly
7 teams and that `my_team_name` is one of them.

---

## Entry point

`refresh_fantrax_snapshots(date=None, max_pool_results=None) -> (Path, Path)`

Authenticate once, then write both snapshots: rosters + player pool assembled
into `raw/fantrax`, standings into `raw/standings`. `date` defaults to today;
`max_pool_results` caps player-pool rows (`None` = 5000, the API max).
`main()` is just this with defaults, and is what the CLI calls.

| Function | Network | Returns |
|---|---|---|
| `create_session()` | no | authenticated `requests.Session` |
| `test_auth(session)` | yes | `bool` |
| `fetch_team_roster(session, team_id, owner)` | yes | `list[dict]`, one per rostered player |
| `fetch_rosters(session)` | yes | player-grain `DataFrame`, all 7 teams |
| `fetch_player_pool(session, max_results=None)` | yes | player-grain `DataFrame`, whole pool |
| `assemble_fantrax_snapshot(rosters, player_pool)` | **no** | the `raw/fantrax` frame |
| `fetch_standings(session)` | yes | team-grain `DataFrame` |
| `refresh_fantrax_snapshots(…)` | yes | both paths written |

`assemble_fantrax_snapshot` is deliberately pure — all the de-duplication logic
is testable without a live cookie.

---

## Critical: `getPlayerStats` pagination

**Do not rely on pagination parameters.** They are unreliable. Use a single
request with `maxResultsPerPage=5000` (the API max) — see `fetch_player_pool`.
The response's `paginatedResultSet.totalNumResults` is read only to print how
many players exist versus how many were fetched.

## The player pool is not filtered to free agents

It would be the obvious thing to do, and it is wrong: `getPlayerStats` returns
rostered players too, and it is the **only** source of `fantrax_score` and
`pct_rostered`. Filtering it to `is_free_agent` would silently strip those two
columns from every owned player. So the pool comes back whole and
`assemble_fantrax_snapshot` de-duplicates against the rosters instead.

## `assemble_fantrax_snapshot(rosters, player_pool)`

Not a cross-source join — both inputs are Fantrax. It is the de-duplication
that makes the snapshot one row per player.

The two fetches supply overlapping but different columns (rosters have `owner`,
`roster_status`, `status_id`, `adp`; the pool has `fantrax_score`,
`pct_rostered`, `roster_trend`, `fantrax_rank`). They are concatenated
**rosters first**, then `groupby("fantrax_id").first()`, which takes the first
*non-null* value per column. That is "roster wins field-by-field, pool fills the
gaps" in one pass — the roster endpoint being authoritative for positions and
slot.

Asserted on the way through:

- both frames non-empty (an empty fetch is a failure, not an empty league);
- `fantrax_id` present and non-null on every row of both;
- no player on two rosters;
- every column in `FANTRAX_SNAPSHOT_COLUMNS` supplied by one fetch or the other;
- `owner` non-null count **equals** `len(rosters)` — the dedup lost or merged
  nobody;
- every owned player has a `roster_status`, and no unowned player has one.

### `fantrax_id` is first-class

`fantrax_id` (the API's `scorer.scorerId`) is a real column in the snapshot and
`build.merge_fantrax` carries it through onto `players`. It used to be captured
by the parsers and then dropped by the old name-based merge, which meant the one
stable Fantrax key was thrown away at exactly the step that needed a key.

It is also the natural join key to other Fantrax-keyed sources —
`market.fetch_fantrax_adp` returns a `fantrax_id` column. (As of now
`build.merge_market` still matches the ADP source on normalized name, so the id
is preserved but not yet used there.)

### Snapshot columns

`FANTRAX_SNAPSHOT_COLUMNS`, 20 of them, asserted exactly:

```
fantrax_id  name  Position  mlb_team  player_type  age
owner  roster_status  status_id  injury_status  injury_detail
rookie  minors_eligible  eligible_positions
adp  fantrax_score  pct_rostered  roster_trend  fantrax_rank  is_free_agent
```

`owner` and `roster_status` are null for unowned players — that null **is** the
free-agent flag downstream. Names are stored exactly as Fantrax spells them.

---

## `statusId` (roster slot) — authoritative values

`ROSTER_STATUS_BY_ID`, taken from the API's own `statusTotals` block. Do **not**
guess these:

| `statusId` | `roster_status` |
|---|---|
| `1` | `active` |
| `2` | `reserve` |
| `3` | `IR` |
| `9` | `minors` |

Anything else decodes to `"unknown"`.

`statusId` is the **fantasy roster slot the manager chose**, not a real-world
injury signal: injured players are frequently left in `active` / `reserve`
slots. And every player the roster endpoint returns is **owned**, whatever slot
he sits in — a `minors` or `IR` player is still on someone's roster.

## Injury status — `scorer.icons`

Real-world injury state lives in `scorer.icons`, a list of `{tooltip, typeId}`
(catalogued from live packets across all 7 rosters):

| `typeId` | Meaning | `injury_status` |
|---|---|---|
| `1` | Day-to-Day (e.g. "Oblique - Day-to-Day"); also non-injury absences like Paternity Leave | `DTD` |
| `2` | On the Injured List (e.g. "Injured List - 10-day IL - Oblique") | `IL` |

`_parse_injury()` returns `(injury_status, injury_detail)` with **IL beating
DTD** when both icons are present; `injury_detail` is the raw Fantrax tooltip.
Both the roster and pool parsers populate these. Every other `typeId` is a
lineup / handedness / batting-order / news marker and is ignored.

---

## Request shape

```text
POST https://www.fantrax.com/fxpa/req?leagueId=<id>
Content-Type: application/json

{"msgs": [{"method": "<name>", "data": {...}}]}
```

| Method | Used by | `data` |
|---|---|---|
| `getFantasyLeagueInfo` | `test_auth` | `leagueId` |
| `getTeamRosterInfo` | `fetch_team_roster` | `leagueId`, `teamId`, `view="STATS"` |
| `getPlayerStats` | `fetch_player_pool` | `leagueId`, `maxResultsPerPage` |
| `getStandings` | `fetch_standings` | `leagueId` |

Responses are unwrapped by `_response_data(response, what)`, which reads
`responses[0].data`.

## Cell mapping

The two endpoints put different things at the same indices — the single easiest
thing to get wrong in this module.

**`getTeamRosterInfo`** (`_parse_roster_rows`): `cells[0]` = age,
`cells[2]` = ADP. Identity comes from `row.scorer`: `scorerId`, `name`,
`posShortNames`, `teamShortName`, `rookie`, `minorsEligible`, `posIds`. Slot
comes from `row.statusId`. Tables live under `tables` or `tableList`.

**`getPlayerStats`** (`_parse_pool_rows`): `cells[0]` = rank, `[1]` = status
(`"FA"` ⇒ `is_free_agent`), `[2]` = age, `[3]` = Fantrax score,
`[4]` = % rostered, `[5]` = roster trend. Rows live under `statsTable`.

`_parse_cell(cells, idx, as_float=False)` returns `None` for a short array, a
non-dict cell, empty content or `"-"`, strips `%`, and rejects anything
non-numeric rather than raising.

---

## Standings

`fetch_standings` returns one row per team: `team_id`, `team_name`,
`overall_rank`, `total_points`, plus lowercase category totals `r hr rbi sb ops
w sv k era whip` and `ab` / `ip`. This is the contract
`optimizer.banked.standings_to_banked_totals` consumes.

Parsing is exact, not heuristic. `_parse_standings_data` looks for the table
whose caption contains **"stat totals"** — that one holds the real
season-to-date values — maps its `header.cells` `shortName`s to column indices
via `_STANDINGS_SHORTNAME_TO_COL`, and reads each row's `cells` by index. Team
identity comes from `row.fixedCells` (falling back to `cells`). An earlier
title-based heuristic accidentally read the *roto-points* table for ERA and
WHIP, corrupting those rates.

`AB` and `IP` are captured as playing-time weights for ratio blending
downstream. `_parse_ip` converts baseball IP notation to decimal innings
(418.2 → 418.667, since `.1` = ⅓ and `.2` = ⅔); a plain float parse is off by up
to ~0.27 IP.

If the Stat-Totals table is absent, team rows come back without category
columns and a warning prints; the optimizer then runs rest-of-season-only.
`fetch_standings` still asserts the parsed row count **equals** the 7 configured
teams — a partial parse would bank the wrong totals.

---

## API robustness

1. Check `isinstance(x, dict)` before `.get()` on nested values.
2. Tables may appear under `tables` **or** `tableList`.
3. HTTP 200 with `pageError` present and `data` absent means expired cookies —
   handle before reading rows (`_response_data`).
4. A parse that yields 0 rows is a hard error, not an empty result. Both
   `fetch_team_roster` and `_parse_pool_rows` assert, and `raw_io.write_raw`
   refuses an empty frame outright.

---

## Downstream: matching to FanGraphs

`build.merge_fantrax` is where the snapshot meets projections. It appends the
`-H`/`-P` suffix to the Fantrax `name` from `player_type`, then cascades:
`(normalized suffixed name, mlb_team)` first, normalized name alone second.
Team disambiguation earns its place because Fantrax and FanGraphs disagree on
spelling often enough — and because two different players can share a name.

It carries over `Position`, `owner`, `roster_status`, `injury_status`,
`injury_detail`, `age`, `fantrax_score`, `pct_rostered`, `fantrax_id`, and warns
by name about any **rostered** Fantrax player with no projection match, since
that player is missing from the table entirely.

`get_player_type(position)` returns `"pitcher"` if `SP` or `RP` appears in the
position string, else `"hitter"`. It is applied at parse time to populate
`player_type` in the snapshot, and `build.merge_fantrax` falls back to calling
it on `Position` if that column is somehow absent.

`FANTRAX_NAME_CORRECTIONS` maps the handful of Fantrax strings that
normalization cannot rescue — different names entirely, not accent or suffix
variants (`"Logan OHoppe"` → `"Logan O'Hoppe"`, `"Leodalis De Vries"` →
`"Leo De Vries"`). It is intentionally **not** applied to the snapshot; the raw
layer stores names as Fantrax spells them. Note that nothing currently reads it
either — `build.py` does not apply it during reconciliation, so these two names
match on the cascade or not at all.
