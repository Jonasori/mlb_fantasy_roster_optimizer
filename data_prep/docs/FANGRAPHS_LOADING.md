# FanGraphs projections

FanGraphs is the source of projected stats, and the only **load-bearing** source
in the pipeline: a missing Fantrax or market snapshot degrades a column, a
missing projections snapshot is a hard error.

Two steps, in two layers:

```
scrape_fangraphs.scrape_projections()      ->  data/raw/projections/{steamer,atc}/YYYY-MM-DD.parquet
build.prepare_projections(raw)             ->  the `players` base frame
build.apply_volume_floors(players)         ->  after the Fantrax merge (needs ownership)
```

**Fetch:** `uv run fetch projections` (or `uv run python -m data_prep.scrape_fangraphs`).

There are no CSVs and no download folder any more. Nothing is read from disk —
`scrape_projections` calls the internal JSON API and writes the snapshot itself.

---

## Feeds

`PROJECTION_TYPES` maps our system name to the FanGraphs API `type` param:

| System | API `type` | What it is |
|---|---|---|
| `steamer` | `steamerr` | Steamer, **rest-of-season** |
| `atc` | `ratcdc` | ATC in-season **DC** variant |

These are the in-season rest-of-season feeds, not the preseason-frozen
`steamer` / `atc` feeds (those never update once the season starts). FanGraphs
prefixes RoS feeds with `r`. `ratcdc` is used because the full-season `atcdc`
endpoint returns HTTP 500 — it is the only working updated ATC feed.

Consequence for anything downstream: `PA` and `IP` are **remaining** playing
time, not full-season. A player already shut down for the year projects ~0.

Two requests per system, `stats=bat` and `stats=pit`, `pos=all`.

## Auth

Browser cookies, read straight out of a logged-in browser by `browser_cookie3`
— nothing to paste, nothing in `config.json`. `get_fangraphs_session()` probes
Brave, Chrome, Edge, Vivaldi, Opera, Firefox, Safari in that order and takes the
first one carrying a `wordpress_logged_in` cookie for `.fangraphs.com`.

The rest-of-season feeds require a paid FanGraphs **membership**; a free account
authenticates but cannot read `steamerr` / `ratcdc`. Two failure modes the
assertion message spells out: Chrome's cookies are keychain-encrypted (run from
a GUI terminal and approve the prompt — an SSH session cannot decrypt them), and
Safari's need Full Disk Access for whatever is running Python.

---

## Raw snapshot contract

`build_snapshot()` concatenates the two per-type frames from
`build_type_frame()` into **19 columns**, asserted exactly:

| Group | Columns |
|---|---|
| Shared (5) | `Name`, `Team`, `player_type`, `PlayerId`, `MLBAMID` |
| Hitter (8) | `PA`, `AB`, `R`, `HR`, `RBI`, `SB`, `OPS`, `WAR` |
| Pitcher (7) | `IP`, `W`, `SV`, `SO`, `ERA`, `WHIP`, `WAR` |

`WAR` is in both lists, so 5 + 8 + 7 − 1 = 19.

Raw-layer discipline, all of it deliberate and all of it `build.py`'s problem
instead:

- Each row carries values only for its own `player_type`; the opposite type's
  stat columns are **NULL, not zero**. Zero-filling is a join-step decision.
- `Name` is **unsuffixed** — no `-H` / `-P`.
- `SO` stays `SO`. It is not yet renamed to the league's category name `K`.
- Nothing is merged in: no positions, no ages, no market value.
- `MLBAMID` is a nullable `Int64` so it never stringifies as `"677951.0"`.
- `PlayerId` is a `str`, because market data joins on it as a string.
- API field renames are the only rewriting done: `PlayerName`→`Name`,
  `xMLBAMID`→`MLBAMID`, `playerids`→`PlayerId` (`API_RENAMES`). The stat keys
  already match.

### Why the columns are selected per type

This is the load-bearing reason the snapshot is a ~19-column subset of the
~74-column FanGraphs export rather than the two frames concatenated whole:
**the hitter and pitcher feeds reuse stat keys with opposite meanings.**

| Key | In the `bat` feed | In the `pit` feed |
|---|---|---|
| `R` | runs **scored** | runs **allowed** |
| `HR` | home runs **hit** | home runs **allowed** |
| `SO` | strikeouts **taken** | strikeouts **recorded** |

A naive `pd.concat` of the full frames would stack those into one column each,
silently, with no error anywhere. Restricting each side to its own scoring
categories is what makes the shared columns genuinely shared.

### Fetch-time validation

- Every expected column must be present, per feed. A missing one is a hard
  error naming the field, not a silently empty column — if FanGraphs renames
  something, the failure surfaces at the rename, not 200 lines downstream.
- Zero null `PlayerId`s. It is the only join key to Ottoneu market values, so a
  null there would silently drop the player from price matching.
- Both `player_type` values present in the snapshot (catches one feed returning
  empty).
- `raw_io.write_raw` refuses to write an empty frame at all.

---

## `build.prepare_projections(raw)`

Turns one raw snapshot into the base `players` frame. Requires `Name`, `Team`,
`player_type`, `PlayerId`, `MLBAMID`, `WAR`.

| Step | Detail |
|---|---|
| Split | on `player_type`; asserts both sides are non-empty |
| Rename | pitcher `SO` → **`K`**, the league's category name |
| Suffix | `Name` + `-H` for hitters, `-P` for pitchers |
| Dedup | duplicate suffixed `Name`, `keep="first"`, with a printed count |
| Zero-fill | hitters get `PITCHING_STATS` = 0.0, pitchers get `HITTING_STATS` = 0.0 |
| `Team` | NaN → `"FA"` (FanGraphs blanks it for unsigned players) |
| `WAR`, `AB` | NaN → 0.0 |
| Placeholders | `Position`, `owner`, `roster_status`, `injury_status`, `injury_detail`, `age`, `fantrax_score`, `pct_rostered` set to `None` for the merges to fill |

The suffix is what keeps a two-way player's two sides distinct as separate rows
all the way through the pipeline. `normalize_name` preserves it on purpose (see
`data_prep/names.py`), so a hitter row can never match a pitcher row.

Zero-filling is why the MEW formula needs no hitter/pitcher branching: all 12
scoring stats (`PA R HR RBI SB OPS` / `IP W SV K ERA WHIP`) are present and
non-null on every row, asserted before return.

`AB` is carried through purely so `apply_volume_floors` can use it later; it is
not a scoring stat.

Note `Position` starts as `None`, not a `DH`/`RP` placeholder. Position
eligibility comes from Fantrax or it is missing.

## `build.apply_volume_floors(players)`

Drops negligible-volume players — `AB < MIN_AB` (10) for hitters, `IP < MIN_IP`
(5) for pitchers — **from free agents only**. Anyone with a non-null `owner` is
exempt.

The floor exists to trim the free-agent pool: a player projected for almost no
remaining playing time is not a realistic pickup, but he does drag the z-score
population FV is computed against.

The free-agents-only carve-out is not a nicety. It is why this runs **after**
the Fantrax merge rather than at load time: ownership is what decides who is
exempt. A rostered minor-league prospect's rest-of-season *MLB* projection is
near zero by definition, so an unconditional floor deleted the entire farm
system — exactly the dynasty assets worth reasoning about — and did it quietly.

Both counts are printed: how many free agents were dropped, and how many
rostered players were kept below the floor.

---

## Switching systems

Both systems are scraped on every `uv run fetch projections`, into separate
directories. Choosing one is a read, not a re-fetch:

```bash
uv run fetch build                   # atc (default)
uv run fetch build --system steamer  # same snapshots, different projections
```

`build_players(system=..., on_or_before=date(...))` does the same from Python,
and `on_or_before` reproduces a past day from the snapshots that existed then.
