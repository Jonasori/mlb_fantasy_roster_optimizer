# FanGraphs projection loading

Ported from v1 `implementation_specs …/01b_fangraphs_loading.md`, trimmed to what **`data_prep.load_projections`** implements.

## Overview

FanGraphs CSVs are the source of projected stats. Fantrax supplies roster ownership and positions after merge.

**Module:** `data_prep.load_projections` — `load_hitter_projections`, `load_pitcher_projections`  
**Combined merge:** `data_prep.build.load_fangraphs`

---

## Why FanGraphs (for this layer)

| Data | FanGraphs | Fantrax |
|------|-----------|---------|
| PA / IP | Yes | Not used for silver volume |
| WAR | Yes | No |
| MLBAMID | Yes | No |
| Full player universe | ~1500 rows | Pool + rosters |

---

## CSV expectations

### Hitters

| Column | Notes |
|--------|--------|
| `Name` | Becomes `Name` + `-H` |
| `Team` | NaN → `FA` |
| `PA`, `R`, `HR`, `RBI`, `SB`, `OPS` | Required; **OPS from file, do not recompute** |
| `WAR` | Optional in file; else 0 |
| `MLBAMID` | Optional |

Initial `Position` is placeholder `DH`; Fantrax merge overwrites where matched.

### Pitchers

| Column | Notes |
|--------|--------|
| `Name` | Becomes `Name` + `-P` |
| `Team` | NaN → `FA` |
| `SO` | Renamed to **`K`** on load |
| `IP`, `W`, `SV`, `K`, `ERA`, `WHIP` | Required |
| `WAR`, `MLBAMID` | As hitters |

Initial `Position` is `RP`; Fantrax merge overwrites.

### Duplicates

Duplicate `Name` rows (after suffix) are dropped with `keep="first"` and a printed note.

---

## Validation

After load:

- Required columns exist and have no NaNs (per loader asserts).
- Names end with `-H` or `-P`.
- Pitcher CSV supplies `K` after rename from `SO`.

---

## Out of scope in `data_prep`

The original spec also described SGP, optimal lineup MILP, and team totals. Those live in the **optimizer** packages (v1/v2), not in `data_prep`.
