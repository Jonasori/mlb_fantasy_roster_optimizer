# MLB Stats API (identity)

Player identity — MLBAM id, full name, birth date — from the public MLB Stats
API. Identity is the slowest-changing data in the system, so it is its own raw
source rather than something rebuilt on every projections refresh:

```
data/raw/identity/YYYY-MM-DD.parquet     ~never, no auth
```

**Module:** `data_prep.mlb_api`
**Fetch:** `uv run fetch identity`
**Join:** `build.merge_identity(players, identity)`

---

## API

```http
GET https://statsapi.mlb.com/api/v1/people?personIds={comma-separated-ids}
```

Reached through the `statsapi` package (`statsapi.get("people", {"personIds": …})`),
not raw `requests`.

- No auth.
- `batch_size=100` ids per request — the limit is URL length, not a documented
  page size.
- 100 ms sleep between batches (skipped after the last one).

---

## Functions

### `fetch_player_ages(mlbam_ids, batch_size=100) -> DataFrame`

The batching layer. Dedupes ids (order-preserving), walks them in batches, and
flattens the `people` array.

Columns: `mlbam_id`, `name` (the API's `fullName`), `birth_date` (`birthDate`),
`age` (the API's `currentAge`).

Asserts at least one id in, at least one row out. Warns with the first 10
offending ids if any row comes back with a null age.

### `age_from_birth_date(birth_date, on=None) -> Series`

Whole years elapsed since `birth_date`, exact at the birthday boundary. `on`
defaults to today, normalized. Nulls in, NaN out (`errors="coerce"`).

The birthday comparison is vectorized by packing month and day into an integer:

```
not_yet = (birth.month·100 + birth.day) > (on.month·100 + on.day)
age     = on.year − birth.year − not_yet
```

Returns floats, so the Series stays null-friendly.

### `fetch_identity_snapshot(mlbam_ids, batch_size=100) -> Path`

Fetches via `fetch_player_ages` and writes today's snapshot. Four columns:

| Column | Type | Source |
|---|---|---|
| `MLBAMID` | `int64` | `mlbam_id` |
| `full_name` | str | `fullName` |
| `birth_date` | str (ISO) | `birthDate` |
| `age` | float | **derived** from `birth_date` |

**`age` is derived, not stored from the API.** `fetch_player_ages` does return
the API's `currentAge`, and `fetch_identity_snapshot` throws it away in favour
of `age_from_birth_date(birth_date)`. A stored integer age silently rots: the
snapshot is refreshed roughly never, so a `currentAge` captured last spring
would keep reading as gospel a year later, and nothing downstream could tell.
`birth_date` is the durable field; `age` stays in the frame only because
`build.py` reads it.

Refuses to write if *every* row is missing a birth date — that means the
`people` response shape changed, and a snapshot with no durable field would be
worse than none. Prints the row count, how many have a birth date, and the age
range.

---

## Join

`build.merge_identity` matches on **`MLBAMID` only** — no name fallback. It is
MLB's own id, present on both sides (FanGraphs exports it as `xMLBAMID`), so
there is nothing a name pass could add. Both sides are stringified through
`str(int(v))` first so an `Int64` column and an `int64` column compare equal.

`birth_date` and `age` are filled where the identity snapshot has a value,
falling back to what is already on the row. In practice that **overwrites the
age Fantrax supplied**, deliberately: MLB is authoritative on it.

Matched count is printed. The whole step is skipped when no identity snapshot
exists (`include_identity=False`, or an empty `data/raw/identity/`), and the
table simply has no `birth_date` / `age` from MLB.

---

## Known gap

`uv run fetch identity` currently raises
`TypeError: fetch_identity_snapshot() missing 1 required positional argument:
'mlbam_ids'` — `cli.cmd_identity` calls it with no arguments, and the function
has no way to work out which ids to fetch on its own. Until that is wired up
(the natural source is the non-null `MLBAMID`s in the latest projections
snapshot), refresh identity from Python:

```python
from data_prep.mlb_api import fetch_identity_snapshot
from data_prep.raw_io import read_latest_raw

raw, _ = read_latest_raw("projections/atc")
fetch_identity_snapshot(raw["MLBAMID"].dropna().astype(int).tolist())
```
