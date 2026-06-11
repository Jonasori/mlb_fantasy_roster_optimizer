# MLB Stats API (ages)

Ported from v1 `implementation_specs …/01f_mlb_stats_api.md`, updated for `data_prep`.

## Overview

Fetches **current age** for MLBAM IDs via the public Stats API. Used by **`data_prep.build.merge_mlb_ages`** after the silver table is built from CSVs + Fantrax.

**Module:** `data_prep.mlb_api`  
**Function:** `fetch_player_ages(mlbam_ids) -> DataFrame`

---

## API

```http
GET https://statsapi.mlb.com/api/v1/people?personIds={comma-separated-ids}
```

- Batch size 100 IDs per request (URI length).
- No auth.
- 100ms delay between batches in the implementation.

---

## Return value

DataFrame columns: `mlbam_id`, `name`, `birth_date`, `age`.

Some players may have missing `age` in the API response; those rows are skipped when merging onto `players`.

---

## Usage in pipeline

`build_silver_table(..., skip_mlb_api=False)` collects non-null `MLBAMID` from the combined projections, calls `fetch_player_ages`, then fills `players["age"]` where MLBAMID matches. Fantrax-supplied ages on roster/pool rows may already be present; MLB pass updates/fills by ID.
