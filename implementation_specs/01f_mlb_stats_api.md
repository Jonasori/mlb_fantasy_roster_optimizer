# MLB Stats API Integration

## Overview

Fetches player ages from the MLB Stats API.

**Module:** `optimizer/mlb_api.py`  
**Function:** `fetch_player_ages(mlbam_ids) -> DataFrame`

**Used by:** `refresh_all_data()` in [01d_database.md](01d_database.md)

---

## API

```
GET https://statsapi.mlb.com/api/v1/people?personIds={comma-separated-ids}
```

- Max ~100 IDs per request (URI length limit)
- No authentication required
- Use 100ms delay between batches

---

## Implementation

```python
"""MLB Stats API integration for player ages."""

import time

import pandas as pd
import statsapi


def fetch_player_ages(mlbam_ids: list[int], batch_size: int = 100) -> pd.DataFrame:
    """
    Fetch player ages from MLB Stats API.
    
    Args:
        mlbam_ids: List of MLBAM IDs to fetch
        batch_size: IDs per request (default 100)
    
    Returns:
        DataFrame with columns: mlbam_id, name, birth_date, age
    """
    assert len(mlbam_ids) > 0, "Must provide at least one MLBAM ID"
    
    unique_ids = list(dict.fromkeys(mlbam_ids))  # dedupe
    
    print(f"Fetching ages for {len(unique_ids)} players from MLB Stats API...")
    start = time.time()
    
    results = []
    for i in range(0, len(unique_ids), batch_size):
        batch = unique_ids[i:i + batch_size]
        ids_str = ','.join(str(id) for id in batch)
        
        data = statsapi.get('people', {'personIds': ids_str})
        
        for person in data.get('people', []):
            results.append({
                'mlbam_id': person['id'],
                'name': person.get('fullName', ''),
                'birth_date': person.get('birthDate'),
                'age': person.get('currentAge'),
            })
        
        if i + batch_size < len(unique_ids):
            time.sleep(0.1)
    
    print(f"  Fetched {len(results)} players in {time.time() - start:.1f}s")
    
    df = pd.DataFrame(results)
    assert len(df) > 0, "No ages returned from API"
    assert df['age'].notna().all(), "Some players missing age"
    
    return df
```

---

## Dependency

```toml
# pyproject.toml (already added)
dependencies = ["MLB-StatsAPI>=1.9.0"]
```
