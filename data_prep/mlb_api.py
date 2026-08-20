"""
MLB Stats API integration: player identity (MLBAM id, birth date, name).

Identity is the slowest-changing data in the system, so it is its own raw
source (`data/raw/identity/<YYYY-MM-DD>.parquet`) rather than something rebuilt
on every projections refresh.
"""

import time
from pathlib import Path

import pandas as pd
import statsapi

from . import raw_io


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
        batch = unique_ids[i : i + batch_size]
        ids_str = ",".join(str(id) for id in batch)

        data = statsapi.get("people", {"personIds": ids_str})

        for person in data.get("people", []):
            results.append(
                {
                    "mlbam_id": person["id"],
                    "name": person.get("fullName", ""),
                    "birth_date": person.get("birthDate"),
                    "age": person.get("currentAge"),
                }
            )

        if i + batch_size < len(unique_ids):
            time.sleep(0.1)

    print(f"  Fetched {len(results)} players in {time.time() - start:.1f}s")

    df = pd.DataFrame(results)
    assert len(df) > 0, "No ages returned from API"

    missing_age_count = df["age"].isna().sum()
    if missing_age_count > 0:
        missing_ids = df[df["age"].isna()]["mlbam_id"].tolist()
        print(
            f"  Warning: {missing_age_count} players missing age data (MLBAM IDs: {missing_ids[:10]}{'...' if len(missing_ids) > 10 else ''})"
        )

    return df


def age_from_birth_date(birth_date: pd.Series, on: pd.Timestamp | None = None) -> pd.Series:
    """
    Whole years elapsed since `birth_date`, exact at the birthday boundary.

    Args:
        birth_date: ISO date strings or datetimes. Nulls give NaN.
        on: Reference date. Defaults to today.

    Returns:
        Float Series (nullable-friendly) of ages in whole years.
    """
    if on is None:
        on = pd.Timestamp.today().normalize()
    birth = pd.to_datetime(birth_date, errors="coerce")
    # Subtract one year when the birthday has not yet passed this year;
    # MMDD as an integer makes that comparison vectorizable.
    not_yet = (birth.dt.month * 100 + birth.dt.day) > (on.month * 100 + on.day)
    return (on.year - birth.dt.year - not_yet).astype(float)


def fetch_identity_snapshot(mlbam_ids: list[int], batch_size: int = 100) -> Path:
    """
    Fetch player identity from MLB Stats API and write today's raw snapshot.

    Args:
        mlbam_ids: MLBAM IDs to look up. Deduped by the underlying fetch.
        batch_size: IDs per request; the API has a URL-length limit.

    Returns:
        Path written, `data/raw/identity/<YYYY-MM-DD>.parquet`.

    Note:
        `birth_date` is the durable field and `age` is derived from it at
        snapshot time, not taken from the API's `currentAge` — a stored integer
        age silently rots. `age` stays in the frame because build.py reads it.
    """
    people = fetch_player_ages(mlbam_ids, batch_size=batch_size)
    identity = pd.DataFrame(
        {
            "MLBAMID": people["mlbam_id"].astype("int64"),
            "full_name": people["name"],
            "birth_date": people["birth_date"],
            "age": age_from_birth_date(people["birth_date"]),
        }
    )

    missing_birth = identity["birth_date"].isna().sum()
    assert missing_birth < len(identity), (
        f"All {len(identity)} identity rows are missing birthDate — the "
        f"MLB Stats API 'people' response shape has changed. Refusing to write "
        f"a snapshot with no durable field."
    )
    print(
        f"  Identity: {len(identity)} players, "
        f"{len(identity) - missing_birth} with a birth date, "
        f"ages {identity['age'].min():.0f}-{identity['age'].max():.0f}"
    )
    return raw_io.write_raw(identity, "identity")
