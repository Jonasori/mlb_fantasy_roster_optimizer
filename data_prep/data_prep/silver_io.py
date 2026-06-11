"""Write the silver `players` table to CSV or Parquet."""

from pathlib import Path

import pandas as pd


def write_silver_table(players: pd.DataFrame, path: str | Path) -> Path:
    """
    Persist the silver table for downstream optimizers.

    Format is chosen by file extension: ``.csv`` or ``.parquet``.

    Args:
        players: Silver DataFrame (Name as column, not index).
        path: Output file path.

    Returns:
        Resolved path that was written.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    assert suffix in (".csv", ".parquet"), (
        f"Unsupported silver table extension {suffix!r}; use .csv or .parquet"
    )
    if suffix == ".csv":
        players.to_csv(out, index=False)
    else:
        players.to_parquet(out, index=False)
    print(f"Wrote silver table: {out} ({len(players)} rows)")
    return out.resolve()
