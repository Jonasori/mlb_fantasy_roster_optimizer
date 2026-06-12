"""
Build the silver `players` DataFrame: FanGraphs CSVs + Fantrax + MLB ages.

No fantasy value (FV), lineup assignment, or gold-table enrichment.
"""

import pandas as pd

from .config import DATA_DIR
from .fantrax_api import (
    FANTRAX_NAME_CORRECTIONS,
    create_session,
    fetch_all_fantrax_data,
    get_player_type,
    test_auth,
)
from .load_projections import load_hitter_projections, load_pitcher_projections
from .mlb_api import fetch_player_ages
from .names import normalize_name


def load_fangraphs(hitter_csv: str, pitcher_csv: str) -> pd.DataFrame:
    """Load FanGraphs CSVs and combine into one silver-stage DataFrame.

    Does not compute FV or other gold-table columns.
    """
    hitters = load_hitter_projections(hitter_csv)
    pitchers = load_pitcher_projections(pitcher_csv)

    hitting_stat_cols = ["PA", "R", "HR", "RBI", "SB", "OPS"]
    pitching_stat_cols = ["IP", "W", "SV", "K", "ERA", "WHIP"]

    for col in pitching_stat_cols:
        hitters[col] = 0.0
    for col in hitting_stat_cols:
        pitchers[col] = 0.0

    all_cols = (
        ["Name", "Team", "Position", "player_type"]
        + hitting_stat_cols
        + pitching_stat_cols
        + ["WAR", "MLBAMID"]
    )
    combined = pd.concat([hitters[all_cols], pitchers[all_cols]], ignore_index=True)

    combined["owner"] = None
    combined["roster_status"] = None
    combined["injury_status"] = None
    combined["injury_detail"] = None
    combined["age"] = None
    combined["fantrax_score"] = None
    combined["pct_rostered"] = None

    print(
        f"Combined projections: {len(combined)} players "
        f"({len(hitters)} hitters, {len(pitchers)} pitchers)"
    )
    return combined


def _build_name_lookup(players: pd.DataFrame) -> dict:
    """Map (normalized_name, team) -> projection name, with name-only fallback."""
    lookup = {}
    for name, team in zip(players["Name"], players["Team"]):
        norm = normalize_name(name)
        lookup[(norm, team)] = name
        lookup.setdefault(norm, name)
    return lookup


def merge_fantrax(players: pd.DataFrame, fantrax_data: dict) -> pd.DataFrame:
    """Merge Fantrax ownership, positions, and ages into players DataFrame.

    Player pool positions are applied first (broad coverage),
    then roster detail positions overwrite (authoritative for rostered players).
    """
    players = players.copy()
    lookup = _build_name_lookup(players)

    player_pool = fantrax_data["player_pool"]
    if len(player_pool) > 0:
        pool_updated = 0
        for _, row in player_pool.iterrows():
            raw_name = row.get("name")
            if raw_name is None or pd.isna(raw_name):
                continue
            if raw_name in FANTRAX_NAME_CORRECTIONS:
                raw_name = FANTRAX_NAME_CORRECTIONS[raw_name]

            position = row.get("position", "")
            player_type = row.get("player_type", get_player_type(str(position)))
            suffix = "-P" if player_type == "pitcher" else "-H"
            suffixed = raw_name + suffix

            mlb_team = row.get("mlb_team", "")
            norm = normalize_name(suffixed)
            proj_name = lookup.get((norm, mlb_team)) or lookup.get(norm)
            if proj_name is None:
                continue

            mask = players["Name"] == proj_name

            if position and not pd.isna(position) and position != "":
                players.loc[mask, "Position"] = position
                pool_updated += 1

            for col in ("fantrax_score", "pct_rostered"):
                val = row[col]
                if val is not None and not pd.isna(val):
                    players.loc[mask, col] = val

            injury_status = row.get("injury_status")
            if injury_status is not None and not pd.isna(injury_status):
                players.loc[mask, "injury_status"] = injury_status
                players.loc[mask, "injury_detail"] = row.get("injury_detail")

        print(f"Merged positions for {pool_updated} players from player pool")

    roster_sets = fantrax_data["roster_sets"]
    roster_details = fantrax_data["rosters"]

    player_info: dict[str, dict] = {}
    for team_name, team_players in roster_details.items():
        for player in team_players:
            raw_name = player["name"]
            if raw_name is None:
                continue
            if raw_name in FANTRAX_NAME_CORRECTIONS:
                raw_name = FANTRAX_NAME_CORRECTIONS[raw_name]

            if raw_name.endswith("-H") or raw_name.endswith("-P"):
                suffixed = raw_name
            else:
                suffix = "-P" if player["player_type"] == "pitcher" else "-H"
                suffixed = raw_name + suffix

            status_id = str(player.get("status_id", ""))
            # Authoritative slot meanings from the API's own `statusTotals`
            # (Active / Reserve / Inj Res / Minors). Do NOT guess these.
            status = {
                "1": "active",
                "2": "reserve",
                "3": "IR",
                "9": "minors",
            }.get(status_id, "unknown")
            player_info[suffixed] = {
                "age": player.get("age"),
                "status": status,
                "status_id": status_id,
                "injury_status": player.get("injury_status"),
                "injury_detail": player.get("injury_detail"),
                "position": player.get("position"),
                "mlb_team": player.get("team"),
            }

    not_found = []
    for team_name, names in roster_sets.items():
        for fantrax_name in names:
            info = player_info.get(fantrax_name, {})
            normalized = normalize_name(fantrax_name)
            mlb_team = info.get("mlb_team", "")
            proj_name = lookup.get((normalized, mlb_team)) or lookup.get(normalized)

            if proj_name is None:
                not_found.append(fantrax_name)
                continue

            mask = players["Name"] == proj_name
            players.loc[mask, "owner"] = team_name
            players.loc[mask, "roster_status"] = info.get("status")
            players.loc[mask, "injury_status"] = info.get("injury_status")
            players.loc[mask, "injury_detail"] = info.get("injury_detail")
            if info.get("age") is not None:
                players.loc[mask, "age"] = info["age"]
            if info.get("position"):
                players.loc[mask, "Position"] = info["position"]

    if not_found:
        print(f"WARNING: {len(not_found)} rostered players not found in projections:")
        for name in sorted(not_found)[:10]:
            print(f"  - {name}")
        if len(not_found) > 10:
            print(f"  ... and {len(not_found) - 10} more")

    rostered = players["owner"].notna().sum()
    print(
        f"Merged Fantrax data: {rostered} rostered players across "
        f"{len(roster_sets)} teams"
    )
    return players


def merge_mlb_ages(players: pd.DataFrame) -> pd.DataFrame:
    """Fetch and merge ages from MLB Stats API into the age column."""
    players = players.copy()
    mlbam_ids = players["MLBAMID"].dropna().astype(int).tolist()

    if len(mlbam_ids) == 0:
        print("No MLBAM IDs — skipping age fetch")
        return players

    ages_df = fetch_player_ages(mlbam_ids)

    updated = 0
    for _, row in ages_df.iterrows():
        mlbam_id = row.get("mlbam_id")
        age = row.get("age")
        if mlbam_id is None or pd.isna(mlbam_id):
            continue
        if age is None or pd.isna(age):
            continue
        mask = players["MLBAMID"] == int(mlbam_id)
        if mask.any():
            players.loc[mask, "age"] = int(age)
            updated += 1

    print(f"Merged ages for {updated} players from MLB Stats API")
    return players


def build_silver_table(
    hitter_proj_path: str,
    pitcher_proj_path: str,
    skip_mlb_api: bool = False,
) -> pd.DataFrame:
    """Build the silver players table (projections + Fantrax + ages).

    Steps: load CSVs -> merge Fantrax -> merge MLB ages.

    Returns:
        DataFrame with one row per player; Name is a column (not the index).
        Matches the v2 silver table input contract (no FV, optimal_slot, etc.).
    """
    print("=== Building silver players table ===")

    print("\n1. Loading FanGraphs projections...")
    players = load_fangraphs(hitter_proj_path, pitcher_proj_path)

    print("\n2. Fetching and merging Fantrax data...")
    session = create_session()
    assert test_auth(session), (
        "Fantrax authentication failed — update cookies in config.json"
    )
    fantrax_data = fetch_all_fantrax_data(session)
    players = merge_fantrax(players, fantrax_data)

    # Persist standings (banked YTD category totals) next to the silver table.
    # The optimizer reads this to add the banked half of season totals — see
    # optimizer/banked.py. Written even if category parsing is incomplete; the
    # consumer validates and falls back to rest-of-season-only if so.
    standings = fantrax_data.get("standings")
    if standings is not None and len(standings) > 0:
        standings_path = DATA_DIR / "standings.parquet"
        standings.to_parquet(standings_path, index=False)
        print(f"Wrote standings: {standings_path} ({len(standings)} teams)")

    if not skip_mlb_api:
        print("\n3. Fetching ages from MLB Stats API...")
        players = merge_mlb_ages(players)
    else:
        print("\n3. Skipping MLB API ages")

    print(f"\n=== Silver table complete: {len(players)} players ===")
    return players
