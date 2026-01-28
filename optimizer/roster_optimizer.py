"""
Dynasty Roto Roster Optimizer - Core optimization logic.

All rostered players' stats count toward team totals (not just starters).
Starting lineup slots exist only to enforce positional requirements.
"""

import sqlite3
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import pulp
from tqdm.auto import tqdm

# === LEAGUE CONFIGURATION ===

# Scoring categories
HITTING_CATEGORIES = ["R", "HR", "RBI", "SB", "OPS"]
PITCHING_CATEGORIES = ["W", "SV", "K", "ERA", "WHIP"]
ALL_CATEGORIES = HITTING_CATEGORIES + PITCHING_CATEGORIES

# Categories where lower is better
NEGATIVE_CATEGORIES = {"ERA", "WHIP"}

# Ratio stats and their weighting denominators
# Key = category, Value = column used for weighting
RATIO_STATS = {
    "OPS": "PA",  # Team OPS = sum(PA_i * OPS_i) / sum(PA_i)
    "ERA": "IP",  # Team ERA = sum(IP_i * ERA_i) / sum(IP_i)
    "WHIP": "IP",  # Team WHIP = sum(IP_i * WHIP_i) / sum(IP_i)
}

# Roster construction
ROSTER_SIZE = 26  # Active roster only (excludes 15 prospect slots and 1 IR slot)

# Starting lineup slots
# Maps slot type -> number of slots of that type
# NOTE: These slots enforce positional validity only. ALL rostered players
# contribute stats, not just starters. The slot constraints ensure the roster
# CAN field a valid starting lineup.
HITTING_SLOTS = {
    "C": 1,
    "1B": 1,
    "2B": 1,
    "SS": 1,
    "3B": 1,
    "OF": 3,
    "UTIL": 1,
}  # Total: 9 hitting slots

PITCHING_SLOTS = {
    "SP": 5,
    "RP": 2,
}  # Total: 7 pitching slots

# Total starting slots: 16
# Remaining roster spots (26 - 16 = 10) are bench
# Bench players' stats count equally toward team totals

# Position eligibility: which player positions can fill which lineup slots
# Note: Database already consolidates LF/CF/RF into 'OF', so we only see 'OF' in practice.
# DH players can ONLY fill UTIL slots (not position-specific slots).
SLOT_ELIGIBILITY = {
    "C": {"C"},
    "1B": {"1B"},
    "2B": {"2B"},
    "SS": {"SS"},
    "3B": {"3B"},
    "OF": {"OF"},  # Database already normalizes LF/CF/RF to OF
    "UTIL": {"C", "1B", "2B", "SS", "3B", "OF", "DH"},  # Any hitter position
    "SP": {"SP"},
    "RP": {"RP"},
}

# League structure
NUM_OPPONENTS = 6  # Hardcoded: 7-team league (me + 6 opponents)

# Roster composition bounds (for full 26-man roster, not just starters)
MIN_HITTERS = 12
MAX_HITTERS = 16
MIN_PITCHERS = 10
MAX_PITCHERS = 14

# MILP Constants
EPSILON_COUNTING = 0.5  # For integer-valued stats (ensures strict inequality)
EPSILON_RATIO = 0.001  # For continuous-valued ratio stats
BIG_M_COUNTING = 10000  # Big-M for counting stats
BIG_M_RATIO = 5000  # Big-M for ratio stat linearized forms

# Team abbreviation normalization (Fantrax uses short forms, FanGraphs uses longer)
# Maps common short forms to FanGraphs standard abbreviations
TEAM_ABBREVIATION_MAP = {
    "ARI": "ARI",
    "ATL": "ATL",
    "BAL": "BAL",
    "BOS": "BOS",
    "CHC": "CHC",
    "CHW": "CHW",
    "CWS": "CHW",
    "CIN": "CIN",
    "CLE": "CLE",
    "COL": "COL",
    "DET": "DET",
    "HOU": "HOU",
    "KC": "KCR",
    "KCR": "KCR",
    "LAA": "LAA",
    "LAD": "LAD",
    "MIA": "MIA",
    "MIL": "MIL",
    "MIN": "MIN",
    "NYM": "NYM",
    "NYY": "NYY",
    "OAK": "OAK",
    "PHI": "PHI",
    "PIT": "PIT",
    "SD": "SDP",
    "SDP": "SDP",  # San Diego: SD → SDP
    "SF": "SFG",
    "SFG": "SFG",  # San Francisco: SF → SFG
    "SEA": "SEA",
    "STL": "STL",
    "TB": "TBR",
    "TBR": "TBR",  # Tampa Bay: TB → TBR
    "TEX": "TEX",
    "TOR": "TOR",
    "WSH": "WSH",
    "WAS": "WSH",
}


def _normalize_team_abbreviation(team: str) -> str:
    """Normalize team abbreviation to FanGraphs standard form."""
    if not team:
        return ""
    team_upper = team.upper().strip()
    return TEAM_ABBREVIATION_MAP.get(team_upper, team_upper)


# === FANTRAX ROSTER CONVERSION ===

# Status mapping from Fantrax format to pipeline format
FANTRAX_STATUS_MAP = {
    "Act": "active",  # Active/starting player
    "Res": "active",  # Reserve/bench (still counts toward 26-man roster)
    "Min": "prospect",  # Minor league player
    "IR": "IR",  # Injured reserve
}


def parse_fantrax_roster(filepath: str) -> pd.DataFrame:
    """
    Parse a single Fantrax roster CSV file.

    Fantrax exports have a specific format:
    - Separate "Hitting" and "Pitching" sections
    - Header row: "", "Hitting" or "", "Pitching"
    - Column headers on the next row
    - Player data follows

    Args:
        filepath: Path to the Fantrax roster CSV file

    Returns:
        DataFrame with columns: name, team, position, fantrax_status, status, player_type
        where status is mapped to pipeline format (active, prospect, IR)
    """
    # Read raw CSV to parse the multi-section format
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find section boundaries
    hitting_start = None
    pitching_start = None

    for i, line in enumerate(lines):
        # Look for section headers (empty cell, section name)
        stripped = line.strip().strip('"')
        if stripped.startswith(",") or stripped == "":
            # Check if next part indicates section
            parts = line.strip().split(",")
            if len(parts) >= 2:
                section = parts[1].strip().strip('"')
                if section == "Hitting":
                    hitting_start = i
                elif section == "Pitching":
                    pitching_start = i

    assert hitting_start is not None, f"Could not find 'Hitting' section in {filepath}"
    assert pitching_start is not None, (
        f"Could not find 'Pitching' section in {filepath}"
    )

    # Parse hitting section (skip section header, read from column headers)
    hitting_lines = lines[hitting_start + 1 : pitching_start]
    hitting_df = pd.read_csv(
        pd.io.common.StringIO("".join(hitting_lines)), quotechar='"'
    )
    hitting_df["player_type"] = "hitter"

    # Parse pitching section
    pitching_lines = lines[pitching_start + 1 :]
    pitching_df = pd.read_csv(
        pd.io.common.StringIO("".join(pitching_lines)), quotechar='"'
    )
    pitching_df["player_type"] = "pitcher"

    # Combine and extract relevant columns
    combined = pd.concat([hitting_df, pitching_df], ignore_index=True)

    # Validate required columns exist
    assert "Player" in combined.columns, f"Missing 'Player' column in {filepath}"
    assert "Status" in combined.columns, f"Missing 'Status' column in {filepath}"
    assert "Team" in combined.columns, f"Missing 'Team' column in {filepath}"

    # Map status values
    combined["fantrax_status"] = combined["Status"]
    combined["status"] = combined["Status"].map(FANTRAX_STATUS_MAP)

    # Check for unmapped status values
    unmapped = combined[combined["status"].isna()]["Status"].unique()
    assert len(unmapped) == 0, (
        f"Unknown Fantrax status values in {filepath}: {list(unmapped)}. "
        f"Expected one of: {list(FANTRAX_STATUS_MAP.keys())}"
    )

    # Extract position from Pos column if available, otherwise use Eligible
    if "Pos" in combined.columns:
        combined["position"] = combined["Pos"]
    elif "Eligible" in combined.columns:
        combined["position"] = combined["Eligible"]
    else:
        combined["position"] = ""

    # Rename and select columns - now including team and position
    result = combined[
        ["Player", "Team", "position", "fantrax_status", "status", "player_type"]
    ].copy()
    result = result.rename(columns={"Player": "name", "Team": "team"})

    # Clean team column - handle "(N/A)" as empty
    result["team"] = result["team"].replace("(N/A)", "")

    # Remove empty rows (some CSVs have trailing empty rows)
    result = result[result["name"].notna() & (result["name"] != "")]

    # Append '-H' or '-P' suffix to ALL player names for global uniqueness
    # This matches what we do in projections: "Luis Castillo-P" vs "Luis Castillo-H"
    # Fantrax already uses this convention for Ohtani (-H, -P), so this is consistent
    def add_suffix(row):
        name = row["name"]
        # If already has suffix (Ohtani-H, Ohtani-P from Fantrax), keep it
        if name.endswith("-H") or name.endswith("-P"):
            return name
        # Otherwise add suffix based on player_type
        suffix = "-H" if row["player_type"] == "hitter" else "-P"
        return name + suffix

    result["name"] = result.apply(add_suffix, axis=1)

    return result


def convert_fantrax_rosters(
    my_team_path: str,
    opponent_team_paths: dict[int, str],
    output_dir: str,
) -> tuple[str, str]:
    """
    Convert Fantrax roster exports to the pipeline's expected format.

    This function reads raw Fantrax roster CSVs and writes two output files:
    - my-roster.csv: My team's roster in name,status format
    - opponent-rosters.csv: All opponent rosters in team_id,name,status format

    Args:
        my_team_path: Path to my team's Fantrax roster CSV
        opponent_team_paths: Dict mapping team_id (1-6) to path of opponent's roster CSV
            Example: {1: 'data/raw_rosters/team_aidan.csv', 2: 'data/raw_rosters/team_oliver.csv', ...}
        output_dir: Directory to write output files (e.g., 'data/')

    Returns:
        Tuple of (my_roster_path, opponent_rosters_path) for the created files

    Example usage:
        my_roster_path, opponent_rosters_path = convert_fantrax_rosters(
            my_team_path='data/raw_rosters/my_team.csv',
            opponent_team_paths={
                1: 'data/raw_rosters/team_aidan.csv',
                2: 'data/raw_rosters/team_oliver.csv',
                3: 'data/raw_rosters/team_future.csv',
                4: 'data/raw_rosters/team_paranoia.csv',
                5: 'data/raw_rosters/team_reasonable.csv',
                6: 'data/raw_rosters/team_shohei.csv',
            },
            output_dir='data/',
        )
    """
    print("=== Converting Fantrax Rosters ===")

    # Validate opponent team_ids
    expected_team_ids = {1, 2, 3, 4, 5, 6}
    actual_team_ids = set(opponent_team_paths.keys())
    assert actual_team_ids == expected_team_ids, (
        f"opponent_team_paths must have exactly team_ids {expected_team_ids}, "
        f"got {actual_team_ids}"
    )

    output_path = Path(output_dir)

    # Parse and convert my roster
    print(f"Parsing my roster: {my_team_path}")
    my_roster = parse_fantrax_roster(my_team_path)

    # Count by status
    status_counts = my_roster["status"].value_counts().to_dict()
    print(
        f"  Found: {status_counts.get('active', 0)} active, "
        f"{status_counts.get('prospect', 0)} prospects, "
        f"{status_counts.get('IR', 0)} IR"
    )

    # Write my roster - include team and player_type for robust matching
    my_roster_output = my_roster[["name", "team", "player_type", "status"]].copy()
    my_roster_path = output_path / "my-roster.csv"
    my_roster_output.to_csv(my_roster_path, index=False)
    print(f"  Wrote: {my_roster_path}")

    # Parse and convert opponent rosters
    all_opponent_rows = []

    for team_id in sorted(opponent_team_paths.keys()):
        team_path = opponent_team_paths[team_id]
        print(f"Parsing opponent {team_id}: {team_path}")

        opponent_roster = parse_fantrax_roster(team_path)
        opponent_roster["team_id"] = team_id

        # Count by status
        status_counts = opponent_roster["status"].value_counts().to_dict()
        print(
            f"  Found: {status_counts.get('active', 0)} active, "
            f"{status_counts.get('prospect', 0)} prospects, "
            f"{status_counts.get('IR', 0)} IR"
        )

        all_opponent_rows.append(opponent_roster)

    # Combine all opponent rosters
    all_opponents = pd.concat(all_opponent_rows, ignore_index=True)

    # Write opponent rosters - include team and player_type for robust matching
    opponent_output = all_opponents[
        ["team_id", "name", "team", "player_type", "status"]
    ].copy()
    opponent_rosters_path = output_path / "opponent-rosters.csv"
    opponent_output.to_csv(opponent_rosters_path, index=False)
    print(f"  Wrote: {opponent_rosters_path}")

    # Summary
    total_active = (all_opponents["status"] == "active").sum() + (
        my_roster["status"] == "active"
    ).sum()
    total_players = len(all_opponents) + len(my_roster)
    print(f"\n=== Conversion Complete ===")
    print(f"Total players: {total_players} ({total_active} active across all teams)")

    return str(my_roster_path), str(opponent_rosters_path)


def find_name_mismatches(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
) -> pd.DataFrame:
    """
    Find roster names that don't match FanGraphs projections and suggest corrections.

    This utility helps diagnose name matching issues between Fantrax exports
    and FanGraphs projections. Common issues include:
    - Accented characters: "Andres Munoz" vs "Andrés Muñoz"
    - Suffixes: "Shohei Ohtani-H" vs "Shohei Ohtani"
    - Spelling variations

    Args:
        roster_path: Path to the converted roster CSV (my-roster.csv or opponent-rosters.csv)
        projections: Combined projections DataFrame
        is_opponent_file: True if this is opponent-rosters.csv format

    Returns:
        DataFrame with columns: roster_name, suggested_match, similarity_score
        Only includes names that don't have an exact match.
    """
    # Read roster
    roster_df = pd.read_csv(roster_path)

    if is_opponent_file:
        roster_names = set(roster_df["name"].tolist())
    else:
        roster_names = set(roster_df["name"].tolist())

    projection_names = set(projections["Name"].tolist())

    # Find unmatched names
    unmatched = roster_names - projection_names

    if not unmatched:
        print("All roster names match projections!")
        return pd.DataFrame(
            columns=["roster_name", "suggested_match", "similarity_score"]
        )

    print(f"Found {len(unmatched)} unmatched names")

    # Known corrections for tricky names that normalization doesn't catch
    # Note: These need -H/-P suffixes since all names now have them
    known_corrections = {
        "Logan OHoppe-H": "Logan O'Hoppe-H",
        "Leodalis De Vries-H": "Leo De Vries-H",
        "Leodalis De Vries-P": "Leo De Vries-P",
    }

    # Build normalized projection name lookup
    proj_name_lookup = {}
    for name in projection_names:
        norm = _normalize_name_for_comparison(name)
        if norm not in proj_name_lookup:
            proj_name_lookup[norm] = []
        proj_name_lookup[norm].append(name)

    # Find matches
    results = []
    for roster_name in sorted(unmatched):
        # Check known corrections first
        if roster_name in known_corrections:
            correction = known_corrections[roster_name]
            if correction in projection_names:
                results.append(
                    {
                        "roster_name": roster_name,
                        "suggested_match": correction,
                        "similarity_score": 1.0,
                    }
                )
                continue

        norm_roster = _normalize_name_for_comparison(roster_name)

        if norm_roster in proj_name_lookup:
            # Exact normalized match
            matches = proj_name_lookup[norm_roster]
            for match in matches:
                results.append(
                    {
                        "roster_name": roster_name,
                        "suggested_match": match,
                        "similarity_score": 1.0,
                    }
                )
        else:
            # No exact normalized match - find closest
            best_match = None
            best_score = 0

            for proj_name in projection_names:
                norm_proj = _normalize_name_for_comparison(proj_name)

                # Simple overlap score
                roster_parts = set(norm_roster.split())
                proj_parts = set(norm_proj.split())

                if roster_parts and proj_parts:
                    intersection = roster_parts & proj_parts
                    union = roster_parts | proj_parts
                    score = len(intersection) / len(union) if union else 0

                    if score > best_score:
                        best_score = score
                        best_match = proj_name

            results.append(
                {
                    "roster_name": roster_name,
                    "suggested_match": best_match
                    if best_score > 0.3
                    else "(no good match)",
                    "similarity_score": best_score,
                }
            )

    return pd.DataFrame(results)


def apply_name_corrections(
    roster_path: str,
    projections: pd.DataFrame,
    is_opponent_file: bool = False,
    min_similarity: float = 0.9,
) -> None:
    """
    Apply automatic name corrections to a roster file based on fuzzy matching.

    This function finds names that don't match projections but have a high-confidence
    match based on normalized comparison, and updates the roster file in place.

    Args:
        roster_path: Path to the roster CSV file (my-roster.csv or opponent-rosters.csv)
        projections: Combined projections DataFrame
        is_opponent_file: True if this is opponent-rosters.csv format (unused, kept for API compat)
        min_similarity: Minimum similarity score to auto-correct (default 0.9)

    Note:
        - Corrections with similarity >= min_similarity are applied automatically
        - Lower-confidence matches are printed but not applied
    """
    mismatches = find_name_mismatches(roster_path, projections, is_opponent_file)

    if len(mismatches) == 0:
        return

    # Separate high-confidence and low-confidence matches
    auto_correct = mismatches[mismatches["similarity_score"] >= min_similarity]
    manual_review = mismatches[mismatches["similarity_score"] < min_similarity]

    if len(auto_correct) > 0:
        print(
            f"\nApplying {len(auto_correct)} automatic corrections (similarity >= {min_similarity}):"
        )

        roster_df = pd.read_csv(roster_path)

        correction_map = dict(
            zip(auto_correct["roster_name"], auto_correct["suggested_match"])
        )

        for old_name, new_name in correction_map.items():
            if old_name != new_name:
                print(f"  '{old_name}' -> '{new_name}'")
                roster_df["name"] = roster_df["name"].replace(old_name, new_name)

        roster_df.to_csv(roster_path, index=False)
        print(f"Updated {roster_path}")

    if len(manual_review) > 0:
        print(f"\n*** MANUAL REVIEW NEEDED ({len(manual_review)} names) ***")
        print("These names have low-confidence matches and need manual correction:")
        for _, row in manual_review.iterrows():
            print(
                f"  '{row['roster_name']}' -> '{row['suggested_match']}' (score: {row['similarity_score']:.2f})"
            )
        print("\nEdit the roster file manually or update the source data.")


def convert_fantrax_rosters_from_dir(
    raw_rosters_dir: str,
    my_team_filename: str = "my_team.csv",
    output_dir: str = None,
) -> tuple[str, str]:
    """
    Convenience wrapper that auto-discovers opponent roster files in a directory.

    This function looks for all CSV files in the raw_rosters_dir that are not
    the my_team file and assigns them team_ids 1-6 in alphabetical order.

    Args:
        raw_rosters_dir: Directory containing Fantrax roster CSV files
            (e.g., 'data/raw_rosters/')
        my_team_filename: Filename of my team's roster (default: 'my_team.csv')
        output_dir: Directory to write output files. If None, uses parent of raw_rosters_dir.

    Returns:
        Tuple of (my_roster_path, opponent_rosters_path) for the created files

    Example usage:
        my_roster_path, opponent_rosters_path = convert_fantrax_rosters_from_dir(
            raw_rosters_dir='data/raw_rosters/',
        )
    """
    raw_dir = Path(raw_rosters_dir)
    assert raw_dir.exists(), f"Directory does not exist: {raw_rosters_dir}"

    # Find my team file
    my_team_path = raw_dir / my_team_filename
    assert my_team_path.exists(), f"My team file not found: {my_team_path}"

    # Find all other CSV files (opponent rosters)
    all_csvs = sorted(raw_dir.glob("*.csv"))
    opponent_files = [f for f in all_csvs if f.name != my_team_filename]

    assert len(opponent_files) == 6, (
        f"Expected exactly 6 opponent roster files, found {len(opponent_files)}: "
        f"{[f.name for f in opponent_files]}"
    )

    # Assign team_ids 1-6 in alphabetical order
    opponent_team_paths = {i + 1: str(f) for i, f in enumerate(opponent_files)}

    print("Auto-discovered opponent rosters (assigned in alphabetical order):")
    for team_id, path in opponent_team_paths.items():
        print(f"  Team {team_id}: {Path(path).name}")
    print()

    # Determine output directory
    if output_dir is None:
        output_dir = str(raw_dir.parent)

    return convert_fantrax_rosters(
        my_team_path=str(my_team_path),
        opponent_team_paths=opponent_team_paths,
        output_dir=output_dir,
    )


# === DATA LOADING ===


def load_positions_from_db(db_path: str) -> dict[int, str]:
    """
    Load player positions from SQLite database.

    Args:
        db_path: Path to mlb_stats.db (typically '../mlb_player_comps_dashboard/mlb_stats.db')

    Returns:
        Dict mapping MLBAMID (int) to position (str).
        Example: {592450: 'OF', 677951: 'SS', ...}
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT player_id, position FROM players")
    positions = {int(row[0]): row[1] for row in cursor if row[1] is not None}
    conn.close()

    print(f"Loaded positions for {len(positions)} players from database")
    return positions


def load_hitter_projections(filepath: str, positions: dict[int, str]) -> pd.DataFrame:
    """
    Load hitter projections from FanGraphs CSV export.

    Args:
        filepath: Path to fangraphs-steamer-projections-hitters.csv
        positions: Dict mapping MLBAMID to position (from load_positions_from_db)

    Returns:
        DataFrame with columns: Name, Team, Position, PA, R, HR, RBI, SB, OBP, SLG, OPS
        Plus a 'player_type' column set to 'hitter' for all rows.

    Note: All player Names are suffixed with '-H' to ensure global uniqueness.
    """
    df = pd.read_csv(filepath)

    # OPS column already exists—use it directly (do NOT recompute)
    assert "OPS" in df.columns, "OPS column must exist in hitter projections CSV"

    # Join position from positions dict using MLBAMID column
    assert "MLBAMID" in df.columns, "MLBAMID column required for position lookup"

    df["Position"] = df["MLBAMID"].map(positions)

    # Count how many got positions from DB vs defaulted to DH
    from_db = df["Position"].notna().sum()
    defaulted = df["Position"].isna().sum()

    # For players not found in positions dict, assign Position = 'DH' (fallback)
    df["Position"] = df["Position"].fillna("DH")

    # Add player_type
    df["player_type"] = "hitter"

    # Append '-H' suffix to all names for global uniqueness
    # This makes "Luis Castillo-H" (hitter) distinct from "Luis Castillo-P" (pitcher)
    df["Name"] = df["Name"] + "-H"

    # Select and keep required columns
    required_cols = [
        "Name",
        "Team",
        "Position",
        "PA",
        "R",
        "HR",
        "RBI",
        "SB",
        "OBP",
        "SLG",
        "OPS",
        "player_type",
        "MLBAMID",
    ]
    df = df[required_cols].copy()

    # Assert no null values in required columns
    for col in ["Name", "PA", "R", "HR", "RBI", "SB", "OPS"]:
        assert df[col].notna().all(), (
            f"Null values found in required hitter column: {col}"
        )

    # Handle duplicate names by keeping first occurrence (higher-projected player)
    # This is necessary because FanGraphs includes minor leaguers with common names
    duplicates = df[df["Name"].duplicated()]["Name"].tolist()
    if duplicates:
        print(
            f"  Note: Removing {len(duplicates)} duplicate hitter names (keeping first/highest projected)"
        )
        df = df.drop_duplicates(subset="Name", keep="first")

    print(
        f"Loaded {len(df)} hitter projections ({from_db} with positions from DB, {defaulted} defaulted to DH)"
    )
    return df


def load_pitcher_projections(filepath: str) -> pd.DataFrame:
    """
    Load pitcher projections from FanGraphs CSV export.

    Args:
        filepath: Path to fangraphs-steamer-projections-pitchers.csv

    Returns:
        DataFrame with columns: Name, Team, Position, IP, W, SV, K, ERA, WHIP
        Plus a 'player_type' column set to 'pitcher' for all rows.

    Note: All player Names are suffixed with '-P' to ensure global uniqueness.
    """
    df = pd.read_csv(filepath)

    # Rename SO column to K
    assert "SO" in df.columns, (
        "SO (strikeouts) column must exist in pitcher projections CSV"
    )
    df = df.rename(columns={"SO": "K"})

    # Determine Position: 'SP' if GS >= 3, else 'RP'
    assert "GS" in df.columns, (
        "GS (games started) column required for SP/RP determination"
    )
    df["Position"] = np.where(df["GS"] >= 3, "SP", "RP")

    # Add player_type
    df["player_type"] = "pitcher"

    # Append '-P' suffix to all names for global uniqueness
    # This makes "Luis Castillo-P" (pitcher) distinct from "Luis Castillo-H" (hitter)
    df["Name"] = df["Name"] + "-P"

    # Select and keep required columns
    required_cols = [
        "Name",
        "Team",
        "Position",
        "IP",
        "W",
        "SV",
        "K",
        "ERA",
        "WHIP",
        "player_type",
        "MLBAMID",
    ]
    df = df[required_cols].copy()

    # Assert no null values in required columns
    for col in ["Name", "IP", "W", "SV", "K", "ERA", "WHIP"]:
        assert df[col].notna().all(), (
            f"Null values found in required pitcher column: {col}"
        )

    # Handle duplicate names by keeping first occurrence (higher-projected player)
    duplicates = df[df["Name"].duplicated()]["Name"].tolist()
    if duplicates:
        print(
            f"  Note: Removing {len(duplicates)} duplicate pitcher names (keeping first/highest projected)"
        )
        df = df.drop_duplicates(subset="Name", keep="first")

    sp_count = (df["Position"] == "SP").sum()
    rp_count = (df["Position"] == "RP").sum()

    print(f"Loaded {len(df)} pitcher projections ({sp_count} SP, {rp_count} RP)")
    return df


def load_projections(hitter_path: str, pitcher_path: str, db_path: str) -> pd.DataFrame:
    """
    Load and combine hitter and pitcher projections.

    Args:
        hitter_path: Path to hitter projections CSV
        pitcher_path: Path to pitcher projections CSV
        db_path: Path to SQLite database with position data

    Returns:
        Combined DataFrame with all players.
        Hitters have pitching columns (IP, W, SV, K, ERA, WHIP) set to 0.
        Pitchers have hitting columns (PA, R, HR, RBI, SB, OBP, SLG, OPS) set to 0.
    """
    # Load positions from database
    positions = load_positions_from_db(db_path)

    # Load hitters and pitchers separately
    hitters = load_hitter_projections(hitter_path, positions)
    pitchers = load_pitcher_projections(pitcher_path)

    # Add missing columns with value 0 to each
    # Hitters need pitcher columns
    for col in ["IP", "W", "SV", "K", "ERA", "WHIP"]:
        hitters[col] = 0

    # Pitchers need hitter columns
    for col in ["PA", "R", "HR", "RBI", "SB", "OBP", "SLG", "OPS"]:
        pitchers[col] = 0

    # Ensure same column order
    all_cols = [
        "Name",
        "Team",
        "Position",
        "player_type",
        "PA",
        "R",
        "HR",
        "RBI",
        "SB",
        "OBP",
        "SLG",
        "OPS",
        "IP",
        "W",
        "SV",
        "K",
        "ERA",
        "WHIP",
        "MLBAMID",
    ]
    hitters = hitters[all_cols]
    pitchers = pitchers[all_cols]

    # Concatenate into single DataFrame
    combined = pd.concat([hitters, pitchers], ignore_index=True)

    # Two-way players: Ohtani appears separately in hitters and pitchers files,
    # which is fine—they have different stats. Assert no duplicates WITHIN each
    # player_type, but allow same name to appear once as hitter and once as pitcher.
    # (Already checked in individual load functions)

    print(
        f"Combined projections: {len(combined)} total players ({len(hitters)} hitters, {len(pitchers)} pitchers)"
    )
    return combined


def load_my_roster(
    filepath: str, projections: pd.DataFrame
) -> tuple[set[str], set[str]]:
    """
    Load my roster from CSV.

    Supports two matching modes:
    1. If 'team' column exists: Uses [normalized_name + team] for robust matching
    2. If no 'team' column: Falls back to exact name matching (with warnings)

    Args:
        filepath: Path to my-roster.csv
        projections: Combined projections DataFrame (for name validation)

    Returns:
        Tuple of:
        - active_names: Set of PROJECTION player names on active roster (FanGraphs spelling)
        - inactive_names: Set of player names on prospect/IR slots (for reference)
    """
    df = pd.read_csv(filepath)

    assert "name" in df.columns, "my-roster.csv must have 'name' column"
    assert "status" in df.columns, "my-roster.csv must have 'status' column"

    # Check if team column exists for robust matching
    has_team_column = "team" in df.columns

    if has_team_column:
        print("  Using team-based matching (robust mode)")
        active_names, inactive_names = _match_roster_with_team(df, projections)
    else:
        print("  Using name-only matching (add 'team' column for better matching)")
        active_names, inactive_names = _match_roster_by_name_only(df, projections)

    print(
        f"My roster: {len(active_names)} active, {len(inactive_names)} inactive (prospects/IR)"
    )

    return active_names, inactive_names


def _match_roster_with_team(
    roster_df: pd.DataFrame, projections: pd.DataFrame
) -> tuple[set[str], set[str]]:
    """
    Match roster players to projections using [normalized_name + team].

    This is more robust than name-only matching because:
    - Handles accent differences (Julio Rodriguez vs Julio Rodríguez)
    - Disambiguates players with same name on different teams
    - Handles Fantrax two-way player suffixes (Ohtani-H → hitter, Ohtani-P → pitcher)

    Returns projection Names (FanGraphs spelling) for matched players.
    """
    # Build simple lookup: (normalized_name, normalized_team) -> projection Name
    # Names are globally unique with -H/-P suffix, so no player_type disambiguation needed
    proj_lookup = {}  # (norm_name, team) -> proj_name
    proj_lookup_by_name = {}  # norm_name -> proj_name (fallback for team mismatches)

    for _, row in projections.iterrows():
        norm_name = _normalize_name_for_comparison(row["Name"])
        team = _normalize_team_abbreviation(
            str(row["Team"]) if pd.notna(row["Team"]) else ""
        )

        proj_lookup[(norm_name, team)] = row["Name"]
        if norm_name not in proj_lookup_by_name:
            proj_lookup_by_name[norm_name] = row["Name"]

    # Split roster by status
    active_df = roster_df[roster_df["status"] == "active"]
    inactive_df = roster_df[roster_df["status"].isin(["prospect", "IR"])]

    active_names = set()
    unmatched = []

    for _, row in active_df.iterrows():
        roster_name = row["name"]  # Already has -H/-P suffix from parse_fantrax_roster
        roster_team = _normalize_team_abbreviation(
            str(row["team"]) if pd.notna(row["team"]) and row["team"] else ""
        )
        norm_name = _normalize_name_for_comparison(roster_name)

        # Try exact match (normalized name + team)
        if (norm_name, roster_team) in proj_lookup:
            proj_name = proj_lookup[(norm_name, roster_team)]
            active_names.add(proj_name)
            # Only print if spelling differs (accent normalization)
            if (
                _normalize_name_for_comparison(proj_name) == norm_name
                and proj_name != roster_name
            ):
                print(f"    Matched: '{roster_name}' → '{proj_name}'")
            continue

        # Fallback: match by name only (handles team abbreviation differences or free agents)
        if norm_name in proj_lookup_by_name:
            proj_name = proj_lookup_by_name[norm_name]
            active_names.add(proj_name)
            print(
                f"    Matched (team mismatch): '{roster_name}' ({roster_team}) → '{proj_name}'"
            )
            continue

        unmatched.append(f"{roster_name} ({roster_team})")

    assert len(unmatched) == 0, (
        f"These active roster players not found in projections: {unmatched}"
    )

    # Inactive names - just collect raw names
    inactive_names = set(inactive_df["name"].tolist())

    return active_names, inactive_names


def _match_roster_by_name_only(
    roster_df: pd.DataFrame, projections: pd.DataFrame
) -> tuple[set[str], set[str]]:
    """
    Match roster players to projections using name only (legacy mode).

    This is less robust - warns about potential mismatches.
    """
    # Split by status column
    active_df = roster_df[roster_df["status"] == "active"]
    inactive_df = roster_df[roster_df["status"].isin(["prospect", "IR"])]

    active_names = set(active_df["name"].tolist())
    inactive_names = set(inactive_df["name"].tolist())

    # Get all valid projection names
    all_proj_names = set(projections["Name"].tolist())

    # Check for unmatched names
    unmatched = active_names - all_proj_names
    assert len(unmatched) == 0, (
        f"These active roster players not found in projections (check spelling/accents): "
        f"{sorted(unmatched)}"
    )

    # Validate for suspicious matches (potential name collisions)
    _validate_roster_matches(active_names, projections, "my roster")

    return active_names, inactive_names


def _normalize_name_for_comparison(name: str) -> str:
    """
    Normalize player name for fuzzy comparison.

    Handles:
    - Accented characters (Rodríguez → rodriguez)
    - Apostrophe variations (O'Hoppe, OHoppe → o'hoppe)
    - Standardizes apostrophe characters
    - Removes Jr./Sr./II/III suffixes for matching

    Preserves -H/-P suffix since that's part of the unique player identifier.
    Example: "Aaron Judge-H" → "aaron judge-h"
    """
    import re
    import unicodedata

    # Extract and preserve -H/-P suffix
    suffix = ""
    if name.endswith("-H") or name.endswith("-P"):
        suffix = name[-2:].lower()
        name = name[:-2]

    # Remove accents
    normalized = unicodedata.normalize("NFKD", name)
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))

    # Handle "OHoppe" pattern BEFORE lowercasing (insert apostrophe between lowercase-Uppercase)
    normalized = re.sub(r"([a-z])([A-Z])", r"\1'\2", normalized)

    # Lowercase
    normalized = normalized.lower()

    # Standardize apostrophe characters
    normalized = normalized.replace("'", "'").replace("'", "'").replace("`", "'")

    # Remove Jr./Sr./II/III suffixes for matching
    for suffix_to_strip in [" jr.", " jr", " sr.", " sr", " ii", " iii"]:
        if normalized.endswith(suffix_to_strip):
            normalized = normalized[: -len(suffix_to_strip)]
            break

    return normalized.strip() + suffix


def _validate_roster_matches(
    roster_names: set[str],
    projections: pd.DataFrame,
    roster_label: str,
) -> None:
    """
    Validate roster names and warn about suspicious matches.

    Checks for:
    1. Names that match only low-volume players (potential minor league name collisions)
    2. Names where a different spelling (with accents) might be the intended player
    3. Names where hitter/pitcher versions exist but matched the wrong one
    """
    warnings = []

    # Build lookup of normalized names to actual projection names
    proj_names_normalized = {}
    for proj_name in projections["Name"].unique():
        norm = _normalize_name_for_comparison(proj_name)
        if norm not in proj_names_normalized:
            proj_names_normalized[norm] = []
        proj_names_normalized[norm].append(proj_name)

    for name in sorted(roster_names):
        matches = projections[projections["Name"] == name]

        if len(matches) == 0:
            continue  # Already caught by assertion

        # Check for accent mismatch - is there a different spelling that's a star player?
        norm_name = _normalize_name_for_comparison(name)
        if norm_name in proj_names_normalized:
            alternate_names = [n for n in proj_names_normalized[norm_name] if n != name]
            for alt_name in alternate_names:
                alt_player = projections[projections["Name"] == alt_name].iloc[0]
                current_player = matches.iloc[0]

                # Compare playing time - if alternate is much higher, warn
                if alt_player["player_type"] == "hitter" and alt_player["PA"] > 300:
                    if (
                        current_player["player_type"] == "pitcher"
                        or current_player["PA"] < 100
                    ):
                        warnings.append(
                            f"  🚨 '{name}' may be wrong player! Found '{alt_name}' with {alt_player['PA']:.0f} PA. "
                            f"Current match: {current_player['player_type']} with "
                            f"{current_player['PA'] if current_player['player_type'] == 'hitter' else current_player['IP']:.0f} "
                            f"{'PA' if current_player['player_type'] == 'hitter' else 'IP'}"
                        )
                elif alt_player["player_type"] == "pitcher" and alt_player["IP"] > 100:
                    if (
                        current_player["player_type"] == "hitter"
                        or current_player["IP"] < 30
                    ):
                        warnings.append(
                            f"  🚨 '{name}' may be wrong player! Found '{alt_name}' with {alt_player['IP']:.0f} IP. "
                            f"Current match: {current_player['player_type']}"
                        )

        # Check each match for low volume
        for _, player in matches.iterrows():
            player_type = player["player_type"]

            if player_type == "hitter":
                pa = player["PA"]
                if pa < 50:
                    warnings.append(
                        f"  ⚠ '{name}' matched a hitter with only {pa:.0f} PA - "
                        f"possible name collision with minor leaguer"
                    )
            else:  # pitcher
                ip = player["IP"]
                if ip < 20:
                    warnings.append(
                        f"  ⚠ '{name}' matched a pitcher with only {ip:.1f} IP - "
                        f"possible name collision with minor leaguer"
                    )

        # Check for hitter/pitcher mismatch - if name appears in both, warn
        hitter_match = projections[
            (projections["Name"] == name) & (projections["player_type"] == "hitter")
        ]
        pitcher_match = projections[
            (projections["Name"] == name) & (projections["player_type"] == "pitcher")
        ]

        if len(hitter_match) > 0 and len(pitcher_match) > 0:
            h_pa = hitter_match.iloc[0]["PA"]
            p_ip = pitcher_match.iloc[0]["IP"]

            # If one is much more significant than the other, note it
            if h_pa > 200 and p_ip < 20:
                warnings.append(
                    f"  ℹ '{name}' exists as both hitter ({h_pa:.0f} PA) and pitcher ({p_ip:.1f} IP) - "
                    f"both will be counted if rostered"
                )
            elif p_ip > 50 and h_pa < 50:
                warnings.append(
                    f"  ℹ '{name}' exists as both pitcher ({p_ip:.1f} IP) and hitter ({h_pa:.0f} PA) - "
                    f"both will be counted if rostered"
                )

    if warnings:
        print(f"\n  === Roster Validation Warnings for {roster_label} ===")
        for w in warnings:
            print(w)
        print()


def load_opponent_rosters(
    filepath: str, projections: pd.DataFrame
) -> dict[int, set[str]]:
    """
    Load opponent rosters from CSV.

    Supports two matching modes:
    1. If 'team' column exists: Uses [normalized_name + team] for robust matching
    2. If no 'team' column: Falls back to exact name matching

    Args:
        filepath: Path to opponent-rosters.csv
        projections: Combined projections DataFrame (for name validation)

    Returns:
        Dict mapping team_id to set of active PROJECTION player names (FanGraphs spelling).
        Example: {1: {'Mike Trout', 'Mookie Betts', ...}, 2: {...}, ...}
    """
    df = pd.read_csv(filepath)

    assert "team_id" in df.columns, "opponent-rosters.csv must have 'team_id' column"
    assert "name" in df.columns, "opponent-rosters.csv must have 'name' column"
    assert "status" in df.columns, "opponent-rosters.csv must have 'status' column"

    # Check if team column exists
    has_team_column = "team" in df.columns

    # Filter to status == 'active'
    active_df = df[df["status"] == "active"]

    # Build simple lookup: (normalized_name, normalized_team) -> projection Name
    # Names are globally unique with -H/-P suffix, so no player_type disambiguation needed
    proj_lookup = {}  # (norm_name, team) -> proj_name
    proj_lookup_by_name = {}  # norm_name -> proj_name (fallback)

    for _, row in projections.iterrows():
        norm_name = _normalize_name_for_comparison(row["Name"])
        mlb_team = _normalize_team_abbreviation(
            str(row["Team"]) if pd.notna(row["Team"]) else ""
        )

        proj_lookup[(norm_name, mlb_team)] = row["Name"]
        if norm_name not in proj_lookup_by_name:
            proj_lookup_by_name[norm_name] = row["Name"]

    all_proj_names = set(projections["Name"].tolist())

    # Group by team_id and match to projections
    opponent_rosters = {}
    unmatched_all = []

    for fantasy_team_id in sorted(active_df["team_id"].unique()):
        team_df = active_df[active_df["team_id"] == fantasy_team_id]
        matched_names = set()

        for _, row in team_df.iterrows():
            roster_name = row["name"]  # Already has -H/-P suffix

            if has_team_column:
                roster_team = _normalize_team_abbreviation(
                    str(row["team"]) if pd.notna(row["team"]) and row["team"] else ""
                )
                norm_name = _normalize_name_for_comparison(roster_name)

                # Try exact match (name + team)
                if (norm_name, roster_team) in proj_lookup:
                    matched_names.add(proj_lookup[(norm_name, roster_team)])
                # Fallback: match by name only
                elif norm_name in proj_lookup_by_name:
                    matched_names.add(proj_lookup_by_name[norm_name])
                else:
                    unmatched_all.append(
                        f"{roster_name} ({roster_team}) [team {fantasy_team_id}]"
                    )
            else:
                # Name-only matching (legacy mode)
                if roster_name in all_proj_names:
                    matched_names.add(roster_name)
                else:
                    unmatched_all.append(f"{roster_name} [team {fantasy_team_id}]")

        opponent_rosters[int(fantasy_team_id)] = matched_names

    # Assert team_ids are exactly {1, 2, 3, 4, 5, 6}
    expected_teams = {1, 2, 3, 4, 5, 6}
    actual_teams = set(opponent_rosters.keys())
    assert actual_teams == expected_teams, (
        f"Expected opponent team_ids {expected_teams}, got {actual_teams}. "
        f"Missing: {expected_teams - actual_teams}, Extra: {actual_teams - expected_teams}"
    )

    assert len(unmatched_all) == 0, (
        f"These opponent roster players not found in projections: {unmatched_all}"
    )

    # Assert no player appears on multiple fantasy teams
    # (With -H/-P suffixes, Ohtani-H and Ohtani-P are distinct names, no special handling needed)
    seen_players = {}
    for fantasy_team_id, names in opponent_rosters.items():
        for name in names:
            if name in seen_players:
                assert False, (
                    f"Player '{name}' appears on both team {seen_players[name]} and team {fantasy_team_id}"
                )
            seen_players[name] = fantasy_team_id

    roster_sizes = [len(opponent_rosters[i]) for i in range(1, 7)]
    print(
        f"Loaded 6 opponent rosters: {', '.join(map(str, roster_sizes))} active players each"
    )
    return opponent_rosters


def load_all_data(
    hitter_proj_path: str,
    pitcher_proj_path: str,
    my_roster_path: str,
    opponent_rosters_path: str,
    db_path: str = "../mlb_player_comps_dashboard/mlb_stats.db",
) -> tuple[pd.DataFrame, set[str], dict[int, set[str]]]:
    """
    Load all data files and validate.

    This is the main entry point for data loading.

    Args:
        hitter_proj_path: Path to hitter projections CSV
        pitcher_proj_path: Path to pitcher projections CSV
        my_roster_path: Path to my roster CSV
        opponent_rosters_path: Path to opponent rosters CSV
        db_path: Path to SQLite database with position data

    Returns:
        projections: Combined projections DataFrame
        my_active_roster: Set of player names on my active roster
        opponent_rosters: Dict mapping team_id to set of active player names
    """
    print("=== Loading Data ===")

    projections = load_projections(hitter_proj_path, pitcher_proj_path, db_path)
    my_active_roster, _ = load_my_roster(my_roster_path, projections)
    opponent_rosters = load_opponent_rosters(opponent_rosters_path, projections)

    print("=== Data Loading Complete ===\n")
    return projections, my_active_roster, opponent_rosters


# === TEAM TOTAL COMPUTATION ===


def compute_team_totals(
    player_names: Iterable[str],
    projections: pd.DataFrame,
) -> dict[str, float]:
    """
    Compute a team's projected totals in all 10 scoring categories.

    Args:
        player_names: Iterable of player names on the team (set, list, etc.)
        projections: Combined projections DataFrame

    Returns:
        Dict mapping category name to team total.
        Example: {'R': 823, 'HR': 245, 'RBI': 780, 'SB': 95, 'OPS': 0.758,
                  'W': 78, 'SV': 55, 'K': 1250, 'ERA': 3.85, 'WHIP': 1.22}
    """
    player_names_set = set(player_names)

    # Filter projections to players in player_names
    team_df = projections[projections["Name"].isin(player_names_set)]

    # Assert all player_names found in projections
    found_names = set(team_df["Name"].tolist())
    missing = player_names_set - found_names
    assert len(missing) == 0, f"Players not found in projections: {sorted(missing)}"

    # Separate into hitters and pitchers
    hitters = team_df[team_df["player_type"] == "hitter"]
    pitchers = team_df[team_df["player_type"] == "pitcher"]

    # Assertions for valid ratio stat computation
    total_pa = hitters["PA"].sum()
    total_ip = pitchers["IP"].sum()

    assert total_pa > 0, (
        f"Team has no plate appearances (PA=0). Check that hitters are on the roster."
    )
    assert total_ip > 0, (
        f"Team has no innings pitched (IP=0). Check that pitchers are on the roster."
    )

    totals = {}

    # Counting stats (R, HR, RBI, SB, W, SV, K): just sum
    for cat in ["R", "HR", "RBI", "SB"]:
        totals[cat] = hitters[cat].sum()

    for cat in ["W", "SV", "K"]:
        totals[cat] = pitchers[cat].sum()

    # Ratio stats: weighted average
    # OPS: sum(PA * OPS) / sum(PA) over hitters
    totals["OPS"] = (hitters["PA"] * hitters["OPS"]).sum() / total_pa

    # ERA: sum(IP * ERA) / sum(IP) over pitchers
    totals["ERA"] = (pitchers["IP"] * pitchers["ERA"]).sum() / total_ip

    # WHIP: sum(IP * WHIP) / sum(IP) over pitchers
    totals["WHIP"] = (pitchers["IP"] * pitchers["WHIP"]).sum() / total_ip

    return totals


def compute_all_opponent_totals(
    opponent_rosters: dict[int, set[str]],
    projections: pd.DataFrame,
) -> dict[int, dict[str, float]]:
    """
    Compute totals for all 6 opponents.

    Args:
        opponent_rosters: Dict mapping team_id to set of player names
        projections: Combined projections DataFrame

    Returns:
        Dict mapping team_id to their category totals.
        {1: {'R': 800, 'HR': 230, ...}, 2: {'R': 815, ...}, ...}
    """
    opponent_totals = {}

    for team_id in tqdm(
        sorted(opponent_rosters.keys()), desc="Computing opponent totals"
    ):
        opponent_totals[team_id] = compute_team_totals(
            opponent_rosters[team_id], projections
        )

    return opponent_totals


# === CANDIDATE PREFILTERING ===


def compute_quality_scores(projections: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a scalar quality score for each player.

    Methodology:
        Hitters: Mean z-score of (R, HR, RBI, SB).
                 Do NOT include OPS (ratio stat, z-score doesn't apply cleanly).

        Pitchers: Mean z-score of (W, SV, K, -ERA, -WHIP).
                  Negate ERA and WHIP before z-scoring so lower = better becomes higher = better.

    Returns:
        DataFrame with columns [Name, player_type, quality_score]
    """
    results = []

    # Process hitters
    hitters = projections[projections["player_type"] == "hitter"].copy()

    if len(hitters) > 0:
        hitter_z_scores = []
        for col in ["R", "HR", "RBI", "SB"]:
            std = hitters[col].std()
            assert std > 0, f"Standard deviation of {col} is 0 for hitters"
            z = (hitters[col] - hitters[col].mean()) / std
            hitter_z_scores.append(z)

        hitters["quality_score"] = np.mean(hitter_z_scores, axis=0)
        results.append(hitters[["Name", "player_type", "quality_score"]])

    # Process pitchers
    pitchers = projections[projections["player_type"] == "pitcher"].copy()

    if len(pitchers) > 0:
        pitcher_z_scores = []

        # Positive stats: W, SV, K
        for col in ["W", "SV", "K"]:
            std = pitchers[col].std()
            assert std > 0, f"Standard deviation of {col} is 0 for pitchers"
            z = (pitchers[col] - pitchers[col].mean()) / std
            pitcher_z_scores.append(z)

        # Negative stats: -ERA, -WHIP (negate so lower = better becomes higher score)
        for col in ["ERA", "WHIP"]:
            std = pitchers[col].std()
            assert std > 0, f"Standard deviation of {col} is 0 for pitchers"
            # Negate before z-scoring
            neg_values = -pitchers[col]
            z = (neg_values - neg_values.mean()) / neg_values.std()
            pitcher_z_scores.append(z)

        pitchers["quality_score"] = np.mean(pitcher_z_scores, axis=0)
        results.append(pitchers[["Name", "player_type", "quality_score"]])

    return pd.concat(results, ignore_index=True)


def filter_candidates(
    projections: pd.DataFrame,
    quality_scores: pd.DataFrame,
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    top_n_per_position: int = 30,
    top_n_per_category: int = 10,
) -> pd.DataFrame:
    """
    Filter to candidate players for optimization.

    Args:
        projections: Combined projections DataFrame
        quality_scores: DataFrame with quality_score per player
        my_roster_names: Set of player names currently on my roster
        opponent_roster_names: Set of ALL player names on ANY opponent roster (unavailable)
        top_n_per_position: How many players to keep per position
        top_n_per_category: How many top players per category to ensure are included

    Returns:
        Filtered projections DataFrame containing only candidate players.
    """
    # Create available_pool = projections excluding opponent_roster_names
    available_pool = projections[
        ~projections["Name"].isin(opponent_roster_names)
    ].copy()

    # Join quality_scores to available_pool
    available_pool = available_pool.merge(
        quality_scores[["Name", "quality_score"]], on="Name", how="left"
    )

    candidate_names = set()

    # Add all my_roster_names to candidate set (must be candidates even if low quality)
    candidate_names.update(my_roster_names)

    # For each slot type in SLOT_ELIGIBILITY keys:
    # Find players whose Position is in SLOT_ELIGIBILITY[slot]
    # Take top N by quality_score
    for slot, eligible_positions in SLOT_ELIGIBILITY.items():
        eligible_players = available_pool[
            available_pool["Position"].isin(eligible_positions)
        ]
        top_players = eligible_players.nlargest(top_n_per_position, "quality_score")[
            "Name"
        ]
        candidate_names.update(top_players)

    # For each scoring category:
    # Take top M players by that category's value
    for cat in ALL_CATEGORIES:
        if cat in NEGATIVE_CATEGORIES:
            # For ERA/WHIP, lower is better - but we want high volume pitchers
            # So take top by IP (most innings) among pitchers
            pitchers = available_pool[available_pool["player_type"] == "pitcher"]
            top_players = pitchers.nsmallest(top_n_per_category, cat)["Name"]
        else:
            top_players = available_pool.nlargest(top_n_per_category, cat)["Name"]
        candidate_names.update(top_players)

    # Return projections filtered to candidate set
    candidates = projections[projections["Name"].isin(candidate_names)].copy()

    hitter_count = (candidates["player_type"] == "hitter").sum()
    pitcher_count = (candidates["player_type"] == "pitcher").sum()

    print(
        f"Filtered to {len(candidates)} candidates from {len(projections)} total players"
    )
    print(f"  - {hitter_count} hitters, {pitcher_count} pitchers")

    return candidates


# === MILP FORMULATION AND SOLVING ===


def _build_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    check_eligibility: bool = True,
) -> tuple[pulp.LpProblem, dict, dict, dict, list, list]:
    """
    Build the MILP problem with all variables and constraints.

    This is the shared core logic used by both build_and_solve_milp and
    sensitivity analysis. Extracted to avoid code duplication.

    Args:
        candidates: DataFrame of candidate players (must be reset_index'd)
        opponent_totals: Dict mapping team_id to dict of category totals
        check_eligibility: If True, assert enough candidates for each slot

    Returns:
        prob: The LpProblem
        x: Dict of player binary variables {i: LpVariable}
        a: Dict of slot assignment variables {(i, s): LpVariable}
        y: Dict of beat indicator variables {(j, c): LpVariable}
        I_H: List of hitter indices
        I_P: List of pitcher indices
    """
    I = list(range(len(candidates)))
    I_H = [i for i in I if candidates.iloc[i]["player_type"] == "hitter"]
    I_P = [i for i in I if candidates.iloc[i]["player_type"] == "pitcher"]
    J = [1, 2, 3, 4, 5, 6]  # Opponent team_ids
    S = list(SLOT_ELIGIBILITY.keys())

    prob = pulp.LpProblem("roster_optimization", pulp.LpMaximize)

    # Decision variables
    x = {i: pulp.LpVariable(f"x_{i}", cat="Binary") for i in I}

    # Slot assignment variables (only for eligible player-slot pairs)
    a = {}
    for i in I:
        player_position = candidates.iloc[i]["Position"]
        for s in S:
            if player_position in SLOT_ELIGIBILITY[s]:
                a[i, s] = pulp.LpVariable(f"a_{i}_{s}", cat="Binary")

    # Beat indicator variables
    y = {
        (j, c): pulp.LpVariable(f"y_{j}_{c}", cat="Binary")
        for j in J
        for c in ALL_CATEGORIES
    }

    # Objective: maximize opponent-category wins
    prob += pulp.lpSum(y[j, c] for j in J for c in ALL_CATEGORIES)

    # C1: Roster Size
    prob += pulp.lpSum(x[i] for i in I) == ROSTER_SIZE

    # C2: Slot Assignment Requires Rostering
    for (i, s), var in a.items():
        prob += var <= x[i]

    # C3: Each Player Assigned to At Most One Slot
    for i in I:
        slots_for_player = [a[i, s] for s in S if (i, s) in a]
        if slots_for_player:
            prob += pulp.lpSum(slots_for_player) <= 1

    # C4: Starting Slots Must Be Filled
    all_slots = {**HITTING_SLOTS, **PITCHING_SLOTS}
    for s, required_count in all_slots.items():
        players_for_slot = [a[i, s] for i in I if (i, s) in a]
        if check_eligibility:
            assert len(players_for_slot) >= required_count, (
                f"Not enough candidates for {s} slot: need {required_count}, "
                f"but only {len(players_for_slot)} candidates are eligible."
            )
        if len(players_for_slot) >= required_count:
            prob += pulp.lpSum(players_for_slot) == required_count

    # C5: Roster Composition Bounds
    prob += pulp.lpSum(x[i] for i in I_H) >= MIN_HITTERS
    prob += pulp.lpSum(x[i] for i in I_H) <= MAX_HITTERS
    prob += pulp.lpSum(x[i] for i in I_P) >= MIN_PITCHERS
    prob += pulp.lpSum(x[i] for i in I_P) <= MAX_PITCHERS

    # C6: Beat Constraints for Counting Stats
    counting_stats = {"R", "HR", "RBI", "SB", "W", "SV", "K"}
    for c in counting_stats:
        I_relevant = I_H if c in HITTING_CATEGORIES else I_P
        for j in J:
            my_total = pulp.lpSum(candidates.iloc[i][c] * x[i] for i in I_relevant)
            opp_total = opponent_totals[j][c]
            prob += my_total >= opp_total + EPSILON_COUNTING - BIG_M_COUNTING * (
                1 - y[j, c]
            )

    # C7: Beat Constraints for OPS (higher is better)
    for j in J:
        opp_ops = opponent_totals[j]["OPS"]
        my_weighted_diff = pulp.lpSum(
            candidates.iloc[i]["PA"] * (candidates.iloc[i]["OPS"] - opp_ops) * x[i]
            for i in I_H
        )
        prob += my_weighted_diff >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, "OPS"])

    # C8: Beat Constraints for ERA and WHIP (lower is better)
    for c in ["ERA", "WHIP"]:
        for j in J:
            opp_val = opponent_totals[j][c]
            my_weighted_diff = pulp.lpSum(
                candidates.iloc[i]["IP"] * (opp_val - candidates.iloc[i][c]) * x[i]
                for i in I_P
            )
            prob += my_weighted_diff >= EPSILON_RATIO - BIG_M_RATIO * (1 - y[j, c])

    return prob, x, a, y, I_H, I_P


def build_and_solve_milp(
    candidates: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    current_roster_names: set[str],
) -> tuple[list[str], dict]:
    """
    Build and solve the MILP for optimal roster construction.

    Args:
        candidates: DataFrame of candidate players (filtered projections).
        opponent_totals: Dict mapping team_id to dict of category totals.
        current_roster_names: Set of player names currently on my roster.
                              (Not used in constraints, but useful for logging changes.)

    Returns:
        optimal_roster_names: List of player Names for the optimal 26-man roster
        solution_info: Dict with 'objective', 'solve_time', and 'status'
    """
    import time

    print(f"Building MILP with {len(candidates)} candidate players...")

    candidates = candidates.reset_index(drop=True)
    prob, x, a, y, I_H, I_P = _build_milp(candidates, opponent_totals)

    print(
        f"  Variables: {len(x)} player, {len(a)} slot assignment, {len(y)} beat indicators"
    )
    print("Solving...")

    start_time = time.time()
    solver = pulp.HiGHS(msg=1, timeLimit=300)
    status = prob.solve(solver)
    solve_time = time.time() - start_time

    assert status == pulp.LpStatusOptimal, (
        f"Solver failed with status: {pulp.LpStatus[status]}. "
        f"Check that enough candidates are eligible for each position slot."
    )

    optimal_roster_names = [
        candidates.iloc[i]["Name"] for i in x if pulp.value(x[i]) > 0.5
    ]
    objective_value = pulp.value(prob.objective)

    print(
        f"Solved in {solve_time:.1f}s — objective: {objective_value:.0f}/60 opponent-category wins"
    )

    return optimal_roster_names, {
        "objective": objective_value,
        "solve_time": solve_time,
        "status": pulp.LpStatus[status],
    }


# === SOLUTION OUTPUT ===


def compute_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute projected standings for each category.

    Returns:
        DataFrame with columns: [category, my_value, opp_1, opp_2, ..., opp_6, my_rank, wins]

        my_rank: 1 = first place, 7 = last place in that category
        wins: number of opponents I beat in that category (0-6)

    For negative categories (ERA, WHIP), lower value = better rank.
    """
    rows = []

    for cat in ALL_CATEGORIES:
        row = {"category": cat, "my_value": my_totals[cat]}

        # Add opponent values
        for team_id in range(1, 7):
            row[f"opp_{team_id}"] = opponent_totals[team_id][cat]

        # Compute rank and wins
        all_values = [my_totals[cat]] + [
            opponent_totals[tid][cat] for tid in range(1, 7)
        ]

        if cat in NEGATIVE_CATEGORIES:
            # Lower is better - sort ascending, my rank = position in sorted list
            sorted_values = sorted(all_values)
            my_rank = sorted_values.index(my_totals[cat]) + 1
            wins = sum(
                1 for tid in range(1, 7) if my_totals[cat] < opponent_totals[tid][cat]
            )
        else:
            # Higher is better - sort descending
            sorted_values = sorted(all_values, reverse=True)
            my_rank = sorted_values.index(my_totals[cat]) + 1
            wins = sum(
                1 for tid in range(1, 7) if my_totals[cat] > opponent_totals[tid][cat]
            )

        row["my_rank"] = my_rank
        row["wins"] = wins
        rows.append(row)

    return pd.DataFrame(rows)


def compute_projected_final_standings(
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
) -> pd.DataFrame:
    """
    Compute projected final roto standings with points for all 7 teams.

    Returns:
        DataFrame with columns: [team, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP, roto_points, rank]
        Sorted by roto_points descending (1st place first).
    """
    # Combine all teams' totals
    all_teams = {"Me": my_totals}
    for tid in range(1, 7):
        all_teams[f"Opp{tid}"] = opponent_totals[tid]

    # Compute rank in each category for each team
    ranks = {team: {} for team in all_teams}

    for cat in ALL_CATEGORIES:
        values = [(team, all_teams[team][cat]) for team in all_teams]

        if cat in NEGATIVE_CATEGORIES:
            # Lower is better
            sorted_values = sorted(values, key=lambda x: x[1])
        else:
            # Higher is better
            sorted_values = sorted(values, key=lambda x: x[1], reverse=True)

        for rank, (team, _) in enumerate(sorted_values, 1):
            ranks[team][cat] = rank

    # Compute roto points (8 - rank) for each team
    rows = []
    for team in all_teams:
        row = {"team": team}
        row.update(all_teams[team])
        row["roto_points"] = sum(8 - ranks[team][cat] for cat in ALL_CATEGORIES)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("roto_points", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def _display_name(name: str) -> str:
    """Strip -H/-P suffix for cleaner display."""
    return name[:-2] if name.endswith(("-H", "-P")) else name


def _print_player_table(players: list, player_type: str, indent: str = "") -> None:
    """Print formatted table of players (hitters or pitchers)."""
    if not players:
        return

    if player_type == "hitter":
        print(
            f"{indent}{'Pos':<4} {'Name':<22} {'Team':<4} {'PA':>5} {'R':>4} {'HR':>4} {'RBI':>4} {'SB':>4} {'OPS':>6}"
        )
        for p in players:
            print(
                f"{indent}{p['Position']:<4} {_display_name(p['Name']):<22} {str(p['Team']):<4} "
                f"{p['PA']:>5.0f} {p['R']:>4.0f} {p['HR']:>4.0f} {p['RBI']:>4.0f} {p['SB']:>4.0f} {p['OPS']:>6.3f}"
            )
    else:
        print(
            f"{indent}{'Pos':<4} {'Name':<22} {'Team':<4} {'IP':>5} {'W':>4} {'SV':>4} {'K':>5} {'ERA':>5} {'WHIP':>5}"
        )
        for p in players:
            print(
                f"{indent}{p['Position']:<4} {_display_name(p['Name']):<22} {str(p['Team']):<4} "
                f"{p['IP']:>5.1f} {p['W']:>4.1f} {p['SV']:>4.1f} {p['K']:>5.0f} {p['ERA']:>5.2f} {p['WHIP']:>5.3f}"
            )


def _split_by_player_type(
    names: set[str], projections: pd.DataFrame
) -> tuple[list, list]:
    """Split player names into hitters and pitchers lists."""
    hitters, pitchers = [], []
    for name in sorted(names):
        matches = projections[projections["Name"] == name]
        if len(matches) == 0:
            continue
        player = matches.iloc[0]
        (hitters if player["player_type"] == "hitter" else pitchers).append(player)
    return hitters, pitchers


def print_roster_summary(
    roster_names: list[str],
    projections: pd.DataFrame,
    my_totals: dict[str, float],
    opponent_totals: dict[int, dict[str, float]],
    old_roster_names: set[str] = None,
) -> None:
    """Print a formatted summary of the optimal roster."""
    roster_df = projections[projections["Name"].isin(roster_names)].copy()
    hitters = roster_df[roster_df["player_type"] == "hitter"].copy()
    pitchers = roster_df[roster_df["player_type"] == "pitcher"].copy()

    # Sort hitters by position priority, then by OPS
    position_order = {"C": 0, "1B": 1, "2B": 2, "SS": 3, "3B": 4, "OF": 5, "DH": 6}
    hitters["pos_order"] = hitters["Position"].map(position_order)
    hitters = hitters.sort_values(["pos_order", "OPS"], ascending=[True, False])

    # Sort pitchers by position (SP first), then by K
    pitchers["pos_order"] = pitchers["Position"].map({"SP": 0, "RP": 1})
    pitchers = pitchers.sort_values(["pos_order", "K"], ascending=[True, False])

    print("\n" + "=" * 80)
    print("                           OPTIMAL ROSTER")
    print("=" * 80)

    print(f"\nHITTERS ({len(hitters)}):")
    print("-" * 80)
    _print_player_table([row for _, row in hitters.iterrows()], "hitter")

    print(f"\nPITCHERS ({len(pitchers)}):")
    print("-" * 80)
    _print_player_table([row for _, row in pitchers.iterrows()], "pitcher")

    # Changes section
    if old_roster_names is not None:
        new_roster_set = set(roster_names)
        added = new_roster_set - old_roster_names
        dropped = old_roster_names - new_roster_set

        if added or dropped:
            print("\n" + "=" * 80)
            print("                         CHANGES FROM CURRENT ROSTER")
            print("=" * 80)

            if dropped:
                print("\nDROPPED:")
                print("-" * 80)
                dropped_h, dropped_p = _split_by_player_type(dropped, projections)
                _print_player_table(dropped_h, "hitter", "  ")
                _print_player_table(dropped_p, "pitcher", "  ")

            if added:
                print("\nADDED:")
                print("-" * 80)
                added_h, added_p = _split_by_player_type(added, projections)
                _print_player_table(added_h, "hitter", "  ")
                _print_player_table(added_p, "pitcher", "  ")

    # Final standings projection
    final_standings = compute_projected_final_standings(my_totals, opponent_totals)

    print("\n" + "=" * 80)
    print("                     PROJECTED FINAL STANDINGS")
    print("=" * 80)
    print(f"{'Rank':<6} {'Team':<8} {'Roto Pts':>10}")
    print("-" * 30)
    for _, row in final_standings.iterrows():
        marker = " ←" if row["team"] == "Me" else ""
        print(f"{row['rank']:<6} {row['team']:<8} {row['roto_points']:>10.0f}{marker}")
    print("-" * 30)

    # Category-by-category standings
    standings = compute_standings(my_totals, opponent_totals)

    print("\n" + "=" * 80)
    print("                      CATEGORY-BY-CATEGORY BREAKDOWN")
    print("=" * 80)
    print(
        f"{'Category':<10} {'Me':>10} {'Opp1':>8} {'Opp2':>8} {'Opp3':>8} {'Opp4':>8} {'Opp5':>8} {'Opp6':>8} {'Rank':>5} {'Wins':>5}"
    )
    print("-" * 80)

    for _, row in standings.iterrows():
        cat = row["category"]
        my_val = row["my_value"]

        # Format value based on category type
        if cat in ["OPS"]:
            fmt = ".3f"
        elif cat in ["ERA", "WHIP"]:
            fmt = ".2f"
        else:
            fmt = ".0f"

        # Mark wins with asterisk
        my_str = f"{my_val:{fmt}}"
        if row["wins"] > 0:
            my_str += "*"

        opp_strs = []
        for tid in range(1, 7):
            opp_val = row[f"opp_{tid}"]
            opp_str = f"{opp_val:{fmt}}"

            opp_strs.append(opp_str)

        print(
            f"{cat:<10} {my_str:>10} {opp_strs[0]:>8} {opp_strs[1]:>8} {opp_strs[2]:>8} {opp_strs[3]:>8} {opp_strs[4]:>8} {opp_strs[5]:>8} {row['my_rank']:>5.0f} {row['wins']:>5.0f}"
        )

    # Summary
    total_wins = standings["wins"].sum()
    total_roto_points = sum(8 - row["my_rank"] for _, row in standings.iterrows())

    print("\n" + "=" * 80)
    print("                              SUMMARY")
    print("=" * 80)
    print(f"Opponent-category wins: {total_wins:.0f} / 60")
    print(f"Projected roto points:  {total_roto_points:.0f} / 70")
    print("=" * 80 + "\n")
