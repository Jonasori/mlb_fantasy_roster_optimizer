"""Playing time adjustment algorithm."""

import numpy as np
import pandas as pd

from .config import (
    AGE_PENALTY_PER_YEAR,
    AGE_PENALTY_START_HITTER,
    AGE_PENALTY_START_PITCHER,
    ADJUSTMENT_WEIGHT,
    FULL_SEASON_IP_RP,
    FULL_SEASON_IP_SP,
    FULL_SEASON_PA,
    PROJECTION_WEIGHT,
    TALENT_PENALTY_FACTOR,
    TALENT_PENALTY_PERCENTILE,
)


def compute_historical_ratio(
    actual_current: float | None,
    actual_prior: float | None,
    full_season: float,
) -> float:
    """
    Compute ratio of actual to expected playing time.

    Args:
        actual_current: Actual PA/IP from most recent season (e.g., 2024)
        actual_prior: Actual PA/IP from prior season (e.g., 2023)
        full_season: Expected full-season PA/IP (600 for hitters, 180/70 for pitchers)

    Returns:
        Float in range [0.0, 1.0].

    Implementation:
        - If both seasons available: avg = (current + prior) / 2
        - If only one season: use that value
        - If neither: return 1.0 (trust projection for rookies)
        - Ratio = min(1.0, avg / full_season)

    The cap at 1.0 prevents bonus for historically high PT.
    """
    # Handle None/NaN values
    has_current = actual_current is not None and not pd.isna(actual_current)
    has_prior = actual_prior is not None and not pd.isna(actual_prior)

    if has_current and has_prior:
        avg = (actual_current + actual_prior) / 2
    elif has_current:
        avg = actual_current
    elif has_prior:
        avg = actual_prior
    else:
        # No historical data - trust projection for rookies
        return 1.0

    # Compute ratio, capped at 1.0
    ratio = avg / full_season
    return min(1.0, ratio)


def compute_age_factor(
    age: int | None,
    is_pitcher: bool,
) -> float:
    """
    Compute age-based adjustment factor.

    Args:
        age: Player age (or None if unknown)
        is_pitcher: True for pitchers, False for hitters

    Returns:
        Float in range [0.5, 1.0].

    Examples:
        - Age 30 hitter → 1.0
        - Age 35 hitter → 1.0 - 0.05*4 = 0.80
        - Age 38 pitcher → 1.0 - 0.05*6 = 0.70
    """
    if age is None or pd.isna(age):
        return 1.0

    threshold = AGE_PENALTY_START_PITCHER if is_pitcher else AGE_PENALTY_START_HITTER

    if age < threshold:
        return 1.0

    years_over = age - threshold + 1
    return max(0.5, 1.0 - AGE_PENALTY_PER_YEAR * years_over)


def compute_talent_factor(
    player_value: float,
    percentile_25: float,
) -> float:
    """
    Compute talent-based adjustment factor.

    Args:
        player_value: Player's value metric (WAR or computed SGP)
        percentile_25: 25th percentile value for player type

    Returns:
        1.0 if player_value >= percentile_25
        0.85 if player_value < percentile_25 (15% penalty for weak players)
    """
    if pd.isna(player_value) or pd.isna(percentile_25):
        return 1.0

    if player_value < percentile_25:
        return TALENT_PENALTY_FACTOR

    return 1.0


def adjust_playing_time(
    projected: float,
    historical_ratio: float,
    age_factor: float,
    talent_factor: float,
) -> float:
    """
    Apply three-factor adjustment and blend with projection.

    Formula:
        adjusted = projected * historical_ratio * age_factor * talent_factor
        blended = 0.65 * projected + 0.35 * adjusted

    Returns:
        Blended playing time value (float)
    """
    adjusted = projected * historical_ratio * age_factor * talent_factor
    blended = PROJECTION_WEIGHT * projected + ADJUSTMENT_WEIGHT * adjusted
    return blended


def apply_adjustments(
    projections: pd.DataFrame,
    historical: dict[str, pd.DataFrame],
    ages: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply playing time adjustments to all players.

    Args:
        projections: DataFrame from load_atc_projections()
        historical: Dict from load_historical_actuals()
        ages: DataFrame from load_ages()

    Returns:
        projections with:
        - PA_original, IP_original columns (preserved originals)
        - PA, IP columns (adjusted values)
        - age column (joined from ages df)
    """
    df = projections.copy()

    # Store originals
    df["PA_original"] = df["PA"]
    df["IP_original"] = df["IP"]

    # --- Pivot historical to wide format ---

    # Hitters: mlbam_id -> pa_2024, pa_2023
    hitters_hist = historical["hitters"]
    if len(hitters_hist) > 0:
        hitters_pivot = hitters_hist.pivot(
            index="mlbam_id", columns="season", values="pa"
        )
        hitters_pivot = hitters_pivot.add_prefix("pa_")
        hitters_pivot = hitters_pivot.reset_index()
        # Ensure columns exist
        if "pa_2024" not in hitters_pivot.columns:
            hitters_pivot["pa_2024"] = np.nan
        if "pa_2023" not in hitters_pivot.columns:
            hitters_pivot["pa_2023"] = np.nan
    else:
        hitters_pivot = pd.DataFrame(columns=["mlbam_id", "pa_2024", "pa_2023"])

    # Pitchers: mlbam_id -> ip_2024, ip_2023, gs_2024
    pitchers_hist = historical["pitchers"]
    if len(pitchers_hist) > 0:
        # Pivot IP
        pitchers_ip_pivot = pitchers_hist.pivot(
            index="mlbam_id", columns="season", values="ip"
        )
        pitchers_ip_pivot = pitchers_ip_pivot.add_prefix("ip_")

        # Pivot GS (need 2024 GS to determine SP vs RP)
        pitchers_gs_pivot = pitchers_hist.pivot(
            index="mlbam_id", columns="season", values="gs"
        )
        pitchers_gs_pivot = pitchers_gs_pivot.add_prefix("gs_")

        pitchers_pivot = pitchers_ip_pivot.join(
            pitchers_gs_pivot, how="outer"
        ).reset_index()

        # Ensure columns exist
        for col in ["ip_2024", "ip_2023", "gs_2024"]:
            if col not in pitchers_pivot.columns:
                pitchers_pivot[col] = np.nan
    else:
        pitchers_pivot = pd.DataFrame(
            columns=["mlbam_id", "ip_2024", "ip_2023", "gs_2024"]
        )

    # --- Join historical to projections ---
    df = df.merge(hitters_pivot, left_on="MLBAMID", right_on="mlbam_id", how="left")
    df = df.merge(
        pitchers_pivot,
        left_on="MLBAMID",
        right_on="mlbam_id",
        how="left",
        suffixes=("", "_p"),
    )

    # Drop duplicate mlbam_id columns
    df = df.drop(columns=["mlbam_id", "mlbam_id_p"], errors="ignore")

    # --- Join ages ---
    if len(ages) > 0:
        df = df.merge(ages, left_on="MLBAMID", right_on="mlbam_id", how="left")
        df = df.drop(columns=["mlbam_id"], errors="ignore")
    else:
        df["age"] = np.nan

    # --- Compute value percentiles ---
    hitters_mask = df["player_type"] == "hitter"
    pitchers_mask = df["player_type"] == "pitcher"

    hitter_war_25 = df.loc[hitters_mask, "WAR"].quantile(TALENT_PENALTY_PERCENTILE)
    pitcher_war_25 = df.loc[pitchers_mask, "WAR"].quantile(TALENT_PENALTY_PERCENTILE)

    # --- Apply adjustments ---
    hitters_adjusted = 0
    pitchers_adjusted = 0
    missing_historical = 0
    missing_age = 0
    hitter_pa_changes = []
    pitcher_ip_changes = []

    for idx, row in df.iterrows():
        is_pitcher = row["player_type"] == "pitcher"

        # Get percentile for talent factor
        percentile_25 = pitcher_war_25 if is_pitcher else hitter_war_25

        # Compute factors
        talent_factor = compute_talent_factor(row["WAR"], percentile_25)
        age_factor = compute_age_factor(row.get("age"), is_pitcher)

        if is_pitcher:
            # Determine full season IP based on role (use projected GS, fallback to historical)
            is_sp = row.get("Position") == "SP"
            full_season_ip = FULL_SEASON_IP_SP if is_sp else FULL_SEASON_IP_RP

            actual_current = row.get("ip_2024")
            actual_prior = row.get("ip_2023")

            historical_ratio = compute_historical_ratio(
                actual_current, actual_prior, full_season_ip
            )

            # Track missing data
            if pd.isna(actual_current) and pd.isna(actual_prior):
                missing_historical += 1

            original_ip = row["IP_original"]
            new_ip = adjust_playing_time(
                original_ip, historical_ratio, age_factor, talent_factor
            )

            df.at[idx, "IP"] = new_ip

            if original_ip != new_ip:
                pitchers_adjusted += 1
                pitcher_ip_changes.append(new_ip - original_ip)

        else:
            # Hitter
            actual_current = row.get("pa_2024")
            actual_prior = row.get("pa_2023")

            historical_ratio = compute_historical_ratio(
                actual_current, actual_prior, FULL_SEASON_PA
            )

            # Track missing data
            if pd.isna(actual_current) and pd.isna(actual_prior):
                missing_historical += 1

            original_pa = row["PA_original"]
            new_pa = adjust_playing_time(
                original_pa, historical_ratio, age_factor, talent_factor
            )

            df.at[idx, "PA"] = int(round(new_pa))

            if original_pa != int(round(new_pa)):
                hitters_adjusted += 1
                hitter_pa_changes.append(new_pa - original_pa)

        # Track missing age
        if pd.isna(row.get("age")):
            missing_age += 1

    # --- Print summary ---
    avg_pa_change = np.mean(hitter_pa_changes) if hitter_pa_changes else 0
    avg_ip_change = np.mean(pitcher_ip_changes) if pitcher_ip_changes else 0

    print("Playing time adjustments:")
    print(
        f"  Hitters: {hitters_adjusted} adjusted, avg change = {avg_pa_change:+.1f} PA"
    )
    print(
        f"  Pitchers: {pitchers_adjusted} adjusted, avg change = {avg_ip_change:+.1f} IP"
    )
    print(f"  {missing_historical} players missing historical data (rookies)")
    print(f"  {missing_age} players missing age data")

    # Clean up intermediate columns
    cols_to_drop = ["pa_2024", "pa_2023", "ip_2024", "ip_2023", "gs_2024"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors="ignore")

    return df
