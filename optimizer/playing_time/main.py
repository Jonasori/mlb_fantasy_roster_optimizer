"""CLI entry point for playing time adjustment."""

import argparse
from pathlib import Path

import pandas as pd

from .adjust import apply_adjustments
from .config import ATC_HITTERS, ATC_PITCHERS, MLB_STATS_DB, OPTIMIZER_DB, OUTPUT_PATH
from .load import load_ages, load_atc_projections, load_historical_actuals


def main():
    """
    Adjust ATC projections for playing time bias.

    Usage:
        python -m optimizer.playing_time.main
        python -m optimizer.playing_time.main --output custom.csv
        python -m optimizer.playing_time.main --no-age-adjustment
    """
    parser = argparse.ArgumentParser(
        description="Adjust ATC projections for playing time"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--no-age-adjustment",
        action="store_true",
        help="Skip age-based adjustment",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=OPTIMIZER_DB,
        help=f"Path to optimizer database (default: {OPTIMIZER_DB})",
    )
    parser.add_argument(
        "--mlb-stats-db",
        type=Path,
        default=MLB_STATS_DB,
        help=f"Path to MLB stats database (default: {MLB_STATS_DB})",
    )
    parser.add_argument(
        "--hitters",
        type=Path,
        default=ATC_HITTERS,
        help=f"Path to ATC hitters CSV (default: {ATC_HITTERS})",
    )
    parser.add_argument(
        "--pitchers",
        type=Path,
        default=ATC_PITCHERS,
        help=f"Path to ATC pitchers CSV (default: {ATC_PITCHERS})",
    )
    args = parser.parse_args()

    print("=== Playing Time Adjustment ===\n")

    # Step 1: Load ATC projections
    print("Step 1: Loading ATC projections...")
    projections = load_atc_projections(args.hitters, args.pitchers)

    # Step 2: Load historical actuals
    print("\nStep 2: Loading historical actuals...")
    historical = load_historical_actuals(args.mlb_stats_db, [2024, 2023])

    # Step 3: Load ages from database
    print("\nStep 3: Loading ages from database...")
    if args.no_age_adjustment:
        print("  Skipping (--no-age-adjustment)")
        ages = pd.DataFrame(columns=["mlbam_id", "age"])
    else:
        ages = load_ages(args.db)

    # Step 4: Apply adjustments
    print("\nStep 4: Applying adjustments...")
    adjusted = apply_adjustments(projections, historical, ages)

    # Step 5: Validate results
    print("\nStep 5: Validating results...")
    _validate_results(adjusted)

    # Step 6: Write output
    print(f"\nStep 6: Writing to {args.output}...")
    # Ensure parent directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    adjusted.to_csv(args.output, index=False)
    print(f"  Wrote {len(adjusted)} players")

    print("\n=== Done ===")


def _validate_results(adjusted: pd.DataFrame) -> None:
    """Run validation checks on adjusted projections."""
    # Basic checks
    assert adjusted["MLBAMID"].notna().all(), "All players must have MLBAMID"
    assert (adjusted["PA"] >= 0).all(), "PA must be non-negative"
    assert (adjusted["IP"] >= 0).all(), "IP must be non-negative"

    hitters = adjusted[adjusted["player_type"] == "hitter"]
    pitchers = adjusted[adjusted["player_type"] == "pitcher"]

    # Adjustments should generally reduce PT, not increase it
    total_pa = hitters["PA"].sum()
    total_pa_original = hitters["PA_original"].sum()
    pa_ratio = total_pa / total_pa_original if total_pa_original > 0 else 1.0

    total_ip = pitchers["IP"].sum()
    total_ip_original = pitchers["IP_original"].sum()
    ip_ratio = total_ip / total_ip_original if total_ip_original > 0 else 1.0

    print(f"  Total PA: {total_pa_original:.0f} -> {total_pa:.0f} ({pa_ratio:.1%})")
    print(f"  Total IP: {total_ip_original:.0f} -> {total_ip:.0f} ({ip_ratio:.1%})")

    assert pa_ratio <= 1.05, (
        f"Total PA should not increase significantly: {pa_ratio:.1%}"
    )
    assert ip_ratio <= 1.05, (
        f"Total IP should not increase significantly: {ip_ratio:.1%}"
    )

    # Spot check: older players should have reduced PT
    if "age" in adjusted.columns:
        old_hitters = hitters[hitters["age"] >= 35]
        if len(old_hitters) > 0:
            old_reduced = (old_hitters["PA"] <= old_hitters["PA_original"]).mean()
            print(
                f"  Old hitters (35+) with reduced PA: {old_reduced:.0%} ({len(old_hitters)} players)"
            )
            # Allow some flexibility - not all old players must have reduced PA
            # (e.g., if they have strong historical performance)
            if old_reduced < 0.5:
                print(f"  WARNING: Less than 50% of old hitters have reduced PA")

    print("  Validation passed!")


if __name__ == "__main__":
    main()
