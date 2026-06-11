"""
Deep analysis: MEW vs EW — Is the linear approximation valid?

MEW is the first-order Taylor approximation of the EW surface.
The lineup MILP maximizes Σ MEW(starters). This script investigates
whether that reliably maximizes EW, and quantifies the approximation error.

Research questions:
  1. Ordinal consistency: when Σ MEW(L₁) > Σ MEW(L₂), is EW(L₁) > EW(L₂)?
  2. Swap accuracy: how well does MEW(add) - MEW(drop) predict exact MSV?
  3. Curvature: how nonlinear is the EW surface at the operating point?
  4. Error magnitude: how much EW do we leave on the table?
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))

from optimizer.config import ALL_CATEGORIES, MY_TEAM_NAME, NEGATIVE_CATEGORIES
from optimizer.league_state import compute_league_state
from optimizer.lineup_solver import compute_totals_for_starters, solve_lineup
from optimizer.player_scoring import add_fantasy_value, add_mew
from optimizer.players import get_eligible_slots, strip_name_suffix
from optimizer.rosters import get_main_roster
from optimizer.win_model import (
    compute_ew_gradient,
    compute_win_probability,
    estimate_projection_uncertainty,
)

# ============================================================================
# LOAD DATA AND COMPUTE BASELINE STATE
# ============================================================================


def load_and_prepare():
    """Load silver table and run pipeline to get baseline state."""
    silver = Path("../data_prep/data/silver_table.parquet")
    assert silver.exists(), f"Silver table not found: {silver}"
    players = pd.read_parquet(silver)
    print(f"Loaded {len(players)} players")

    players = add_fantasy_value(players)
    state = compute_league_state(players)
    players = add_mew(players, state["my_totals"], state["gradient"])

    return players, state


# ============================================================================
# ANALYSIS 1: What does Σ MEW actually compute?
# ============================================================================


def analyze_mew_sum_vs_ew(players, state):
    """Show what Σ MEW(starters) actually equals, compared to EW."""
    print("\n" + "=" * 70)
    print("ANALYSIS 1: What is Σ MEW, and why does it differ from EW?")
    print("=" * 70)

    my_starters = state["my_starters"]
    starter_df = players[players["Name"].isin(my_starters)]

    total_mew = float(starter_df["MEW"].sum())
    current_ew = state["current_ew"]
    gradient = state["gradient"]
    my_totals = state["my_totals"]

    # Decompose Σ MEW into counting stat contribution + ratio stat contribution
    counting_contribution = 0.0
    for cat in ("R", "HR", "RBI", "SB", "W", "SV", "K"):
        counting_contribution += gradient[cat] * my_totals[cat]

    # Ratio stat terms should sum to ~0 for current starters
    hitters = starter_df[starter_df["player_type"] == "hitter"]
    pitchers = starter_df[starter_df["player_type"] == "pitcher"]

    ops_term = (
        gradient["OPS"]
        * float((hitters["PA"] * (hitters["OPS"] - my_totals["OPS"])).sum())
        / my_totals["PA"]
    )

    era_term = (
        gradient["ERA"]
        * float((pitchers["IP"] * (pitchers["ERA"] - my_totals["ERA"])).sum())
        / my_totals["IP"]
    )

    whip_term = (
        gradient["WHIP"]
        * float((pitchers["IP"] * (pitchers["WHIP"] - my_totals["WHIP"])).sum())
        / my_totals["IP"]
    )

    print(f"\nΣ MEW(starters) = {total_mew:.4f}")
    print(f"  Counting stat contribution (Σ g_c × T_c): {counting_contribution:.4f}")
    print(f"  Ratio OPS contribution: {ops_term:.6f}")
    print(f"  Ratio ERA contribution: {era_term:.6f}")
    print(f"  Ratio WHIP contribution: {whip_term:.6f}")
    print(f"  Sum of ratio terms: {ops_term + era_term + whip_term:.6f} (should be ~0)")

    print(f"\nEW = {current_ew:.4f}")
    print(f"Ratio: Σ MEW / EW = {total_mew / current_ew:.2f}x")

    print(f"\nThese are DIFFERENT QUANTITIES:")
    print(f"  Σ MEW = Σ_c g_c × T_c  (dot product of gradient and team totals)")
    print(f"  EW    = Σ_c Σ_o Φ(z_{{c,o}})  (sum of 60 CDF values)")
    print(f"  They share the same gradient but apply it differently.")

    # Show per-category breakdown
    print(f"\nPer-category breakdown:")
    print(f"{'Cat':>6} {'g_c':>10} {'T_c':>10} {'g*T':>10} {'Σ_o Φ(z)':>10}")
    print("-" * 52)

    _, diag = compute_win_probability(
        my_totals, state["opponent_totals"], state["category_sigmas"]
    )
    for cat in ALL_CATEGORIES:
        g = gradient[cat]
        t = my_totals[cat]
        g_t = g * t
        cat_ew = sum(diag["beat_probs"][cat].values())
        print(f"{cat:>6} {g:>10.4f} {t:>10.2f} {g_t:>10.2f} {cat_ew:>10.4f}")

    return total_mew, current_ew


# ============================================================================
# ANALYSIS 2: Ordinal consistency — lineup ranking
# ============================================================================


def analyze_ordinal_consistency(players, state):
    """Compare MEW ranking vs EW ranking across many alternative lineups.

    For each bench player, swap them in for each starter they're position-eligible
    to replace. Compute both Σ MEW and exact EW for the modified lineup.
    Check if higher Σ MEW always means higher EW.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Ordinal consistency — does higher Σ MEW mean higher EW?")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_starters = state["my_starters"]
    my_bench = my_roster - my_starters
    my_lineup = state["my_lineup"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    current_ew = state["current_ew"]
    gradient = state["gradient"]
    my_totals = state["my_totals"]

    starter_mew = players[players["Name"].isin(my_starters)].set_index("Name")["MEW"]
    bench_mew = players[players["Name"].isin(my_bench)].set_index("Name")["MEW"]

    baseline_sum_mew = float(starter_mew.sum())

    # Generate alternative lineups by swapping one bench player in for one starter
    swap_results = []

    for bench_name in sorted(my_bench):
        bench_row = players[players["Name"] == bench_name].iloc[0]
        bench_eligible = get_eligible_slots(str(bench_row["Position"]))

        for starter_name, slot in my_lineup.items():
            if slot not in bench_eligible:
                continue

            # Compute Σ MEW for modified lineup
            delta_mew = float(bench_mew[bench_name]) - float(starter_mew[starter_name])
            new_sum_mew = baseline_sum_mew + delta_mew

            # Compute exact EW for modified lineup
            new_starters = (my_starters - {starter_name}) | {bench_name}
            new_totals = compute_totals_for_starters(new_starters, players)
            new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)

            swap_results.append(
                {
                    "bench": strip_name_suffix(bench_name),
                    "bench_name_full": bench_name,
                    "starter": strip_name_suffix(starter_name),
                    "starter_name_full": starter_name,
                    "slot": slot,
                    "delta_mew": delta_mew,
                    "new_sum_mew": new_sum_mew,
                    "delta_ew": new_ew - current_ew,
                    "new_ew": new_ew,
                }
            )

    df = pd.DataFrame(swap_results)
    if len(df) == 0:
        print("No valid bench-for-starter swaps found.")
        return df

    print(f"\nGenerated {len(df)} bench-for-starter swap scenarios")

    # Check ordinal consistency: does sign(ΔMEW) == sign(ΔEW)?
    df["mew_says_good"] = df["delta_mew"] > 0
    df["ew_says_good"] = df["delta_ew"] > 0
    df["ordinal_agree"] = df["mew_says_good"] == df["ew_says_good"]

    n_agree = df["ordinal_agree"].sum()
    n_total = len(df)
    pct_agree = 100 * n_agree / n_total

    print(f"\nORDINAL CONSISTENCY (sign of ΔMEW vs sign of ΔEW):")
    print(f"  Agree: {n_agree}/{n_total} ({pct_agree:.1f}%)")
    print(f"  Disagree: {n_total - n_agree}/{n_total} ({100 - pct_agree:.1f}%)")

    # Show disagreements
    disagree = df[~df["ordinal_agree"]].sort_values("delta_ew", ascending=False)
    if len(disagree) > 0:
        print(f"\nDISAGREEMENTS (MEW and EW disagree on direction):")
        for _, row in disagree.iterrows():
            direction = "MEW+/EW-" if row["delta_mew"] > 0 else "MEW-/EW+"
            print(
                f"  {direction}: {row['bench']} for {row['starter']} at {row['slot']} "
                f"(ΔMEW={row['delta_mew']:.4f}, ΔEW={row['delta_ew']:.4f})"
            )

    # Correlation
    corr = df["delta_mew"].corr(df["delta_ew"])
    print(f"\nCorrelation(ΔMEW, ΔEW) = {corr:.6f}")

    # Regression: ΔEW = α + β × ΔMEW
    from numpy.polynomial.polynomial import polyfit

    beta, alpha = np.polyfit(df["delta_mew"], df["delta_ew"], 1)
    residuals = df["delta_ew"] - (alpha + beta * df["delta_mew"])
    rmse = float(np.sqrt((residuals**2).mean()))
    print(f"Linear fit: ΔEW ≈ {alpha:.6f} + {beta:.6f} × ΔMEW")
    print(f"  RMSE = {rmse:.6f} EW units")
    print(f"  β (scaling factor): {beta:.6f}")

    # Show largest magnitude swaps
    print(f"\nLargest ΔMEW swaps and their actual ΔEW:")
    top = df.nlargest(5, "delta_mew")
    for _, row in top.iterrows():
        print(
            f"  {row['bench']:>20s} for {row['starter']:<20s} at {row['slot']}: "
            f"ΔMEW={row['delta_mew']:+.4f}, ΔEW={row['delta_ew']:+.4f}"
        )
    bot = df.nsmallest(5, "delta_mew")
    for _, row in bot.iterrows():
        print(
            f"  {row['bench']:>20s} for {row['starter']:<20s} at {row['slot']}: "
            f"ΔMEW={row['delta_mew']:+.4f}, ΔEW={row['delta_ew']:+.4f}"
        )

    return df


# ============================================================================
# ANALYSIS 3: Curvature — the Hessian of EW
# ============================================================================


def analyze_curvature(state):
    """Compute the Hessian of EW and quantify nonlinearity."""
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Curvature of the EW surface (Hessian)")
    print("=" * 70)

    my_totals = state["my_totals"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    gradient = state["gradient"]

    print(f"\nPer-category curvature (∂²EW/∂(my_c)²):")
    print(
        f"{'Cat':>6} {'g_c':>10} {'d²EW/dc²':>12} {'z_avg':>8} {'z_min':>8} {'z_max':>8}"
    )
    print("-" * 60)

    curvature = {}
    z_scores = {}

    for cat in ALL_CATEGORIES:
        sigma = max(sigmas[cat], 0.001)
        denom = sigma * np.sqrt(2)

        zs = []
        for opp_totals in opponent_totals.values():
            if cat in NEGATIVE_CATEGORIES:
                z = (opp_totals[cat] - my_totals[cat]) / denom
            else:
                z = (my_totals[cat] - opp_totals[cat]) / denom
            zs.append(z)

        zs = np.array(zs)
        z_scores[cat] = zs

        # ∂²EW/∂(my_c)² = -Σ_o z_{c,o} × φ(z_{c,o}) / (σ_c √2)²
        # For C⁻ (ERA, WHIP): the chain rule gives an extra (-1)² = 1, same formula
        second_deriv = -np.sum(zs * stats.norm.pdf(zs)) / denom**2
        curvature[cat] = second_deriv

        print(
            f"{cat:>6} {gradient[cat]:>10.4f} {second_deriv:>12.6f} "
            f"{np.mean(zs):>8.3f} {np.min(zs):>8.3f} {np.max(zs):>8.3f}"
        )

    # Interpret curvature
    print(f"\nInterpretation:")
    for cat in ALL_CATEGORIES:
        zs = z_scores[cat]
        h = curvature[cat]
        if h < 0:
            regime = "CONCAVE (diminishing returns — you're mostly winning)"
        elif h > 0:
            regime = "CONVEX (accelerating returns — you're mostly losing)"
        else:
            regime = "FLAT (near inflection point)"
        print(f"  {cat}: {regime}")
        print(f"    z-scores vs opponents: {', '.join(f'{z:.2f}' for z in zs)}")

    # Quantify second-order error for a typical lineup swap
    print(f"\nSecond-order error estimate for typical swap (Δ = 10% of team total):")
    for cat in ["HR", "K", "OPS", "ERA"]:
        if cat in ("OPS", "ERA", "WHIP"):
            # For ratio stats, a typical swap changes the average by ~1-2%
            delta = my_totals[cat] * 0.02 if cat == "OPS" else my_totals[cat] * 0.03
        else:
            delta = my_totals[cat] * 0.10
        first_order = abs(gradient[cat] * delta)
        second_order = abs(0.5 * curvature[cat] * delta**2)
        ratio = second_order / first_order if first_order > 0 else float("inf")
        print(
            f"  {cat}: Δ={delta:.2f}, |1st order|={first_order:.4f}, "
            f"|2nd order|={second_order:.4f}, ratio={ratio:.4f}"
        )

    return curvature, z_scores


# ============================================================================
# ANALYSIS 4: Direct comparison of MEW-optimal vs EW-optimal lineups
# ============================================================================


def find_ew_optimal_lineup(players, state):
    """Brute-force search among feasible lineups to find EW-optimal.

    We can't enumerate all lineups, but we can:
    1. Solve the MEW-optimal lineup (current)
    2. Solve the FV-optimal lineup (alternative)
    3. Perturb the MEW-optimal lineup systematically
    4. Compute EW for each and compare
    """
    print("\n" + "=" * 70)
    print("ANALYSIS 4: Is the MEW-optimal lineup also EW-optimal?")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_lineup = state["my_lineup"]
    my_starters = state["my_starters"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    current_ew = state["current_ew"]
    my_totals = state["my_totals"]
    gradient = state["gradient"]

    # Current (MEW-optimal) lineup EW
    print(f"\nCurrent MEW-optimal lineup: EW = {current_ew:.4f}")
    mew_sum = float(players[players["Name"].isin(my_starters)]["MEW"].sum())
    print(f"  Σ MEW = {mew_sum:.4f}")

    # FV-optimal lineup
    fv_lineup = solve_lineup(my_roster, players, "FV")
    fv_starters = set(fv_lineup.keys())
    fv_totals = compute_totals_for_starters(fv_starters, players)
    fv_ew, _ = compute_win_probability(fv_totals, opponent_totals, sigmas)
    fv_mew_sum = float(players[players["Name"].isin(fv_starters)]["MEW"].sum())
    print(f"\nFV-optimal lineup: EW = {fv_ew:.4f}")
    print(f"  Σ MEW = {fv_mew_sum:.4f}")

    # Show the differences
    mew_only = my_starters - fv_starters
    fv_only = fv_starters - my_starters
    if mew_only or fv_only:
        print(f"\n  Players in MEW lineup but not FV lineup:")
        for n in sorted(mew_only):
            row = players[players["Name"] == n].iloc[0]
            print(
                f"    {strip_name_suffix(n)} (MEW={row['MEW']:.3f}, FV={row['FV']:.3f})"
            )
        print(f"  Players in FV lineup but not MEW lineup:")
        for n in sorted(fv_only):
            row = players[players["Name"] == n].iloc[0]
            print(
                f"    {strip_name_suffix(n)} (MEW={row['MEW']:.3f}, FV={row['FV']:.3f})"
            )
    else:
        print("  MEW and FV lineups are identical!")

    # Now do an exhaustive local search: try all 1-swap perturbations of MEW lineup
    # and see if any has higher EW
    print(f"\nLocal search: trying all 1-swap perturbations of MEW-optimal lineup...")

    my_bench = my_roster - my_starters
    best_ew = current_ew
    best_swap = None
    improvements = []

    for bench_name in sorted(my_bench):
        bench_row = players[players["Name"] == bench_name].iloc[0]
        bench_eligible = get_eligible_slots(str(bench_row["Position"]))

        for starter_name, slot in my_lineup.items():
            if slot not in bench_eligible:
                continue

            new_starters = (my_starters - {starter_name}) | {bench_name}
            new_totals = compute_totals_for_starters(new_starters, players)
            new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)

            if new_ew > current_ew:
                improvements.append(
                    {
                        "bench": strip_name_suffix(bench_name),
                        "starter": strip_name_suffix(starter_name),
                        "slot": slot,
                        "delta_ew": new_ew - current_ew,
                    }
                )

            if new_ew > best_ew:
                best_ew = new_ew
                best_swap = (bench_name, starter_name, slot)

    if improvements:
        print(
            f"\n  FOUND {len(improvements)} 1-swap improvement(s) over MEW-optimal lineup!"
        )
        print(f"  The MEW-optimal lineup is NOT locally EW-optimal.")
        for imp in sorted(improvements, key=lambda x: -x["delta_ew"]):
            print(
                f"    {imp['bench']} for {imp['starter']} at {imp['slot']}: "
                f"ΔEW = +{imp['delta_ew']:.4f}"
            )
        print(f"\n  Best improvement: ΔEW = +{best_ew - current_ew:.4f}")
        print(f"  MEW-optimal EW: {current_ew:.4f}")
        print(f"  Best local EW:  {best_ew:.4f}")
        print(f"  EW gap from linearization: {best_ew - current_ew:.4f}")
    else:
        print(
            f"\n  MEW-optimal lineup IS locally EW-optimal (no 1-swap improvement found)."
        )

    return {
        "mew_ew": current_ew,
        "fv_ew": fv_ew,
        "best_local_ew": best_ew,
        "n_improvements": len(improvements),
    }


# ============================================================================
# ANALYSIS 5: Swap-level accuracy for FA screening
# ============================================================================


def analyze_swap_accuracy(players, state):
    """Compare MEW-based MSV_approx vs exact MSV for top FA candidates."""
    print("\n" + "=" * 70)
    print("ANALYSIS 5: Swap-level accuracy (MSV_approx vs MSV_exact)")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_starters = state["my_starters"]
    my_bench = my_roster - my_starters
    my_lineup = state["my_lineup"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    current_ew = state["current_ew"]
    gradient = state["gradient"]
    my_totals = state["my_totals"]

    # Get top FAs by MEW
    fa_mask = players["owner"].isna()
    fas = players[fa_mask].nlargest(20, "MEW")

    # Get droppable bench players (lowest MEW)
    bench_df = players[players["Name"].isin(my_bench)].nsmallest(5, "MEW")

    results = []
    for _, fa_row in fas.iterrows():
        fa_name = fa_row["Name"]
        fa_mew = fa_row["MEW"]

        for _, bench_row in bench_df.iterrows():
            drop_name = bench_row["Name"]
            drop_mew = bench_row["MEW"]

            # MSV_approx = MEW(add) - MEW(drop)
            msv_approx = fa_mew - drop_mew

            # MSV_exact: re-solve lineup and compute EW
            new_roster = (my_roster - {drop_name}) | {fa_name}
            new_lineup = solve_lineup(new_roster, players, "MEW")
            new_starters = set(new_lineup.keys())
            new_totals = compute_totals_for_starters(new_starters, players)
            new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
            msv_exact = new_ew - current_ew

            results.append(
                {
                    "fa": strip_name_suffix(fa_name),
                    "drop": strip_name_suffix(drop_name),
                    "fa_mew": fa_mew,
                    "drop_mew": drop_mew,
                    "msv_approx": msv_approx,
                    "msv_exact": msv_exact,
                    "error": msv_approx - msv_exact,
                    "pct_error": (
                        100 * (msv_approx - msv_exact) / abs(msv_exact)
                        if abs(msv_exact) > 0.001
                        else float("nan")
                    ),
                }
            )

    df = pd.DataFrame(results)

    # Ordinal consistency for swaps
    n_pairs = 0
    n_agree = 0
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            n_pairs += 1
            approx_order = df.iloc[i]["msv_approx"] > df.iloc[j]["msv_approx"]
            exact_order = df.iloc[i]["msv_exact"] > df.iloc[j]["msv_exact"]
            if approx_order == exact_order:
                n_agree += 1

    if n_pairs > 0:
        print(
            f"\nPairwise ordinal consistency: {n_agree}/{n_pairs} ({100 * n_agree / n_pairs:.1f}%)"
        )

    corr = df["msv_approx"].corr(df["msv_exact"])
    print(f"Correlation(MSV_approx, MSV_exact) = {corr:.4f}")

    # Error statistics
    print(f"\nError statistics (MSV_approx - MSV_exact):")
    print(f"  Mean error:   {df['error'].mean():.4f}")
    print(f"  Median error: {df['error'].median():.4f}")
    print(f"  Std error:    {df['error'].std():.4f}")
    print(f"  Max |error|:  {df['error'].abs().max():.4f}")

    # Show top swaps
    print(f"\nTop 10 swaps by MSV_approx vs their exact MSV:")
    top = df.nlargest(10, "msv_approx")
    print(
        f"{'FA':>20s} {'Drop':>20s} {'MSVapx':>8} {'MSVexact':>8} {'Error':>8} {'%Err':>8}"
    )
    for _, row in top.iterrows():
        pct = f"{row['pct_error']:.1f}%" if not np.isnan(row["pct_error"]) else "  N/A"
        print(
            f"{row['fa']:>20s} {row['drop']:>20s} "
            f"{row['msv_approx']:>8.4f} {row['msv_exact']:>8.4f} "
            f"{row['error']:>8.4f} {pct:>8s}"
        )

    # Critical question: would the best MSV_approx swap also be the best MSV_exact swap?
    best_approx_idx = df["msv_approx"].idxmax()
    best_exact_idx = df["msv_exact"].idxmax()
    print(
        f"\nBest swap by MSV_approx: {df.loc[best_approx_idx, 'fa']} for {df.loc[best_approx_idx, 'drop']} ({df.loc[best_approx_idx, 'msv_approx']:.4f})"
    )
    print(
        f"Best swap by MSV_exact:  {df.loc[best_exact_idx, 'fa']} for {df.loc[best_exact_idx, 'drop']} ({df.loc[best_exact_idx, 'msv_exact']:.4f})"
    )
    print(f"Same best swap? {best_approx_idx == best_exact_idx}")

    return df


# ============================================================================
# ANALYSIS 6: The z-score regime — where nonlinearity lives
# ============================================================================


def analyze_z_score_regime(state):
    """Detailed analysis of z-scores and where the approximation is most/least valid."""
    print("\n" + "=" * 70)
    print("ANALYSIS 6: z-score regime analysis")
    print("=" * 70)

    my_totals = state["my_totals"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    gradient = state["gradient"]

    _, diag = compute_win_probability(my_totals, opponent_totals, sigmas)
    z_scores = diag["normalized_gaps"]
    beat_probs = diag["beat_probs"]

    # Φ is approximately linear near z=0 (slope = φ(0)/σ√2).
    # The linearization error grows as |z| increases because φ(z) → 0
    # (the gradient at the current point overstates sensitivity in decided categories).

    print(f"\nz-scores (positive = winning):")
    print(f"{'Cat':>6}", end="")
    for opp_id in sorted(opponent_totals.keys()):
        print(f" {'Opp' + str(opp_id):>8}", end="")
    print(f" {'Σ|z|':>8} {'Regime':>12}")
    print("-" * (6 + 9 * len(opponent_totals) + 22))

    for cat in ALL_CATEGORIES:
        print(f"{cat:>6}", end="")
        zs = []
        for opp_id in sorted(opponent_totals.keys()):
            z = z_scores[cat][opp_id]
            zs.append(z)
            print(f" {z:>8.3f}", end="")
        total_abs_z = sum(abs(z) for z in zs)
        avg_abs_z = np.mean(np.abs(zs))

        if avg_abs_z < 0.5:
            regime = "COMPETITIVE"
        elif avg_abs_z < 1.5:
            regime = "LEANING"
        else:
            regime = "DECIDED"

        print(f" {total_abs_z:>8.2f} {regime:>12}")

    # Show beat probabilities
    print(f"\nBeat probabilities Φ(z):")
    print(f"{'Cat':>6}", end="")
    for opp_id in sorted(opponent_totals.keys()):
        print(f" {'Opp' + str(opp_id):>8}", end="")
    print(f" {'Sum':>8}")
    print("-" * (6 + 9 * (len(opponent_totals) + 1)))

    for cat in ALL_CATEGORIES:
        print(f"{cat:>6}", end="")
        cat_sum = 0
        for opp_id in sorted(opponent_totals.keys()):
            p = beat_probs[cat][opp_id]
            cat_sum += p
            print(f" {p:>8.4f}", end="")
        print(f" {cat_sum:>8.4f}")

    # Gradient density: φ(z) — this is what drives the gradient
    print(f"\nGradient density φ(z) — contribution to gradient from each opponent:")
    print(f"{'Cat':>6}", end="")
    for opp_id in sorted(opponent_totals.keys()):
        print(f" {'Opp' + str(opp_id):>8}", end="")
    print(f" {'Total':>8} {'g_c':>10}")
    print("-" * (6 + 9 * (len(opponent_totals) + 1) + 11))

    for cat in ALL_CATEGORIES:
        print(f"{cat:>6}", end="")
        total_phi = 0
        for opp_id in sorted(opponent_totals.keys()):
            z = z_scores[cat][opp_id]
            phi = stats.norm.pdf(z)
            total_phi += phi
            print(f" {phi:>8.4f}", end="")
        print(f" {total_phi:>8.4f} {gradient[cat]:>10.4f}")


# ============================================================================
# ANALYSIS 7: Sensitivity — how much does gradient change across lineup swaps?
# ============================================================================


def analyze_gradient_stability(players, state):
    """Measure how much the gradient changes when the lineup changes."""
    print("\n" + "=" * 70)
    print("ANALYSIS 7: Gradient stability across lineup changes")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_starters = state["my_starters"]
    my_bench = my_roster - my_starters
    my_lineup = state["my_lineup"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    gradient = state["gradient"]
    my_totals = state["my_totals"]

    # For each valid bench-for-starter swap, compute the new gradient
    # and compare to the current gradient
    swap_gradients = []

    for bench_name in sorted(my_bench):
        bench_row = players[players["Name"] == bench_name].iloc[0]
        bench_eligible = get_eligible_slots(str(bench_row["Position"]))

        for starter_name, slot in my_lineup.items():
            if slot not in bench_eligible:
                continue

            new_starters = (my_starters - {starter_name}) | {bench_name}
            new_totals = compute_totals_for_starters(new_starters, players)
            new_gradient = compute_ew_gradient(new_totals, opponent_totals, sigmas)

            max_pct_change = 0.0
            changes = {}
            for cat in ALL_CATEGORIES:
                old_g = gradient[cat]
                new_g = new_gradient[cat]
                if abs(old_g) > 1e-10:
                    pct = 100 * (new_g - old_g) / abs(old_g)
                else:
                    pct = 0.0
                changes[cat] = pct
                max_pct_change = max(max_pct_change, abs(pct))

            swap_gradients.append(
                {
                    "bench": strip_name_suffix(bench_name),
                    "starter": strip_name_suffix(starter_name),
                    "slot": slot,
                    "max_pct_change": max_pct_change,
                    "changes": changes,
                }
            )

    df = pd.DataFrame(swap_gradients)
    print(f"\n{len(df)} bench-for-starter swaps analyzed")

    print(f"\nGradient change statistics (max % change in any g_c):")
    print(f"  Mean:   {df['max_pct_change'].mean():.2f}%")
    print(f"  Median: {df['max_pct_change'].median():.2f}%")
    print(f"  Max:    {df['max_pct_change'].max():.2f}%")
    print(f"  90th percentile: {df['max_pct_change'].quantile(0.9):.2f}%")

    # Show the most destabilizing swaps
    print(f"\nMost destabilizing swaps (largest gradient change):")
    top = df.nlargest(5, "max_pct_change")
    for _, row in top.iterrows():
        print(
            f"  {row['bench']:>20s} for {row['starter']:<20s} at {row['slot']}: "
            f"max Δg = {row['max_pct_change']:.2f}%"
        )
        big_changes = {k: v for k, v in row["changes"].items() if abs(v) > 2.0}
        if big_changes:
            for cat, pct in sorted(big_changes.items(), key=lambda x: -abs(x[1])):
                print(f"      {cat}: {pct:+.2f}%")

    return df


# ============================================================================
# ANALYSIS 8: Comprehensive EW surface exploration
# ============================================================================


def analyze_ew_surface(players, state):
    """Explore the EW surface more systematically by varying team totals."""
    print("\n" + "=" * 70)
    print("ANALYSIS 8: EW surface exploration — nonlinearity quantification")
    print("=" * 70)

    my_totals = state["my_totals"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    gradient = state["gradient"]
    current_ew = state["current_ew"]

    # For each category, sweep team total from -20% to +20% of current value
    # and compare actual EW vs linear prediction from gradient
    for cat in ["HR", "SB", "K", "OPS", "ERA", "SV"]:
        print(f"\n--- {cat} sweep ---")
        base_val = my_totals[cat]

        if cat in ("OPS", "ERA", "WHIP"):
            deltas = np.linspace(-0.05 * base_val, 0.05 * base_val, 21)
        else:
            deltas = np.linspace(-0.30 * base_val, 0.30 * base_val, 21)

        print(f"  Base value: {base_val:.3f}, gradient: {gradient[cat]:.6f}")
        print(
            f"  {'Delta':>10} {'Actual EW':>10} {'Linear EW':>10} {'Error':>10} {'%Err':>8}"
        )

        for delta in deltas:
            modified = dict(my_totals)
            modified[cat] = base_val + delta
            actual_ew, _ = compute_win_probability(modified, opponent_totals, sigmas)
            linear_ew = current_ew + gradient[cat] * delta
            error = linear_ew - actual_ew
            pct_err = 100 * error / abs(actual_ew) if abs(actual_ew) > 0.01 else 0

            marker = " ***" if abs(pct_err) > 1.0 else ""
            print(
                f"  {delta:>10.3f} {actual_ew:>10.4f} {linear_ew:>10.4f} "
                f"{error:>10.4f} {pct_err:>7.2f}%{marker}"
            )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    players, state = load_and_prepare()

    analyze_mew_sum_vs_ew(players, state)
    swap_df = analyze_ordinal_consistency(players, state)
    analyze_curvature(state)
    find_ew_optimal_lineup(players, state)
    analyze_swap_accuracy(players, state)
    analyze_z_score_regime(state)
    analyze_gradient_stability(players, state)
    analyze_ew_surface(players, state)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
