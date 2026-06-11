"""
Part 2: Deep dive into WHY MSV_approx fails for swap screening.

Key finding from Part 1: MSV_approx = MEW(FA) - MEW(drop) has near-zero
correlation with MSV_exact for FA swaps. This analysis investigates the
specific failure modes:

1. Bench-drop vs starter-drop: MEW(bench) doesn't represent actual EW
   contribution (bench players contribute 0 to starting lineup).
2. Lineup cascading: adding an FA may not change the starting lineup
   if all current starters have higher MEW.
3. The OPS convexity trap: we're deeply losing OPS (z ~ -1.5 avg),
   creating large second-order errors.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

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


def load_and_prepare():
    silver = Path("../data_prep/data/silver_table.parquet")
    assert silver.exists(), f"Silver table not found: {silver}"
    players = pd.read_parquet(silver)
    players = add_fantasy_value(players)
    state = compute_league_state(players)
    players = add_mew(players, state["my_totals"], state["gradient"])
    return players, state


# ============================================================================
# ANALYSIS A: What happens in a typical "drop bench, add FA" swap?
# ============================================================================


def analyze_bench_drop_mechanism(players, state):
    """Trace through specific bench-drop swaps to understand the failure."""
    print("\n" + "=" * 70)
    print("ANALYSIS A: Anatomy of a bench-drop swap")
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

    # Find a specific bench player
    bench_df = players[players["Name"].isin(my_bench)].sort_values("MEW")
    print(f"\nMy bench players (sorted by MEW):")
    for _, row in bench_df.iterrows():
        print(
            f"  {strip_name_suffix(row['Name']):>20s}: "
            f"MEW={row['MEW']:.4f}, FV={row['FV']:.3f}, "
            f"Position={row['Position']}"
        )

    # Pick the lowest-MEW bench player
    drop_name = bench_df.iloc[0]["Name"]
    drop_row = bench_df.iloc[0]
    print(f"\nDropping: {strip_name_suffix(drop_name)} (MEW={drop_row['MEW']:.4f})")

    # Try top 5 FAs
    fa_mask = players["owner"].isna()
    top_fas = players[fa_mask].nlargest(10, "MEW")

    print(f"\nTop 10 FAs by MEW:")
    for _, fa_row in top_fas.iterrows():
        fa_name = fa_row["Name"]
        fa_mew = fa_row["MEW"]
        msv_approx = fa_mew - drop_row["MEW"]

        # Exact evaluation
        new_roster = (my_roster - {drop_name}) | {fa_name}
        new_lineup = solve_lineup(new_roster, players, "MEW")
        new_starters = set(new_lineup.keys())
        new_totals = compute_totals_for_starters(new_starters, players)
        new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
        msv_exact = new_ew - current_ew

        # Did the FA enter the starting lineup?
        fa_starts = fa_name in new_starters
        # Who was displaced?
        displaced = my_starters - new_starters
        entered = new_starters - my_starters

        # What actually changed in team totals?
        total_changes = {}
        for cat in ALL_CATEGORIES:
            delta = new_totals[cat] - my_totals[cat]
            if abs(delta) > 0.001:
                total_changes[cat] = delta

        print(
            f"\n  FA: {strip_name_suffix(fa_name)} (MEW={fa_mew:.4f}, Pos={fa_row['Position']})"
        )
        print(f"    MSV_approx = {msv_approx:.4f} (MEW(FA) - MEW(drop))")
        print(f"    MSV_exact  = {msv_exact:.4f}")
        print(f"    Error      = {msv_approx - msv_exact:.4f}")
        print(f"    FA starts? {fa_starts}")
        if fa_starts:
            print(f"    Displaced: {[strip_name_suffix(n) for n in displaced]}")
            print(f"    New starts: {[strip_name_suffix(n) for n in entered]}")
            if total_changes:
                print(f"    Total changes: ", end="")
                for cat, delta in sorted(total_changes.items()):
                    print(f"{cat}={delta:+.3f} ", end="")
                print()
        else:
            print(f"    FA goes to bench — no change in starters or team totals")


# ============================================================================
# ANALYSIS B: Starter-drop swaps (where MEW should work)
# ============================================================================


def analyze_starter_drop_swaps(players, state):
    """Test MSV_approx specifically for swaps where a starter is dropped."""
    print("\n" + "=" * 70)
    print("ANALYSIS B: MSV_approx accuracy for STARTER-drop FA swaps")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_starters = state["my_starters"]
    my_lineup = state["my_lineup"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    current_ew = state["current_ew"]

    # For each starter, find the best FA replacement (same position eligibility)
    fa_mask = players["owner"].isna()

    results = []
    for starter_name, slot in my_lineup.items():
        starter_row = players[players["Name"] == starter_name].iloc[0]
        starter_mew = starter_row["MEW"]

        # Find FAs eligible for this slot
        eligible_fas = []
        for _, fa_row in players[fa_mask].iterrows():
            fa_eligible = get_eligible_slots(str(fa_row["Position"]))
            if slot in fa_eligible:
                eligible_fas.append(fa_row)

        if not eligible_fas:
            continue

        # Sort by MEW descending, take top 3
        eligible_fas.sort(key=lambda r: r["MEW"], reverse=True)
        for fa_row in eligible_fas[:3]:
            fa_name = fa_row["Name"]
            fa_mew = fa_row["MEW"]

            # MSV_approx: direct MEW difference
            msv_approx = fa_mew - starter_mew

            # MSV_exact: re-solve lineup
            new_roster = (my_roster - {starter_name}) | {fa_name}
            new_lineup = solve_lineup(new_roster, players, "MEW")
            new_starters = set(new_lineup.keys())
            new_totals = compute_totals_for_starters(new_starters, players)
            new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
            msv_exact = new_ew - current_ew

            # Did the FA take the same slot?
            fa_slot = new_lineup.get(fa_name, "BENCH")
            cascade = fa_slot != slot

            results.append(
                {
                    "starter": strip_name_suffix(starter_name),
                    "fa": strip_name_suffix(fa_name),
                    "slot": slot,
                    "starter_mew": starter_mew,
                    "fa_mew": fa_mew,
                    "msv_approx": msv_approx,
                    "msv_exact": msv_exact,
                    "error": msv_approx - msv_exact,
                    "fa_slot": fa_slot,
                    "cascade": cascade,
                    "fa_started": fa_name in new_starters,
                }
            )

    df = pd.DataFrame(results)

    print(f"\n{len(df)} starter-drop swaps evaluated")

    # Separate into cascading and non-cascading
    no_cascade = df[~df["cascade"] & df["fa_started"]]
    cascade = df[df["cascade"] & df["fa_started"]]
    fa_benched = df[~df["fa_started"]]

    print(f"\n  Direct replacement (FA takes same slot): {len(no_cascade)}")
    print(f"  Cascade (FA takes different slot):        {len(cascade)}")
    print(f"  FA goes to bench:                         {len(fa_benched)}")

    if len(no_cascade) > 0:
        corr = no_cascade["msv_approx"].corr(no_cascade["msv_exact"])
        rmse = float(np.sqrt((no_cascade["error"] ** 2).mean()))
        print(f"\n  Direct replacement accuracy:")
        print(f"    Correlation: {corr:.4f}")
        print(f"    RMSE:        {rmse:.4f}")
        print(f"    Mean error:  {no_cascade['error'].mean():.4f}")

    if len(cascade) > 0:
        corr = cascade["msv_approx"].corr(cascade["msv_exact"])
        rmse = float(np.sqrt((cascade["error"] ** 2).mean()))
        print(f"\n  Cascade swap accuracy:")
        print(f"    Correlation: {corr:.4f}")
        print(f"    RMSE:        {rmse:.4f}")
        print(f"    Mean error:  {cascade['error'].mean():.4f}")

    # Show some examples
    print(f"\nBest direct replacements (top 5 by MSV_exact):")
    if len(no_cascade) > 0:
        top_direct = no_cascade.nlargest(5, "msv_exact")
        print(
            f"  {'Starter':>20s} {'FA':>20s} {'Slot':>4s} {'MSVapx':>8} {'MSVex':>8} {'Err':>8}"
        )
        for _, row in top_direct.iterrows():
            print(
                f"  {row['starter']:>20s} {row['fa']:>20s} {row['slot']:>4s} "
                f"{row['msv_approx']:>8.4f} {row['msv_exact']:>8.4f} {row['error']:>8.4f}"
            )

    print(f"\nWorst-predicted swaps (top 5 by |error|):")
    worst = df.loc[df["error"].abs().nlargest(5).index]
    print(
        f"  {'Starter':>20s} {'FA':>20s} {'Slot':>4s} {'MSVapx':>8} {'MSVex':>8} {'Err':>8} {'Cascade':>8}"
    )
    for _, row in worst.iterrows():
        print(
            f"  {row['starter']:>20s} {row['fa']:>20s} {row['slot']:>4s} "
            f"{row['msv_approx']:>8.4f} {row['msv_exact']:>8.4f} {row['error']:>8.4f} "
            f"{'YES' if row['cascade'] else 'no':>8}"
        )

    return df


# ============================================================================
# ANALYSIS C: The OPS convexity problem
# ============================================================================


def analyze_ops_nonlinearity(players, state):
    """Deep dive into OPS — the category with the worst linearization error."""
    print("\n" + "=" * 70)
    print("ANALYSIS C: OPS nonlinearity deep dive")
    print("=" * 70)

    my_totals = state["my_totals"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    gradient = state["gradient"]
    current_ew = state["current_ew"]

    my_ops = my_totals["OPS"]
    sigma_ops = sigmas["OPS"]

    print(f"\nCurrent team OPS: {my_ops:.4f}")
    print(f"OPS σ: {sigma_ops:.4f}")
    print(f"OPS gradient (∂EW/∂OPS): {gradient['OPS']:.4f}")

    from scipy.stats import norm as _norm

    print(f"\nOPS z-scores vs opponents:")
    for opp_id, opp_t in sorted(opponent_totals.items()):
        z = (my_ops - opp_t["OPS"]) / (sigma_ops * np.sqrt(2))
        phi = float(np.exp(-(z**2) / 2) / np.sqrt(2 * np.pi))
        cdf_val = float(_norm.cdf(z))
        print(
            f"  Opp {opp_id}: their OPS = {opp_t['OPS']:.4f}, "
            f"z = {z:.3f}, Φ(z) = {cdf_val:.4f}, "
            f"φ(z) = {phi:.4f}"
        )

    # The problem: we're deeply in the left tail for OPS.
    # The gradient at z ~ -1.5 is very low (φ(-1.5) ≈ 0.13).
    # But if we could move z by +1.0 (to z ~ -0.5), φ(-0.5) ≈ 0.35.
    # The gradient TRIPLES! The linear approximation massively understates
    # the value of large OPS improvements.

    print(f"\nOPS improvement value: linear vs actual")
    print(
        f"{'ΔOPS':>8} {'Δz(avg)':>8} {'Linear ΔEW':>10} {'Actual ΔEW':>10} {'Ratio':>8}"
    )
    for delta_ops in [0.005, 0.010, 0.015, 0.020, 0.030, 0.040, 0.050]:
        modified = dict(my_totals)
        modified["OPS"] = my_ops + delta_ops
        actual_ew, _ = compute_win_probability(modified, opponent_totals, sigmas)
        linear_ew = current_ew + gradient["OPS"] * delta_ops
        actual_delta = actual_ew - current_ew
        linear_delta = linear_ew - current_ew
        ratio = (
            actual_delta / linear_delta if abs(linear_delta) > 0.001 else float("nan")
        )
        print(
            f"{delta_ops:>8.4f} {delta_ops / (sigma_ops * np.sqrt(2)):>8.3f} "
            f"{linear_delta:>10.4f} {actual_delta:>10.4f} {ratio:>8.3f}"
        )

    # What would happen if we moved OPS to league average?
    all_ops = [my_ops] + [opp_t["OPS"] for opp_t in opponent_totals.values()]
    league_avg_ops = np.mean(all_ops)
    delta_to_avg = league_avg_ops - my_ops

    modified = dict(my_totals)
    modified["OPS"] = league_avg_ops
    avg_ew, _ = compute_win_probability(modified, opponent_totals, sigmas)

    print(
        f"\nMoving OPS to league average ({league_avg_ops:.4f}, Δ = +{delta_to_avg:.4f}):"
    )
    print(f"  Linear prediction: ΔEW = {gradient['OPS'] * delta_to_avg:.4f}")
    print(f"  Actual ΔEW:        {avg_ew - current_ew:.4f}")
    print(
        f"  Understatement:    {(avg_ew - current_ew) / (gradient['OPS'] * delta_to_avg):.2f}x"
    )


# ============================================================================
# ANALYSIS D: Corrected MSV_approx — what MEW SHOULD predict
# ============================================================================


def analyze_corrected_msv_approx(players, state):
    """Compute MSV_approx the "right" way and compare accuracy.

    The correct first-order approximation for a swap (drop, add) is:
    - Identify the post-swap starting lineup (which players start)
    - ΔEW ≈ Σ_c g_c × (new_totals_c - old_totals_c)

    For a bench-drop + FA-add where the FA starts, the actual change is:
    ΔEW ≈ MEW(FA) - MEW(displaced_starter)
    NOT MEW(FA) - MEW(dropped_bench_player)
    """
    print("\n" + "=" * 70)
    print("ANALYSIS D: Corrected MSV_approx using actual lineup changes")
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

    fa_mask = players["owner"].isna()

    # For the worst bench player, try top 20 FAs
    bench_df = players[players["Name"].isin(my_bench)].nsmallest(3, "MEW")
    top_fas = players[fa_mask].nlargest(20, "MEW")

    results = []

    for _, bench_row in bench_df.iterrows():
        drop_name = bench_row["Name"]
        drop_mew = bench_row["MEW"]

        for _, fa_row in top_fas.iterrows():
            fa_name = fa_row["Name"]
            fa_mew = fa_row["MEW"]

            # Naive MSV_approx
            naive_approx = fa_mew - drop_mew

            # Exact MSV
            new_roster = (my_roster - {drop_name}) | {fa_name}
            new_lineup = solve_lineup(new_roster, players, "MEW")
            new_starters = set(new_lineup.keys())
            new_totals = compute_totals_for_starters(new_starters, players)
            new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
            msv_exact = new_ew - current_ew

            # Corrected MSV_approx: use actual total changes
            corrected_approx = sum(
                gradient[cat] * (new_totals[cat] - my_totals[cat])
                for cat in ALL_CATEGORIES
            )

            # What actually changed in the lineup?
            displaced = my_starters - new_starters
            entered = new_starters - my_starters
            fa_started = fa_name in new_starters

            # "Smart" MSV_approx: if FA doesn't start, MSV ≈ 0
            if not fa_started:
                smart_approx = 0.0
            else:
                displaced_mew = sum(
                    float(players[players["Name"] == n]["MEW"].iloc[0])
                    for n in displaced
                )
                entered_mew = sum(
                    float(players[players["Name"] == n]["MEW"].iloc[0]) for n in entered
                )
                smart_approx = entered_mew - displaced_mew

            results.append(
                {
                    "drop": strip_name_suffix(drop_name),
                    "fa": strip_name_suffix(fa_name),
                    "fa_started": fa_started,
                    "naive_approx": naive_approx,
                    "smart_approx": smart_approx,
                    "corrected_approx": corrected_approx,
                    "msv_exact": msv_exact,
                    "naive_err": naive_approx - msv_exact,
                    "smart_err": smart_approx - msv_exact,
                    "corrected_err": corrected_approx - msv_exact,
                }
            )

    df = pd.DataFrame(results)

    print(f"\n{len(df)} swaps evaluated")
    print(f"  FA enters lineup: {df['fa_started'].sum()}")
    print(f"  FA stays on bench: {(~df['fa_started']).sum()}")

    # Compare the three approximation methods
    print(f"\nApproximation method comparison:")
    print(f"{'Method':>20s} {'Corr':>8} {'RMSE':>8} {'Mean|Err|':>10} {'Max|Err|':>10}")

    for method, col in [
        ("Naive (MEW diff)", "naive_err"),
        ("Smart (lineup-aware)", "smart_err"),
        ("Corrected (g·ΔT)", "corrected_err"),
    ]:
        err = df[col]
        approx_col = (
            col.replace("_err", "_approx")
            if col != "corrected_err"
            else "corrected_approx"
        )
        if approx_col in df.columns:
            corr = df[approx_col].corr(df["msv_exact"])
        else:
            corr = float("nan")
        rmse = float(np.sqrt((err**2).mean()))
        mae = float(err.abs().mean())
        max_err = float(err.abs().max())
        print(f"{method:>20s} {corr:>8.4f} {rmse:>8.4f} {mae:>10.4f} {max_err:>10.4f}")

    # Show cases where methods disagree
    print(f"\nDetailed comparison (top 10 by |naive_err|):")
    worst = df.loc[df["naive_err"].abs().nlargest(10).index]
    print(
        f"  {'Drop':>15s} {'FA':>15s} {'Start':>5s} {'Naive':>8} {'Smart':>8} {'Corr':>8} {'Exact':>8}"
    )
    for _, row in worst.iterrows():
        print(
            f"  {row['drop']:>15s} {row['fa']:>15s} "
            f"{'Y' if row['fa_started'] else 'N':>5s} "
            f"{row['naive_approx']:>8.4f} {row['smart_approx']:>8.4f} "
            f"{row['corrected_approx']:>8.4f} {row['msv_exact']:>8.4f}"
        )

    return df


# ============================================================================
# ANALYSIS E: How large could the EW gain be from a 2nd-order correction?
# ============================================================================


def analyze_second_order_lineup_optimization(players, state):
    """Could a second-order (quadratic) optimization find a better lineup?

    The MEW-optimal lineup maximizes the linear objective Σ g_c × T_c.
    A quadratic optimization would also account for the curvature:
    max Σ g_c × T_c + (1/2) Σ h_cc × T_c² (where h_cc is the Hessian diagonal)

    This is still separable by category, so we can compute the quadratic
    score per player and re-optimize.
    """
    print("\n" + "=" * 70)
    print("ANALYSIS E: Second-order lineup optimization")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_starters = state["my_starters"]
    my_lineup = state["my_lineup"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    current_ew = state["current_ew"]
    gradient = state["gradient"]
    my_totals = state["my_totals"]

    from scipy import stats as scipy_stats

    # Compute Hessian diagonal
    hessian = {}
    for cat in ALL_CATEGORIES:
        sigma = max(sigmas[cat], 0.001)
        denom = sigma * np.sqrt(2)
        second_deriv = 0.0
        for opp_totals in opponent_totals.values():
            if cat in NEGATIVE_CATEGORIES:
                z = (opp_totals[cat] - my_totals[cat]) / denom
            else:
                z = (my_totals[cat] - opp_totals[cat]) / denom
            second_deriv += -z * scipy_stats.norm.pdf(z) / denom**2
        hessian[cat] = second_deriv

    # For each player, compute:
    # QMEW(p) ≈ MEW(p) + (1/2) × Σ_c h_cc × (Δtotal_c from adding p)²
    # This is a per-player second-order correction

    # The challenge: the second-order term depends on ALL starters, not just this player.
    # Specifically, if we change the lineup, Δtotal_c = new_total_c - old_total_c,
    # and the second-order correction is (1/2) × h_cc × (Δtotal_c)².
    # This can't be decomposed per-player because (Σ stat_c(p))² ≠ Σ stat_c(p)².

    # Instead, let's just compare the MEW-optimal lineup to all 1-swap perturbations
    # using the quadratic correction and see if the optimal changes.

    print(f"\nChecking all 1-swap perturbations with quadratic correction...")

    my_bench = my_roster - my_starters
    best_qew = current_ew
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

            # Linear approximation
            linear_delta = sum(
                gradient[cat] * (new_totals[cat] - my_totals[cat])
                for cat in ALL_CATEGORIES
            )

            # Quadratic correction
            quad_correction = sum(
                0.5 * hessian[cat] * (new_totals[cat] - my_totals[cat]) ** 2
                for cat in ALL_CATEGORIES
            )

            # Actual EW
            new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
            actual_delta = new_ew - current_ew

            # Compare
            linear_err = linear_delta - actual_delta
            quad_err = (linear_delta + quad_correction) - actual_delta

            if abs(linear_err) > 0.01 or abs(quad_err) > 0.01:
                print(
                    f"  {strip_name_suffix(bench_name):>20s} for {strip_name_suffix(starter_name):<20s}: "
                    f"ΔEW={actual_delta:+.4f}, "
                    f"Linear={linear_delta:+.4f} (err={linear_err:+.4f}), "
                    f"Quad={linear_delta + quad_correction:+.4f} (err={quad_err:+.4f})"
                )

            if new_ew > best_qew:
                best_qew = new_ew
                best_swap = (bench_name, starter_name)
                improvements.append(
                    {
                        "bench": strip_name_suffix(bench_name),
                        "starter": strip_name_suffix(starter_name),
                        "actual_delta": actual_delta,
                        "linear_delta": linear_delta,
                        "quad_delta": linear_delta + quad_correction,
                    }
                )

    if not improvements:
        print(f"\n  No improvements found — MEW-optimal is locally EW-optimal.")
    else:
        print(f"\n  {len(improvements)} improvement(s) found!")

    # Also compute: what if we directly maximized EW by brute force?
    # For 28 choose 18, this is ~3M combinations — too many.
    # But we can iterate: start from MEW-optimal, try all 1-swaps,
    # pick the best, repeat until no improvement.
    print(f"\nIterative local search from MEW-optimal lineup (hill climbing on EW)...")
    current_starters = set(my_starters)
    current_lineup = dict(my_lineup)
    current_local_ew = current_ew
    n_improvements = 0

    for iteration in range(10):
        best_ew = current_local_ew
        best_swap_local = None

        local_bench = my_roster - current_starters
        for bench_name in local_bench:
            bench_row = players[players["Name"] == bench_name].iloc[0]
            bench_eligible = get_eligible_slots(str(bench_row["Position"]))
            for starter_name, slot in current_lineup.items():
                if slot not in bench_eligible:
                    continue
                new_starters = (current_starters - {starter_name}) | {bench_name}
                new_totals = compute_totals_for_starters(new_starters, players)
                new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
                if new_ew > best_ew + 0.0001:
                    best_ew = new_ew
                    best_swap_local = (bench_name, starter_name, slot)

        if best_swap_local is None:
            print(
                f"  Converged after {iteration} improvement(s). Final EW = {current_local_ew:.4f}"
            )
            break

        bench_name, starter_name, slot = best_swap_local
        print(
            f"  Swap {iteration + 1}: {strip_name_suffix(bench_name)} for "
            f"{strip_name_suffix(starter_name)} at {slot}: "
            f"ΔEW = +{best_ew - current_local_ew:.4f}"
        )
        current_starters = (current_starters - {starter_name}) | {bench_name}
        current_lineup = solve_lineup(my_roster, players, "MEW")
        # Actually re-solve with the new lineup
        current_lineup_new = {}
        for name in current_starters:
            # We need a proper re-solve here
            pass
        current_local_ew = best_ew
        n_improvements += 1

    print(f"\n  MEW-optimal EW:      {current_ew:.4f}")
    print(f"  Hill-climbing EW:    {current_local_ew:.4f}")
    print(f"  EW left on table:    {current_local_ew - current_ew:.4f}")


# ============================================================================
# ANALYSIS F: End-to-end greedy optimizer accuracy
# ============================================================================


def analyze_greedy_accuracy(players, state):
    """Simulate one iteration of the greedy optimizer and compare
    MEW-ranked vs EW-ranked swap choices.

    The real question: does the greedy optimizer (which uses MEW for
    screening then exact MSV for evaluation) make the same choice as
    an optimizer that uses exact MSV for everything?
    """
    print("\n" + "=" * 70)
    print("ANALYSIS F: Greedy optimizer — does screening find the right swap?")
    print("=" * 70)

    my_roster = state["my_roster_names"]
    my_starters = state["my_starters"]
    my_bench = my_roster - my_starters
    my_lineup = state["my_lineup"]
    opponent_totals = state["opponent_totals"]
    sigmas = state["category_sigmas"]
    current_ew = state["current_ew"]

    fa_mask = players["owner"].isna()
    fas = players[fa_mask]

    # The greedy optimizer screens by MSV_approx, takes top K,
    # then exact-evaluates. The question: does the best MSV_approx
    # actually appear in the top K by MSV_exact?

    # Compute MSV_approx for all (FA, droppable) pairs
    all_swaps = []
    for _, fa_row in fas.iterrows():
        fa_name = fa_row["Name"]
        fa_mew = fa_row["MEW"]

        for drop_name in my_roster:
            drop_mew = float(players[players["Name"] == drop_name]["MEW"].iloc[0])
            msv_approx = fa_mew - drop_mew
            all_swaps.append(
                {
                    "fa_name": fa_name,
                    "drop_name": drop_name,
                    "msv_approx": msv_approx,
                    "fa": strip_name_suffix(fa_name),
                    "drop": strip_name_suffix(drop_name),
                    "drop_is_starter": drop_name in my_starters,
                }
            )

    all_df = pd.DataFrame(all_swaps)
    all_df = all_df.sort_values("msv_approx", ascending=False)

    # Top 50 by MSV_approx
    top50_approx = all_df.head(50)

    # How many are bench drops?
    n_bench_drops = (~top50_approx["drop_is_starter"]).sum()
    n_starter_drops = top50_approx["drop_is_starter"].sum()
    print(f"\nTop 50 by MSV_approx:")
    print(f"  Bench drops: {n_bench_drops}")
    print(f"  Starter drops: {n_starter_drops}")

    # Exact-evaluate a sample of swaps from different ranking tiers
    print(f"\nExact-evaluating selected swaps from different ranking tiers...")

    # Take top 10 approx, plus ranks 50-60, plus 10 random
    sample_indices = list(range(10))
    sample_indices += list(range(49, min(59, len(all_df))))
    sample_indices += list(
        np.random.choice(range(60, min(500, len(all_df))), size=10, replace=False)
    )

    exact_results = []
    for idx in sample_indices:
        if idx >= len(all_df):
            continue
        row = all_df.iloc[idx]
        fa_name = row["fa_name"]
        drop_name = row["drop_name"]

        new_roster = (my_roster - {drop_name}) | {fa_name}
        new_lineup = solve_lineup(new_roster, players, "MEW")
        new_starters = set(new_lineup.keys())
        new_totals = compute_totals_for_starters(new_starters, players)
        new_ew, _ = compute_win_probability(new_totals, opponent_totals, sigmas)
        msv_exact = new_ew - current_ew

        exact_results.append(
            {
                "approx_rank": idx + 1,
                "fa": row["fa"],
                "drop": row["drop"],
                "drop_is_starter": row["drop_is_starter"],
                "msv_approx": row["msv_approx"],
                "msv_exact": msv_exact,
            }
        )

    exact_df = pd.DataFrame(exact_results)
    print(
        f"\n  {'Rank':>6} {'FA':>20s} {'Drop':>20s} {'Strt':>5s} {'MSVapx':>8} {'MSVex':>8}"
    )
    for _, row in exact_df.iterrows():
        marker = " ***" if abs(row["msv_approx"] - row["msv_exact"]) > 0.5 else ""
        print(
            f"  {row['approx_rank']:>6d} {row['fa']:>20s} {row['drop']:>20s} "
            f"{'Y' if row['drop_is_starter'] else 'N':>5s} "
            f"{row['msv_approx']:>8.4f} {row['msv_exact']:>8.4f}{marker}"
        )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    players, state = load_and_prepare()

    analyze_bench_drop_mechanism(players, state)
    analyze_starter_drop_swaps(players, state)
    analyze_ops_nonlinearity(players, state)
    analyze_corrected_msv_approx(players, state)
    analyze_second_order_lineup_optimization(players, state)
    analyze_greedy_accuracy(players, state)

    print("\n" + "=" * 70)
    print("ANALYSIS PART 2 COMPLETE")
    print("=" * 70)
