"""
Swap evaluation: screening, exact MSV, gradient-based BV, greedy optimizer.

Implements Mathematical Framework Sections 4, 5, 6, 8.
A "swap" is the universal operation: drop N players, add N players.
"""

import pandas as pd
import pulp
from pulp import LpVariable, lpSum
from tqdm.auto import tqdm

from .config import (
    HITTING_SLOTS,
    MAX_HITTERS,
    MAX_PITCHERS,
    MIN_HITTERS,
    MIN_PITCHERS,
    MY_TEAM_NAME,
    N_STARTER_SLOTS,
    PITCHING_SLOTS,
    ROSTER_SIZE,
)
from .league_state import compute_league_state
from .lineup_solver import (
    assign_optimal_slots,
    compute_totals_for_starters,
    get_solver,
    maybe_blend,
    solve_lineup,
)
from .player_scoring import add_mew
from .players import get_startable_slots
from .win_model import (
    compute_ew_gradient,
    compute_win_probability,
    estimate_projection_uncertainty,
)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_SCREEN_TOP_K: int = 50
DEFAULT_VALUE_THRESHOLD: float = 0.1

POSITION_ABSENCE_RATES: dict[str, float] = {
    "C": 0.25,
    "1B": 0.18,
    "2B": 0.20,
    "SS": 0.22,
    "3B": 0.20,
    "OF": 0.20,
    "UTIL": 0.15,
    "SP": 0.35,
    "RP": 0.25,
}

_ALL_SLOTS: dict[str, int] = {**HITTING_SLOTS, **PITCHING_SLOTS}

# ============================================================================
# Sandbox: validate_transaction — Positional feasibility check
# ============================================================================


def validate_transaction(
    drop_names: set[str],
    add_names: set[str],
    my_roster_names: set[str],
    players: pd.DataFrame,
) -> dict:
    """Validate that a proposed transaction preserves roster feasibility.

    Checks:
    1. All drop names are on my roster.
    2. No add names are already on my roster.
    3. Roster size stays within bounds (28 - min/max hitter/pitcher balance).
    4. Every lineup slot can still be filled after the transaction.

    Returns:
        {
            'valid': bool,
            'errors': list[str],   — blocking problems
            'warnings': list[str], — non-blocking notes
            'new_roster': set[str] | None,
        }
    """
    errors: list[str] = []
    warnings: list[str] = []

    # 1. Drop names must be on roster
    missing_drops = drop_names - my_roster_names
    if missing_drops:
        from .players import strip_name_suffix

        errors.append(
            f"Cannot drop {len(missing_drops)} player(s) not on your roster: "
            + ", ".join(strip_name_suffix(n) for n in sorted(missing_drops))
        )

    # 2. Add names must not already be on roster
    already_rostered = add_names & my_roster_names
    if already_rostered:
        from .players import strip_name_suffix

        errors.append(
            f"{len(already_rostered)} player(s) already on your roster: "
            + ", ".join(strip_name_suffix(n) for n in sorted(already_rostered))
        )

    # 3. Roster size — a transaction changes the count only when the number of
    # players dropped differs from the number added. (my_roster_names is the
    # working roster, which includes IR players held beyond the active 28, so
    # comparing against the ROSTER_SIZE constant would false-alarm here.)
    cur_size = len(my_roster_names)
    new_size = cur_size - len(drop_names) + len(add_names)
    if len(drop_names) != len(add_names):
        warnings.append(
            f"Roster size changes from {cur_size} to {new_size} "
            f"(dropping {len(drop_names)}, adding {len(add_names)}). "
            f"Transaction will be padded or trimmed for evaluation."
        )

    if errors:
        return {
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "new_roster": None,
        }

    new_roster = (my_roster_names - drop_names) | add_names
    new_roster_df = players[players["Name"].isin(new_roster)]

    # 4. Check each slot can be filled
    n_hitters = len(new_roster_df[new_roster_df["player_type"] == "hitter"])
    n_pitchers = len(new_roster_df[new_roster_df["player_type"] == "pitcher"])

    total_hitting_slots = sum(HITTING_SLOTS.values())
    total_pitching_slots = sum(PITCHING_SLOTS.values())

    if n_hitters < total_hitting_slots:
        errors.append(
            f"Only {n_hitters} hitters on new roster but need "
            f"{total_hitting_slots} to fill hitting slots. "
            f"You're dropping too many hitters."
        )
    if n_pitchers < total_pitching_slots:
        errors.append(
            f"Only {n_pitchers} pitchers on new roster but need "
            f"{total_pitching_slots} to fill pitching slots. "
            f"You're dropping too many pitchers."
        )

    # League roster-composition bounds (Fantrax enforces these at execution).
    # IR players sit outside the active 28 and don't count toward the bounds.
    if "roster_status" in new_roster_df.columns:
        active_df = new_roster_df[new_roster_df["roster_status"] != "IR"]
    else:
        active_df = new_roster_df
    n_active_hitters = int((active_df["player_type"] == "hitter").sum())
    n_active_pitchers = int((active_df["player_type"] == "pitcher").sum())
    if not (MIN_HITTERS <= n_active_hitters <= MAX_HITTERS):
        errors.append(
            f"New roster has {n_active_hitters} active hitters; league requires "
            f"{MIN_HITTERS}-{MAX_HITTERS}."
        )
    if not (MIN_PITCHERS <= n_active_pitchers <= MAX_PITCHERS):
        errors.append(
            f"New roster has {n_active_pitchers} active pitchers; league requires "
            f"{MIN_PITCHERS}-{MAX_PITCHERS}."
        )

    # Per-slot feasibility — injury-aware: IL players cannot fill a starting
    # slot, so a roster whose only healthy catcher is dropped must fail here
    # even if an IL catcher remains on paper.
    has_inj = "injury_status" in new_roster_df.columns
    for slot, count in _ALL_SLOTS.items():
        startable_count = 0
        for _, row in new_roster_df.iterrows():
            slots = get_startable_slots(
                str(row["Position"]),
                row["injury_status"] if has_inj else None,
            )
            if slot in slots:
                startable_count += 1
        if startable_count < count:
            errors.append(
                f"Cannot fill {slot} slot: need {count} startable player(s) "
                f"but only {startable_count} healthy/eligible on new roster."
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "new_roster": new_roster if len(errors) == 0 else None,
    }


# ============================================================================
# Starter comparison — "Should I start A or B?"
# ============================================================================


def compare_starters(
    candidate_names: list[str],
    my_roster_names: set[str],
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    my_banked_totals: dict[str, float] | None = None,
) -> list[dict]:
    """Compare the impact of forcing each candidate into the starting lineup.

    For each candidate, solves the lineup MILP with that player constrained
    to start, then computes team totals and EW. The rest of the lineup is
    optimized freely around the forced starter.

    Args:
        candidate_names: Internal names (with -H/-P suffix) to compare.
            All must be on my_roster_names.
        my_roster_names: Current 28-player roster.
        players: Players DataFrame with MEW column.
        opponent_totals: {opp_id: {cat: total}}.
        category_sigmas: σ_c per category.

    Returns:
        List of dicts (one per candidate, same order as candidate_names):
        [
            {
                'name': str,
                'lineup': dict,  — full lineup assignment
                'totals': dict,  — team totals
                'ew': float,
                'cat_ew': dict,  — per-category EW
            },
            ...
        ]
    """
    from scipy import stats as sp_stats

    from .config import ALL_CATEGORIES, NEGATIVE_CATEGORIES

    assert len(candidate_names) >= 2, (
        f"compare_starters: need at least 2 candidates, got {len(candidate_names)}"
    )
    for name in candidate_names:
        assert name in my_roster_names, f"compare_starters: '{name}' not on roster"
    assert "MEW" in players.columns, "compare_starters: players must have MEW column"

    results = []
    for name in candidate_names:
        lineup = solve_lineup(my_roster_names, players, "MEW", force_start={name})
        starters = set(lineup.keys())
        ros = compute_totals_for_starters(starters, players)
        totals = maybe_blend(my_banked_totals, ros)
        ew, _ = compute_win_probability(totals, opponent_totals, category_sigmas)

        cat_ew: dict[str, float] = {}
        for cat in ALL_CATEGORIES:
            total = 0.0
            for opp in opponent_totals.values():
                denom = category_sigmas.get(cat, 1.0) * (2.0**0.5)
                if denom < 1e-9:
                    denom = 1e-9
                if cat in NEGATIVE_CATEGORIES:
                    z = (opp[cat] - totals[cat]) / denom
                else:
                    z = (totals[cat] - opp[cat]) / denom
                total += float(sp_stats.norm.cdf(z))
            cat_ew[cat] = total

        results.append(
            {
                "name": name,
                "lineup": lineup,
                "totals": totals,
                "ew": ew,
                "cat_ew": cat_ew,
            }
        )

    return results


# ============================================================================
# 9a. compute_exact_msv — Ground-truth swap value
# ============================================================================


def compute_exact_msv(
    drop_names: set[str],
    add_names: set[str],
    my_roster_names: set[str],
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    my_banked_totals: dict[str, float] | None = None,
    baseline_lineup: dict[str, str] | None = None,
) -> dict:
    """Compute exact Marginal Swap Value by re-solving the lineup.

    MSV = EW(new_roster) − EW(current_roster)

    This is the universal evaluator for any transaction: 1-for-1 FA swap,
    N-for-N trade, or a batch of moves. Same math regardless.

    Hitter and pitcher lineups are INDEPENDENT subproblems (disjoint players
    and slots). When ``baseline_lineup`` is given and the swap touches only one
    player type, the untouched half is reused verbatim and only the affected
    half is re-solved. This is both exact and essential: re-solving the whole
    lineup lets MILP ties in the untouched half flip arbitrarily (e.g. a
    hitter swap silently swapping which two relievers start), injecting
    phantom category deltas the move never caused. Holding the untouched half
    fixed guarantees those categories are unchanged by construction.

    Args:
        drop_names: Players to remove from my roster.
        add_names: Players to add to my roster.
        my_roster_names: Current roster names.
        players: Players DataFrame with MEW column.
        opponent_totals: {opp_id: {cat: total}}. SEASON totals (banked + ros)
            when banked modeling is active; ros-only otherwise.
        category_sigmas: σ_c per category.
        current_ew: EW of current roster.
        my_banked_totals: My banked YTD totals (compute_totals_for_starters
            shape), or None. When given, the post-swap ros totals are blended
            with banked before computing EW, and 'new_totals' is the season
            total so downstream gradient/MEW/BV stay season-consistent.
        baseline_lineup: Current {name: slot} lineup. When given, the untouched
            player-type half is held fixed for single-type swaps (see above).

    Returns:
        {
            'msv': float,
            'new_ew': float,
            'new_totals': dict,    # SEASON totals when banked given, else ros
            'new_lineup': dict,
        }
    """
    assert len(drop_names) == len(add_names), (
        f"compute_exact_msv: |drop| = {len(drop_names)} != "
        f"|add| = {len(add_names)}. Roster size must be preserved."
    )
    assert drop_names <= my_roster_names, (
        f"compute_exact_msv: dropping players not on roster: "
        f"{sorted(drop_names - my_roster_names)}"
    )

    new_roster = (my_roster_names - drop_names) | add_names
    assert len(new_roster) == len(my_roster_names), (
        f"compute_exact_msv: new roster size {len(new_roster)} != "
        f"original {len(my_roster_names)}. drop={sorted(drop_names)}, "
        f"add={sorted(add_names)}"
    )

    new_lineup = _solve_lineup_holding_half(
        new_roster, players, drop_names | add_names, baseline_lineup
    )
    new_ros = compute_totals_for_starters(set(new_lineup.keys()), players)
    new_totals = maybe_blend(my_banked_totals, new_ros)
    new_ew, _ = compute_win_probability(new_totals, opponent_totals, category_sigmas)
    msv = new_ew - current_ew

    return {
        "msv": msv,
        "new_ew": new_ew,
        "new_totals": new_totals,
        "new_lineup": new_lineup,
    }


def _solve_lineup_holding_half(
    new_roster: set[str],
    players: pd.DataFrame,
    changed_names: set[str],
    baseline_lineup: dict[str, str] | None,
) -> dict[str, str]:
    """Solve the post-swap MEW lineup, holding the untouched player type fixed.

    If the swap touches only hitters (or only pitchers) and a baseline lineup
    is available, the other type's starters are carried over unchanged and only
    the affected half is re-optimized. Otherwise the full lineup is re-solved.
    """
    if baseline_lineup is None:
        return solve_lineup(new_roster, players, "MEW")

    type_lookup = players.set_index("Name")["player_type"].to_dict()
    changed_types = {type_lookup.get(n) for n in changed_names}

    if changed_types == {"hitter"}:
        affected_slots, kept_slots = HITTING_SLOTS, PITCHING_SLOTS
    elif changed_types == {"pitcher"}:
        affected_slots, kept_slots = PITCHING_SLOTS, HITTING_SLOTS
    else:
        # Mixed-type (or unknown) swap: both halves can change.
        return solve_lineup(new_roster, players, "MEW")

    kept = {
        name: slot
        for name, slot in baseline_lineup.items()
        if slot in kept_slots and name in new_roster
    }
    affected = solve_lineup(new_roster, players, "MEW", slots=affected_slots)
    return {**kept, **affected}


# ============================================================================
# 9d. add_bench_value — Gradient-based bench insurance
# ============================================================================


def add_bench_value(
    players: pd.DataFrame,
    my_lineup: dict[str, str],
    my_roster_names: set[str],
) -> pd.DataFrame:
    """Add 'BV' column: analytical bench insurance value.

    BV is computed via the gradient-based formula from MATHEMATICAL_FRAMEWORK §6.

    Formula:
        BV(b) = Σ_{k : best_bench(k) = b} P_absent(k) × max(0, MEW(b) − MEW(best_FA(k)))

    where:
        k indexes starter slots
        best_bench(k) = highest-MEW bench player eligible for slot k
        best_FA(k) = highest-MEW free agent eligible for slot k
        P_absent(k) = POSITION_ABSENCE_RATES[slot_type(k)]

    A bench player who is the best option for multiple slots accumulates
    BV from all of them.

    Args:
        my_lineup: state['my_lineup'] from compute_league_state — a dict
            mapping starter name → assigned slot.
        my_roster_names: Canonical 28-player optimizer roster.
            Bench = my_roster_names − starters.

    Requires: Name, Position, MEW.
    Adds: BV.
    """
    players = players.copy()
    assert "MEW" in players.columns, (
        "add_bench_value: players must have MEW column. Call add_mew() first."
    )

    my_starters = set(my_lineup.keys())
    my_bench_mask = players["Name"].isin(my_roster_names - my_starters)
    # Exclude roster members from the FA pool: when evaluating a hypothetical
    # swap, the owner column still shows the just-added player as a free
    # agent, but they are on my_roster_names and no longer claimable.
    fa_mask = players["owner"].isna() & ~players["Name"].isin(my_roster_names)

    bench_df = players[my_bench_mask]
    fa_df = players[fa_mask]

    # Precompute STARTABLE slots (injury-aware): an IL bench player provides
    # no insurance, and an IL free agent is not a startable replacement.
    has_inj = "injury_status" in players.columns

    bench_eligible: dict[str, set[str]] = {}
    for _, row in bench_df.iterrows():
        bench_eligible[row["Name"]] = get_startable_slots(
            str(row["Position"]), row["injury_status"] if has_inj else None
        )

    fa_eligible: dict[str, set[str]] = {}
    for _, row in fa_df.iterrows():
        fa_eligible[row["Name"]] = get_startable_slots(
            str(row["Position"]), row["injury_status"] if has_inj else None
        )

    bench_mew = bench_df.set_index("Name")["MEW"].to_dict()
    fa_mew = fa_df.set_index("Name")["MEW"].to_dict()

    # For each starter slot k, find best_bench and best_FA
    bv_accumulator: dict[str, float] = {name: 0.0 for name in bench_mew}

    for starter_name, slot in my_lineup.items():
        p_absent = POSITION_ABSENCE_RATES.get(slot, 0.20)

        # Best bench player eligible for this slot
        best_bench_name = None
        best_bench_mew = -float("inf")
        for name, elig in bench_eligible.items():
            if slot in elig and bench_mew[name] > best_bench_mew:
                best_bench_mew = bench_mew[name]
                best_bench_name = name

        if best_bench_name is None:
            continue

        # Best FA eligible for this slot (MEW=0 if no FA eligible at all)
        eligible_fa_mews = [
            fa_mew[name] for name, elig in fa_eligible.items() if slot in elig
        ]
        best_fa_mew_val = max(eligible_fa_mews) if eligible_fa_mews else 0.0

        premium = max(0.0, best_bench_mew - best_fa_mew_val)
        bv_accumulator[best_bench_name] += p_absent * premium

    # Set BV column
    players["BV"] = 0.0
    for name, bv in bv_accumulator.items():
        players.loc[players["Name"] == name, "BV"] = bv

    return players


# ============================================================================
# 9b. screen_swaps — Unified screening for all FA swaps
# ============================================================================


def compute_critical_slots(
    my_roster_names: set[str],
    players: pd.DataFrame,
) -> dict[str, set[str]]:
    """For each roster player, the slots that would become unfillable without them.

    Coverage is INJURY-AWARE: an IL player cannot start, so they neither
    provide coverage nor need protection. A slot s with count k is "critical"
    for player p when p is one of <= k startable covers — dropping p would
    leave the slot short.

    This is the basis for conditional drop protection: dropping p is only
    sound if the incoming player covers all of p's critical slots (e.g. your
    only startable catcher may be swapped for a better catcher, but not for
    an outfielder).

    Returns:
        {player_name: set of critical slots}, only for players with at least
        one critical slot.
    """
    roster_df = players[players["Name"].isin(my_roster_names)]
    has_inj = "injury_status" in roster_df.columns

    startable: dict[str, set[str]] = {}
    for _, row in roster_df.iterrows():
        startable[row["Name"]] = get_startable_slots(
            str(row["Position"]),
            row["injury_status"] if has_inj else None,
        )

    critical: dict[str, set[str]] = {}
    for slot, count in _ALL_SLOTS.items():
        covers = [name for name, slots in startable.items() if slot in slots]
        if len(covers) <= count:
            for name in covers:
                critical.setdefault(name, set()).add(slot)

    return critical


def find_protected_players(
    my_roster_names: set[str],
    players: pd.DataFrame,
) -> set[str]:
    """Roster players that cannot be dropped for a non-covering replacement.

    Injury-aware: a player is protected when they are one of the last
    startable covers for some slot (see compute_critical_slots). They may
    still be dropped for an incoming player who covers those slots — that
    conditional check lives in screen_swaps.
    """
    return set(compute_critical_slots(my_roster_names, players))


def screen_swaps(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_lineup: dict[str, str],
    top_k: int = DEFAULT_SCREEN_TOP_K,
) -> pd.DataFrame:
    """Screen all possible 1-for-1 FA swaps, ranked by approximate Value.

    For each (FA, roster_player) pair, compute a **lineup-aware** MSV_approx
    plus ΔBV_approx. Approximate Value = MSV_approx + ΔBV_approx.

    Lineup-aware MSV_approx (MATHEMATICAL_FRAMEWORK §4):
        The EW change from a swap depends on who enters and leaves the
        *starting lineup*, not who enters and leaves the *roster*.

        - Bench drop, FA starts (displacing starter S):
            MSV_approx = MEW(FA) − MEW(S)
        - Bench drop, FA doesn't start:
            MSV_approx = 0  (value comes from ΔBV only)
        - Starter drop, FA eligible for vacated slot:
            MSV_approx = MEW(FA) − MEW(dropped_starter)
        - Starter drop, FA takes different slot (cascade):
            Approximate conservatively; exact evaluation handles cascades.

    Args:
        my_lineup: state['my_lineup'] from compute_league_state.
        top_k: Number of top candidates to return.

    Requires columns: Name, owner, optimal_slot, MEW, Position, player_type.

    Returns:
        DataFrame with columns: fa_name, drop_name, msv_approx,
        delta_bv_approx, value_approx. Sorted by value_approx descending.
    """
    assert "MEW" in players.columns, (
        "screen_swaps: players must have MEW. Call add_mew() first."
    )
    assert "BV" in players.columns, (
        "screen_swaps: players must have BV. Call add_bench_value() first."
    )

    # 1. Slot-coverage criticality (injury-aware). A player who is one of the
    # last startable covers for a slot may only be dropped for an FA who also
    # covers that slot — "replace your only catcher with a better catcher".
    critical = compute_critical_slots(my_roster_names, players)
    droppable = set(my_roster_names)

    if critical:
        print(
            f"Conditionally protected (last startable cover): {len(critical)} players"
        )

    # 2. Precompute lookups
    my_starters = set(my_lineup.keys())
    my_bench = my_roster_names - my_starters
    fa_mask = players["owner"].isna()
    fa_names = set(players[fa_mask]["Name"])

    mew_lookup = players.set_index("Name")["MEW"].to_dict()
    bv_lookup = players.set_index("Name")["BV"].to_dict()
    pos_lookup = players.set_index("Name")["Position"].to_dict()
    type_lookup = players.set_index("Name")["player_type"].to_dict()
    inj_lookup = (
        players.set_index("Name")["injury_status"].to_dict()
        if "injury_status" in players.columns
        else {}
    )
    status_lookup = (
        players.set_index("Name")["roster_status"].to_dict()
        if "roster_status" in players.columns
        else {}
    )

    def _startable(name: str) -> set[str]:
        """Injury-aware startable slots for any player by name."""
        return get_startable_slots(str(pos_lookup.get(name, "")), inj_lookup.get(name))

    # Active-roster composition (IR players sit outside the bounds).
    n_act_h = sum(
        1
        for n in my_roster_names
        if status_lookup.get(n) != "IR" and type_lookup.get(n) == "hitter"
    )
    n_act_p = sum(
        1
        for n in my_roster_names
        if status_lookup.get(n) != "IR" and type_lookup.get(n) == "pitcher"
    )

    def _composition_ok(fa_name: str, drop_name: str) -> bool:
        """Would the swap keep the active roster inside league h/p bounds?"""
        h, p = n_act_h, n_act_p
        if status_lookup.get(drop_name) != "IR":
            if type_lookup.get(drop_name) == "hitter":
                h -= 1
            else:
                p -= 1
        if type_lookup.get(fa_name) == "hitter":
            h += 1
        else:
            p += 1
        return MIN_HITTERS <= h <= MAX_HITTERS and MIN_PITCHERS <= p <= MAX_PITCHERS

    current_total_bv = sum(bv_lookup.get(n, 0.0) for n in my_bench)

    # 2a. Precompute per-slot starter info for lineup-aware MSV
    # For each slot, the weakest starter (lowest MEW) — the one an FA would displace
    slot_to_starters: dict[str, list[tuple[str, float]]] = {}
    for name, slot in my_lineup.items():
        slot_to_starters.setdefault(slot, []).append((name, mew_lookup[name]))
    for slot in slot_to_starters:
        slot_to_starters[slot].sort(key=lambda x: x[1])

    # 2b. Precompute per-slot best bench/FA MEW for BV approximation
    bench_slot_mew: dict[str, list[tuple[str, float]]] = {}
    fa_slot_mew: dict[str, list[tuple[str, float]]] = {}

    for slot in set(my_lineup.values()):
        bench_eligible = []
        for name in my_bench:
            if slot in _startable(name):
                bench_eligible.append((name, mew_lookup.get(name, 0.0)))
        bench_eligible.sort(key=lambda x: x[1], reverse=True)
        bench_slot_mew[slot] = bench_eligible

        fa_eligible = []
        for name in fa_names:
            if slot in _startable(name):
                fa_eligible.append((name, mew_lookup.get(name, 0.0)))
        fa_eligible.sort(key=lambda x: x[1], reverse=True)
        fa_slot_mew[slot] = fa_eligible

    def _fa_displacement(fa_mew: float, fa_eligible_slots: set[str]) -> float:
        """How much EW does the FA gain by displacing the weakest eligible starter?

        Returns max(0, FA_MEW − weakest_starter_MEW) across eligible slots.
        Zero if the FA wouldn't start.
        """
        best = 0.0
        for slot in fa_eligible_slots:
            starters = slot_to_starters.get(slot, [])
            if starters:
                weakest_mew = starters[0][1]
                disp = fa_mew - weakest_mew
                if disp > best:
                    best = disp
        return best

    def _lineup_aware_msv(
        fa_name: str,
        fa_mew: float,
        fa_eligible_slots: set[str],
        drop_name: str,
        drop_mew: float,
    ) -> float:
        """Compute lineup-aware MSV_approx for a 1-for-1 swap.

        The key insight: EW changes come from who enters/leaves the starting
        lineup, not who enters/leaves the roster. A bench player's MEW does
        not represent their actual EW contribution (which is 0 from the
        starting lineup; their value is captured by BV separately).
        """
        drop_is_starter = drop_name in my_lineup

        if not drop_is_starter:
            # Bench drop: lineup only changes if FA displaces a current starter
            return _fa_displacement(fa_mew, fa_eligible_slots)

        # Starter drop
        drop_slot = my_lineup[drop_name]

        if drop_slot in fa_eligible_slots:
            # FA can directly replace the dropped starter
            return fa_mew - drop_mew

        # Cascade: FA can't fill the vacated slot. A bench player fills it,
        # and the FA may displace a different starter. Approximate as:
        # (best bench eligible for vacated slot − dropped starter) + FA displacement
        best_fill_mew = -float("inf")
        for bn, bm in bench_slot_mew.get(drop_slot, []):
            best_fill_mew = bm
            break
        if best_fill_mew == -float("inf"):
            return -float("inf")

        fill_cost = best_fill_mew - drop_mew
        fa_disp = _fa_displacement(fa_mew, fa_eligible_slots)
        return fill_cost + fa_disp

    def _approx_delta_bv(drop_name: str, fa_name: str) -> float:
        """Approximate ΔBV for a 1-for-1 swap without lineup re-solve.

        Affected slots are those where the incoming FA is eligible (bench
        insurance may improve) OR where the dropped player is eligible
        (insurance may be lost). Slots untouched by both players contribute
        identically before and after, so they cancel in the delta.
        """
        drop_is_bench = drop_name in my_bench
        if not drop_is_bench:
            return 0.0

        fa_mew_val = mew_lookup.get(fa_name, 0.0)
        fa_eligible_slots = _startable(fa_name)
        drop_eligible_slots = _startable(drop_name)
        affected_slots = fa_eligible_slots | drop_eligible_slots

        delta = 0.0
        for starter_name, slot in my_lineup.items():
            if slot not in affected_slots:
                continue
            p_absent = POSITION_ABSENCE_RATES.get(slot, 0.20)
            current_bench_list = bench_slot_mew.get(slot, [])
            current_fa_list = fa_slot_mew.get(slot, [])

            # Before the swap
            best_b_old = current_bench_list[0][1] if current_bench_list else None
            best_f_old = current_fa_list[0][1] if current_fa_list else 0.0
            if best_b_old is not None:
                delta -= p_absent * max(0.0, best_b_old - best_f_old)

            # After the swap: drop leaves the bench; FA joins it (if eligible)
            best_b_new = None
            for bn, bm in current_bench_list:
                if bn == drop_name:
                    continue
                best_b_new = bm
                break
            if slot in fa_eligible_slots:
                if best_b_new is None or fa_mew_val > best_b_new:
                    best_b_new = fa_mew_val

            best_f_new = 0.0
            for fn, fm in current_fa_list:
                if fn == fa_name:
                    continue
                best_f_new = fm
                break

            if best_b_new is not None:
                delta += p_absent * max(0.0, best_b_new - best_f_new)

        return delta

    # 3. Screen: for each FA, find best droppable roster player
    rows = []
    fa_df = players[fa_mask].copy()

    for _, fa_row in fa_df.iterrows():
        fa_name = fa_row["Name"]
        fa_mew = fa_row["MEW"]
        fa_eligible_slots = _startable(fa_name)

        best_value = -float("inf")
        best_drop = None
        best_msv = 0.0
        best_dbv = 0.0

        for drop_name in sorted(droppable):
            # Conditional protection: the FA must cover every slot for which
            # the dropped player is the last startable cover.
            critical_slots = critical.get(drop_name)
            if critical_slots and not critical_slots <= fa_eligible_slots:
                continue
            # League roster-composition bounds (12-18 hitters, 10-14 pitchers).
            if not _composition_ok(fa_name, drop_name):
                continue
            # IL stash protection: an IL player's ROS projection is their
            # post-return value (the feed already discounts for missed time),
            # but the lineup model scores them zero because they can't start
            # TODAY. Never trade that future value for a lesser player —
            # dropping an IL player requires an FA with higher MEW.
            if inj_lookup.get(drop_name) == "IL" and fa_mew <= mew_lookup.get(
                drop_name, 0.0
            ):
                continue

            drop_mew = mew_lookup.get(drop_name, 0.0)
            msv_approx = _lineup_aware_msv(
                fa_name,
                fa_mew,
                fa_eligible_slots,
                drop_name,
                drop_mew,
            )
            delta_bv = _approx_delta_bv(drop_name, fa_name)
            value_approx = msv_approx + delta_bv

            if value_approx > best_value:
                best_value = value_approx
                best_drop = drop_name
                best_msv = msv_approx
                best_dbv = delta_bv

        if best_drop is not None:
            rows.append(
                {
                    "fa_name": fa_name,
                    "drop_name": best_drop,
                    "msv_approx": best_msv,
                    "delta_bv_approx": best_dbv,
                    "value_approx": best_value,
                }
            )

    result = pd.DataFrame(rows)
    if len(result) == 0:
        print("screen_swaps: no candidates found")
        return result

    result = result.sort_values("value_approx", ascending=False).head(top_k)
    result = result.reset_index(drop=True)

    n_positive = (result["value_approx"] > 0).sum()
    print(
        f"Screened {len(fa_df)} FAs × {len(droppable)} droppable: "
        f"{n_positive} positive-value candidates in top {len(result)}"
    )
    return result


# ============================================================================
# 9c. evaluate_top_k — Exact Value for top candidates
# ============================================================================


def evaluate_top_k(
    candidates: pd.DataFrame,
    my_roster_names: set[str],
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    include_bv: bool = True,
    my_banked_totals: dict[str, float] | None = None,
    my_lineup: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute exact Value = ΔEW + ΔBV for each screened swap candidate.

    For each candidate (fa_name, drop_name):
        1. result = compute_exact_msv({drop_name}, {fa_name}, ...)
        2. If include_bv:
           a. new_gradient from new_totals
           b. new_MEW for all players via add_mew
           c. new_total_bv via add_bench_value
           d. delta_bv = new_total_bv − baseline_bv
        3. value = msv + delta_bv

    ΔBV scale consistency: when ``my_lineup`` is given, the baseline bench
    value is recomputed for the CURRENT roster under the candidate's
    post-swap gradient, so delta_bv compares both rosters on one MEW scale.
    Comparing new-gradient BV against the pre-computed old-gradient
    ``current_total_bv`` lets global gradient rescaling masquerade as value —
    the greedy loop then farms phantom ΔBV by churning players in and out.

    Returns: candidates with added columns msv_exact, delta_bv, value, new_ew.
    Sorted by value descending.
    """
    if len(candidates) == 0:
        return candidates

    results = []
    for _, row in tqdm(
        candidates.iterrows(),
        total=len(candidates),
        desc="Exact evaluation",
    ):
        fa_name = row["fa_name"]
        drop_name = row["drop_name"]

        msv_result = compute_exact_msv(
            {drop_name},
            {fa_name},
            my_roster_names,
            players,
            opponent_totals,
            category_sigmas,
            current_ew,
            my_banked_totals=my_banked_totals,
            baseline_lineup=my_lineup,
        )

        delta_bv = 0.0
        if include_bv:
            new_gradient = compute_ew_gradient(
                msv_result["new_totals"], opponent_totals, category_sigmas
            )
            work = add_mew(players, msv_result["new_totals"], new_gradient)
            new_roster = (my_roster_names - {drop_name}) | {fa_name}
            scored = add_bench_value(work, msv_result["new_lineup"], new_roster)
            new_bench = new_roster - set(msv_result["new_lineup"].keys())
            new_total_bv = float(scored[scored["Name"].isin(new_bench)]["BV"].sum())

            if my_lineup is not None:
                # Same-scale baseline: current roster's BV under the SAME
                # post-swap gradient (see docstring).
                base = add_bench_value(work, my_lineup, my_roster_names)
                old_bench = my_roster_names - set(my_lineup.keys())
                baseline_bv = float(base[base["Name"].isin(old_bench)]["BV"].sum())
            else:
                baseline_bv = current_total_bv
            delta_bv = new_total_bv - baseline_bv

        value = msv_result["msv"] + delta_bv

        results.append(
            {
                "fa_name": fa_name,
                "drop_name": drop_name,
                "msv_exact": msv_result["msv"],
                "delta_bv": delta_bv,
                "value": value,
                "new_ew": msv_result["new_ew"],
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("value", ascending=False).reset_index(drop=True)
    return result_df


# ============================================================================
# 9e. compute_ew_ceiling — Diagnostic for gap-to-optimal
# ============================================================================


def compute_ew_ceiling(
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    my_roster_names: set[str],
    current_ew: float,
    my_team_name: str | None = None,
    my_banked_totals: dict[str, float] | None = None,
) -> dict:
    """Compute best achievable EW from full candidate pool (diagnostic).

    Solves a larger MILP: pick ROSTER_SIZE players from all available
    (my roster + FA pool) and assign N_STARTER_SLOTS to starter slots,
    maximizing Σ MEW(starters).

    After solving, compute exact EW from the ceiling roster.

    Args:
        players: Players DataFrame with MEW column.
        opponent_totals: {opp_id: {cat: total}}.
        category_sigmas: σ_c per category.
        my_roster_names: Canonical 28-player roster from compute_league_state.
        current_ew: EW of current roster (from compute_league_state).
        my_team_name: Team name for pool filtering. Defaults to MY_TEAM_NAME.

    Returns:
        {
            'ceiling_ew': float,
            'ceiling_roster': set[str],
            'ceiling_lineup': dict,
            'gap': float,
        }
    """
    if my_team_name is None:
        my_team_name = MY_TEAM_NAME

    assert "MEW" in players.columns, "compute_ew_ceiling: players must have MEW column"

    # Candidate pool: my roster + FA pool
    pool_mask = (players["owner"] == my_team_name) | players["owner"].isna()
    pool = players[pool_mask].reset_index(drop=True)
    n = len(pool)

    print(
        f"Ceiling MILP: {n} candidates, {ROSTER_SIZE} roster spots, "
        f"{N_STARTER_SLOTS} starter slots"
    )

    has_injury_col = "injury_status" in pool.columns
    eligibility = {
        i: get_startable_slots(
            str(pool.iloc[i]["Position"]),
            pool.iloc[i]["injury_status"] if has_injury_col else None,
        )
        for i in range(n)
    }

    prob = pulp.LpProblem("Ceiling", pulp.LpMaximize)

    x = {i: LpVariable(f"x_{i}", cat="Binary") for i in range(n)}
    a: dict[tuple[int, str], LpVariable] = {}
    for i in range(n):
        for s in eligibility[i]:
            a[i, s] = LpVariable(f"a_{i}_{s}", cat="Binary")

    # Objective: maximize Σ MEW(starters)
    prob += lpSum(
        pool.iloc[i]["MEW"] * lpSum(a[i, s] for s in eligibility[i] if (i, s) in a)
        for i in range(n)
    )

    # Roster size constraint
    prob += lpSum(x[i] for i in range(n)) <= ROSTER_SIZE, "roster_size"

    # Assignment requires roster membership
    for i in range(n):
        for s in eligibility[i]:
            if (i, s) in a:
                prob += a[i, s] <= x[i], f"roster_{i}_{s}"

    # Each player at most one slot
    for i in range(n):
        slots_for_player = [s for s in eligibility[i] if (i, s) in a]
        if slots_for_player:
            prob += lpSum(a[i, s] for s in slots_for_player) <= 1, f"one_slot_{i}"

    # Slot constraints: fill all slots (IS §9e says "assign 18 to starter slots")
    for slot, count in _ALL_SLOTS.items():
        eligible_for_slot = [i for i in range(n) if slot in eligibility[i]]
        slots_to_fill = min(len(eligible_for_slot), count)
        if slots_to_fill > 0:
            prob += (
                lpSum(a[i, slot] for i in eligible_for_slot if (i, slot) in a)
                == slots_to_fill,
                f"fill_{slot}",
            )

    prob.solve(get_solver())

    assert prob.status == pulp.LpStatusOptimal, (
        f"Ceiling MILP failed: {pulp.LpStatus[prob.status]}"
    )

    ceiling_roster: set[str] = set()
    ceiling_lineup: dict[str, str] = {}
    for i in range(n):
        if pulp.value(x[i]) > 0.5:
            ceiling_roster.add(pool.iloc[i]["Name"])
        for s in eligibility[i]:
            if (i, s) in a and pulp.value(a[i, s]) > 0.5:
                ceiling_lineup[pool.iloc[i]["Name"]] = s

    ceiling_ros = compute_totals_for_starters(set(ceiling_lineup.keys()), players)
    ceiling_totals = maybe_blend(my_banked_totals, ceiling_ros)
    ceiling_ew, _ = compute_win_probability(
        ceiling_totals, opponent_totals, category_sigmas
    )

    gap = ceiling_ew - current_ew

    print(f"Ceiling EW: {ceiling_ew:.2f}, current EW: {current_ew:.2f}, gap: {gap:.2f}")

    return {
        "ceiling_ew": ceiling_ew,
        "ceiling_roster": ceiling_roster,
        "ceiling_lineup": ceiling_lineup,
        "gap": gap,
    }


# ============================================================================
# 9f. run_greedy_optimization — The optimizer
# ============================================================================


def run_greedy_optimization(
    players: pd.DataFrame,
    max_moves: int = 10,
    value_threshold: float = DEFAULT_VALUE_THRESHOLD,
    top_k: int = DEFAULT_SCREEN_TOP_K,
    include_bv: bool = True,
    my_team_name: str | None = None,
    banked_totals: dict[str, dict[str, float]] | None = None,
    season_fraction_remaining: float | None = None,
) -> dict:
    """Find the best reachable roster via greedy FA swaps.

    Each iteration:
        1. Compute league state (MEW-lineup fixed-point)
        2. Enrich players with MEW and BV
        3. Screen all FA swaps
        4. Exact-evaluate top K candidates
        5. If best Value > threshold: execute swap, loop
        6. Else: stop

    MEW is recomputed each iteration (W3).

    Args:
        players: Silver table with FV column.
        max_moves: Maximum swaps.
        value_threshold: Stop when best Value < this.
        top_k: Candidates to exact-evaluate per iteration.
        include_bv: Include ΔBV in Value.
        my_team_name: Team to optimize for. Defaults to MY_TEAM_NAME.

    Returns:
        {
            'moves': list[dict],
            'drops': set[str],
            'adds': set[str],
            'starting_ew': float,
            'final_ew': float,
            'total_value': float,
        }
    """
    if my_team_name is None:
        my_team_name = MY_TEAM_NAME
    assert "FV" in players.columns, (
        "run_greedy_optimization: players must have FV. Call add_fantasy_value() first."
    )

    players = players.copy()
    moves: list[dict] = []
    all_drops: set[str] = set()
    all_adds: set[str] = set()
    starting_ew = None
    current_ew = None

    for iteration in range(max_moves):
        print(f"\n=== Greedy iteration {iteration + 1}/{max_moves} ===")

        # 1. Compute league state
        state = compute_league_state(
            players,
            my_team_name=my_team_name,
            banked_totals=banked_totals,
            season_fraction_remaining=season_fraction_remaining,
        )

        if starting_ew is None:
            starting_ew = state["current_ew"]
        current_ew = state["current_ew"]

        # 2. Enrich with optimal_slot, MEW, BV
        players = assign_optimal_slots(
            players,
            state["my_lineup"],
            state["opponent_lineups"],
            state["opponent_teams"],
        )
        players = add_mew(players, state["my_totals"], state["gradient"])
        my_roster_names = state["my_roster_names"]
        players = add_bench_value(players, state["my_lineup"], my_roster_names)

        # Compute current total BV
        my_bench = my_roster_names - state["my_starters"]
        current_total_bv = float(players[players["Name"].isin(my_bench)]["BV"].sum())

        # 3. Screen
        candidates = screen_swaps(
            players,
            my_roster_names,
            state["my_lineup"],
            top_k=top_k,
        )

        if len(candidates) == 0:
            print("No candidates found. Stopping.")
            break

        # 4. Exact-evaluate top K
        evaluated = evaluate_top_k(
            candidates,
            my_roster_names,
            players,
            state["opponent_totals"],
            state["category_sigmas"],
            current_ew,
            current_total_bv,
            include_bv=include_bv,
            my_banked_totals=state["my_banked"],
            my_lineup=state["my_lineup"],
        )

        # Churn guard: never re-add a player dropped earlier in this run or
        # re-drop one we just added. Greedy oscillation (A out, A back in)
        # signals candidates whose "value" is approximation noise, not signal.
        if len(evaluated) > 0:
            churn = evaluated["fa_name"].isin(all_drops) | evaluated["drop_name"].isin(
                all_adds
            )
            if churn.any():
                print(f"Churn guard: skipping {int(churn.sum())} reversal candidates")
                evaluated = evaluated[~churn].reset_index(drop=True)

        if len(evaluated) == 0 or evaluated.iloc[0]["value"] < value_threshold:
            print(
                f"Best value {evaluated.iloc[0]['value']:.2f} < "
                f"threshold {value_threshold}. Stopping."
                if len(evaluated) > 0
                else "No evaluated candidates. Stopping."
            )
            break

        # 5. Execute best swap
        best = evaluated.iloc[0]
        drop_name = best["drop_name"]
        add_name = best["fa_name"]

        print(
            f"Move {iteration + 1}: drop {drop_name}, add {add_name} "
            f"(ΔEW={best['msv_exact']:.2f}, ΔBV={best['delta_bv']:.2f}, "
            f"value={best['value']:.2f})"
        )

        moves.append(
            {
                "drop": drop_name,
                "add": add_name,
                "delta_ew": best["msv_exact"],
                "delta_bv": best["delta_bv"],
                "value": best["value"],
            }
        )
        all_drops.add(drop_name)
        all_adds.add(add_name)

        # Update roster in players DataFrame
        players.loc[players["Name"] == drop_name, "owner"] = None
        players.loc[players["Name"] == drop_name, "roster_status"] = None
        players.loc[players["Name"] == add_name, "owner"] = my_team_name
        players.loc[players["Name"] == add_name, "roster_status"] = "active"

        current_ew = best["new_ew"]

    total_value = sum(m["value"] for m in moves)
    final_ew = current_ew if current_ew is not None else starting_ew

    print(
        f"\nGreedy optimization: {len(moves)} moves, "
        f"EW {starting_ew:.2f} → {final_ew:.2f} "
        f"(total value: {total_value:.2f})"
    )

    return {
        "moves": moves,
        "drops": all_drops,
        "adds": all_adds,
        "starting_ew": starting_ew,
        "final_ew": final_ew,
        "total_value": total_value,
    }
