"""
Trade evaluation: score specific trades and search for good trades.

Trades are mathematically identical to FA swaps — same compute_exact_msv,
same Value metric. The differences:
1. The "adds" come from an opponent's roster, not the FA pool.
2. Two PV constraints must hold (both aggregate and per-player max).
3. The affected opponent's totals change post-trade (1 extra MILP).

PV feasibility uses two checks:
  - Aggregate: opponent's total PV loss ≤ pv_max_loss_frac of what they give up.
  - Per-player max: the highest-PV player received can't vastly exceed the
    highest-PV player sent. This prevents "trade up by quantity" — aggregating
    mid-tier players to acquire a superstar.
"""

from itertools import combinations

import pandas as pd
from tqdm.auto import tqdm

from .config import MY_TEAM_NAME
from .lineup_solver import (
    compute_totals_for_starters,
    solve_lineup,
)
from .player_scoring import add_mew
from .players import get_eligible_slots
from .swap_evaluator import add_bench_value
from .win_model import (
    compute_ew_gradient,
    compute_win_probability,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Maximum fraction of PV the opponent will accept losing.
# 0.15 = opponent accepts trades where they lose up to 15% of their PV.
DEFAULT_PV_MAX_LOSS_FRAC: float = 0.15
DEFAULT_TRADE_MAX_SIZE: int = 2

# ============================================================================
# 10a. evaluate_trade — Score a specific trade proposal
# ============================================================================


def evaluate_trade(
    send_names: set[str],
    receive_names: set[str],
    my_roster_names: set[str],
    opponent_roster_names: set[str],
    trade_opponent_id: int,
    players: pd.DataFrame,
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    pv_max_loss_frac: float = DEFAULT_PV_MAX_LOSS_FRAC,
    my_lineup: dict[str, str] | None = None,
) -> dict:
    """Evaluate a specific trade, including opponent roster change and ΔBV.

    Same math as any swap (Value = MSV + ΔBV), plus PV check and opponent
    lineup re-solve. See MATHEMATICAL_FRAMEWORK §7.

    Supports imbalanced trades:
    - 2-for-1: Auto-fills with best FA to maintain roster size.
    - 1-for-2: Auto-drops lowest-MEW bench player to maintain roster size.

    PV feasibility uses a **relative** check: the opponent rejects if they
    lose more than pv_max_loss_frac of their PV in the trade.  E.g. 0.15
    means the opponent accepts up to a 15 % PV loss.

    Args:
        send_names: Players I send (leave my roster).
        receive_names: Players I receive (join my roster).
        my_roster_names: Current my roster.
        opponent_roster_names: Current opponent's roster.
        trade_opponent_id: 1-indexed opponent ID.
        players: Players DataFrame with FV, MEW, PV.
        opponent_totals: All opponents' totals.
        category_sigmas: σ_c per category.
        current_ew: Current EW.
        current_total_bv: Current total BV of my bench.
        pv_max_loss_frac: Max fraction of PV the opponent will accept losing.
            0.15 = opponent accepts up to 15% PV loss.
        my_lineup: Optional {name: slot} dict for current lineup.
            Needed for 1-for-2 trades to identify bench players to drop.

    Returns:
        {
            'msv': float,
            'new_ew': float,
            'delta_bv': float,
            'value': float,
            'pv_balance': float,
            'opp_pv_loss_pct': float,  # % PV the opponent loses (positive = bad for opp)
            'pv_feasible': bool,
            'new_totals': dict,
            'new_lineup': dict,
            'auto_fa_add': str | None,  # FA player auto-added (2-for-1)
            'auto_drop': str | None,     # Bench player auto-dropped (1-for-2)
        }
    """
    assert "PV" in players.columns, "evaluate_trade: players must have PV column"

    # Convert to mutable sets for potential modification
    send_names = set(send_names)
    receive_names = set(receive_names)
    auto_fa_add = None
    auto_drop = None

    # Handle imbalanced trades
    n_send = len(send_names)
    n_recv = len(receive_names)
    mew_lookup = players.set_index("Name")["MEW"].to_dict()

    if n_send > n_recv:
        # 2-for-1: I send more, need to add FA to receive
        deficit = n_send - n_recv
        fa_names = set(players[players["owner"].isna()]["Name"])
        fa_names -= receive_names  # Don't double-count already-receiving FAs
        if not fa_names:
            return {
                "msv": 0.0,
                "new_ew": current_ew,
                "delta_bv": 0.0,
                "value": 0.0,
                "pv_balance": 0.0,
                "pv_feasible": False,
                "new_totals": {},
                "new_lineup": {},
                "auto_fa_add": None,
                "auto_drop": None,
                "error": "No FAs available to balance 2-for-1 trade",
            }
        # Pick best FA by MEW
        for _ in range(deficit):
            best_fa = max(fa_names, key=lambda n: mew_lookup.get(n, 0.0))
            receive_names.add(best_fa)
            fa_names.discard(best_fa)
            auto_fa_add = best_fa  # Store last one added for reporting

    elif n_recv > n_send:
        # 1-for-2: I receive more, need to drop bench player
        deficit = n_recv - n_send
        if my_lineup is None:
            # Fall back: compute lineup to find bench
            starters = set(solve_lineup(my_roster_names, players, "MEW").keys())
        else:
            starters = set(my_lineup.keys())
        bench = my_roster_names - starters - send_names
        if len(bench) < deficit:
            return {
                "msv": 0.0,
                "new_ew": current_ew,
                "delta_bv": 0.0,
                "value": 0.0,
                "pv_balance": 0.0,
                "pv_feasible": False,
                "new_totals": {},
                "new_lineup": {},
                "auto_fa_add": None,
                "auto_drop": None,
                "error": f"Not enough bench players ({len(bench)}) to drop for 1-for-{n_recv} trade",
            }
        # Drop lowest-MEW bench players
        sorted_bench = sorted(bench, key=lambda n: mew_lookup.get(n, 0.0))
        for i in range(deficit):
            drop_name = sorted_bench[i]
            send_names.add(drop_name)
            auto_drop = drop_name  # Store last one dropped for reporting

    # After balancing, send and receive should be equal
    assert len(send_names) == len(receive_names), (
        f"evaluate_trade: |send| = {len(send_names)} != "
        f"|receive| = {len(receive_names)} after balancing. This is a bug."
    )

    # 1. PV check: only on opponent-routed portion (MF §1)
    # recv_from_opp: players leaving opponent's roster and coming to me
    # send_to_opp: players leaving my roster and going to the opponent
    #   (NOT including players I'm dropping to FA as part of an imbalanced trade)
    pv_lookup = players.set_index("Name")["PV"].to_dict()
    recv_from_opp = receive_names & opponent_roster_names
    recv_from_fa = receive_names - recv_from_opp

    # Infer send_to_opp: in an imbalanced trade like 1-for-2 + FA drop,
    # some of send_names go to FA rather than the opponent. The opponent-routed
    # sends are the highest-PV players from send_names (up to the trade size).
    # For balanced trades (1-for-1, 2-for-2), all sends go to the opponent.
    # For 2-for-1+FA fill, all sends go to the opponent.
    # For 1-for-2+FA drop, only some sends go to the opponent.
    send_from_roster = send_names & my_roster_names
    n_fa_drops = len(recv_from_fa)
    if n_fa_drops > 0 and len(send_from_roster) > len(recv_from_opp):
        # Imbalanced: some of my sends are FA drops, not trades to opponent.
        # The FA-dropped players are the lowest-PV ones (least trade-worthy).
        sorted_sends = sorted(
            send_from_roster, key=lambda n: pv_lookup.get(n, 0.0), reverse=True
        )
        n_to_opp = len(send_from_roster) - n_fa_drops
        send_to_opp = set(sorted_sends[:n_to_opp])
    else:
        send_to_opp = send_from_roster

    pv_send_val = sum(pv_lookup.get(n, 0.0) for n in send_to_opp)
    pv_recv_val = sum(pv_lookup.get(n, 0.0) for n in recv_from_opp)
    pv_balance = pv_send_val - pv_recv_val

    # Two-part PV feasibility check:
    #   1. Aggregate: opponent's total PV loss ≤ threshold
    #   2. Per-player max: can't get a star without sending one back
    #
    # opp gives up: pv_recv_val (I receive from them)
    # opp receives: pv_send_val (I send to them)
    if pv_recv_val > 0:
        opp_pv_loss_frac = (pv_recv_val - pv_send_val) / pv_recv_val
    else:
        opp_pv_loss_frac = 0.0
    opp_pv_loss_pct = round(opp_pv_loss_frac * 100, 1)
    agg_ok = opp_pv_loss_frac <= pv_max_loss_frac

    max_pv_sent = max((pv_lookup.get(n, 0.0) for n in send_to_opp), default=0.0)
    max_pv_recv = max((pv_lookup.get(n, 0.0) for n in recv_from_opp), default=0.0)
    max_ok = (
        max_pv_sent >= max_pv_recv * (1 - pv_max_loss_frac) if max_pv_recv > 0 else True
    )

    pv_feasible = agg_ok and max_ok

    if not pv_feasible:
        return {
            "msv": 0.0,
            "new_ew": current_ew,
            "delta_bv": 0.0,
            "value": 0.0,
            "pv_balance": pv_balance,
            "opp_pv_loss_pct": opp_pv_loss_pct,
            "pv_feasible": False,
            "new_totals": {},
            "new_lineup": {},
            "auto_fa_add": auto_fa_add,
            "auto_drop": auto_drop,
        }

    # 2. My new roster
    my_new_roster = (my_roster_names - send_names) | receive_names
    assert len(my_new_roster) == len(my_roster_names), (
        f"evaluate_trade: my new roster size {len(my_new_roster)} != "
        f"original {len(my_roster_names)}"
    )

    # 3. My new lineup (MEW objective)
    my_new_lineup = solve_lineup(my_new_roster, players, "MEW")
    my_new_totals = compute_totals_for_starters(set(my_new_lineup.keys()), players)

    # 4. Opponent's new roster and lineup (FV objective)
    # Only send_to_opp goes to the opponent; FA drops don't.
    opp_new_roster = (opponent_roster_names - recv_from_opp) | send_to_opp
    opp_new_lineup = solve_lineup(opp_new_roster, players, "FV")
    opp_new_totals = compute_totals_for_starters(set(opp_new_lineup.keys()), players)

    # 5. Updated opponent totals
    updated_opponent_totals = {**opponent_totals}
    updated_opponent_totals[trade_opponent_id] = opp_new_totals

    # 6. New EW
    new_ew, _ = compute_win_probability(
        my_new_totals, updated_opponent_totals, category_sigmas
    )
    msv = new_ew - current_ew

    # 7. ΔBV
    new_gradient = compute_ew_gradient(
        my_new_totals, updated_opponent_totals, category_sigmas
    )
    work = add_mew(players, my_new_totals, new_gradient)
    work = add_bench_value(work, my_new_lineup, my_new_roster)

    new_bench = my_new_roster - set(my_new_lineup.keys())
    new_total_bv = float(work[work["Name"].isin(new_bench)]["BV"].sum())
    delta_bv = new_total_bv - current_total_bv

    value = msv + delta_bv

    return {
        "msv": msv,
        "new_ew": new_ew,
        "delta_bv": delta_bv,
        "value": value,
        "pv_balance": pv_balance,
        "opp_pv_loss_pct": opp_pv_loss_pct,
        "pv_feasible": pv_feasible,
        "new_totals": my_new_totals,
        "new_lineup": my_new_lineup,
        "auto_fa_add": auto_fa_add,
        "auto_drop": auto_drop,
    }


# ============================================================================
# 10b. search_trades — Find good trades automatically
# ============================================================================


def search_trades(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_lineup: dict[str, str],
    opponent_rosters: dict[int, set[str]],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    pv_max_loss_frac: float = DEFAULT_PV_MAX_LOSS_FRAC,
    max_trade_size: int = DEFAULT_TRADE_MAX_SIZE,
    top_k: int = 100,
    min_value: float = 0.0,
    my_team_name: str | None = None,
) -> list[dict]:
    """Enumerate and rank PV-feasible trades, including imbalanced.

    For each opponent o:
        1. TARGETS: their players with high MEW (I want them)
        2. CHIPS: my players sorted by PV (tradeable value to opponents)
        3. Enumerate trade shapes
        4. Filter by relative PV constraint
        5. Score by MEW-based approximation
        6. Exact-evaluate top K

    Args:
        players: DataFrame with FV, MEW, PV.
        my_roster_names: My current roster.
        my_lineup: {name: slot} for my team.
        opponent_rosters: {opp_id: set of names}.
        opponent_totals: {opp_id: {cat: total}}.
        category_sigmas: σ_c per category.
        current_ew: Current EW.
        current_total_bv: Current total BV.
        pv_max_loss_frac: Max fraction of PV opponent will accept losing.
        max_trade_size: Max players per side.
        top_k: Top trades to exact-evaluate.
        min_value: Minimum value to include in results.
        my_team_name: Team name for filtering. Defaults to MY_TEAM_NAME.

    Returns list of:
        {
            'send': list[str], 'receive': list[str], 'opponent': str,
            'msv_exact': float, 'delta_bv': float, 'value': float,
            'pv_balance': float, 'new_ew': float,
        }
    """
    if my_team_name is None:
        my_team_name = MY_TEAM_NAME
    assert "MEW" in players.columns, "search_trades: need MEW column"
    assert "PV" in players.columns, "search_trades: need PV column"

    mew_lookup = players.set_index("Name")["MEW"].to_dict()
    pv_lookup = players.set_index("Name")["PV"].to_dict()

    my_starters = set(my_lineup.keys())
    fa_names = set(players[players["owner"].isna()]["Name"])

    # Get best FA by MEW (precompute once)
    best_fa = None
    best_fa_mew = -float("inf")
    for fa_name in fa_names:
        fa_mew = mew_lookup.get(fa_name, 0.0)
        if fa_mew > best_fa_mew:
            best_fa_mew = fa_mew
            best_fa = fa_name

    # All my roster players sorted by PV — every player is a potential chip
    my_players = players[players["Name"].isin(my_roster_names)].copy()
    chips = my_players.sort_values("PV", ascending=False)["Name"].tolist()

    # For 2-player combos, cap to avoid truly degenerate explosion.
    # C(20,2) = 190 combos per side → 190×190 = 36,100 per opponent.
    # With ~9 opponents that's ~325K approximate evals — still sub-second.
    COMBO_CAP = 20

    def _opp_would_reject(
        pv_send: float,
        pv_recv: float,
        max_pv_send: float | None = None,
        max_pv_recv: float | None = None,
    ) -> bool:
        """True if opponent rejects (aggregate loss OR per-player max violation).

        For 1-for-1 trades, max_pv defaults are the same as the aggregates.
        For multi-player trades, pass the max individual PV on each side.
        """
        if pv_recv <= 0:
            return False
        if (pv_recv - pv_send) / pv_recv > pv_max_loss_frac:
            return True
        ms = max_pv_send if max_pv_send is not None else pv_send
        mr = max_pv_recv if max_pv_recv is not None else pv_recv
        if mr > 0 and ms < mr * (1 - pv_max_loss_frac):
            return True
        return False

    approximate_trades: list[dict] = []

    for opp_id, opp_roster in tqdm(opponent_rosters.items(), desc="Searching trades"):
        opp_players = players[players["Name"].isin(opp_roster)].copy()
        if len(opp_players) == 0:
            continue

        # All opponent players sorted by MEW (value to me)
        targets = opp_players.sort_values("MEW", ascending=False)["Name"].tolist()

        # --- 1-for-1 trades (full roster × full roster) ---
        for target in targets:
            pv_recv = pv_lookup.get(target, 0.0)
            for chip in chips:
                pv_send = pv_lookup.get(chip, 0.0)
                if _opp_would_reject(pv_send, pv_recv):
                    continue
                msv_approx = mew_lookup.get(target, 0.0) - mew_lookup.get(chip, 0.0)
                approximate_trades.append(
                    {
                        "send": [chip],
                        "receive": [target],
                        "opponent_id": opp_id,
                        "msv_approx": msv_approx,
                    }
                )

        if max_trade_size < 2:
            continue

        _targets_2 = targets[:COMBO_CAP]
        _chips_2 = chips[:COMBO_CAP]

        # --- 2-for-2 trades ---
        for t1, t2 in combinations(_targets_2, 2):
            pv_t1, pv_t2 = pv_lookup.get(t1, 0.0), pv_lookup.get(t2, 0.0)
            pv_recv = pv_t1 + pv_t2
            max_recv = max(pv_t1, pv_t2)
            for c1, c2 in combinations(_chips_2, 2):
                pv_c1, pv_c2 = pv_lookup.get(c1, 0.0), pv_lookup.get(c2, 0.0)
                pv_send = pv_c1 + pv_c2
                if _opp_would_reject(pv_send, pv_recv, max(pv_c1, pv_c2), max_recv):
                    continue
                msv_approx = (
                    mew_lookup.get(t1, 0.0)
                    + mew_lookup.get(t2, 0.0)
                    - mew_lookup.get(c1, 0.0)
                    - mew_lookup.get(c2, 0.0)
                )
                approximate_trades.append(
                    {
                        "send": [c1, c2],
                        "receive": [t1, t2],
                        "opponent_id": opp_id,
                        "msv_approx": msv_approx,
                    }
                )

        # --- 2-for-1 + FA fill ---
        if best_fa is not None:
            for target in _targets_2:
                pv_recv = pv_lookup.get(target, 0.0)
                for c1, c2 in combinations(_chips_2, 2):
                    pv_c1, pv_c2 = pv_lookup.get(c1, 0.0), pv_lookup.get(c2, 0.0)
                    pv_send = pv_c1 + pv_c2
                    if _opp_would_reject(pv_send, pv_recv, max(pv_c1, pv_c2), pv_recv):
                        continue
                    msv_approx = (
                        mew_lookup.get(target, 0.0)
                        + best_fa_mew
                        - mew_lookup.get(c1, 0.0)
                        - mew_lookup.get(c2, 0.0)
                    )
                    approximate_trades.append(
                        {
                            "send": [c1, c2],
                            "receive": [target, best_fa],
                            "opponent_id": opp_id,
                            "msv_approx": msv_approx,
                        }
                    )

        # --- 1-for-2 + FA drop ---
        for t1, t2 in combinations(_targets_2, 2):
            pv_t1, pv_t2 = pv_lookup.get(t1, 0.0), pv_lookup.get(t2, 0.0)
            pv_recv = pv_t1 + pv_t2
            max_recv = max(pv_t1, pv_t2)
            for chip in _chips_2:
                pv_send = pv_lookup.get(chip, 0.0)
                if _opp_would_reject(pv_send, pv_recv, pv_send, max_recv):
                    continue
                bench = my_roster_names - my_starters - {chip}
                if not bench:
                    continue
                drop_name = min(bench, key=lambda n: mew_lookup.get(n, 0.0))
                msv_approx = (
                    mew_lookup.get(t1, 0.0)
                    + mew_lookup.get(t2, 0.0)
                    - mew_lookup.get(chip, 0.0)
                    - mew_lookup.get(drop_name, 0.0)
                )
                approximate_trades.append(
                    {
                        "send": [chip, drop_name],
                        "receive": [t1, t2],
                        "opponent_id": opp_id,
                        "msv_approx": msv_approx,
                    }
                )

    if not approximate_trades:
        print("search_trades: no PV-feasible candidates found")
        return []

    # Rank by approximate MSV, take top K for exact evaluation
    approximate_trades.sort(key=lambda t: t["msv_approx"], reverse=True)
    top_candidates = approximate_trades[:top_k]

    print(
        f"search_trades: {len(approximate_trades)} PV-feasible candidates, "
        f"exact-evaluating top {len(top_candidates)}"
    )

    # Exact evaluation
    results: list[dict] = []
    opponent_teams = sorted(
        t
        for t in players[players["owner"].notna()]["owner"].unique()
        if t != my_team_name
    )

    for trade in tqdm(top_candidates, desc="Evaluating trades"):
        opp_id = trade["opponent_id"]
        send_set = set(trade["send"])
        recv_set = set(trade["receive"])

        result = evaluate_trade(
            send_names=send_set,
            receive_names=recv_set,
            my_roster_names=my_roster_names,
            opponent_roster_names=opponent_rosters[opp_id],
            trade_opponent_id=opp_id,
            players=players,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
            current_ew=current_ew,
            current_total_bv=current_total_bv,
            pv_max_loss_frac=pv_max_loss_frac,
            my_lineup=my_lineup,
        )

        if not result["pv_feasible"]:
            continue

        if result["value"] < min_value:
            continue

        opp_name = (
            opponent_teams[opp_id - 1]
            if opp_id - 1 < len(opponent_teams)
            else f"Opponent {opp_id}"
        )
        results.append(
            {
                "send": trade["send"],
                "receive": trade["receive"],
                "opponent": opp_name,
                "msv_exact": result["msv"],
                "delta_bv": result["delta_bv"],
                "value": result["value"],
                "opp_pv_loss_pct": result["opp_pv_loss_pct"],
                "new_ew": result["new_ew"],
            }
        )

    results.sort(key=lambda r: r["value"], reverse=True)

    print(f"search_trades: {len(results)} trades above min_value {min_value}")
    return results


# ============================================================================
# 10c. search_trades_for_players — Find trades involving specific players
# ============================================================================


def search_trades_for_players(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_lineup: dict[str, str],
    opponent_rosters: dict[int, set[str]],
    opponent_totals: dict[int, dict[str, float]],
    category_sigmas: dict[str, float],
    current_ew: float,
    current_total_bv: float,
    pv_max_loss_frac: float = DEFAULT_PV_MAX_LOSS_FRAC,
    must_send: set[str] | None = None,
    must_receive: set[str] | None = None,
    opponent_filter: set[int] | None = None,
    top_k: int = 50,
    min_value: float = -1.0,
    my_team_name: str | None = None,
) -> list[dict]:
    """Search for trades involving specific players.

    Use cases:
    - must_send only: "What can I get for Cody Bellinger?"
    - must_receive only: "Who do I need to give up for Juan Soto?"
    - Both: Evaluate specific trade shapes

    Args:
        players: DataFrame with FV, MEW, PV.
        my_roster_names: My current roster.
        my_lineup: {name: slot} for my team.
        opponent_rosters: {opp_id: set of names}.
        opponent_totals: {opp_id: {cat: total}}.
        category_sigmas: σ_c per category.
        current_ew: Current EW.
        current_total_bv: Current total BV.
        pv_max_loss_frac: Max fraction of PV opponent will accept losing.
        must_send: Players that must be in the send side (my players).
        must_receive: Players that must be in the receive side (opponent players).
        opponent_filter: If set, only search these opponent IDs.
        top_k: Max trades to exact-evaluate.
        min_value: Minimum value to include in results.
        my_team_name: Team name for filtering. Defaults to MY_TEAM_NAME.

    Returns list of trade dicts with same structure as search_trades.
    """
    if my_team_name is None:
        my_team_name = MY_TEAM_NAME
    assert "MEW" in players.columns, "search_trades_for_players: need MEW column"
    assert "PV" in players.columns, "search_trades_for_players: need PV column"

    must_send = must_send or set()
    must_receive = must_receive or set()

    mew_lookup = players.set_index("Name")["MEW"].to_dict()
    pv_lookup = players.set_index("Name")["PV"].to_dict()

    my_starters = set(my_lineup.keys())
    fa_names = set(players[players["owner"].isna()]["Name"])

    # Identify additional trade chips (beyond must_send)
    my_players = players[players["Name"].isin(my_roster_names - must_send)].copy()
    if len(my_players) > 0:
        my_players["chip_score"] = my_players["PV"].rank(pct=True) - my_players[
            "MEW"
        ].rank(pct=True)
        extra_chips = my_players.nlargest(min(10, len(my_players)), "chip_score")[
            "Name"
        ].tolist()
    else:
        extra_chips = []

    # Determine which opponents to search
    if must_receive:
        # Find which opponent owns the must_receive players
        recv_owners = set()
        for name in must_receive:
            owner = players.loc[players["Name"] == name, "owner"].iloc[0]
            if pd.notna(owner) and owner != MY_TEAM_NAME:
                for oid, roster in opponent_rosters.items():
                    if name in roster:
                        recv_owners.add(oid)
                        break
        if opponent_filter:
            search_opps = recv_owners & opponent_filter
        else:
            search_opps = recv_owners
    elif opponent_filter:
        search_opps = opponent_filter
    else:
        search_opps = set(opponent_rosters.keys())

    def _opp_would_reject(
        pv_send: float,
        pv_recv: float,
        max_pv_send: float | None = None,
        max_pv_recv: float | None = None,
    ) -> bool:
        """True if opponent rejects (aggregate loss OR per-player max violation)."""
        if pv_recv <= 0:
            return False
        if (pv_recv - pv_send) / pv_recv > pv_max_loss_frac:
            return True
        ms = max_pv_send if max_pv_send is not None else pv_send
        mr = max_pv_recv if max_pv_recv is not None else pv_recv
        if mr > 0 and ms < mr * (1 - pv_max_loss_frac):
            return True
        return False

    approximate_trades: list[dict] = []

    for opp_id in search_opps:
        opp_roster = opponent_rosters[opp_id]
        opp_players = players[players["Name"].isin(opp_roster - must_receive)].copy()

        if len(opp_players) > 0:
            extra_targets = opp_players.nlargest(min(15, len(opp_players)), "MEW")[
                "Name"
            ].tolist()
        else:
            extra_targets = []

        n_must_send = len(must_send)
        n_must_recv = len(must_receive)

        # Case: must_send specified, find what we can get
        if must_send and not must_receive:
            # 1-for-1: send must_send, get target
            for target in extra_targets:
                if target in opp_roster:
                    for chip in must_send:
                        pv_send = pv_lookup.get(chip, 0.0)
                        pv_recv = pv_lookup.get(target, 0.0)
                        if _opp_would_reject(pv_send, pv_recv):
                            continue
                        msv_approx = mew_lookup.get(target, 0.0) - mew_lookup.get(
                            chip, 0.0
                        )
                        approximate_trades.append(
                            {
                                "send": [chip],
                                "receive": [target],
                                "opponent_id": opp_id,
                                "msv_approx": msv_approx,
                            }
                        )

            # 1-for-2: send must_send + drop bench, get 2 targets
            bench = my_roster_names - my_starters - must_send
            if bench and n_must_send == 1:
                drop_name = min(bench, key=lambda n: mew_lookup.get(n, 0.0))
                for i, t1 in enumerate(extra_targets[:8]):
                    for t2 in extra_targets[i + 1 : 8]:
                        pv_send_total = sum(pv_lookup.get(c, 0.0) for c in must_send)
                        pv_t1, pv_t2 = pv_lookup.get(t1, 0.0), pv_lookup.get(t2, 0.0)
                        pv_recv = pv_t1 + pv_t2
                        if _opp_would_reject(
                            pv_send_total, pv_recv, pv_send_total, max(pv_t1, pv_t2)
                        ):
                            continue
                        msv_approx = (
                            mew_lookup.get(t1, 0.0)
                            + mew_lookup.get(t2, 0.0)
                            - sum(mew_lookup.get(c, 0.0) for c in must_send)
                            - mew_lookup.get(drop_name, 0.0)
                        )
                        approximate_trades.append(
                            {
                                "send": list(must_send) + [drop_name],
                                "receive": [t1, t2],
                                "opponent_id": opp_id,
                                "msv_approx": msv_approx,
                            }
                        )

            # 2-for-1: send must_send + extra chip, get 1 target + FA fill
            if n_must_send >= 1:
                best_fa = (
                    max(fa_names, key=lambda n: mew_lookup.get(n, 0.0))
                    if fa_names
                    else None
                )
                if best_fa:
                    for target in extra_targets[:10]:
                        pv_recv = pv_lookup.get(target, 0.0)
                        for extra_chip in extra_chips[:10]:
                            send_list = list(must_send) + [extra_chip]
                            send_pvs = [pv_lookup.get(c, 0.0) for c in send_list]
                            pv_send = sum(send_pvs)
                            if _opp_would_reject(
                                pv_send, pv_recv, max(send_pvs), pv_recv
                            ):
                                continue
                            msv_approx = (
                                mew_lookup.get(target, 0.0)
                                + mew_lookup.get(best_fa, 0.0)
                                - sum(mew_lookup.get(c, 0.0) for c in send_list)
                            )
                            approximate_trades.append(
                                {
                                    "send": send_list,
                                    "receive": [target, best_fa],
                                    "opponent_id": opp_id,
                                    "msv_approx": msv_approx,
                                }
                            )

        # Case: must_receive specified, find what we need to give up
        elif must_receive and not must_send:
            must_recv_list = list(must_receive)
            # 1-for-1: give chip, get must_receive
            for chip in extra_chips:
                for target in must_recv_list:
                    pv_send = pv_lookup.get(chip, 0.0)
                    pv_recv = pv_lookup.get(target, 0.0)
                    if _opp_would_reject(pv_send, pv_recv):
                        continue
                    msv_approx = mew_lookup.get(target, 0.0) - mew_lookup.get(chip, 0.0)
                    approximate_trades.append(
                        {
                            "send": [chip],
                            "receive": [target],
                            "opponent_id": opp_id,
                            "msv_approx": msv_approx,
                        }
                    )

            # 2-for-1: give 2 chips, get must_receive + FA
            if n_must_recv == 1 and fa_names:
                best_fa = max(fa_names, key=lambda n: mew_lookup.get(n, 0.0))
                recv_pvs = [pv_lookup.get(t, 0.0) for t in must_recv_list]
                pv_recv = sum(recv_pvs)
                max_recv = max(recv_pvs)
                for i, c1 in enumerate(extra_chips[:8]):
                    for c2 in extra_chips[i + 1 : 8]:
                        pv_c1, pv_c2 = pv_lookup.get(c1, 0.0), pv_lookup.get(c2, 0.0)
                        pv_send = pv_c1 + pv_c2
                        if _opp_would_reject(
                            pv_send, pv_recv, max(pv_c1, pv_c2), max_recv
                        ):
                            continue
                        msv_approx = (
                            sum(mew_lookup.get(t, 0.0) for t in must_recv_list)
                            + mew_lookup.get(best_fa, 0.0)
                            - mew_lookup.get(c1, 0.0)
                            - mew_lookup.get(c2, 0.0)
                        )
                        approximate_trades.append(
                            {
                                "send": [c1, c2],
                                "receive": must_recv_list + [best_fa],
                                "opponent_id": opp_id,
                                "msv_approx": msv_approx,
                            }
                        )

    if not approximate_trades:
        print("search_trades_for_players: no PV-feasible candidates found")
        return []

    # Rank and exact-evaluate
    approximate_trades.sort(key=lambda t: t["msv_approx"], reverse=True)
    top_candidates = approximate_trades[:top_k]

    print(
        f"search_trades_for_players: {len(approximate_trades)} candidates, "
        f"exact-evaluating top {len(top_candidates)}"
    )

    results: list[dict] = []
    opponent_teams = sorted(
        t
        for t in players[players["owner"].notna()]["owner"].unique()
        if t != my_team_name
    )

    for trade in tqdm(top_candidates, desc="Evaluating trades"):
        opp_id = trade["opponent_id"]
        send_set = set(trade["send"])
        recv_set = set(trade["receive"])

        result = evaluate_trade(
            send_names=send_set,
            receive_names=recv_set,
            my_roster_names=my_roster_names,
            opponent_roster_names=opponent_rosters[opp_id],
            trade_opponent_id=opp_id,
            players=players,
            opponent_totals=opponent_totals,
            category_sigmas=category_sigmas,
            current_ew=current_ew,
            current_total_bv=current_total_bv,
            pv_max_loss_frac=pv_max_loss_frac,
            my_lineup=my_lineup,
        )

        if not result["pv_feasible"]:
            continue

        if result["value"] < min_value:
            continue

        opp_name = (
            opponent_teams[opp_id - 1]
            if opp_id - 1 < len(opponent_teams)
            else f"Opponent {opp_id}"
        )
        results.append(
            {
                "send": trade["send"],
                "receive": trade["receive"],
                "opponent": opp_name,
                "msv_exact": result["msv"],
                "delta_bv": result["delta_bv"],
                "value": result["value"],
                "opp_pv_loss_pct": result["opp_pv_loss_pct"],
                "new_ew": result["new_ew"],
            }
        )

    results.sort(key=lambda r: r["value"], reverse=True)
    print(
        f"search_trades_for_players: {len(results)} trades above min_value {min_value}"
    )
    return results
