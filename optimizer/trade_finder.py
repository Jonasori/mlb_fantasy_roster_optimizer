"""
Trade evaluation: score specific trades and search for good trades.

Trades are mathematically identical to FA swaps — same compute_exact_msv,
same Value metric. The differences:
1. The "adds" come from an opponent's roster, not the FA pool.
2. Two trade-fairness constraints must hold (aggregate and per-player max).
3. The affected opponent's totals change post-trade (1 extra MILP).

**Trade value column.** Fairness is judged on one configurable per-player
column, `value_column` (default `"market_value"`, Ottoneu median auction
salary): whatever number best approximates what the rest of the league thinks
a player is worth. It is a column name, not a formula.

It must be EXOGENOUS. FV was the historical default and is wrong for this job:
it is built from our own projections, so corr(FV, MEW) = 0.96 and the
constraint ends up checking our valuation against itself, which is exactly the
circularity the fairness model exists to avoid. FV is also a z-score sum, so a
third of rostered players carry a negative value and both RELATIVE conditions
below lose their meaning.

Feasibility uses two checks on that column:
  - Aggregate: opponent's total value loss ≤ max_value_loss_frac of what
    they give up. Keeps the overall package roughly fair.
  - Per-player max: the highest-value player received can't vastly exceed
    the highest-value player sent. This prevents "trade up by quantity" —
    aggregating mid-tier players to acquire a superstar. Opponents price
    stars above the sum of their parts, so the aggregate check alone is
    not enough.
"""

from itertools import combinations

import pandas as pd
from tqdm.auto import tqdm

from .config import MAX_VALUE_LOSS_FRAC, MY_TEAM_NAME
from .lineup_solver import (
    compute_totals_for_starters,
    maybe_blend,
    solve_lineup,
)
from .player_scoring import add_mew
from .swap_evaluator import (
    _solve_lineup_holding_half,
    add_bench_value,
    composition_violation,
)
from .win_model import (
    compute_ew_gradient,
    compute_win_probability,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Maximum fraction of trade value the opponent will accept losing (from
# config.json trade_engine.max_value_loss_frac; default 0.15 = up to a 15% loss).
DEFAULT_MAX_VALUE_LOSS_FRAC: float = MAX_VALUE_LOSS_FRAC
DEFAULT_TRADE_MAX_SIZE: int = 2

# Caps on how many players enter the 2-player combinatorics.
# Broad search: C(20,2) = 190 combos per side → 190×190 = 36,100 per
# opponent; with 6 opponents that is ~217K approximate evals — still fast.
BROAD_COMBO_CAP: int = 20
# Targeted search casts a smaller net: the must-send/must-receive player is
# already fixed, so only the most plausible partners are worth enumerating.
TARGETED_CHIP_CAP: int = 10
TARGETED_TARGET_CAP: int = 15
TARGETED_PAIR_CAP: int = 8


def _opp_would_reject(
    send_value: float,
    recv_value: float,
    max_value_loss_frac: float,
    max_send_value: float | None = None,
    max_recv_value: float | None = None,
) -> bool:
    """True if the opponent rejects (aggregate loss OR per-player max violation).

    Args:
        send_value: Total value of players I send to the opponent.
        recv_value: Total value of players I receive from the opponent.
        max_value_loss_frac: Max fraction of value the opponent will lose.
        max_send_value: Largest single-player value I send. Defaults to
            send_value, which is exact for 1-for-1 trades.
        max_recv_value: Largest single-player value I receive. Defaults to
            recv_value, exact for 1-for-1 trades.

    Returns:
        True if either fairness check fails.
    """
    assert recv_value > 0 and send_value > 0, (
        f"_opp_would_reject: trade values must be positive market prices, got "
        f"send={send_value}, recv={recv_value}. Both fairness conditions are "
        f"RELATIVE, so a zero or negative side makes them meaningless — the "
        f"previous `return False` silently waved such trades through as fair. "
        f"Point value_column at a real market price (e.g. 'market_value')."
    )
    if (recv_value - send_value) / recv_value > max_value_loss_frac:
        return True
    ms = max_send_value if max_send_value is not None else send_value
    mr = max_recv_value if max_recv_value is not None else recv_value
    if mr > 0 and ms < mr * (1 - max_value_loss_frac):
        return True
    return False


def _i_overpay_by(send_value: float, recv_value: float) -> float:
    """Fraction of value at stake that I hand over above a fair exchange.

    The two spec conditions bound only the OPPONENT's loss, so a trade that
    massively overpays passes both — §7 asks for value balance ≈ 0, and this
    is the other half of ≈. Positive means I am giving up more than I get.
    """
    return (send_value - recv_value) / recv_value if recv_value > 0 else 0.0


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
    value_column: str = "market_value",
    max_value_loss_frac: float = DEFAULT_MAX_VALUE_LOSS_FRAC,
    my_lineup: dict[str, str] | None = None,
    send_to_opp_names: set[str] | None = None,
    my_banked_totals: dict[str, float] | None = None,
    trade_opponent_banked: dict[str, float] | None = None,
) -> dict:
    """Evaluate a specific trade, including opponent roster change and ΔBV.

    Same math as any swap (Value = MSV + ΔBV), plus the fairness check and an
    opponent lineup re-solve. See MATHEMATICAL_FRAMEWORK §7.

    Supports imbalanced trades:
    - 2-for-1: Auto-fills with best FA to maintain roster size.
    - 1-for-2: Auto-drops lowest-MEW bench player to maintain roster size.

    Feasibility is a **relative** check on `value_column`: the opponent
    rejects if they lose more than max_value_loss_frac of the value they
    give up, or if the best player they send out dwarfs the best player they
    get back.

    Args:
        send_names: Players I send (leave my roster).
        receive_names: Players I receive (join my roster).
        my_roster_names: Current my roster.
        opponent_roster_names: Current opponent's roster.
        trade_opponent_id: 1-indexed opponent ID.
        players: Players DataFrame with FV, MEW and `value_column`.
        opponent_totals: All opponents' totals.
        category_sigmas: σ_c per category.
        current_ew: Current EW.
        current_total_bv: Current total BV of my bench.
        value_column: Column holding each player's trade-market value —
            what opponents think they are worth. Default "FV".
        max_value_loss_frac: Max fraction of `value_column` the opponent will
            accept losing. 0.15 = opponent accepts up to a 15% loss.
        my_lineup: Optional {name: slot} dict for current lineup.
            Needed for 1-for-2 trades to identify bench players to drop.
        send_to_opp_names: Which of the (original) send_names are routed to
            the opponent. Players auto-added for balancing are routed
            automatically (auto FA fills come from FA; auto drops go to FA).
            Default None routes ALL original send_names to the opponent —
            correct for balanced trades and 2-for-1 + FA fill. For
            1-for-2 + FA drop shapes where the caller pre-included the FA
            drop in send_names, pass the opponent-routed subset explicitly
            (MATHEMATICAL_FRAMEWORK §1: moves carry dest/src tags).
        my_banked_totals: My banked YTD totals, or None. Blended with my
            post-trade ros totals before computing EW (banked-YTD model).
        trade_opponent_banked: The traded opponent's banked YTD totals, or
            None. Blended with their post-trade ros totals so the updated
            opponent standings reflect banked + ros, consistent with my side.

    Returns:
        {
            'msv': float,
            'new_ew': float,
            'delta_bv': float,
            'value': float,
            'value_balance': float,   # value sent − value received
            'opp_value_loss_pct': float,  # % of value the opponent loses
            'trade_feasible': bool,
            'new_totals': dict,
            'new_lineup': dict,
            'auto_fa_add': str | None,  # FA player auto-added (2-for-1)
            'auto_drop': str | None,     # Bench player auto-dropped (1-for-2)
        }
    """
    assert value_column in players.columns, (
        f"evaluate_trade: players must have the value column '{value_column}'. "
        f"Available columns: {sorted(players.columns)}"
    )

    # Convert to mutable sets for potential modification
    send_names = set(send_names)
    receive_names = set(receive_names)
    original_send_names = set(send_names)
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
                "value_balance": 0.0,
                "trade_feasible": False,
                "opp_value_loss_pct": 0.0,
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
                "value_balance": 0.0,
                "trade_feasible": False,
                "opp_value_loss_pct": 0.0,
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

    # 1. Fairness check: only on opponent-routed portion (MF §1)
    # recv_from_opp: players leaving opponent's roster and coming to me
    # send_to_opp: players leaving my roster and going to the opponent
    #   (NOT including players I'm dropping to FA, nor auto-balancing drops)
    value_lookup = players.set_index("Name")[value_column].to_dict()
    recv_from_opp = receive_names & opponent_roster_names

    if send_to_opp_names is None:
        # Default routing: every player the caller explicitly sent goes to
        # the opponent. Auto-balancing drops (added above) go to FA.
        send_to_opp = original_send_names & my_roster_names
    else:
        send_to_opp = set(send_to_opp_names)
        assert send_to_opp <= send_names, (
            f"evaluate_trade: send_to_opp_names must be a subset of send_names. "
            f"Extra: {sorted(send_to_opp - send_names)}"
        )

    send_value = sum(value_lookup.get(n, 0.0) for n in send_to_opp)
    recv_value = sum(value_lookup.get(n, 0.0) for n in recv_from_opp)
    value_balance = send_value - recv_value

    # Two-part feasibility check:
    #   1. Aggregate: opponent's total value loss ≤ threshold
    #   2. Per-player max: can't get a star without sending one back
    #
    # opp gives up: recv_value (I receive from them)
    # opp receives: send_value (I send to them)
    assert recv_value > 0 and send_value > 0, (
        f"evaluate_trade: '{value_column}' must be a positive market price for "
        f"every traded player. Got send={send_value}, recv={recv_value} for "
        f"send={sorted(send_to_opp)}, recv={sorted(recv_from_opp)}. A relative "
        f"fairness check on a zero or negative side is meaningless."
    )
    opp_value_loss_frac = (recv_value - send_value) / recv_value
    opp_value_loss_pct = round(opp_value_loss_frac * 100, 1)
    agg_ok = opp_value_loss_frac <= max_value_loss_frac

    max_sent = max((value_lookup.get(n, 0.0) for n in send_to_opp), default=0.0)
    max_recv = max((value_lookup.get(n, 0.0) for n in recv_from_opp), default=0.0)
    max_ok = max_sent >= max_recv * (1 - max_value_loss_frac) if max_recv > 0 else True

    trade_feasible = agg_ok and max_ok

    if not trade_feasible:
        return {
            "msv": 0.0,
            "new_ew": current_ew,
            "delta_bv": 0.0,
            "value": 0.0,
            "value_balance": value_balance,
            "opp_value_loss_pct": opp_value_loss_pct,
            "trade_feasible": False,
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

    # Roster composition, per §8 SCREEN and §9c. FA swaps already enforce this;
    # trades did not, so the highest-value proposals were free to take a roster
    # already short of pitchers and remove another one. Monotone, not absolute,
    # for the same reason screening is: an already-illegal roster must still be
    # allowed to make neutral and repairing moves.
    if composition_violation(my_new_roster, players) > composition_violation(
        my_roster_names, players
    ):
        return {
            "msv": 0.0,
            "new_ew": current_ew,
            "delta_bv": 0.0,
            "value": 0.0,
            "value_balance": value_balance,
            "opp_value_loss_pct": opp_value_loss_pct,
            "trade_feasible": False,
            "infeasible_reason": "worsens roster hitter/pitcher composition",
            "new_totals": {},
            "new_lineup": {},
            "auto_fa_add": auto_fa_add,
            "auto_drop": auto_drop,
        }

    # 3. My new lineup (MEW objective) — hold the untouched player-type half
    # fixed for single-type trades so unaffected categories don't drift from
    # MILP tie-breaking (see compute_exact_msv / _solve_lineup_holding_half).
    my_new_lineup = _solve_lineup_holding_half(
        my_new_roster, players, send_names | receive_names, my_lineup
    )
    my_new_ros = compute_totals_for_starters(set(my_new_lineup.keys()), players)
    my_new_totals = maybe_blend(my_banked_totals, my_new_ros)

    # 4. Opponent's new roster and lineup (FV objective)
    # Only send_to_opp goes to the opponent; FA drops don't.
    opp_new_roster = (opponent_roster_names - recv_from_opp) | send_to_opp
    opp_new_lineup = solve_lineup(opp_new_roster, players, "FV")
    opp_new_ros = compute_totals_for_starters(set(opp_new_lineup.keys()), players)
    opp_new_totals = maybe_blend(trade_opponent_banked, opp_new_ros)

    # 5. Updated opponent totals
    updated_opponent_totals = {**opponent_totals}
    updated_opponent_totals[trade_opponent_id] = opp_new_totals

    # 6. New EW
    new_ew, _ = compute_win_probability(
        my_new_totals, updated_opponent_totals, category_sigmas
    )
    msv = new_ew - current_ew

    # 7. ΔBV — baseline recomputed under the post-trade gradient so both
    # rosters are valued on one MEW scale (avoids phantom ΔBV from global
    # gradient rescaling; see evaluate_top_k).
    new_gradient = compute_ew_gradient(
        my_new_totals, updated_opponent_totals, category_sigmas
    )
    work = add_mew(players, my_new_totals, new_gradient)
    scored = add_bench_value(work, my_new_lineup, my_new_roster)

    new_bench = my_new_roster - set(my_new_lineup.keys())
    new_total_bv = float(scored[scored["Name"].isin(new_bench)]["BV"].sum())

    if my_lineup is not None:
        base = add_bench_value(work, my_lineup, my_roster_names)
        old_bench = my_roster_names - set(my_lineup.keys())
        baseline_bv = float(base[base["Name"].isin(old_bench)]["BV"].sum())
    else:
        baseline_bv = current_total_bv
    delta_bv = new_total_bv - baseline_bv

    value = msv + delta_bv

    return {
        "msv": msv,
        "new_ew": new_ew,
        "delta_bv": delta_bv,
        "value": value,
        "value_balance": value_balance,
        "opp_value_loss_pct": opp_value_loss_pct,
        "trade_feasible": trade_feasible,
        "new_totals": my_new_totals,
        "new_lineup": my_new_lineup,
        "auto_fa_add": auto_fa_add,
        "auto_drop": auto_drop,
    }


# ============================================================================
# 10b. Candidate enumeration (approximate stage)
# ============================================================================


def _enumerate_broad(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_starters: set[str],
    search_rosters: dict[int, set[str]],
    chips: list[str],
    mew_lookup: dict[str, float],
    value_lookup: dict[str, float],
    best_fa: str | None,
    max_value_loss_frac: float,
    max_trade_size: int,
) -> list[dict]:
    """Enumerate candidate trades across every searched opponent's roster.

    1-for-1 covers the full cross product (my roster × their roster);
    multi-player shapes are capped at BROAD_COMBO_CAP players per side.
    `chips` is my roster sorted by trade value descending, `search_rosters`
    is {opp_id: roster}, `best_fa` is the highest-MEW free agent or None.

    Returns:
        List of {send, send_to_opp, receive, opponent_id, msv_approx} dicts.
    """
    best_fa_mew = mew_lookup.get(best_fa, 0.0) if best_fa is not None else 0.0
    candidates: list[dict] = []

    for opp_id, opp_roster in tqdm(search_rosters.items(), desc="Searching trades"):
        opp_players = players[players["Name"].isin(opp_roster)]
        if len(opp_players) == 0:
            continue

        # All opponent players sorted by MEW (value to me)
        targets = opp_players.sort_values("MEW", ascending=False)["Name"].tolist()

        # --- 1-for-1 trades (full roster × full roster) ---
        for target in targets:
            recv_value = value_lookup.get(target, 0.0)
            for chip in chips:
                send_value = value_lookup.get(chip, 0.0)
                if _opp_would_reject(send_value, recv_value, max_value_loss_frac):
                    continue
                msv_approx = mew_lookup.get(target, 0.0) - mew_lookup.get(chip, 0.0)
                candidates.append(
                    {
                        "send": [chip],
                        "send_to_opp": [chip],
                        "receive": [target],
                        "opponent_id": opp_id,
                        "msv_approx": msv_approx,
                    }
                )

        if max_trade_size < 2:
            continue

        _targets_2 = targets[:BROAD_COMBO_CAP]
        _chips_2 = chips[:BROAD_COMBO_CAP]

        # --- 2-for-2 trades ---
        for t1, t2 in combinations(_targets_2, 2):
            v_t1, v_t2 = value_lookup.get(t1, 0.0), value_lookup.get(t2, 0.0)
            recv_value = v_t1 + v_t2
            max_recv = max(v_t1, v_t2)
            for c1, c2 in combinations(_chips_2, 2):
                v_c1, v_c2 = value_lookup.get(c1, 0.0), value_lookup.get(c2, 0.0)
                send_value = v_c1 + v_c2
                if _opp_would_reject(
                    send_value,
                    recv_value,
                    max_value_loss_frac,
                    max(v_c1, v_c2),
                    max_recv,
                ):
                    continue
                msv_approx = (
                    mew_lookup.get(t1, 0.0)
                    + mew_lookup.get(t2, 0.0)
                    - mew_lookup.get(c1, 0.0)
                    - mew_lookup.get(c2, 0.0)
                )
                candidates.append(
                    {
                        "send": [c1, c2],
                        "send_to_opp": [c1, c2],
                        "receive": [t1, t2],
                        "opponent_id": opp_id,
                        "msv_approx": msv_approx,
                    }
                )

        # --- 2-for-1 + FA fill ---
        if best_fa is not None:
            for target in _targets_2:
                recv_value = value_lookup.get(target, 0.0)
                for c1, c2 in combinations(_chips_2, 2):
                    v_c1, v_c2 = value_lookup.get(c1, 0.0), value_lookup.get(c2, 0.0)
                    send_value = v_c1 + v_c2
                    if _opp_would_reject(
                        send_value,
                        recv_value,
                        max_value_loss_frac,
                        max(v_c1, v_c2),
                        recv_value,
                    ):
                        continue
                    msv_approx = (
                        mew_lookup.get(target, 0.0)
                        + best_fa_mew
                        - mew_lookup.get(c1, 0.0)
                        - mew_lookup.get(c2, 0.0)
                    )
                    candidates.append(
                        {
                            "send": [c1, c2],
                            "send_to_opp": [c1, c2],
                            "receive": [target, best_fa],
                            "opponent_id": opp_id,
                            "msv_approx": msv_approx,
                        }
                    )

        # --- 1-for-2 + FA drop ---
        for t1, t2 in combinations(_targets_2, 2):
            v_t1, v_t2 = value_lookup.get(t1, 0.0), value_lookup.get(t2, 0.0)
            recv_value = v_t1 + v_t2
            max_recv = max(v_t1, v_t2)
            for chip in _chips_2:
                send_value = value_lookup.get(chip, 0.0)
                if _opp_would_reject(
                    send_value, recv_value, max_value_loss_frac, send_value, max_recv
                ):
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
                candidates.append(
                    {
                        "send": [chip, drop_name],
                        "send_to_opp": [chip],  # drop_name goes to FA
                        "receive": [t1, t2],
                        "opponent_id": opp_id,
                        "msv_approx": msv_approx,
                    }
                )

    return candidates


def _enumerate_targeted(
    players: pd.DataFrame,
    my_roster_names: set[str],
    my_starters: set[str],
    search_rosters: dict[int, set[str]],
    mew_lookup: dict[str, float],
    value_lookup: dict[str, float],
    value_column: str,
    best_fa: str | None,
    max_value_loss_frac: float,
    must_send: set[str],
    must_receive: set[str],
) -> list[dict]:
    """Enumerate candidate trades that all include the required players.

    Partner chips are picked by `chip_score` = value-rank − MEW-rank: the
    players the market likes more than my team needs, i.e. the cheapest
    real currency I have. Caps are tighter than the broad search because
    one side of every trade is already pinned. At least one of must_send /
    must_receive must be non-empty; `search_rosters` is {opp_id: roster}
    and `best_fa` the highest-MEW free agent or None.

    Returns:
        List of {send, send_to_opp, receive, opponent_id, msv_approx} dicts.
    """
    # Additional trade chips (beyond must_send): market likes them more than
    # my lineup needs them.
    my_players = players[players["Name"].isin(my_roster_names - must_send)].copy()
    if len(my_players) > 0:
        my_players["chip_score"] = my_players[value_column].rank(pct=True) - my_players[
            "MEW"
        ].rank(pct=True)
        extra_chips = my_players.nlargest(
            min(TARGETED_CHIP_CAP, len(my_players)), "chip_score"
        )["Name"].tolist()
    else:
        extra_chips = []

    candidates: list[dict] = []

    for opp_id, opp_roster in search_rosters.items():
        opp_players = players[players["Name"].isin(opp_roster - must_receive)]
        extra_targets = opp_players.nlargest(
            min(TARGETED_TARGET_CAP, len(opp_players)), "MEW"
        )["Name"].tolist()

        n_must_send = len(must_send)
        n_must_recv = len(must_receive)

        # Case: both sides pinned — one explicit shape, nothing to enumerate.
        if must_send and must_receive:
            send_list = list(must_send)
            recv_list = list(must_receive & opp_roster)
            if not recv_list:
                continue
            send_values = [value_lookup.get(c, 0.0) for c in send_list]
            recv_values = [value_lookup.get(t, 0.0) for t in recv_list]
            if _opp_would_reject(
                sum(send_values),
                sum(recv_values),
                max_value_loss_frac,
                max(send_values),
                max(recv_values),
            ):
                continue
            candidates.append(
                {
                    "send": send_list,
                    "send_to_opp": send_list,
                    "receive": recv_list,
                    "opponent_id": opp_id,
                    "msv_approx": sum(mew_lookup.get(t, 0.0) for t in recv_list)
                    - sum(mew_lookup.get(c, 0.0) for c in send_list),
                }
            )

        # Case: must_send specified, find what we can get
        elif must_send:
            # 1-for-1: send must_send, get target
            for target in extra_targets:
                recv_value = value_lookup.get(target, 0.0)
                for chip in must_send:
                    send_value = value_lookup.get(chip, 0.0)
                    if _opp_would_reject(send_value, recv_value, max_value_loss_frac):
                        continue
                    msv_approx = mew_lookup.get(target, 0.0) - mew_lookup.get(chip, 0.0)
                    candidates.append(
                        {
                            "send": [chip],
                            "send_to_opp": [chip],
                            "receive": [target],
                            "opponent_id": opp_id,
                            "msv_approx": msv_approx,
                        }
                    )

            # 1-for-2: send must_send + drop bench, get 2 targets
            bench = my_roster_names - my_starters - must_send
            if bench and n_must_send == 1:
                drop_name = min(bench, key=lambda n: mew_lookup.get(n, 0.0))
                send_value = sum(value_lookup.get(c, 0.0) for c in must_send)
                for i, t1 in enumerate(extra_targets[:TARGETED_PAIR_CAP]):
                    for t2 in extra_targets[i + 1 : TARGETED_PAIR_CAP]:
                        v_t1, v_t2 = (
                            value_lookup.get(t1, 0.0),
                            value_lookup.get(t2, 0.0),
                        )
                        if _opp_would_reject(
                            send_value,
                            v_t1 + v_t2,
                            max_value_loss_frac,
                            send_value,
                            max(v_t1, v_t2),
                        ):
                            continue
                        msv_approx = (
                            mew_lookup.get(t1, 0.0)
                            + mew_lookup.get(t2, 0.0)
                            - sum(mew_lookup.get(c, 0.0) for c in must_send)
                            - mew_lookup.get(drop_name, 0.0)
                        )
                        candidates.append(
                            {
                                "send": list(must_send) + [drop_name],
                                "send_to_opp": list(must_send),  # drop goes to FA
                                "receive": [t1, t2],
                                "opponent_id": opp_id,
                                "msv_approx": msv_approx,
                            }
                        )

            # 2-for-1: send must_send + extra chip, get 1 target + FA fill
            if best_fa is not None:
                for target in extra_targets[:TARGETED_CHIP_CAP]:
                    recv_value = value_lookup.get(target, 0.0)
                    for extra_chip in extra_chips[:TARGETED_CHIP_CAP]:
                        send_list = list(must_send) + [extra_chip]
                        send_values = [value_lookup.get(c, 0.0) for c in send_list]
                        if _opp_would_reject(
                            sum(send_values),
                            recv_value,
                            max_value_loss_frac,
                            max(send_values),
                            recv_value,
                        ):
                            continue
                        msv_approx = (
                            mew_lookup.get(target, 0.0)
                            + mew_lookup.get(best_fa, 0.0)
                            - sum(mew_lookup.get(c, 0.0) for c in send_list)
                        )
                        candidates.append(
                            {
                                "send": send_list,
                                "send_to_opp": list(send_list),
                                "receive": [target, best_fa],
                                "opponent_id": opp_id,
                                "msv_approx": msv_approx,
                            }
                        )

        # Case: must_receive specified, find what we need to give up
        else:
            must_recv_list = list(must_receive)
            # 1-for-1: give chip, get must_receive
            for chip in extra_chips:
                send_value = value_lookup.get(chip, 0.0)
                for target in must_recv_list:
                    recv_value = value_lookup.get(target, 0.0)
                    if _opp_would_reject(send_value, recv_value, max_value_loss_frac):
                        continue
                    msv_approx = mew_lookup.get(target, 0.0) - mew_lookup.get(chip, 0.0)
                    candidates.append(
                        {
                            "send": [chip],
                            "send_to_opp": [chip],
                            "receive": [target],
                            "opponent_id": opp_id,
                            "msv_approx": msv_approx,
                        }
                    )

            # 2-for-1: give 2 chips, get must_receive + FA
            if n_must_recv == 1 and best_fa is not None:
                recv_values = [value_lookup.get(t, 0.0) for t in must_recv_list]
                recv_value = sum(recv_values)
                max_recv = max(recv_values)
                for i, c1 in enumerate(extra_chips[:TARGETED_PAIR_CAP]):
                    for c2 in extra_chips[i + 1 : TARGETED_PAIR_CAP]:
                        v_c1, v_c2 = (
                            value_lookup.get(c1, 0.0),
                            value_lookup.get(c2, 0.0),
                        )
                        if _opp_would_reject(
                            v_c1 + v_c2,
                            recv_value,
                            max_value_loss_frac,
                            max(v_c1, v_c2),
                            max_recv,
                        ):
                            continue
                        msv_approx = (
                            sum(mew_lookup.get(t, 0.0) for t in must_recv_list)
                            + mew_lookup.get(best_fa, 0.0)
                            - mew_lookup.get(c1, 0.0)
                            - mew_lookup.get(c2, 0.0)
                        )
                        candidates.append(
                            {
                                "send": [c1, c2],
                                "send_to_opp": [c1, c2],
                                "receive": must_recv_list + [best_fa],
                                "opponent_id": opp_id,
                                "msv_approx": msv_approx,
                            }
                        )

    return candidates


# ============================================================================
# 10c. search_trades — Find good trades automatically
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
    value_column: str = "market_value",
    max_value_loss_frac: float = DEFAULT_MAX_VALUE_LOSS_FRAC,
    max_trade_size: int = DEFAULT_TRADE_MAX_SIZE,
    must_send: set[str] | None = None,
    must_receive: set[str] | None = None,
    opponent_filter: set[int] | None = None,
    top_k: int | None = None,
    min_value: float | None = None,
    my_team_name: str | None = None,
    my_banked_totals: dict[str, float] | None = None,
    opponent_banked: dict[int, dict[str, float] | None] | None = None,
) -> list[dict]:
    """Enumerate and rank feasible trades, including imbalanced ones.

    Two enumeration modes, chosen by whether any player is pinned:

    - **Broad** (must_send and must_receive both None): for each searched
      opponent, cross every one of my players (chips, sorted by trade value)
      against every one of theirs (targets, sorted by MEW) 1-for-1, then
      2-for-2 / 2-for-1+FA / 1-for-2+FA shapes over the top
      BROAD_COMBO_CAP of each side. Defaults: top_k=100, min_value=0.0.
    - **Targeted** (must_send and/or must_receive set): every candidate
      contains the pinned players; partners come from a `chip_score`
      shortlist (market value rank − MEW rank). Answers "what can I get for
      X?" (must_send), "what do I have to give up for Y?" (must_receive),
      or scores one explicit shape (both). Defaults: top_k=50,
      min_value=-1.0.

    Both modes then approximate-score every candidate by ΔMEW, exact-evaluate
    the top_k via evaluate_trade, and return the survivors sorted by value.

    Args:
        players: DataFrame with FV, MEW and `value_column`.
        my_roster_names: My current roster.
        my_lineup: {name: slot} for my team.
        opponent_rosters: {opp_id: set of names}.
        opponent_totals: {opp_id: {cat: total}}.
        category_sigmas: σ_c per category.
        current_ew: Current EW.
        current_total_bv: Current total BV.
        value_column: Column holding each player's trade-market value.
            Default "FV". Drives fairness checks and chip ranking.
        max_value_loss_frac: Max fraction of value the opponent will lose.
        max_trade_size: Max players per side (broad mode only).
        must_send: Players that must be on the send side (my players).
            Switches to targeted mode.
        must_receive: Players that must be on the receive side (opponent
            players). Switches to targeted mode, and restricts the search
            to the opponents who own them.
        opponent_filter: If set, only search these opponent IDs. Works in
            either mode; on its own it does NOT switch modes.
        top_k: Trades to exact-evaluate. None → 100 broad / 50 targeted.
        min_value: Minimum value to keep. None → 0.0 broad / -1.0 targeted.
        my_team_name: Team name for filtering. Defaults to MY_TEAM_NAME.
        my_banked_totals: My banked YTD totals, blended into post-trade totals.
        opponent_banked: {opp_id: banked totals} for the same blending on
            the opponent's side.

    Returns list of:
        {
            'send': list[str], 'send_to_opp': list[str],
            'receive': list[str], 'opponent': str,
            'msv_exact': float, 'delta_bv': float, 'value': float,
            'opp_value_loss_pct': float, 'new_ew': float,
        }
    """
    if my_team_name is None:
        my_team_name = MY_TEAM_NAME
    assert "MEW" in players.columns, "search_trades: need MEW column"
    assert value_column in players.columns, (
        f"search_trades: need the value column '{value_column}'. "
        f"Available columns: {sorted(players.columns)}"
    )

    must_send = set(must_send) if must_send else set()
    must_receive = set(must_receive) if must_receive else set()
    targeted = bool(must_send or must_receive)
    if top_k is None:
        top_k = 50 if targeted else 100
    if min_value is None:
        min_value = -1.0 if targeted else 0.0

    mew_lookup = players.set_index("Name")["MEW"].to_dict()
    value_lookup = players.set_index("Name")[value_column].to_dict()

    # A player the market has not priced cannot be checked for fairness: every
    # comparison against NaN is False, so an unpriced player silently satisfies
    # BOTH conditions and can be used to pad any package. Drop them from the
    # tradeable universe and say so, rather than letting them through.
    unpriced = {
        n
        for n in my_roster_names | {p for r in opponent_rosters.values() for p in r}
        if not (pd.notna(value_lookup.get(n)) and float(value_lookup.get(n, 0.0)) > 0)
    }
    if unpriced:
        print(
            f"search_trades: {len(unpriced)} player(s) have no positive "
            f"'{value_column}' and are excluded from trades (cannot price "
            f"fairness). Examples: {sorted(unpriced)[:5]}"
        )
    my_roster_names = my_roster_names - unpriced
    opponent_rosters = {
        oid: roster - unpriced for oid, roster in opponent_rosters.items()
    }

    my_starters = set(my_lineup.keys())
    fa_names = set(players[players["owner"].isna()]["Name"])
    best_fa = max(fa_names, key=lambda n: mew_lookup.get(n, 0.0)) if fa_names else None

    # Which opponents to search: those owning must_receive players, narrowed
    # by opponent_filter if given.
    if must_receive:
        search_opps = {
            oid for oid, roster in opponent_rosters.items() if roster & must_receive
        }
        if opponent_filter:
            search_opps &= opponent_filter
    elif opponent_filter:
        search_opps = set(opponent_filter) & set(opponent_rosters)
    else:
        search_opps = set(opponent_rosters)
    search_rosters = {oid: opponent_rosters[oid] for oid in sorted(search_opps)}

    if targeted:
        approximate_trades = _enumerate_targeted(
            players=players,
            my_roster_names=my_roster_names,
            my_starters=my_starters,
            search_rosters=search_rosters,
            mew_lookup=mew_lookup,
            value_lookup=value_lookup,
            value_column=value_column,
            best_fa=best_fa,
            max_value_loss_frac=max_value_loss_frac,
            must_send=must_send,
            must_receive=must_receive,
        )
    else:
        # Every player on my roster is a potential chip, most valuable first.
        my_players = players[players["Name"].isin(my_roster_names)]
        chips = my_players.sort_values(value_column, ascending=False)["Name"].tolist()
        approximate_trades = _enumerate_broad(
            players=players,
            my_roster_names=my_roster_names,
            my_starters=my_starters,
            search_rosters=search_rosters,
            chips=chips,
            mew_lookup=mew_lookup,
            value_lookup=value_lookup,
            best_fa=best_fa,
            max_value_loss_frac=max_value_loss_frac,
            max_trade_size=max_trade_size,
        )

    if not approximate_trades:
        print("search_trades: no feasible candidates found")
        return []

    # Rank by approximate MSV, take top K for exact evaluation
    approximate_trades.sort(key=lambda t: t["msv_approx"], reverse=True)
    top_candidates = approximate_trades[:top_k]

    print(
        f"search_trades: {len(approximate_trades)} feasible candidates, "
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
        send_to_opp = set(trade.get("send_to_opp", trade["send"]))

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
            value_column=value_column,
            max_value_loss_frac=max_value_loss_frac,
            my_lineup=my_lineup,
            send_to_opp_names=send_to_opp,
            my_banked_totals=my_banked_totals,
            trade_opponent_banked=(
                opponent_banked.get(opp_id) if opponent_banked else None
            ),
        )

        if not result["trade_feasible"]:
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
                "send_to_opp": sorted(send_to_opp),
                "receive": trade["receive"],
                "opponent": opp_name,
                "msv_exact": result["msv"],
                "delta_bv": result["delta_bv"],
                "value": result["value"],
                "opp_value_loss_pct": result["opp_value_loss_pct"],
                "new_ew": result["new_ew"],
            }
        )

    results.sort(key=lambda r: r["value"], reverse=True)

    # Prune trades where I overpay past the same tolerance the opponent gets.
    # Without this the top of the list fills with giveaways: the fairness
    # conditions only ever asked whether the OPPONENT was losing value.
    overpaying = [
        r
        for r in results
        if r.get("opp_value_loss_pct") is not None
        and r["opp_value_loss_pct"] < -100 * max_value_loss_frac
    ]
    if overpaying:
        results = [r for r in results if r not in overpaying]
        print(
            f"search_trades: dropped {len(overpaying)} trade(s) where I overpay "
            f"by more than {100 * max_value_loss_frac:.0f}% of the value at stake"
        )

    print(f"search_trades: {len(results)} trades above min_value {min_value}")
    return results


def search_trades_for_players(**kwargs) -> list[dict]:
    """Thin wrapper: search_trades in targeted mode. Kept for existing callers.

    Takes the same keyword arguments as search_trades; must_send and/or
    must_receive are what make the search targeted.
    """
    return search_trades(**kwargs)
