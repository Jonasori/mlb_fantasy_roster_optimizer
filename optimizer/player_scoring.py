"""
Per-player scoring: FV, MEW.

All functions are DataFrame enrichment: players in → players with new column(s) out.
Depends on config.
"""

import pandas as pd

from .config import MIN_STAT_STANDARD_DEVIATION

# ============================================================================
# FANTASY VALUE (FV)
# ============================================================================


def add_fantasy_value(players: pd.DataFrame) -> pd.DataFrame:
    """Add 'FV' column: sum of z-scores across 5 relevant scoring categories.

    Hitters: z(R) + z(HR) + z(RBI) + z(SB) + z(OPS)
    Pitchers: z(W) + z(SV) + z(K) + z(−ERA) + z(−WHIP)

    z-scores are computed within each player_type population (all hitters
    in the DataFrame, all pitchers in the DataFrame). This includes FAs,
    so FV is comparable across rostered players and free agents.

    For negative stats (ERA, WHIP), negate BEFORE computing z-score so
    that lower ERA → higher z-score → higher FV.

    COUNTING vs RATIO stats are treated differently, and the difference is
    load-bearing. A counting stat is always defined: 0 HR is a fact, and a
    player projected for nothing correctly z-scores far below the mean. A RATIO
    stat over zero playing time is not a fact, it is undefined — 0 PA means
    "this player has no OPS", not "this player has an OPS of 0.000". So:

      * ratio z-scores (OPS, ERA, WHIP) are computed over the positive-volume
        population only, and a zero-volume player contributes 0 for them.

    Anything else invents information. Reading the stored rate at face value is
    actively dangerous for the negated categories: a stored ERA of 0.00 against
    a ~4.00 mean is a large POSITIVE z, so a zero-volume player would be scored
    as the best pitcher in the league. Filling the stored value with the
    population mean instead would be just as arbitrary in the other direction —
    it asserts a rate the player never earned. Excluding the undefined value is
    the only choice that adds no information, and it makes the stored fill value
    irrelevant to FV, which is why data_prep can leave it at 0.0: team totals
    weight ratio stats by PA/IP, so a zero-volume row is inert there too.

    Requires: player_type, PA, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Adds: FV.
    """
    players = players.copy()
    players["FV"] = 0.0

    def _add_z(row_mask: pd.Series, stat: str, sign: float, pop_mask: pd.Series) -> None:
        """Add sign·z(stat) to FV for `row_mask`, standardized over `pop_mask`."""
        population = players.loc[pop_mask, stat]
        std = population.std()
        assert std > MIN_STAT_STANDARD_DEVIATION, (
            f"FV: Standard deviation of {stat} is {std:.6f} over "
            f"{int(pop_mask.sum())} players, below minimum "
            f"{MIN_STAT_STANDARD_DEVIATION}. Check that projections have "
            f"meaningful variance."
        )
        z = (players.loc[row_mask, stat] - population.mean()) / std
        players.loc[row_mask, "FV"] += sign * z

    h_mask = players["player_type"] == "hitter"
    p_mask = players["player_type"] == "pitcher"
    # Volume gates: who has a DEFINED rate for their type.
    h_rated = h_mask & (players["PA"] > 0)
    p_rated = p_mask & (players["IP"] > 0)

    for stat in ("R", "HR", "RBI", "SB"):
        _add_z(h_mask, stat, 1.0, h_mask)
    _add_z(h_rated, "OPS", 1.0, h_rated)

    for stat in ("W", "SV", "K"):
        _add_z(p_mask, stat, 1.0, p_mask)
    for stat in ("ERA", "WHIP"):
        _add_z(p_rated, stat, -1.0, p_rated)

    n_unrated = int((h_mask & ~h_rated).sum() + (p_mask & ~p_rated).sum())
    print(
        f"FV computed for {len(players)} players "
        f"(hitters: {h_mask.sum()}, pitchers: {p_mask.sum()})"
        + (
            f"; {n_unrated} with zero playing time scored on counting stats only"
            if n_unrated
            else ""
        )
    )
    return players


# ============================================================================
# MARGINAL EXPECTED WINS (MEW)
# ============================================================================


def add_mew(
    players: pd.DataFrame,
    my_totals: dict[str, float],
    gradient: dict[str, float],
) -> pd.DataFrame:
    """Add 'MEW' column: first-order marginal EW contribution per player.

    MEW is the central player-evaluation metric. It uses the EW gradient
    to score every player — hitters and pitchers alike — in one unified
    formula with no conditional logic (MATHEMATICAL_FRAMEWORK §4):

        MEW(p) = Σ_{c ∈ C_count} g_c × stat_c(p)
               + g_OPS  × PA(p) × (OPS(p)  − my_OPS)  / total_PA
               + g_ERA  × IP(p) × (ERA(p)  − my_ERA)  / total_IP
               + g_WHIP × IP(p) × (WHIP(p) − my_WHIP) / total_IP

    where C_count = {R, HR, RBI, SB, W, SV, K}.

    No hitter/pitcher branching needed: for hitters, IP = 0 so all
    pitching terms vanish; for pitchers, PA = 0 so all hitting terms
    vanish. The data encodes the player type; the formula is universal.

    The gradient is a pre-computed input (from compute_league_state),
    NOT recomputed here.

    SIGN VERIFICATION:
        g_ERA < 0. Good pitcher: (ERA − my_ERA) < 0. Product: positive. ✓

    Args:
        players: DataFrame with stat columns.
        my_totals: Converged team totals dict. Must contain all 10 category
            keys plus 'PA' and 'IP'.
        gradient: Pre-computed ∂EW/∂(my_c) from compute_ew_gradient.

    Requires: PA, IP, R, HR, RBI, SB, OPS, W, SV, K, ERA, WHIP.
    Adds: MEW.
    """
    players = players.copy()

    total_pa = my_totals["PA"]
    total_ip = my_totals["IP"]
    my_ops = my_totals["OPS"]
    my_era = my_totals["ERA"]
    my_whip = my_totals["WHIP"]

    assert total_pa > 0, (
        f"add_mew: total_PA is {total_pa}. "
        f"my_totals must come from compute_totals_for_starters (includes PA)."
    )
    assert total_ip > 0, (
        f"add_mew: total_IP is {total_ip}. "
        f"my_totals must come from compute_totals_for_starters (includes IP)."
    )

    # Fail fast on NaN stats: the unified MEW formula relies on the silver-table
    # invariant that every scoring stat is a real number (0 for the opposite
    # player type, never NaN). A NaN here silently poisons the player's MEW and
    # would propagate into the lineup objective with no error otherwise.
    _stat_cols = ["PA", "IP", "R", "HR", "RBI", "SB", "OPS", "W", "SV", "K", "ERA", "WHIP"]
    _nan_mask = players[_stat_cols].isna().any(axis=1)
    assert not _nan_mask.any(), (
        f"add_mew: NaN found in scoring stats for {int(_nan_mask.sum())} player(s): "
        f"{sorted(players.loc[_nan_mask, 'Name'])[:10]}. "
        f"The silver table must fill all 12 scoring stats (0 for the opposite "
        f"player type). Fix data_prep ingestion before scoring."
    )

    mew = pd.Series(0.0, index=players.index)

    for cat in ("R", "HR", "RBI", "SB", "W", "SV", "K"):
        mew += gradient[cat] * players[cat]

    mew += gradient["OPS"] * players["PA"] * (players["OPS"] - my_ops) / total_pa
    mew += gradient["ERA"] * players["IP"] * (players["ERA"] - my_era) / total_ip
    mew += gradient["WHIP"] * players["IP"] * (players["WHIP"] - my_whip) / total_ip

    players["MEW"] = mew
    return players
